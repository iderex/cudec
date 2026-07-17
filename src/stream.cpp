/* Pinned-host streaming LZ4 block decode: host-resident compressed chunks
 * decoded on the GPU with the H2D copy overlapped against decode across N
 * CUDA streams (masterplan section 4, the asset-streaming memory path, M2).
 *
 * This adds NO kernel and NO parser code: it is host-side copy/stream
 * choreography around the unchanged, already-fuzzed cudec_lz4_decompress_batch,
 * which is reused verbatim per wave. Per-chunk output and result indices are
 * disjoint, so stream interleaving cannot change the bytes - the output is
 * bit-identical for every stream count (determinism by construction). The
 * per-frame staging in frame.cpp is the future consumer of this path (#24's
 * note in frame.cpp); wiring frame.cpp onto it is a separate change. */
#include "cudec.h"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace {

constexpr unsigned kDefaultStreams = 4;
constexpr unsigned kMaxStreams = 32;
constexpr size_t kWaveChunks = 64; /* chunks per wave: amortizes launch cost */
constexpr size_t kRingDepth = 2;   /* ring slots per stream (double-buffer) */

/* The batch entry's own launch limit; a stream batch cannot exceed it
 * either. Kept in sync with validate_batch_args in lz4_decode.cuh. */
constexpr size_t kMaxChunks = static_cast<size_t>(INT32_MAX) * 32;

bool MulOverflows(size_t a, size_t b) { return a != 0 && b > SIZE_MAX / a; }

/* RAII owners so every device/pinned/stream/event allocation is freed on
 * every return path, including the reject and exception paths. */
struct DevPtr {
    void* p = nullptr;
    DevPtr() = default;
    DevPtr(const DevPtr&) = delete;
    DevPtr& operator=(const DevPtr&) = delete;
    ~DevPtr() {
        if (p != nullptr) {
            (void)cudaFree(p);
        }
    }
    cudaError_t alloc(size_t bytes) { return cudaMalloc(&p, bytes); }
};

struct PinnedPtr {
    void* p = nullptr;
    PinnedPtr() = default;
    PinnedPtr(const PinnedPtr&) = delete;
    PinnedPtr& operator=(const PinnedPtr&) = delete;
    ~PinnedPtr() {
        if (p != nullptr) {
            (void)cudaFreeHost(p);
        }
    }
    cudaError_t alloc(size_t bytes) {
        return cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    }
};

struct StreamPtr {
    cudaStream_t s = nullptr;
    StreamPtr() = default;
    StreamPtr(const StreamPtr&) = delete;
    StreamPtr& operator=(const StreamPtr&) = delete;
    ~StreamPtr() {
        if (s != nullptr) {
            (void)cudaStreamDestroy(s);
        }
    }
    cudaError_t create() {
        /* Non-blocking so the overlap does not depend on the legacy default
         * stream staying idle (a caller with default-stream work in the same
         * context would otherwise serialize every wave). */
        return cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    }
};

struct EventPtr {
    cudaEvent_t e = nullptr;
    EventPtr() = default;
    EventPtr(const EventPtr&) = delete;
    EventPtr& operator=(const EventPtr&) = delete;
    ~EventPtr() {
        if (e != nullptr) {
            (void)cudaEventDestroy(e);
        }
    }
    cudaError_t create() { return cudaEventCreate(&e); }
};

/* Per ring slot: one pinned staging buffer + one device buffer for the
 * wave's compressed sources, a pinned+device metadata buffer for the batch
 * entry's per-chunk pointer/size arrays, an optional device destination
 * staging buffer (host-output only), and a completion event. Slots are
 * reused round-robin; a slot's event gates its reuse. */
struct Slot {
    PinnedPtr p_src;
    DevPtr d_src;
    PinnedPtr p_meta;
    DevPtr d_meta;
    DevPtr d_dst; /* host-output staging only */
    EventPtr ev;
};

/* Metadata layout inside p_meta/d_meta for a wave of up to kWaveChunks
 * chunks: [src_ptrs][src_sizes][dst_ptrs][dst_caps], each kWaveChunks wide,
 * 8-byte elements. */
constexpr size_t kMetaStride = kWaveChunks * sizeof(void*);
constexpr size_t kMetaBytes = 4 * kMetaStride;

cudec_status DecodeStream(const void* const* h_src_ptrs,
                          const size_t* h_src_sizes, void* const* dst_ptrs,
                          const size_t* dst_caps, size_t chunk_count,
                          cudec_mem_space dst_space, unsigned streams,
                          cudec_chunk_result* h_results) {
#define STREAM_CUDA(call)                       \
    do {                                        \
        if ((call) != cudaSuccess) {            \
            return CUDEC_ERR_CUDA;              \
        }                                       \
    } while (0)

    const bool host_out = (dst_space == CUDEC_MEM_HOST);
    const size_t wave_count = (chunk_count + kWaveChunks - 1) / kWaveChunks;

    unsigned n_streams = (streams == 0) ? kDefaultStreams : streams;
    if (n_streams > kMaxStreams) {
        n_streams = kMaxStreams;
    }
    if (static_cast<size_t>(n_streams) > wave_count) {
        n_streams = static_cast<unsigned>(wave_count);
    }
    /* Host output drains each wave before issuing the next (the readback to
     * pageable caller memory is synchronous), so extra streams and ring slots
     * buy no overlap - collapse to a single stream and avoid provisioning a
     * fan-out that never engages. */
    if (host_out) {
        n_streams = 1;
    }
    const size_t n_slots = static_cast<size_t>(n_streams) * kRingDepth;

    /* Largest single wave's compressed bytes and destination bytes decide
     * the per-slot buffer sizes (fixed for the whole call, so a hostile
     * chunk_count cannot drive per-wave growth). */
    size_t max_wave_src = 0;
    size_t max_wave_dst = 0;
    for (size_t w = 0; w < wave_count; w++) {
        const size_t begin = w * kWaveChunks;
        const size_t end =
            (begin + kWaveChunks < chunk_count) ? begin + kWaveChunks
                                                : chunk_count;
        size_t wsrc = 0, wdst = 0;
        for (size_t i = begin; i < end; i++) {
            if (SIZE_MAX - wsrc < h_src_sizes[i]) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            wsrc += h_src_sizes[i];
            if (host_out) {
                if (SIZE_MAX - wdst < dst_caps[i]) {
                    return CUDEC_ERR_CORRUPT_INPUT;
                }
                wdst += dst_caps[i];
            }
        }
        if (wsrc > max_wave_src) {
            max_wave_src = wsrc;
        }
        if (wdst > max_wave_dst) {
            max_wave_dst = wdst;
        }
    }
    if (max_wave_src == 0) {
        max_wave_src = 1; /* cudaMalloc(0) is legal but avoid the corner */
    }
    if (host_out && max_wave_dst == 0) {
        max_wave_dst = 1;
    }

    /* Shared results: one device buffer + a pinned mirror so the per-wave
     * result readback stays async (a device->pageable copy would sync and
     * serialize the pipeline). */
    if (MulOverflows(chunk_count, sizeof(cudec_chunk_result))) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const size_t results_bytes = chunk_count * sizeof(cudec_chunk_result);
    DevPtr d_results;
    PinnedPtr p_results;
    STREAM_CUDA(d_results.alloc(results_bytes));
    STREAM_CUDA(p_results.alloc(results_bytes));
    /* 0xFF is a non-OK status sentinel: any wave that fails to complete
     * before an error return leaves its result records reading not-OK rather
     * than stale caller memory. */
    std::memset(p_results.p, 0xFF, results_bytes);

    std::vector<StreamPtr> stream_pool(n_streams);
    for (unsigned s = 0; s < n_streams; s++) {
        STREAM_CUDA(stream_pool[s].create());
    }
    std::vector<Slot> slots(n_slots);
    for (size_t s = 0; s < n_slots; s++) {
        STREAM_CUDA(slots[s].p_src.alloc(max_wave_src));
        STREAM_CUDA(slots[s].d_src.alloc(max_wave_src));
        STREAM_CUDA(slots[s].p_meta.alloc(kMetaBytes));
        STREAM_CUDA(slots[s].d_meta.alloc(kMetaBytes));
        if (host_out) {
            STREAM_CUDA(slots[s].d_dst.alloc(max_wave_dst));
        }
        STREAM_CUDA(slots[s].ev.create());
    }

    /* On an error inside the wave loop we break rather than return, so the
     * unconditional drain below completes every in-flight stream before the
     * RAII owners free the buffers those streams are still reading/writing
     * (not relying on cudaFree's implicit device sync). WAVE_FAIL records the
     * failure and stops the loop. */
    cudec_status wave_status = CUDEC_OK;
    /* Used only inside braced if-blocks in the wave loop; the break exits the
     * wave loop (not a do-while) so control reaches the unconditional drain. */
#define WAVE_FAIL(st)       \
    {                       \
        wave_status = (st); \
        break;              \
    }

    for (size_t w = 0; w < wave_count; w++) {
        const size_t begin = w * kWaveChunks;
        const size_t end =
            (begin + kWaveChunks < chunk_count) ? begin + kWaveChunks
                                                : chunk_count;
        const size_t wn = end - begin;
        const unsigned si = static_cast<unsigned>(w % n_streams);
        Slot& slot = slots[w % n_slots];
        cudaStream_t stream = stream_pool[si].s;

        /* Reuse gate: wait for the slot's previous wave to complete before
         * overwriting its buffers. A never-recorded event returns at once. */
        if (cudaEventSynchronize(slot.ev.e) != cudaSuccess) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }

        /* Stage the wave's compressed sources contiguously and build the
         * per-chunk device pointer/size arrays into the pinned metadata. */
        unsigned char* p_src = static_cast<unsigned char*>(slot.p_src.p);
        unsigned char* d_src = static_cast<unsigned char*>(slot.d_src.p);
        unsigned char* d_dst = static_cast<unsigned char*>(slot.d_dst.p);
        const void** m_src =
            reinterpret_cast<const void**>(static_cast<unsigned char*>(
                slot.p_meta.p));
        size_t* m_ssz = reinterpret_cast<size_t*>(
            static_cast<unsigned char*>(slot.p_meta.p) + kMetaStride);
        void** m_dst = reinterpret_cast<void**>(
            static_cast<unsigned char*>(slot.p_meta.p) + 2 * kMetaStride);
        size_t* m_cap = reinterpret_cast<size_t*>(
            static_cast<unsigned char*>(slot.p_meta.p) + 3 * kMetaStride);

        size_t src_off = 0;
        size_t dst_off = 0;
        for (size_t j = 0; j < wn; j++) {
            const size_t i = begin + j;
            const size_t ssz = h_src_sizes[i];
            if (ssz != 0) {
                std::memcpy(p_src + src_off, h_src_ptrs[i], ssz);
            }
            m_src[j] = d_src + src_off;
            m_ssz[j] = ssz;
            m_cap[j] = dst_caps[i];
            if (host_out) {
                m_dst[j] = d_dst + dst_off;
                dst_off += dst_caps[i];
            } else {
                m_dst[j] = dst_ptrs[i]; /* decode straight into caller VRAM */
            }
            src_off += ssz;
        }

        /* Device metadata pointers into d_meta (same layout as p_meta). */
        unsigned char* dm = static_cast<unsigned char*>(slot.d_meta.p);
        const void* const* dd_src =
            reinterpret_cast<const void* const*>(dm);
        const size_t* dd_ssz =
            reinterpret_cast<const size_t*>(dm + kMetaStride);
        void* const* dd_dst =
            reinterpret_cast<void* const*>(dm + 2 * kMetaStride);
        const size_t* dd_cap =
            reinterpret_cast<const size_t*>(dm + 3 * kMetaStride);

        if (src_off != 0 &&
            cudaMemcpyAsync(d_src, p_src, src_off, cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }
        if (cudaMemcpyAsync(slot.d_meta.p, slot.p_meta.p, kMetaBytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }

        /* The per-wave result slice is offset*16 into a cudaMalloc base, so
         * it satisfies the batch entry's 16-byte-alignment requirement. */
        cudec_chunk_result* d_res =
            static_cast<cudec_chunk_result*>(d_results.p) + begin;
        const cudec_status launched = cudec_lz4_decompress_batch(
            dd_src, dd_ssz, dd_dst, dd_cap, wn, d_res, stream);
        if (launched != CUDEC_OK) {
            WAVE_FAIL(launched);
        }

        /* Result readback into the pinned mirror (async). */
        if (cudaMemcpyAsync(
                static_cast<cudec_chunk_result*>(p_results.p) + begin, d_res,
                wn * sizeof(cudec_chunk_result), cudaMemcpyDeviceToHost,
                stream) != cudaSuccess) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }

        if (host_out) {
            /* Copy exactly bytes_written per chunk into the caller's host
             * buffer, leaving the tail beyond it untouched - the same
             * contract as the device-output path, and no cross-chunk residue.
             * The exact length needs the per-chunk result, so this wave is
             * drained first; the D2H targets pageable caller memory and is
             * therefore synchronous, so host-output does not overlap in this
             * cut (device-output is the overlapped path). Overlapping the
             * host readback needs a pinned output ring - a later change. */
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                WAVE_FAIL(CUDEC_ERR_CUDA);
            }
            const cudec_chunk_result* wr =
                static_cast<cudec_chunk_result*>(p_results.p) + begin;
            size_t off = 0;
            bool copy_failed = false;
            for (size_t j = 0; j < wn; j++) {
                const size_t i = begin + j;
                const size_t bw = static_cast<size_t>(wr[j].bytes_written);
                if (bw != 0 && dst_ptrs[i] != nullptr &&
                    cudaMemcpy(dst_ptrs[i], d_dst + off, bw,
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    copy_failed = true;
                    break;
                }
                off += dst_caps[i];
            }
            if (copy_failed) {
                WAVE_FAIL(CUDEC_ERR_CUDA);
            }
        }

        /* Gate slot reuse on the whole wave's completion. */
        if (cudaEventRecord(slot.ev.e, stream) != cudaSuccess) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }
    }
#undef WAVE_FAIL

    /* Drain every stream unconditionally so no async work outlives the RAII
     * teardown, and surface any async fault as a defined error. */
    cudec_status drain = wave_status;
    for (unsigned s = 0; s < n_streams; s++) {
        if (cudaStreamSynchronize(stream_pool[s].s) != cudaSuccess &&
            drain == CUDEC_OK) {
            drain = CUDEC_ERR_CUDA;
        }
    }
    if (cudaGetLastError() != cudaSuccess && drain == CUDEC_OK) {
        drain = CUDEC_ERR_CUDA;
    }

    /* Publish whatever per-chunk results completed (0xFF non-OK sentinel for
     * any wave that did not), then decide the aggregate. */
    std::memcpy(h_results, p_results.p, results_bytes);
    if (drain != CUDEC_OK) {
        return drain;
    }

    /* Aggregate: OK iff every chunk decoded OK, else the first non-OK in
     * index order - a lazy caller checking only the return still fails
     * closed. */
    for (size_t i = 0; i < chunk_count; i++) {
        if (h_results[i].status != CUDEC_OK) {
            return static_cast<cudec_status>(h_results[i].status);
        }
    }
    return CUDEC_OK;
#undef STREAM_CUDA
}

}  // namespace

cudec_status cudec_lz4_decompress_stream(const void* const* h_src_ptrs,
                                         const size_t* h_src_sizes,
                                         void* const* dst_ptrs,
                                         const size_t* dst_caps,
                                         size_t chunk_count,
                                         cudec_mem_space dst_space,
                                         unsigned streams,
                                         cudec_chunk_result* h_results) {
    if (h_src_ptrs == nullptr || h_src_sizes == nullptr ||
        dst_ptrs == nullptr || dst_caps == nullptr || h_results == nullptr ||
        chunk_count == 0 || chunk_count > kMaxChunks ||
        (dst_space != CUDEC_MEM_HOST && dst_space != CUDEC_MEM_DEVICE)) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    /* A NULL source with a non-zero size (a host read that would segfault) or
     * a NULL destination for a chunk that claims capacity is a caller error;
     * reject the whole call before touching the device. */
    for (size_t i = 0; i < chunk_count; i++) {
        if ((h_src_ptrs[i] == nullptr && h_src_sizes[i] != 0) ||
            (dst_ptrs[i] == nullptr && dst_caps[i] != 0)) {
            return CUDEC_ERR_INVALID_ARGUMENT;
        }
    }
    try {
        return DecodeStream(h_src_ptrs, h_src_sizes, dst_ptrs, dst_caps,
                            chunk_count, dst_space, streams, h_results);
    } catch (...) {
        /* A host allocation failed; never let it cross the C ABI. */
        return CUDEC_ERR_CUDA;
    }
}
