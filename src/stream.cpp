/* Reusable single-stream context for pinned-host streaming LZ4 block decode:
 * host-resident compressed chunks decoded on the GPU, with the per-call CUDA
 * allocation of the earlier N-stream ring amortized into a caller-held context
 * that is created once and reused across decodes (masterplan section 4, the
 * asset-streaming memory path, M2).
 *
 * This adds NO kernel and NO parser code: it is host-side copy/stream
 * choreography around the unchanged, already-fuzzed cudec_lz4_decompress_batch,
 * which is reused verbatim per wave. Per-chunk output and result indices are
 * disjoint and a single stream executes every wave in order, so the output is
 * bit-identical on every path (determinism by construction) - across a fresh
 * context and a reused one, and before and after a grow.
 *
 * Why a single stream, no ring: the #24 measurement showed the copy/decode
 * overlap ceiling is only ~25% for LZ4 (the compressed H2D is ~4 ms against a
 * ~12 ms decode - LZ4's ~2:1 ratio keeps the input small), so the N-stream ring
 * bought almost no overlap. Measuring a context that allocates nothing in steady
 * state also corrected #24's own diagnosis: the per-call allocation is NOT the
 * dominant cost - reuse saves only ~8 ms (cold - steady) of the ~230 ms wall.
 * The dominant streaming cost was the per-wave serial submission (~51 waves for
 * Silesia at the fixed 64-chunk wave), and issue #33 took that lever: the wave
 * is now sized by the staging it costs, Silesia submits once on the
 * device-output path, and the steady-state device wall is 27.7 ms against
 * 235.4 ms. Under correctness > measured performance > minimal code, the
 * N-stream ring is therefore dropped for a single stream whose one reusable
 * staging set is grown on demand and reused. See docs/BENCHMARKS.md for the
 * corrected overlap analysis, the wave measurement, and the output-D2H future
 * lever. */
#include "batch_limits.h"
#include "cudec.h"
#include "stream_decode.h"

#include "vendor_raii.h"

#include "vendor_rt.h"

#include <cstdint>
#include <cstring>
#include <new>

namespace cudec_stream_detail {

/* The wave is what one submission carries, and the submission count is the
 * streaming wall on this platform: issue #33 measured Silesia's ~230 ms
 * steady-state against a ~12 ms device-resident decode and a ~4 ms compressed
 * H2D, and attributed the ~213 ms residual to the 51 serial submissions a
 * fixed 64-chunk wave produced. So the wave is sized by the staging it costs
 * rather than by a chunk count, and both bounds below are needed.
 *
 * The byte budget bounds peak staging: pinned + device compressed source, and
 * (host-output only) the device destination arena. It is charged against the
 * LARGEST chunk in the call rather than the mean, so the bound holds for any
 * distribution including a hostile one - a batch of tiny chunks with one huge
 * one cannot talk the sizer into a wave that allocates far past the budget.
 *
 * The chunk ceiling bounds the metadata staging independently, at
 * 4 * kWaveChunksMax * 8 bytes. Without it a batch of zero-length chunks
 * divides the budget by one and asks for a wave of kMaxBatchChunks entries. */
constexpr size_t kWaveStagingBudget = size_t{384} << 20;
constexpr size_t kWaveChunksMax = 4096;

/* The batch entry's launch limit bounds a stream batch too, and it is
 * cudec_detail::kMaxBatchChunks in src/batch_limits.h for both of them - the
 * duplicate that used to live here is what issue #39 was about. */
using cudec_detail::kMaxBatchChunks;

/* Metadata layout inside p_meta/d_meta for a wave of `wave_chunks` chunks:
 * [src_ptrs][src_sizes][dst_ptrs][dst_caps], each `wave_chunks` wide, 8-byte
 * elements. The stride is per call because the wave size is. */
inline size_t MetaStride(size_t wave_chunks) {
    return wave_chunks * sizeof(void*);
}

/* Chunks per wave for this call. Never zero: a single chunk larger than the
 * budget still has to be decoded, and it goes alone rather than being
 * refused - the budget sizes a wave, it is not a capacity limit on the ABI. */
inline size_t ChooseWaveChunks(const size_t* h_src_sizes,
                               const size_t* dst_caps, size_t chunk_count,
                               bool host_out) {
    size_t per_chunk_max = 1;
    for (size_t i = 0; i < chunk_count; i++) {
        size_t bytes = h_src_sizes[i];
        if (host_out) {
            /* Saturating: the sum only feeds a division that picks a wave
             * size, so clamping it costs one chunk per wave and never a
             * wrong bound. The real per-wave totals are summed exactly
             * below, where an overflow is a reject. */
            bytes = (SIZE_MAX - bytes < dst_caps[i]) ? SIZE_MAX
                                                     : bytes + dst_caps[i];
        }
        if (bytes > per_chunk_max) {
            per_chunk_max = bytes;
        }
    }
    size_t wave = kWaveStagingBudget / per_chunk_max;
    if (wave == 0) {
        wave = 1;
    }
    if (wave > kWaveChunksMax) {
        wave = kWaveChunksMax;
    }
    if (wave > chunk_count) {
        wave = chunk_count;
    }
    return wave;
}

inline bool MulOverflows(size_t a, size_t b) {
    return a != 0 && b > SIZE_MAX / a;
}

cudec_status CopyWaveToHost(cudec_chunk_result* wr, const unsigned char* d_dst,
                            void* const* dst_ptrs, const size_t* dst_caps,
                            size_t begin, size_t wn);

}  // namespace cudec_stream_detail

/* The opaque context (the header forward-declares `struct cudec_stream_ctx`).
 * One non-blocking stream, one grow-only pinned+device compressed-source
 * staging pair, one grow-only pinned+device metadata pair, one grow-only
 * device destination staging (host-output only), one grow-only device result
 * buffer with a pinned mirror, and one reuse event that gates reuse of the
 * single pinned-source buffer across waves. `poisoned` latches on a CUDA fault
 * during a decode: only destruction is valid afterwards.
 *
 * Not thread-safe: one context per thread, no internal locking (documented on
 * the ABI). The member types have external linkage (the shared cudec_cuda
 * namespace, not an anonymous one) so this ABI-visible struct triggers no
 * subobject-linkage diagnostic. */
struct cudec_stream_ctx {
    cudec_rt::StreamOwner stream;
    cudec_rt::EventOwner reuse_ev;
    cudec_rt::PinnedBuf p_src;
    cudec_rt::DevBuf d_src;
    cudec_rt::PinnedBuf p_meta;
    cudec_rt::DevBuf d_meta;
    cudec_rt::DevBuf d_dst; /* host-output staging only */
    cudec_rt::DevBuf d_results;
    cudec_rt::PinnedBuf p_results;
    bool poisoned = false;
};

namespace cudec_stream_detail {

/* Copies one drained wave's decoded bytes out of the device staging into the
 * caller's host destinations: exactly bytes_written per chunk, leaving the tail
 * beyond it untouched, with the staging slots laid out at dst_caps strides.
 * `wr` is the wave's slice of the pinned result mirror, `d_dst` its staging
 * base, and `dst_ptrs`/`dst_caps` the caller's arrays indexed from `begin`.
 * Returns CUDEC_ERR_CUDA on a fault, which fails the wave.
 *
 * bytes_written is device-reported and it becomes a HOST write length here, so
 * it is bounded by the chunk's own capacity before it is used, and a length
 * past that capacity is treated as a device fault rather than trusted: the
 * chunk is stamped non-OK with no claimed output and the wave fails, with
 * nothing written to the caller's buffer. The frame path already bounds the
 * same value against its internal block_max after its result readback; this is
 * the streaming path's half of it.
 *
 * Its own function with external linkage, rather than a block inside the wave
 * loop, so the negative test can drive it with an injected result record: the
 * lengths it acts on come from the device, and the current kernel cannot
 * produce the ones this has to survive. */
cudec_status CopyWaveToHost(cudec_chunk_result* wr, const unsigned char* d_dst,
                            void* const* dst_ptrs, const size_t* dst_caps,
                            size_t begin, size_t wn) {
    size_t off = 0;
    for (size_t j = 0; j < wn; j++) {
        const size_t i = begin + j;
        const size_t bw = static_cast<size_t>(wr[j].bytes_written);
        if (bw > dst_caps[i]) {
            wr[j].status = static_cast<int32_t>(CUDEC_ERR_CUDA);
            wr[j].reserved = 0;
            wr[j].bytes_written = 0;
            return CUDEC_ERR_CUDA;
        }
        if (bw != 0 && dst_ptrs[i] != nullptr &&
            cudec_rt::memcpy(dst_ptrs[i], d_dst + off, bw,
                       cudec_rt::memcpy_d2h) != cudec_rt::success) {
            return CUDEC_ERR_CUDA;
        }
        off += dst_caps[i];
    }
    return CUDEC_OK;
}

}  // namespace cudec_stream_detail

namespace {

using namespace cudec_stream_detail;

/* Pure argument validation: no CUDA call, no context needed. Both the context
 * entry and the one-shot wrapper run this BEFORE touching the device, so a
 * malformed call returns CUDEC_ERR_INVALID_ARGUMENT synchronously on any host
 * (including the GPU-less CI runner, where the conformance test exercises it). */
cudec_status ValidateStreamArgs(const void* const* h_src_ptrs,
                                const size_t* h_src_sizes,
                                void* const* dst_ptrs, const size_t* dst_caps,
                                size_t chunk_count, cudec_mem_space dst_space,
                                const cudec_chunk_result* h_results) {
    if (h_src_ptrs == nullptr || h_src_sizes == nullptr ||
        dst_ptrs == nullptr || dst_caps == nullptr || h_results == nullptr ||
        chunk_count == 0 || chunk_count > kMaxBatchChunks ||
        (dst_space != CUDEC_MEM_HOST && dst_space != CUDEC_MEM_DEVICE)) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    /* A NULL source with a non-zero size (a host read that would segfault) or a
     * NULL destination for a chunk that claims capacity is a caller error. */
    for (size_t i = 0; i < chunk_count; i++) {
        if ((h_src_ptrs[i] == nullptr && h_src_sizes[i] != 0) ||
            (dst_ptrs[i] == nullptr && dst_caps[i] != 0)) {
            return CUDEC_ERR_INVALID_ARGUMENT;
        }
    }
    return CUDEC_OK;
}

/* Stamps a defined non-OK status into every per-chunk record. Used to leave the
 * per-chunk channel fail-closed with a DEFINED cudec_status value (never a stale
 * or out-of-enum byte pattern) on any post-validation path that returns without
 * the device reporting a chunk. `status` is stored as int32; bytes_written is
 * zeroed, matching the "no output on error" contract. */
void StampNotDecoded(cudec_chunk_result* results, size_t chunk_count,
                     cudec_status status) {
    for (size_t i = 0; i < chunk_count; i++) {
        results[i].status = static_cast<int32_t>(status);
        results[i].reserved = 0;
        results[i].bytes_written = 0;
    }
}

/* Decodes the whole batch on the context's single stream, submitting each wave
 * through `entry`. Grows the staging to this call's high-water mark first
 * (reusing it when already large enough), then stages and launches each wave in
 * order. A CUDA-level fault (a failed copy/launch/sync, or a grow allocation
 * failure) poisons the context and returns CUDEC_ERR_CUDA; per-chunk decode
 * rejects are reported in h_results and never poison.
 *
 * Nothing below names a format. `entry` is the only thing that does, and it is
 * the seam src/stream_decode.h argues for. */
cudec_status DecodeStreamCtx(cudec_stream_ctx& ctx, BatchEntry entry,
                             const void* const* h_src_ptrs,
                             const size_t* h_src_sizes, void* const* dst_ptrs,
                             const size_t* dst_caps, size_t chunk_count,
                             cudec_mem_space dst_space,
                             cudec_chunk_result* h_results) {
    const bool host_out = (dst_space == CUDEC_MEM_HOST);
    const size_t wave_chunks =
        ChooseWaveChunks(h_src_sizes, dst_caps, chunk_count, host_out);
    const size_t wave_count =
        (chunk_count + wave_chunks - 1) / wave_chunks;
    const size_t meta_stride = MetaStride(wave_chunks);
    const size_t meta_bytes = 4 * meta_stride;

    /* Fail-closed the per-chunk channel up front: any post-validation early
     * return below (a size-overflow reject or a staging-grow failure) then
     * leaves every h_results[k] reading a DEFINED not-produced status rather than
     * stale caller memory. Successful and mid-wave-failure returns overwrite this
     * with the real per-chunk outcomes via the pinned-mirror publish. */
    StampNotDecoded(h_results, chunk_count, CUDEC_ERR_CUDA);

    /* Largest single wave's compressed bytes and (host-output) destination
     * bytes set the grow-to sizes: fixed for the whole call, so a hostile
     * chunk_count cannot drive per-wave growth. */
    size_t max_wave_src = 0;
    size_t max_wave_dst = 0;
    for (size_t w = 0; w < wave_count; w++) {
        const size_t begin = w * wave_chunks;
        const size_t end = (begin + wave_chunks < chunk_count)
                               ? begin + wave_chunks
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
        /* A zero-size device allocation is legal but avoid the corner. */
        max_wave_src = 1;
    }
    if (host_out && max_wave_dst == 0) {
        max_wave_dst = 1;
    }

    if (MulOverflows(chunk_count, sizeof(cudec_chunk_result))) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const size_t results_bytes = chunk_count * sizeof(cudec_chunk_result);

    /* Grow-only staging. A grow allocation failure (e.g. an oversized device
     * allocation) leaves the context's buffers partially grown; poison so only
     * destruction is valid afterwards. This is a DEFINED failure, reachable
     * through the public API without any undefined behavior. */
#define GROW(call) \
    CUDEC_RT_CHECK(call, { ctx.poisoned = true; return CUDEC_ERR_CUDA; })
    GROW(ctx.p_src.ensure(max_wave_src));
    GROW(ctx.d_src.ensure(max_wave_src));
    GROW(ctx.p_meta.ensure(meta_bytes));
    GROW(ctx.d_meta.ensure(meta_bytes));
    if (host_out) {
        GROW(ctx.d_dst.ensure(max_wave_dst));
    }
    GROW(ctx.d_results.ensure(results_bytes));
    GROW(ctx.p_results.ensure(results_bytes));
#undef GROW

    cudec_rt::stream_t stream = ctx.stream.s;
    cudec_rt::event_t reuse = ctx.reuse_ev.e;

    /* Seed the pinned result mirror with a DEFINED not-produced status: any wave
     * that does not complete before an error return then publishes CUDEC_ERR_CUDA
     * (not a stale or out-of-enum value) for its chunks, while completed waves
     * overwrite their slice via the result readback below. */
    StampNotDecoded(static_cast<cudec_chunk_result*>(ctx.p_results.p),
                    chunk_count, CUDEC_ERR_CUDA);

    /* On an error inside the wave loop we break rather than return, so the
     * unconditional drain below completes the in-flight stream before the RAII
     * owners (at context destruction) free the buffers it is still
     * reading/writing. WAVE_FAIL records the fault and stops the loop. */
    cudec_status wave_status = CUDEC_OK;
    bool have_pending_src = false; /* whether reuse has been recorded */
#define WAVE_FAIL(st)       \
    {                       \
        wave_status = (st); \
        break;              \
    }

    for (size_t w = 0; w < wave_count; w++) {
        const size_t begin = w * wave_chunks;
        const size_t end = (begin + wave_chunks < chunk_count)
                               ? begin + wave_chunks
                               : chunk_count;
        const size_t wn = end - begin;

        /* Reuse gate: the single pinned source/metadata staging is overwritten
         * every wave, so wait for the previous wave's H2D of it to finish
         * before the host memcpy clobbers it. This is the one cheap overlap the
         * single-stream design keeps: while the host is blocked here, the GPU
         * runs the previous wave's decode and result readback. The first wave
         * never recorded the event. */
        if (have_pending_src &&
            cudec_rt::event_synchronize(reuse) != cudec_rt::success) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }

        /* Stage the wave's compressed sources contiguously and build the
         * per-chunk device pointer/size arrays into the pinned metadata. */
        unsigned char* p_src = static_cast<unsigned char*>(ctx.p_src.p);
        unsigned char* d_src = static_cast<unsigned char*>(ctx.d_src.p);
        unsigned char* d_dst = static_cast<unsigned char*>(ctx.d_dst.p);
        const void** m_src = reinterpret_cast<const void**>(
            static_cast<unsigned char*>(ctx.p_meta.p));
        size_t* m_ssz = reinterpret_cast<size_t*>(
            static_cast<unsigned char*>(ctx.p_meta.p) + meta_stride);
        void** m_dst = reinterpret_cast<void**>(
            static_cast<unsigned char*>(ctx.p_meta.p) + 2 * meta_stride);
        size_t* m_cap = reinterpret_cast<size_t*>(
            static_cast<unsigned char*>(ctx.p_meta.p) + 3 * meta_stride);

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
        unsigned char* dm = static_cast<unsigned char*>(ctx.d_meta.p);
        const void* const* dd_src = reinterpret_cast<const void* const*>(dm);
        const size_t* dd_ssz =
            reinterpret_cast<const size_t*>(dm + meta_stride);
        void* const* dd_dst =
            reinterpret_cast<void* const*>(dm + 2 * meta_stride);
        const size_t* dd_cap =
            reinterpret_cast<const size_t*>(dm + 3 * meta_stride);

        if (src_off != 0 &&
            cudec_rt::memcpy_async(d_src, p_src, src_off, cudec_rt::memcpy_h2d,
                            stream) != cudec_rt::success) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }
        if (cudec_rt::memcpy_async(ctx.d_meta.p, ctx.p_meta.p, meta_bytes,
                                   cudec_rt::memcpy_h2d,
                                   stream) != cudec_rt::success) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }
        /* Record after both H2D copies so the next wave's reuse gate waits for
         * the pinned source AND metadata reads to complete. */
        if (cudec_rt::event_record(reuse, stream) != cudec_rt::success) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }
        have_pending_src = true;

        /* The per-wave result slice is offset*16 into a device-allocation
         * base, so it
         * satisfies the batch entry's 16-byte-alignment requirement. */
        cudec_chunk_result* d_res =
            static_cast<cudec_chunk_result*>(ctx.d_results.p) + begin;
        const cudec_status launched = entry(dd_src, dd_ssz, dd_dst, dd_cap, wn,
                                            d_res,
                                            cudec_rt::abi_stream(stream));
        if (launched != CUDEC_OK) {
            WAVE_FAIL(launched);
        }

        /* Result readback into the pinned mirror (async). */
        if (cudec_rt::memcpy_async(
                static_cast<cudec_chunk_result*>(ctx.p_results.p) + begin,
                d_res, wn * sizeof(cudec_chunk_result), cudec_rt::memcpy_d2h,
                stream) != cudec_rt::success) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }

        if (host_out) {
            /* Copy exactly bytes_written per chunk into the caller's host
             * buffer, leaving the tail beyond it untouched - the same contract
             * as the device-output path, and no cross-chunk residue. The exact
             * length needs the per-chunk result, so this wave is drained first;
             * the D2H targets pageable caller memory and is therefore
             * synchronous, so host-output does not overlap (device-output is
             * the overlapped path). */
            if (cudec_rt::stream_synchronize(stream) != cudec_rt::success) {
                WAVE_FAIL(CUDEC_ERR_CUDA);
            }
            cudec_chunk_result* wr =
                static_cast<cudec_chunk_result*>(ctx.p_results.p) + begin;
            const cudec_status copied =
                CopyWaveToHost(wr, d_dst, dst_ptrs, dst_caps, begin, wn);
            if (copied != CUDEC_OK) {
                WAVE_FAIL(copied);
            }
        }
    }
#undef WAVE_FAIL

    /* Drain the stream unconditionally so no async work outlives this call
     * (the entry is synchronous), and surface any async fault as a defined
     * error. */
    cudec_status drain = wave_status;
    if (cudec_rt::stream_synchronize(stream) != cudec_rt::success &&
        drain == CUDEC_OK) {
        drain = CUDEC_ERR_CUDA;
    }
    if (cudec_rt::get_last_error() != cudec_rt::success && drain == CUDEC_OK) {
        drain = CUDEC_ERR_CUDA;
    }

    /* Publish whatever per-chunk results completed (the defined CUDEC_ERR_CUDA
     * not-produced seed from above for any wave that did not), then decide the
     * aggregate. */
    std::memcpy(h_results, ctx.p_results.p, results_bytes);
    if (drain != CUDEC_OK) {
        /* A CUDA-level fault happened; the context is dead. */
        ctx.poisoned = true;
        return drain;
    }

    /* Aggregate: OK iff every chunk decoded OK, else the first non-OK in index
     * order - a lazy caller checking only the return still fails closed. A
     * per-chunk reject is a normal fail-closed outcome and does NOT poison. */
    for (size_t i = 0; i < chunk_count; i++) {
        if (h_results[i].status != CUDEC_OK) {
            return static_cast<cudec_status>(h_results[i].status);
        }
    }
    return CUDEC_OK;
}

}  // namespace

cudec_status cudec_stream_ctx_create(cudec_stream_ctx** out_ctx) {
    if (out_ctx == nullptr) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    *out_ctx = nullptr;
    cudec_stream_ctx* ctx = new (std::nothrow) cudec_stream_ctx();
    if (ctx == nullptr) {
        return CUDEC_ERR_CUDA; /* host OOM - never crosses the ABI as throw */
    }
    /* The only create-time device resources are the single stream and the
     * reuse event; the staging is allocated lazily on first decode (grow-only,
     * so create takes no sizing parameters). */
    if (ctx->stream.create() != cudec_rt::success ||
        ctx->reuse_ev.create() != cudec_rt::success) {
        delete ctx; /* RAII frees whichever of the two succeeded */
        return CUDEC_ERR_CUDA;
    }
    *out_ctx = ctx;
    return CUDEC_OK;
}

namespace cudec_stream_detail {

cudec_status DecodeOnStreamCtx(cudec_stream_ctx* ctx, BatchEntry entry,
                               const void* const* h_src_ptrs,
                               const size_t* h_src_sizes,
                               void* const* dst_ptrs, const size_t* dst_caps,
                               size_t chunk_count, cudec_mem_space dst_space,
                               cudec_chunk_result* h_results) {
    if (ctx == nullptr || entry == nullptr) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    /* Argument rejects are synchronous, make no CUDA call, and never poison the
     * context. */
    const cudec_status args = ValidateStreamArgs(h_src_ptrs, h_src_sizes,
                                                 dst_ptrs, dst_caps, chunk_count,
                                                 dst_space, h_results);
    if (args != CUDEC_OK) {
        return args;
    }
    /* A context poisoned by an earlier CUDA fault decodes nothing further. This
     * is a post-validation non-OK return, so fail-closed the per-chunk channel
     * with the defined not-produced status the ABI promises. */
    if (ctx->poisoned) {
        StampNotDecoded(h_results, chunk_count, CUDEC_ERR_CUDA);
        return CUDEC_ERR_CUDA;
    }
    try {
        return DecodeStreamCtx(*ctx, entry, h_src_ptrs, h_src_sizes, dst_ptrs,
                               dst_caps, chunk_count, dst_space, h_results);
    } catch (...) {
        /* A host allocation failed mid-decode; never let it cross the C ABI,
         * and poison since the context's state is now unknown. */
        ctx->poisoned = true;
        return CUDEC_ERR_CUDA;
    }
}

}  // namespace cudec_stream_detail

cudec_status cudec_lz4_decompress_stream_ctx(
    cudec_stream_ctx* ctx, const void* const* h_src_ptrs,
    const size_t* h_src_sizes, void* const* dst_ptrs, const size_t* dst_caps,
    size_t chunk_count, cudec_mem_space dst_space,
    cudec_chunk_result* h_results) {
    return cudec_stream_detail::DecodeOnStreamCtx(
        ctx, cudec_lz4_decompress_batch, h_src_ptrs, h_src_sizes, dst_ptrs,
        dst_caps, chunk_count, dst_space, h_results);
}

void cudec_stream_ctx_destroy(cudec_stream_ctx* ctx) {
    if (ctx == nullptr) {
        return;
    }
    /* The decode entry drains on every return, so nothing is normally pending;
     * this defensive sync is valid on a poisoned context too (it simply
     * surfaces the fault, which is ignored) and guarantees no async work
     * touches the buffers the destructor is about to free. */
    if (ctx->stream.s != nullptr) {
        (void)cudec_rt::stream_synchronize(ctx->stream.s);
    }
    delete ctx;
}

cudec_status cudec_lz4_decompress_stream(const void* const* h_src_ptrs,
                                         const size_t* h_src_sizes,
                                         void* const* dst_ptrs,
                                         const size_t* dst_caps,
                                         size_t chunk_count,
                                         cudec_mem_space dst_space,
                                         cudec_chunk_result* h_results) {
    /* One-shot: create a context, decode once, destroy it. The reusable
     * context (create/decode/destroy) is the amortized path; this wrapper pays
     * the full per-call setup and is here for the single-shot caller.
     *
     * Validate the arguments BEFORE creating the context, so a malformed call
     * returns CUDEC_ERR_INVALID_ARGUMENT synchronously without any CUDA call -
     * matching the documented contract and the GPU-less conformance test (where
     * cudec_stream_ctx_create would otherwise fail for lack of a device and
     * mask the argument error). */
    const cudec_status args = ValidateStreamArgs(
        h_src_ptrs, h_src_sizes, dst_ptrs, dst_caps, chunk_count, dst_space,
        h_results);
    if (args != CUDEC_OK) {
        return args;
    }
    cudec_stream_ctx* ctx = nullptr;
    const cudec_status created = cudec_stream_ctx_create(&ctx);
    if (created != CUDEC_OK) {
        return created;
    }
    const cudec_status st = cudec_lz4_decompress_stream_ctx(
        ctx, h_src_ptrs, h_src_sizes, dst_ptrs, dst_caps, chunk_count,
        dst_space, h_results);
    cudec_stream_ctx_destroy(ctx);
    return st;
}
