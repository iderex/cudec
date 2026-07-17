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
 * The dominant streaming cost is the per-wave serial submission (~51 waves for
 * Silesia), tracked as the real lever in issue #33. Under correctness > measured
 * performance > minimal code, the N-stream ring is therefore dropped for a
 * single stream whose one reusable staging set is grown on demand and reused;
 * the amortization this context buys is real but small. See docs/BENCHMARKS.md
 * for the corrected overlap analysis and the output-D2H future lever. */
#include "cudec.h"

#include "cuda_raii.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <new>

namespace cudec_stream_detail {

constexpr size_t kWaveChunks = 64; /* chunks per wave: amortizes launch cost */

/* The batch entry's own launch limit; a stream batch cannot exceed it either.
 * Kept in sync with validate_batch_args in lz4_decode.cuh. */
constexpr size_t kMaxChunks = static_cast<size_t>(INT32_MAX) * 32;

/* Metadata layout inside p_meta/d_meta for a wave of up to kWaveChunks chunks:
 * [src_ptrs][src_sizes][dst_ptrs][dst_caps], each kWaveChunks wide, 8-byte
 * elements. Fixed size, so the metadata staging never grows past this. */
constexpr size_t kMetaStride = kWaveChunks * sizeof(void*);
constexpr size_t kMetaBytes = 4 * kMetaStride;

inline bool MulOverflows(size_t a, size_t b) {
    return a != 0 && b > SIZE_MAX / a;
}

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
    cudec_cuda::StreamOwner stream;
    cudec_cuda::EventOwner reuse_ev;
    cudec_cuda::PinnedBuf p_src;
    cudec_cuda::DevBuf d_src;
    cudec_cuda::PinnedBuf p_meta;
    cudec_cuda::DevBuf d_meta;
    cudec_cuda::DevBuf d_dst; /* host-output staging only */
    cudec_cuda::DevBuf d_results;
    cudec_cuda::PinnedBuf p_results;
    bool poisoned = false;
};

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
        chunk_count == 0 || chunk_count > kMaxChunks ||
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

/* Decodes the whole batch on the context's single stream. Grows the staging to
 * this call's high-water mark first (reusing it when already large enough),
 * then stages and launches each wave in order. A CUDA-level fault (a failed
 * copy/launch/sync, or a grow allocation failure) poisons the context and
 * returns CUDEC_ERR_CUDA; per-chunk decode rejects are reported in h_results
 * and never poison. */
cudec_status DecodeStreamCtx(cudec_stream_ctx& ctx,
                             const void* const* h_src_ptrs,
                             const size_t* h_src_sizes, void* const* dst_ptrs,
                             const size_t* dst_caps, size_t chunk_count,
                             cudec_mem_space dst_space,
                             cudec_chunk_result* h_results) {
    const bool host_out = (dst_space == CUDEC_MEM_HOST);
    const size_t wave_count = (chunk_count + kWaveChunks - 1) / kWaveChunks;

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
        const size_t begin = w * kWaveChunks;
        const size_t end = (begin + kWaveChunks < chunk_count)
                               ? begin + kWaveChunks
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

    if (MulOverflows(chunk_count, sizeof(cudec_chunk_result))) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const size_t results_bytes = chunk_count * sizeof(cudec_chunk_result);

    /* Grow-only staging. A grow allocation failure (e.g. an oversized
     * cudaMalloc) leaves the context's buffers partially grown; poison so only
     * destruction is valid afterwards. This is a DEFINED failure, reachable
     * through the public API without any undefined behavior. */
#define GROW(call) \
    CUDEC_CUDA_CHECK(call, { ctx.poisoned = true; return CUDEC_ERR_CUDA; })
    GROW(ctx.p_src.ensure(max_wave_src));
    GROW(ctx.d_src.ensure(max_wave_src));
    GROW(ctx.p_meta.ensure(kMetaBytes));
    GROW(ctx.d_meta.ensure(kMetaBytes));
    if (host_out) {
        GROW(ctx.d_dst.ensure(max_wave_dst));
    }
    GROW(ctx.d_results.ensure(results_bytes));
    GROW(ctx.p_results.ensure(results_bytes));
#undef GROW

    cudaStream_t stream = ctx.stream.s;
    cudaEvent_t reuse = ctx.reuse_ev.e;

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
        const size_t begin = w * kWaveChunks;
        const size_t end = (begin + kWaveChunks < chunk_count)
                               ? begin + kWaveChunks
                               : chunk_count;
        const size_t wn = end - begin;

        /* Reuse gate: the single pinned source/metadata staging is overwritten
         * every wave, so wait for the previous wave's H2D of it to finish
         * before the host memcpy clobbers it. This is the one cheap overlap the
         * single-stream design keeps: while the host is blocked here, the GPU
         * runs the previous wave's decode and result readback. The first wave
         * never recorded the event. */
        if (have_pending_src &&
            cudaEventSynchronize(reuse) != cudaSuccess) {
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
            static_cast<unsigned char*>(ctx.p_meta.p) + kMetaStride);
        void** m_dst = reinterpret_cast<void**>(
            static_cast<unsigned char*>(ctx.p_meta.p) + 2 * kMetaStride);
        size_t* m_cap = reinterpret_cast<size_t*>(
            static_cast<unsigned char*>(ctx.p_meta.p) + 3 * kMetaStride);

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
        if (cudaMemcpyAsync(ctx.d_meta.p, ctx.p_meta.p, kMetaBytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }
        /* Record after both H2D copies so the next wave's reuse gate waits for
         * the pinned source AND metadata reads to complete. */
        if (cudaEventRecord(reuse, stream) != cudaSuccess) {
            WAVE_FAIL(CUDEC_ERR_CUDA);
        }
        have_pending_src = true;

        /* The per-wave result slice is offset*16 into a cudaMalloc base, so it
         * satisfies the batch entry's 16-byte-alignment requirement. */
        cudec_chunk_result* d_res =
            static_cast<cudec_chunk_result*>(ctx.d_results.p) + begin;
        const cudec_status launched = cudec_lz4_decompress_batch(
            dd_src, dd_ssz, dd_dst, dd_cap, wn, d_res, stream);
        if (launched != CUDEC_OK) {
            WAVE_FAIL(launched);
        }

        /* Result readback into the pinned mirror (async). */
        if (cudaMemcpyAsync(
                static_cast<cudec_chunk_result*>(ctx.p_results.p) + begin,
                d_res, wn * sizeof(cudec_chunk_result), cudaMemcpyDeviceToHost,
                stream) != cudaSuccess) {
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
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                WAVE_FAIL(CUDEC_ERR_CUDA);
            }
            const cudec_chunk_result* wr =
                static_cast<cudec_chunk_result*>(ctx.p_results.p) + begin;
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
    }
#undef WAVE_FAIL

    /* Drain the stream unconditionally so no async work outlives this call
     * (the entry is synchronous), and surface any async fault as a defined
     * error. */
    cudec_status drain = wave_status;
    if (cudaStreamSynchronize(stream) != cudaSuccess && drain == CUDEC_OK) {
        drain = CUDEC_ERR_CUDA;
    }
    if (cudaGetLastError() != cudaSuccess && drain == CUDEC_OK) {
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
    if (ctx->stream.create() != cudaSuccess ||
        ctx->reuse_ev.create() != cudaSuccess) {
        delete ctx; /* RAII frees whichever of the two succeeded */
        return CUDEC_ERR_CUDA;
    }
    *out_ctx = ctx;
    return CUDEC_OK;
}

cudec_status cudec_lz4_decompress_stream_ctx(
    cudec_stream_ctx* ctx, const void* const* h_src_ptrs,
    const size_t* h_src_sizes, void* const* dst_ptrs, const size_t* dst_caps,
    size_t chunk_count, cudec_mem_space dst_space,
    cudec_chunk_result* h_results) {
    if (ctx == nullptr) {
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
        return DecodeStreamCtx(*ctx, h_src_ptrs, h_src_sizes, dst_ptrs,
                               dst_caps, chunk_count, dst_space, h_results);
    } catch (...) {
        /* A host allocation failed mid-decode; never let it cross the C ABI,
         * and poison since the context's state is now unknown. */
        ctx->poisoned = true;
        return CUDEC_ERR_CUDA;
    }
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
        (void)cudaStreamSynchronize(ctx->stream.s);
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
