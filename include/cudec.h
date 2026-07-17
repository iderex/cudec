/* cudec - open-source GPU decompression for the standard formats.
 *
 * Public C ABI. Everything in this header is C-compatible and requires no
 * CUDA headers; the library never throws across this boundary and never
 * reports output it did not fully validate.
 */
#ifndef CUDEC_H
#define CUDEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDEC_VERSION_MAJOR 0
#define CUDEC_VERSION_MINOR 0
#define CUDEC_VERSION_PATCH 1

/* Returns the runtime library version as (major * 10000 +
 * minor * 100 + patch), for ABI sanity checks against these macros. */
int cudec_version(void);

typedef enum cudec_status {
    CUDEC_OK = 0,
    CUDEC_ERR_INVALID_ARGUMENT = 1,
    CUDEC_ERR_CORRUPT_INPUT = 2,
    CUDEC_ERR_OUTPUT_TOO_SMALL = 3,
    CUDEC_ERR_CUDA = 4,
    /* Reserved for an API entry point or decode path declared here but not
     * yet implemented (e.g. a future format). No current entry point
     * returns it; kept at this fixed value so a caller's switch stays
     * exhaustive once it is wired up. */
    CUDEC_ERR_NOT_IMPLEMENTED = 5,
    /* A well-formed frame that uses a feature cudec does not decode
     * (block-linked mode, a dictionary id). Distinct from CORRUPT_INPUT:
     * the input is valid, just outside cudec's supported subset. */
    CUDEC_ERR_UNSUPPORTED = 6
} cudec_status;

/* Binary-compatible with cudaStream_t without pulling in the CUDA headers:
 * both are pointers to the driver's CUstream_st. Pass a cudaStream_t
 * directly; NULL means the legacy default stream (callers built with
 * per-thread default streams pass cudaStreamPerThread explicitly). */
typedef struct CUstream_st* cudec_stream_t;

/* Per-chunk outcome, written by the device into a caller-provided device
 * buffer. Fixed 16-byte layout on both sides of the ABI. */
typedef struct cudec_chunk_result {
    int32_t status;         /* a cudec_status value */
    uint32_t reserved;      /* written as zero */
    uint64_t bytes_written; /* valid output bytes when status is CUDEC_OK */
} cudec_chunk_result;

/* The device writes these records straight into caller memory, so every
 * compiler on either side of the ABI must agree on the layout. The first
 * member is at offset 0 by definition; with the total size and the last
 * member's offset pinned, no other layout is possible. */
#if defined(__cplusplus) && __cplusplus >= 201103L
static_assert(sizeof(cudec_chunk_result) == 16 &&
                  offsetof(cudec_chunk_result, bytes_written) == 8,
              "cudec_chunk_result must keep its fixed 16-byte ABI layout");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(cudec_chunk_result) == 16 &&
                   offsetof(cudec_chunk_result, bytes_written) == 8,
               "cudec_chunk_result must keep its fixed 16-byte ABI layout");
#else
/* Pre-C11 / pre-C++11: a negative array size fails compilation on drift. */
typedef char cudec_chunk_result_layout_check
    [(sizeof(cudec_chunk_result) == 16 &&
      offsetof(cudec_chunk_result, bytes_written) == 8)
         ? 1
         : -1];
#endif

/* Batch LZ4 block decode. Each chunk is an independent LZ4 block; on
 * success the chunk's result reports CUDEC_OK and bytes_written, and the
 * destination holds exactly bytes_written decoded bytes. A malformed
 * chunk reports a defined error (CUDEC_ERR_CORRUPT_INPUT /
 * CUDEC_ERR_OUTPUT_TOO_SMALL) with bytes_written == 0; its destination
 * contents are then unspecified but never presented as a valid decode.
 *
 * All array arguments are device memory holding chunk_count entries, and
 * the pointers those arrays contain are device pointers; d_results must be
 * 16-byte aligned (any cudaMalloc allocation is). The call is asynchronous
 * on `stream`: the synchronous return value covers argument validation and
 * launch submission only; per-chunk outcomes land in d_results and are
 * valid once the stream reaches the end of this launch.
 *
 * Validation rejects the whole call synchronously with
 * CUDEC_ERR_INVALID_ARGUMENT and launches nothing: any NULL array
 * argument, a misaligned d_results, an empty batch (chunk_count == 0), and
 * a batch beyond the implementation's launch limit (rejected, never
 * truncated). A rejected call makes no CUDA call and leaves the thread's
 * pending CUDA error state untouched; a call that passes validation
 * consumes that pending state (cudaGetLastError semantics), so the
 * returned status reflects this submission alone. */
cudec_status cudec_lz4_decompress_batch(const void* const* d_src_ptrs,
                                        const size_t* d_src_sizes,
                                        void* const* d_dst_ptrs,
                                        const size_t* d_dst_capacities,
                                        size_t chunk_count,
                                        cudec_chunk_result* d_results,
                                        cudec_stream_t stream);

/* Decode a single LZ4 frame (the .lz4 container: magic, frame descriptor,
 * data blocks, end mark, optional checksums) from host memory into host
 * memory, using the GPU batch decoder internally. Synchronous.
 *
 * `frame`/`frame_size` is the whole frame in host memory; the decoded
 * output is written to `dst` (host, `dst_capacity` bytes) and the produced
 * size is returned in `*bytes_written`.
 *
 * Supported subset: block-INDEPENDENT frames (compress with
 * LZ4F_blockIndependent). The header, block, and content checksums and the
 * optional declared content size, when present, are verified fail-closed.
 * Returns CUDEC_ERR_UNSUPPORTED for a valid frame cudec does not decode
 * (block-linked mode - the default of liblz4's frame compressor - or a
 * dictionary id), CUDEC_ERR_CORRUPT_INPUT for a malformed frame, a checksum
 * mismatch, or a declared content size that does not match the decoded
 * size, CUDEC_ERR_OUTPUT_TOO_SMALL when `dst_capacity` is too small, and
 * CUDEC_ERR_CUDA on a device or host resource failure. A NULL `frame` or
 * `bytes_written`, or a NULL `dst` with a non-zero `dst_capacity`, returns
 * CUDEC_ERR_INVALID_ARGUMENT. On any error `*bytes_written` is 0 and no
 * partial output is presented as a valid decode. */
cudec_status cudec_lz4f_decompress(const void* frame, size_t frame_size,
                                   void* dst, size_t dst_capacity,
                                   size_t* bytes_written);

/* Memory space of the streaming decoder's per-chunk destinations. */
typedef enum cudec_mem_space {
    CUDEC_MEM_HOST = 0,   /* dst_ptrs are host pointers; output is copied D2H */
    CUDEC_MEM_DEVICE = 1  /* dst_ptrs are device pointers; output stays in VRAM */
} cudec_mem_space;

/* A reusable streaming-decode context (opaque). It owns the CUDA stream and
 * the pinned/device staging the streaming decoder needs; create it once and
 * reuse it across decodes so the staging allocation is paid once, not per
 * call. The staging grows on demand to the largest batch decoded so far and is
 * reused for same-or-smaller batches (grow-only high-water sizing), so create
 * takes no sizing parameters. Not thread-safe: use one context per thread. */
typedef struct cudec_stream_ctx cudec_stream_ctx;

/* Creates a streaming-decode context and stores it in *out_ctx. Allocates only
 * the CUDA stream and a small event up front; the staging is allocated lazily
 * on the first decode. Returns CUDEC_ERR_INVALID_ARGUMENT if out_ctx is NULL,
 * CUDEC_ERR_CUDA on a stream/event/host allocation failure. On any error
 * *out_ctx is set to NULL. Never throws across this boundary. */
cudec_status cudec_stream_ctx_create(cudec_stream_ctx** out_ctx);

/* Streaming batch LZ4 block decode from HOST-resident compressed chunks, on a
 * reusable context. Synchronous: the work is fully drained before return, so
 * every dst[k] and h_results[k] is valid on a CUDEC_OK return.
 *
 * All array arguments are HOST arrays of chunk_count entries. h_src_ptrs[k] /
 * h_src_sizes[k] is the k-th compressed block in host memory; all input is
 * staged through the context's own pinned buffer, so the H2D copy does not
 * depend on the caller pinning its input. dst_ptrs[k] / dst_caps[k] is the
 * k-th output slot, in the space named by dst_space: for CUDEC_MEM_DEVICE the
 * decode writes device memory directly; for CUDEC_MEM_HOST the decoded bytes
 * are read back to host memory. On a successful chunk the caller's destination
 * holds exactly bytes_written bytes and the space beyond is left untouched, in
 * both spaces. h_results[k] receives the k-th outcome. The decoded output is
 * BIT-IDENTICAL whether the context is fresh or reused (including after the
 * staging has grown) and matches a single-stream reference decode.
 *
 * Fail-closed: a NULL ctx, a NULL array, a NULL h_src_ptrs[k] with a non-zero
 * h_src_sizes[k], a NULL dst[k] with a non-zero dst_caps[k], an unknown
 * dst_space, chunk_count == 0, or a batch beyond the launch limit returns
 * CUDEC_ERR_INVALID_ARGUMENT, decodes nothing, and does NOT poison the
 * context. A rejected chunk reports its defined error in h_results[k] with
 * bytes_written 0; neighbours are unaffected and its destination is
 * unspecified but never presented as a valid decode. The aggregate return is
 * CUDEC_OK iff every chunk decoded OK; otherwise a host/device resource
 * failure (CUDEC_ERR_CUDA) takes precedence, then the first non-OK chunk's
 * status in index order.
 *
 * On any non-OK return once the arguments are accepted, every h_results[k]
 * holds a DEFINED non-OK cudec_status and bytes_written 0: its own decode
 * error where the decoder reached the chunk, otherwise CUDEC_ERR_CUDA for a
 * chunk the call did not produce (a staging-grow or CUDA fault) - never a
 * stale or out-of-enum value. (A CUDEC_ERR_INVALID_ARGUMENT reject, where an
 * array argument may itself be NULL, leaves h_results untouched.)
 *
 * A CUDA fault during a decode POISONS the context: it returns CUDEC_ERR_CUDA,
 * every later decode on it returns CUDEC_ERR_CUDA without touching the device,
 * and only cudec_stream_ctx_destroy is valid on it thereafter. Never throws
 * across this boundary. */
cudec_status cudec_lz4_decompress_stream_ctx(
    cudec_stream_ctx* ctx, const void* const* h_src_ptrs,
    const size_t* h_src_sizes, void* const* dst_ptrs, const size_t* dst_caps,
    size_t chunk_count, cudec_mem_space dst_space,
    cudec_chunk_result* h_results);

/* Destroys a streaming-decode context and frees everything it owns. NULL-safe
 * and valid on a poisoned context. After it returns the pointer is dangling. */
void cudec_stream_ctx_destroy(cudec_stream_ctx* ctx);

/* One-shot streaming batch LZ4 block decode: equivalent to
 * cudec_stream_ctx_create, one cudec_lz4_decompress_stream_ctx, and
 * cudec_stream_ctx_destroy. It pays the full staging allocation on every call;
 * a caller decoding repeatedly should hold a context and call the _ctx entry
 * instead. Same arguments, contract, and fail-closed behavior as the _ctx
 * entry (a NULL array, a NULL h_src_ptrs[k] with a non-zero h_src_sizes[k], a
 * NULL dst[k] with a non-zero dst_caps[k], an unknown dst_space, chunk_count
 * == 0, or a batch beyond the launch limit returns CUDEC_ERR_INVALID_ARGUMENT
 * and decodes nothing). Never throws across this boundary. */
cudec_status cudec_lz4_decompress_stream(const void* const* h_src_ptrs,
                                         const size_t* h_src_sizes,
                                         void* const* dst_ptrs,
                                         const size_t* dst_caps,
                                         size_t chunk_count,
                                         cudec_mem_space dst_space,
                                         cudec_chunk_result* h_results);

#ifdef __cplusplus
}
#endif

#endif /* CUDEC_H */
