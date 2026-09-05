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

/* The one place the version is written. The top-level CMakeLists reads these
 * three macros to set the project version, so a bump here is the whole bump
 * and nothing restates the number. */
#define CUDEC_VERSION_MAJOR 0
#define CUDEC_VERSION_MINOR 1
#define CUDEC_VERSION_PATCH 0

/* Returns the runtime library version as (major * 10000 +
 * minor * 100 + patch), for ABI sanity checks against these macros. */
int cudec_version(void);

typedef enum cudec_status {
    CUDEC_OK = 0,
    CUDEC_ERR_INVALID_ARGUMENT = 1,
    CUDEC_ERR_CORRUPT_INPUT = 2,
    CUDEC_ERR_OUTPUT_TOO_SMALL = 3,
    CUDEC_ERR_CUDA = 4,
    /* An API entry point or decode path declared here but not implemented
     * in this build. It is what a declared-but-unbuilt entry answers a
     * batch it has already accepted, so the freeze on that entry's symbol
     * and signature never depends on a build configuration:
     * cudec_zstd_decompress_batch returns it today, and the value is fixed
     * so a caller's switch stays exhaustive across the build that stops
     * returning it - which cudec_gdeflate_decompress_batch did when its
     * kernel landed, with the symbol and every reject class unmoved. */
    CUDEC_ERR_NOT_IMPLEMENTED = 5,
    /* A well-formed frame that uses a feature cudec does not decode, or a
     * legal frame type it declines (block-linked mode, a dictionary id, a
     * skippable frame). Distinct from CORRUPT_INPUT: the input is valid,
     * just outside cudec's supported subset. */
    CUDEC_ERR_UNSUPPORTED = 6
} cudec_status;

/* Binary-compatible with cudaStream_t without pulling in the CUDA headers:
 * both are pointers to the driver's CUstream_st. Pass a cudaStream_t
 * directly; NULL means the legacy default stream (callers built with
 * per-thread default streams pass cudaStreamPerThread explicitly).
 *
 * The same type on the HIP build: this header carries no backend define, so
 * the ABI a consumer compiles against is one pointer on either build. A HIP
 * caller passes its hipStream_t through this typedef with a cast - both
 * handles are a pointer to the driver's stream object, NULL is the default
 * stream on both, and the per-thread sentinel is hipStreamPerThread there. */
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
 * truncated). That limit is a fixed property of this ABI: it is the same
 * number on every backend and every device, and in particular it does not
 * widen on a GPU whose wave is 64 lanes rather than 32. A chunk_count this
 * call accepts is accepted everywhere, and one it refuses is refused
 * everywhere. A rejected call makes no CUDA call and leaves the thread's
 * pending CUDA error state untouched; a call that passes validation
 * consumes that pending state (cudaGetLastError semantics), so the
 * returned status reflects this submission alone.
 *
 * A call that passes validation then asks the runtime for the current
 * device's wave width and launches the kernel built for it. Two synchronous
 * failures come from that step and neither one launches: the query itself
 * failing returns CUDEC_ERR_CUDA, and a device reporting a wave width this
 * build has no kernel for returns CUDEC_ERR_UNSUPPORTED. The second one is
 * refused rather than approximated on purpose - launching a kernel built for
 * a different width would map chunks onto lanes that are not there, so the
 * refusal is the fail-closed answer and not a missing feature that will be
 * papered over later. Neither status is reachable on any CUDA device, all of
 * which report 32. */
cudec_status cudec_lz4_decompress_batch(const void* const* d_src_ptrs,
                                        const size_t* d_src_sizes,
                                        void* const* d_dst_ptrs,
                                        const size_t* d_dst_capacities,
                                        size_t chunk_count,
                                        cudec_chunk_result* d_results,
                                        cudec_stream_t stream);

/* Batch Snappy block decode. The contract above holds argument for
 * argument: the same device-side arrays, the same per-chunk result
 * semantics, the same synchronous validation rejects with the same
 * CUDEC_ERR_INVALID_ARGUMENT and the same launch limit, the same
 * asynchronous launch on `stream`, and the same pending-CUDA-error
 * semantics. Nothing about the batch contract varies by format.
 *
 * The supported surface is the RAW Snappy stream - the varint-prefixed
 * block google/snappy's Compress and Uncompress produce. The Snappy FRAMING
 * format (stream identifier, framed chunks, masked CRC-32C) is not
 * decoded here and no entry point decodes it, so a framed stream is a
 * malformed raw one and is rejected as CUDEC_ERR_CORRUPT_INPUT rather than
 * partially decoded.
 *
 * The declared uncompressed length that opens every raw stream is
 * attacker-controlled and is never used to size anything: it is checked
 * against that chunk's d_dst_capacities entry before an element is parsed,
 * and a declaration above the capacity reports CUDEC_ERR_OUTPUT_TOO_SMALL.
 * Every later overrun of the declared length reports
 * CUDEC_ERR_CORRUPT_INPUT: once a stream has declared its own length, an
 * element that runs past it is inconsistent rather than short of room, and
 * a larger destination would not make it valid. */
cudec_status cudec_snappy_decompress_batch(const void* const* d_src_ptrs,
                                           const size_t* d_src_sizes,
                                           void* const* d_dst_ptrs,
                                           const size_t* d_dst_capacities,
                                           size_t chunk_count,
                                           cudec_chunk_result* d_results,
                                           cudec_stream_t stream);

/* Batch GDeflate page decode. The batch contract above holds argument for
 * argument: the same device-side arrays of chunk_count entries holding
 * device pointers, the same 16-byte-aligned d_results, the same per-chunk
 * result semantics, the same synchronous CUDEC_ERR_INVALID_ARGUMENT reject
 * classes against the same launch limit, and the same asynchronous launch
 * on `stream`. Nothing about the batch contract varies by format.
 *
 * The supported surface is the RAW GDeflate page - the unit the format
 * decompresses into one 64 KiB tile - with its compressed size in
 * d_src_sizes and its output capacity in d_dst_capacities, both supplied
 * by the caller. No container is parsed here and none ever will be: the
 * TileStream envelope that records where a file's pages are is a
 * caller-side layer, so a whole TileStream handed to this entry is a
 * malformed page and is rejected as CUDEC_ERR_CORRUPT_INPUT rather than
 * stepped into. The bound on every write is that page's
 * d_dst_capacities entry and never a length read out of the page.
 *
 * Per page, and isolated to that page: CUDEC_ERR_CORRUPT_INPUT for a
 * malformed page, CUDEC_ERR_OUTPUT_TOO_SMALL when the decode would pass
 * the supplied capacity, bytes_written == 0 on either, and the
 * destination contents then unspecified but never presented as a valid
 * decode. A page is decoded by one warp, its 32 lanes being the format's
 * 32 substreams, and a page that overlaps another chunk's destination is
 * the caller's error exactly as it is for the two entries above.
 *
 * The contract was frozen before the kernel stood behind it, and the
 * freeze held: the symbol, the signature and every reject class above did
 * not move when the kernel landed; only the answer to an accepted batch
 * did, from CUDEC_ERR_NOT_IMPLEMENTED to the launch. */
cudec_status cudec_gdeflate_decompress_batch(const void* const* d_src_ptrs,
                                             const size_t* d_src_sizes,
                                             void* const* d_dst_ptrs,
                                             const size_t* d_dst_capacities,
                                             size_t chunk_count,
                                             cudec_chunk_result* d_results,
                                             cudec_stream_t stream);

/* Batch Zstd frame decode. The batch contract above holds argument for
 * argument: the same device-side arrays of chunk_count entries holding
 * device pointers, the same 16-byte-aligned d_results, the same per-chunk
 * result semantics, the same synchronous CUDEC_ERR_INVALID_ARGUMENT reject
 * classes against the same launch limit, and the same asynchronous launch
 * on `stream`. Nothing about the batch contract varies by format.
 *
 * THE SUPPORTED UNIT IS ONE WHOLE FRAME PER CHUNK, and that is the format's
 * choice rather than this ABI's. Blocks inside a frame are not independent -
 * a match reaches back across a block boundary and entropy tables carry from
 * one block to the next - while frames are, so the frame is the smallest
 * thing this entry can be handed. A chunk holding several concatenated
 * frames, or part of one, is a malformed frame and is rejected as such
 * rather than stepped into. d_src_sizes carries the frame's compressed size
 * and d_dst_capacities the caller's output capacity; the bound on every
 * write is that capacity and never a length read out of the frame.
 *
 * The accepted envelope is docs/MASTERPLAN.md section 12.2 and is not
 * restated here, because a copy of that table in this header would drift
 * against the parser that decides it. What the classes mean does belong
 * here: CUDEC_ERR_UNSUPPORTED names a frame this decoder declines - a
 * skippable frame, a dictionary id, an absent content size, a window past
 * the range this decoder authorises - and CUDEC_ERR_CORRUPT_INPUT names one
 * that is malformed. The two are different answers because only one of them
 * can be retried elsewhere.
 *
 * Per frame, and isolated to that frame: those two statuses,
 * CUDEC_ERR_OUTPUT_TOO_SMALL when the decode would pass the supplied
 * capacity, bytes_written == 0 on any of them, and the destination contents
 * then unspecified but never presented as a valid decode.
 *
 * THIS BUILD CARRIES NO ZSTD KERNEL, AND THAT IS A DEFINED STATUS RATHER
 * THAN AN ABSENT SYMBOL. A batch that passes the validation above returns
 * CUDEC_ERR_NOT_IMPLEMENTED, having made no CUDA call at all - so this
 * entry, like the GDeflate one above, also leaves the thread's pending CUDA
 * error state untouched on a call it accepts. The symbol, the signature and
 * every reject class above are frozen now and do not move when the kernel
 * lands; only the answer to an accepted batch does. A symbol that was absent
 * instead would make the freeze conditional on a build configuration, which
 * is two contracts wearing one name. */
cudec_status cudec_zstd_decompress_batch(const void* const* d_src_ptrs,
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
 * (block-linked mode - the default of liblz4's frame compressor - a
 * dictionary id, or a skippable frame, which is a legal member of the
 * container this entry declines to step over),
 * CUDEC_ERR_CORRUPT_INPUT for a malformed frame, a checksum
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

/* Reads the GDeflate TileStream envelope at `stream[0 .. stream_size)` and
 * reports what a decode of it would produce: the number of tiles it declares
 * and the total uncompressed size those tiles decode to. Host-only - it makes
 * no CUDA call, touches no device, and reads no tile payload - so a caller
 * can size its destination and its per-tile result array on a machine with no
 * GPU.
 *
 * THE ENVELOPE IS NOT THE PAGE FORMAT, AND THIS ENTRY IS THE SEAM. A
 * TileStream is the DirectStorage-side container that records where a file's
 * GDeflate pages are; the pages themselves are what
 * cudec_gdeflate_decompress_batch decodes. Nothing about the container
 * crosses into device code, which is why this entry can answer without one.
 *
 * On CUDEC_OK, *tile_count and *uncompressed_size hold the envelope's
 * declaration, each already checked against the bytes actually present.
 * CUDEC_ERR_INVALID_ARGUMENT for a NULL argument; CUDEC_ERR_CORRUPT_INPUT for
 * anything the bytes get wrong - a truncated header or table, an unknown tile
 * size index, a reserved bit set, a tile count of zero, a table whose offsets
 * do not strictly ascend, or a last tile running past the end; and
 * CUDEC_ERR_CUDA for a HOST resource failure, since sizing the tile table is
 * an allocation and a failed one must cross this boundary as a status rather
 * than as a throw. That last one is named despite the entry making no CUDA
 * call, because the enum's spelling is older than the distinction and a
 * caller switching on the set below must not fall through. On any error
 * *tile_count and *uncompressed_size are 0. The container carries no optional
 * features, so nothing here returns CUDEC_ERR_UNSUPPORTED.
 *
 * THE CEILING IS WORTH KNOWING BEFORE SIZING ANYTHING FROM THE ANSWER. The
 * count field is 16 bits and a tile is 65536 bytes, so a 320 KB envelope can
 * legitimately report an uncompressed size of 4294901760. The number is the
 * container's declaration, not a promise that the payload is there; the decode
 * refuses a stream whose tiles do not produce exactly it. */
cudec_status cudec_gdeflate_tilestream_info(const void* stream,
                                            size_t stream_size,
                                            size_t* tile_count,
                                            size_t* uncompressed_size);

/* Decodes a whole GDeflate TileStream on a reusable streaming context: the
 * envelope is parsed on the host, and its tiles are driven through the
 * GDeflate page batch decoder as independent pages. Synchronous - the work is
 * drained before return, so `dst`, `*bytes_written` and every
 * h_tile_results[i] are valid on return.
 *
 * `stream`/`stream_size` is the whole container in HOST memory. The decoded
 * tiles are written back to back into `dst`, in tile order, in the space named
 * by dst_space: CUDEC_MEM_DEVICE writes device memory directly, CUDEC_MEM_HOST
 * reads the bytes back into host memory. `dst_capacity` must be at least the
 * `uncompressed_size` cudec_gdeflate_tilestream_info reports for the same
 * bytes, and h_tile_results must be a HOST array of at least that entry's
 * `tile_count` records; both are checked against the parsed envelope and
 * refused rather than truncated.
 *
 * THE STAGING IS THE STREAMING CONTEXT'S, NOT A SECOND ONE. The context owns
 * the CUDA stream and the grow-only pinned/device staging, so a caller
 * decoding many streams pays the allocation once; that is the whole reason
 * this entry takes a context rather than being one-shot only.
 *
 * EVERY TILE MUST PRODUCE EXACTLY WHAT THE ENVELOPE DECLARED FOR IT. The
 * declared size is a bound in both directions, and the second direction is the
 * one nothing else enforces: a GDeflate page ends at its final block and
 * reports what it produced without comparing that against the capacity it was
 * given, so a page decoding to fewer bytes than its tile was declared to hold
 * is a well-formed page inside a container that lied about it. Accepted, it
 * would leave the gap up to the next tile's declared offset holding whatever
 * was in `dst` - and a decoder whose output depends on that is neither
 * fail-closed nor deterministic. Such a tile is CUDEC_ERR_CORRUPT_INPUT, the
 * same as one that overruns.
 *
 * A tile that fails to decode is isolated to itself: h_tile_results[i] carries
 * that tile's own defined status with bytes_written 0, its siblings decode
 * normally and report their own bytes, and nothing about the failed tile's
 * region of `dst` is presented as valid output. On CUDEC_MEM_DEVICE that
 * region may hold bytes the kernel wrote before it refused; they are not
 * presented - *bytes_written is 0 and the tile's status is not OK - and
 * nothing promises they were left alone.
 *
 * THE WHOLE-STREAM ANSWER FOLLOWS THE FRAME PRECEDENT RATHER THAN THE BATCH
 * ONE, because this entry, like cudec_lz4f_decompress, produces ONE object out
 * of many pieces. A stream with any failed tile is not a partially decoded
 * file: *bytes_written is 0 and the return is non-OK - CUDEC_ERR_CUDA where a
 * device or host resource failed, CUDEC_ERR_UNSUPPORTED where the page decoder
 * declines the device it was asked to launch on, otherwise the first non-OK
 * tile's status in tile order. *bytes_written equals the envelope's declared
 * total, and is reported only when every tile decoded exactly its declared
 * size - it is derived from the validated envelope rather than summed from the
 * device-written records, because it is a length the caller will read `dst`
 * with.
 *
 * Fail-closed, synchronously and with nothing launched: a NULL ctx, a NULL
 * `stream`, `bytes_written` or h_tile_results, a NULL `dst` with a non-zero
 * `dst_capacity`, an unknown dst_space, or an envelope this build refuses
 * returns CUDEC_ERR_INVALID_ARGUMENT or CUDEC_ERR_CORRUPT_INPUT before any
 * CUDA call. A `dst_capacity` below the envelope's total returns
 * CUDEC_ERR_OUTPUT_TOO_SMALL, and an h_tile_results array shorter than the
 * declared tile count returns CUDEC_ERR_INVALID_ARGUMENT; both are decided
 * from the parsed envelope, so both are content-dependent rejects and neither
 * writes a byte of `dst`.
 *
 * WHERE h_tile_results IS WRITTEN, stated because CUDEC_OK is zero and a
 * caller cannot otherwise tell an untouched array from a decoded one. Every
 * reject above - the argument classes, CUDEC_ERR_CORRUPT_INPUT,
 * CUDEC_ERR_OUTPUT_TOO_SMALL, the short results array - leaves it UNTOUCHED,
 * because none of them has accepted the call. From the point the call is
 * accepted onward every entry in [0, tile_count) holds a defined
 * cudec_status: its own decode outcome, or CUDEC_ERR_CUDA for a tile the call
 * did not produce. Entries past tile_count are never written.
 *
 * A CUDA fault POISONS the context exactly as the LZ4 streaming entry
 * documents: only cudec_stream_ctx_destroy is valid on it thereafter, and
 * every later decode on it refuses without touching the device. The refusal is
 * CUDEC_ERR_CUDA once the envelope has been accepted - the envelope walk runs
 * first and is pure host arithmetic, so a poisoned context handed malformed
 * bytes answers for the bytes. Fail-closed either way, and worth knowing for a
 * caller using the return value to detect poisoning. Never throws across this
 * boundary. */
cudec_status cudec_gdeflate_tilestream_decompress_ctx(
    cudec_stream_ctx* ctx, const void* stream, size_t stream_size, void* dst,
    size_t dst_capacity, cudec_mem_space dst_space,
    cudec_chunk_result* h_tile_results, size_t tile_results_capacity,
    size_t* bytes_written);

/* One-shot whole-TileStream decode: equivalent to cudec_stream_ctx_create,
 * one cudec_gdeflate_tilestream_decompress_ctx, and
 * cudec_stream_ctx_destroy. It pays the full staging allocation on every
 * call; a caller decoding repeatedly should hold a context and call the _ctx
 * entry instead. Same arguments, contract and fail-closed behavior as the
 * _ctx entry, and every synchronous reject class above is answered here
 * without a context ever being created. Never throws across this
 * boundary. */
cudec_status cudec_gdeflate_tilestream_decompress(
    const void* stream, size_t stream_size, void* dst, size_t dst_capacity,
    cudec_mem_space dst_space, cudec_chunk_result* h_tile_results,
    size_t tile_results_capacity, size_t* bytes_written);

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
