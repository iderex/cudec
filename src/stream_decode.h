/* The streaming decode core, shared by every format that drives the reusable
 * context (issue #177). Internal header, not part of the ABI.
 *
 * WHY THIS SEAM EXISTS AT ALL. src/stream.cpp owns the only staging path in
 * the library that grows once and is reused: a pinned/device source pair, the
 * metadata pair, the host-output destination staging, the result buffer and
 * its pinned mirror, waved so a hostile chunk_count cannot drive per-wave
 * growth. Every one of those is format-blind - the only line in the wave loop
 * that names a format is the batch entry it launches. A second format that
 * copied the loop to change that one line would be a second staging path to
 * keep in agreement with this one, and the two would drift on exactly the
 * questions (poisoning, the not-produced seed, the drain) that are hardest to
 * notice going wrong.
 *
 * A FUNCTION POINTER RATHER THAN A TEMPLATE, deliberately, and the opposite of
 * what src/batch.cu does for its launchers. There the indirection would cost
 * the device compiler the inlining the parser depends on; here the callee is
 * a host function that submits one launch per wave, so the call is paid once
 * per wave against a kernel launch and buys one instantiation of a 250-line
 * body instead of one per format. */
#ifndef CUDEC_STREAM_DECODE_H
#define CUDEC_STREAM_DECODE_H

#include "cudec.h"

#include <stddef.h>

namespace cudec_stream_detail {

/* The shape of every batch entry in include/cudec.h. Spelled from the ABI
 * rather than from any one entry, so a signature change reds every format at
 * once instead of silently binding the wrong one. */
typedef cudec_status (*BatchEntry)(const void* const* d_src_ptrs,
                                   const size_t* d_src_sizes,
                                   void* const* d_dst_ptrs,
                                   const size_t* d_dst_capacities,
                                   size_t chunk_count,
                                   cudec_chunk_result* d_results,
                                   cudec_stream_t stream);

/* Drives `entry` over the whole batch on `ctx`'s stream and staging. The
 * arguments after `entry` are the streaming ABI's, argument for argument, and
 * the contract is the one cudec_lz4_decompress_stream_ctx documents: a NULL
 * ctx or a malformed argument is refused synchronously with no CUDA call and
 * no poisoning, a poisoned context refuses with the per-chunk channel stamped
 * fail-closed, a per-chunk reject is reported in h_results and does not
 * poison, and a CUDA-level fault poisons and returns CUDEC_ERR_CUDA. Never
 * throws: a host allocation failure inside becomes CUDEC_ERR_CUDA. */
cudec_status DecodeOnStreamCtx(cudec_stream_ctx* ctx, BatchEntry entry,
                               const void* const* h_src_ptrs,
                               const size_t* h_src_sizes,
                               void* const* dst_ptrs, const size_t* dst_caps,
                               size_t chunk_count, cudec_mem_space dst_space,
                               cudec_chunk_result* h_results);

}  // namespace cudec_stream_detail

#endif /* CUDEC_STREAM_DECODE_H */
