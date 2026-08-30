#include "cudec.h"
#include "chunk_decode.cuh"
#include "lz4_block.h"
#include "snappy_block.h"

#include "vendor_rt.h"

namespace {

/* The two entries differ in one template argument and in nothing else, so
 * the body they share is written once. Templated rather than handed a
 * function pointer: a launch through an indirect call would put the kernel
 * address in device memory and cost the compiler the inlining the parser
 * depends on. */
template <class Parser>
cudec_status submit_batch(const void* const* d_src_ptrs,
                          const size_t* d_src_sizes, void* const* d_dst_ptrs,
                          const size_t* d_dst_capacities, size_t chunk_count,
                          cudec_chunk_result* d_results,
                          cudec_stream_t stream) {
    const cudec_status valid = cudec_detail::validate_batch_args(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results);
    if (valid != CUDEC_OK) {
        return valid;
    }

    /* Drain any error already pending on this thread so the post-launch
     * check reports this submission alone; the header documents that the
     * call consumes the pending error state. */
    (void)cudec_rt::get_last_error();

    cudec_detail::chunk_decode_batch<Parser, false, cudec_detail::kCudaWaveSize>
        <<<cudec_detail::decode_grid_blocks(chunk_count),
           cudec_detail::kBlockThreads, 0, stream>>>(
            d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
            d_results);
    return cudec_rt::get_last_error() == cudec_rt::success ? CUDEC_OK
                                                           : CUDEC_ERR_CUDA;
}

}  // namespace

cudec_status cudec_lz4_decompress_batch(const void* const* d_src_ptrs,
                                        const size_t* d_src_sizes,
                                        void* const* d_dst_ptrs,
                                        const size_t* d_dst_capacities,
                                        size_t chunk_count,
                                        cudec_chunk_result* d_results,
                                        cudec_stream_t stream) {
    return submit_batch<cudec_detail::Lz4Parser>(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results, stream);
}

cudec_status cudec_snappy_decompress_batch(const void* const* d_src_ptrs,
                                           const size_t* d_src_sizes,
                                           void* const* d_dst_ptrs,
                                           const size_t* d_dst_capacities,
                                           size_t chunk_count,
                                           cudec_chunk_result* d_results,
                                           cudec_stream_t stream) {
    return submit_batch<cudec_detail::SnappyParser>(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results, stream);
}

/* The GDeflate entry is the frozen contract without the kernel behind it
 * yet (issue #216). It shares the validator with the two above rather than
 * growing its own, so the reject classes cannot drift apart while they are
 * being frozen, and then stops: no launch, and deliberately no
 * cudec_rt::get_last_error() drain either. Draining is how the entries above
 * buy a
 * post-launch check that reports their own submission; with nothing
 * submitted there is nothing to report, and consuming a caller's pending
 * error on the way to answering "not built here" would be this entry
 * altering CUDA state it never used.
 *
 * That absence is what tests/launch_fail.cpp reads: with no visible device
 * an entry that made any CUDA call would answer CUDEC_ERR_CUDA, so the
 * not-implemented answer there is the evidence that this path touches the
 * runtime at all only through the validator, which touches it not at
 * all. */
cudec_status cudec_gdeflate_decompress_batch(const void* const* d_src_ptrs,
                                             const size_t* d_src_sizes,
                                             void* const* d_dst_ptrs,
                                             const size_t* d_dst_capacities,
                                             size_t chunk_count,
                                             cudec_chunk_result* d_results,
                                             cudec_stream_t stream) {
    (void)stream;
    const cudec_status valid = cudec_detail::validate_batch_args(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results);
    if (valid != CUDEC_OK) {
        return valid;
    }
    return CUDEC_ERR_NOT_IMPLEMENTED;
}
