#include "cudec.h"
#include "chunk_decode.cuh"
#include "lz4_block.h"
#include "snappy_block.h"

#include <cuda_runtime.h>

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
    (void)cudaGetLastError();

    cudec_detail::chunk_decode_batch<Parser, false>
        <<<cudec_detail::decode_grid_blocks(chunk_count),
           cudec_detail::kBlockThreads, 0, stream>>>(
            d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
            d_results);
    return cudaGetLastError() == cudaSuccess ? CUDEC_OK : CUDEC_ERR_CUDA;
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
