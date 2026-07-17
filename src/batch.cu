#include "cudec.h"
#include "lz4_decode.cuh"

#include <cuda_runtime.h>

cudec_status cudec_lz4_decompress_batch(const void* const* d_src_ptrs,
                                        const size_t* d_src_sizes,
                                        void* const* d_dst_ptrs,
                                        const size_t* d_dst_capacities,
                                        size_t chunk_count,
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

    cudec_detail::lz4_decode_batch<false>
        <<<cudec_detail::decode_grid_blocks(chunk_count),
           cudec_detail::kBlockThreads, 0, stream>>>(
            d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
            d_results);
    return cudaGetLastError() == cudaSuccess ? CUDEC_OK : CUDEC_ERR_CUDA;
}
