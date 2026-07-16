#include "cudec.h"

#include <cuda_runtime.h>

namespace {

constexpr unsigned kBlockThreads = 256;

/* M1 replaces this with the LZ4 decoder; the stub proves the batch
 * contract end to end (validation, launch, per-chunk device reporting). */
__global__ void report_not_implemented(cudec_chunk_result* results,
                                       size_t chunk_count) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < chunk_count) {
        results[i].status = CUDEC_ERR_NOT_IMPLEMENTED;
        results[i].reserved = 0;
        results[i].bytes_written = 0;
    }
}

}  // namespace

cudec_status cudec_lz4_decompress_batch(const void* const* d_src_ptrs,
                                        const size_t* d_src_sizes,
                                        void* const* d_dst_ptrs,
                                        const size_t* d_dst_capacities,
                                        size_t chunk_count,
                                        cudec_chunk_result* d_results,
                                        cudec_stream_t stream) {
    /* Fail closed: reject the whole call rather than guess at intent. The
     * grid bound keeps the block count inside gridDim.x's int32 range; the
     * alignment bound keeps a single 16-byte result store per chunk open
     * for the real decoder. */
    constexpr size_t kMaxChunks =
        static_cast<size_t>(INT32_MAX) * kBlockThreads;
    static_assert(kMaxChunks < SIZE_MAX,
                  "the SIZE_MAX over-limit contract test relies on this");
    if (d_src_ptrs == nullptr || d_src_sizes == nullptr ||
        d_dst_ptrs == nullptr || d_dst_capacities == nullptr ||
        d_results == nullptr ||
        reinterpret_cast<uintptr_t>(d_results) % 16 != 0 || chunk_count == 0 ||
        chunk_count > kMaxChunks) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }

    /* Drain any error already pending on this thread so the post-launch
     * check reports this submission alone; the header documents that the
     * call consumes the pending error state. */
    (void)cudaGetLastError();

    const unsigned grid_blocks =
        static_cast<unsigned>((chunk_count + kBlockThreads - 1) / kBlockThreads);
    report_not_implemented<<<grid_blocks, kBlockThreads, 0, stream>>>(
        d_results, chunk_count);
    return cudaGetLastError() == cudaSuccess ? CUDEC_OK : CUDEC_ERR_CUDA;
}
