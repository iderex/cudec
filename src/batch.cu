#include "cudec.h"
#include "lz4_block.h"

#include <cuda_runtime.h>

namespace {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kBlockWarps = 4;
constexpr unsigned kBlockThreads = kWarpSize * kBlockWarps;  /* 128 */

/* Warp-per-chunk LZ4 block decode (masterplan section 9). All 32 lanes run
 * the validated parser redundantly in lockstep - identical bytes, identical
 * arithmetic, identical registers, so no leader lane and no shuffles - and
 * fan out by lane for every copy. Overlapping matches use the closed-form
 * modular gather dst[m+i] = dst[m-off + (i mod off)], which reads only from
 * the region finalized before this match; off >= 1 is guaranteed by the
 * parser's offset==0 rejection, so the modulo never divides by zero. */
__global__ void __launch_bounds__(kBlockThreads)
    lz4_decode_batch(const void* const* src_ptrs, const size_t* src_sizes,
                     void* const* dst_ptrs, const size_t* dst_caps,
                     size_t chunk_count, cudec_chunk_result* results) {
    const unsigned lane = threadIdx.x % kWarpSize;
    /* The global thread id fits in 32 bits because the host caps the grid
     * at kMaxBlocks (8192) x kBlockThreads (128) ~ 1.05M threads; if that
     * cap were ever raised past ~33M blocks (~4.3B threads), widen this
     * to size_t. */
    const size_t warp_in_grid =
        (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const size_t total_warps =
        (static_cast<size_t>(gridDim.x) * blockDim.x) / kWarpSize;

    for (size_t chunk = warp_in_grid; chunk < chunk_count;
         chunk += total_warps) {
        const unsigned char* src =
            static_cast<const unsigned char*>(src_ptrs[chunk]);
        /* dst must NOT be __restrict__-qualified: the cross-lane read after
         * write below relies on __syncwarp() forcing a reload from global
         * memory, and __restrict__ could legally let the compiler cache a
         * dst[] value across the barrier, silently breaking lane-to-lane
         * visibility. */
        unsigned char* dst = static_cast<unsigned char*>(dst_ptrs[chunk]);

        cudec_detail::Lz4Parser parser{src, src_sizes[chunk], dst_caps[chunk]};
        cudec_detail::Lz4Sequence seq;
        cudec_status status = CUDEC_OK;
        bool done = false;
        while (true) {
            status = parser.Next(&seq, &done);
            if (status != CUDEC_OK) {
                break;
            }
            for (uint64_t i = lane; i < seq.literals_len; i += kWarpSize) {
                dst[seq.literals_dst + i] = src[seq.literals_src + i];
            }
            /* The match may read literal bytes just written above. */
            __syncwarp();
            if (seq.match_len != 0) {
                const uint64_t offset = seq.match_dst - seq.match_src;
                for (uint64_t i = lane; i < seq.match_len; i += kWarpSize) {
                    dst[seq.match_dst + i] =
                        dst[seq.match_src + (i % offset)];
                }
                /* The next sequence may read bytes this match wrote. */
                __syncwarp();
            }
            if (done) {
                break;
            }
        }

        /* All lanes agree on status and dst_pos (redundant parse); one lane
         * writes the 16-byte result. bytes_written is set on full success
         * only - a rejected chunk reports zero and never presents its
         * partial output as valid. */
        if (lane == 0) {
            results[chunk].status = status;
            results[chunk].reserved = 0;
            results[chunk].bytes_written =
                (status == CUDEC_OK) ? parser.dst_pos : 0;
        }
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
     * bound rejects absurd counts while staying well under SIZE_MAX (the
     * grid is capped independently, so the launch geometry never
     * overflows); d_results is 16-byte aligned so each per-chunk result
     * record lands in a single aligned 16-byte slot. */
    constexpr size_t kMaxChunks = static_cast<size_t>(INT32_MAX) * kWarpSize;
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

    /* One warp per chunk; the grid is sized to cover the batch in a single
     * wave up to a cap that already fills the target GPU, and the kernel's
     * grid-stride loop handles any remainder. */
    constexpr size_t kMaxBlocks = 8192;
    size_t blocks = (chunk_count + kBlockWarps - 1) / kBlockWarps;
    if (blocks > kMaxBlocks) {
        blocks = kMaxBlocks;
    }
    lz4_decode_batch<<<static_cast<unsigned>(blocks), kBlockThreads, 0,
                       stream>>>(d_src_ptrs, d_src_sizes, d_dst_ptrs,
                                 d_dst_capacities, chunk_count, d_results);
    return cudaGetLastError() == cudaSuccess ? CUDEC_OK : CUDEC_ERR_CUDA;
}
