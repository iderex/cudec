/* The warp-per-chunk LZ4 decode kernel, templated on the copy mode so the
 * shipped decoder and the bench's parse-only ceiling share ONE parse (an
 * honest ceiling - masterplan section 9). The public entry (src/batch.cu)
 * instantiates Full; the bench instantiates ParseOnly. Each translation
 * unit emits only the instantiation it launches, so the shipped library
 * carries no parse-only kernel. Internal header, not part of the ABI. */
#ifndef CUDEC_LZ4_DECODE_CUH
#define CUDEC_LZ4_DECODE_CUH

#include "cudec.h"
#include "lz4_block.h"

#include <cuda_runtime.h>

namespace cudec_detail {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kBlockWarps = 4;
constexpr unsigned kBlockThreads = kWarpSize * kBlockWarps;  /* 128 */

/* One warp per chunk over a grid-stride loop. All 32 lanes run the
 * validated parser redundantly in lockstep and fan out by lane for every
 * copy; overlapping matches use the closed-form modular gather
 * dst[m+i] = dst[m-off + (i mod off)] (off >= 1 by the parser's offset==0
 * rejection, so no modulo-by-zero). __syncwarp() carries intra-warp memory
 * ordering at the two copy boundaries. ParseOnly elides the copies and the
 * syncs to isolate the parse cost - it writes no output and is a bench
 * ceiling only, never shipped. */
template <bool ParseOnly>
__global__ void __launch_bounds__(kBlockThreads)
    lz4_decode_batch(const void* const* src_ptrs, const size_t* src_sizes,
                     void* const* dst_ptrs, const size_t* dst_caps,
                     size_t chunk_count, cudec_chunk_result* results) {
    const unsigned lane = threadIdx.x % kWarpSize;
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
         * dst[] value across the barrier. */
        unsigned char* dst = static_cast<unsigned char*>(dst_ptrs[chunk]);

        Lz4Parser parser{src, src_sizes[chunk], dst_caps[chunk]};
        Lz4Sequence seq;
        cudec_status status = CUDEC_OK;
        bool done = false;
        while (true) {
            status = parser.Next(&seq, &done);
            if (status != CUDEC_OK) {
                break;
            }
            if constexpr (!ParseOnly) {
                for (uint64_t i = lane; i < seq.literals_len; i += kWarpSize) {
                    dst[seq.literals_dst + i] = src[seq.literals_src + i];
                }
                /* The match may read literal bytes just written above. */
                __syncwarp();
                if (seq.match_len != 0) {
                    const uint64_t offset = seq.match_dst - seq.match_src;
                    for (uint64_t i = lane; i < seq.match_len;
                         i += kWarpSize) {
                        dst[seq.match_dst + i] =
                            dst[seq.match_src + (i % offset)];
                    }
                    /* The next sequence may read bytes this match wrote. */
                    __syncwarp();
                }
            }
            if (done) {
                break;
            }
        }

        /* All lanes agree on status and dst_pos (redundant parse); one lane
         * writes the 16-byte result. bytes_written is set on full success
         * only - a rejected chunk reports zero and never presents partial
         * output as valid. */
        if (lane == 0) {
            results[chunk].status = status;
            results[chunk].reserved = 0;
            results[chunk].bytes_written =
                (status == CUDEC_OK) ? parser.dst_pos : 0;
        }
    }
}

/* One warp per chunk up to a cap that already fills the target GPU; the
 * grid-stride loop covers any remainder. */
inline unsigned decode_grid_blocks(size_t chunk_count) {
    constexpr size_t kMaxBlocks = 8192;
    size_t blocks = (chunk_count + kBlockWarps - 1) / kBlockWarps;
    return static_cast<unsigned>(blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

/* Fail-closed argument validation, shared by the public entry and the
 * bench. The bound rejects absurd counts while staying under SIZE_MAX (the
 * grid is capped independently); d_results is 16-byte aligned so each
 * per-chunk result record lands in a single aligned slot. */
inline cudec_status validate_batch_args(const void* const* d_src_ptrs,
                                        const size_t* d_src_sizes,
                                        void* const* d_dst_ptrs,
                                        const size_t* d_dst_capacities,
                                        size_t chunk_count,
                                        cudec_chunk_result* d_results) {
    constexpr size_t kMaxChunks = static_cast<size_t>(INT32_MAX) * kWarpSize;
    static_assert(kMaxChunks < SIZE_MAX,
                  "the SIZE_MAX over-limit contract test relies on this");
    if (d_src_ptrs == nullptr || d_src_sizes == nullptr ||
        d_dst_ptrs == nullptr || d_dst_capacities == nullptr ||
        d_results == nullptr ||
        reinterpret_cast<uintptr_t>(d_results) % 16 != 0 ||
        chunk_count == 0 || chunk_count > kMaxChunks) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_LZ4_DECODE_CUH */
