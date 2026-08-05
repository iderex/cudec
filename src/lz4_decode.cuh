/* The warp-per-chunk LZ4 decode kernel, templated on the copy mode so the
 * shipped decoder and the bench's parse-only ceiling share ONE parse (an
 * honest ceiling - masterplan section 9). The public entry (src/batch.cu)
 * instantiates Full; the bench instantiates ParseOnly. Each translation
 * unit emits only the instantiation it launches, so the shipped library
 * carries no parse-only kernel. Internal header, not part of the ABI. */
#ifndef CUDEC_LZ4_DECODE_CUH
#define CUDEC_LZ4_DECODE_CUH

#include "batch_limits.h"
#include "cudec.h"
#include "lz4_block.h"

#include <cuda_runtime.h>

namespace cudec_detail {

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
    /* The chunk-to-lane mapping is sound only when the grid holds at least
     * one whole warp and every block is a whole number of warps. Both
     * failures are geometry rather than input, and both are silent:
     *  - fewer than one whole warp in the grid makes the grid-stride below
     *    advance by zero and spin forever;
     *  - a block that is not a warp multiple splits a chunk's warp across two
     *    blocks (<<<2, 16>>> and <<<2, 48>>> both do), so `lane` only ever
     *    takes the low half of its range while the copy loops stride by
     *    kWarpSize - every output byte at i % 32 >= 16 is written by nobody,
     *    and the full-mask __syncwarp() is executed by half a warp.
     * The shipped entry always launches kBlockThreads, but the kernel must
     * not depend on its caller for either property. Both tests are
     * grid-uniform (gridDim/blockDim only), so neither can strand lanes at a
     * __syncwarp().
     *
     * warp_in_grid above stays 32-bit, so this kernel additionally requires
     * gridDim.x * blockDim.x <= 2^32. That bound is DOCUMENTED rather than
     * enforced, and the reason is the asymmetry in what breaking it costs:
     * past it the index wraps modulo 2^32, but the guard below has already
     * forced blockDim.x to a multiple of kWarpSize, so the wrapped base stays
     * warp-aligned and the aliased block recomputes the SAME warp_in_grid
     * values with a full 0-31 lane range. Two blocks then decode one chunk
     * redundantly and write byte-identical values to the same addresses -
     * duplicated work, not missing bytes and not a hang, and the output stays
     * bit-identical. Enforcing it costs an occupancy step: three spellings
     * were measured (widening the expression, `total_warps > 1 << 27`, and a
     * pure 32-bit `gridDim.x > UINT_MAX / blockDim.x`) and all three take the
     * kernel from 48 to 52 registers, dropping sm_86 from 40 to 36 warps/SM
     * (docs/BENCHMARKS.md). Trading real throughput on every launch against
     * redundant-but-correct output on a geometry needing 33 million blocks -
     * against a decode_grid_blocks cap of 8192, and unreachable through the
     * public ABI - is not a trade worth making. */
    if (total_warps == 0 || blockDim.x % kWarpSize != 0) {
        return;
    }

    for (size_t chunk = warp_in_grid; chunk < chunk_count;
         chunk += total_warps) {
        const unsigned char* src =
            static_cast<const unsigned char*>(src_ptrs[chunk]);
        /* dst must NOT be __restrict__-qualified: the cross-lane read after
         * write below relies on __syncwarp() forcing a reload from global
         * memory, and __restrict__ could legally let the compiler cache a
         * dst[] value across the barrier. */
        unsigned char* dst = static_cast<unsigned char*>(dst_ptrs[chunk]);

        const size_t src_size = src_sizes[chunk];
        Lz4Parser parser{src, src_size, dst_caps[chunk]};
        Lz4Sequence seq;
        cudec_status status = CUDEC_OK;
        bool done = false;
        /* Fuel: Next consumes at least the token byte, so a chunk of n
         * source bytes admits at most n + 1 calls - a budget no input the
         * parser accepts or rejects can reach. The saturating form matters:
         * a plain n + 1 would wrap to 0 for a SIZE_MAX chunk size and cap a
         * parse that nothing else would have stopped. It is lane-uniform
         * (every lane parses the same bytes), so it cannot make one lane
         * leave the loop early and strand the others at a __syncwarp() below.
         * A hung warp is a fail-closed violation like an out-of-bounds read,
         * and this is what makes a future parser bug a rejected chunk. */
        uint64_t fuel = src_size == SIZE_MAX ? SIZE_MAX : src_size + 1;
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
            /* The fuel test rides the exit branch that already exists rather
             * than adding one at the top of the loop. That is a structural
             * choice, not a measured one - the formulations were not
             * separable above run-to-run noise; what IS measured is the cost
             * of having a cap at all (docs/BENCHMARKS.md). */
            if (done || fuel-- == 0) {
                break;
            }
        }
        /* Fuel exhaustion is a reject, never a success: leaving the loop
         * without `done` must not report the parse as complete. */
        if (status == CUDEC_OK && !done) {
            status = CUDEC_ERR_CORRUPT_INPUT;
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
    if (d_src_ptrs == nullptr || d_src_sizes == nullptr ||
        d_dst_ptrs == nullptr || d_dst_capacities == nullptr ||
        d_results == nullptr ||
        reinterpret_cast<uintptr_t>(d_results) % 16 != 0 ||
        chunk_count == 0 || chunk_count > kMaxBatchChunks) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_LZ4_DECODE_CUH */
