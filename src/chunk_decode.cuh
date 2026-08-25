/* The warp-per-chunk chunk decoder: one kernel, templated on the format
 * parser, on the copy mode and on the wave width. The parser is a parameter
 * rather
 * than a hard-wired type because everything below it - the fuel cap, the
 * two copy loops, the __syncwarp() placement, the geometry guards, the
 * failure contract - is format-independent, and a second copy of it per
 * format would be a second place for the fail-closed discipline to drift
 * (masterplan section 10, "The seam"). src/batch.cu instantiates it once
 * per format; the bench instantiates ParseOnly for its honest parse-only
 * ceiling (masterplan section 9). Each translation unit emits only the
 * instantiations it launches, so the shipped library carries no parse-only
 * kernel and, for the same reason, no wave64 kernel: the CUDA launches all
 * name kCudaWaveSize. Internal header, not part of the ABI.
 *
 * THE WIDTH IS A PARAMETER BECAUSE THE HARDWARE HERE CANNOT FALSIFY A WRONG
 * ONE (issue #241). Every lane-stride and geometry constant below is derived
 * from WaveSize, so a hardcoded lane count cannot survive in a place only a
 * wave64 device would expose. The parse uses no shuffle and no ballot, so no
 * 64-bit lane mask arises and the port is the strides and the geometry
 * guards rather than a second decode.
 *
 * What a Parser owes, and what this kernel is entitled to assume. The
 * requirements are stated here because this is the code that breaks when
 * one of them is not met:
 *
 *  - construction from the chunk's source pointer, source size and
 *    destination capacity, and nothing else;
 *  - one entry point, Next(DecodeSequence*, bool*), returning exactly three
 *    outcomes: an element to execute and call again, the single success
 *    exit with *done set, or a reject status;
 *  - LIVENESS: every call that hands back a non-terminal element consumes
 *    at least one source byte. The fuel bound below is derived from the
 *    source size, so a parser that can return without advancing breaks
 *    termination for the whole family, not only for its own format;
 *  - LANE-UNIFORMITY: the parser reads source bytes and its own state and
 *    nothing else. Every lane of the wave runs it redundantly in lockstep,
 *    and that is sound only while every lane computes the same answer;
 *  - ALL input validation inside the parser. The copy loops below perform
 *    no input-derived bound check of their own, by design: a parser that
 *    hands back an element it has not fully bounded has already failed
 *    closed-ness and nothing here will catch it;
 *  - a dst_pos member holding the bytes produced, read once on success;
 *  - a reported match offset (match_dst - match_src) that is at least 1 and
 *    below 2^32. The lower bound keeps the closed-form gather from a
 *    modulo by zero; the upper one is what makes the 32-bit gather arm
 *    lossless. Both hold structurally from the offset field's width in
 *    every format on this seam, and neither is re-checked here.
 *
 * The parse is redundant across lanes rather than broadcast, and the
 * element type is shared across formats rather than associated with each
 * parser - both settled at the #85 design panel and not re-opened here. */
#ifndef CUDEC_CHUNK_DECODE_CUH
#define CUDEC_CHUNK_DECODE_CUH

#include "batch_limits.h"
#include "cudec.h"
#include "decode_sequence.h"

#include <cuda_runtime.h>

namespace cudec_detail {

constexpr unsigned kBlockWarps = 4;

/* The block shape is derived from the wave width rather than written down:
 * four waves to a block, whatever a wave is on the device being built for
 * (issue #241). At wave32 this is the 128 threads the shipped launches have
 * always used; at wave64 it is 256, which is the same four waves and not a
 * different block shape. */
template <int WaveSize>
inline constexpr unsigned kBlockThreadsFor =
    static_cast<unsigned>(WaveSize) * kBlockWarps;

/* The one width this CUDA build instantiates. It is named here rather than
 * written as digits at each launch, for the reason issue #211 gives, and it
 * is a compile-time constant on purpose: the runtime-width dispatch is a
 * sibling issue, and until it lands the CUDA translation units emit exactly
 * one kernel per format. */
constexpr int kCudaWaveSize = static_cast<int>(kWarpSize);

constexpr unsigned kBlockThreads = kBlockThreadsFor<kCudaWaveSize>;

/* One warp per chunk over a grid-stride loop. All 32 lanes run the
 * validated parser redundantly in lockstep and fan out by lane for every
 * copy; overlapping matches use the closed-form modular gather
 * dst[m+i] = dst[m-off + (i mod off)] (off >= 1 by the parser's offset==0
 * rejection, so no modulo-by-zero). __syncwarp() carries intra-warp memory
 * ordering at the two copy boundaries. ParseOnly elides the copies and the
 * syncs to isolate the parse cost - it writes no output and is a bench
 * ceiling only, never shipped. */
template <class Parser, bool ParseOnly, int WaveSize>
__global__ void __launch_bounds__(kBlockThreadsFor<WaveSize>)
    chunk_decode_batch(const void* const* src_ptrs, const size_t* src_sizes,
                       void* const* dst_ptrs, const size_t* dst_caps,
                       size_t chunk_count, cudec_chunk_result* results) {
    /* Every lane-stride and geometry constant below is this one value. The
     * cast is here rather than at each use so the parameter's own type stays
     * the `int` the width family is written in, and the arithmetic below
     * stays unsigned. */
    constexpr unsigned kWaveSize = static_cast<unsigned>(WaveSize);

    const unsigned lane = threadIdx.x % kWaveSize;
    const size_t warp_in_grid =
        (blockIdx.x * blockDim.x + threadIdx.x) / kWaveSize;
    const size_t total_warps =
        (static_cast<size_t>(gridDim.x) * blockDim.x) / kWaveSize;
    /* The chunk-to-lane mapping is sound only when the grid holds at least
     * one whole warp and every block is a whole number of warps. Both
     * failures are geometry rather than input, and both are silent:
     *  - fewer than one whole warp in the grid makes the grid-stride below
     *    advance by zero and spin forever;
     *  - a block that is not a wave multiple splits a chunk's wave across two
     *    blocks (a block of half a wave, or of one and a half, both do), so
     *    `lane` only ever takes the low part of its range while the copy
     *    loops stride by kWaveSize - every output byte whose index modulo the
     *    wave width lands in the missing part is written by nobody, and the
     *    full-mask __syncwarp() is executed by a fraction of a wave. Both
     *    halves of that argument scale with the width rather than with the
     *    digits they used to be written in.
     * The shipped entry always launches kBlockThreads, but the kernel must
     * not depend on its caller for either property. Both tests are
     * grid-uniform (gridDim/blockDim only), so neither can strand lanes at a
     * __syncwarp().
     *
     * warp_in_grid above stays 32-bit, so this kernel additionally requires
     * gridDim.x * blockDim.x <= 2^32. That bound is DOCUMENTED rather than
     * enforced, and the reason is the asymmetry in what breaking it costs:
     * past it the index wraps modulo 2^32, but the guard below has already
     * forced blockDim.x to a multiple of kWaveSize, so the wrapped base stays
     * wave-aligned and the aliased block recomputes the SAME warp_in_grid
     * values with a full lane range. That argument rests on the wrap modulus
     * being a multiple of the wave width, which holds for every power-of-two
     * width and so survives the port unchanged. Two blocks then decode one chunk
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
    if (total_warps == 0 || blockDim.x % kWaveSize != 0) {
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
        Parser parser{src, src_size, dst_caps[chunk]};
        DecodeSequence seq;
        cudec_status status = CUDEC_OK;
        bool done = false;
        /* Fuel: the seam's liveness requirement says every non-terminal
         * call consumes at least one source byte, so a chunk of n
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
                for (uint64_t i = lane; i < seq.literals_len;
                     i += kWaveSize) {
                    dst[seq.literals_dst + i] = src[seq.literals_src + i];
                }
                /* The match may read literal bytes just written above. */
                __syncwarp();
                if (seq.match_len != 0) {
                    const uint64_t offset = seq.match_dst - seq.match_src;
                    /* Same closed-form gather at two arithmetic widths.
                     * sm_86 has no integer-divide hardware, so a modulo by
                     * a runtime divisor is a software sequence, and the
                     * 64-bit one costs roughly twice the 32-bit one - paid
                     * once per match byte per lane. `offset` comes out of a
                     * format's offset FIELD, whose width bounds it below
                     * 2^32 structurally rather than by a check on parsed
                     * input - two bytes for LZ4, at most four for Snappy -
                     * which is the seam requirement stated at the top of
                     * this header, and it is why off32 loses nothing. `i`
                     * is bounded only by match_len, which the ABI's size_t
                     * capacities admit above 2^32, so the narrowing is
                     * sound only under the test. No field is narrowed and
                     * no bound is taken from the 64 KiB convention: the
                     * stored sequence stays 64-bit and a match longer than
                     * 2^32 still decodes, through the other arm. The test
                     * is warp-uniform (every lane parsed the same bytes),
                     * so it costs no divergence and cannot strand a lane at
                     * the __syncwarp() below. */
                    if (seq.match_len <= UINT32_MAX) {
                        const uint32_t off32 = static_cast<uint32_t>(offset);
                        for (uint64_t i = lane; i < seq.match_len;
                             i += kWaveSize) {
                            dst[seq.match_dst + i] =
                                dst[seq.match_src +
                                    (static_cast<uint32_t>(i) % off32)];
                        }
                    } else {
                        for (uint64_t i = lane; i < seq.match_len;
                             i += kWaveSize) {
                            dst[seq.match_dst + i] =
                                dst[seq.match_src + (i % offset)];
                        }
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

#endif /* CUDEC_CHUNK_DECODE_CUH */
