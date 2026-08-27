/* The batch geometry both decode entry points are bounded by, defined once.
 *
 * The limit used to be written twice - once in validate_batch_args and once in
 * the streaming path's pre-check - and kept in agreement by a comment saying
 * so (issue #39). Nothing failed if they drifted, and a drift means either a
 * chunk_count the streaming pre-check accepts and the batch entry then rejects
 * per wave, or a needless reject. Both are wrong answers on a fail-closed
 * boundary, so the bound has one definition and both callers read it.
 *
 * Nothing CUDA-specific here on purpose: this header is included by a .cu
 * translation unit and by a host .cpp compiled as plain C++, so it must not
 * pull in the CUDA runtime. Internal header, not part of the ABI. */
#ifndef CUDEC_BATCH_LIMITS_H
#define CUDEC_BATCH_LIMITS_H

#include <cstddef>
#include <cstdint>

namespace cudec_detail {

/* Lanes per warp on this backend, and the kernel's chunk-to-lane divisor.
 * The one place the width's digits are written (issue #211). The width port
 * landed and did NOT turn this into the parameter: the launch's width is the
 * WaveSize template argument in src/chunk_decode.cuh and kCudaWaveSize seeds
 * it from here, so this stays the CUDA build's width rather than becoming the
 * family's. */
constexpr unsigned kWarpSize = 32;

/* The wave width the ABI's chunk-count ceiling below is FROZEN at, on every
 * backend. It is a second name for the same digits as kWarpSize on purpose,
 * and the two must be free to differ: this one is an ABI constant and that one
 * is a device property (masterplan section 15.5, issue #242). */
constexpr unsigned kAbiChunkLimitWaveSize = 32;

/* One chunk per warp and at most INT32_MAX warps in a grid, so this is the
 * largest chunk_count either entry point can map onto a launch.
 *
 * IT DOES NOT MOVE WITH THE WAVE WIDTH, and the tempting spelling is the one
 * that does: INT32_MAX times the width the kernel is launched at. A wave64
 * device would then admit twice as many chunks, so the same call succeeds on
 * one GPU and returns CUDEC_ERR_INVALID_ARGUMENT on another - a contract that
 * varies by hardware, in a library whose claim is that the same input produces
 * the same result on every path. The narrower bound costs nothing anybody will
 * meet: INT32_MAX * 32 chunks is far past any real batch.
 *
 * The launch-shape refusal is the other half of the same question and it goes
 * the other way. A block that is not a whole number of waves leaves a slice of
 * a destination written by nobody (docs/DETERMINISM.md), so THAT guard is
 * derived from the wave width, in src/chunk_decode.cuh. There it is a
 * statement about the launch; here it is a statement about the ABI. */
constexpr size_t kMaxBatchChunks =
    static_cast<size_t>(INT32_MAX) * kAbiChunkLimitWaveSize;

/* The over-limit contract test hands SIZE_MAX to both entry points and expects
 * a reject; that is only a test of the bound while the bound is below
 * SIZE_MAX. */
static_assert(kMaxBatchChunks < SIZE_MAX,
              "the SIZE_MAX over-limit contract test relies on this");

}  // namespace cudec_detail

#endif /* CUDEC_BATCH_LIMITS_H */
