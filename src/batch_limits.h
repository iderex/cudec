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

/* Lanes per warp, the kernel's chunk-to-lane divisor and the multiplier below.
 * The one place the width's digits are written (issue #211); the width port
 * turns this into the WaveSize parameter (#241). */
constexpr unsigned kWarpSize = 32;

/* One chunk per warp and at most INT32_MAX warps in a grid, so this is the
 * largest chunk_count either entry point can map onto a launch. */
constexpr size_t kMaxBatchChunks = static_cast<size_t>(INT32_MAX) * kWarpSize;

/* The over-limit contract test hands SIZE_MAX to both entry points and expects
 * a reject; that is only a test of the bound while the bound is below
 * SIZE_MAX. */
static_assert(kMaxBatchChunks < SIZE_MAX,
              "the SIZE_MAX over-limit contract test relies on this");

}  // namespace cudec_detail

#endif /* CUDEC_BATCH_LIMITS_H */
