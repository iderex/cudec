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

/* WHICH KERNEL THE DISPATCH LAUNCHES FOR A WIDTH THE DEVICE REPORTED, and the
 * one place that decision is written (issue #242). Host logic with nothing
 * device-specific in it, which is why it lives in this header rather than in
 * src/chunk_decode.cuh: the wave64 answer can then be exercised on a machine
 * with no wave64 device, and on a machine with no device at all.
 *
 * THE DEFAULT ARM REFUSES AND MUST NOT FALL BACK TO 32. A width this build
 * emits no kernel for is not a hardware detail to paper over: the kernel maps
 * chunks onto lanes by the width it was instantiated at, so launching the
 * 32-lane kernel on a device whose wave is some other size gives a block that
 * is not a whole number of waves - and that is the case src/chunk_decode.cuh
 * documents as leaving every output byte whose index modulo the width lands in
 * the missing part written by nobody. A silent wrong answer, on the fail-closed
 * side of a decoder. The caller gets CUDEC_ERR_UNSUPPORTED instead.
 *
 * The enumerators carry the widths as their values so a reader cannot pair the
 * wrong one with a launch, and the type still refuses to convert to a width
 * implicitly. */
constexpr int kWaveWidth32 = 32;
constexpr int kWaveWidth64 = 64;

enum class WaveInstantiation {
    kUnsupported = 0,
    kWave32 = kWaveWidth32,
    kWave64 = kWaveWidth64
};

inline WaveInstantiation select_wave_instantiation(int reported_width) {
    switch (reported_width) {
        case kWaveWidth32:
            return WaveInstantiation::kWave32;
        case kWaveWidth64:
            return WaveInstantiation::kWave64;
        default:
            return WaveInstantiation::kUnsupported;
    }
}

}  // namespace cudec_detail

#endif /* CUDEC_BATCH_LIMITS_H */
