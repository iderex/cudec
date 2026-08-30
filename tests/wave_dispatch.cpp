/* The width-to-instantiation selection the batch entries dispatch on (issue
 * #242). Host-side and GPU-less: nothing here launches and nothing here calls
 * into a device runtime.
 *
 * WHY A MOCKED WIDTH IS THE ONLY WAY THIS LEG EXISTS AT ALL. The answer that
 * matters is the wave64 one, and no machine in this project has ever had a
 * wave64 device - no hosted runner carries an AMD GPU either. Handing the
 * selector the number the runtime would have reported is the whole of what
 * can be executed here, and it is worth executing: the failure it catches is
 * a dispatch that reads 64 and launches the 32-lane kernel anyway, which is a
 * silent wrong answer rather than a crash.
 *
 * WHAT IT DOES NOT PROVE, stated here because the file's name invites the
 * larger reading. It does not prove that a wave64 DEVICE launches the wave64
 * kernel, that the kernel decodes correctly at that width, or that the
 * runtime reports 64 where this project expects it to. Those need the
 * hardware, and the pull request that landed this says so in the same words.
 * What is executed here is the choice, and only the choice.
 *
 * THE UNSUPPORTED ARM IS THE ONE WITH TEETH. A selector that fell back to the
 * 32-lane kernel for an unknown width would pass every assertion about 32 and
 * 64 above it, so the widths below are not decoration: they are the whole
 * difference between a fail-closed refusal and a launch at a geometry the
 * kernel's own comment describes as leaving output bytes written by nobody. */
#include "batch_limits.h"
#include "require.h"

#include <cstdio>

using cudec_detail::select_wave_instantiation;
using cudec_detail::WaveInstantiation;

int main() {
    /* The two widths this build emits a kernel for. 32 is every CUDA device
     * and RDNA; 64 is CDNA. */
    REQUIRE(select_wave_instantiation(32) == WaveInstantiation::kWave32);
    REQUIRE(select_wave_instantiation(64) == WaveInstantiation::kWave64);

    /* Everything else refuses. The list is chosen rather than swept: each one
     * is a value a runtime could plausibly hand back, and the two nearest
     * neighbours of each supported width are the mistakes an off-by-one in a
     * range test would let through.
     *
     * 0 is what a query that wrote nothing leaves in an initialised variable,
     * so it is the value a caller sees if the dispatch ever ignores a failed
     * status - and it must not select a kernel. */
    const int kRefused[] = {0,  1,  16, 31, 33, 48,
                            63, 65, 96, 128, 256, -1};
    for (int width : kRefused) {
        REQUIRE(select_wave_instantiation(width) ==
                WaveInstantiation::kUnsupported);
    }

    /* The enumerators carry their widths, which is what lets a reader check a
     * launch against the arm that selected it. A renumbering that broke this
     * would leave every assertion above green. */
    REQUIRE(static_cast<int>(WaveInstantiation::kWave32) == 32);
    REQUIRE(static_cast<int>(WaveInstantiation::kWave64) == 64);
    REQUIRE(static_cast<int>(WaveInstantiation::kUnsupported) == 0);

    std::printf("PASS: the dispatch selects 32 and 64 by the reported width "
                "and refuses %zu other widths\n",
                sizeof(kRefused) / sizeof(kRefused[0]));
    return 0;
}
