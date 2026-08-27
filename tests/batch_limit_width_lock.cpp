/* The ABI's chunk-count ceiling does not move with the wave width (issue
 * #242, masterplan section 15.5). Host-side and GPU-less: nothing here
 * launches and nothing here calls into CUDA.
 *
 * WHY THIS TEST WRITES THE NUMBER OUT AGAIN WHEN batch_limit_parity.cpp
 * DELIBERATELY DOES NOT. That test compares the two entry points against each
 * other at the boundary and says in its own comment that it must not carry a
 * third copy of the value, because a copy drifts silently against the header.
 * The failure guarded here is the opposite one, and a copy is the only thing
 * that catches it: the header's own constant being re-derived from the width
 * the kernel launches at. Both sides of such a change move together, so every
 * test that reads the value out of the header agrees with it afterwards and
 * stays green. The digits below are the reference, not a duplicate of it.
 *
 * WHAT THE FAILURE LOOKS LIKE IF IT SHIPS. INT32_MAX * 64 is twice INT32_MAX *
 * 32, so a width-derived ceiling accepts a chunk_count on a wave64 device that
 * the same call is refused with CUDEC_ERR_INVALID_ARGUMENT on a wave32 one -
 * a contract that varies by hardware, in a library whose claim is that the
 * same input produces the same result on every path. The comment on kWarpSize
 * used to name that derivation as the port's next step, which is why the
 * refusal is executed here rather than left in prose.
 *
 * WHY THE PROOF IS COMPILE-TIME AND THERE IS NO BEHAVIOURAL LEG. Showing
 * behaviourally that the accepted set has not GROWN means calling an entry
 * point with a count inside the grown region - and when the defect is present
 * that call passes validation and launches on the fake pointer arrays a
 * host-side test has to use. batch_limit_parity.cpp refuses that for the same
 * reason and stays on the reject side of the boundary throughout. So the
 * comparison below is made where it costs nothing to make it: against digits
 * this file owns, at compile time, before any call exists. */
#include "batch_limits.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

/* The wave32 value, spelled from digits, so the assertions compare the header
 * against something that does not move when the header does. */
constexpr size_t kFrozenAbiChunkCeiling =
    static_cast<size_t>(INT32_MAX) * 32u;

/* What a ceiling derived from the launch width would be on a wave64 device.
 * Named rather than written inline: it is the accepted set this test exists to
 * keep out, not an arbitrary larger number. */
constexpr size_t kWave64DerivedCeiling = static_cast<size_t>(INT32_MAX) * 64u;

static_assert(kWave64DerivedCeiling > kFrozenAbiChunkCeiling,
              "a width-derived ceiling is the LARGER accepted set; if these "
              "two are equal the assertions below compare nothing");

static_assert(cudec_detail::kMaxBatchChunks == kFrozenAbiChunkCeiling,
              "the batch ABI's chunk-count ceiling is frozen at the wave32 "
              "value on every backend (masterplan section 15.5)");

static_assert(cudec_detail::kMaxBatchChunks != kWave64DerivedCeiling,
              "the batch ABI's chunk-count ceiling has been re-derived from "
              "the launch width; section 15.5 refuses that");

static_assert(cudec_detail::kAbiChunkLimitWaveSize == 32u,
              "the width the ABI ceiling is frozen at is an ABI constant, not "
              "the device's width");

/* The two constants coincide on this backend and must be free to differ: a
 * static_assert tying them together would re-create the derivation above by
 * another route. Asserted the only way that leaves them independent - each
 * against its own reference. */
static_assert(cudec_detail::kWarpSize == 32u,
              "the CUDA build's wave width is 32");

}  // namespace

int main() {
    std::printf("PASS: the batch ABI ceiling is %zu on every backend; the "
                "wave64-derived %zu is not it\n",
                cudec_detail::kMaxBatchChunks, kWave64DerivedCeiling);
    return 0;
}
