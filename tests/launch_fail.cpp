/* The launch-failure branch of both batch entry points, exercised with
 * defined behavior (deferred gap 1 from the #2 review): a process with
 * zero visible CUDA devices makes every launch submission fail
 * deterministically, so CI's GPU-lessness becomes the test condition and
 * the same test runs on the GPU machine via the environment override. No
 * destroyed-stream UB anywhere: the fake device pointers below pass
 * validation and are never dereferenced because the launch dies at
 * submission. */
#include "cudec.h"
#include "require.h"

#include <cstdio>
#include <cstdlib>

int main() {
    /* First statement, before any CUDA runtime touch; belt-and-suspenders
     * with the ENVIRONMENT property on the ctest entry. */
    REQUIRE(setenv("CUDA_VISIBLE_DEVICES", "", 1) == 0);

    alignas(16) static cudec_chunk_result results[1];
    static const void* srcs[1];
    static size_t sizes[1];
    static void* dsts[1];
    static size_t caps[1];

    /* Validation is pure host logic: it rejects without a CUDA call even
     * when no device exists. */
    REQUIRE(cudec_lz4_decompress_batch(nullptr, sizes, dsts, caps, 1, results,
                                       nullptr) == CUDEC_ERR_INVALID_ARGUMENT);

    /* A validation-passing call must reach the launch and report its
     * submission failure - the branch under test. Twice: the branch is
     * stable, not a one-shot artifact of first-touch context init. */
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, 1, results,
                                       nullptr) == CUDEC_ERR_CUDA);
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, 1, results,
                                       nullptr) == CUDEC_ERR_CUDA);

    /* The Snappy entry's two halves, in the same order and for the same
     * reason (issue #152). The second one is what a shared validator cannot
     * prove on its own: validation passing is the point where the two
     * entries stop being the same code, and an entry that dropped the
     * post-launch error check would return CUDEC_OK here while submitting
     * nothing. */
    REQUIRE(cudec_snappy_decompress_batch(nullptr, sizes, dsts, caps, 1,
                                          results, nullptr) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, dsts, caps, 1, results,
                                          nullptr) == CUDEC_ERR_CUDA);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, dsts, caps, 1, results,
                                          nullptr) == CUDEC_ERR_CUDA);

    /* The GDeflate entry's two halves, the same shape as the two above now
     * that its kernel stands behind the frozen contract (issues #216, #214).
     * Until then the second half was the OPPOSITE assertion - the defined
     * not-implemented status, as the evidence that an accepted batch reached
     * the runtime not at all - and the day the kernel landed it flipped to
     * the launch-failure answer, which is the cleanest record of the landing
     * this file can hold. Twice, for the reason the calls above are twice. */
    REQUIRE(cudec_gdeflate_decompress_batch(nullptr, sizes, dsts, caps, 1,
                                            results, nullptr) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, 1,
                                            results, nullptr) == CUDEC_ERR_CUDA);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, 1,
                                            results, nullptr) == CUDEC_ERR_CUDA);

    /* The Zstd entry, the same opposite assertion and for the same reason
     * (issue #427). Written out rather than folded into a loop over the two
     * not-implemented entries: what this file is for is the point where two
     * entries stop being the same code, and a loop would assume they are. */
    REQUIRE(cudec_zstd_decompress_batch(nullptr, sizes, dsts, caps, 1, results,
                                        nullptr) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, caps, 1, results,
                                        nullptr) == CUDEC_ERR_NOT_IMPLEMENTED);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, caps, 1, results,
                                        nullptr) == CUDEC_ERR_NOT_IMPLEMENTED);

    std::printf("PASS: with zero visible devices the three kernel-backed batch "
                "entries report CUDEC_ERR_CUDA and the Zstd entry reports "
                "CUDEC_ERR_NOT_IMPLEMENTED\n");
    return 0;
}
