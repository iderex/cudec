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

    /* The GDeflate entry, whose second half is the OPPOSITE assertion and
     * is why it belongs in this file at all (issue #216). With no visible
     * device, any CUDA call this entry made would fail, so an entry that
     * launched, or merely drained the pending error state, would answer
     * CUDEC_ERR_CUDA here. Answering the defined not-implemented status
     * instead is the evidence that a batch it accepts reaches the CUDA
     * runtime not at all. Twice, for the reason the calls above are twice:
     * a first-touch context init failing once would look the same. */
    REQUIRE(cudec_gdeflate_decompress_batch(nullptr, sizes, dsts, caps, 1,
                                            results, nullptr) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, 1,
                                            results, nullptr) ==
            CUDEC_ERR_NOT_IMPLEMENTED);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, 1,
                                            results, nullptr) ==
            CUDEC_ERR_NOT_IMPLEMENTED);

    std::printf("PASS: with zero visible devices the two kernel-backed batch "
                "entries report CUDEC_ERR_CUDA and the GDeflate entry "
                "reports CUDEC_ERR_NOT_IMPLEMENTED\n");
    return 0;
}
