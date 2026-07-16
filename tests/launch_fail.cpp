/* The launch-failure branch of cudec_lz4_decompress_batch, exercised with
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

    std::printf("PASS: launch-failure branch reports CUDEC_ERR_CUDA with "
                "zero visible devices\n");
    return 0;
}
