/* The two decode entry points are bounded by the same chunk_count limit, and
 * this asserts they answer the same way at that boundary (issue #39). The
 * limit comes from src/batch_limits.h, so the test cannot carry a third copy
 * of the number it is checking.
 *
 * Host-side and GPU-less: every call below is rejected by argument validation
 * before any CUDA call, which is what makes it safe to hand both entry points
 * pointer arrays with one element and a chunk_count of tens of billions. No
 * call here passes validation, deliberately - a validation-passing call would
 * launch on fake pointers.
 *
 * What this test does NOT catch, measured rather than reasoned: a streaming
 * bound raised above the batch one. Set to twice the canonical value, every
 * assertion below still passes - the over-limit count then clears the
 * streaming count check and the per-chunk walk reads past these one-element
 * arrays, which is undefined and happened to reject. No caller can hand that
 * validator an array of tens of billions of entries, so no behavioural test
 * can separate "rejected for the count" from "rejected for something else" at
 * this magnitude. The lock against a divergent bound is therefore the
 * configure-time one in tests/CMakeLists.txt, which reads the comparison out
 * of both sources; what this test carries is that the two entry points still
 * answer identically at the boundary and on both of its ends. */
#include "batch_limits.h"
#include "cudec.h"
#include "require.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
    alignas(16) static cudec_chunk_result results[1];
    static const void* srcs[1] = {nullptr};
    static size_t sizes[1] = {0};
    static void* dsts[1] = {nullptr};
    static size_t caps[1] = {0};

    const size_t over = cudec_detail::kMaxBatchChunks + 1;
    /* The bound is the thing under test, so its shape is asserted rather than
     * assumed: an over-limit count that had silently become 0 through some
     * arithmetic change would make every check below pass vacuously, since 0
     * is a reject class of its own. */
    REQUIRE(over > cudec_detail::kMaxBatchChunks);
    REQUIRE(cudec_detail::kMaxBatchChunks > 0);

    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, over, results,
                                       nullptr) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_stream(srcs, sizes, dsts, caps, over,
                                        CUDEC_MEM_DEVICE, results) ==
            CUDEC_ERR_INVALID_ARGUMENT);

    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, SIZE_MAX,
                                       results, nullptr) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_stream(srcs, sizes, dsts, caps, SIZE_MAX,
                                        CUDEC_MEM_DEVICE, results) ==
            CUDEC_ERR_INVALID_ARGUMENT);

    /* Zero is the other end of the same field, and both entry points name it
     * as a reject class of its own. Cheap to include and it keeps the pair
     * being compared over the whole boundary rather than one side of it. */
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, 0, results,
                                       nullptr) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_stream(srcs, sizes, dsts, caps, 0,
                                        CUDEC_MEM_DEVICE, results) ==
            CUDEC_ERR_INVALID_ARGUMENT);

    std::printf("PASS: both entry points refuse chunk_count 0, %zu and "
                "SIZE_MAX against the shared limit %zu\n",
                over, cudec_detail::kMaxBatchChunks);
    return 0;
}
