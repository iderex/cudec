/* Conformance: the public header is consumable from plain C99 - it
 * compiles as C, resolves every public symbol with C linkage, and the
 * documented contract holds for a C caller. C99 deliberately: it routes
 * the header's layout check through the pre-C11 negative-array-size
 * fallback, which no other translation unit compiles.
 *
 * Every path exercised here is a documented synchronous reject that makes
 * no CUDA call (see cudec.h), so this test runs green on the GPU-less CI
 * runner - the pointers below are host memory and are never dereferenced. */
#include "cudec.h"

#include <stdint.h>
#include <stdio.h>

#define REQUIRE(cond)                                                     \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,       \
                    #cond);                                               \
            return 1;                                                     \
        }                                                                 \
    } while (0)

int main(void) {
    unsigned char raw[2 * sizeof(cudec_chunk_result) + 16];
    const uintptr_t base = ((uintptr_t)raw + 15) & ~(uintptr_t)15;
    cudec_chunk_result* aligned = (cudec_chunk_result*)base;
    cudec_chunk_result* misaligned = (cudec_chunk_result*)(base + 8);
    const void* srcs[1] = {0};
    size_t sizes[1] = {0};
    void* dsts[1] = {0};
    size_t caps[1] = {0};

    REQUIRE(cudec_version() == CUDEC_VERSION_MAJOR * 10000 +
                                   CUDEC_VERSION_MINOR * 100 +
                                   CUDEC_VERSION_PATCH);

    /* Every documented reject class, one call each: null arrays, null
     * results, misaligned results, empty batch, over-limit batch. */
    REQUIRE(cudec_lz4_decompress_batch(0, sizes, dsts, caps, 1, aligned, 0) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_batch(srcs, 0, dsts, caps, 1, aligned, 0) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, 0, caps, 1, aligned, 0) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, 0, 1, aligned, 0) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, 1, 0, 0) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, 1, misaligned,
                                       0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, 0, aligned,
                                       0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, SIZE_MAX,
                                       aligned, 0) ==
            CUDEC_ERR_INVALID_ARGUMENT);

    /* The frame entry point resolves its own C linkage here. Every call
     * below is a documented argument reject that returns before any CUDA
     * call, so it runs on the GPU-less runner and never dereferences the
     * host pointers. */
    {
        size_t written = 123;
        unsigned char byte = 0;
        REQUIRE(cudec_lz4f_decompress(0, 0, &byte, 1, &written) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null frame */
        REQUIRE(written == 0);
        REQUIRE(cudec_lz4f_decompress(&byte, 1, &byte, 1, 0) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null bytes_written */
        REQUIRE(cudec_lz4f_decompress(&byte, 1, 0, 1, &written) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null dst, nonzero capacity */
        REQUIRE(written == 0);
    }

    /* The streaming entry and its enum resolve their C linkage here. Every
     * call is an argument reject that returns before any CUDA call. */
    {
        const void* one_src[1] = {0};
        size_t one_size[1] = {0};
        void* one_dst[1] = {0};
        size_t one_cap[1] = {0};
        cudec_chunk_result sres[1];
        REQUIRE(cudec_lz4_decompress_stream(0, one_size, one_dst, one_cap, 1,
                                            CUDEC_MEM_DEVICE, 4, sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null array */
        REQUIRE(cudec_lz4_decompress_stream(one_src, one_size, one_dst, one_cap,
                                            0, CUDEC_MEM_DEVICE, 4, sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* zero chunks */
        REQUIRE(cudec_lz4_decompress_stream(one_src, one_size, one_dst, one_cap,
                                            1, (cudec_mem_space)7, 4, sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* unknown dst_space */
    }

    printf("PASS: plain-C caller exercised every public symbol\n");
    return 0;
}
