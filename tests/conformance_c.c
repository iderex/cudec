/* Conformance: the public header is consumable from plain C99 - it
 * compiles as C, resolves every public symbol with C linkage, and the
 * documented contract holds for a C caller. C99 deliberately: it routes
 * the header's layout check through the pre-C11 negative-array-size
 * fallback, which no other translation unit compiles.
 *
 * Every path exercised here returns synchronously without making a CUDA
 * call (see cudec.h): the documented argument rejects, and the one
 * documented ACCEPT that answers CUDEC_ERR_NOT_IMPLEMENTED because this
 * build carries no GDeflate kernel. So this test runs green on the
 * GPU-less CI runner - the pointers below are host memory and are never
 * dereferenced. */
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

    /* The line above compares two things derived from the same three
     * macros, so it holds for any version and pins none. This one pins the
     * current one. It exists to red on a partial bump: a release that edits
     * the header but leaves a number written out somewhere else stops here
     * with the arithmetic in front of the reader, instead of shipping two
     * versions. Bumping the version means editing this literal too, and
     * that is the point of it rather than a cost of it.
     *
     * 0.1.0 encodes as 0 * 10000 + 1 * 100 + 0 = 100. The three-digit look
     * of it is the encoding working, not a typo: 10000 is 1.0.0. */
    REQUIRE(cudec_version() == 100); /* 0.1.0 */

    /* The build system derives PROJECT_VERSION from the macros above and
     * hands the three numbers back as CUDEC_CMAKE_VERSION_*. A derivation
     * that matched the wrong line in the header would otherwise configure
     * clean and ship package metadata disagreeing with cudec_version(). An
     * undefined macro here is a compile error, not a zero. */
    REQUIRE(CUDEC_CMAKE_VERSION_MAJOR == CUDEC_VERSION_MAJOR);
    REQUIRE(CUDEC_CMAKE_VERSION_MINOR == CUDEC_VERSION_MINOR);
    REQUIRE(CUDEC_CMAKE_VERSION_PATCH == CUDEC_VERSION_PATCH);

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

    /* The same eight classes on the Snappy entry (issue #152). The contract
     * is frozen and format-independent, and the two entries share one
     * validator - which is exactly why this is written out call for call
     * rather than trusted: a future entry that grew its own validation, or
     * one wired to a different one, would pass a test that only counted on
     * the sharing. Resolving the symbol from C99 at all is half of what
     * this file is for. */
    REQUIRE(cudec_snappy_decompress_batch(0, sizes, dsts, caps, 1, aligned,
                                          0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, 0, dsts, caps, 1, aligned,
                                          0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, 0, caps, 1, aligned,
                                          0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, dsts, 0, 1, aligned,
                                          0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, dsts, caps, 1, 0, 0) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, dsts, caps, 1,
                                          misaligned,
                                          0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, dsts, caps, 0, aligned,
                                          0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_snappy_decompress_batch(srcs, sizes, dsts, caps, SIZE_MAX,
                                          aligned,
                                          0) == CUDEC_ERR_INVALID_ARGUMENT);

    /* The same eight classes on the GDeflate entry (issue #216), written
     * out call for call for the reason the Snappy block above is: a future
     * entry wired to its own validator, or to none, would pass a test that
     * only counted on the sharing. */
    REQUIRE(cudec_gdeflate_decompress_batch(0, sizes, dsts, caps, 1, aligned,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, 0, dsts, caps, 1, aligned,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, 0, caps, 1, aligned,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, 0, 1, aligned,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, 1, 0,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, 1,
                                            misaligned,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, 0,
                                            aligned,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_gdeflate_decompress_batch(srcs, sizes, dsts, caps, SIZE_MAX,
                                            aligned,
                                            0) == CUDEC_ERR_INVALID_ARGUMENT);

    /* The ninth class this entry carried while it had no kernel - a
     * validation-passing batch answered with the defined not-implemented
     * status - is gone with the kernel (issue #214), and no call replaces it
     * here: an accepted batch now reaches a launch, and these are host
     * pointers this GPU-less test must never hand to one. What that class
     * froze still holds and is asserted where a device can read it: the
     * symbol and the eight reject classes above did not move, and
     * tests/launch_fail.cpp holds the launch half on a GPU-less process. */

    /* The same nine classes on the Zstd entry (issue #427), written out call
     * for call for the reason the two blocks above are: a future entry wired
     * to its own validator, or to none, would pass a test that only counted
     * on the sharing. */
    REQUIRE(cudec_zstd_decompress_batch(0, sizes, dsts, caps, 1, aligned,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, 0, dsts, caps, 1, aligned,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, 0, caps, 1, aligned,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, 0, 1, aligned,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, caps, 1, 0,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, caps, 1, misaligned,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, caps, 0, aligned,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, caps, SIZE_MAX,
                                        aligned,
                                        0) == CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_zstd_decompress_batch(srcs, sizes, dsts, caps, 1, aligned,
                                        0) == CUDEC_ERR_NOT_IMPLEMENTED);

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

    /* The streaming entries and the enum resolve their C linkage here. Every
     * call is an argument reject that returns before any CUDA call. */
    {
        const void* one_src[1] = {0};
        size_t one_size[1] = {0};
        void* one_dst[1] = {0};
        size_t one_cap[1] = {0};
        cudec_chunk_result sres[1];

        /* The one-shot wrapper (no `streams` parameter). */
        REQUIRE(cudec_lz4_decompress_stream(0, one_size, one_dst, one_cap, 1,
                                            CUDEC_MEM_DEVICE, sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null array */
        REQUIRE(cudec_lz4_decompress_stream(one_src, one_size, one_dst, one_cap,
                                            0, CUDEC_MEM_DEVICE, sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* zero chunks */
        REQUIRE(cudec_lz4_decompress_stream(one_src, one_size, one_dst, one_cap,
                                            1, (cudec_mem_space)7, sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* unknown dst_space */

        /* The reusable-context entries. create with a NULL out_ctx, and the
         * decode entry with a NULL ctx / zero chunks / unknown dst_space, all
         * reject synchronously before any CUDA call. The decode rejects use a
         * NULL ctx so no context is created on the GPU-less runner. */
        REQUIRE(cudec_stream_ctx_create(0) == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(cudec_lz4_decompress_stream_ctx(0, one_src, one_size, one_dst,
                                                one_cap, 1, CUDEC_MEM_DEVICE,
                                                sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null ctx */
        REQUIRE(cudec_lz4_decompress_stream_ctx(0, one_src, one_size, one_dst,
                                                one_cap, 0, CUDEC_MEM_DEVICE,
                                                sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null ctx + zero chunks */
        REQUIRE(cudec_lz4_decompress_stream_ctx(0, one_src, one_size, one_dst,
                                                one_cap, 1, (cudec_mem_space)7,
                                                sres) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null ctx + unknown dst_space */

        /* destroy is NULL-safe. */
        cudec_stream_ctx_destroy(0);
    }

    /* The TileStream entries resolve their C linkage here (issue #177). Every
     * call below either rejects on an argument or refuses the bytes, and both
     * happen before any CUDA call, so the whole block runs on the GPU-less
     * runner.
     *
     * The envelope rejects are the interesting half: they are decided by the
     * host walker, so this GPU-less test reaches the parse and not just the
     * pointer checks. `env` is a well-formed one-tile envelope - id/magic
     * complement, tileSizeIdx 1, one 4-byte table entry naming a 1-byte last
     * tile, and that byte present - so the truncations below differ from an
     * accepted stream by the bytes named and by nothing else. */
    {
        unsigned char env[13];
        unsigned char out[16];
        cudec_chunk_result tres[1];
        size_t written = 123;
        size_t tiles = 123;
        size_t total = 123;

        env[0] = 0x04;  /* id = magic ^ 0xFF */
        env[1] = 0xFB;  /* magic */
        env[2] = 1;     /* tile count, low byte */
        env[3] = 0;
        env[4] = (unsigned char)(1u | (1u << 2)); /* idx 1, last tile 1 byte */
        env[5] = 0;
        env[6] = 0;
        env[7] = 0;
        env[8] = 1; /* table entry 0: the last tile's compressed size */
        env[9] = 0;
        env[10] = 0;
        env[11] = 0;
        env[12] = 0xAB; /* the one payload byte the table accounts for */

        REQUIRE(cudec_gdeflate_tilestream_info(0, sizeof env, &tiles, &total) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null stream */
        REQUIRE(tiles == 0 && total == 0);
        REQUIRE(cudec_gdeflate_tilestream_info(env, sizeof env, 0, &total) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null tile_count */
        REQUIRE(cudec_gdeflate_tilestream_info(env, sizeof env, &tiles, 0) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null uncompressed_size */
        REQUIRE(cudec_gdeflate_tilestream_info(env, 4, &tiles, &total) ==
                CUDEC_ERR_CORRUPT_INPUT); /* header truncated */
        REQUIRE(tiles == 0 && total == 0);
        REQUIRE(cudec_gdeflate_tilestream_info(env, sizeof env, &tiles,
                                               &total) == CUDEC_OK);
        REQUIRE(tiles == 1 && total == 1);

        /* The one-shot entry. Nothing here reaches cudec_stream_ctx_create:
         * every case is refused above it. */
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    0, sizeof env, out, sizeof out, CUDEC_MEM_HOST, tres, 1,
                    &written) == CUDEC_ERR_INVALID_ARGUMENT); /* null stream */
        REQUIRE(written == 0);
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    env, sizeof env, out, sizeof out, CUDEC_MEM_HOST, tres, 1,
                    0) == CUDEC_ERR_INVALID_ARGUMENT); /* null bytes_written */
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    env, sizeof env, out, sizeof out, CUDEC_MEM_HOST, 0, 1,
                    &written) == CUDEC_ERR_INVALID_ARGUMENT); /* null results */
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    env, sizeof env, 0, sizeof out, CUDEC_MEM_HOST, tres, 1,
                    &written) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* null dst, nonzero capacity */
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    env, sizeof env, out, sizeof out, (cudec_mem_space)7, tres,
                    1, &written) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* unknown dst_space */
        REQUIRE(written == 0);

        /* A NULL ctx is the _ctx entry's own reject, and it is taken before
         * the envelope is looked at, so no context exists on this runner. */
        REQUIRE(cudec_gdeflate_tilestream_decompress_ctx(
                    0, env, sizeof env, out, sizeof out, CUDEC_MEM_HOST, tres,
                    1, &written) == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(written == 0);

        /* The two content-dependent rejects: they need the envelope parsed, so
         * they prove the one-shot entry reaches the walker before it reaches a
         * device. A one-tile stream declares 1 byte of output and 1 result. */
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    env, sizeof env, out, 0, CUDEC_MEM_HOST, tres, 1,
                    &written) == CUDEC_ERR_OUTPUT_TOO_SMALL);
        REQUIRE(written == 0);
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    env, sizeof env, out, sizeof out, CUDEC_MEM_HOST, tres, 0,
                    &written) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* results array too short */
        REQUIRE(written == 0);
        REQUIRE(cudec_gdeflate_tilestream_decompress(
                    env, 4, out, sizeof out, CUDEC_MEM_HOST, tres, 1,
                    &written) == CUDEC_ERR_CORRUPT_INPUT); /* truncated */
        REQUIRE(written == 0);
    }

    printf("PASS: plain-C caller exercised every public symbol\n");
    return 0;
}
