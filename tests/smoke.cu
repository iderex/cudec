/* M0 smoke test: the library links, the version matches the header, and
 * the batch entry point honors its documented contract on a real device -
 * fail-closed argument validation, asynchronous submission, per-chunk
 * device-side reporting, and a real empty-block decode. */
#include "cudec.h"
#include "require.h"

#include "vendor_rt_test.h"

#include <cstdio>

int main() {
    REQUIRE(cudec_version() == CUDEC_VERSION_MAJOR * 10000 +
                                   CUDEC_VERSION_MINOR * 100 +
                                   CUDEC_VERSION_PATCH);

    /* Validation needs no CUDA context: this rejects before the process
     * has made a single CUDA call. */
    const void* const* no_srcs = nullptr;
    const size_t* no_sizes = nullptr;
    void* const* no_dsts = nullptr;
    const size_t* no_caps = nullptr;
    cudec_chunk_result* no_results = nullptr;
    REQUIRE(cudec_lz4_decompress_batch(no_srcs, no_sizes, no_dsts, no_caps, 1,
                                       no_results, nullptr) ==
            CUDEC_ERR_INVALID_ARGUMENT);

    /* 257 chunks span many blocks of the warp-per-chunk launch (128
     * threads = 4 chunks per block), so the multi-block index arithmetic
     * actually executes on device. */
    const size_t chunk_count = 257;
    const size_t chunk_bytes = 64;

    unsigned char* buffers[2 * chunk_count];
    const void* h_src_ptrs[chunk_count];
    void* h_dst_ptrs[chunk_count];
    size_t h_sizes[chunk_count];
    /* Each chunk is a valid empty LZ4 block (a lone 0x00 token) so the
     * batch exercises a real successful decode end to end at the ABI
     * boundary; the fixture test covers content-bearing streams. */
    for (size_t i = 0; i < chunk_count; i++) {
        REQUIRE_RT(cudec_rt::device_malloc(&buffers[i], chunk_bytes));
        REQUIRE_RT(cudec_rt::device_memset(buffers[i], 0, 1));
        REQUIRE_RT(
            cudec_rt::device_malloc(&buffers[chunk_count + i], chunk_bytes));
        h_src_ptrs[i] = buffers[i];
        h_dst_ptrs[i] = buffers[chunk_count + i];
        h_sizes[i] = 1;
    }

    const void** d_src_ptrs;
    void** d_dst_ptrs;
    size_t* d_src_sizes;
    size_t* d_dst_caps;
    cudec_chunk_result* d_results;
    REQUIRE_RT(cudec_rt::device_malloc(&d_src_ptrs, sizeof(h_src_ptrs)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_dst_ptrs, sizeof(h_dst_ptrs)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_src_sizes, sizeof(h_sizes)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_dst_caps, sizeof(h_sizes)));
    REQUIRE_RT(
        cudec_rt::device_malloc(&d_results, chunk_count * sizeof(*d_results)));
    REQUIRE_RT(cudec_rt::memcpy(d_src_ptrs, h_src_ptrs, sizeof(h_src_ptrs),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_dst_ptrs, h_dst_ptrs, sizeof(h_dst_ptrs),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_src_sizes, h_sizes, sizeof(h_sizes),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_dst_caps, h_sizes, sizeof(h_sizes),
                            cudec_rt::memcpy_h2d));

    /* The public header's stream type is one pointer type on both backends
     * (masterplan section 15.6): abi_stream is the identity on CUDA and the
     * one cast a HIP caller makes, and the binary compatibility the header
     * claims is exercised right here. */
    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    cudec_stream_t api_stream = cudec_rt::abi_stream(stream);

    /* Every reject path, one at a time, so dropping any single validation
     * check fails the test: each missing array, a misaligned result
     * buffer, an empty batch, and a batch no launch can cover. Rejection
     * happens before anything is dereferenced or launched. */
    cudec_chunk_result* misaligned = reinterpret_cast<cudec_chunk_result*>(
        reinterpret_cast<char*>(d_results) + 8);
#define REQUIRE_REJECTED(srcs, sizes, dsts, caps, count, results)          \
    REQUIRE(cudec_lz4_decompress_batch(srcs, sizes, dsts, caps, count,     \
                                       results, api_stream) ==             \
            CUDEC_ERR_INVALID_ARGUMENT)
    REQUIRE_REJECTED(nullptr, d_src_sizes, d_dst_ptrs, d_dst_caps,
                     chunk_count, d_results);
    REQUIRE_REJECTED(d_src_ptrs, nullptr, d_dst_ptrs, d_dst_caps,
                     chunk_count, d_results);
    REQUIRE_REJECTED(d_src_ptrs, d_src_sizes, nullptr, d_dst_caps,
                     chunk_count, d_results);
    REQUIRE_REJECTED(d_src_ptrs, d_src_sizes, d_dst_ptrs, nullptr,
                     chunk_count, d_results);
    REQUIRE_REJECTED(d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_caps,
                     chunk_count, nullptr);
    REQUIRE_REJECTED(d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_caps,
                     chunk_count, misaligned);
    REQUIRE_REJECTED(d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_caps, 0,
                     d_results);
    REQUIRE_REJECTED(d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_caps,
                     SIZE_MAX, d_results);
#undef REQUIRE_REJECTED

    /* Poison the result buffer so the per-field checks below prove the
     * kernel wrote every field, not that the allocation happened to be
     * zero-filled. */
    REQUIRE_RT(cudec_rt::device_memset(d_results, 0xFF,
                                       chunk_count * sizeof(*d_results)));

    /* A stale caller-side error must not be misreported as this call's
     * failure: plant one, verify it is actually pending, then expect a
     * clean submission. */
    REQUIRE(cudec_rt::set_device(12345) != cudec_rt::success);
    REQUIRE(cudec_rt::peek_at_last_error() != cudec_rt::success);

    /* A rejected call makes no CUDA call, so the planted error must still
     * be pending afterwards - only the passing call below consumes it. */
    REQUIRE(cudec_lz4_decompress_batch(d_src_ptrs, d_src_sizes, d_dst_ptrs,
                                       d_dst_caps, 0, d_results,
                                       api_stream) ==
            CUDEC_ERR_INVALID_ARGUMENT);
    REQUIRE(cudec_rt::peek_at_last_error() != cudec_rt::success);

    REQUIRE(cudec_lz4_decompress_batch(d_src_ptrs, d_src_sizes, d_dst_ptrs,
                                       d_dst_caps, chunk_count, d_results,
                                       api_stream) == CUDEC_OK);
    REQUIRE_RT(cudec_rt::stream_synchronize(stream));

    cudec_chunk_result h_results[chunk_count];
    REQUIRE_RT(cudec_rt::memcpy(h_results, d_results, sizeof(h_results),
                            cudec_rt::memcpy_d2h));
    for (size_t i = 0; i < chunk_count; i++) {
        REQUIRE(h_results[i].status == CUDEC_OK);
        REQUIRE(h_results[i].reserved == 0);
        REQUIRE(h_results[i].bytes_written == 0);
    }

    REQUIRE_RT(cudec_rt::stream_destroy(stream));
    std::printf("PASS: version + fail-closed validation + %zu-chunk "
                "empty-block decode on device\n",
                chunk_count);
    return 0;
}
