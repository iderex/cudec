/* The match longer than 2^32, decoded on the device (issue #331).
 *
 * src/chunk_decode.cuh runs the closed-form overlap gather at two arithmetic
 * widths and picks the 32-bit one under `seq.match_len <= UINT32_MAX`. Every
 * other test in the tree stays far below that bound, so deleting the test and
 * leaving the narrow arm unconditional left the whole suite green - a guard
 * the suite could not prove. This reaches it.
 *
 * One valid LZ4 block: three literals, one match of 2^32 + 4096 bytes at
 * offset 3, a five-byte literals-only tail. The output is "ABC" repeating,
 * so the expected byte at index i is "ABC"[i % 3] for as far as the match
 * reaches, and no 4 GiB comparison is needed to separate the two decoders:
 * 2^32 mod 3 == 1, so a gather that truncates its loop index to 32 bits
 * shifts the phase of the window by one at the 4 GiB mark. A 64-byte window
 * astride that mark carries the whole verdict.
 *
 * The block's well-formedness is liblz4's verdict rather than an assertion
 * about the format: the same construction at a small match length is decoded
 * by LZ4_decompress_safe first, and only then is the large one built.
 *
 * It allocates 4 GiB and walks all of it with a single warp, because one
 * chunk is one warp by design. That is why it is not in the default run, and
 * why a device that cannot hold the allocation FAILS here rather than
 * skipping: a skip on the only test that reaches an arm reports the same
 * silence as a pass. */
#include "cudec.h"
#include "fixtures.h"
#include "require.h"

#include "vendor_rt_test.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

const unsigned char kSeed[3] = {'A', 'B', 'C'};
const unsigned char kTail[5] = {'w', 'x', 'y', 'z', '.'};

/* LZ4 block: token, literals, 2-byte offset, match-length continuation, then
 * a terminal literals-only sequence. Nothing here is cudec's encoder - the
 * bytes are hand-built so the reference can rule on them. */
std::vector<unsigned char> MakeBlock(uint64_t match_len) {
    std::vector<unsigned char> out;
    const uint64_t encoded = match_len - 4; /* LZ4 minmatch */
    const uint64_t low = encoded < 15 ? encoded : 15;
    out.push_back(static_cast<unsigned char>((3u << 4) | low));
    out.insert(out.end(), kSeed, kSeed + sizeof(kSeed));
    out.push_back(3); /* offset, little-endian */
    out.push_back(0);
    if (encoded >= 15) {
        uint64_t rest = encoded - 15;
        while (rest >= 255) {
            out.push_back(0xFF);
            rest -= 255;
        }
        out.push_back(static_cast<unsigned char>(rest));
    }
    out.push_back(static_cast<unsigned char>(sizeof(kTail) << 4));
    out.insert(out.end(), kTail, kTail + sizeof(kTail));
    return out;
}

uint64_t DecodedSize(uint64_t match_len) {
    return sizeof(kSeed) + match_len + sizeof(kTail);
}

unsigned char ExpectedByte(uint64_t match_len, uint64_t i) {
    const uint64_t copied = sizeof(kSeed) + match_len;
    if (i < copied) {
        return kSeed[i % sizeof(kSeed)];
    }
    return kTail[i - copied];
}

/* The reference's verdict on the shape, at a length that fits in a host
 * buffer. If liblz4 refuses this, the large block below is not a decoder
 * disagreement, it is a malformed stream, and saying so here keeps the two
 * apart. */
int CheckShapeAgainstOracle() {
    const uint64_t match_len = 4096;
    const std::vector<unsigned char> block = MakeBlock(match_len);
    const size_t decoded_size = static_cast<size_t>(DecodedSize(match_len));
    std::vector<unsigned char> decoded;
    REQUIRE(OracleDecodes(block, decoded_size, &decoded));
    REQUIRE(decoded.size() == decoded_size);
    for (size_t i = 0; i < decoded_size; i++) {
        REQUIRE_CTX(decoded[i] == ExpectedByte(match_len, i),
                    "oracle byte %zu is 0x%02X, expected 0x%02X", i,
                    decoded[i], ExpectedByte(match_len, i));
    }
    return 0;
}

struct DeviceBatch {
    unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    const void** src_ptrs = nullptr;
    void** dst_ptrs = nullptr;
    size_t* src_sizes = nullptr;
    size_t* dst_caps = nullptr;
    cudec_chunk_result* results = nullptr;
};

}  // namespace

int main() {
    if (CheckShapeAgainstOracle() != 0) {
        return 1;
    }

    /* 2^32 + 4096: past the bound by more than a page, so the divergence is
     * not a one-byte edge effect, and 4096 mod 3 != 0 so the tail of the
     * match is not accidentally in phase either way. */
    const uint64_t match_len = (uint64_t{1} << 32) + 4096;
    const std::vector<unsigned char> block = MakeBlock(match_len);
    const uint64_t dst_capacity = DecodedSize(match_len);

    /* Refused, never skipped. The free-memory query is asked before anything is
     * allocated so the message names the shortfall instead of an
     * out-of-memory return several calls later. */
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    REQUIRE_RT(cudec_rt::mem_get_info(&free_bytes, &total_bytes));
    const uint64_t needed = dst_capacity + block.size();
    REQUIRE_CTX(free_bytes >= needed,
                "this test needs %llu bytes on the device and %zu are free "
                "of %zu total - it is refused rather than skipped, because "
                "it is the only test that reaches the >4 GiB match arm",
                static_cast<unsigned long long>(needed), free_bytes,
                total_bytes);

    DeviceBatch b;
    REQUIRE_RT(cudec_rt::device_malloc(&b.src, block.size()));
    REQUIRE_RT(
        cudec_rt::device_malloc(&b.dst, static_cast<size_t>(dst_capacity)));
    REQUIRE_RT(cudec_rt::memcpy(b.src, block.data(), block.size(),
                            cudec_rt::memcpy_h2d));

    const void* h_src_ptr = b.src;
    void* h_dst_ptr = b.dst;
    const size_t h_src_size = block.size();
    const size_t h_dst_cap = static_cast<size_t>(dst_capacity);
    REQUIRE_RT(cudec_rt::device_malloc(&b.src_ptrs, sizeof(h_src_ptr)));
    REQUIRE_RT(cudec_rt::device_malloc(&b.dst_ptrs, sizeof(h_dst_ptr)));
    REQUIRE_RT(cudec_rt::device_malloc(&b.src_sizes, sizeof(h_src_size)));
    REQUIRE_RT(cudec_rt::device_malloc(&b.dst_caps, sizeof(h_dst_cap)));
    REQUIRE_RT(cudec_rt::device_malloc(&b.results, sizeof(*b.results)));
    REQUIRE_RT(cudec_rt::memcpy(b.src_ptrs, &h_src_ptr, sizeof(h_src_ptr),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b.dst_ptrs, &h_dst_ptr, sizeof(h_dst_ptr),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b.src_sizes, &h_src_size, sizeof(h_src_size),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b.dst_caps, &h_dst_cap, sizeof(h_dst_cap),
                            cudec_rt::memcpy_h2d));

    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    REQUIRE(cudec_lz4_decompress_batch(
                b.src_ptrs, b.src_sizes, b.dst_ptrs, b.dst_caps, 1, b.results,
                cudec_rt::abi_stream(stream)) == CUDEC_OK);
    REQUIRE_RT(cudec_rt::stream_synchronize(stream));

    cudec_chunk_result result{};
    REQUIRE_RT(cudec_rt::memcpy(&result, b.results, sizeof(result),
                            cudec_rt::memcpy_d2h));
    REQUIRE_CTX(result.status == CUDEC_OK, "chunk status %d", result.status);
    REQUIRE_CTX(result.bytes_written == dst_capacity,
                "bytes_written %llu, expected %llu",
                static_cast<unsigned long long>(result.bytes_written),
                static_cast<unsigned long long>(dst_capacity));

    /* The 64-byte window astride the 4 GiB mark. 3 + 2^32 is the first byte
     * a truncated loop index can reach, and it sits inside this window. */
    const uint64_t window_begin = (uint64_t{1} << 32) - 29;
    const size_t window_bytes = 64;
    unsigned char window[window_bytes];
    REQUIRE_RT(cudec_rt::memcpy(window, b.dst + window_begin, window_bytes,
                            cudec_rt::memcpy_d2h));
    size_t differing = 0;
    for (size_t i = 0; i < window_bytes; i++) {
        if (window[i] != ExpectedByte(match_len, window_begin + i)) {
            differing++;
        }
    }
    REQUIRE_CTX(differing == 0,
                "window [%llu, %llu): %zu of %zu bytes differ",
                static_cast<unsigned long long>(window_begin),
                static_cast<unsigned long long>(window_begin + window_bytes),
                differing, window_bytes);

    /* Both ends too, so a decode that got the window right by writing
     * nothing at all cannot pass. */
    unsigned char head[16];
    unsigned char tail[16];
    REQUIRE_RT(
        cudec_rt::memcpy(head, b.dst, sizeof(head), cudec_rt::memcpy_d2h));
    REQUIRE_RT(cudec_rt::memcpy(tail, b.dst + dst_capacity - sizeof(tail),
                            sizeof(tail), cudec_rt::memcpy_d2h));
    for (size_t i = 0; i < sizeof(head); i++) {
        REQUIRE_CTX(head[i] == ExpectedByte(match_len, i),
                    "head byte %zu is 0x%02X, expected 0x%02X", i, head[i],
                    ExpectedByte(match_len, i));
    }
    for (size_t i = 0; i < sizeof(tail); i++) {
        const uint64_t at = dst_capacity - sizeof(tail) + i;
        REQUIRE_CTX(tail[i] == ExpectedByte(match_len, at),
                    "tail byte %llu is 0x%02X, expected 0x%02X",
                    static_cast<unsigned long long>(at), tail[i],
                    ExpectedByte(match_len, at));
    }

    REQUIRE_RT(cudec_rt::stream_destroy(stream));
    REQUIRE_RT(cudec_rt::device_free(b.results));
    REQUIRE_RT(cudec_rt::device_free(b.dst_caps));
    REQUIRE_RT(cudec_rt::device_free(b.src_sizes));
    REQUIRE_RT(cudec_rt::device_free(b.dst_ptrs));
    REQUIRE_RT(cudec_rt::device_free(b.src_ptrs));
    REQUIRE_RT(cudec_rt::device_free(b.dst));
    REQUIRE_RT(cudec_rt::device_free(b.src));

    std::printf(
        "PASS: 2^32 + 4096 byte match at offset 3 decodes bit-exact across "
        "the 4 GiB mark\n");
    return 0;
}
