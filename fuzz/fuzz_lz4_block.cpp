/* Differential fuzz target over the LZ4 block parser (issue #140).
 *
 * Property, the fail-open direction: whenever the twin ACCEPTS, liblz4 must
 * accept the same bytes at the same capacity and produce the identical output
 * and size. The other direction is not asserted, because the twin is
 * deliberately stricter than liblz4 in at least the offset==0 family
 * (tests/parser_twin.cpp pins that divergence); asserting it would red on
 * legitimate strictness.
 *
 * Both sides get a tight allocation for the stream and for the output, so an
 * over-read past the last stream byte or an over-write past the capacity lands
 * in an ASan redzone rather than in vector slack.
 */
#include "cudec.h"
#include "lz4_block.h"

#include "lz4.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

/* Ceilings, so the fuzzer explores the parser instead of the allocator. */
constexpr size_t kMaxCapacity = 1u << 16;
constexpr size_t kMaxStream = 1u << 14;

struct Decoded {
    bool ok;
    size_t size;
    std::unique_ptr<unsigned char[]> bytes; /* tight: exactly `capacity` */
};

/* Sequential reference execution of the parsed sequences - the same driver
 * shape as TwinDecode in tests/parser_twin.cpp. */
Decoded TwinDecode(const unsigned char* stream, size_t stream_size,
                   size_t capacity) {
    Decoded d{false, 0, nullptr};
    d.bytes = std::make_unique<unsigned char[]>(capacity);
    std::memset(d.bytes.get(), 0, capacity);

    cudec_detail::Lz4Parser parser{stream, stream_size, capacity};
    cudec_detail::Lz4Sequence seq;
    bool done = false;
    while (true) {
        const cudec_status status = parser.Next(&seq, &done);
        if (status != CUDEC_OK) {
            return d;
        }
        for (uint64_t i = 0; i < seq.literals_len; i++) {
            d.bytes[seq.literals_dst + i] = stream[seq.literals_src + i];
        }
        for (uint64_t i = 0; i < seq.match_len; i++) {
            d.bytes[seq.match_dst + i] = d.bytes[seq.match_src + i];
        }
        if (done) {
            break;
        }
    }
    d.ok = true;
    d.size = static_cast<size_t>(parser.dst_pos);
    return d;
}

Decoded OracleDecode(const unsigned char* stream, size_t stream_size,
                     size_t capacity) {
    Decoded d{false, 0, nullptr};
    d.bytes = std::make_unique<unsigned char[]>(capacity);
    std::memset(d.bytes.get(), 0, capacity);
    if (stream_size == 0) {
        return d; /* never hand liblz4 a null source pointer */
    }
    const int written = LZ4_decompress_safe(
        reinterpret_cast<const char*>(stream),
        reinterpret_cast<char*>(d.bytes.get()), static_cast<int>(stream_size),
        static_cast<int>(capacity));
    if (written < 0) {
        return d;
    }
    d.ok = true;
    d.size = static_cast<size_t>(written);
    return d;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    /* The fuzzer controls the output capacity as well as the stream: with a
     * fixed capacity, CUDEC_ERR_OUTPUT_TOO_SMALL swamps the corpus and the
     * deeper reject branches stay unreached. Two header bytes, then the
     * stream. Both sides are handed the identical capacity. */
    if (size < 2) {
        return 0;
    }
    const size_t capacity =
        (static_cast<size_t>(data[0]) << 8 | static_cast<size_t>(data[1])) %
        (kMaxCapacity + 1);
    const uint8_t* body = data + 2;
    size_t body_size = size - 2;
    if (body_size > kMaxStream) {
        body_size = kMaxStream;
    }

    /* Tight copy of the stream: an over-read past the last byte must land in
     * an ASan redzone, not in slack the allocator happened to round up. */
    auto tight = std::make_unique<unsigned char[]>(body_size);
    if (body_size != 0) {
        std::memcpy(tight.get(), body, body_size);
    }

    const Decoded twin = TwinDecode(tight.get(), body_size, capacity);
    const Decoded oracle = OracleDecode(tight.get(), body_size, capacity);

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to prove the comparison below is live:
     * a second binary built with this defined perturbs an accepted output, so
     * a harness that had silently stopped comparing would pass where this one
     * traps. Never define it in a build whose findings are being believed. */
    if (twin.ok && twin.size != 0) {
        const_cast<Decoded&>(twin).bytes[0] ^= 0xFFu;
    }
#endif

    if (!twin.ok) {
        return 0; /* the twin may be stricter; that direction is not asserted */
    }
    if (!oracle.ok) {
        std::fprintf(stderr,
                     "FAIL-OPEN: twin accepted (%zu bytes) a stream liblz4 "
                     "rejected; capacity=%zu stream=%zu\n",
                     twin.size, capacity, body_size);
        __builtin_trap();
    }
    if (twin.size != oracle.size) {
        std::fprintf(stderr,
                     "SIZE DIVERGENCE: twin=%zu oracle=%zu capacity=%zu "
                     "stream=%zu\n",
                     twin.size, oracle.size, capacity, body_size);
        __builtin_trap();
    }
    if (twin.size != 0 &&
        std::memcmp(twin.bytes.get(), oracle.bytes.get(), twin.size) != 0) {
        std::fprintf(stderr,
                     "BYTE DIVERGENCE: %zu bytes, capacity=%zu stream=%zu\n",
                     twin.size, capacity, body_size);
        __builtin_trap();
    }
    return 0;
}

/* Built and run so far without a CMake seam, because no machine available to
 * this work has CMake and Clang together. The exact invocation that produced
 * the findings recorded on issue #140, against the liblz4 release tarball
 * pinned in tests/CMakeLists.txt and hash-verified before use:
 *
 *   clang -std=c11 -O2 -g -fsanitize=address,undefined \
 *     -fno-sanitize-recover=all -isystem lz4-1.10.0/lib \
 *     -c lz4-1.10.0/lib/lz4.c -o lz4_oracle.o
 *   clang++ -std=c++17 -g -O1 -fsanitize=fuzzer,address,undefined \
 *     -fno-sanitize-recover=all -I include -I src \
 *     -isystem lz4-1.10.0/lib -o fuzz_lz4_block \
 *     fuzz/fuzz_lz4_block.cpp lz4_oracle.o
 *
 * The CMake option that replaces it, and the committed seed corpus, are what
 * issue #140 still owes.
 */
