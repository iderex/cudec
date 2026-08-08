/* Differential fuzz target over the LZ4 block parser (issue #140).
 *
 * Property, the fail-open direction: whenever the twin ACCEPTS, liblz4 must
 * accept the same bytes at the same capacity and produce the identical output
 * and size. The other direction is not asserted, because the twin is
 * deliberately stricter than liblz4 in at least the offset==0 family
 * (tests/parser_twin.cpp pins that divergence); asserting it would red on
 * legitimate strictness.
 *
 * The twin side is tests/lz4_twin.h - the same driver tests/parser_twin.cpp
 * runs, not a second copy of it, so this target cannot go green about a
 * parser execution the test net never performs. Both sides get tight
 * allocations, so an over-read past the last stream byte or a write past the
 * declared capacity lands in an ASan redzone rather than in slack. */
#include "cudec.h"
#include "lz4_twin.h"

#include "lz4.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

/* Ceilings, so the fuzzer explores the parser instead of the allocator. The
 * capacity ceiling is what two header bytes can express; the stream ceiling
 * is set below libFuzzer's own -max_len so a length-prefix blow-up cannot
 * turn a corpus entry into an allocation. */
constexpr size_t kMaxStream = 1u << 14;

struct Decoded {
    bool ok;
    size_t size;
    std::unique_ptr<unsigned char[]> bytes; /* tight: exactly `capacity` */
};

Decoded OracleDecode(const unsigned char* stream, size_t stream_size,
                     size_t capacity) {
    Decoded decoded{false, 0, std::make_unique<unsigned char[]>(capacity)};
    if (capacity != 0) {
        std::memset(decoded.bytes.get(), 0, capacity);
    }
    if (stream_size == 0) {
        return decoded; /* never hand liblz4 a null source pointer */
    }
    const int written = LZ4_decompress_safe(
        reinterpret_cast<const char*>(stream),
        reinterpret_cast<char*>(decoded.bytes.get()),
        static_cast<int>(stream_size), static_cast<int>(capacity));
    if (written < 0) {
        return decoded;
    }
    decoded.ok = true;
    decoded.size = static_cast<size_t>(written);
    return decoded;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    /* The fuzzer controls the output capacity as well as the stream: with a
     * fixed capacity, CUDEC_ERR_OUTPUT_TOO_SMALL swamps the corpus and the
     * deeper reject branches stay unreached. Two big-endian header bytes,
     * then the stream. Both sides are handed the identical capacity, so a
     * divergence is never an artefact of comparing two different contracts. */
    if (size < 2) {
        return 0;
    }
    const size_t capacity =
        static_cast<size_t>(data[0]) << 8 | static_cast<size_t>(data[1]);
    const uint8_t* body = data + 2;
    size_t body_size = size - 2;
    if (body_size > kMaxStream) {
        body_size = kMaxStream;
    }

    /* libFuzzer hands out a slice of a buffer sized to -max_len, not to this
     * input, so a read past `body_size` would land in that slack and stay
     * green. Both sides are run over one exactly-sized copy instead. The twin
     * driver tightens again internally - that is its own guarantee to its
     * other caller, and it costs one copy of at most 16 KiB here. */
    auto stream = std::make_unique<unsigned char[]>(body_size);
    if (body_size != 0) {
        std::memcpy(stream.get(), body, body_size);
    }

    const cudec_test::Lz4TwinResult twin =
        cudec_test::Lz4TwinDecode(stream.get(), body_size, capacity);
    Decoded oracle = OracleDecode(stream.get(), body_size, capacity);

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to prove the comparison below is live
     * without waiting for a real divergence: a second binary built with this
     * defined perturbs an accepted output, so a harness that had silently
     * stopped comparing would pass where this one traps. Never define it in a
     * build whose findings are being believed. */
    if (oracle.ok && oracle.size != 0) {
        oracle.bytes[0] ^= 0xFFu;
    }
#endif

    if (twin.status != CUDEC_OK) {
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
