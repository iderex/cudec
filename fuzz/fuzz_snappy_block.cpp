/* Differential fuzz target over the Snappy element parser (issue #90), the
 * sibling of fuzz_lz4_block.cpp and built to the shape that one settled.
 *
 * Two properties, and the second is the one a parity comparison cannot see.
 *
 * Parity, in the fail-open direction: whenever the twin ACCEPTS, snappy must
 * accept the same bytes and produce the identical output and size. The other
 * direction is not asserted here. The twin is not knowingly stricter than
 * snappy anywhere - tests/snappy_parser_twin.cpp requires its
 * stricter-than-oracle count over the mutation corpus to be zero - but a
 * fuzzer reaching bytes that corpus never produced is exactly where a
 * legitimate strictness could first appear, and trapping on one would say
 * "fail-open" about the opposite of a fail-open.
 *
 * Bounds, unconditionally: the driver counts every element the caller could
 * not have executed without leaving its buffers, every cursor left outside
 * its own bounds, and every parse that did not terminate. That count must
 * stay zero whatever the verdict is, because a rejected stream whose reject
 * arrived only after an over-read is still an over-read.
 *
 * The whole input is the stream. Unlike the LZ4 block format, a Snappy stream
 * carries its own uncompressed length, so there is no capacity to fuzz
 * alongside it: both sides size the destination from the declaration and
 * refuse the same declarations above the same bound.
 */
#include "cudec.h"
#include "snappy_twin.h"

#include "snappy.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

/* The shared refusal both sides apply. A stream may declare up to 4 GiB - 1
 * of output and a mutated one will, so it is bounded here rather than
 * allocated; the number is this harness's and not snappy's, and it is handed
 * to the twin as its capacity so the two refuse the same declarations for the
 * same reason. Well below tests/fixtures.h's 256 MiB, because a fuzzer that
 * spends its time in a quarter-gigabyte memset explores nothing. */
constexpr uint64_t kMaxDeclared = 1u << 22;

/* Bounded so libFuzzer explores the parser rather than the allocator. */
constexpr size_t kMaxStream = 1u << 14;

struct Decoded {
    bool ok;
    std::vector<unsigned char> bytes;
};

/* snappy's exact-consume, exact-produce decode, under this harness's bound.
 * The same shape as SnappyOracleDecodes in tests/fixtures.cpp, restated here
 * for the same reason the LZ4 target calls LZ4_decompress_safe directly: the
 * fuzz binary links the oracle, not the test-fixture library. */
Decoded OracleDecode(const unsigned char* stream, size_t stream_size) {
    Decoded decoded{false, {}};
    if (stream_size == 0) {
        return decoded; /* never hand snappy a null source pointer */
    }
    const char* src = reinterpret_cast<const char*>(stream);
    size_t declared = 0;
    if (!snappy::GetUncompressedLength(src, stream_size, &declared)) {
        return decoded;
    }
    if (declared > kMaxDeclared) {
        return decoded;
    }
    decoded.bytes.assign(declared, 0);
    if (!snappy::RawUncompress(src, stream_size,
                               reinterpret_cast<char*>(decoded.bytes.data()))) {
        /* A rejected stream produces no output: a caller that reads the
         * buffer after a false verdict must find nothing to mistake for one. */
        decoded.bytes.clear();
        return decoded;
    }
    decoded.ok = true;
    return decoded;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    size_t stream_size = size;
    if (stream_size > kMaxStream) {
        stream_size = kMaxStream;
    }

    /* libFuzzer hands out a slice of a buffer sized to -max_len, not to this
     * input, so a read past `stream_size` would land in that slack and stay
     * green. Both sides run over one exactly-sized copy instead. The twin
     * driver tightens again internally, which is its own guarantee to its
     * other caller. */
    auto stream = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(stream.get(), data, stream_size);
    }

    cudec_test::SnappyTwinObserver observer;
    std::vector<unsigned char> twin_bytes;
    const cudec_status twin = cudec_test::SnappyTwinDecode(
        stream.get(), stream_size, kMaxDeclared, &twin_bytes, &observer);
    Decoded oracle = OracleDecode(stream.get(), stream_size);

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to prove the comparisons below are
     * live without waiting for a real divergence: a second binary built with
     * this defined perturbs an accepted oracle output, so a harness that had
     * silently stopped comparing would pass where this one traps. Never
     * define it in a build whose findings are being believed. */
    if (oracle.ok && !oracle.bytes.empty()) {
        oracle.bytes[0] ^= 0xFFu;
    }
#endif

    if (observer.bounds_violations != 0) {
        std::fprintf(stderr,
                     "BOUNDS VIOLATION: %zu on a %zu-byte stream (status %d)\n",
                     observer.bounds_violations, stream_size,
                     static_cast<int>(twin));
        __builtin_trap();
    }
    if (twin != CUDEC_OK) {
        return 0; /* the stricter direction is not asserted */
    }
    if (!oracle.ok) {
        std::fprintf(stderr,
                     "FAIL-OPEN: twin accepted (%zu bytes) a stream snappy "
                     "rejected; stream=%zu\n",
                     twin_bytes.size(), stream_size);
        __builtin_trap();
    }
    if (twin_bytes.size() != oracle.bytes.size()) {
        std::fprintf(stderr,
                     "SIZE DIVERGENCE: twin=%zu oracle=%zu stream=%zu\n",
                     twin_bytes.size(), oracle.bytes.size(), stream_size);
        __builtin_trap();
    }
    if (!twin_bytes.empty() &&
        std::memcmp(twin_bytes.data(), oracle.bytes.data(),
                    twin_bytes.size()) != 0) {
        std::fprintf(stderr, "BYTE DIVERGENCE: %zu bytes, stream=%zu\n",
                     twin_bytes.size(), stream_size);
        __builtin_trap();
    }
    return 0;
}
