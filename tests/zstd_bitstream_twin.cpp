/* The CPU twin of the Zstd backward bitstream reader (src/zstd_bitstream.h),
 * the sibling of tests/parser_twin.cpp and tests/snappy_parser_twin.cpp: the
 * single-source unit executed on the host, on the GPU-less CI runner, before
 * any FSE or Huffman code exists to run over it.
 *
 * WHAT THE EXPECTATIONS ARE, because this is the whole weight of the test.
 * Every vector below carries a hand-written bit string: the bits the reader
 * must hand back, in the order it must hand them back, spelled out one
 * character each. They are NOT computed from the reader's own formula, which
 * would only prove the formula equals itself. A reviewer checks a vector by
 * reading the two bytes and the string beside them.
 *
 * NOT DONE, and stated rather than left to be noticed: the vectors are not
 * cross-checked against libzstd's own BIT_initDStream / BIT_readBits. That
 * half of the proof belongs with the M5 oracle and is not reachable from this
 * change. What stands here is arithmetic a person can check, which is less
 * than a second implementation agreeing. */
#include "require.h"
#include "zstd_bitstream.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* Which reject rungs a declared negative reached. Same discipline as the
 * Snappy twin: the enumeration lives once, in the header, and main() requires
 * every rung to have been named by a negative written to reach it. A rung
 * added to the reader with no negative behind it reds this test. */
bool g_reject_covered[cudec_detail::kZstdBitRejectCount] = {false};

void CoverRung(cudec_detail::ZstdBitReject rung) {
    if (rung != cudec_detail::kZstdBitRejectNone) {
        g_reject_covered[rung] = true;
    }
}

/* A stream and the bits it must yield, in consumption order.
 *
 * The final byte carries the start marker: its highest set bit is consumed as
 * the marker and every bit below it is live data, so the padding is the
 * marker plus the zeros above it. The eight vectors below walk that padding
 * from 1 bit wide to 8 bits wide, which is every legal width there is.
 *
 * Both bytes are written out in binary beside each vector so the expected
 * string can be checked without decoding hex. */
struct Vector {
    const char* name;
    Bytes stream;
    const char* bits;
};

std::vector<Vector> MakeVectors() {
    std::vector<Vector> out;
    /* stream[0] = 0xA5 = 1010 0101 in every vector: eight live bits, read
     * from bit 7 down to bit 0, after the final byte is exhausted. */

    /* final 0xD5 = 1101 0101, marker at bit 7, padding 1 bit wide.
     * live from the final byte: bits 6..0 = 101 0101 */
    out.push_back({"padding-1", {0xA5, 0xD5}, "1010101" "10100101"});
    /* final 0x55 = 0101 0101, marker at bit 6, padding 2 bits wide.
     * live: bits 5..0 = 01 0101 */
    out.push_back({"padding-2", {0xA5, 0x55}, "010101" "10100101"});
    /* final 0x2A = 0010 1010, marker at bit 5, padding 3 bits wide.
     * live: bits 4..0 = 0 1010 */
    out.push_back({"padding-3", {0xA5, 0x2A}, "01010" "10100101"});
    /* final 0x15 = 0001 0101, marker at bit 4, padding 4 bits wide.
     * live: bits 3..0 = 0101 */
    out.push_back({"padding-4", {0xA5, 0x15}, "0101" "10100101"});
    /* final 0x0A = 0000 1010, marker at bit 3, padding 5 bits wide.
     * live: bits 2..0 = 010 */
    out.push_back({"padding-5", {0xA5, 0x0A}, "010" "10100101"});
    /* final 0x05 = 0000 0101, marker at bit 2, padding 6 bits wide.
     * live: bits 1..0 = 01 */
    out.push_back({"padding-6", {0xA5, 0x05}, "01" "10100101"});
    /* final 0x02 = 0000 0010, marker at bit 1, padding 7 bits wide.
     * live: bit 0 = 0 */
    out.push_back({"padding-7", {0xA5, 0x02}, "0" "10100101"});
    /* final 0x01 = 0000 0001, marker at bit 0, padding 8 bits wide: the
     * final byte contributes NOTHING but the marker, which is the degenerate
     * end of the range and the one a reader that assumed at least one live
     * bit per byte would get wrong. */
    out.push_back({"padding-8", {0xA5, 0x01}, "10100101"});

    /* One byte only, so the whole stream is the final byte: 0xB3 =
     * 1011 0011, marker at bit 7, live bits 6..0 = 011 0011. */
    out.push_back({"single-byte", {0xB3}, "0110011"});
    /* One byte whose marker is its only set bit: zero live bits. A legal
     * stream that carries no data, and the reader must report exactly that
     * rather than reaching for a byte that is not there. */
    out.push_back({"single-byte-empty", {0x01}, ""});
    /* Three bytes, so the walk crosses more than one byte boundary going
     * backwards. 0x0F = 0000 1111 marker at bit 3, live bits 2..0 = 111;
     * then 0x00 = 0000 0000 whole; then 0xFF = 1111 1111 whole. The zero
     * middle byte is deliberate: only the FINAL byte may not be zero. */
    out.push_back({"three-bytes", {0xFF, 0x00, 0x0F},
                   "111" "00000000" "11111111"});
    return out;
}

uint64_t ValueOf(const std::string& bits) {
    uint64_t value = 0;
    for (const char c : bits) {
        value = (value << 1) | static_cast<uint64_t>(c == '1' ? 1 : 0);
    }
    return value;
}

}  // namespace

int main() {
    const auto vectors = MakeVectors();
    size_t bits_checked = 0;

    for (const auto& v : vectors) {
        const std::string expected = v.bits;

        /* One bit at a time, against the string, character by character. */
        {
            cudec_detail::ZstdBitReader reader{v.stream.data(),
                                               v.stream.size()};
            REQUIRE_CTX(reader.Start() == CUDEC_OK, "vector %s", v.name);
            REQUIRE_CTX(reader.BitsRemaining() == expected.size(),
                        "vector %s: %llu live bits, expected %zu", v.name,
                        static_cast<unsigned long long>(reader.BitsRemaining()),
                        expected.size());
            for (size_t i = 0; i < expected.size(); i++) {
                uint64_t bit = 0;
                REQUIRE_CTX(reader.ReadBits(1, &bit) == CUDEC_OK,
                            "vector %s, bit %zu", v.name, i);
                REQUIRE_CTX(bit == (expected[i] == '1' ? 1u : 0u),
                            "vector %s, bit %zu: read %llu", v.name, i,
                            static_cast<unsigned long long>(bit));
                bits_checked++;
            }
            REQUIRE_CTX(reader.AtEnd(), "vector %s: bits left over", v.name);
            REQUIRE_CTX(reader.BitsRemaining() == 0, "vector %s", v.name);
        }

        /* And in wider reads, which is how every caller will actually use it:
         * the same bits must come back in the same order however they are
         * grouped. Widths 2, 3, 5 and 7 are coprime with 8, so the groups
         * straddle byte boundaries at every offset rather than lining up with
         * them - which is where a backward walk goes wrong if it is going to.
         */
        for (const unsigned width : {2u, 3u, 5u, 7u}) {
            cudec_detail::ZstdBitReader reader{v.stream.data(),
                                               v.stream.size()};
            REQUIRE(reader.Start() == CUDEC_OK);
            size_t at = 0;
            while (expected.size() - at >= width) {
                uint64_t value = 0;
                REQUIRE_CTX(reader.ReadBits(width, &value) == CUDEC_OK,
                            "vector %s, width %u at %zu", v.name, width, at);
                REQUIRE_CTX(value == ValueOf(expected.substr(at, width)),
                            "vector %s, width %u at %zu: read %llu", v.name,
                            width, at,
                            static_cast<unsigned long long>(value));
                at += width;
            }
            REQUIRE_CTX(reader.BitsRemaining() == expected.size() - at,
                        "vector %s, width %u", v.name, width);
        }

        /* A zero-width read yields zero and moves nothing - the FSE and
         * Huffman decoders ask for zero-width fields as a matter of course,
         * and a reader that consumed a bit for one would desynchronise the
         * whole stream silently. */
        {
            cudec_detail::ZstdBitReader reader{v.stream.data(),
                                               v.stream.size()};
            REQUIRE(reader.Start() == CUDEC_OK);
            const uint64_t before = reader.BitsRemaining();
            uint64_t value = 0xFFu;
            REQUIRE(reader.ReadBits(0, &value) == CUDEC_OK);
            REQUIRE(value == 0);
            REQUIRE(reader.BitsRemaining() == before);
        }
    }

    /* Start is idempotent: a caller that reaches for the live-bit count
     * before reading must not consume anything by asking. */
    {
        const Bytes stream = {0xA5, 0xD5};
        cudec_detail::ZstdBitReader reader{stream.data(), stream.size()};
        REQUIRE(reader.Start() == CUDEC_OK);
        const uint64_t total = reader.BitsRemaining();
        REQUIRE(reader.Start() == CUDEC_OK);
        REQUIRE(reader.BitsRemaining() == total);
    }

    /* The negatives, one per rung of the ladder in src/zstd_bitstream.h. Each
     * pins the status AND the rung: the statuses repeat, so a status
     * comparison alone cannot say which refusal was reached. */
    {
        /* Empty stream. */
        cudec_detail::ZstdBitReader reader{nullptr, 0};
        REQUIRE(reader.Start() == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(reader.reject == cudec_detail::kZstdBitRejectEmptyStream);
        CoverRung(reader.reject);
    }
    {
        /* A zero final byte carries no start marker. Not a clamp: the width
         * of the padding would have to be invented, and an invented width
         * decodes a bit sequence the encoder never wrote. */
        const Bytes stream = {0xA5, 0x00};
        cudec_detail::ZstdBitReader reader{stream.data(), stream.size()};
        REQUIRE(reader.Start() == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(reader.reject == cudec_detail::kZstdBitRejectNoStartMarker);
        CoverRung(reader.reject);
        /* And it stays refused: Start is idempotent on success, never on
         * failure. */
        REQUIRE(reader.Start() == CUDEC_ERR_CORRUPT_INPUT);
    }
    {
        /* ReadBits before Start. The reader does not start itself here: a
         * reader whose Start failed must not become usable by asking it for
         * bits instead. */
        const Bytes stream = {0xA5, 0xD5};
        cudec_detail::ZstdBitReader reader{stream.data(), stream.size()};
        uint64_t value = 1;
        REQUIRE(reader.ReadBits(1, &value) == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(reader.reject == cudec_detail::kZstdBitRejectNotStarted);
        REQUIRE(value == 0);
        CoverRung(reader.reject);
    }
    {
        /* A read wider than the accumulator's usable width. */
        const Bytes stream = {0xA5, 0xD5};
        cudec_detail::ZstdBitReader reader{stream.data(), stream.size()};
        REQUIRE(reader.Start() == CUDEC_OK);
        uint64_t value = 1;
        REQUIRE(reader.ReadBits(cudec_detail::kZstdBitReadMax + 1, &value) ==
                CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(reader.reject == cudec_detail::kZstdBitRejectReadTooWide);
        REQUIRE(value == 0);
        CoverRung(reader.reject);
        /* The widest legal read is still legal, so the bound is not off by
         * one - it refuses on 58 and would have to accept 57 if the stream
         * carried that many, which this one does not. The reject it gives
         * instead is the exhaustion rung, not the width rung. */
        cudec_detail::ZstdBitReader wide{stream.data(), stream.size()};
        REQUIRE(wide.Start() == CUDEC_OK);
        REQUIRE(wide.ReadBits(cudec_detail::kZstdBitReadMax, &value) ==
                CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(wide.reject == cudec_detail::kZstdBitRejectPastStart);
    }
    {
        /* Reading past the start of the buffer: the fail-open this unit
         * exists to prevent. The stream carries 15 live bits; asking for 16
         * must refuse rather than pad with zeros. */
        const Bytes stream = {0xA5, 0xD5};
        cudec_detail::ZstdBitReader reader{stream.data(), stream.size()};
        REQUIRE(reader.Start() == CUDEC_OK);
        REQUIRE(reader.BitsRemaining() == 15);
        uint64_t value = 1;
        REQUIRE(reader.ReadBits(16, &value) == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(reader.reject == cudec_detail::kZstdBitRejectPastStart);
        REQUIRE(value == 0);
        REQUIRE(reader.overrun);
        CoverRung(reader.reject);

        /* Sticky. The next read is refused for having overrun already, even
         * though one bit would otherwise have been available - a caller that
         * missed the first return value must not be able to walk back into a
         * stream that already ended. */
        REQUIRE(reader.ReadBits(1, &value) == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(reader.reject == cudec_detail::kZstdBitRejectAlreadyOverrun);
        REQUIRE(value == 0);
        CoverRung(reader.reject);
    }
    {
        /* Exhaustion at exactly the boundary is NOT an overrun: consuming the
         * last bit leaves the reader at end, and only the read after it
         * refuses. The off-by-one in the other direction would refuse a
         * legal final symbol. */
        const Bytes stream = {0xA5, 0x01}; /* eight live bits */
        cudec_detail::ZstdBitReader reader{stream.data(), stream.size()};
        REQUIRE(reader.Start() == CUDEC_OK);
        uint64_t value = 0;
        REQUIRE(reader.ReadBits(8, &value) == CUDEC_OK);
        REQUIRE(value == 0xA5u);
        REQUIRE(reader.AtEnd());
        REQUIRE(!reader.overrun);
        REQUIRE(reader.ReadBits(1, &value) == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(reader.reject == cudec_detail::kZstdBitRejectPastStart);
    }
    {
        /* Exact consumption as the callers will assert it: a stream left
         * unconsumed reports bits remaining, which is the predicate the FSE
         * and literal decoders end on. */
        const Bytes stream = {0xA5, 0xD5};
        cudec_detail::ZstdBitReader reader{stream.data(), stream.size()};
        REQUIRE(reader.Start() == CUDEC_OK);
        uint64_t value = 0;
        REQUIRE(reader.ReadBits(7, &value) == CUDEC_OK);
        REQUIRE(!reader.AtEnd());
        REQUIRE(reader.BitsRemaining() == 8);
    }

    /* Every rung named by a negative written to reach it, walked rather than
     * restated: a rung added to the reader with none behind it lands here as
     * a hole with its number on it. */
    {
        size_t rungs_covered = 0;
        for (int rung = cudec_detail::kZstdBitRejectNone + 1;
             rung < cudec_detail::kZstdBitRejectCount; rung++) {
            REQUIRE_CTX(g_reject_covered[rung],
                        "reject rung %d has no declared negative that reaches "
                        "it - add one, or the rung is untested",
                        rung);
            rungs_covered++;
        }
        REQUIRE(rungs_covered ==
                static_cast<size_t>(cudec_detail::kZstdBitRejectCount) - 1);
    }

    std::printf("PASS: %zu vectors over every legal padding width, %zu bits "
                "checked one at a time against hand-written expectations and "
                "again at widths 2/3/5/7; %d of %d reject rungs covered by a "
                "declared negative\n",
                vectors.size(), bits_checked,
                static_cast<int>(cudec_detail::kZstdBitRejectCount) - 1,
                static_cast<int>(cudec_detail::kZstdBitRejectCount) - 1);
    return 0;
}
