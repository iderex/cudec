/* The decode behaviours the M3 research took from a source read of snappy
 * 1.2.2, executed against the pinned oracle instead of argued from the code
 * (issue #155). Every assertion below names the reference line it was read
 * from, and the streams are hand-built byte sequences rather than compressor
 * output: a behaviour the compressor never emits is exactly the one a hostile
 * stream will carry.
 *
 * Line citations are into the pinned tree, google/snappy 1.2.2, as fetched by
 * the URL_HASH in tests/CMakeLists.txt. They move only when that pin moves.
 *
 * No cudec code is under test here. This is the reference's behaviour, which
 * the fail-closed matrix is then allowed to lock against. */
#include "fixtures.h"
#include "require.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

void Append(Bytes* out, const Bytes& more) {
    out->insert(out->end(), more.begin(), more.end());
}

/* A literal element carrying `data`, in the short form: the tag's upper six
 * bits hold len-1 (snappy.cc:1613-1614), which reaches 60 bytes before the
 * length needs bytes of its own (snappy.cc:1624). */
Bytes Literal(const Bytes& data) {
    Bytes out;
    out.push_back(static_cast<unsigned char>((data.size() - 1) << 2));
    Append(&out, data);
    return out;
}

/* A copy element with a 2-byte offset, tag 10: the length comes out of the
 * kLengthMinusOffset table (snappy.cc:1657, snappy.cc:1660), so lengths 1..64
 * are encodable - including the 1..3 that the 1-byte-offset form cannot
 * express at all. */
Bytes Copy2(size_t length, uint16_t offset) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((length - 1) << 2) | 2));
    out.push_back(static_cast<unsigned char>(offset & 0xff));
    out.push_back(static_cast<unsigned char>((offset >> 8) & 0xff));
    return out;
}

/* A copy element with a 4-byte offset, tag 11, whose length is computed
 * directly rather than through the table (snappy.cc:1650-1652). */
Bytes Copy4(size_t length, uint32_t offset) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((length - 1) << 2) | 3));
    for (int i = 0; i < 4; i++) {
        out.push_back(static_cast<unsigned char>((offset >> (8 * i)) & 0xff));
    }
    return out;
}

/* The preamble: the uncompressed length as a varint32. */
Bytes Preamble(uint32_t length) {
    Bytes out;
    uint32_t v = length;
    while (v >= 0x80) {
        out.push_back(static_cast<unsigned char>((v & 0x7f) | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<unsigned char>(v));
    return out;
}

Bytes Ascii(const std::string& text) {
    return Bytes(text.begin(), text.end());
}

bool DecodesTo(const Bytes& stream, const std::string& expected) {
    Bytes decoded;
    if (!SnappyOracleDecodes(stream, &decoded)) {
        std::fprintf(stderr, "oracle rejected a stream expected to decode\n");
        return false;
    }
    if (decoded.size() != expected.size()) {
        std::fprintf(stderr, "decoded %zu bytes, expected %zu\n",
                     decoded.size(), expected.size());
        return false;
    }
    return equal_bytes(decoded.data(), expected.data(), decoded.size());
}

}  // namespace

int main() {
    /* Probe 1: a non-minimal varint preamble is accepted.
     * Varint::Parse32WithLimit (snappy-stubs-internal.h:455-475) accumulates
     * seven bits per byte and stops at the first byte without the
     * continuation bit. Nothing there rejects an encoding that could have
     * been shorter, so 4 written as two bytes must decode like 4 written as
     * one. Both spellings of the same stream, so the only difference between
     * the verdicts can be the preamble. */
    const Bytes payload = Ascii("abcd");
    Bytes minimal;
    Append(&minimal, Preamble(4));
    Append(&minimal, Literal(payload));
    REQUIRE(minimal.size() == 6);
    REQUIRE(DecodesTo(minimal, "abcd"));

    Bytes non_minimal;
    non_minimal.push_back(0x84); /* 4, with the continuation bit set */
    non_minimal.push_back(0x00); /* and a second group contributing zero */
    Append(&non_minimal, Literal(payload));
    REQUIRE(non_minimal.size() == 7);
    REQUIRE(SnappyOracleAccepts(non_minimal));
    REQUIRE(DecodesTo(non_minimal, "abcd"));

    /* Probe 2: a trailing byte after an otherwise complete stream is
     * rejected. InternalUncompressAllTags returns
     * `decompressor->eof() && writer->CheckLength()` (snappy.cc:1789): the
     * output length can be exactly right and the verdict still false because
     * input remains. This is the half a decoder that only counts output
     * bytes would lose. */
    Bytes trailing = minimal;
    trailing.push_back(0x00);
    REQUIRE(!SnappyOracleAccepts(trailing));
    Bytes trailing_out;
    REQUIRE(!SnappyOracleDecodes(trailing, &trailing_out));
    REQUIRE(trailing_out.empty());

    /* Probe 3: a truncated stream is rejected - the CheckLength half of the
     * same line (snappy.cc:1789, SnappyArrayWriter::CheckLength at
     * snappy.cc:2167 for the RawUncompress writer). Every prefix of a valid
     * stream, so the ladder is walked rather than sampled at one point. */
    for (size_t keep = 1; keep < minimal.size(); keep++) {
        const Bytes prefix(minimal.begin(), minimal.begin() + keep);
        Bytes prefix_out;
        REQUIRE_CTX(!SnappyOracleAccepts(prefix), "prefix of %zu bytes", keep);
        REQUIRE_CTX(!SnappyOracleDecodes(prefix, &prefix_out),
                    "prefix of %zu bytes", keep);
    }

    /* Probe 4: tag-11 copies, the 4-byte-offset form, are accepted. This one
     * is decode-side only: snappy's own compressor cannot emit it at all -
     * EmitCopyAtMost64 asserts `offset < 65536` and writes the 1-byte or
     * 2-byte form (snappy.cc:667-693), and EmitCopy has no other path
     * (snappy.cc:696-722). So no compressor output in existence exercises
     * this branch, while snappy.cc:1650-1652 decodes it without ever asking
     * which form would have been cheaper - which is exactly the asymmetry a
     * hand-built stream lives in. */
    Bytes tag11;
    Append(&tag11, Preamble(8));
    Append(&tag11, Literal(payload));
    Append(&tag11, Copy4(4, 4));
    REQUIRE(SnappyOracleAccepts(tag11));
    REQUIRE(DecodesTo(tag11, "abcdabcd"));

    /* Probe 5: copy lengths 1 to 3 are accepted. The 1-byte-offset form
     * cannot encode them at all - it carries len-4 in three bits
     * (snappy-internal.h:367), and the compressor asserts `len >= 4` before
     * emitting any copy (snappy.cc:669). Their acceptance rests on the table
     * form, whose length is entry & 0xff (snappy.cc:1660). Both forms that
     * can express them are probed, since they decode through different
     * code. */
    for (size_t length = 1; length <= 3; length++) {
        const std::string expected = "abcd" + std::string(length, 'd');
        Bytes copy2;
        Append(&copy2, Preamble(static_cast<uint32_t>(4 + length)));
        Append(&copy2, Literal(payload));
        Append(&copy2, Copy2(length, 1));
        REQUIRE_CTX(SnappyOracleAccepts(copy2), "copy length %zu, tag 10",
                    length);
        REQUIRE_CTX(DecodesTo(copy2, expected), "copy length %zu, tag 10",
                    length);

        Bytes copy4;
        Append(&copy4, Preamble(static_cast<uint32_t>(4 + length)));
        Append(&copy4, Literal(payload));
        Append(&copy4, Copy4(length, 1));
        REQUIRE_CTX(SnappyOracleAccepts(copy4), "copy length %zu, tag 11",
                    length);
        REQUIRE_CTX(DecodesTo(copy4, expected), "copy length %zu, tag 11",
                    length);
    }

    std::printf("PASS: 5 snappy 1.2.2 decode behaviours confirmed by "
                "execution\n");
    return 0;
}
