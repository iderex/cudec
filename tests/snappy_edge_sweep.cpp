/* The two branch classes a mutation corpus cannot structurally build, swept
 * against the twin and the oracle (issue #172).
 *
 * Mutating a real stream damages bytes; it does not synthesise a shape the
 * compressor never emits. Both LZ4 fail-opens ever found here needed exactly
 * such a shape, and for Snappy the two classes out of reach of the corpus are
 * the literal length's extra bytes and the varint preamble's edge encodings.
 * Every stream below is hand-built, and the verdict it is held to is the
 * reference's own rather than this file's opinion: whenever snappy rejects,
 * the twin must reject; where both accept, the bytes must match.
 *
 * The neighbouring test carries the per-rung negatives and the pinned 32-bit
 * literal-length wrap, one stream per ladder branch. This one is breadth over
 * those same branches: the width-class boundaries where an off-by-one in the
 * header arithmetic lives, and the preamble encodings a compressor has no
 * reason to write.
 *
 * Line citations are into the pinned tree, google/snappy 1.2.2, as fetched by
 * the URL_HASH in tests/CMakeLists.txt. They move only when that pin moves. */
#include "cudec.h"
#include "fixtures.h"
#include "require.h"
#include "snappy_block.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

size_t g_accept_side_divergences = 0;
size_t g_reject_side_divergences = 0;
size_t g_bounds_violations = 0;
size_t g_cases = 0;

void Append(Bytes* out, const Bytes& more) {
    out->insert(out->end(), more.begin(), more.end());
}

/* The preamble, minimally encoded: seven bits per byte, low group first,
 * continuation bit on every byte but the last
 * (snappy-stubs-internal.h:477-490, Varint::Encode32). */
Bytes VarintMinimal(uint32_t value) {
    Bytes out;
    uint32_t rest = value;
    while (rest >= 0x80) {
        out.push_back(static_cast<unsigned char>((rest & 0x7f) | 0x80));
        rest >>= 7;
    }
    out.push_back(static_cast<unsigned char>(rest));
    return out;
}

/* The same value in EXACTLY `groups` bytes, padding with continuation bytes
 * that carry no payload. Nothing in the reference's preamble readers refuses
 * an encoding that could have been shorter: Varint::Parse32WithLimit
 * (snappy-stubs-internal.h:455-474) stops at the first byte below 128 and
 * asks nothing about minimality, and ReadUncompressedLength (snappy.cc:1533)
 * only refuses a shift of 32 or more and a group whose shifted value would
 * overflow. So these must be ACCEPTED, which is the direction that matters:
 * over-strictness here would reject streams the reference decodes. */
Bytes VarintPadded(uint32_t value, unsigned groups) {
    Bytes out;
    uint32_t rest = value;
    for (unsigned i = 0; i + 1 < groups; i++) {
        out.push_back(static_cast<unsigned char>((rest & 0x7f) | 0x80));
        rest >>= 7;
    }
    out.push_back(static_cast<unsigned char>(rest & 0x7f));
    return out;
}

/* A literal in the short form: the tag's upper six bits hold length-1, which
 * reaches 60 bytes before the length needs bytes of its own
 * (snappy.cc:1613-1614, snappy.cc:1624). */
Bytes Literal(const Bytes& data) {
    Bytes out;
    out.push_back(static_cast<unsigned char>((data.size() - 1) << 2));
    Append(&out, data);
    return out;
}

/* A literal whose length-1 travels in `extra` little-endian bytes, selected
 * by tag values 60 to 63 (snappy.cc:1624-1626: the tag value above 59 is the
 * count of length bytes, and the length is read little-endian from them). */
Bytes LiteralHeader(uint32_t length_minus_one, unsigned extra) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((59 + extra) << 2)));
    for (unsigned i = 0; i < extra; i++) {
        out.push_back(static_cast<unsigned char>((length_minus_one >> (8 * i)) &
                                                 0xff));
    }
    return out;
}

Bytes LongLiteral(const Bytes& data, unsigned extra) {
    Bytes out = LiteralHeader(static_cast<uint32_t>(data.size() - 1), extra);
    Append(&out, data);
    return out;
}

Bytes Filler(size_t length) {
    Bytes out;
    out.reserve(length);
    for (size_t i = 0; i < length; i++) {
        out.push_back(static_cast<unsigned char>('a' + (i % 26)));
    }
    return out;
}

/* Sequential execution of the parsed elements, from an EXACTLY-sized copy of
 * the stream so an over-read past src_size lands in a redzone rather than in
 * a vector's rounded-up slack. A copy is the chasing byte copy, which is the
 * pattern-repeating semantics an overlapping Snappy copy has.
 *
 * The bounds counter is not decoration: a verdict alone cannot prove a bounds
 * guard, because a cursor that ran past the end is usually caught by a later
 * branch AFTER the read already happened. What the guard prevents is that
 * read, so that is what is counted. */
cudec_status TwinDecode(const Bytes& stream, Bytes* out) {
    out->clear();
    const size_t stream_size = stream.size();
    auto tight = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(tight.get(), stream.data(), stream_size);
    }
    cudec_detail::SnappyParser parser{tight.get(), stream_size,
                                      kSnappyOracleMaxOutput};
    const cudec_status header = parser.Begin();
    if (header != CUDEC_OK) {
        return header;
    }
    out->assign(static_cast<size_t>(parser.declared), 0);
    cudec_detail::DecodeSequence element;
    bool done = false;
    uint64_t fuel = static_cast<uint64_t>(stream_size) + 2;
    while (true) {
        if (fuel-- == 0) {
            std::fprintf(stderr, "the parser did not terminate on a %zu-byte "
                                 "stream\n",
                         stream_size);
            g_bounds_violations++;
            out->clear();
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        const cudec_status status = parser.Next(&element, &done);
        if (status != CUDEC_OK) {
            out->clear();
            return status;
        }
        const bool bounded =
            parser.src_pos <= parser.src_size &&
            parser.dst_pos <= parser.declared &&
            element.literals_dst + element.literals_len <= parser.dst_pos &&
            element.literals_src + element.literals_len <= parser.src_pos &&
            element.match_dst + element.match_len <= parser.dst_pos &&
            (element.match_len == 0 || element.match_src < element.match_dst);
        if (!bounded) {
            std::fprintf(stderr, "the parser left its bounds on a %zu-byte "
                                 "stream\n",
                         stream_size);
            g_bounds_violations++;
            out->clear();
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        for (uint64_t i = 0; i < element.literals_len; i++) {
            (*out)[static_cast<size_t>(element.literals_dst + i)] =
                tight[static_cast<size_t>(element.literals_src + i)];
        }
        for (uint64_t i = 0; i < element.match_len; i++) {
            (*out)[static_cast<size_t>(element.match_dst + i)] =
                (*out)[static_cast<size_t>(element.match_src + i)];
        }
        if (done) {
            break;
        }
    }
    out->resize(static_cast<size_t>(parser.dst_pos));
    return CUDEC_OK;
}

/* One stream, both directions. The oracle decides: an accept the twin refuses
 * is a divergence, a reject the twin accepts is a fail-open, and two accepts
 * that produce different bytes are neither but are just as wrong.
 *
 * Two reference calls, each in the direction where it is authoritative. The
 * decoding one carries the harness's own 256 MiB bound, so it is the one the
 * verdicts are compared through; the no-write validator carries no bound at
 * all, so it is asked only about streams the twin ACCEPTED, where the
 * declaration is inside the bound by construction. A stream cudec accepts
 * that snappy's own validator calls invalid is a fail-open however the
 * decoding wrapper answered. */
void Parity(const std::string& name, const Bytes& stream) {
    g_cases++;
    Bytes oracle_out;
    const bool oracle_ok = SnappyOracleDecodes(stream, &oracle_out);
    Bytes twin_out;
    const cudec_status twin = TwinDecode(stream, &twin_out);

    if (twin == CUDEC_OK && !SnappyOracleAccepts(stream)) {
        std::fprintf(stderr,
                     "%s: the twin accepts a stream IsValidCompressedBuffer "
                     "refuses - a fail-open\n",
                     name.c_str());
        g_reject_side_divergences++;
        return;
    }

    if (oracle_ok && twin != CUDEC_OK) {
        std::fprintf(stderr,
                     "%s: snappy accepts, the twin refuses with status %d\n",
                     name.c_str(), static_cast<int>(twin));
        g_accept_side_divergences++;
        return;
    }
    if (!oracle_ok && twin == CUDEC_OK) {
        std::fprintf(stderr, "%s: snappy refuses, the twin accepts - a "
                             "fail-open\n",
                     name.c_str());
        g_reject_side_divergences++;
        return;
    }
    if (oracle_ok && twin_out != oracle_out) {
        std::fprintf(stderr,
                     "%s: both accept and the bytes differ (%zu vs %zu)\n",
                     name.c_str(), twin_out.size(), oracle_out.size());
        g_accept_side_divergences++;
    }
}

/* The literal-length extra bytes. The tag value selects HOW MANY bytes carry
 * length-1 rather than carrying the length itself, so every transition
 * between two width classes is an off-by-one waiting in the header
 * arithmetic: 60 bytes is the last length the tag carries alone, 61 the first
 * that needs a byte, and 2^8 and 2^16 are where the byte count steps again.
 *
 * Both sides of each transition, and each length in every width class that
 * can express it - a length of 61 is legal in the 1, 2, 3 and 4-byte classes
 * alike, and only the shortest is what a compressor writes. */
void SweepLiteralLengths() {
    const size_t transitions[] = {59, 60, 61, 254, 255, 256, 257, 65535, 65536,
                                  65537};
    for (size_t length : transitions) {
        const Bytes payload = Filler(length);
        if (length <= 60) {
            Bytes s = VarintMinimal(static_cast<uint32_t>(length));
            Append(&s, Literal(payload));
            Parity("literal-inline-" + std::to_string(length), s);
        }
        for (unsigned extra = 1; extra <= 4; extra++) {
            /* A class can only be used where length-1 fits its bytes. */
            const uint64_t bound = static_cast<uint64_t>(1) << (8 * extra);
            if (length - 1 >= bound) {
                continue;
            }
            Bytes s = VarintMinimal(static_cast<uint32_t>(length));
            Append(&s, LongLiteral(payload, extra));
            Parity("literal-" + std::to_string(length) + "-in-" +
                       std::to_string(extra) + "-byte-class",
                   s);
        }
    }
}

/* The length bytes cut off inside the field, one truncation per width class,
 * at every count of present bytes. The header runs past the end of the source
 * here, which is the read the header-truncation rung exists to prevent, so
 * this is also where a missing bound shows up as a bounds violation rather
 * than as a verdict. */
void SweepTruncatedLengthFields() {
    for (unsigned extra = 2; extra <= 4; extra++) {
        for (unsigned present = 1; present < extra; present++) {
            Bytes s = VarintMinimal(64);
            Bytes header = LiteralHeader(63, extra);
            header.resize(1 + present);
            Append(&s, header);
            Parity("literal-header-" + std::to_string(extra) + "-byte-class-" +
                       std::to_string(present) + "-present",
                   s);
        }
    }
    /* The tag byte alone, with the whole length field missing. */
    for (unsigned extra = 1; extra <= 4; extra++) {
        Bytes s = VarintMinimal(64);
        s.push_back(static_cast<unsigned char>((59 + extra) << 2));
        Parity("literal-header-" + std::to_string(extra) +
                   "-byte-class-length-absent",
               s);
    }
}

/* A declared literal length that fits one bound and not the other, in both
 * orders. These are the two rungs a literal can fail at after its header is
 * read, and the pair matters because a decoder checking only one of them
 * fails open on the other: a literal inside the source but past the declared
 * output overruns the destination, and one inside the declaration but past
 * the source over-reads the input. */
void SweepLiteralAgainstBothBounds() {
    /* Fits the remaining source, past the remaining declared output: 16
     * literal bytes present, 8 declared. */
    {
        Bytes s = VarintMinimal(8);
        Append(&s, LongLiteral(Filler(16), 1));
        Parity("literal-inside-source-past-declaration", s);
    }
    /* The converse: 64 declared, a literal claiming 64 bytes, 16 present. */
    {
        Bytes s = VarintMinimal(64);
        Append(&s, LiteralHeader(63, 1));
        Append(&s, Filler(16));
        Parity("literal-inside-declaration-past-source", s);
    }
    /* Both at once, at the top of the four-byte class: a length no source
     * could carry, which is refused on both sides for the source bound. The
     * neighbouring value, 0xFFFFFFFF, is where the reference's 32-bit
     * accumulator wraps, and that ONE divergence is pinned in
     * tests/snappy_parser_twin.cpp rather than swept here. */
    {
        Bytes s = VarintMinimal(64);
        Append(&s, LiteralHeader(0xFFFFFFFEu, 4));
        Append(&s, Filler(16));
        Parity("literal-length-4294967295-no-source", s);
    }
    /* An exact fit against both bounds is the accept these rejects are
     * measured against: without it, a parser that refused every long-form
     * literal would pass this sweep. */
    {
        Bytes s = VarintMinimal(64);
        Append(&s, LongLiteral(Filler(64), 4));
        Parity("literal-exact-fit-both-bounds", s);
    }
}

/* The preamble. A varint32 of one to five bytes, and the encodings a
 * compressor has no reason to emit are exactly the ones an attacker writes:
 * a value padded out to more groups than it needs, a fifth group carrying
 * bits that would overflow 32, and a sixth group that cannot exist at all. */
void SweepPreambleEncodings() {
    const uint32_t values[] = {0, 1, 4, 127, 128, 16383, 16384};
    for (uint32_t value : values) {
        const Bytes payload = Filler(value);
        /* Minimal, the shape the compressor writes. */
        {
            Bytes s = VarintMinimal(value);
            if (value != 0) {
                Append(&s, LongLiteral(payload, 4));
            }
            Parity("preamble-minimal-" + std::to_string(value), s);
        }
        /* Non-minimal, in every group count that can still hold the value.
         * These must be ACCEPTED: refusing them would reject streams the
         * reference decodes. */
        for (unsigned groups = 2; groups <= 5; groups++) {
            const unsigned needed = static_cast<unsigned>(
                VarintMinimal(value).size());
            if (groups < needed) {
                continue;
            }
            Bytes s = VarintPadded(value, groups);
            if (value != 0) {
                Append(&s, LongLiteral(payload, 4));
            }
            Parity("preamble-" + std::to_string(value) + "-in-" +
                       std::to_string(groups) + "-groups",
                   s);
        }
    }
    /* A fifth group at and above 16. Four bits are all the fifth group can
     * contribute before a 32-bit accumulator overflows, which is why the
     * reference refuses a fifth byte of 16 or more outright
     * (snappy-stubs-internal.h:469, `if (b < 16) goto done;`) rather than
     * masking the excess away.
     *
     * All three of these come back refused, and only the top two for that
     * reason: a fifth group of 15 declares just under 4 GiB, which the
     * harness wrapper refuses at its own 256 MiB bound and the twin refuses
     * against the capacity it was given. What the case pins is that the two
     * sides agree on the whole neighbourhood of the boundary, not which rung
     * caught it. */
    for (unsigned top = 15; top <= 17; top++) {
        Bytes s = {0x80, 0x80, 0x80, 0x80, static_cast<unsigned char>(top)};
        Append(&s, Literal(Filler(4)));
        Parity("preamble-fifth-group-" + std::to_string(top), s);
    }
    /* A sixth group. The fifth byte carries its continuation bit, so the
     * value never terminates inside the five bytes a varint32 has
     * (snappy.cc:1538, `if (shift >= 32) return false`). */
    {
        Bytes s = {0x80, 0x80, 0x80, 0x80, 0x80, 0x00};
        Append(&s, Literal(Filler(4)));
        Parity("preamble-sixth-group", s);
    }
    /* The preamble cut off inside itself, at every length: every byte
     * continues and the stream ends. */
    for (unsigned present = 1; present <= 4; present++) {
        Bytes s(present, 0x80);
        Parity("preamble-truncated-" + std::to_string(present) + "-bytes", s);
    }
    /* No preamble at all. */
    Parity("preamble-empty-stream", Bytes());
}

/* The capacity axis, which the format cannot express and the oracle therefore
 * has no verdict on: the destination the CALLER owns, against the length the
 * STREAM declares. Driven at the parser directly, and the status is required
 * rather than merely non-OK, because the difference between "this stream is
 * corrupt" and "your buffer is too small" is what a caller acts on. */
void SweepDeclaredAgainstCapacity() {
    struct Case {
        uint32_t declared;
        uint64_t capacity;
        cudec_status expected;
    };
    const Case cases[] = {
        /* Nothing declared fits any capacity, including none. */
        {0, 0, CUDEC_OK},
        {0, 1, CUDEC_OK},
        {1, 0, CUDEC_ERR_OUTPUT_TOO_SMALL},
        {1, 1, CUDEC_OK},
        {2, 1, CUDEC_ERR_OUTPUT_TOO_SMALL},
        /* The largest a varint32 can declare, against a capacity no machine
         * here would hand it and against one it dwarfs. Nothing is allocated
         * from the declaration on either path, which is the property. */
        {0xFFFFFFFFu, 0xFFFFFFFFull, CUDEC_OK},
        {0xFFFFFFFFu, 0xFFFFFFFEull, CUDEC_ERR_OUTPUT_TOO_SMALL},
        {0xFFFFFFFFu, 1024, CUDEC_ERR_OUTPUT_TOO_SMALL},
    };
    for (const Case& c : cases) {
        const Bytes preamble = VarintMinimal(c.declared);
        cudec_detail::SnappyParser parser{preamble.data(), preamble.size(),
                                          c.capacity};
        const cudec_status status = parser.Begin();
        if (status != c.expected) {
            std::fprintf(stderr,
                         "declared %u against capacity %llu: expected status "
                         "%d, got %d\n",
                         c.declared,
                         static_cast<unsigned long long>(c.capacity),
                         static_cast<int>(c.expected),
                         static_cast<int>(status));
            g_reject_side_divergences++;
        }
        if (status == CUDEC_OK && parser.declared != c.declared) {
            std::fprintf(stderr,
                         "declared %u read back as %llu\n", c.declared,
                         static_cast<unsigned long long>(parser.declared));
            g_reject_side_divergences++;
        }
        g_cases++;
    }
}

}  // namespace

int main() {
    SweepLiteralLengths();
    SweepTruncatedLengthFields();
    SweepLiteralAgainstBothBounds();
    SweepPreambleEncodings();
    SweepDeclaredAgainstCapacity();

    std::printf("snappy edge sweep: %zu streams, %zu accept-side divergences, "
                "%zu reject-side divergences, %zu bounds violations\n",
                g_cases, g_accept_side_divergences, g_reject_side_divergences,
                g_bounds_violations);
    REQUIRE(g_accept_side_divergences == 0);
    REQUIRE(g_reject_side_divergences == 0);
    REQUIRE(g_bounds_violations == 0);
    /* A sweep that swept nothing prints the same line as one that swept
     * everything and found nothing, so the count is asserted too. */
    REQUIRE(g_cases >= 60);
    std::printf("PASS\n");
    return 0;
}
