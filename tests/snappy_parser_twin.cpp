/* The CPU twin of the Snappy decoder core, the sibling of
 * tests/parser_twin.cpp: the single-source parser and validation ladder
 * from src/snappy_block.h, executed sequentially on the host and held to
 * oracle parity on the GPU-less CI runner - the fail-closed heart of M3
 * lands under CI before any kernel exists. Parity is the authority:
 * wherever snappy rejects, the twin must reject; where both accept, the
 * bytes and the size must match the reference's own output.
 *
 * The stream carries its own uncompressed length, so a verdict needs no
 * size argument and the twin sizes its destination the way
 * SnappyOracleDecodes does - from the declared length, refused above the
 * harness bound in fixtures.h. That keeps the two sides comparable on the
 * mutants that declare an absurd length. The capacity axis the format
 * cannot express, a destination smaller than the declaration, is driven
 * against the parser directly further down. */
#include "cudec.h"
#include "fixtures.h"
#include "require.h"
#include "snappy_block.h"
#include "snappy_twin.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* The driver's own two counters, in the shared header that owns the driver
 * (tests/snappy_twin.h). main() requires the bounds count to be zero; the
 * status the driver returns is left alone, so a violation cannot be swallowed
 * by a parity comparison that only reads verdicts. The references keep this
 * file's call sites reading as they did before the driver moved out.
 *
 * Why a count rather than a status: a verdict alone cannot prove a bounds
 * guard. Removing the check that a long-form literal's length bytes are
 * present, or that a copy's offset bytes are, still rejects every crafted
 * stream - the cursor runs past the source end, the next subtraction wraps,
 * and a later branch catches it after the read has already happened. What the
 * guard actually prevents is that read, so that is what is counted. Measured,
 * three of those removals left a verdict-only suite green. */
cudec_test::SnappyTwinObserver g_twin;
size_t& g_bounds_violations = g_twin.bounds_violations;
cudec_detail::SnappyReject& g_last_reject = g_twin.last_reject;

void ObserveReject(cudec_detail::SnappyReject branch) {
    g_last_reject = branch;
}

/* Which ladder rungs a DECLARED NEGATIVE reached (issue #173). The
 * enumeration lives once, in src/snappy_block.h, and main() requires this set
 * to be the whole of it apart from the one rung declared unreachable there.
 * So a rung added to the ladder with no negative behind it reds this test,
 * and deleting the negative that reached a rung reds it too.
 *
 * ONLY the declared negative sets mark coverage - never the fixtures, never
 * the mutation corpus. That is the difference between this check biting and
 * merely looking as if it does, and it is MEASURED: with the corpus counted,
 * deleting the copy-over-production negative left this test green, because a
 * mutant that happened to trip the same rung stood in for it. A rung is
 * tested when a stream written to reach it reaches it, not when some stream
 * does. */
bool g_reject_covered[cudec_detail::kSnappyRejectCount] = {false};

void CoverBranch(cudec_detail::SnappyReject branch) {
    if (branch != cudec_detail::kSnappyRejectNone) {
        g_reject_covered[branch] = true;
    }
}

/* The two vector-shaped entries this file uses, over the shared driver in
 * tests/snappy_twin.h - one copy with the fuzz target (issue #90). The
 * capacity bound handed to the whole-stream entry is the harness bound rather
 * than the declaration itself: that is the same refusal SnappyOracleDecodes
 * applies, so a mutant declaring 4 GiB is refused on both sides for the same
 * reason instead of reading as a parity failure. */
cudec_status Drive(const unsigned char* src, uint64_t src_size,
                   uint64_t capacity, Bytes* out) {
    return cudec_test::SnappyTwinDrive(src, src_size, capacity, out, &g_twin);
}

cudec_status TwinDecode(const Bytes& stream, Bytes* out) {
    return cudec_test::SnappyTwinDecode(stream.data(), stream.size(),
                                        kSnappyOracleMaxOutput, out, &g_twin);
}

/* Drives the parser for its liveness contract alone, producing nothing.
 * Returns false, with a reason, when a non-terminal call consumed no source
 * byte, when a call after the terminal one succeeded, or when the parser did
 * not terminate inside a bound every live parse fits in. A lane that never
 * leaves its element loop holds a whole launch, so this is a fail-closed
 * property and not a nicety. */
bool LivenessHolds(const Bytes& stream, const char** why) {
    const size_t stream_size = stream.size();
    auto tight = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(tight.get(), stream.data(), stream_size);
    }
    cudec_detail::SnappyParser parser{tight.get(), stream_size,
                                      kSnappyOracleMaxOutput};
    cudec_detail::DecodeSequence element;
    bool done = false;
    uint64_t fuel = static_cast<uint64_t>(stream_size) + 2;
    while (true) {
        if (fuel-- == 0) {
            *why = "the parser did not terminate";
            return false;
        }
        const uint64_t before = parser.src_pos;
        const cudec_status status = parser.Next(&element, &done);
        if (status != CUDEC_OK) {
            ObserveReject(parser.reject);
            return true; /* a reject ends any driver loop */
        }
        if (done) {
            /* The terminal call consumes nothing, so the parser itself has
             * to stop a driver that ignores *done. */
            bool again = false;
            if (parser.Next(&element, &again) == CUDEC_OK) {
                *why = "a call after the terminal element succeeded";
                return false;
            }
            ObserveReject(parser.reject);
            return true;
        }
        if (parser.src_pos <= before) {
            *why = "a non-terminal call consumed no source byte";
            return false;
        }
    }
}

/* Stream builders. tests/snappy_probes.cpp carries its own copies of five
 * of these, where they exist to ask the reference what it accepts and their
 * comments cite the reference lines each encoding was read from. These are
 * duplicates of that arithmetic, kept local because the citations belong
 * with the probes; a shared builder header is the right home and is not
 * this change. */

void Append(Bytes* out, const Bytes& more) {
    out->insert(out->end(), more.begin(), more.end());
}

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

/* A short-form literal: the tag's upper six bits hold length-1, which
 * reaches 60 bytes before the length needs bytes of its own. */
Bytes Literal(const Bytes& data) {
    Bytes out;
    out.push_back(static_cast<unsigned char>((data.size() - 1) << 2));
    Append(&out, data);
    return out;
}

/* A long-form literal: `extra` little-endian bytes carry length-1, and the
 * tag's six bits select how many. */
Bytes LongLiteral(const Bytes& data, unsigned extra) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((59 + extra) << 2)));
    const uint64_t encoded = data.size() - 1;
    for (unsigned i = 0; i < extra; i++) {
        out.push_back(static_cast<unsigned char>((encoded >> (8 * i)) & 0xff));
    }
    Append(&out, data);
    return out;
}

Bytes Copy1(unsigned length, unsigned offset) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((offset >> 8) << 5) |
                                             ((length - 4) << 2) | 1));
    out.push_back(static_cast<unsigned char>(offset & 0xff));
    return out;
}

Bytes Copy2(unsigned length, unsigned offset) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((length - 1) << 2) | 2));
    out.push_back(static_cast<unsigned char>(offset & 0xff));
    out.push_back(static_cast<unsigned char>((offset >> 8) & 0xff));
    return out;
}

Bytes Copy4(unsigned length, uint32_t offset) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((length - 1) << 2) | 3));
    for (int i = 0; i < 4; i++) {
        out.push_back(static_cast<unsigned char>((offset >> (8 * i)) & 0xff));
    }
    return out;
}

Bytes Ascii(const std::string& text) {
    return Bytes(text.begin(), text.end());
}

struct CraftedNegative {
    const char* name;
    Bytes stream;
    cudec_status expected;
    /* The ladder branch this stream must reach, named from the enumeration
     * in src/snappy_block.h. Pinned per entry rather than only in aggregate:
     * a negative that keeps rejecting for a DIFFERENT reason than the one it
     * was written for still leaves its own branch unproven, and the coverage
     * count alone would not notice as long as some other entry happened to
     * reach it. */
    cudec_detail::SnappyReject branch;
};

/* One crafted stream per validation-ladder branch. Each pins the twin's
 * specific status AND asserts snappy rejects the same bytes, so the branch
 * is demonstrably in parity rather than merely present. Byte layouts
 * verified by trace against src/snappy_block.h. */
std::vector<CraftedNegative> MakeCraftedNegatives() {
    const Bytes abcd = Ascii("abcd");
    std::vector<CraftedNegative> out;

    /* Preamble: no byte at all. */
    out.push_back({"preamble-absent",
                   {},
                   CUDEC_ERR_CORRUPT_INPUT,
                   cudec_detail::kSnappyRejectPreambleByteMissing});
    /* Preamble: a continuation bit with nothing behind it. */
    out.push_back({"preamble-truncated",
                   {0x80},
                   CUDEC_ERR_CORRUPT_INPUT,
                   cudec_detail::kSnappyRejectPreambleByteMissing});
    /* Preamble: a sixth group. The fifth byte still carries the
     * continuation bit, which the reference refuses outright. */
    out.push_back({"preamble-six-groups",
                   {0x80, 0x80, 0x80, 0x80, 0x80, 0x00},
                   CUDEC_ERR_CORRUPT_INPUT,
                   cudec_detail::kSnappyRejectPreambleFinalGroup});
    /* Preamble: five groups whose value needs a 33rd bit. */
    out.push_back({"preamble-overflows-32-bit",
                   {0x80, 0x80, 0x80, 0x80, 0x10},
                   CUDEC_ERR_CORRUPT_INPUT,
                   cudec_detail::kSnappyRejectPreambleFinalGroup});

    /* Elements: a declaration with no elements behind it. */
    {
        Bytes s = Preamble(4);
        out.push_back({"under-production-empty", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectUnderProduction});
    }
    /* Elements: production stops short of the declaration. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        out.push_back({"under-production-short", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectUnderProduction});
    }
    /* A long-form literal whose length bytes run past the source end. The
     * declaration is deliberately larger than any length those missing
     * bytes could have spelled - 256 for the one-byte class, 65536 for the
     * two-byte one - so that removing the guard does not merely trade one
     * reject for another: whatever lies past the buffer, the element is
     * accepted and the cursor leaves the stream, which is what the guard
     * exists to prevent and what Drive's bounds check then reports. */
    {
        Bytes s = Preamble(300);
        s.push_back(0xF0); /* class 60: one length byte follows, and none does */
        out.push_back({"literal-header-truncated", s,
                       CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectLiteralHeaderTruncated});
    }
    {
        Bytes s = Preamble(70000);
        s.push_back(0xF4); /* class 61: two length bytes follow, and none do */
        out.push_back({"literal-header-truncated-wide", s,
                       CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectLiteralHeaderTruncated});
    }
    /* A literal whose payload runs past the source end. */
    {
        Bytes s = Preamble(4);
        Append(&s, Literal(abcd));
        s.resize(s.size() - 2);
        out.push_back({"literal-payload-truncated", s,
                       CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectLiteralPayloadTruncated});
    }
    /* A literal longer than the declaration leaves room for. */
    {
        Bytes s = Preamble(2);
        Append(&s, Literal(abcd));
        out.push_back({"literal-over-production", s,
                       CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectLiteralOverProduction});
    }
    /* A copy whose offset bytes run past the source end. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        s.push_back(0x0E); /* a two-byte-offset copy tag, with no offset */
        out.push_back({"copy-header-truncated", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyHeaderTruncated});
    }
    /* The same for the narrow form, whose header is one byte shorter and
     * whose guard is therefore a separate branch. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        s.push_back(0x01); /* a one-byte-offset copy tag, with no offset */
        out.push_back({"copy1-header-truncated", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyHeaderTruncated});
    }
    /* And for the four-byte-offset form, which is missing three of them. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy4(4, 4));
        s.resize(s.size() - 3);
        out.push_back({"copy4-header-truncated", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyHeaderTruncated});
    }
    /* A copy with no byte to replicate, in each of the three forms, both
     * with output behind it and with none. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy2(4, 0));
        out.push_back({"copy-offset-zero-tag-10", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyOffsetZero});
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy4(4, 0));
        out.push_back({"copy-offset-zero-tag-11", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyOffsetZero});
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy1(4, 0));
        out.push_back({"copy-offset-zero-tag-01", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyOffsetZero});
    }
    {
        Bytes s = Preamble(4);
        Append(&s, Copy2(4, 0));
        out.push_back({"copy-offset-zero-at-start", s,
                       CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyOffsetZero});
    }
    /* A copy reaching before the start of the produced output. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy2(4, 5));
        out.push_back({"copy-offset-before-start", s,
                       CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyOffsetBeforeStart});
    }
    /* The same, through the four-byte-offset form, where the offset is
     * wide enough to be a pointer-sized mistake rather than a small one. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy4(4, 0xFFFFFFFFu));
        out.push_back({"copy-offset-huge", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyOffsetBeforeStart});
    }
    /* A copy longer than the declaration leaves room for. */
    {
        Bytes s = Preamble(6);
        Append(&s, Literal(abcd));
        Append(&s, Copy2(4, 1));
        out.push_back({"copy-over-production", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectCopyOverProduction});
    }
    /* A complete stream with one more element behind it: the declared
     * length is already produced, so the trailing element over-produces. */
    {
        Bytes s = Preamble(4);
        Append(&s, Literal(abcd));
        Append(&s, Literal(Ascii("x")));
        out.push_back({"trailing-element", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectLiteralOverProduction});
    }
    /* A complete stream with one dangling tag byte: the source is not
     * consumed exactly, and the tag's own payload is missing too. */
    {
        Bytes s = Preamble(4);
        Append(&s, Literal(abcd));
        s.push_back(0x00);
        out.push_back({"trailing-tag", s, CUDEC_ERR_CORRUPT_INPUT,
                       cudec_detail::kSnappyRejectLiteralPayloadTruncated});
    }
    return out;
}

/* The rung no malformed stream can reach, and the reason it needs its own set
 * rather than an entry above: it is not a statement about the bytes. The
 * capacity rung refuses a stream snappy ACCEPTS, because the destination the
 * caller owns is too small for the length the stream declares - so the loop
 * above, which requires the reference to reject the same bytes, cannot hold
 * one. */
struct CapacityNegative {
    const char* name;
    Bytes stream;
    uint64_t capacity;
    cudec_detail::SnappyReject branch;
};

std::vector<CapacityNegative> MakeCapacityNegatives() {
    const Bytes abcd = Ascii("abcd");
    std::vector<CapacityNegative> out;
    Bytes s = Preamble(4);
    Append(&s, Literal(abcd));
    out.push_back({"capacity-one-short", s, 3,
                   cudec_detail::kSnappyRejectDeclaredOverCapacity});
    out.push_back({"capacity-zero", s, 0,
                   cudec_detail::kSnappyRejectDeclaredOverCapacity});
    return out;
}

/* Streams the reference accepts, which the ladder must not over-reject.
 * Over-strictness is a parity failure and not extra safety: every entry
 * here is a shape snappy's own compressor never emits, so nothing in the
 * fixture corpus covers them. */
struct AcceptCase {
    std::string name;
    Bytes stream;
    std::string expected;
};

std::vector<AcceptCase> MakeAcceptCases() {
    const Bytes abcd = Ascii("abcd");
    std::vector<AcceptCase> out;

    /* A declaration of nothing, which is a whole valid stream. */
    out.push_back({"empty-output", Preamble(0), ""});

    /* A non-minimal preamble. Nothing in Varint::Parse32WithLimit rejects
     * an encoding that could have been shorter (snappy_probes.cpp probe 1). */
    {
        Bytes s = {0x84, 0x00};
        Append(&s, Literal(abcd));
        out.push_back({"preamble-non-minimal", s, "abcd"});
    }
    /* The longest legal preamble for a small number: five groups. */
    {
        Bytes s = {0x84, 0x80, 0x80, 0x80, 0x00};
        Append(&s, Literal(abcd));
        out.push_back({"preamble-five-groups", s, "abcd"});
    }
    /* The four-byte-offset copy form, which no compressor emits at all
     * (snappy_probes.cpp probe 4). */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy4(4, 4));
        out.push_back({"copy-tag-11", s, "abcdabcd"});
    }
    /* Copy lengths 1 to 3, which the narrow form cannot express and the
     * compressor never emits (snappy_probes.cpp probe 5). Both wide forms,
     * since they decode through different arithmetic. */
    for (unsigned length = 1; length <= 3; length++) {
        const std::string expected = "abcd" + std::string(length, 'd');
        Bytes wide2 = Preamble(static_cast<uint32_t>(4 + length));
        Append(&wide2, Literal(abcd));
        Append(&wide2, Copy2(length, 1));
        out.push_back({"copy-length-" + std::to_string(length) + "-tag-10",
                       wide2, expected});
        Bytes wide4 = Preamble(static_cast<uint32_t>(4 + length));
        Append(&wide4, Literal(abcd));
        Append(&wide4, Copy4(length, 1));
        out.push_back({"copy-length-" + std::to_string(length) + "-tag-11",
                       wide4, expected});
    }
    /* The narrow form itself, whose offset arrives in two pieces. */
    {
        Bytes s = Preamble(9);
        Append(&s, Literal(Ascii("abcde")));
        Append(&s, Copy1(4, 5));
        out.push_back({"copy-tag-01", s, "abcdeabcd"});
    }
    /* A one-byte overlapping copy repeating its pattern, the semantics an
     * offset smaller than the length has. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(Ascii("ab")));
        Append(&s, Copy2(6, 1));
        out.push_back({"copy-overlapping", s, "abbbbbbb"});
    }
    /* A long-form literal in each of its four width classes. */
    for (unsigned extra = 1; extra <= 4; extra++) {
        const Bytes payload = Ascii(std::string(64, 'z'));
        Bytes s = Preamble(64);
        Append(&s, LongLiteral(payload, extra));
        out.push_back({"literal-long-form-" + std::to_string(extra) + "-bytes",
                       s, std::string(64, 'z')});
    }
    return out;
}

/* An offset at or above 65536 needs that much produced output behind it, so
 * it gets its own builder rather than a line in the table above. snappy's
 * compressor asserts `offset < 65536` before emitting any copy, so this
 * shape exists only in hand-built streams and in nothing the corpus holds. */
Bytes WideOffsetStream(uint32_t offset, unsigned length) {
    const uint32_t produced = offset + length;
    Bytes payload;
    payload.reserve(offset);
    for (uint32_t i = 0; i < offset; i++) {
        payload.push_back(static_cast<unsigned char>('A' + (i % 26)));
    }
    Bytes s = Preamble(produced);
    Append(&s, LongLiteral(payload, 4));
    Append(&s, Copy4(length, offset));
    return s;
}

}  // namespace

int main() {
    const auto fixtures = MakeSnappyFixtures();
    REQUIRE(!fixtures.empty());

    /* Every fixture decodes to its original through the twin. */
    for (const auto& f : fixtures) {
        Bytes out;
        REQUIRE_CTX(TwinDecode(f.compressed, &out) == CUDEC_OK, "fixture %s",
                    f.name.c_str());
        REQUIRE_CTX(out.size() == f.original.size(), "fixture %s",
                    f.name.c_str());
        REQUIRE_CTX(equal_bytes(out.data(), f.original.data(), out.size()),
                    "fixture %s", f.name.c_str());
    }

    /* Full mutant parity, as the two security-critical directions of a
     * fail-closed decoder:
     *   1. where the twin ACCEPTS, snappy must accept and the bytes and
     *      size must match - the twin never invents an acceptance;
     *   2. where snappy REJECTS, the twin must reject - never more lenient.
     * Together these make the twin's accept set a subset of the
     * reference's, with bit-exact output on it. Over this corpus the count
     * of streams the twin refuses while snappy accepts is pinned at zero
     * rather than merely reported. The one class where it is not zero is
     * unreachable by mutation - it needs a length field the generator never
     * writes - and is pinned by name further down. */
    size_t mutant_total = 0;
    size_t rejected_total = 0;
    size_t stricter_than_oracle = 0;
    for (const auto& f : fixtures) {
        for (const auto& m : MutateSnappyStream(f.compressed, f.seed)) {
            Bytes oracle_out;
            const bool oracle_accepts =
                SnappyOracleDecodes(m.stream, &oracle_out);
            Bytes twin_out;
            const cudec_status twin_status = TwinDecode(m.stream, &twin_out);
            mutant_total++;
            if (twin_status == CUDEC_OK) {
                REQUIRE_CTX(oracle_accepts,
                            "twin accepts what snappy rejects: %s/%s",
                            f.name.c_str(), m.description.c_str());
                REQUIRE_CTX(twin_out.size() == oracle_out.size(),
                            "size parity: %s/%s", f.name.c_str(),
                            m.description.c_str());
                REQUIRE_CTX(equal_bytes(twin_out.data(), oracle_out.data(),
                                        twin_out.size()),
                            "byte parity: %s/%s", f.name.c_str(),
                            m.description.c_str());
            } else {
                REQUIRE_CTX(twin_status == CUDEC_ERR_CORRUPT_INPUT ||
                                twin_status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                            "undefined reject status %d: %s/%s",
                            static_cast<int>(twin_status), f.name.c_str(),
                            m.description.c_str());
                if (oracle_accepts) {
                    stricter_than_oracle++;
                    std::fprintf(stderr,
                                 "stricter than snappy: %s/%s (status %d)\n",
                                 f.name.c_str(), m.description.c_str(),
                                 static_cast<int>(twin_status));
                } else {
                    rejected_total++;
                }
            }
        }
    }
    REQUIRE(mutant_total > 0);
    REQUIRE(rejected_total > 0);
    REQUIRE(stricter_than_oracle == 0);

    /* One crafted negative per ladder branch: pins the twin's specific
     * status AND that the stream is malformed by the reference's reckoning. */
    const auto crafted = MakeCraftedNegatives();
    for (const auto& c : crafted) {
        Bytes twin_out;
        g_last_reject = cudec_detail::kSnappyRejectNone;
        REQUIRE_CTX(TwinDecode(c.stream, &twin_out) == c.expected,
                    "crafted %s", c.name);
        /* The branch, not merely the status: eight of these entries share
         * CUDEC_ERR_CORRUPT_INPUT with some other entry, so a status
         * comparison cannot tell which rung refused. */
        REQUIRE_CTX(g_last_reject == c.branch,
                    "crafted %s: reached ladder rung %d, not %d", c.name,
                    static_cast<int>(g_last_reject),
                    static_cast<int>(c.branch));
        CoverBranch(g_last_reject);
        REQUIRE_CTX(twin_out.empty(),
                    "crafted %s: a rejected stream produced output", c.name);
        Bytes oracle_out;
        REQUIRE_CTX(!SnappyOracleDecodes(c.stream, &oracle_out),
                    "crafted %s: snappy unexpectedly accepts", c.name);
        REQUIRE_CTX(!SnappyOracleAccepts(c.stream),
                    "crafted %s: snappy's validator unexpectedly accepts",
                    c.name);
    }

    /* The capacity rung, on streams the reference accepts. Its ACCEPTANCE is
     * asserted here where the crafted loop asserts rejection: a capacity
     * negative that had drifted into being malformed would otherwise pass for
     * the wrong reason and leave the rung untested while looking covered. */
    const auto capacity_negatives = MakeCapacityNegatives();
    for (const auto& c : capacity_negatives) {
        Bytes oracle_out;
        REQUIRE_CTX(SnappyOracleDecodes(c.stream, &oracle_out),
                    "capacity %s: snappy rejects the stream itself", c.name);
        Bytes twin_out;
        g_last_reject = cudec_detail::kSnappyRejectNone;
        REQUIRE_CTX(Drive(c.stream.data(), c.stream.size(), c.capacity,
                          &twin_out) == CUDEC_ERR_OUTPUT_TOO_SMALL,
                    "capacity %s", c.name);
        REQUIRE_CTX(g_last_reject == c.branch,
                    "capacity %s: reached ladder rung %d, not %d", c.name,
                    static_cast<int>(g_last_reject),
                    static_cast<int>(c.branch));
        CoverBranch(g_last_reject);
        REQUIRE_CTX(twin_out.empty(),
                    "capacity %s: a refused decode produced output", c.name);
    }

    /* The past-terminal rung, which no stream reaches either: it takes a
     * DRIVER that ignores *done, and that is the case it exists for - a warp
     * lane looping past the terminal element must be refused rather than left
     * to spin against the launch. LivenessHolds proves the refusal over every
     * stream this test has; this case is what NAMES the rung, so deleting it
     * leaves the rung uncovered. */
    {
        Bytes valid = Preamble(4);
        Append(&valid, Literal(Ascii("abcd")));
        cudec_detail::SnappyParser parser{valid.data(), valid.size(), 4};
        cudec_detail::DecodeSequence element;
        bool done = false;
        uint64_t fuel = valid.size() + 2;
        while (!done) {
            REQUIRE(fuel-- != 0);
            REQUIRE(parser.Next(&element, &done) == CUDEC_OK);
        }
        bool again = false;
        REQUIRE(parser.Next(&element, &again) == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(!again);
        REQUIRE(parser.reject == cudec_detail::kSnappyRejectPastTerminal);
        CoverBranch(parser.reject);
    }

    /* The deliberate divergence, pinned explicitly. The four-byte literal
     * length class spelling 0xFFFFFFFF means length-1, and the reference
     * adds the 1 in 32 bits, so the value wraps to zero and snappy accepts
     * the stream as carrying a zero-length literal. cudec refuses it: a
     * length that exists only because an accumulator wrapped is not a
     * length. The oracle's acceptance is asserted rather than assumed -
     * that it accepts is the whole reason this case is written down - and
     * the twin's refusal must be a DEFINED status. This is the Snappy
     * analogue of LZ4's offset == 0 divergence. */
    {
        const Bytes wrapped = {0x00, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF};
        Bytes oracle_out;
        REQUIRE(SnappyOracleDecodes(wrapped, &oracle_out));
        REQUIRE(oracle_out.empty());
        Bytes twin_out;
        REQUIRE(TwinDecode(wrapped, &twin_out) == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(twin_out.empty());
        /* The neighbouring value needs no wrap: it asks for a literal of
         * 0xFFFFFFFF bytes, which no source here carries, so both sides
         * refuse it on the bytes that are missing. That is what makes the
         * refusal above the wrap and not the four-byte class itself. */
        Bytes near = {0x00, 0xFC, 0xFE, 0xFF, 0xFF, 0xFF};
        Bytes near_out;
        REQUIRE(!SnappyOracleDecodes(near, &near_out));
        REQUIRE(TwinDecode(near, &near_out) == CUDEC_ERR_CORRUPT_INPUT);
        /* The wrap is not confined to an otherwise empty stream: with the
         * zero-length literal followed by a real one, snappy decodes four
         * bytes and cudec still refuses. Both spellings are pinned so a
         * future change cannot silence one and leave the other. */
        const Bytes wrapped_then_literal = {0x04, 0xFC, 0xFF, 0xFF, 0xFF,
                                            0xFF, 0x0C, 'a',  'b',  'c',
                                            'd'};
        Bytes tail_out;
        REQUIRE(SnappyOracleDecodes(wrapped_then_literal, &tail_out));
        REQUIRE(tail_out.size() == 4);
        Bytes tail_twin;
        REQUIRE(TwinDecode(wrapped_then_literal, &tail_twin) ==
                CUDEC_ERR_CORRUPT_INPUT);
    }

    /* The band above the harness bound, disclosed rather than left silent.
     * TwinDecode and SnappyOracleDecodes both refuse a declaration above
     * kSnappyOracleMaxOutput, so every mutant declaring more than that is
     * in parity for the HARNESS's reason and tests nothing about either
     * decoder - the corpus's own preamble-declares-max mutant included.
     * What is testable without allocating gigabytes is that the parser
     * refuses it on the capacity rung and says so, rather than letting a
     * 4 GiB declaration reach an element. */
    {
        const Bytes huge = {0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x0C,
                            'a',  'b',  'c',  'd'};
        uint64_t declared = 0;
        uint64_t preamble_size = 0;
        REQUIRE(cudec_detail::SnappyDeclaredLength(huge.data(), huge.size(),
                                                   &declared,
                                                   &preamble_size) ==
                CUDEC_OK);
        REQUIRE(declared == 0xFFFFFFFFu);
        REQUIRE(preamble_size == 5);
        /* Driven through Begin rather than Drive: the refusal has to happen
         * before anything is sized, and a driver that allocated the
         * capacity first would be testing the allocator. */
        cudec_detail::SnappyParser parser{huge.data(), huge.size(),
                                          declared - 1};
        REQUIRE(parser.Begin() == CUDEC_ERR_OUTPUT_TOO_SMALL);
        REQUIRE(parser.dst_pos == 0);
        Bytes twin_out;
        REQUIRE(TwinDecode(huge, &twin_out) == CUDEC_ERR_OUTPUT_TOO_SMALL);
        Bytes oracle_out;
        REQUIRE(!SnappyOracleDecodes(huge, &oracle_out));
    }

    /* The accept set, held open in both directions: snappy accepts these
     * and so must the twin, byte for byte. */
    const auto accepted = MakeAcceptCases();
    for (const auto& a : accepted) {
        Bytes oracle_out;
        REQUIRE_CTX(SnappyOracleDecodes(a.stream, &oracle_out),
                    "accept case %s: snappy rejects it", a.name.c_str());
        Bytes twin_out;
        REQUIRE_CTX(TwinDecode(a.stream, &twin_out) == CUDEC_OK,
                    "accept case %s: the twin over-rejects", a.name.c_str());
        REQUIRE_CTX(twin_out.size() == a.expected.size(), "accept case %s",
                    a.name.c_str());
        REQUIRE_CTX(equal_bytes(twin_out.data(), a.expected.data(),
                                twin_out.size()),
                    "accept case %s", a.name.c_str());
        REQUIRE_CTX(twin_out.size() == oracle_out.size() &&
                        equal_bytes(twin_out.data(), oracle_out.data(),
                                    twin_out.size()),
                    "accept case %s: oracle bytes", a.name.c_str());
    }

    /* Offsets at and above the 65536 the compressor cannot emit, swept
     * across the boundary in both directions. */
    for (uint32_t offset : {65535u, 65536u, 65537u, 131072u}) {
        const Bytes s = WideOffsetStream(offset, 4);
        Bytes oracle_out;
        REQUIRE_CTX(SnappyOracleDecodes(s, &oracle_out), "wide offset %u",
                    offset);
        Bytes twin_out;
        REQUIRE_CTX(TwinDecode(s, &twin_out) == CUDEC_OK, "wide offset %u",
                    offset);
        REQUIRE_CTX(twin_out.size() == oracle_out.size(), "wide offset %u",
                    offset);
        REQUIRE_CTX(
            equal_bytes(twin_out.data(), oracle_out.data(), twin_out.size()),
            "wide offset %u", offset);
        /* And one byte past what the stream produced, which must reject. */
        const Bytes past = WideOffsetStream(offset, 4);
        Bytes over = past;
        over[over.size() - 4] = static_cast<unsigned char>((offset + 1) & 0xff);
        over[over.size() - 3] =
            static_cast<unsigned char>(((offset + 1) >> 8) & 0xff);
        over[over.size() - 2] =
            static_cast<unsigned char>(((offset + 1) >> 16) & 0xff);
        over[over.size() - 1] =
            static_cast<unsigned char>(((offset + 1) >> 24) & 0xff);
        Bytes over_out;
        REQUIRE_CTX(TwinDecode(over, &over_out) == CUDEC_ERR_CORRUPT_INPUT,
                    "wide offset %u, one past the produced output", offset);
        Bytes over_oracle;
        REQUIRE_CTX(!SnappyOracleDecodes(over, &over_oracle),
                    "wide offset %u: snappy accepts a copy past its output",
                    offset);
    }

    /* The capacity axis, which the format cannot express and the oracle
     * therefore cannot be asked about: the declared length is checked
     * against the caller's capacity before any element is parsed, and the
     * refusal is CUDEC_ERR_OUTPUT_TOO_SMALL rather than a corrupt-input
     * verdict on a stream that is not corrupt. */
    {
        const auto& f = fixtures.front();
        const size_t size = f.compressed.size();
        auto tight = std::make_unique<unsigned char[]>(size);
        std::memcpy(tight.get(), f.compressed.data(), size);
        uint64_t declared = 0;
        uint64_t preamble_size = 0;
        REQUIRE(cudec_detail::SnappyDeclaredLength(tight.get(), size,
                                                   &declared,
                                                   &preamble_size) ==
                CUDEC_OK);
        REQUIRE(declared == f.original.size());
        REQUIRE(preamble_size >= 1);

        Bytes out;
        REQUIRE(Drive(tight.get(), size, declared - 1, &out) ==
                CUDEC_ERR_OUTPUT_TOO_SMALL);
        REQUIRE(out.empty());
        /* Exactly enough still decodes, so the bound is not off by one. */
        REQUIRE(Drive(tight.get(), size, declared, &out) == CUDEC_OK);
        REQUIRE(out.size() == f.original.size());
        REQUIRE(equal_bytes(out.data(), f.original.data(), out.size()));
        /* And a capacity beyond the declaration produces the declared
         * length, never the capacity - the anti-pattern a decoder that
         * sizes its output from the caller would have. */
        REQUIRE(Drive(tight.get(), size, declared + 4096, &out) == CUDEC_OK);
        REQUIRE(out.size() == f.original.size());
        REQUIRE(equal_bytes(out.data(), f.original.data(), out.size()));
    }

    /* Liveness, over every stream this test has: valid, mutated and
     * crafted. Every non-terminal call consumes at least one source byte,
     * the terminal call is reached, and a further call after it is refused
     * - so a driver that ignores *done terminates on a reject instead of
     * spinning a warp lane against the launch. */
    size_t liveness_streams = 0;
    {
        std::vector<Bytes> streams;
        for (const auto& f : fixtures) {
            streams.push_back(f.compressed);
            for (const auto& m : MutateSnappyStream(f.compressed, f.seed)) {
                streams.push_back(m.stream);
            }
        }
        for (const auto& c : crafted) {
            streams.push_back(c.stream);
        }
        for (const auto& a : accepted) {
            streams.push_back(a.stream);
        }
        /* Plus every stream of up to two bytes, exhaustively: that is where
         * a parser mishandling its own terminal state shows first. */
        streams.push_back(Bytes{});
        for (unsigned first = 0; first < 256; first++) {
            streams.push_back(Bytes{static_cast<unsigned char>(first)});
            for (unsigned second = 0; second < 256; second++) {
                streams.push_back(Bytes{static_cast<unsigned char>(first),
                                        static_cast<unsigned char>(second)});
            }
        }
        for (const auto& stream : streams) {
            const char* why = "";
            REQUIRE_CTX(LivenessHolds(stream, &why),
                        "liveness on a %zu-byte stream: %s", stream.size(),
                        why);
            liveness_streams++;
        }
    }

    /* No element the parser handed back was outside the buffers the caller
     * could execute it in, on any stream above. */
    REQUIRE(g_bounds_violations == 0);

    /* The branch-count lock (issue #173): every rung of the ladder in
     * src/snappy_block.h was reached by a DECLARED NEGATIVE, and the one rung
     * declared unreachable there was reached by none of them.
     *
     * The enumeration is not restated here - it is walked. A rung added to
     * the header with no negative behind it leaves a hole this loop names,
     * and deleting the negative that reached a rung opens the same hole;
     * tests/CMakeLists.txt holds the other end, refusing a refusal that does
     * not pass through the enumerated choke point at all.
     *
     * The unreachable rung is fail-closed in BOTH directions. It carries no
     * negative because the argument for it is that no input reaches it - so
     * if one ever does, the argument was wrong and that has to red rather
     * than pass as extra coverage. */
    {
        size_t branches_covered = 0;
        for (int branch = cudec_detail::kSnappyRejectNone + 1;
             branch < cudec_detail::kSnappyRejectCount; branch++) {
            const bool covered = g_reject_covered[branch];
            if (branch == cudec_detail::kSnappyRejectPreambleUnreachable) {
                REQUIRE_CTX(!covered,
                            "ladder rung %d is declared unreachable in "
                            "src/snappy_block.h and a negative reached it",
                            branch);
                continue;
            }
            REQUIRE_CTX(covered,
                        "ladder rung %d has no declared negative that reaches "
                        "it - add one, or the rung is untested (issue #173)",
                        branch);
            branches_covered++;
        }
        REQUIRE(branches_covered ==
                static_cast<size_t>(cudec_detail::kSnappyRejectCount) - 2);
    }

    std::printf("PASS: %zu snappy fixtures + %zu mutants in oracle parity "
                "(%zu oracle-rejected, %zu stricter-than-snappy); %zu crafted "
                "negatives; %zu accept cases held open; wide-offset sweep and "
                "the capacity axis pinned; the 32-bit literal-length wrap "
                "pinned as the one divergence; liveness on %zu streams; %d of "
                "%d ladder rungs covered by a declared negative, 1 declared "
                "unreachable\n",
                fixtures.size(), mutant_total, rejected_total,
                stricter_than_oracle, crafted.size(), accepted.size(),
                liveness_streams,
                static_cast<int>(cudec_detail::kSnappyRejectCount) - 2,
                static_cast<int>(cudec_detail::kSnappyRejectCount) - 1);
    return 0;
}
