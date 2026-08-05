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

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* Counts elements the parser handed back that the caller could not have
 * executed without leaving its buffers, and cursors left outside their own
 * bounds. main() requires it to be zero; the status Drive returns is left
 * alone, so a violation cannot be swallowed by a parity comparison that
 * only reads verdicts.
 *
 * It exists because a verdict alone cannot prove a bounds guard. Removing
 * the check that a long-form literal's length bytes are present, or that a
 * copy's offset bytes are, still rejects every crafted stream: the cursor
 * runs past the source end, the next subtraction wraps, and a later branch
 * catches it after the read has already happened. What the guard actually
 * prevents is that read, so that is what is counted. Measured - three of
 * those removals left a verdict-only suite green. */
size_t g_bounds_violations = 0;

/* Sequential reference execution of the parsed elements. A copy is the
 * chasing byte copy - reads may land on just-written bytes, which is the
 * pattern-repeating semantics an overlapping Snappy copy has. */
cudec_status Drive(const unsigned char* src, uint64_t src_size,
                   uint64_t capacity, Bytes* out) {
    out->assign(static_cast<size_t>(capacity), 0);
    cudec_detail::SnappyParser parser{src, src_size, capacity};
    cudec_detail::SnappyElement element;
    bool done = false;
    /* Fuel, for the same reason every driver loop in this tree carries one:
     * a parser that stops making progress must red the run rather than hang
     * it. Every non-terminal call consumes at least one source byte, so one
     * call per byte plus the terminal one is more than a live parser needs. */
    uint64_t fuel = src_size + 2;
    while (true) {
        if (fuel-- == 0) {
            std::fprintf(stderr, "the parser did not terminate on a %llu-byte "
                                 "stream\n",
                         static_cast<unsigned long long>(src_size));
            g_bounds_violations++;
            out->clear();
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        const cudec_status status = parser.Next(&element, &done);
        if (status != CUDEC_OK) {
            /* A rejected stream produces nothing, the same contract
             * SnappyOracleDecodes holds its caller to. */
            out->clear();
            return status;
        }
        /* Both cursors stay inside their buffers, the produced length stays
         * inside the declaration, and the element lies inside what the
         * parser says it produced. A literal's source range is inside the
         * stream; a copy reaches strictly backwards, which is what makes the
         * chasing read below defined. */
        const bool bounded =
            parser.src_pos <= parser.src_size &&
            parser.dst_pos <= parser.declared &&
            parser.declared <= capacity &&
            element.to + element.length <= parser.dst_pos &&
            (element.is_copy ? element.from < element.to
                             : element.from + element.length <=
                                   parser.src_pos);
        if (!bounded) {
            std::fprintf(stderr,
                         "the parser left its bounds: src_pos=%llu/%llu "
                         "dst_pos=%llu/%llu element from=%llu to=%llu len=%llu "
                         "copy=%d\n",
                         static_cast<unsigned long long>(parser.src_pos),
                         static_cast<unsigned long long>(parser.src_size),
                         static_cast<unsigned long long>(parser.dst_pos),
                         static_cast<unsigned long long>(parser.declared),
                         static_cast<unsigned long long>(element.from),
                         static_cast<unsigned long long>(element.to),
                         static_cast<unsigned long long>(element.length),
                         element.is_copy ? 1 : 0);
            g_bounds_violations++;
            out->clear();
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        if (element.is_copy) {
            for (uint64_t i = 0; i < element.length; i++) {
                (*out)[static_cast<size_t>(element.to + i)] =
                    (*out)[static_cast<size_t>(element.from + i)];
            }
        } else {
            for (uint64_t i = 0; i < element.length; i++) {
                (*out)[static_cast<size_t>(element.to + i)] =
                    src[element.from + i];
            }
        }
        if (done) {
            break;
        }
    }
    out->resize(static_cast<size_t>(parser.dst_pos));
    return CUDEC_OK;
}

/* Parse from an EXACTLY-sized copy of the stream, as TwinDecode in
 * tests/parser_twin.cpp does: the mutation corpus truncates by
 * resize()-DOWN, which leaves the vector's rounded-up capacity readable, so
 * an over-read past src_size would land in that slack and leave ASan green.
 * A tight allocation puts a redzone right after the last stream byte. */
cudec_status TwinDecode(const Bytes& stream, Bytes* out) {
    out->clear();
    const size_t stream_size = stream.size();
    auto tight = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(tight.get(), stream.data(), stream_size);
    }
    /* The declaration is read through the parser's own Begin, so the varint
     * is parsed once and the destination is sized from a value that has
     * already been compared against a capacity. The capacity here is the
     * harness bound rather than the declaration itself: that is the same
     * refusal SnappyOracleDecodes applies, so a mutant declaring 4 GiB is
     * refused on both sides for the same reason instead of reading as a
     * parity failure. */
    cudec_detail::SnappyParser probe{tight.get(), stream_size,
                                     kSnappyOracleMaxOutput};
    const cudec_status header = probe.Begin();
    if (header != CUDEC_OK) {
        return header;
    }
    return Drive(tight.get(), stream_size, probe.declared, out);
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
    cudec_detail::SnappyElement element;
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
};

/* One crafted stream per validation-ladder branch. Each pins the twin's
 * specific status AND asserts snappy rejects the same bytes, so the branch
 * is demonstrably in parity rather than merely present. Byte layouts
 * verified by trace against src/snappy_block.h. */
std::vector<CraftedNegative> MakeCraftedNegatives() {
    const Bytes abcd = Ascii("abcd");
    std::vector<CraftedNegative> out;

    /* Preamble: no byte at all. */
    out.push_back({"preamble-absent", {}, CUDEC_ERR_CORRUPT_INPUT});
    /* Preamble: a continuation bit with nothing behind it. */
    out.push_back({"preamble-truncated", {0x80}, CUDEC_ERR_CORRUPT_INPUT});
    /* Preamble: a sixth group. The fifth byte still carries the
     * continuation bit, which the reference refuses outright. */
    out.push_back({"preamble-six-groups",
                   {0x80, 0x80, 0x80, 0x80, 0x80, 0x00},
                   CUDEC_ERR_CORRUPT_INPUT});
    /* Preamble: five groups whose value needs a 33rd bit. */
    out.push_back({"preamble-overflows-32-bit",
                   {0x80, 0x80, 0x80, 0x80, 0x10},
                   CUDEC_ERR_CORRUPT_INPUT});

    /* Elements: a declaration with no elements behind it. */
    {
        Bytes s = Preamble(4);
        out.push_back({"under-production-empty", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    /* Elements: production stops short of the declaration. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        out.push_back({"under-production-short", s, CUDEC_ERR_CORRUPT_INPUT});
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
                       CUDEC_ERR_CORRUPT_INPUT});
    }
    {
        Bytes s = Preamble(70000);
        s.push_back(0xF4); /* class 61: two length bytes follow, and none do */
        out.push_back({"literal-header-truncated-wide", s,
                       CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A literal whose payload runs past the source end. */
    {
        Bytes s = Preamble(4);
        Append(&s, Literal(abcd));
        s.resize(s.size() - 2);
        out.push_back({"literal-payload-truncated", s,
                       CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A literal longer than the declaration leaves room for. */
    {
        Bytes s = Preamble(2);
        Append(&s, Literal(abcd));
        out.push_back({"literal-over-production", s,
                       CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A copy whose offset bytes run past the source end. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        s.push_back(0x0E); /* a two-byte-offset copy tag, with no offset */
        out.push_back({"copy-header-truncated", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    /* The same for the narrow form, whose header is one byte shorter and
     * whose guard is therefore a separate branch. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        s.push_back(0x01); /* a one-byte-offset copy tag, with no offset */
        out.push_back({"copy1-header-truncated", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    /* And for the four-byte-offset form, which is missing three of them. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy4(4, 4));
        s.resize(s.size() - 3);
        out.push_back({"copy4-header-truncated", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A copy with no byte to replicate, in each of the three forms, both
     * with output behind it and with none. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy2(4, 0));
        out.push_back({"copy-offset-zero-tag-10", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy4(4, 0));
        out.push_back({"copy-offset-zero-tag-11", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy1(4, 0));
        out.push_back({"copy-offset-zero-tag-01", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    {
        Bytes s = Preamble(4);
        Append(&s, Copy2(4, 0));
        out.push_back({"copy-offset-zero-at-start", s,
                       CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A copy reaching before the start of the produced output. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy2(4, 5));
        out.push_back({"copy-offset-before-start", s,
                       CUDEC_ERR_CORRUPT_INPUT});
    }
    /* The same, through the four-byte-offset form, where the offset is
     * wide enough to be a pointer-sized mistake rather than a small one. */
    {
        Bytes s = Preamble(8);
        Append(&s, Literal(abcd));
        Append(&s, Copy4(4, 0xFFFFFFFFu));
        out.push_back({"copy-offset-huge", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A copy longer than the declaration leaves room for. */
    {
        Bytes s = Preamble(6);
        Append(&s, Literal(abcd));
        Append(&s, Copy2(4, 1));
        out.push_back({"copy-over-production", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A complete stream with one more element behind it: the declared
     * length is already produced, so the trailing element over-produces. */
    {
        Bytes s = Preamble(4);
        Append(&s, Literal(abcd));
        Append(&s, Literal(Ascii("x")));
        out.push_back({"trailing-element", s, CUDEC_ERR_CORRUPT_INPUT});
    }
    /* A complete stream with one dangling tag byte: the source is not
     * consumed exactly, and the tag's own payload is missing too. */
    {
        Bytes s = Preamble(4);
        Append(&s, Literal(abcd));
        s.push_back(0x00);
        out.push_back({"trailing-tag", s, CUDEC_ERR_CORRUPT_INPUT});
    }
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
        REQUIRE_CTX(TwinDecode(c.stream, &twin_out) == c.expected,
                    "crafted %s", c.name);
        REQUIRE_CTX(twin_out.empty(),
                    "crafted %s: a rejected stream produced output", c.name);
        Bytes oracle_out;
        REQUIRE_CTX(!SnappyOracleDecodes(c.stream, &oracle_out),
                    "crafted %s: snappy unexpectedly accepts", c.name);
        REQUIRE_CTX(!SnappyOracleAccepts(c.stream),
                    "crafted %s: snappy's validator unexpectedly accepts",
                    c.name);
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

    std::printf("PASS: %zu snappy fixtures + %zu mutants in oracle parity "
                "(%zu oracle-rejected, %zu stricter-than-snappy); %zu crafted "
                "negatives; %zu accept cases held open; wide-offset sweep and "
                "the capacity axis pinned; the 32-bit literal-length wrap "
                "pinned as the one divergence; liveness on %zu streams\n",
                fixtures.size(), mutant_total, rejected_total,
                stricter_than_oracle, crafted.size(), accepted.size(),
                liveness_streams);
    return 0;
}
