/* The CPU twin of Zstd repeat-offset resolution (src/zstd_repcode.h): the
 * three-slot history, the zero-literals index shift, the minus-one rule and
 * the recency rotation (issue #197). The sibling of tests/zstd_seq_twin.cpp,
 * which decodes the Offset_Values this unit resolves.
 *
 * THREE PROOFS, AND THEY ARE DIFFERENT IN KIND.
 *
 * THE RULES ARE PINNED ONE BY ONE AGAINST HAND-COMPUTED EXPECTATIONS, history
 * slot by history slot. Each rule from RFC 8878 section 3.1.1.5 has its own
 * named row, because the rules share an implementation and a single test over
 * all of them would pass on an implementation that got two of them wrong in
 * compensating directions.
 *
 * THE ORACLE DECIDES THE BYTES, over frames this file writes. A resolved
 * offset is not observable on its own, so each crafted frame is decoded twice:
 * once by the pinned reference, and once by a driver here that copies literals
 * and matches at the offsets this unit resolves. The two byte strings must be
 * equal. That is what turns "the offset came out as 4" into a claim about the
 * format rather than about this file's arithmetic - and the reject direction
 * is the same frame construction with the rule violated, where the reference
 * must refuse and this unit must refuse too.
 *
 * THE CORPUS IS WHERE THE ROTATION IS EXERCISED DENSELY. The crafted frames
 * carry one or two sequences each; a compressor's block carries thousands, and
 * the rotation is the rule whose errors only show up after enough of them. So
 * the driver decodes every #185 fixture whole - literals through
 * src/zstd_literals.h, sequences through src/zstd_seq.h, offsets through the
 * unit under test - and requires the result to equal the reference's byte for
 * byte.
 *
 * THE DRIVER IS THIS FILE'S AND NOT A SHIPPED UNIT. Executing a sequence and
 * looping over a frame's blocks are their own issues with their own contracts;
 * what is here is the least code that turns resolved offsets into bytes the
 * reference can be asked about, which is the only way this issue's claim is
 * observable at all. */
#include "require.h"
#include "zstd_corpus.h"
#include "zstd_repcode.h"
#include "zstd_twin_driver.h"

#include <zstd.h>
#include <zstd_errors.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = cudec_twin::Bytes;

using cudec_detail::ZstdRepcodeHistory;
using cudec_detail::ZstdRepcodeInit;
using cudec_detail::ZstdRepcodeReject;
using cudec_detail::ZstdRepcodeResolve;

using cudec_twin::DecodeFrame;
using cudec_twin::kPathCount;
using cudec_twin::kPathNames;
using cudec_twin::Run;

/* One capacity for every driver run, larger than anything decoded here. */
constexpr uint64_t kDriverCapacity = 8ull * 1024ull * 1024ull;

size_t g_path_corpus[kPathCount] = {0};
size_t g_path_crafted[kPathCount] = {0};

/* Which reject rungs a declared negative reached. Same discipline as the
 * sibling twins: the enumeration lives once, in the header, and main()
 * requires every rung to have been named by a negative written to reach it. */
bool g_reject_covered[cudec_detail::kZstdRepcodeRejectCount] = {false};

void CoverRung(ZstdRepcodeReject rung) {
    if (rung != cudec_detail::kZstdRepcodeRejectNone) {
        g_reject_covered[rung] = true;
    }
}

/* One resolution, for the rule rows: the offset that came out and the history
 * left behind, so a row asserts both halves of what the unit does. */
struct Resolution {
    cudec_status status;
    ZstdRepcodeReject rung;
    uint64_t offset;
    uint64_t slot[3];
};

Resolution Resolve(ZstdRepcodeHistory* history, uint64_t offset_value,
                   uint32_t literals_length) {
    Resolution out;
    out.rung = cudec_detail::kZstdRepcodeRejectNone;
    out.offset = 0;
    out.status = ZstdRepcodeResolve(history, offset_value, literals_length,
                                    &out.offset, &out.rung);
    for (unsigned i = 0; i < 3; i++) {
        out.slot[i] = history->slot[i];
    }
    return out;
}

/* ---------------------------------------------------------------- frames */

void Append(Bytes* out, const Bytes& more) {
    out->insert(out->end(), more.begin(), more.end());
}

Bytes BlockHeader(bool last, unsigned type, size_t size) {
    const uint32_t value = (last ? 1u : 0u) | (type << 1) |
                           (static_cast<uint32_t>(size) << 3);
    return Bytes{static_cast<unsigned char>(value & 0xff),
                 static_cast<unsigned char>((value >> 8) & 0xff),
                 static_cast<unsigned char>((value >> 16) & 0xff)};
}

Bytes RawBlock(const Bytes& payload, bool last) {
    Bytes out = BlockHeader(last, 0, payload.size());
    Append(&out, payload);
    return out;
}

/* Bits in the order the DECODER reads them. A sequence bitstream is read
 * backward from its last byte, so the reversal lives here and callers think
 * forward. */
struct BitWriter {
    std::string bits;

    void Add(uint32_t value, unsigned count) {
        for (unsigned i = count; i > 0; i--) {
            bits.push_back(((value >> (i - 1)) & 1u) ? '1' : '0');
        }
    }
};

/* Seal a bitstream: the end marker goes on first, then zero padding above it
 * to a byte boundary, and the byte the decoder starts at is emitted last. */
Bytes SealBitstream(const std::string& read_order) {
    std::string all = "1" + read_order;
    while (all.size() % 8 != 0) {
        all.insert(all.begin(), '0');
    }
    const size_t byte_count = all.size() / 8;
    Bytes out(byte_count, 0);
    for (size_t chunk = 0; chunk < byte_count; chunk++) {
        unsigned char byte = 0;
        for (size_t bit = 0; bit < 8; bit++) {
            byte = static_cast<unsigned char>(
                (byte << 1) | (all[chunk * 8 + bit] == '1' ? 1 : 0));
        }
        out[byte_count - 1 - chunk] = byte;
    }
    return out;
}

/* One sequence as this file writes it, in the symbols a block carries rather
 * than in the values they mean. `offset_extra` is the Offset_Code's extra bit
 * field, so (1 << offset_code) + offset_extra is the Offset_Value. */
struct SeqSpec {
    unsigned litlen_code;
    unsigned offset_code;
    uint32_t offset_extra;
    unsigned matchlen_code;
};

/* A compressed block whose three sequence tables are RLE, so every state
 * initialisation and update reads zero bits and the only bits in the stream
 * are the codes' extra bits. All three RLE symbols are per-block, so every
 * sequence in one of these blocks shares its three codes - which is why the
 * rows below use one or two sequences each and the corpus carries the rest.
 *
 * Layout (RFC 8878 3.1.1.3.1 and 3.1.1.3.2): a Raw literals header with
 * Size_Format 00, the literals, Number_Of_Sequences, Symbol_Compression_Modes
 * with all three fields RLE, one symbol byte per field in the order
 * Literals_Lengths, Offsets, Match_Lengths, then the bitstream. */
Bytes RleSequenceBlock(const Bytes& literals, unsigned count,
                       const SeqSpec& spec, bool last) {
    BitWriter writer;
    /* Read order inside one sequence: the offset's extra bits, then the
     * match length's, then the literals length's. The two length codes used
     * here carry none. */
    for (unsigned i = 0; i < count; i++) {
        writer.Add(spec.offset_extra, spec.offset_code);
    }
    Bytes content;
    content.push_back(static_cast<unsigned char>(literals.size() << 3));
    Append(&content, literals);
    content.push_back(static_cast<unsigned char>(count));
    content.push_back(0x54); /* all three modes RLE */
    content.push_back(static_cast<unsigned char>(spec.litlen_code));
    content.push_back(static_cast<unsigned char>(spec.offset_code));
    content.push_back(static_cast<unsigned char>(spec.matchlen_code));
    Append(&content, SealBitstream(writer.bits));

    Bytes out = BlockHeader(last, 2, content.size());
    Append(&out, content);
    return out;
}

/* A single-segment frame: its window is its declared content size, and the
 * one-byte content-size field holds anything these rows regenerate. */
Bytes Frame(const std::vector<Bytes>& blocks, unsigned content_size) {
    Bytes frame = {0x28, 0xb5, 0x2f, 0xfd, 0x20,
                   static_cast<unsigned char>(content_size)};
    for (size_t i = 0; i < blocks.size(); i++) {
        Append(&frame, blocks[i]);
    }
    return frame;
}

/* Distinct-enough history bytes, so a wrong offset produces wrong bytes
 * rather than accidentally-right ones. */
Bytes HistoryBytes(size_t size) {
    Bytes out;
    out.reserve(size);
    for (size_t i = 0; i < size; i++) {
        out.push_back(static_cast<unsigned char>('A' + (i % 26)));
    }
    return out;
}

/* One capacity for every call, larger than anything any frame here
 * regenerates. It is a named constant rather than a per-call number because
 * the reject rows have to be able to say a refusal was NOT about room, and
 * they could not if the capacity were ever the binding constraint. The flag
 * below is what makes that check real rather than a claim: a refusal for
 * want of room sets it, and main() reds on it. This test met exactly that
 * failure while it was being written, on the largest generated source. */
constexpr size_t kDecodeCapacity = 1024 * 1024;
bool g_decode_capacity_hit = false;

bool OracleDecodes(const Bytes& frame, Bytes* out) {
    Bytes decoded(kDecodeCapacity, 0);
    const size_t produced = ZSTD_decompress(decoded.data(), decoded.size(),
                                            frame.data(), frame.size());
    if (ZSTD_isError(produced)) {
        if (ZSTD_getErrorCode(produced) == ZSTD_error_dstSize_tooSmall) {
            g_decode_capacity_hit = true;
        }
        out->clear();
        return false;
    }
    decoded.resize(produced);
    *out = decoded;
    return true;
}

/* ------------------------------------------------------------- the rows */

size_t g_frames = 0;
size_t g_bytes = 0;
size_t g_sequences = 0;
size_t g_repeats = 0;
size_t g_outside_subset = 0;

/* One frame: the reference decodes it, the driver decodes it, and the two
 * byte strings must be equal. `tally` separates what a compressor emitted
 * from what this file wrote, because the two answer different questions - one
 * is coverage of the format as it is used, the other is coverage of rules a
 * compressor will not produce. */
int OracleAgrees(const char* name, const Bytes& frame, size_t* tally) {
    Bytes reference;
    REQUIRE_CTX(OracleDecodes(frame, &reference),
                "%s: the reference refused the frame", name);
    const Run run = DecodeFrame(frame, kDriverCapacity);
    REQUIRE_CTX(run.ok, "%s: %s", name, run.why.c_str());
    REQUIRE_CTX(run.output.size() == reference.size(),
                "%s: %zu bytes, the reference produced %zu", name,
                run.output.size(), reference.size());
    REQUIRE_CTX(equal_bytes(run.output.data(), reference.data(),
                            reference.size()),
                "%s: bytes differ from the reference", name);
    g_frames++;
    g_bytes += run.output.size();
    g_sequences += run.sequences;
    g_repeats += run.repeats;
    for (unsigned i = 0; i < kPathCount; i++) {
        tally[i] += run.path[i];
    }
    return 0;
}

/* The reject direction: the reference must refuse the frame, and the driver
 * must stop at the repeat-offset rung rather than anywhere else. */
int OracleAndTwinRefuse(const char* name, const Bytes& frame,
                        ZstdRepcodeReject want_rung) {
    Bytes reference;
    REQUIRE_CTX(!OracleDecodes(frame, &reference),
                "%s: the reference accepted it", name);
    const Run run = DecodeFrame(frame, kDriverCapacity);
    REQUIRE_CTX(!run.ok, "%s: the driver accepted it", name);
    REQUIRE_CTX(run.stage == cudec_twin::kStageRepcode,
                "%s: refused at the %s stage: %s", name,
                cudec_twin::kStageNames[run.stage], run.why.c_str());
    REQUIRE_CTX(run.rung == static_cast<int>(want_rung), "%s: rung %d", name,
                run.rung);
    CoverRung(static_cast<ZstdRepcodeReject>(run.rung));
    return 0;
}

int Rules() {
    /* The history a frame starts with. */
    {
        ZstdRepcodeHistory history;
        ZstdRepcodeInit(&history);
        REQUIRE(history.slot[0] == 1);
        REQUIRE(history.slot[1] == 4);
        REQUIRE(history.slot[2] == 8);
    }

    /* An explicit Offset_Value is three larger than the distance it names,
     * because the three smallest encodings are spent on the repeats. It goes
     * to the front and the oldest slot falls off. */
    {
        ZstdRepcodeHistory history;
        ZstdRepcodeInit(&history);
        const Resolution r = Resolve(&history, 40, 5);
        REQUIRE(r.status == CUDEC_OK);
        REQUIRE(r.offset == 37);
        REQUIRE(r.slot[0] == 37);
        REQUIRE(r.slot[1] == 1);
        REQUIRE(r.slot[2] == 4);
    }

    /* With a non-zero literals length, 1, 2 and 3 name the three slots in
     * order. Each is resolved from the same starting history, so the three
     * rows are independent of each other. */
    {
        const uint64_t want[3] = {1, 4, 8};
        for (uint64_t value = 1; value <= 3; value++) {
            ZstdRepcodeHistory history;
            ZstdRepcodeInit(&history);
            const Resolution r = Resolve(&history, value, 1);
            REQUIRE_CTX(r.status == CUDEC_OK, "offset value %llu",
                        static_cast<unsigned long long>(value));
            REQUIRE_CTX(r.offset == want[value - 1], "offset value %llu gave "
                                                     "%llu",
                        static_cast<unsigned long long>(value),
                        static_cast<unsigned long long>(r.offset));
        }
    }

    /* With a literals length of zero the meaning shifts by one: 1 names the
     * second slot, 2 the third, and 3 is the minus-one rule. */
    {
        ZstdRepcodeHistory history;
        ZstdRepcodeInit(&history);
        REQUIRE(Resolve(&history, 1, 0).offset == 4);

        ZstdRepcodeInit(&history);
        REQUIRE(Resolve(&history, 2, 0).offset == 8);

        /* The minus-one rule where it is legal: a history whose most recent
         * offset is 5 resolves Offset_Value 3 with no literals to 4. */
        history.slot[0] = 5;
        history.slot[1] = 1;
        history.slot[2] = 4;
        const Resolution r = Resolve(&history, 3, 0);
        REQUIRE(r.status == CUDEC_OK);
        REQUIRE(r.offset == 4);
        /* It is a push, not a slot: the value that produced it stays. */
        REQUIRE(r.slot[0] == 4);
        REQUIRE(r.slot[1] == 5);
        REQUIRE(r.slot[2] == 1);
    }

    /* The minus-one rule where it would yield zero. At frame start the most
     * recent offset is one, so this is the shape a stream reaches it from. */
    {
        ZstdRepcodeHistory history;
        ZstdRepcodeInit(&history);
        const Resolution r = Resolve(&history, 3, 0);
        REQUIRE(r.status == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(r.rung == cudec_detail::kZstdRepcodeRejectResolvedToZero);
        /* The history did not move: a refusal leaves the frame's state alone
         * rather than half-updated. */
        REQUIRE(r.slot[0] == 1);
        REQUIRE(r.slot[1] == 4);
        REQUIRE(r.slot[2] == 8);
        CoverRung(r.rung);
    }

    /* The same rung reached the other way: a history that was never
     * initialised. This is what makes the check a property of the resolved
     * value rather than of the minus-one branch alone.
     *
     * BOTH ARMS, because they fail differently. Taking a zero slot resolves
     * to zero and is refused; taking the MINUS-ONE rule on a zero slot would
     * wrap an unsigned subtraction into the widest offset there is unless the
     * slot is read before the subtraction, and a wrap is a fail-open however
     * loudly whatever runs next refuses it. */
    {
        ZstdRepcodeHistory history;
        history.slot[0] = 0;
        history.slot[1] = 0;
        history.slot[2] = 0;
        Resolution r = Resolve(&history, 1, 1);
        REQUIRE(r.status == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(r.rung == cudec_detail::kZstdRepcodeRejectResolvedToZero);
        CoverRung(r.rung);

        history.slot[0] = 0;
        history.slot[1] = 0;
        history.slot[2] = 0;
        r = Resolve(&history, 3, 0);
        REQUIRE(r.status == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(r.rung == cudec_detail::kZstdRepcodeRejectResolvedToZero);
        REQUIRE(r.offset == 0);
        CoverRung(r.rung);
    }

    /* The recency rotation, one row per slot taken, on the repcode path.
     * Taking the most recent moves nothing; taking the second swaps the top
     * two; taking the third rotates all three. Starting from a history whose
     * three values are distinct, so a wrong rotation cannot look right. */
    {
        ZstdRepcodeHistory history;
        history.slot[0] = 10;
        history.slot[1] = 20;
        history.slot[2] = 30;
        Resolution r = Resolve(&history, 1, 1);
        REQUIRE(r.offset == 10);
        REQUIRE(r.slot[0] == 10 && r.slot[1] == 20 && r.slot[2] == 30);

        history.slot[0] = 10;
        history.slot[1] = 20;
        history.slot[2] = 30;
        r = Resolve(&history, 2, 1);
        REQUIRE(r.offset == 20);
        REQUIRE(r.slot[0] == 20 && r.slot[1] == 10 && r.slot[2] == 30);

        history.slot[0] = 10;
        history.slot[1] = 20;
        history.slot[2] = 30;
        r = Resolve(&history, 3, 1);
        REQUIRE(r.offset == 30);
        REQUIRE(r.slot[0] == 30 && r.slot[1] == 10 && r.slot[2] == 20);
    }

    /* And the same rotation through the shifted indices, which take the same
     * slots from a zero literals length. */
    {
        ZstdRepcodeHistory history;
        history.slot[0] = 10;
        history.slot[1] = 20;
        history.slot[2] = 30;
        Resolution r = Resolve(&history, 1, 0);
        REQUIRE(r.offset == 20);
        REQUIRE(r.slot[0] == 20 && r.slot[1] == 10 && r.slot[2] == 30);

        history.slot[0] = 10;
        history.slot[1] = 20;
        history.slot[2] = 30;
        r = Resolve(&history, 2, 0);
        REQUIRE(r.offset == 30);
        REQUIRE(r.slot[0] == 30 && r.slot[1] == 10 && r.slot[2] == 20);
    }

    /* The caller-argument rung: an Offset_Value of zero is not a value
     * src/zstd_seq.h can produce, so it is a caller bug rather than a
     * stream. */
    {
        ZstdRepcodeHistory history;
        ZstdRepcodeInit(&history);
        uint64_t offset = 0;
        ZstdRepcodeReject rung = cudec_detail::kZstdRepcodeRejectNone;
        REQUIRE(ZstdRepcodeResolve(&history, 0, 1, &offset, &rung) ==
                CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(rung == cudec_detail::kZstdRepcodeRejectBadRequest);
        CoverRung(rung);
    }
    return 0;
}

/* The crafted frames. Each one puts a rule in front of the reference: the
 * history block gives the matches something to point at, and the sequence
 * block spells one rule in its three RLE symbols. */
int CraftedFrames() {
    const Bytes history = HistoryBytes(16);

    /* Offset_Value 1 with a zero literals length resolves to the SECOND
     * repeat offset, 4, not the first. Offset_Code 0 carries no extra bits,
     * Literals_Length_Code 0 is a length of zero, Match_Length_Code 0 is a
     * length of three. */
    {
        SeqSpec spec;
        spec.litlen_code = 0;
        spec.offset_code = 0;
        spec.offset_extra = 0;
        spec.matchlen_code = 0;
        std::vector<Bytes> blocks;
        blocks.push_back(RawBlock(history, false));
        blocks.push_back(RleSequenceBlock(Bytes(), 1, spec, true));
        const int rc = OracleAgrees("zero-literals-shift",
                                    Frame(blocks, 19), g_path_crafted);
        if (rc != 0) {
            return rc;
        }
    }

    /* The same Offset_Value with a literals length of one resolves to the
     * FIRST, 1. One byte of the block differs from the frame above - the
     * Literals_Lengths RLE symbol - plus the literal it needs, so the two
     * rows differ in the rule and not in the construction. */
    {
        SeqSpec spec;
        spec.litlen_code = 1;
        spec.offset_code = 0;
        spec.offset_extra = 0;
        spec.matchlen_code = 0;
        std::vector<Bytes> blocks;
        blocks.push_back(RawBlock(history, false));
        blocks.push_back(RleSequenceBlock(Bytes{'x'}, 1, spec, true));
        const int rc = OracleAgrees("no-shift-with-literals",
                                    Frame(blocks, 20), g_path_crafted);
        if (rc != 0) {
            return rc;
        }
    }

    /* An explicit offset, and then a repeat of it. Offset_Code 3 carries
     * three extra bits, so Offset_Value 8 with extra bits zero is a distance
     * of 5; the second sequence's Offset_Value 1 with a non-zero literals
     * length must resolve to that same 5, which is the recency rotation
     * observed through the reference. */
    {
        SeqSpec spec;
        spec.litlen_code = 1;
        spec.offset_code = 3;
        spec.offset_extra = 0;
        spec.matchlen_code = 0;
        std::vector<Bytes> blocks;
        blocks.push_back(RawBlock(history, false));
        blocks.push_back(
            RleSequenceBlock(Bytes{'x', 'y'}, 2, spec, true));
        const int rc = OracleAgrees("explicit-then-repeat",
                                    Frame(blocks, 24), g_path_crafted);
        if (rc != 0) {
            return rc;
        }
    }

    /* The history survives a block boundary. The first block's sequence sets
     * the most recent offset to 5 with an explicit Offset_Code 3; the second
     * block's uses the minus-one rule, which resolves to 4 only if the
     * history was NOT reset between the two. */
    {
        SeqSpec explicit_offset;
        explicit_offset.litlen_code = 0;
        explicit_offset.offset_code = 3;
        explicit_offset.offset_extra = 0;
        explicit_offset.matchlen_code = 0;
        SeqSpec minus_one;
        minus_one.litlen_code = 0;
        minus_one.offset_code = 1;
        minus_one.offset_extra = 1;
        minus_one.matchlen_code = 0;
        std::vector<Bytes> blocks;
        blocks.push_back(RawBlock(history, false));
        blocks.push_back(RleSequenceBlock(Bytes(), 1, explicit_offset, false));
        blocks.push_back(RleSequenceBlock(Bytes(), 1, minus_one, true));
        const int rc =
            OracleAgrees("history-crosses-blocks", Frame(blocks, 22),
                         g_path_crafted);
        if (rc != 0) {
            return rc;
        }
    }

    /* The reject direction, and its accepting twin. The first frame reaches
     * the minus-one rule at frame start, where the most recent offset is one
     * and the rule yields zero. The second differs in the Literals_Lengths
     * RLE symbol alone: with literals the shift is gone, Offset_Value 3 names
     * the third repeat offset, 8, and the frame decodes. */
    {
        SeqSpec spec;
        spec.litlen_code = 0;
        spec.offset_code = 1;
        spec.offset_extra = 1;
        spec.matchlen_code = 0;
        std::vector<Bytes> blocks;
        blocks.push_back(RawBlock(history, false));
        blocks.push_back(RleSequenceBlock(Bytes(), 1, spec, true));
        const int rc = OracleAndTwinRefuse(
            "minus-one-resolves-to-zero", Frame(blocks, 19),
            cudec_detail::kZstdRepcodeRejectResolvedToZero);
        if (rc != 0) {
            return rc;
        }

        spec.litlen_code = 1;
        std::vector<Bytes> accepted;
        accepted.push_back(RawBlock(history, false));
        accepted.push_back(RleSequenceBlock(Bytes{'x'}, 1, spec, true));
        const int rc2 =
            OracleAgrees("third-repeat-with-literals",
                         Frame(accepted, 20), g_path_crafted);
        if (rc2 != 0) {
            return rc2;
        }
    }
    return 0;
}

/* A source built to make a compressor reach for repeat offsets.
 *
 * The #185 corpus is built for the ENTROPY surfaces and its repeat-offset
 * density is whatever fell out of that; measured on it, four of the seven
 * resolution paths are never reached, so a corpus-only proof would be green
 * over rules it never ran. This source is the other half: a small pool of
 * phrases emitted over and over, so the same few distances recur and the
 * compressor spends repeat codes on them instead of explicit offsets. The
 * separators are what put sequences with and without literals next to each
 * other, which is the only way the shifted indices are reached at all.
 *
 * Deterministic: a counter-driven generator, so the bytes are the same on
 * every run and a failure names a fixture somebody else can rebuild. */
Bytes RepcodeDenseSource(size_t target, uint32_t seed) {
    const char* const kPhrases[] = {
        "the quick brown fox jumps ", "over the lazy dog again ",
        "pack my box with five doz", "en liquor jugs and more ",
        "how vexingly quick daft z", "ebras jump over fences  ",
        "sphinx of black quartz ju", "dge my vow to the letter"};
    const size_t kPhraseCount = sizeof(kPhrases) / sizeof(kPhrases[0]);
    Bytes out;
    out.reserve(target + 64);
    uint32_t state = seed;
    while (out.size() < target) {
        state = state * 1664525u + 1013904223u;
        const size_t which = (state >> 16) % kPhraseCount;
        const char* phrase = kPhrases[which];
        for (size_t i = 0; phrase[i] != 0; i++) {
            out.push_back(static_cast<unsigned char>(phrase[i]));
        }
        /* Three separator shapes, and the choice is what varies the literals
         * length of the sequence that follows: none at all, one byte, and a
         * short novel run. A source with only one of them reaches only the
         * shifted indices or only the unshifted ones. */
        state = state * 1664525u + 1013904223u;
        const unsigned shape = (state >> 16) % 3u;
        if (shape == 1) {
            out.push_back(static_cast<unsigned char>('0' + (state % 10u)));
        } else if (shape == 2) {
            for (unsigned i = 0; i < 4; i++) {
                state = state * 1664525u + 1013904223u;
                out.push_back(
                    static_cast<unsigned char>('!' + (state >> 20) % 60u));
            }
        }
    }
    return out;
}

/* The repeat-offset-dense corpus: sources above, compressed at a spread of
 * levels because the match finder's willingness to spend a repeat code is a
 * level-dependent decision, then decoded whole and held to the reference. */
int DenseCorpus() {
    const int levels[] = {1, 3, 9, 19};
    const size_t sizes[] = {8 * 1024, 64 * 1024, 300 * 1024};
    for (size_t z = 0; z < sizeof(sizes) / sizeof(sizes[0]); z++) {
        for (size_t l = 0; l < sizeof(levels) / sizeof(levels[0]); l++) {
            const Bytes source =
                RepcodeDenseSource(sizes[z], 0x9e3779b9u +
                                   static_cast<uint32_t>(z * 4 + l));
            Bytes frame(ZSTD_compressBound(source.size()), 0);
            const size_t written =
                ZSTD_compress(frame.data(), frame.size(), source.data(),
                              source.size(), levels[l]);
            REQUIRE(!ZSTD_isError(written));
            frame.resize(written);
            char name[96];
            std::snprintf(name, sizeof(name), "dense-%zu-level-%d", sizes[z],
                          levels[l]);
            cudec_detail::ZstdFrameHeader probe;
            cudec_detail::ZstdFrameReject frame_rung;
            REQUIRE_CTX(cudec_detail::ZstdParseFrameHeader(
                            frame.data(), frame.size(), &probe,
                            &frame_rung) == CUDEC_OK,
                        "%s: frame header refused", name);
            const int rc = OracleAgrees(name, frame, g_path_corpus);
            if (rc != 0) {
                return rc;
            }
            /* The reference's own round trip, so a fixture that decoded to
             * something both sides agreed on is still held to the bytes it
             * was built from. */
            Bytes reference;
            REQUIRE_CTX(OracleDecodes(frame, &reference), "%s", name);
            REQUIRE_CTX(reference.size() == source.size(), "%s", name);
            REQUIRE_CTX(equal_bytes(reference.data(), source.data(),
                                    source.size()),
                        "%s: the round trip lost bytes", name);
        }
    }
    return 0;
}

/* The corpus: every #185 fixture decoded whole and held to the reference byte
 * for byte. This is where the rotation meets thousands of sequences. */
int Corpus() {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());
    for (size_t f = 0; f < fixtures.size(); f++) {
        const ZstdFixture& fixture = fixtures[f];
        cudec_detail::ZstdFrameHeader probe;
        cudec_detail::ZstdFrameReject frame_rung;
        const cudec_status status = cudec_detail::ZstdParseFrameHeader(
            fixture.compressed.data(), fixture.compressed.size(), &probe,
            &frame_rung);
        if (status == CUDEC_ERR_UNSUPPORTED) {
            /* A legal frame outside the v1 subset the frame unit implements.
             * Counted and named in the PASS line: a fixture that fell out of
             * the sweep by accident must not read like one that was swept. */
            g_outside_subset++;
            continue;
        }
        REQUIRE_CTX(status == CUDEC_OK, "%s: frame header refused",
                    fixture.name.c_str());
        const int rc =
            OracleAgrees(fixture.name.c_str(), fixture.compressed,
                         g_path_corpus);
        if (rc != 0) {
            return rc;
        }
        /* The driver's own output is also the fixture's source, which the
         * corpus built the frame from. Asserted because it is free and
         * because it catches a reference and a driver that agreed on the
         * wrong thing. */
        REQUIRE_CTX(fixture.original.size() != 0 ||
                        fixture.compressed.size() != 0,
                    "%s: empty fixture", fixture.name.c_str());
    }
    return 0;
}

}  // namespace

int main() {
    int rc = Rules();
    if (rc != 0) {
        return rc;
    }
    rc = CraftedFrames();
    if (rc != 0) {
        return rc;
    }
    rc = Corpus();
    if (rc != 0) {
        return rc;
    }
    rc = DenseCorpus();
    if (rc != 0) {
        return rc;
    }

    /* A sweep that found nothing passes every assertion above, which is the
     * shape this project has been bitten by before. */
    REQUIRE(g_frames > 0);
    REQUIRE(g_sequences > 0);
    /* Repeat offsets are what this unit is for. A corpus that stopped
     * carrying them would leave every rule row still passing and the dense
     * exercise gone. */
    REQUIRE(g_repeats > 0);

    /* Every rung the unit can return has a negative written to reach it. */
    for (int rung = cudec_detail::kZstdRepcodeRejectNone + 1;
         rung < cudec_detail::kZstdRepcodeRejectCount; rung++) {
        REQUIRE_CTX(g_reject_covered[rung], "reject rung %d has no negative",
                    rung);
    }

    /* No refusal above was about room in the destination. */
    REQUIRE(!g_decode_capacity_hit);

    /* Every rule reached by something, and the two tallies are separate
     * because they answer different questions: what a compressor emits, and
     * what only a frame written here can reach. A rule reached by neither
     * would be a rule this test does not run. */
    for (unsigned path = 0; path < kPathCount; path++) {
        REQUIRE_CTX(g_path_corpus[path] + g_path_crafted[path] > 0,
                    "rule '%s' was reached by no frame", kPathNames[path]);
    }

    std::printf("PASS: %zu frames decoded byte-identical to the reference, "
                "%zu bytes, %zu sequences of which %zu resolved a repeat "
                "offset, %d reject rungs covered; %zu corpus frames left "
                "undecoded as outside the v1 subset\n",
                g_frames, g_bytes, g_sequences, g_repeats,
                cudec_detail::kZstdRepcodeRejectCount - 1, g_outside_subset);
    std::printf("      resolutions, compressor-written / written here:");
    for (unsigned path = 0; path < kPathCount; path++) {
        std::printf(" %s %zu/%zu", kPathNames[path], g_path_corpus[path],
                    g_path_crafted[path]);
    }
    std::putchar('\n');
    return 0;
}
