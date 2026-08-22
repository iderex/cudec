/* Zstd sequence execution: the prefix sum over (Literal_Length +
 * Match_Length), the literal and match copies it addresses, and the bound that
 * refuses a match reaching outside what the frame has produced (issue #198).
 * Host-side and GPU-less: every input this unit takes is an array a caller
 * built, so nothing here needs a compressor and nothing here needs a device.
 *
 * WHY THIS TWIN LINKS NO ORACLE, WHICH IS DELIBERATE AND IS NOT A GAP. The
 * reference exposes no entry point that executes a sequence stream: the
 * quantities this unit consumes - the tuples, the decoded literals, the
 * resolved distances - exist only inside its own decode loop. What the
 * reference does have an opinion about is a whole frame, and that is where the
 * parity lives: tests/zstd_twin_driver.h calls this unit for every compressed
 * block it decodes, and tests/zstd_entropy_twin.cpp holds the result to
 * libzstd over both corpora in both directions. A row here that passed while
 * the unit was wrong would red there, and the two are meant to be read
 * together.
 *
 * WHAT IS LEFT FOR THIS FILE IS WHAT A CORPUS CANNOT REACH. A compressor emits
 * legal sequence streams, so it never produces the offset one past the bound,
 * the destination array that disagrees with its own sequences, or the block
 * that regenerates past its maximum - and those are the branches that decide
 * whether the unit is fail-closed. They are written here by hand, one per
 * rung, in the shape the sibling Zstd twins use.
 *
 * THE SWEEP COMPARES TWO IMPLEMENTATIONS AND THAT IS WORTH EXACTLY WHAT IT
 * SOUNDS LIKE. The model below is the serial decoder anybody would write: a
 * running cursor, copy the literals, copy the match. It shares no line with
 * the unit, which addresses every copy through the prefix sum instead, so a
 * scan that is off by one diverges from it on the first sequence. It does not
 * share the unit's blind spots either, but it was written by the same hand
 * against the same reading of the format, so it proves the two agree and not
 * that either is right about RFC 8878. What proves that is the frame-level
 * parity named above. */
#include "require.h"
#include "zstd_exec.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using cudec_detail::kZstdExecRejectBadRequest;
using cudec_detail::kZstdExecRejectBlockTooLarge;
using cudec_detail::kZstdExecRejectCount;
using cudec_detail::kZstdExecRejectDestinationTooSmall;
using cudec_detail::kZstdExecRejectLiteralsExhausted;
using cudec_detail::kZstdExecRejectNone;
using cudec_detail::kZstdExecRejectOffsetBeforeOutput;
using cudec_detail::kZstdExecRejectOffsetPastWindow;
using cudec_detail::kZstdExecRejectOffsetZero;
using cudec_detail::kZstdExecRejectPlanInconsistent;
using cudec_detail::ZstdExecPlan;
using cudec_detail::ZstdExecPrefixSum;
using cudec_detail::ZstdExecReject;
using cudec_detail::ZstdExecuteBlock;
using cudec_detail::ZstdSequence;

namespace {

using Bytes = std::vector<unsigned char>;

/* Which reject rungs a declared negative reached. Same discipline as the
 * sibling twins: the enumeration lives once, in the header, and main()
 * requires every rung to have been named by a negative written to reach it. */
bool g_reject_covered[kZstdExecRejectCount] = {false};

void CoverRung(ZstdExecReject rung) {
    if (rung != kZstdExecRejectNone) {
        g_reject_covered[rung] = true;
    }
}

size_t g_rows = 0;
size_t g_sequences = 0;
size_t g_bytes = 0;

/* One block put through both halves of the unit, with everything a row wants
 * to assert on carried back. */
struct Outcome {
    cudec_status status = CUDEC_OK;
    ZstdExecReject rung = kZstdExecRejectNone;
    /* Where it stopped: false while the prefix sum was still running. */
    bool executed = false;
    uint64_t block_size = 0;
    uint64_t literals_used = 0;
    uint64_t produced = 0;
    std::vector<uint64_t> destinations;
};

/* The window a row means when it does not care about the window bound. Larger
 * than any offset written below, so a refusal is never accidentally the window
 * rung. */
constexpr uint64_t kWideWindow = 1ull << 20;

Outcome Execute(const std::vector<ZstdSequence>& sequences,
                const std::vector<uint64_t>& offsets, const Bytes& literals,
                uint64_t block_maximum, uint64_t window_size, Bytes* dst,
                uint64_t produced) {
    Outcome out;
    out.produced = produced;
    out.destinations.assign(sequences.size() + 1, 0);
    ZstdExecPlan plan;
    plan.block_size = 0;
    plan.literals_used = 0;
    out.status = ZstdExecPrefixSum(
        sequences.empty() ? 0 : sequences.data(),
        static_cast<uint32_t>(sequences.size()), literals.size(),
        block_maximum, out.destinations.data(),
        static_cast<uint32_t>(out.destinations.size()), &plan, &out.rung);
    out.block_size = plan.block_size;
    out.literals_used = plan.literals_used;
    if (out.status != CUDEC_OK) {
        return out;
    }
    out.executed = true;
    out.status = ZstdExecuteBlock(
        sequences.empty() ? 0 : sequences.data(),
        static_cast<uint32_t>(sequences.size()), out.destinations.data(),
        offsets.empty() ? 0 : offsets.data(),
        literals.empty() ? 0 : literals.data(), literals.size(), &plan,
        window_size, dst->data(), dst->size(), &out.produced, &out.rung);
    return out;
}

/* ------------------------------------------------------------- the model */

/* The serial decoder the file header describes: a cursor, the literals, the
 * match. It shares no addressing with the unit, and it is what the randomized
 * sweep compares against. It assumes the row is legal - the bounds are the
 * unit's claim, not this model's - so a caller only hands it rows it has
 * already had accepted. */
bool ModelExecute(const std::vector<ZstdSequence>& sequences,
                  const std::vector<uint64_t>& offsets, const Bytes& literals,
                  Bytes* frame) {
    size_t literal_at = 0;
    for (size_t index = 0; index < sequences.size(); index++) {
        const uint32_t literals_length = sequences[index].literals_length;
        if (literal_at + literals_length > literals.size()) {
            return false;
        }
        for (uint32_t i = 0; i < literals_length; i++) {
            frame->push_back(literals[literal_at + i]);
        }
        literal_at += literals_length;
        const uint64_t offset = offsets[index];
        if (offset == 0 || offset > frame->size()) {
            return false;
        }
        const size_t from = frame->size() - static_cast<size_t>(offset);
        for (uint32_t i = 0; i < sequences[index].match_length; i++) {
            frame->push_back((*frame)[from + i]);
        }
    }
    for (size_t i = literal_at; i < literals.size(); i++) {
        frame->push_back(literals[i]);
    }
    return true;
}

/* ----------------------------------------------------------- the helpers */

ZstdSequence Seq(uint32_t literals_length, uint32_t match_length) {
    ZstdSequence sequence;
    sequence.literals_length = literals_length;
    sequence.match_length = match_length;
    /* Not read by this unit: the Offset_Value is src/zstd_repcode.h's input
     * and the resolved distance is what arrives here. Set to a value that is
     * an explicit offset rather than a repeat, so a reader is not invited to
     * believe the two are related. */
    sequence.offset_value = 4;
    return sequence;
}

Bytes Ascii(const char* text) {
    return Bytes(reinterpret_cast<const unsigned char*>(text),
                 reinterpret_cast<const unsigned char*>(text) +
                     std::strlen(text));
}

/* A positive row: the block decodes, and it decodes to exactly these bytes on
 * top of whatever the frame already held. */
int Accepts(const char* name, const std::vector<ZstdSequence>& sequences,
            const std::vector<uint64_t>& offsets, const Bytes& literals,
            const Bytes& prefix, uint64_t window_size, const char* want) {
    const Bytes expected = Ascii(want);
    Bytes dst(prefix.size() + expected.size() + 64, 0);
    /* Guarded on emptiness rather than called with a zero length: an empty
     * vector's data() is a null pointer, and memcpy is declared never to take
     * one however many bytes it is asked for. UBSan says so, and it said so
     * here. */
    if (!prefix.empty()) {
        if (!prefix.empty()) {
            std::memcpy(dst.data(), prefix.data(), prefix.size());
        }
    }
    const Outcome out =
        Execute(sequences, offsets, literals,
                cudec_detail::kZstdBlockSizeCeiling, window_size, &dst,
                prefix.size());
    REQUIRE_CTX(out.status == CUDEC_OK, "%s: refused, rung %d", name,
                static_cast<int>(out.rung));
    REQUIRE_CTX(out.produced == prefix.size() + expected.size(),
                "%s: produced %llu, want %llu", name,
                static_cast<unsigned long long>(out.produced),
                static_cast<unsigned long long>(prefix.size() +
                                                expected.size()));
    REQUIRE_CTX(out.block_size == expected.size(), "%s: block size %llu", name,
                static_cast<unsigned long long>(out.block_size));
    REQUIRE_CTX(equal_bytes(dst.data() + prefix.size(), expected.data(),
                            expected.size()),
                "%s: bytes", name);
    /* The prefix is not touched by a block that writes after it. A unit that
     * addressed from the block's start instead of the frame's would pass every
     * byte assertion above on the first block of a frame and corrupt the one
     * before it on every block after. */
    REQUIRE_CTX(equal_bytes(dst.data(), prefix.data(), prefix.size()),
                "%s: the earlier output moved", name);
    g_rows++;
    g_sequences += sequences.size();
    g_bytes += expected.size();
    return 0;
}

/* A negative row: the unit refuses, at this rung, with this status. */
int Refuses(const char* name, const std::vector<ZstdSequence>& sequences,
            const std::vector<uint64_t>& offsets, const Bytes& literals,
            uint64_t block_maximum, uint64_t window_size, uint64_t capacity,
            uint64_t produced, ZstdExecReject want_rung,
            cudec_status want_status) {
    Bytes dst(static_cast<size_t>(capacity), 0);
    const Outcome out = Execute(sequences, offsets, literals, block_maximum,
                                window_size, &dst, produced);
    REQUIRE_CTX(out.status == want_status, "%s: status %d, want %d", name,
                static_cast<int>(out.status), static_cast<int>(want_status));
    REQUIRE_CTX(out.rung == want_rung, "%s: rung %d, want %d", name,
                static_cast<int>(out.rung), static_cast<int>(want_rung));
    /* A refusal leaves the caller's byte count where it was: a partial write
     * is not a partial success, and a caller that advanced on one would
     * present bytes it has no reason to believe. */
    REQUIRE_CTX(out.produced == produced, "%s: produced moved to %llu", name,
                static_cast<unsigned long long>(out.produced));
    CoverRung(out.rung);
    g_rows++;
    return 0;
}

/* ------------------------------------------------------------- the rows */

int Positives() {
    /* A block with no sequences at all decodes to its literals. The tail is
     * the whole block, and the prefix sum runs zero times, so this is the one
     * row where every destination the copies use comes from the
     * one-past-the-end entry alone. */
    if (Accepts("literals-only", {}, {}, Ascii("abcdef"), Bytes(), kWideWindow,
                "abcdef") != 0) {
        return 1;
    }

    /* One sequence and a tail: three literals, a match four back over three
     * bytes, then the literals nobody claimed. */
    if (Accepts("one-sequence", {Seq(3, 3)}, {3}, Ascii("abcdef"), Bytes(),
                kWideWindow, "abcabcdef") != 0) {
        return 1;
    }

    /* Offset one over a long match: the run-fill. The source runs into the
     * bytes the match is writing, so a copy that read the range once - a
     * vector load, a memcpy - would produce different bytes rather than the
     * same bytes faster. */
    if (Accepts("run-fill-offset-one", {Seq(1, 9)}, {1}, Ascii("a"), Bytes(),
                kWideWindow, "aaaaaaaaaa") != 0) {
        return 1;
    }

    /* An overlap that is not a run: offset two over five bytes alternates. */
    if (Accepts("overlap-offset-two", {Seq(2, 5)}, {2}, Ascii("ab"), Bytes(),
                kWideWindow, "abababa") != 0) {
        return 1;
    }

    /* A match reaching into what an earlier block produced. This is the case
     * the format is built around - it is what makes the blocks of a frame
     * sequential - and a decoder bounding an offset against the block instead
     * of the frame refuses it. */
    if (Accepts("cross-block-match", {Seq(1, 5)}, {12}, Ascii("!"),
                Ascii("hello world"), kWideWindow, "!hello") != 0) {
        return 1;
    }

    /* The offset exactly at the bound: the match starts at position four of
     * the frame and reaches back four, to the first byte there is. One less
     * and the read is outside the output. */
    if (Accepts("offset-at-the-output-bound", {Seq(4, 4)}, {4}, Ascii("wxyz"),
                Bytes(), kWideWindow, "wxyzwxyz") != 0) {
        return 1;
    }

    /* The offset exactly at the window, which is a different bound and is
     * checked separately. Eight bytes of earlier output, a window of eight, a
     * match reaching all eight back. */
    if (Accepts("offset-at-the-window-bound", {Seq(0, 3)}, {8}, Bytes(),
                Ascii("01234567"), 8, "012") != 0) {
        return 1;
    }

    /* Several sequences and a tail, so the prefix sum has more than one entry
     * to get right. Destinations 0, 5 and 9; the tail at 12. */
    if (Accepts("multi-sequence",
                {Seq(2, 3), Seq(1, 3), Seq(1, 2)}, {2, 5, 1},
                Ascii("abcdXY"), Bytes(), kWideWindow, "ababacbabdddXY") != 0) {
        return 1;
    }

    /* A sequence with no literals at all: the match starts where the previous
     * one ended. */
    if (Accepts("zero-literal-sequence", {Seq(3, 2), Seq(0, 2)}, {3, 5},
                Ascii("xyz"), Bytes(), kWideWindow, "xyzxyxy") != 0) {
        return 1;
    }

    /* A block that regenerates exactly its maximum is legal; one byte more is
     * the negative below. Written with the ceiling handed in explicitly rather
     * than through Accepts(), because the row is about that number. */
    {
        const std::vector<ZstdSequence> sequences = {Seq(4, 4)};
        const std::vector<uint64_t> offsets = {4};
        Bytes dst(64, 0);
        uint64_t produced = 0;
        const Outcome out = Execute(sequences, offsets, Ascii("wxyz"), 8,
                                    kWideWindow, &dst, produced);
        REQUIRE(out.status == CUDEC_OK);
        REQUIRE(out.block_size == 8);
        REQUIRE(out.produced == 8);
        g_rows++;
    }
    return 0;
}

int Negatives() {
    const Bytes literals = Ascii("abcdef");

    /* The destinations array with no room for the one-past-the-end entry. The
     * entry is not a convenience: it is what the execution holds the last
     * sequence's own arithmetic against, and what tells the tail where it
     * starts. */
    {
        std::vector<ZstdSequence> sequences = {Seq(1, 1)};
        uint64_t destinations[1] = {0};
        ZstdExecPlan plan;
        ZstdExecReject rung = kZstdExecRejectNone;
        const cudec_status status = ZstdExecPrefixSum(
            sequences.data(), 1, literals.size(),
            cudec_detail::kZstdBlockSizeCeiling, destinations, 1, &plan,
            &rung);
        REQUIRE(status == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(rung == kZstdExecRejectBadRequest);
        CoverRung(rung);
        g_rows++;
    }

    /* A ceiling past the format's own. Refused rather than used, because every
     * sum in the unit is bounded by it and a ceiling nobody bounded is a bound
     * that proves nothing. */
    if (Refuses("ceiling-past-the-format", {Seq(1, 1)}, {1}, literals,
                cudec_detail::kZstdBlockSizeCeiling + 1, kWideWindow, 256, 0,
                kZstdExecRejectBadRequest, CUDEC_ERR_INVALID_ARGUMENT) != 0) {
        return 1;
    }

    /* More literal bytes claimed than the literals section produced. */
    if (Refuses("literals-exhausted", {Seq(7, 1)}, {1}, literals,
                cudec_detail::kZstdBlockSizeCeiling, kWideWindow, 256, 0,
                kZstdExecRejectLiteralsExhausted, CUDEC_ERR_CORRUPT_INPUT) !=
        0) {
        return 1;
    }

    /* The literals run out on the SECOND sequence, so the row also says the
     * accounting is running rather than per sequence. */
    if (Refuses("literals-exhausted-later", {Seq(4, 1), Seq(4, 1)}, {1, 1},
                literals, cudec_detail::kZstdBlockSizeCeiling, kWideWindow,
                256, 0, kZstdExecRejectLiteralsExhausted,
                CUDEC_ERR_CORRUPT_INPUT) != 0) {
        return 1;
    }

    /* A block one byte past its maximum. */
    if (Refuses("block-past-its-maximum", {Seq(4, 5)}, {4}, Ascii("wxyz"), 8,
                kWideWindow, 256, 0, kZstdExecRejectBlockTooLarge,
                CUDEC_ERR_CORRUPT_INPUT) != 0) {
        return 1;
    }

    /* The tail alone carries a block past its maximum, which the per-sequence
     * bound above cannot see. */
    if (Refuses("tail-past-the-maximum", {Seq(2, 2)}, {2}, literals, 5,
                kWideWindow, 256, 0, kZstdExecRejectBlockTooLarge,
                CUDEC_ERR_CORRUPT_INPUT) != 0) {
        return 1;
    }

    /* A resolved offset of zero. src/zstd_repcode.h refuses one already; this
     * is the offset array that did not come from it. */
    if (Refuses("offset-zero", {Seq(2, 2)}, {0}, literals,
                cudec_detail::kZstdBlockSizeCeiling, kWideWindow, 256, 0,
                kZstdExecRejectOffsetZero, CUDEC_ERR_CORRUPT_INPUT) != 0) {
        return 1;
    }

    /* One past the window, with enough output behind it that the other bound
     * would have accepted it. The two rungs are only distinguishable on a row
     * where exactly one of them bites. */
    if (Refuses("offset-past-the-window", {Seq(0, 2)}, {9}, Bytes(),
                cudec_detail::kZstdBlockSizeCeiling, 8, 256, 32,
                kZstdExecRejectOffsetPastWindow, CUDEC_ERR_CORRUPT_INPUT) !=
        0) {
        return 1;
    }

    /* One past the output, with a window wide enough that the other bound
     * would have accepted it: the case a decoder checking only the window
     * reads outside its own buffer on, which is every frame near its start. */
    if (Refuses("offset-before-the-output", {Seq(3, 2)}, {4}, literals,
                cudec_detail::kZstdBlockSizeCeiling, kWideWindow, 256, 0,
                kZstdExecRejectOffsetBeforeOutput, CUDEC_ERR_CORRUPT_INPUT) !=
        0) {
        return 1;
    }

    /* The destination does not hold the frame's output plus this block. Its
     * status is OUTPUT_TOO_SMALL and not CORRUPT_INPUT: the bytes are good and
     * the buffer is short, and a caller that cannot tell those apart cannot
     * retry with a larger one. */
    if (Refuses("destination-too-small", {Seq(3, 3)}, {3}, literals,
                cudec_detail::kZstdBlockSizeCeiling, kWideWindow, 8, 0,
                kZstdExecRejectDestinationTooSmall,
                CUDEC_ERR_OUTPUT_TOO_SMALL) != 0) {
        return 1;
    }

    /* The destinations array that disagrees with the sequences that write to
     * it. Reached by a scan that is wrong rather than by a stream, which is
     * why it is built here by hand: the prefix sum would never produce it, and
     * the execution is what has to refuse rather than believe it. Three
     * shapes, because they fail in three different places. */
    {
        const std::vector<ZstdSequence> sequences = {Seq(2, 2), Seq(1, 1)};
        const std::vector<uint64_t> offsets = {2, 3};
        ZstdExecPlan plan;
        ZstdExecReject rung = kZstdExecRejectNone;
        std::vector<uint64_t> good(3, 0);
        REQUIRE(ZstdExecPrefixSum(sequences.data(), 2, literals.size(),
                                  cudec_detail::kZstdBlockSizeCeiling,
                                  good.data(), 3, &plan,
                                  &rung) == CUDEC_OK);

        struct Row {
            const char* name;
            unsigned entry;
            uint64_t value;
        };
        const Row rows[] = {
            /* A successor one short: the scan dropped a byte. */
            {"plan-successor-short", 1, 3},
            /* A successor one long: the scan double-counted. */
            {"plan-successor-long", 1, 5},
            /* The last sequence's successor IS the tail's start, so a tail
             * that moved on a block with sequences is caught by the same
             * comparison. The case only that equation sees is the block with
             * none, and it is the row after this loop. */
            {"plan-tail-moved", 2, 7},
        };
        for (const Row& row : rows) {
            std::vector<uint64_t> broken = good;
            broken[row.entry] = row.value;
            Bytes dst(256, 0);
            uint64_t produced = 0;
            rung = kZstdExecRejectNone;
            const cudec_status status = ZstdExecuteBlock(
                sequences.data(), 2, broken.data(), offsets.data(),
                literals.data(), literals.size(), &plan, kWideWindow,
                dst.data(), dst.size(), &produced, &rung);
            REQUIRE_CTX(status == CUDEC_ERR_CORRUPT_INPUT, "%s: status %d",
                        row.name, static_cast<int>(status));
            REQUIRE_CTX(rung == kZstdExecRejectPlanInconsistent,
                        "%s: rung %d", row.name, static_cast<int>(rung));
            REQUIRE_CTX(produced == 0, "%s: produced moved", row.name);
            CoverRung(rung);
            g_rows++;
        }

        /* The literals bound inside the execution, reached by calling it
         * directly with a plan and a destinations array that agree with each
         * other and not with the literals buffer. The prefix sum refuses this
         * one sequence earlier, so nothing that goes through both halves ever
         * arrives here - and this is the guard between a caller that computed
         * its plan somewhere else and a read outside the literals it was
         * handed. The refusal has to come BEFORE the copy, which is why the
         * row exists rather than a comment saying the sum already checked. */
        {
            const std::vector<ZstdSequence> lying_sequences = {Seq(4, 1)};
            const std::vector<uint64_t> lying_offsets = {1};
            const Bytes short_literals = Ascii("ab");
            const uint64_t lying_destinations[2] = {0, 5};
            ZstdExecPlan lying_plan;
            lying_plan.block_size = 5;
            lying_plan.literals_used = 4;
            Bytes short_dst(256, 0);
            uint64_t short_produced = 0;
            ZstdExecReject short_rung = kZstdExecRejectNone;
            const cudec_status short_status = ZstdExecuteBlock(
                lying_sequences.data(), 1, lying_destinations,
                lying_offsets.data(), short_literals.data(),
                short_literals.size(), &lying_plan, kWideWindow,
                short_dst.data(), short_dst.size(), &short_produced,
                &short_rung);
            REQUIRE(short_status == CUDEC_ERR_CORRUPT_INPUT);
            REQUIRE(short_rung == kZstdExecRejectLiteralsExhausted);
            REQUIRE(short_produced == 0);
            CoverRung(short_rung);
            g_rows++;
        }

        /* A block with NO sequences, whose tail was moved. The per-sequence
         * comparison does not run at all here, so the equation at the end is
         * the only thing between a destination somebody else computed and a
         * write outside the block it was bounded against. */
        {
            std::vector<uint64_t> tail_moved(1, 3);
            ZstdExecPlan empty_plan;
            ZstdExecReject empty_rung = kZstdExecRejectNone;
            std::vector<uint64_t> empty_destinations(1, 0);
            REQUIRE(ZstdExecPrefixSum(0, 0, literals.size(),
                                      cudec_detail::kZstdBlockSizeCeiling,
                                      empty_destinations.data(), 1,
                                      &empty_plan, &empty_rung) == CUDEC_OK);
            Bytes empty_dst(256, 0);
            uint64_t empty_produced = 0;
            const cudec_status empty_status = ZstdExecuteBlock(
                0, 0, tail_moved.data(), 0, literals.data(), literals.size(),
                &empty_plan, kWideWindow, empty_dst.data(), empty_dst.size(),
                &empty_produced, &empty_rung);
            REQUIRE(empty_status == CUDEC_ERR_CORRUPT_INPUT);
            REQUIRE(empty_rung == kZstdExecRejectPlanInconsistent);
            REQUIRE(empty_produced == 0);
            CoverRung(empty_rung);
            g_rows++;
        }

        /* A plan whose literals_used disagrees with what the sequences
         * consume. The per-sequence comparison holds destinations against the
         * SUM of two lengths and cannot tell a literal byte from a match one,
         * so this survives it - and a tail sized from that number would run
         * past the block. */
        ZstdExecPlan lying = plan;
        lying.literals_used = plan.literals_used + 1;
        Bytes dst(256, 0);
        uint64_t produced = 0;
        rung = kZstdExecRejectNone;
        const cudec_status status = ZstdExecuteBlock(
            sequences.data(), 2, good.data(), offsets.data(), literals.data(),
            literals.size(), &lying, kWideWindow, dst.data(), dst.size(),
            &produced, &rung);
        REQUIRE(status == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(rung == kZstdExecRejectPlanInconsistent);
        CoverRung(rung);
        g_rows++;
    }
    return 0;
}

/* ------------------------------------------------------------- the sweep */

/* A deterministic generator, so a failure is reproducible from the seed
 * printed with it. xorshift64 rather than the standard library's engines:
 * this file states the sequence it draws instead of depending on an
 * implementation's. */
uint64_t Next(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int Sweep() {
    constexpr int kRuns = 512;
    size_t total_sequences = 0;
    for (int run = 0; run < kRuns; run++) {
        uint64_t state = 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(run);
        /* A prefix that plays the part of earlier blocks, so a sweep row
         * reaches the cross-block case rather than only the first block. */
        Bytes frame;
        const size_t prefix_size = static_cast<size_t>(Next(&state) % 40);
        for (size_t i = 0; i < prefix_size; i++) {
            frame.push_back(static_cast<unsigned char>(Next(&state)));
        }
        const Bytes prefix = frame;

        Bytes literals;
        const size_t literals_size =
            static_cast<size_t>(Next(&state) % 48) + 1;
        for (size_t i = 0; i < literals_size; i++) {
            literals.push_back(
                static_cast<unsigned char>('a' + (Next(&state) % 26)));
        }

        std::vector<ZstdSequence> sequences;
        std::vector<uint64_t> offsets;
        size_t literal_at = 0;
        size_t produced = prefix.size();
        const size_t count = static_cast<size_t>(Next(&state) % 6);
        for (size_t s = 0; s < count; s++) {
            const size_t left = literals.size() - literal_at;
            if (left == 0) {
                break;
            }
            const uint32_t literals_length =
                static_cast<uint32_t>(Next(&state) % (left + 1));
            produced += literals_length;
            literal_at += literals_length;
            if (produced == 0) {
                /* Nothing to match against yet: an offset needs a byte behind
                 * it, and this row would be the negative rather than the
                 * sweep. */
                continue;
            }
            const uint64_t offset = (Next(&state) % produced) + 1;
            const uint32_t match_length =
                static_cast<uint32_t>(Next(&state) % 40) + 1;
            produced += match_length;
            sequences.push_back(Seq(literals_length, match_length));
            offsets.push_back(offset);
        }

        Bytes expected = prefix;
        REQUIRE_CTX(ModelExecute(sequences, offsets, literals, &expected),
                    "run %d: the model refused its own row", run);

        Bytes dst(expected.size() + 64, 0);
        std::memcpy(dst.data(), prefix.data(), prefix.size());
        const Outcome out =
            Execute(sequences, offsets, literals,
                    cudec_detail::kZstdBlockSizeCeiling, kWideWindow, &dst,
                    prefix.size());
        REQUIRE_CTX(out.status == CUDEC_OK, "run %d: refused, rung %d", run,
                    static_cast<int>(out.rung));
        REQUIRE_CTX(out.produced == expected.size(),
                    "run %d: produced %llu, model %zu", run,
                    static_cast<unsigned long long>(out.produced),
                    expected.size());
        REQUIRE_CTX(equal_bytes(dst.data(), expected.data(), expected.size()),
                    "run %d: bytes", run);

        /* The destinations the unit produced, checked against the model's own
         * cursor. A scan that is wrong in a way the bytes happen to survive -
         * two sequences whose lengths swap - is caught here and nowhere
         * else. */
        size_t cursor = 0;
        for (size_t s = 0; s < sequences.size(); s++) {
            REQUIRE_CTX(out.destinations[s] == cursor,
                        "run %d: destination %zu is %llu, want %zu", run, s,
                        static_cast<unsigned long long>(out.destinations[s]),
                        cursor);
            cursor += sequences[s].literals_length + sequences[s].match_length;
        }
        REQUIRE_CTX(out.destinations[sequences.size()] == cursor,
                    "run %d: the tail starts at %llu, want %zu", run,
                    static_cast<unsigned long long>(
                        out.destinations[sequences.size()]),
                    cursor);
        total_sequences += sequences.size();
        g_rows++;
        g_sequences += sequences.size();
        g_bytes += out.block_size;
    }
    /* A sweep that generated nothing passes every assertion in it. */
    REQUIRE(total_sequences > 0);
    std::printf("      sweep: %d rows, %zu sequences\n", kRuns,
                total_sequences);
    return 0;
}

}  // namespace

int main() {
    if (Positives() != 0) {
        return 1;
    }
    if (Negatives() != 0) {
        return 1;
    }
    if (Sweep() != 0) {
        return 1;
    }

    /* Every rung the unit can return has a negative written to reach it. */
    for (int rung = kZstdExecRejectNone + 1; rung < kZstdExecRejectCount;
         rung++) {
        REQUIRE_CTX(g_reject_covered[rung], "reject rung %d has no negative",
                    rung);
    }

    std::printf("PASS: %zu rows, %zu sequences executed, %zu bytes "
                "regenerated, %d reject rungs covered\n",
                g_rows, g_sequences, g_bytes, kZstdExecRejectCount - 1);
    return 0;
}
