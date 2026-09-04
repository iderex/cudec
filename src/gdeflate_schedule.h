/* The GDeflate round/refill schedule (issue #178): which of the 32 lanes
 * consumes which 32-bit word of a page, in which round. Single-sourced for
 * host and device, the sibling of src/lz4_block.h, src/snappy_block.h and
 * src/zstd_frame.h. It decodes no symbol and builds no table; it owns the bit
 * cursor and nothing else.
 *
 * WHY THIS IS A SEPARATE THING FROM THE DECODER. GDeflate carries no checksum
 * anywhere (MASTERPLAN section 11.4) and its 32 substreams have no markers: a
 * boundary is not self-delimiting, and the only thing that says where lane 7's
 * next word is, is the schedule that put it there. So an off-by-one here is
 * not a crash, it is silent corruption that no downstream check catches.
 * Isolating the schedule is what makes it testable before a table or a kernel
 * exists, which is the reason the M4 rungs are cut in this order (#178, then
 * #176, then #182).
 *
 * WHAT A ROUND IS. One shared word cursor walks the page's 32-bit words in
 * increasing order. Advance is the only thing that moves the lane index, and
 * it refills the CURRENT lane before rotating, not after - so lane L's refill
 * is the word that sits at the cursor at the moment lane L is current, and the
 * lane order is 0, 1, ... 31, 0, 1, ... with no exception anywhere in the
 * format. Reset returns to lane 0 without consuming a word: every block header
 * rides lane 0 (dossier 11.2), and the header round is that reset.
 *
 * THE PRIMING ROUND IS PART OF THE FORMAT, NOT AN IMPLEMENTATION CHOICE. A
 * page opens with one full pass that loads a word into each of the 32 lanes in
 * lane order, which is why a valid GDeflate stream cannot be smaller than
 * 4 * 32 = 128 bytes (draft section 5.3, pinned in tests/gdeflate_probes.cpp).
 * GDeflateInit performs it, and a page too short to carry it is refused there
 * rather than half-primed.
 *
 * FAIL-CLOSED, AND STRICTER THAN THE REFERENCE IN TWO PLACES THAT ARE STATED
 * RATHER THAN LEFT TO BE FOUND. The reference refills without checking that a
 * word is present and pops without checking that bits are present, because the
 * format's own watermark discipline makes both true on a well-formed page.
 * Hostile input is the expected case here, so a refill past the last word and
 * a pop wider than the lane holds each set the sticky failure flag and consume
 * nothing. Both are refusals a valid page cannot reach: after the priming
 * round every lane holds exactly 32 bits, and every Advance restores any lane
 * that fell below the 32-bit watermark, so a page that trips either one is
 * malformed. Nothing here ever sizes anything from the stream.
 *
 * THE HOST SHAPE AND THE DEVICE SHAPE ARE THE SAME FUNCTION, NOT THE SAME
 * LAYOUT. GDeflateSchedule holds all 32 lane buffers because the CPU twin runs
 * the lanes sequentially in one thread. In the warp kernel each lane owns its
 * own bit buffer and occupancy in registers while the cursor is warp-uniform,
 * so the kernel instantiates the same arithmetic over a different residency
 * (issue #214). That is why every primitive below is a template over the
 * residency `S` rather than a function of GDeflateSchedule: `S` supplies the
 * current lane's buffer and occupancy through Buf() and Left(), the sticky
 * verdict through `failed` and `reject`, and the page's word count through
 * `word_count`, and the arithmetic is written once. What must not fork is the
 * schedule itself, which is why it lives in one place. The cursor is 64-bit
 * for the reason the other parsers give: the caller's sizes are size_t, and
 * mixing widths in a bound comparison is where an overflow hides.
 *
 * WHERE A RUNG IS NAMED. Each refusal below is named at exactly one site, and
 * the sites that both residencies reach - the width checks of a read, the
 * occupancy check of a remove, the bound of a refill, the shape of a page -
 * name their rung into a caller's slot the way the table construction in
 * src/gdeflate_tables.h does, so that a sequential Pop and a per-lane read on
 * the device raise the same branch from the same line. */
#ifndef CUDEC_GDEFLATE_SCHEDULE_H
#define CUDEC_GDEFLATE_SCHEDULE_H

#include <stdint.h>

/* Guarded: the other single-sourced parsers define the same macro for the same
 * reason, and a device translation unit that decodes two formats includes two
 * of these headers. The definitions are identical, so an unguarded second one
 * is legal rather than an error - which is exactly why it would go unnoticed
 * if they ever stopped being identical. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIP__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* All three are the reference's own constants (deflate_constants.h in the
 * pinned NVIDIA/libdeflate fork), restated here because this header is the
 * thing that has to agree with them. */
constexpr uint32_t kGDeflateNumStreams = 32;
constexpr uint32_t kGDeflateBitsPerPacket = 32;
constexpr uint32_t kGDeflateLowWatermarkBits = 32;

/* The width of one lane's bit buffer, named rather than left as the width of
 * whatever type the struct below happens to use: every shift in this header is
 * safe because of this number, so it is the thing the assertions compare
 * against. */
constexpr uint32_t kGDeflateLaneBufferBits = 64;

/* A refill only fires below the watermark, so a lane holds at most
 * (watermark - 1) + packet bits at any instant. That is the bound the shift in
 * the refill is safe under: shifting one packet left by up to a watermark less
 * one place stays inside the buffer. */
constexpr uint32_t kGDeflateMaxLaneBits =
    kGDeflateLowWatermarkBits - 1 + kGDeflateBitsPerPacket;
static_assert(kGDeflateMaxLaneBits < kGDeflateLaneBufferBits,
              "a refill at the watermark must not shift a packet off the top "
              "of the lane bit buffer");
static_assert(sizeof(uint64_t) * 8 == kGDeflateLaneBufferBits,
              "the lane bit buffer type must be as wide as the constant the "
              "shift bounds are argued against");

/* The widest single read the format asks for is the 16-bit stored-block LEN
 * (dossier 11.3). The ceiling here is not that number: it is the watermark,
 * because a read wider than a lane sitting exactly at the watermark could not
 * be satisfied on a well-formed page, which would make it a caller bug rather
 * than bad input. */
constexpr uint32_t kGDeflateMaxPopBits = kGDeflateLowWatermarkBits;

/* THE REJECT LADDER (issue #183). Every refusal in these three headers names
 * one branch of this enumeration, and each branch is named by exactly one
 * site, so the ladder's branch set is DERIVED from the code rather than
 * restated beside it. tests/CMakeLists.txt refuses a build where that stops
 * being true, and the twins hold the run-time half: a negative has to reach
 * every branch that any page can reach, and a branch declared unreachable has
 * to stay unreached. Adding a rung therefore costs a branch here and a
 * negative there, and a rung that reuses a neighbour's branch to avoid both is
 * what the exactly-one count refuses.
 *
 * ONE ENUMERATION FOR THREE HEADERS, because one sticky field carries it: a
 * page decode fails once, in whichever of the three the defect was found, and
 * a caller reading the verdict should not have to ask which layer to ask.
 * The sections below are the layers, and the sentinels between them are what
 * lets each twin assert coverage over its own header rather than over all
 * three. A sentinel carries an explicit value; a branch never does, which is
 * how the configure-time lock tells them apart without a list to maintain. */
enum GDeflateReject {
    kGDeflateRejectNone = 0,

    /* src/gdeflate_schedule.h - the bit cursor. */
    kGDeflateRejectPagePartialWord,
    kGDeflateRejectPageBelowPrimingRound,
    kGDeflateRejectRefillPastEnd,
    kGDeflateRejectRemovePastLane,
    kGDeflateRejectPopWidthPastFormat,
    kGDeflateRejectPopPastLane,
    kGDeflateRejectScheduleFirst = kGDeflateRejectPagePartialWord,
    kGDeflateRejectScheduleLast = kGDeflateRejectPopPastLane,

    /* src/gdeflate_tables.h - table construction and symbol decode. */
    kGDeflateRejectTablePastCapacity,
    kGDeflateRejectTableLengthPastMax,
    kGDeflateRejectTableOverSubscribed,
    kGDeflateRejectTableIncomplete,
    kGDeflateRejectEmptyTableUsed,
    kGDeflateRejectCodewordNotInCode,
    kGDeflateRejectRepeatNothingBefore,
    kGDeflateRejectRepeatRunPastAlphabet,
    kGDeflateRejectTablesFirst = kGDeflateRejectTablePastCapacity,
    kGDeflateRejectTablesLast = kGDeflateRejectRepeatRunPastAlphabet,

    /* src/gdeflate_block.h - the block loop. */
    kGDeflateRejectBlockTypeReserved,
    kGDeflateRejectStoredPastCap,
    kGDeflateRejectStoredPastPage,
    kGDeflateRejectLiteralPastCap,
    kGDeflateRejectMatchPastCap,
    kGDeflateRejectMatchBeforeOutput,
    kGDeflateRejectRoundFuelExhausted,
    kGDeflateRejectNoFinalBlock,
    kGDeflateRejectBlockFirst = kGDeflateRejectBlockTypeReserved,
    kGDeflateRejectBlockLast = kGDeflateRejectNoFinalBlock,

    kGDeflateRejectCount
};

/* The sections have to stay contiguous and in this order, because each twin
 * walks its own First..Last range. A branch inserted into the wrong section
 * would be asserted by the wrong twin, which is a coverage hole that reads as
 * coverage. */
static_assert(kGDeflateRejectScheduleFirst == kGDeflateRejectNone + 1,
              "the schedule section must start immediately after None");
static_assert(kGDeflateRejectTablesFirst == kGDeflateRejectScheduleLast + 1,
              "the table section must follow the schedule section");
static_assert(kGDeflateRejectBlockFirst == kGDeflateRejectTablesLast + 1,
              "the block section must follow the table section");
static_assert(kGDeflateRejectCount == kGDeflateRejectBlockLast + 1,
              "the block section must be the last one");

/* THE DECLARED STRICTNESS DEPARTURES (issue #183). Three rungs refuse a page
 * that libdeflate's GDeflate decompressor decodes, and each one is a decision
 * argued at its own refusal site rather than an accident:
 *
 *   - kGDeflateRejectRefillPastEnd, at GDeflateRefillLane below: the reference
 *     reads a word past the last one the page holds, which the format's
 *     watermark discipline makes safe on a well-formed page and nothing makes
 *     safe on a hostile one.
 *   - kGDeflateRejectEmptyTableUsed, in src/gdeflate_tables.h: the reference
 *     resolves a use of an empty code to a synthetic symbol the stream never
 *     encoded.
 *   - kGDeflateRejectRepeatRunPastAlphabet, in src/gdeflate_tables.h: the
 *     reference absorbs a code-length repeat run that overruns HLIT + HDIST
 *     into slack entries it then never reads.
 *
 * WHY THE LIST IS A PREDICATE AND NOT A SENTENCE IN A COMMENT. For a format
 * with no checksum anywhere, over-strictness is not a lesser cousin of
 * fail-open: a stream the reference decompresses and this decoder calls
 * corrupt is data the caller cannot recover and cannot appeal, because there
 * is no checksum to say which of the two decoders is right. A twin that is
 * never measured in this direction drifts stricter one refusal at a time, and
 * every step of that drift reads as a bug fix. So the differential target
 * traps the reverse direction and consults this predicate for the exemption,
 * which makes a fourth departure a thing somebody writes down rather than a
 * thing a later reader discovers by finding data that will not decompress.
 *
 * WHAT HOLDS THE LIST TO REALITY IS tests/gdeflate_departure_lock.cpp, which
 * requires a page the reference accepts and this decoder refuses on that rung
 * for every branch named here. A rung added to this predicate to silence a
 * trap therefore fails a test rather than buying silence, and a departure that
 * stops being one fails it too. */
CUDEC_HOST_DEVICE inline bool GDeflateRejectIsDeclaredDeparture(
    GDeflateReject branch) {
    switch (branch) {
        case kGDeflateRejectRefillPastEnd:
        case kGDeflateRejectEmptyTableUsed:
        case kGDeflateRejectRepeatRunPastAlphabet:
            return true;
        default:
            return false;
    }
}

/* The 32 lane bit buffers, the current lane, and the one shared word cursor.
 * The failure flag is sticky: once set, every operation is a no-op, so a
 * caller that checks it once at the end reads the same verdict as one that
 * checks after every call.
 *
 * This is the HOST residency of the schedule: one thread, every lane. Buf()
 * and Left() are what the primitives below are written against, and here they
 * select the current lane; the device residency in src/gdeflate_decode.cuh
 * answers them with the one lane the calling thread owns. */
struct GDeflateSchedule {
    uint64_t bitbuf[kGDeflateNumStreams];
    uint32_t bitsleft[kGDeflateNumStreams];
    uint32_t idx;
    uint64_t cursor;
    uint64_t word_count;
    bool failed;
    /* Which rung refused, and it is the reason the flag alone is not enough:
     * a caller that only knows the page failed cannot tell a page that ran off
     * its own end from one whose code lengths are not a code, and neither can
     * a test asserting that its negative reached the branch it was written
     * for. Set only through the choke point below. */
    GDeflateReject reject;

    CUDEC_HOST_DEVICE uint64_t& Buf() { return bitbuf[idx]; }
    CUDEC_HOST_DEVICE const uint64_t& Buf() const { return bitbuf[idx]; }
    CUDEC_HOST_DEVICE uint32_t& Left() { return bitsleft[idx]; }
    CUDEC_HOST_DEVICE const uint32_t& Left() const { return bitsleft[idx]; }
};

/* THE ONE PLACE A REFUSAL IS RECORDED, for the reason SnappyRefuse is one in
 * src/snappy_block.h: a refusal that never passes through here names no rung,
 * so it is invisible to the configure-time count and to the twins' coverage
 * alike, and adding one would be the one change that could grow the ladder
 * with both halves of the lock still green. It answers false so that a bool
 * caller can `return GDeflateRefuse(...)` and a void one can call it and
 * return.
 *
 * FIRST REFUSAL WINS. The flag is sticky and every operation is a no-op once
 * it is set, so a later call could only overwrite the branch with a
 * consequence of the first one. */
template <class S>
CUDEC_HOST_DEVICE inline bool GDeflateRefuse(S& s, GDeflateReject branch) {
    if (!s.failed) {
        s.failed = true;
        s.reject = branch;
    }
    return false;
}

/* Re-raise a refusal a nested ladder already named. Table construction runs
 * without a schedule - it is called on a length vector and nothing else - so
 * it names its rung into a caller-supplied slot, and the caller that owns the
 * schedule raises it here. Kept as its own verb rather than as a GDeflateRefuse
 * call with a variable argument, because the configure-time lock reads the
 * branch off the call site: a site passing a variable would have to be
 * exempted from that read, and an exemption spelled the same way as the rule
 * is how the rule stops being read. */
template <class S>
CUDEC_HOST_DEVICE inline bool GDeflateRefuseAs(S& s,
                                               GDeflateReject already_named) {
    if (!s.failed) {
        s.failed = true;
        s.reject = already_named;
    }
    return false;
}

/* The same choke point for a ladder with no schedule to record into. `slot`
 * is the caller's, and a null one is a caller that does not want the reason -
 * the refusal still happens, which is what makes this safe to call from a
 * table build that a test drives directly. */
CUDEC_HOST_DEVICE inline bool GDeflateTableRefuse(GDeflateReject* slot,
                                                  GDeflateReject branch) {
    if (slot != nullptr) {
        *slot = branch;
    }
    return false;
}

/* Little-endian, byte by byte, for the reason every other parser in this tree
 * reads its integers this way: the page is a byte buffer with no alignment
 * guarantee, and a cast through a uint32_t pointer would be undefined on the
 * host and differently undefined on the device. */
CUDEC_HOST_DEVICE inline uint32_t GDeflateWordAt(const unsigned char* page,
                                                 uint64_t word_index) {
    const unsigned char* p = page + word_index * 4u;
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

/* A READ AHEAD OF THE LANE, WITHOUT CONSUMING. `consumed` is how many bits the
 * caller has already read ahead and not yet removed, so a caller can resolve a
 * codeword and the extra bits behind it as one speculative read and then
 * remove the whole width at once - or remove nothing, which is how the warp
 * kernel discards the lanes that must not have consumed after an end-of-block
 * (docs/MASTERPLAN.md section 13.2). `why` is the rung the read refused on, and
 * a read past a refusal returns zero and reads nothing more.
 *
 * The two refusals here are the two GDeflatePop always carried, in the same
 * order: a width past the format's widest field is a caller asking for
 * something no page encodes, while a lane that cannot serve a legal width is a
 * page that ran out of bits where the schedule says it should not have. A
 * single rung would let a negative for either satisfy the other. */
struct GDeflateLaneRead {
    uint32_t consumed;
    GDeflateReject why;
};

CUDEC_HOST_DEVICE inline uint32_t GDeflateReadAhead(uint64_t bitbuf,
                                                    uint32_t bitsleft,
                                                    GDeflateLaneRead& r,
                                                    uint32_t n) {
    if (r.why != kGDeflateRejectNone) {
        return 0;
    }
    if (n > kGDeflateMaxPopBits) {
        GDeflateTableRefuse(&r.why, kGDeflateRejectPopWidthPastFormat);
        return 0;
    }
    if (n > bitsleft - r.consumed) {
        GDeflateTableRefuse(&r.why, kGDeflateRejectPopPastLane);
        return 0;
    }
    /* `consumed + n` is at most `bitsleft`, which is below the buffer width,
     * so neither shift can reach the undefined range; n == 0 reads nothing and
     * the mask below is zero for it. */
    const uint64_t window = bitbuf >> r.consumed;
    const uint64_t mask = (static_cast<uint64_t>(1) << n) - 1u;
    r.consumed += n;
    return static_cast<uint32_t>(window & mask);
}

/* Whether a lane holding `bitsleft` bits can give up `n`. The occupancy is
 * unsigned, so an unchecked subtraction would wrap to a lane that appears to
 * hold four billion bits; this is the one site that names that rung, and both
 * the sequential remove and the accelerator hit in src/gdeflate_tables.h ask it
 * before they shift. */
CUDEC_HOST_DEVICE inline bool GDeflateRemoveCheck(uint32_t bitsleft, uint32_t n,
                                                  GDeflateReject* why) {
    if (n > bitsleft) {
        return GDeflateTableRefuse(why, kGDeflateRejectRemovePastLane);
    }
    return true;
}

/* Remove n bits from the current lane. Refuses rather than underflowing. */
template <class S>
CUDEC_HOST_DEVICE inline void GDeflateRemove(S& s, uint32_t n) {
    if (s.failed) {
        return;
    }
    GDeflateReject why = kGDeflateRejectNone;
    if (!GDeflateRemoveCheck(s.Left(), n, &why)) {
        GDeflateRefuseAs(s, why);
        return;
    }
    s.Buf() >>= n;
    s.Left() -= n;
}

/* Read n bits from the current lane without removing them. The width bound is
 * not a policy: a shift of 64 or more is undefined, and this is a const read
 * with no failure flag to set, so the only fail-closed answer available here
 * is to hand back nothing. A caller reaching this is asking for a width the
 * format has no field for, and the Remove that follows a peek carries the
 * refusal that stops it going further. */
template <class S>
CUDEC_HOST_DEVICE inline uint32_t GDeflatePeek(const S& s, uint32_t n) {
    if (n == 0 || n > kGDeflateMaxPopBits) {
        return 0;
    }
    return static_cast<uint32_t>(s.Buf() &
                                 ((static_cast<uint64_t>(1) << n) - 1));
}

/* Peek and remove. No refill: refills belong to Advance, and a read that
 * quietly topped a lane up would move a word at a point the format does not,
 * which is precisely the off-by-one this header exists to prevent. The width
 * checks are GDeflateReadAhead's, so the sequential pop and the speculative
 * per-lane read refuse from one site each. */
template <class S>
CUDEC_HOST_DEVICE inline uint32_t GDeflatePop(S& s, uint32_t n) {
    if (s.failed) {
        return 0;
    }
    GDeflateLaneRead r = {0, kGDeflateRejectNone};
    const uint32_t value = GDeflateReadAhead(s.Buf(), s.Left(), r, n);
    if (r.why != kGDeflateRejectNone) {
        GDeflateRefuseAs(s, r.why);
        return 0;
    }
    GDeflateRemove(s, n);
    return value;
}

/* Hand the current lane the word at `word`. The bound is the page's word
 * count, which came from the caller's compressed size and never from the
 * bits; a well-formed page cannot reach past it, so this is a refusal and not
 * a tail case: a decoder that carried on with a short buffer would be reading
 * the lane's stale bits as if the stream had supplied them. The sequential
 * Ensure below and the warp kernel's collective refill both take their word
 * through here, so the rung has one site. */
template <class S>
CUDEC_HOST_DEVICE inline void GDeflateRefillLane(S& s,
                                                 const unsigned char* page,
                                                 uint64_t word) {
    if (s.failed) {
        return;
    }
    if (word >= s.word_count) {
        GDeflateRefuse(s, kGDeflateRejectRefillPastEnd);
        return;
    }
    s.Buf() |= static_cast<uint64_t>(GDeflateWordAt(page, word)) << s.Left();
    s.Left() += kGDeflateBitsPerPacket;
}

/* Whether a lane is due a word: below the watermark, and not already failed,
 * since a failed lane consumes nothing more. Named so the kernel's collective
 * refill and the sequential one ask the same question. */
template <class S>
CUDEC_HOST_DEVICE inline bool GDeflateLaneNeedsRefill(const S& s) {
    return !s.failed && s.Left() < kGDeflateLowWatermarkBits;
}

/* Refill the CURRENT lane if it has fallen below the watermark. Consumes at
 * most one word and never runs past the page. */
CUDEC_HOST_DEVICE inline void GDeflateEnsure(GDeflateSchedule& s,
                                             const unsigned char* page) {
    if (!GDeflateLaneNeedsRefill(s)) {
        return;
    }
    GDeflateRefillLane(s, page, s.cursor);
    if (s.failed) {
        return;
    }
    s.cursor += 1;
}

/* Ensure the current lane, then rotate. This order IS the schedule: the word a
 * lane takes is the one at the cursor while that lane is still current. */
CUDEC_HOST_DEVICE inline void GDeflateAdvance(GDeflateSchedule& s,
                                              const unsigned char* page) {
    if (s.failed) {
        return;
    }
    GDeflateEnsure(s, page);
    if (s.failed) {
        return;
    }
    s.idx = (s.idx + 1u) % kGDeflateNumStreams;
}

/* Back to lane 0 without consuming anything. Every block header rides lane 0,
 * so this is what starts a block rather than a round of its own. Guarded like
 * everything else, so "once failed, nothing moves" is literally true rather
 * than true of the parts a caller happens to look at. */
CUDEC_HOST_DEVICE inline void GDeflateReset(GDeflateSchedule& s) {
    if (s.failed) {
        return;
    }
    s.idx = 0;
}

/* The shape a page must have before a lane is primed: whole words only, and
 * at least one word per lane. A trailing partial word is not a word this
 * schedule could hand a lane, and rounding down would leave those bytes
 * reachable by nothing while the page still claimed them; and draft section
 * 5.3 says the first round always loads 32 consecutive words, so a stream
 * below 128 bytes cannot be valid and is refused here rather than discovered
 * part-way through the priming round. Both residencies prime through this, so
 * each rung has one site. */
CUDEC_HOST_DEVICE inline bool GDeflatePageShape(uint64_t page_bytes,
                                                uint64_t* word_count,
                                                GDeflateReject* why) {
    *word_count = page_bytes / 4u;
    if (page_bytes % 4u != 0u) {
        return GDeflateTableRefuse(why, kGDeflateRejectPagePartialWord);
    }
    if (*word_count < kGDeflateNumStreams) {
        return GDeflateTableRefuse(why, kGDeflateRejectPageBelowPrimingRound);
    }
    return true;
}

/* The priming round: one word into each lane, in lane order, leaving the
 * cursor at word 32 and the current lane back at 0. Returns false and leaves
 * the schedule failed if the page cannot carry it. */
CUDEC_HOST_DEVICE inline bool GDeflateInit(GDeflateSchedule& s,
                                           const unsigned char* page,
                                           uint64_t page_bytes) {
    for (uint32_t n = 0; n < kGDeflateNumStreams; ++n) {
        s.bitbuf[n] = 0;
        s.bitsleft[n] = 0;
    }
    s.idx = 0;
    s.cursor = 0;
    s.failed = false;
    s.reject = kGDeflateRejectNone;
    GDeflateReject why = kGDeflateRejectNone;
    if (!GDeflatePageShape(page_bytes, &s.word_count, &why)) {
        return GDeflateRefuseAs(s, why);
    }
    for (uint32_t n = 0; n < kGDeflateNumStreams; ++n) {
        GDeflateAdvance(s, page);
    }
    return !s.failed;
}

/* The sum of every lane's occupancy. The reference computes exactly this to
 * bound a stored block's declared length against the bytes still reachable, so
 * it is part of the schedule's surface rather than a test helper. */
CUDEC_HOST_DEVICE inline uint64_t GDeflateBufferedBits(
    const GDeflateSchedule& s) {
    uint64_t total = 0;
    for (uint32_t n = 0; n < kGDeflateNumStreams; ++n) {
        total += s.bitsleft[n];
    }
    return total;
}

}  // namespace cudec_detail

#endif /* CUDEC_GDEFLATE_SCHEDULE_H */
