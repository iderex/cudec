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
 * LAYOUT. This struct holds all 32 lane buffers because the CPU twin runs the
 * lanes sequentially in one thread. In the warp kernel each lane owns its own
 * bit buffer and occupancy in registers while the lane index and the cursor
 * are warp-uniform, so the kernel instantiates the same arithmetic over a
 * different residency. What must not fork is the schedule itself, which is why
 * it lives in one place. The cursor is 64-bit for the reason the other parsers
 * give: the caller's sizes are size_t, and mixing widths in a bound comparison
 * is where an overflow hides. */
#ifndef CUDEC_GDEFLATE_SCHEDULE_H
#define CUDEC_GDEFLATE_SCHEDULE_H

#include <stdint.h>

/* Guarded: the other single-sourced parsers define the same macro for the same
 * reason, and a device translation unit that decodes two formats includes two
 * of these headers. The definitions are identical, so an unguarded second one
 * is legal rather than an error - which is exactly why it would go unnoticed
 * if they ever stopped being identical. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
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

/* The 32 lane bit buffers, the current lane, and the one shared word cursor.
 * The failure flag is sticky: once set, every operation is a no-op, so a
 * caller that checks it once at the end reads the same verdict as one that
 * checks after every call. */
struct GDeflateSchedule {
    uint64_t bitbuf[kGDeflateNumStreams];
    uint32_t bitsleft[kGDeflateNumStreams];
    uint32_t idx;
    uint64_t cursor;
    uint64_t word_count;
    bool failed;
};

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

/* Refill the CURRENT lane if it has fallen below the watermark. Consumes at
 * most one word and never runs past the page. */
CUDEC_HOST_DEVICE inline void GDeflateEnsure(GDeflateSchedule& s,
                                             const unsigned char* page) {
    if (s.failed) {
        return;
    }
    if (s.bitsleft[s.idx] >= kGDeflateLowWatermarkBits) {
        return;
    }
    if (s.cursor >= s.word_count) {
        /* The page has no word left to give this lane. A well-formed page
         * cannot reach here, so this is a refusal and not a tail case: a
         * decoder that carried on with a short buffer would be reading the
         * lane's stale bits as if the stream had supplied them. */
        s.failed = true;
        return;
    }
    s.bitbuf[s.idx] |= static_cast<uint64_t>(GDeflateWordAt(page, s.cursor))
                       << s.bitsleft[s.idx];
    s.cursor += 1;
    s.bitsleft[s.idx] += kGDeflateBitsPerPacket;
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
    s.word_count = page_bytes / 4u;
    /* Whole words only. A trailing partial word is not a word this schedule
     * could hand a lane, and rounding down would leave those bytes reachable
     * by nothing while the page still claimed them. */
    if (page_bytes % 4u != 0u) {
        s.failed = true;
        return false;
    }
    if (s.word_count < kGDeflateNumStreams) {
        /* Draft section 5.3: the first round always loads 32 consecutive
         * words, so a stream below 128 bytes cannot be valid. Refused here
         * rather than discovered part-way through the priming round. */
        s.failed = true;
        return false;
    }
    for (uint32_t n = 0; n < kGDeflateNumStreams; ++n) {
        GDeflateAdvance(s, page);
    }
    return !s.failed;
}

/* Read n bits from the current lane without removing them. The width bound is
 * not a policy: a shift of 64 or more is undefined, and this is a const read
 * with no failure flag to set, so the only fail-closed answer available here
 * is to hand back nothing. A caller reaching this is asking for a width the
 * format has no field for, and the Remove that follows a peek carries the
 * refusal that stops it going further. */
CUDEC_HOST_DEVICE inline uint32_t GDeflatePeek(const GDeflateSchedule& s,
                                               uint32_t n) {
    if (n == 0 || n > kGDeflateMaxPopBits) {
        return 0;
    }
    return static_cast<uint32_t>(s.bitbuf[s.idx] &
                                 ((static_cast<uint64_t>(1) << n) - 1));
}

/* Remove n bits from the current lane. Refuses rather than underflowing: the
 * occupancy is unsigned, so an unchecked subtraction would wrap to a lane that
 * appears to hold four billion bits. */
CUDEC_HOST_DEVICE inline void GDeflateRemove(GDeflateSchedule& s, uint32_t n) {
    if (s.failed) {
        return;
    }
    if (n > s.bitsleft[s.idx]) {
        s.failed = true;
        return;
    }
    s.bitbuf[s.idx] >>= n;
    s.bitsleft[s.idx] -= n;
}

/* Peek and remove. No refill: refills belong to Advance, and a read that
 * quietly topped a lane up would move a word at a point the format does not,
 * which is precisely the off-by-one this header exists to prevent. */
CUDEC_HOST_DEVICE inline uint32_t GDeflatePop(GDeflateSchedule& s, uint32_t n) {
    if (s.failed) {
        return 0;
    }
    if (n > kGDeflateMaxPopBits || n > s.bitsleft[s.idx]) {
        s.failed = true;
        return 0;
    }
    const uint32_t value = GDeflatePeek(s, n);
    GDeflateRemove(s, n);
    return value;
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
