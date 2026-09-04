/* The GDeflate page decode (issue #182): the block loop over stored, static
 * and dynamic blocks, the length and distance tables, and the in-tile LZ77
 * execution. Single-source for host and device, the top of the stack whose
 * lower rungs are src/gdeflate_schedule.h (#178) and src/gdeflate_tables.h
 * (#175, #176). It assembles those rather than restating them: no second bit
 * cursor, no second canonical construction, no second code-length read.
 *
 * THE TABLES HERE ARE NOT RFC 1951'S, AND THAT IS THE WHOLE REASON THIS FILE
 * CARRIES THEM. The pinned fork defines DEFLATE64 unconditionally in
 * deflate_constants.h, so its length and distance alphabets are the DEFLATE64
 * ones: symbol 285 is base 3 with SIXTEEN extra bits, reaching 65538 rather
 * than the fixed 258 of RFC 1951 section 3.2.5, and distance symbols 30 and 31
 * exist with base 32769 and 49153 and fourteen extra bits, reaching 65536 -
 * exactly the 64 KiB tile. A decoder that used the stock tables would decode
 * most pages correctly and silently corrupt the ones that use those symbols,
 * with no checksum anywhere to catch it (docs/MASTERPLAN.md section 11.4).
 * Each of the three is pinned in tests/gdeflate_block_twin.cpp against the
 * oracle AND against the stock reading, so the pin proves something.
 *
 * THE DEFERRED COPY IS THE FORMAT AND NOT AN OPTIMISATION. A length symbol and
 * its distance symbol are decoded on the SAME lane in CONSECUTIVE rounds of
 * that lane, which is thirty-two rounds apart in the page. The round that
 * decodes the length reserves the output range and moves the write cursor past
 * it; the round that decodes the distance performs the copy into that reserved
 * range. So the output is written out of order, and a decoder that performed
 * the copy immediately would consume the distance symbol from the wrong lane
 * and decode a different page. Deferring is what the format says, and the
 * thirty-two rounds after end-of-block exist to drain the copies still
 * outstanding when the block ended.
 *
 * ONE ROUND BODY FOR BOTH RESIDENCIES (issue #214). The loop below is written
 * over the team contract at GDeflateHostTeam in src/gdeflate_tables.h: every
 * lane resolves its symbol speculatively, the team names the first lane that
 * hit end-of-block, and each lane commits only if it sits at or below that
 * lane - a lane that must not have consumed keeps the bits it entered the
 * round with, which is a select rather than a rollback (masterplan 13.2). The
 * lanes above the end-of-block lane take the drain's first steps in the same
 * round, and one more round over the lanes below it takes the rest, which is
 * the reference's thirty-two drain steps starting at the end-of-block lane
 * and read off in that order. Output positions come from the team in lane
 * order, and the copies of one round are performed in lane order, because a
 * copy's source lies before its own reservation and a lower lane's
 * reservation of the same round may be what it reads.
 *
 * FAIL-CLOSED, AND WHERE THE BOUNDS COME FROM. Every length is checked against
 * the output that remains before any byte is written, every distance is
 * checked against the output already produced before any byte is read, and the
 * stored block's declared length is checked against the bytes the page can
 * still supply. None of the three is a bound derived from the stream. A match
 * may never reach before the start of this page's output, which is what makes
 * a tile independent of every other tile - the reference bounds it the same
 * way and against the same origin, because it decompresses one page into one
 * tile too.
 *
 * WHAT IT DOES NOT DO. It decodes ONE page. The tile stream that says where
 * pages are is src/tilestream.h, and the batch surface that drives it is #216.
 * Reject behaviour beyond the refusals named above is filed separately from
 * this issue. */
#ifndef CUDEC_GDEFLATE_BLOCK_H
#define CUDEC_GDEFLATE_BLOCK_H

#include "gdeflate_tables.h"

#include <stdint.h>

namespace cudec_detail {

/* The reference's block types (deflate_constants.h). Type 3 is reserved and is
 * the one BTYPE that carries no meaning at all, so it is refused by name. */
constexpr uint32_t kGDeflateBlockStored = 0;
constexpr uint32_t kGDeflateBlockStatic = 1;
constexpr uint32_t kGDeflateBlockDynamic = 2;
constexpr uint32_t kGDeflateBlockReserved = 3;

/* The block types a decoded page can be made OF, which is one fewer than the
 * BTYPE field can spell. The reserved type is refused before anything is
 * accounted for it, so it never indexes the census below and the census array
 * is deliberately too short to hold it. */
constexpr uint32_t kGDeflateBlockTypeCount = 3;

/* The literal/length alphabet's first length symbol, and the end-of-block
 * symbol below it. */
constexpr uint32_t kGDeflateEndOfBlock = 256;
constexpr uint32_t kGDeflateFirstLengthSym = 257;

/* The shortest match the format can express. It is here because the distance
 * check below is what stops a match reading before the page's output, and a
 * reader wants the minimum beside it. */
constexpr uint32_t kGDeflateMinMatchLen = 3;

/* Length symbol 257 + i: its base, and how many extra bits follow it in the
 * same round. Data from the pinned fork's `litlen_decode_results` under
 * DEFLATE64, restated here because this is the thing that has to agree with
 * it. The last three entries are where the format leaves RFC 1951: symbols
 * 285, 286 and 287 all carry base 3 with sixteen extra bits. */
CUDEC_HOST_DEVICE inline uint32_t GDeflateLengthBase(uint32_t index) {
    const uint16_t kBase[31] = {3,  4,  5,  6,  7,  8,  9,   10,  11,  13, 15,
                                17, 19, 23, 27, 31, 35, 43,  51,  59,  67, 83,
                                99, 115, 131, 163, 195, 227, 3,   3,   3};
    return kBase[index];
}

CUDEC_HOST_DEVICE inline uint32_t GDeflateLengthExtra(uint32_t index) {
    const unsigned char kExtra[31] = {0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1,
                                      1, 2, 2, 2, 2, 3, 3, 3,  3,  4,  4,
                                      4, 4, 5, 5, 5, 5, 16, 16, 16};
    return kExtra[index];
}

/* Distance symbol `sym`: its base and its extra-bit count. Symbols 30 and 31
 * are the second divergence - RFC 1951 stops at 29 and 32768. */
CUDEC_HOST_DEVICE inline uint32_t GDeflateDistBase(uint32_t sym) {
    const uint32_t kBase[kGDeflateNumDistSyms] = {
        1,    2,    3,    4,     5,     7,     9,     13,   17,    25,   33,
        49,   65,   97,   129,   193,   257,   385,   513,  769,   1025, 1537,
        2049, 3073, 4097, 6145,  8193,  12289, 16385, 24577, 32769, 49153};
    return kBase[sym];
}

CUDEC_HOST_DEVICE inline uint32_t GDeflateDistExtra(uint32_t sym) {
    const unsigned char kExtra[kGDeflateNumDistSyms] = {
        0, 0, 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
        7, 7, 8,  8,  9,  9,  10, 10, 11, 11, 12, 12, 13, 13, 14, 14};
    return kExtra[sym];
}

/* The static block's two codes (RFC 1951 section 3.2.6, unchanged by the
 * DEFLATE64 divergences, which move bases and extra bits rather than codeword
 * lengths). Built through the same construction every other table goes
 * through, so a static block and a dynamic block share one decoder. */
CUDEC_HOST_DEVICE inline bool GDeflateStaticTables(GDeflateCodeLengths& scratch,
                                                   GDeflateLitLenTable& litlen,
                                                   GDeflateDistTable& dist,
                                                   GDeflateReject* why =
                                                       nullptr) {
    /* The fixed lengths are written into the team's code-length buffer rather
     * than a local array: on the device a local array of the alphabet's size
     * is a per-thread stack frame paid on every page, static block or not,
     * and the buffer is idle during a static block by construction. */
    unsigned char* lens = scratch.lens;
    for (uint32_t i = 0; i < kGDeflateNumLitLenSyms; i++) {
        if (i < 144) {
            lens[i] = 8;
        } else if (i < 256) {
            lens[i] = 9;
        } else if (i < 280) {
            lens[i] = 7;
        } else {
            lens[i] = 8;
        }
    }
    if (!GDeflateBuildTable(lens, kGDeflateNumLitLenSyms, litlen, why)) {
        return false;
    }
    unsigned char* dist_lens = scratch.lens + kGDeflateNumLitLenSyms;
    for (uint32_t i = 0; i < kGDeflateNumDistSyms; i++) {
        dist_lens[i] = 5;
    }
    return GDeflateBuildTable(dist_lens, kGDeflateNumDistSyms, dist, why);
}

/* What a page turned out to hold: how many blocks, of which types, producing
 * how many bytes each. It is what the decode produced rather than
 * bookkeeping: a page is one block or several and nothing outside says which,
 * so without this the multi-block case cannot be told from the single-block
 * one by anything that reads a decode. A block boundary inside a page is not
 * findable by scanning (docs/MASTERPLAN.md section 11.3), so a decode that
 * walked to a block is the only thing that can attribute a byte to its type,
 * and without this the attribution is unreadable from outside (issue #206).
 * Written only on a successful decode. */
struct GDeflateCensus {
    uint32_t blocks;
    uint32_t type_blocks[kGDeflateBlockTypeCount];
    uint64_t type_bytes[kGDeflateBlockTypeCount];
};

/* Everything a sequential page decode carries: the schedule, the two live
 * tables, the code-length vector, and the census. The deferred copies live in
 * the team. */
struct GDeflatePageState {
    GDeflateSchedule s;
    GDeflateLitLenTable litlen;
    GDeflateDistTable dist;
    GDeflateCodeLengths lens;
    /* The census, flattened into this struct for the readers that had it
     * here before the rounds moved into the team. */
    uint32_t blocks;
    uint32_t type_blocks[kGDeflateBlockTypeCount];
    uint64_t type_bytes[kGDeflateBlockTypeCount];
};

/* What one lane resolved for the round ahead of it, before anything is
 * committed: the symbol, how many bits it and its extra field span, the length
 * or offset those extra bits complete, and the rung the read refused on. */
struct GDeflateLaneStep {
    GDeflateLaneRead read;
    uint32_t sym;
    uint32_t value;
};

/* Resolve the current lane's next literal/length symbol and, for a length,
 * the extra bits behind it, consuming nothing. */
template <class Team>
CUDEC_HOST_DEVICE inline GDeflateLaneStep GDeflateResolveLitLen(Team& lane) {
    GDeflateLaneStep step;
    step.read.consumed = 0;
    step.read.why = kGDeflateRejectNone;
    step.value = 0;
    const uint64_t buf = lane.Bits().Buf();
    const uint32_t left = lane.Bits().Left();
    step.sym = GDeflateResolveSymbol(lane.LitLen(), buf, left, step.read);
    if (step.read.why == kGDeflateRejectNone &&
        step.sym >= kGDeflateFirstLengthSym) {
        const uint32_t index = step.sym - kGDeflateFirstLengthSym;
        step.value = GDeflateLengthBase(index) +
                     GDeflateReadAhead(buf, left, step.read,
                                       GDeflateLengthExtra(index));
    }
    return step;
}

/* Resolve the current lane's distance symbol and its extra bits, which
 * together are the offset of the copy this lane reserved, consuming nothing. */
template <class Team>
CUDEC_HOST_DEVICE inline GDeflateLaneStep GDeflateResolveDist(Team& lane) {
    GDeflateLaneStep step;
    step.read.consumed = 0;
    step.read.why = kGDeflateRejectNone;
    step.value = 0;
    const uint64_t buf = lane.Bits().Buf();
    const uint32_t left = lane.Bits().Left();
    step.sym = GDeflateResolveSymbol(lane.Dist(), buf, left, step.read);
    if (step.read.why == kGDeflateRejectNone) {
        step.value = GDeflateDistBase(step.sym) +
                     GDeflateReadAhead(buf, left, step.read,
                                       GDeflateDistExtra(step.sym));
    }
    return step;
}

/* What one lane does in a symbol round, decided after the team has named the
 * end-of-block lane. */
enum GDeflateLaneMode {
    kGDeflateLaneIdle = 0,
    kGDeflateLaneLiteral,
    kGDeflateLaneReserve,
    kGDeflateLaneEndOfBlock,
    kGDeflateLaneCopy
};

/* Retire the current lane's reservation: commit the distance read, check the
 * source against the output this page has produced, and hand the copy to the
 * team. One function because the symbol round and the drain round both retire
 * copies, and a rung named at two sites is what the ladder lock refuses. */
template <class Team>
CUDEC_HOST_DEVICE inline bool GDeflateRetireCopy(Team& lane,
                                                 const GDeflateLaneStep& step,
                                                 GDeflateDeferredCopy& c) {
    auto& b = lane.Bits();
    if (step.read.why != kGDeflateRejectNone) {
        return GDeflateRefuseAs(b, step.read.why);
    }
    GDeflateRemove(b, step.read.consumed);
    /* The match source may not begin before this page's output. Compared in
     * the 64-bit width the output position is carried in, so the comparison
     * cannot be the place a narrowing hides: the offset is at most 65536 and
     * `c.out_pos` is where the reservation started. */
    if (static_cast<uint64_t>(step.value) > c.out_pos) {
        return GDeflateRefuse(b, kGDeflateRejectMatchBeforeOutput);
    }
    c.pending = false;
    return true;
}

/* Decode one page into `out`, which is the tile it belongs to, over the team
 * `t`. Returns false with the team failed for every page this decoder refuses;
 * `*out_len` and `census` are written only on success.
 *
 * `out_cap` is the caller's capacity and is the only bound the output is
 * checked against. It is never derived from the page: the format states no
 * uncompressed size anywhere inside a page, which is why the tile size lives
 * in the stream header that src/tilestream.h parses. */
template <class Team>
CUDEC_HOST_DEVICE inline bool GDeflateDecodePageRounds(Team& t,
                                                       uint64_t page_bytes,
                                                       unsigned char* out,
                                                       uint64_t out_cap,
                                                       uint64_t* out_len,
                                                       GDeflateCensus* census) {
    if (!t.Prime(page_bytes)) {
        return false;
    }
    t.ClearCopies();
    uint64_t out_pos = 0;

    /* A block costs lane 0 at least the three bits of its own header, and a
     * lane holds at most kGDeflateMaxLaneBits, so at most that many thirds of
     * a lane's worth of blocks can pass between two refills of lane 0 - and
     * lane 0 can be refilled at most once per word the page holds. The cap is
     * therefore generous by a wide margin and exists to make termination a
     * property of the code rather than of an argument about the input. */
    const uint64_t block_fuel =
        (t.WordCount() + 1u) * (kGDeflateMaxLaneBits / kGDeflateMinMatchLen);
    bool final_block = false;
    uint32_t blocks_read = 0;
    uint32_t type_blocks[kGDeflateBlockTypeCount] = {0, 0, 0};
    uint64_t type_bytes[kGDeflateBlockTypeCount] = {0, 0, 0};
    for (uint64_t block = 0; block < block_fuel && !final_block; block++) {
        blocks_read++;
        const uint64_t block_out_start = out_pos;
        t.Reset();
        /* BFINAL and BTYPE ride lane 0 and every lane needs them, packed into
         * one word for one broadcast. */
        uint32_t header = 0;
        t.Lane0([&](Team& lane0) {
            auto& b = lane0.Bits();
            const uint32_t bfinal = GDeflatePop(b, 1);
            const uint32_t type = GDeflatePop(b, 2);
            header = bfinal | (type << 1);
        });
        header = t.Broadcast(header);
        if (t.Failed()) {
            return false;
        }
        final_block = (header & 1u) != 0;
        const uint32_t block_type = header >> 1;
        t.EnsureLane0();
        if (t.Failed()) {
            return false;
        }

        if (block_type == kGDeflateBlockReserved) {
            return GDeflateRefuse(t.Bits(), kGDeflateRejectBlockTypeReserved);
        }

        if (block_type == kGDeflateBlockStored) {
            /* The bytes the page can still supply: whole words the cursor has
             * not reached, plus whatever the lanes already hold. The reference
             * computes exactly this, and it is the only bound a stored block's
             * declared length has - there is no complement field to
             * cross-check it against (dossier 11.3, D3).
             *
             * The reference guards its own sixteen-bit read with a second
             * check that at least two bytes remain. That one is not carried
             * here: it protects an unchecked read out of the input buffer,
             * and GDeflatePop is bounded by the schedule already, so the same
             * check on this side could not be reached by any page and a guard
             * that cannot bite proves nothing. */
            const uint64_t reachable =
                (t.WordCount() - t.Cursor()) * (kGDeflateBitsPerPacket / 8u) +
                (t.BufferedBits() + 7u) / 8u;
            uint32_t len = 0;
            t.Lane0([&](Team& lane0) {
                len = GDeflatePop(lane0.Bits(), 16);
            });
            len = t.Broadcast(len);
            if (t.Failed()) {
                return false;
            }
            /* Two rungs rather than one condition, for the reason the pop
             * width and the lane occupancy are two in the schedule: a length
             * past what the caller can hold and a length past what the page
             * can still supply are different defects, and a single rung would
             * let a negative for either stand in for the other. */
            if (len > out_cap - out_pos) {
                return GDeflateRefuse(t.Bits(), kGDeflateRejectStoredPastCap);
            }
            if (len > reachable) {
                return GDeflateRefuse(t.Bits(), kGDeflateRejectStoredPastPage);
            }
            /* One byte per lane per round, lane order, each byte's lane
             * refilled behind it: the reference's byte loop, thirty-two at a
             * time. */
            for (uint32_t done = 0; done < len;) {
                const uint32_t batch = (len - done < kGDeflateNumStreams)
                                           ? (len - done)
                                           : kGDeflateNumStreams;
                t.Round(batch, [&](Team& lane) -> bool {
                    if (!lane.Active()) {
                        return false;
                    }
                    const uint32_t byte = GDeflatePop(lane.Bits(), 8);
                    if (!lane.Bits().failed) {
                        out[out_pos + lane.Lane()] =
                            static_cast<unsigned char>(byte);
                    }
                    return true;
                });
                if (t.Failed()) {
                    return false;
                }
                out_pos += batch;
                done += batch;
            }
        } else {
            if (block_type == kGDeflateBlockStatic) {
                t.Build([&](Team& one) {
                    GDeflateReject static_why = kGDeflateRejectNone;
                    if (!GDeflateStaticTables(one.Lens(), one.LitLen(),
                                              one.Dist(), &static_why)) {
                        GDeflateRefuseAs(one.Bits(), static_why);
                    }
                });
                if (t.Failed()) {
                    return false;
                }
            } else if (!GDeflateReadDynamicTableRounds(t)) {
                return false;
            }

            t.Reset();
            /* Every step either writes a byte, reserves a match of at least
             * the minimum length, or retires a reservation an earlier step
             * made - so the capacity bounds the reserving steps and the
             * retiring steps alike, and the end-of-block step and the drain
             * are the constant beside them. A round is up to a lane's worth
             * of steps, so the same cap counted in rounds is wider still. */
            const uint64_t round_fuel =
                (out_cap > (UINT64_MAX - kGDeflateNumStreams) / 2u - 1u)
                    ? UINT64_MAX
                    : 2u * (out_cap + 1u) + kGDeflateNumStreams;
            bool block_done = false;
            for (uint64_t round = 0; round < round_fuel && !block_done;
                 round++) {
                uint32_t eob_lane = kGDeflateNoLane;
                t.Round(kGDeflateNumStreams, [&](Team& lane) -> bool {
                    GDeflateDeferredCopy& c = lane.Copy();
                    const bool active = lane.Active();
                    const bool pending = active && c.pending;
                    GDeflateLaneStep step;
                    step.read.consumed = 0;
                    step.read.why = kGDeflateRejectNone;
                    step.sym = kGDeflateNoSymbol;
                    step.value = 0;
                    if (pending) {
                        step = GDeflateResolveDist(lane);
                    } else if (active) {
                        step = GDeflateResolveLitLen(lane);
                    }
                    /* The end-of-block lane is the lowest lane that resolved
                     * code 256 while holding no reservation; a lane above it
                     * must not have consumed, so it takes a drain step
                     * instead, and the reference leaves the loop AT that lane
                     * without advancing, so the drain starts there. */
                    const bool hit_eob = !pending && active &&
                                         step.read.why == kGDeflateRejectNone &&
                                         step.sym == kGDeflateEndOfBlock;
                    const uint32_t first_eob = lane.FirstLane(hit_eob);
                    eob_lane = first_eob;
                    GDeflateLaneMode mode = kGDeflateLaneIdle;
                    if (pending) {
                        mode = kGDeflateLaneCopy;
                    } else if (active && lane.Lane() < first_eob) {
                        mode = (step.read.why != kGDeflateRejectNone ||
                                step.sym < kGDeflateEndOfBlock)
                                   ? kGDeflateLaneLiteral
                                   : kGDeflateLaneReserve;
                    } else if (active && lane.Lane() == first_eob) {
                        mode = kGDeflateLaneEndOfBlock;
                    }
                    /* The reservation, not the copy: the distance symbol this
                     * match needs arrives on this same lane one round of its
                     * own later, and the output cursor moves past the hole
                     * now so the rounds in between write after it. A literal
                     * claims its one byte the same way. */
                    uint64_t claim = 0;
                    if (mode == kGDeflateLaneLiteral) {
                        claim = 1;
                    } else if (mode == kGDeflateLaneReserve &&
                               step.read.why == kGDeflateRejectNone) {
                        claim = step.value;
                    }
                    const uint64_t pos = lane.Claim(out_pos, claim);
                    bool copy_now = false;
                    if (mode == kGDeflateLaneCopy) {
                        copy_now = GDeflateRetireCopy(lane, step, c);
                    } else if (mode != kGDeflateLaneIdle) {
                        auto& b = lane.Bits();
                        if (step.read.why != kGDeflateRejectNone) {
                            GDeflateRefuseAs(b, step.read.why);
                        } else {
                            GDeflateRemove(b, step.read.consumed);
                            if (mode == kGDeflateLaneLiteral) {
                                if (pos >= out_cap) {
                                    GDeflateRefuse(b, kGDeflateRejectLiteralPastCap);
                                } else {
                                    out[pos] =
                                        static_cast<unsigned char>(step.sym);
                                }
                            } else if (mode == kGDeflateLaneReserve) {
                                /* `pos` can exceed the capacity only on the
                                 * warp, behind a lower lane that refused in
                                 * this round and whose claim still moved the
                                 * base; the team fails at the round's end,
                                 * but the reservation must not be recorded
                                 * past the tile even for that one round. */
                                if (pos > out_cap ||
                                    step.value > out_cap - pos) {
                                    GDeflateRefuse(b, kGDeflateRejectMatchPastCap);
                                } else {
                                    c.pending = true;
                                    c.length = step.value;
                                    c.out_pos = pos;
                                }
                            }
                        }
                    }
                    lane.CopyBytes(copy_now && !lane.Bits().failed, out,
                                   c.out_pos, c.out_pos - step.value,
                                   c.length);
                    /* Every lane of the round steps, drain steps included:
                     * the reference advances after each of the thirty-two
                     * drain iterations whether or not that lane held a
                     * reservation. */
                    return active;
                });
                if (t.Failed()) {
                    return false;
                }
                if (eob_lane != kGDeflateNoLane) {
                    /* The rest of the drain: the lanes below the end-of-block
                     * lane, in order, each retiring what it still holds. */
                    t.Round(eob_lane, [&](Team& lane) -> bool {
                        GDeflateDeferredCopy& c = lane.Copy();
                        const bool active = lane.Active();
                        const bool pending = active && c.pending;
                        GDeflateLaneStep step;
                        step.read.consumed = 0;
                        step.read.why = kGDeflateRejectNone;
                        step.sym = kGDeflateNoSymbol;
                        step.value = 0;
                        if (pending) {
                            step = GDeflateResolveDist(lane);
                        }
                        bool copy_now = false;
                        if (pending) {
                            copy_now = GDeflateRetireCopy(lane, step, c);
                        }
                        lane.CopyBytes(copy_now && !lane.Bits().failed, out,
                                       c.out_pos, c.out_pos - step.value,
                                       c.length);
                        return active;
                    });
                    if (t.Failed()) {
                        return false;
                    }
                    block_done = true;
                }
            }
            if (!block_done) {
                /* The round cap was reached with no end-of-block. A page that
                 * ran that far has produced more rounds than its own capacity
                 * admits, so it is refused rather than reported as a decode. */
                return GDeflateRefuse(t.Bits(), kGDeflateRejectRoundFuelExhausted);
            }
        }

        /* BTYPE is two bits and the reserved value returned above, so the
         * index is inside the census by construction rather than by a bound
         * checked here. The byte count is the output cursor's movement across
         * the whole block, drain included: a match reserved inside this block
         * moved the cursor when it was reserved, and the copy that retires it
         * writes into the range that move already claimed. */
        type_blocks[block_type]++;
        type_bytes[block_type] += out_pos - block_out_start;
    }
    if (!final_block) {
        return GDeflateRefuse(t.Bits(), kGDeflateRejectNoFinalBlock);
    }
    census->blocks = blocks_read;
    for (uint32_t n = 0; n < kGDeflateBlockTypeCount; n++) {
        census->type_blocks[n] = type_blocks[n];
        census->type_bytes[n] = type_bytes[n];
    }
    *out_len = out_pos;
    return true;
}

/* The sequential entry point, which the twins, the fuzz targets and the bench
 * drive: the rounds above over the host team, and the census copied into the
 * state where those readers find it. */
inline bool GDeflateDecodePage(GDeflatePageState& st,
                               const unsigned char* page, uint64_t page_bytes,
                               unsigned char* out, uint64_t out_cap,
                               uint64_t* out_len) {
    GDeflateHostTeam t(st.s, page, &st.lens, &st.litlen, &st.dist);
    GDeflateCensus census;
    if (!GDeflateDecodePageRounds(t, page_bytes, out, out_cap, out_len,
                                  &census)) {
        return false;
    }
    st.blocks = census.blocks;
    for (uint32_t n = 0; n < kGDeflateBlockTypeCount; n++) {
        st.type_blocks[n] = census.type_blocks[n];
        st.type_bytes[n] = census.type_bytes[n];
    }
    return true;
}

}  // namespace cudec_detail

#endif /* CUDEC_GDEFLATE_BLOCK_H */
