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
CUDEC_HOST_DEVICE inline bool GDeflateStaticTables(GDeflateLitLenTable& litlen,
                                                   GDeflateDistTable& dist) {
    unsigned char lens[kGDeflateNumLitLenSyms];
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
    if (!GDeflateBuildTable(lens, kGDeflateNumLitLenSyms, litlen)) {
        return false;
    }
    unsigned char dist_lens[kGDeflateNumDistSyms];
    for (uint32_t i = 0; i < kGDeflateNumDistSyms; i++) {
        dist_lens[i] = 5;
    }
    return GDeflateBuildTable(dist_lens, kGDeflateNumDistSyms, dist);
}

/* One lane's outstanding match: the length it reserved and where in the output
 * that reservation starts. Held per lane for the reason GDeflateSchedule holds
 * all 32 bit buffers - the CPU twin runs the lanes sequentially in one thread,
 * while in the warp kernel each lane owns its own copy state in registers. The
 * reference keeps the same thing as a rotating 32-bit mask beside a per-lane
 * array; a mask and a flag per lane are the same set, and the flag is the
 * shape that reads correctly in both residencies. */
struct GDeflateDeferredCopy {
    uint64_t out_pos;
    uint32_t length;
    bool pending;
};

/* Everything a page decode carries: the schedule, the two live tables, and the
 * 32 deferred copies. */
struct GDeflatePageState {
    GDeflateSchedule s;
    GDeflateLitLenTable litlen;
    GDeflateDistTable dist;
    GDeflateCodeLengths lens;
    GDeflateDeferredCopy copies[kGDeflateNumStreams];
    /* How many blocks the page turned out to hold. It is what the decode
     * produced rather than bookkeeping: a page is one block or several and
     * nothing outside says which, so without this the multi-block case cannot
     * be told from the single-block one by anything that reads a decode. */
    uint32_t blocks;
};

/* Perform the copy the current lane reserved: decode the distance symbol off
 * this lane, add its extra bits, and move the bytes. */
CUDEC_HOST_DEVICE inline bool GDeflateDoCopy(GDeflatePageState& st,
                                             unsigned char* out) {
    GDeflateSchedule& s = st.s;
    GDeflateDeferredCopy& c = st.copies[s.idx];
    const uint32_t sym = GDeflateDecodeSymbol(s, st.dist);
    if (sym == kGDeflateNoSymbol) {
        return false;
    }
    const uint32_t offset = GDeflateDistBase(sym) +
                            GDeflatePop(s, GDeflateDistExtra(sym));
    if (s.failed) {
        return false;
    }
    /* The match source may not begin before this page's output. Compared in
     * the 64-bit width the output position is carried in, so the comparison
     * cannot be the place a narrowing hides: `offset` is at most 65536 and
     * `c.out_pos` is where the reservation started. */
    if (static_cast<uint64_t>(offset) > c.out_pos) {
        s.failed = true;
        return false;
    }
    const uint64_t src = c.out_pos - offset;
    /* Byte by byte, forwards, which is what an overlapping match MEANS: a
     * distance below the length is a run that repeats what this copy is itself
     * writing. A word-at-a-time copy would have to special-case that; this
     * does not, and it is the same order on both residencies, which is what
     * makes the output bit-identical (docs/DETERMINISM.md). */
    for (uint32_t n = 0; n < c.length; n++) {
        out[c.out_pos + n] = out[src + n];
    }
    c.pending = false;
    return true;
}

/* Decode one page into `out`, which is the tile it belongs to. Returns false
 * with the schedule failed for every page this decoder refuses; `*out_len` is
 * written only on success.
 *
 * `out_cap` is the caller's capacity and is the only bound the output is
 * checked against. It is never derived from the page: the format states no
 * uncompressed size anywhere inside a page, which is why the tile size lives
 * in the stream header that src/tilestream.h parses. */
CUDEC_HOST_DEVICE inline bool GDeflateDecodePage(GDeflatePageState& st,
                                                 const unsigned char* page,
                                                 uint64_t page_bytes,
                                                 unsigned char* out,
                                                 uint64_t out_cap,
                                                 uint64_t* out_len) {
    GDeflateSchedule& s = st.s;
    if (!GDeflateInit(s, page, page_bytes)) {
        return false;
    }
    for (uint32_t n = 0; n < kGDeflateNumStreams; n++) {
        st.copies[n].pending = false;
        st.copies[n].length = 0;
        st.copies[n].out_pos = 0;
    }
    uint64_t out_pos = 0;

    /* A block costs lane 0 at least the three bits of its own header, and a
     * lane holds at most kGDeflateMaxLaneBits, so at most that many thirds of
     * a lane's worth of blocks can pass between two refills of lane 0 - and
     * lane 0 can be refilled at most once per word the page holds. The cap is
     * therefore generous by a wide margin and exists to make termination a
     * property of the code rather than of an argument about the input. */
    const uint64_t block_fuel =
        (s.word_count + 1u) * (kGDeflateMaxLaneBits / kGDeflateMinMatchLen);
    bool final_block = false;
    uint32_t blocks_read = 0;
    for (uint64_t block = 0; block < block_fuel && !final_block; block++) {
        blocks_read++;
        GDeflateReset(s);
        final_block = GDeflatePop(s, 1) != 0;
        const uint32_t block_type = GDeflatePop(s, 2);
        if (s.failed) {
            return false;
        }
        GDeflateEnsure(s, page);

        if (block_type == kGDeflateBlockReserved) {
            s.failed = true;
            return false;
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
                (s.word_count - s.cursor) * (kGDeflateBitsPerPacket / 8u) +
                (GDeflateBufferedBits(s) + 7u) / 8u;
            const uint32_t len = GDeflatePop(s, 16);
            if (s.failed) {
                return false;
            }
            if (len > out_cap - out_pos || len > reachable) {
                s.failed = true;
                return false;
            }
            for (uint32_t n = 0; n < len; n++) {
                out[out_pos] = static_cast<unsigned char>(GDeflatePop(s, 8));
                out_pos++;
                GDeflateAdvance(s, page);
            }
            if (s.failed) {
                return false;
            }
        } else {
            if (block_type == kGDeflateBlockStatic) {
                if (!GDeflateStaticTables(st.litlen, st.dist)) {
                    s.failed = true;
                    return false;
                }
            } else if (!GDeflateReadDynamicTables(s, page, st.lens, st.litlen,
                                                  st.dist)) {
                return false;
            }

            GDeflateReset(s);
            /* Every round either writes a byte, reserves a match of at least
             * the minimum length, or retires a reservation an earlier round
             * made - so the capacity bounds the reserving rounds and the
             * retiring rounds alike, and the end-of-block round and the drain
             * are the constant beside them. */
            const uint64_t round_fuel = 2u * (out_cap + 1u) + kGDeflateNumStreams;
            bool block_done = false;
            for (uint64_t round = 0; round < round_fuel && !block_done;
                 round++) {
                if (st.copies[s.idx].pending) {
                    if (!GDeflateDoCopy(st, out)) {
                        return false;
                    }
                    GDeflateAdvance(s, page);
                    if (s.failed) {
                        return false;
                    }
                    continue;
                }
                const uint32_t sym = GDeflateDecodeSymbol(s, st.litlen);
                if (sym == kGDeflateNoSymbol) {
                    return false;
                }
                if (sym < kGDeflateEndOfBlock) {
                    if (out_pos == out_cap) {
                        s.failed = true;
                        return false;
                    }
                    out[out_pos] = static_cast<unsigned char>(sym);
                    out_pos++;
                    GDeflateAdvance(s, page);
                    if (s.failed) {
                        return false;
                    }
                    continue;
                }
                if (sym == kGDeflateEndOfBlock) {
                    /* The reference leaves the loop here WITHOUT advancing:
                     * the drain below starts on this lane, not the next. */
                    block_done = true;
                    continue;
                }
                const uint32_t index = sym - kGDeflateFirstLengthSym;
                const uint32_t length =
                    GDeflateLengthBase(index) +
                    GDeflatePop(s, GDeflateLengthExtra(index));
                if (s.failed) {
                    return false;
                }
                if (length > out_cap - out_pos) {
                    s.failed = true;
                    return false;
                }
                /* The reservation, not the copy. The distance symbol this
                 * match needs arrives on this same lane one round of its own
                 * later, and the output cursor moves past the hole now so the
                 * rounds in between write after it. */
                st.copies[s.idx].pending = true;
                st.copies[s.idx].length = length;
                st.copies[s.idx].out_pos = out_pos;
                out_pos += length;
                GDeflateAdvance(s, page);
                if (s.failed) {
                    return false;
                }
            }
            if (!block_done) {
                /* The round cap was reached with no end-of-block. A page that
                 * ran that far has produced more rounds than its own capacity
                 * admits, so it is refused rather than reported as a decode. */
                s.failed = true;
                return false;
            }

            /* The drain: one round per lane, retiring whatever is still
             * outstanding. A block that ends with copies pending is the normal
             * case rather than an error - up to 31 lanes can hold one. */
            for (uint32_t n = 0; n < kGDeflateNumStreams; n++) {
                if (st.copies[s.idx].pending && !GDeflateDoCopy(st, out)) {
                    return false;
                }
                GDeflateAdvance(s, page);
                if (s.failed) {
                    return false;
                }
            }
        }
    }
    if (!final_block) {
        s.failed = true;
        return false;
    }
    st.blocks = blocks_read;
    *out_len = out_pos;
    return true;
}

}  // namespace cudec_detail

#endif /* CUDEC_GDEFLATE_BLOCK_H */
