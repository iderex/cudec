/* Canonical Huffman table construction for GDeflate (issue #175): a code
 * length vector in, a decode table out, and a refusal for every vector the
 * reference refuses. Single-sourced for host and device, the sibling of
 * src/gdeflate_schedule.h, src/lz4_block.h and src/snappy_block.h. It reads no
 * bits of its own beyond the decode helper at the bottom and knows nothing
 * about block headers beyond the dynamic block's own code-length rounds
 * (#176): what a decoded literal/length or distance symbol MEANS is #182's,
 * and so is every other block type.
 *
 * WHY THIS IS THE PIECE THAT GETS ITS OWN REVIEW. A decode table is the one
 * place in a DEFLATE-family decoder where attacker-chosen numbers become an
 * index. The vector is read from the stream, so its length distribution is
 * chosen by whoever wrote the page, and every historical DEFLATE memory-safety
 * defect lives in the gap between "these lengths describe a code" and "these
 * lengths describe something, so let us index with it". GDeflate carries no
 * checksum anywhere (docs/MASTERPLAN.md section 11.4), so a table built from a
 * vector that is not a code produces plausible bytes rather than a caught
 * error.
 *
 * THE LAYOUT IS THE ONE SECTION 13.1 SETTLED, AND ITS SHAPE IS THE ARGUMENT.
 * A canonical decoder over per-length counts, first codes and first indices is
 * complete on its own and its size is fixed by the alphabet rather than by the
 * lengths, so no input can change how much memory a warp holds - which is what
 * killed the two-level table, whose worst legal case needs 2504 lit/len
 * entries. The root arrays here are accelerators over that decoder and not a
 * second decoder: a code no longer than the root resolves in one lookup, a
 * longer one falls back to the length-by-length walk, and both paths return
 * the same symbol by construction because both read the same three arrays.
 *
 * WHAT IT COSTS AGAINST 13.1'S BUDGET, because the table there was computed
 * for 318 symbols and this holds 320. The draft says up to 286 literal/length
 * symbols (dossier 11.2); the reference reads HLIT as five bits and adds 257,
 * so it admits up to 288 and builds a table over all of them. Parity with the
 * reference is the rule the whole M4 ladder rests on, so the capacity follows
 * the reference. Two extra symbols are four bytes, and the two `kind` fields
 * are two more each: 3144 bytes per warp against the 3200-byte budget, where
 * 13.1 computed 3132. The margin narrows from 68 bytes to 56 and the occupancy
 * arithmetic is unmoved.
 *
 * FAIL-CLOSED, AND EXACTLY WHERE THE REFERENCE IS. `build_decode_table` in the
 * pinned fork refuses an over-subscribed vector, refuses an incomplete one,
 * and admits two named exceptions to incompleteness: the empty code, and a
 * code whose whole codespace is one symbol at length 1. Both exceptions are
 * reachable in a valid stream, so both are accepted here rather than tightened
 * away - a distance code with one distance is what a page full of one match
 * distance produces. The empty code is where this header is STRICTER than the
 * reference and says so: the reference fills its table with a synthetic
 * "symbol 0, length 1" entry so that a malformed stream reading from the
 * unused codespace still lands on an initialised entry, which means a match in
 * a block whose distance code is empty decodes to a distance symbol that the
 * stream never encoded. Here that table refuses on use instead. A block that
 * declares no distances and uses none is unaffected, which is the only shape a
 * compressor produces. */
#ifndef CUDEC_GDEFLATE_TABLES_H
#define CUDEC_GDEFLATE_TABLES_H

#include "gdeflate_schedule.h"

#include <stdint.h>

namespace cudec_detail {

/* The reference's own constants (deflate_constants.h in the pinned
 * NVIDIA/libdeflate fork), restated here because this header is the thing that
 * has to agree with them. The literal/length capacity is the reference's 288
 * rather than the draft's 286, for the reason the file comment gives. */
constexpr uint32_t kGDeflateMaxCodeLen = 15;
constexpr uint32_t kGDeflateNumLitLenSyms = 288;
constexpr uint32_t kGDeflateNumDistSyms = 32;

/* Root widths from docs/MASTERPLAN.md section 13.1: 10 bits over the
 * literal/length alphabet and 7 over the distance alphabet, which is the pair
 * that fits the per-warp budget. They are accelerator widths, so moving one
 * changes a footprint and never an answer - #204 is where that is measured. */
constexpr uint32_t kGDeflateLitLenRootBits = 10;
constexpr uint32_t kGDeflateDistRootBits = 7;

/* A root slot packs the symbol above the codeword length. Length 0 is the miss
 * marker rather than a legal answer: a zero-length codeword does not exist, so
 * no symbol can collide with the sentinel. The widest symbol index is 287, so
 * the shifted field needs nine bits above the four length bits and the entry
 * stays 16 bits wide, which is what the section 13.1 footprint is costed at. */
constexpr uint32_t kGDeflateRootLenBits = 4;
constexpr uint32_t kGDeflateRootLenMask = (1u << kGDeflateRootLenBits) - 1u;
static_assert(kGDeflateMaxCodeLen <= kGDeflateRootLenMask,
              "the packed length field must hold the longest legal codeword");
static_assert((kGDeflateNumLitLenSyms - 1) <=
                  (0xFFFFu >> kGDeflateRootLenBits),
              "the packed symbol field must hold the highest symbol index");

/* What a vector turned out to be. `kGDeflateTableComplete` is the only kind
 * that decodes through the arrays; the other two are the reference's two named
 * exceptions to completeness and are handled at their own branch in the decode
 * helper, because neither has a codespace to walk. */
enum GDeflateTableKind {
    kGDeflateTableComplete = 0,
    kGDeflateTableSingle = 1,
    kGDeflateTableEmpty = 2
};

/* `kCapSyms` is the alphabet the table is sized for and `kMaxLen` is the
 * longest codeword the code admits, which is 15 for the literal/length and
 * distance codes and 7 for the precode. The codespace comparison is against
 * `kMaxLen`, so the two are not interchangeable: a vector that is a complete
 * code under one is over-subscribed or incomplete under the other. */
template <uint32_t kCapSyms, uint32_t kRootBits, uint32_t kMaxLen>
struct GDeflateHuffTable {
    /* Symbols in canonical order - by increasing codeword length, then by
     * increasing symbol value - with the zero-length symbols left out. */
    uint16_t sorted[kCapSyms];
    /* Indexed by codeword length, 0 through kMaxLen. `count[0]` is the number
     * of unused symbols and is not part of any code. */
    uint16_t count[kMaxLen + 1];
    uint16_t first_code[kMaxLen + 1];
    uint16_t first_index[kMaxLen + 1];
    uint16_t root[1u << kRootBits];
    uint16_t kind;
};

using GDeflateLitLenTable =
    GDeflateHuffTable<kGDeflateNumLitLenSyms, kGDeflateLitLenRootBits,
                      kGDeflateMaxCodeLen>;
using GDeflateDistTable =
    GDeflateHuffTable<kGDeflateNumDistSyms, kGDeflateDistRootBits,
                      kGDeflateMaxCodeLen>;

/* Reverse the low `len` bits of `code`. DEFLATE packs a codeword so that its
 * most significant bit arrives first, and the lane bit buffer hands bits back
 * least significant first, so the value a root lookup is indexed by is the
 * codeword read backwards. This is the one place the two orders meet; the
 * length-by-length walk below never needs it because it rebuilds the codeword
 * one arriving bit at a time. */
CUDEC_HOST_DEVICE inline uint32_t GDeflateReverseBits(uint32_t code,
                                                      uint32_t len) {
    uint32_t out = 0;
    for (uint32_t i = 0; i < len; i++) {
        out = (out << 1) | ((code >> i) & 1u);
    }
    return out;
}

/* Build the decode table for `num_syms` code lengths. Returns false and leaves
 * the table unusable for every vector the reference refuses, and for the two
 * it refuses that the format cannot produce (a length past `kMaxLen`, an
 * alphabet past the capacity) - both are caller errors rather than bad input,
 * and refusing is the answer that cannot be mistaken for a decode. */
template <uint32_t kCapSyms, uint32_t kRootBits, uint32_t kMaxLen>
CUDEC_HOST_DEVICE inline bool GDeflateBuildTable(
    const unsigned char* lens, uint32_t num_syms,
    GDeflateHuffTable<kCapSyms, kRootBits, kMaxLen>& t) {
    static_assert(kRootBits <= kMaxLen,
                  "a root wider than the longest codeword would index slots "
                  "no code can reach");
    if (num_syms > kCapSyms) {
        return false;
    }
    for (uint32_t len = 0; len <= kMaxLen; len++) {
        t.count[len] = 0;
        t.first_code[len] = 0;
        t.first_index[len] = 0;
    }
    t.kind = kGDeflateTableComplete;
    for (uint32_t sym = 0; sym < num_syms; sym++) {
        const uint32_t len = lens[sym];
        if (len > kMaxLen) {
            /* The format cannot deliver this - explicit lengths arrive as
             * precode symbols below 16 and the precode's own lengths are three
             * bits - so it is a caller error, and it is refused rather than
             * truncated because a truncated length silently describes a
             * different code. */
            return false;
        }
        t.count[len] = static_cast<uint16_t>(t.count[len] + 1u);
    }

    /* Codespace out of 2^kMaxLen, accumulated the way the reference
     * accumulates it: the running total shifted left once per length. An
     * over-subscribed vector overshoots deliberately - that overshoot is the
     * refusal below - so the accumulator has to hold more than a full
     * codespace without wrapping, and the worst case is every symbol at length
     * 1, which reaches `num_syms << (kMaxLen - 1)`. The reference carries the
     * same bound as a static assertion and so does this. */
    static_assert(kCapSyms <= (0xFFFFFFFFu >> (kMaxLen - 1)),
                  "the codespace accumulator must not wrap on the most "
                  "over-subscribed vector the alphabet admits");
    uint32_t codespace = 0;
    for (uint32_t len = 1; len <= kMaxLen; len++) {
        codespace = (codespace << 1) + t.count[len];
    }
    const uint32_t full = 1u << kMaxLen;
    if (codespace > full) {
        /* Over-subscribed: the lengths claim more of the codespace than
         * exists, so at least two symbols share a prefix and no assignment
         * exists. */
        return false;
    }

    /* Canonical order, and the first index of each length. Both are needed by
     * every branch below, including the two degenerate ones, so they are
     * filled before the completeness question is answered. */
    uint32_t next = 0;
    for (uint32_t len = 1; len <= kMaxLen; len++) {
        t.first_index[len] = static_cast<uint16_t>(next);
        next += t.count[len];
    }
    /* first_index doubles as the fill cursor and is wound back by each
     * length's own count afterwards. A separate scratch array would be
     * per-thread local memory in the kernel, in the one place where occupancy
     * is the binding resource (section 13.1), for a value the table already
     * holds. */
    for (uint32_t sym = 0; sym < num_syms; sym++) {
        const uint32_t len = lens[sym];
        if (len != 0) {
            t.sorted[t.first_index[len]++] = static_cast<uint16_t>(sym);
        }
    }
    for (uint32_t len = 1; len <= kMaxLen; len++) {
        t.first_index[len] =
            static_cast<uint16_t>(t.first_index[len] - t.count[len]);
    }

    /* Cleared before the completeness question rather than after it, so no
     * accepted table ever carries an uninitialised root - the degenerate kinds
     * do not read it, and a struct with a live field nothing wrote is the
     * thing a later reader has to reason about instead of trust. */
    for (uint32_t i = 0; i < (1u << kRootBits); i++) {
        t.root[i] = 0;
    }

    if (codespace < full) {
        /* Incomplete, which the reference admits in exactly two shapes and
         * refuses otherwise. Both are pinned against the reference in
         * tests/gdeflate_tables_twin.cpp rather than read out of its source. */
        if (codespace == 0) {
            t.kind = kGDeflateTableEmpty;
            return true;
        }
        if (codespace != (full >> 1) || t.count[1] != 1) {
            return false;
        }
        /* One symbol at length 1. The reference gives it both codewords, 0 and
         * 1, so a single bit resolves it whichever way that bit falls; the
         * decode helper does the same, which is why no root entry is written
         * for this kind. */
        t.kind = kGDeflateTableSingle;
        return true;
    }

    /* Complete. First codes are the canonical recurrence, and they are what
     * the walk compares against. */
    uint32_t code = 0;
    for (uint32_t len = 1; len <= kMaxLen; len++) {
        t.first_code[len] = static_cast<uint16_t>(code);
        code = (code + t.count[len]) << 1;
    }

    /* The root accelerator. A codeword of length L owns every slot whose low L
     * bits are its reversed codeword, which is 2^(kRootBits - L) slots spaced
     * 2^L apart - the same spacing the reference's own table has, and for the
     * same reason. Slots left at zero are the misses that fall through to the
     * walk. */
    for (uint32_t len = 1; len <= kRootBits; len++) {
        for (uint32_t n = 0; n < t.count[len]; n++) {
            const uint32_t sym = t.sorted[t.first_index[len] + n];
            const uint32_t rev =
                GDeflateReverseBits(t.first_code[len] + n, len);
            const uint16_t entry = static_cast<uint16_t>(
                (sym << kGDeflateRootLenBits) | len);
            for (uint32_t i = rev; i < (1u << kRootBits); i += (1u << len)) {
                t.root[i] = entry;
            }
        }
    }
    return true;
}

/* The one sentinel a decode may return. 0xFFFF is outside every alphabet this
 * header sizes for, and every caller of the helper below is required to test
 * for it: the schedule's own failure flag is set at the same moment, so a
 * caller that checks either one is correct and a caller that checks neither is
 * caught by the flag on its next operation. */
constexpr uint32_t kGDeflateNoSymbol = 0xFFFFu;

/* Decode one symbol from the current lane. The accelerator is a peek, so a
 * miss has consumed nothing and the walk starts at length 1 rather than
 * part-way through a codeword.
 *
 * EVERY EXIT THAT IS NOT A SYMBOL SETS THE SCHEDULE'S FAILURE FLAG. A lane
 * that cannot supply the bits a resolved codeword needs is refused by
 * GDeflateRemove rather than allowed to shift a buffer it does not hold, and a
 * walk that reaches the maximum length without matching has read a codeword
 * that the code does not contain, which on a complete code means the bits were
 * not produced by this table. */
template <uint32_t kCapSyms, uint32_t kRootBits, uint32_t kMaxLen>
CUDEC_HOST_DEVICE inline uint32_t GDeflateDecodeSymbol(
    GDeflateSchedule& s,
    const GDeflateHuffTable<kCapSyms, kRootBits, kMaxLen>& t) {
    if (s.failed) {
        return kGDeflateNoSymbol;
    }
    if (t.kind == kGDeflateTableEmpty) {
        /* Stricter than the reference, deliberately: it hands back a symbol
         * the stream never encoded, and a distance built from that symbol is
         * indistinguishable from one the page asked for. */
        s.failed = true;
        return kGDeflateNoSymbol;
    }
    if (t.kind == kGDeflateTableSingle) {
        GDeflateRemove(s, 1);
        if (s.failed) {
            return kGDeflateNoSymbol;
        }
        return t.sorted[0];
    }
    const uint32_t probe = GDeflatePeek(s, kRootBits);
    const uint16_t entry = t.root[probe];
    const uint32_t hit_len = entry & kGDeflateRootLenMask;
    if (hit_len != 0) {
        GDeflateRemove(s, hit_len);
        if (s.failed) {
            return kGDeflateNoSymbol;
        }
        return entry >> kGDeflateRootLenBits;
    }
    uint32_t code = 0;
    for (uint32_t len = 1; len <= kMaxLen; len++) {
        const uint32_t bit = GDeflatePop(s, 1);
        if (s.failed) {
            return kGDeflateNoSymbol;
        }
        code = (code << 1) | bit;
        if (t.count[len] != 0 && code >= t.first_code[len] &&
            (code - t.first_code[len]) < t.count[len]) {
            return t.sorted[t.first_index[len] + (code - t.first_code[len])];
        }
    }
    s.failed = true;
    return kGDeflateNoSymbol;
}


/* THE DYNAMIC BLOCK'S CODE-LENGTH ROUNDS (issue #176). Everything above turns
 * a length vector into a table; this turns a page into the length vectors. It
 * is in this file rather than beside the block loop because the vector is read
 * THROUGH a canonical Huffman code built by the routine above - the precode is
 * a table like any other - and a second copy of that construction is the one
 * thing #175 forbids by name.
 *
 * THE ROUND STRUCTURE IS THE FORMAT AND IT IS NOT RFC 1951'S. In DEFLATE these
 * fields are one serial bit stream. Here HLIT, HDIST and HCLEN ride lane 0
 * with no round between them, each precode length is its own round, and each
 * expanded length - a repeat code and its extra bits together - is its own
 * round as well. An extra-bit field read in the round AFTER its code would
 * decode a different page and raise no error, which is why the extra bits are
 * popped before the Advance rather than after it.
 *
 * WHERE THIS IS STRICTER THAN THE REFERENCE, STATED RATHER THAN LEFT TO BE
 * FOUND. A repeat run reaching past HLIT + HDIST is refused here. The
 * reference does not refuse it: it keeps 137 slack entries after the alphabet
 * precisely so the worst overrun - 138 zeroes with one length left to fill -
 * writes into memory it owns, and the entries past the alphabet are then never
 * read. That is a decoder ABSORBING a malformed vector rather than accepting a
 * legal one: no compressor emits it, and RFC 1951 section 3.2.7 gives the run
 * no meaning past the end of the alphabet. Carrying the slack would spend 137
 * bytes of the per-warp budget section 13.1 measures on bytes nothing may
 * read. The divergence is executed, with the reference's own answer beside it,
 * in tests/gdeflate_header_twin.cpp. */

constexpr uint32_t kGDeflateNumPrecodeSyms = 19;
constexpr uint32_t kGDeflateMaxPrecodeLen = 7;

/* The precode's root width is its maximum codeword length, so every codeword
 * it can carry resolves in one lookup and the length-by-length walk is
 * unreachable on it. The reference rests on the same equality - its
 * PRECODE_TABLEBITS static assertion - and this is that statement in the place
 * that has to agree with it. */
using GDeflatePrecodeTable =
    GDeflateHuffTable<kGDeflateNumPrecodeSyms, kGDeflateMaxPrecodeLen,
                      kGDeflateMaxPrecodeLen>;

/* The order the precode's own lengths are stored in - data rather than a
 * derivation, and the reference's `deflate_precode_lens_permutation`. A
 * permutation reorders the same multiset, so every completeness test answers
 * identically under the permuted and the unpermuted reading (measured as a
 * surviving mutant on #170). Only lengths ASSIGNED TO SYMBOLS separate the
 * two, which is what this file produces and what that one could not. */
CUDEC_HOST_DEVICE inline uint32_t GDeflatePrecodeOrder(uint32_t i) {
    const unsigned char kOrder[kGDeflateNumPrecodeSyms] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    return kOrder[i];
}

/* The three repeat codes, named so the decode below reads as the format
 * rather than as arithmetic over magic numbers. */
constexpr uint32_t kGDeflateRepeatPrev = 16;
constexpr uint32_t kGDeflateRepeatZeroShort = 17;
constexpr uint32_t kGDeflateRepeatZeroLong = 18;

/* The concatenated literal/length and distance length vector, and where the
 * cut between them falls. A caller-provided struct rather than a local array
 * because in the warp kernel these are 320 bytes that have to land in shared
 * memory: a local array would be per-thread local memory in the one place
 * where occupancy is the binding resource (docs/MASTERPLAN.md section 13.1). */
struct GDeflateCodeLengths {
    unsigned char lens[kGDeflateNumLitLenSyms + kGDeflateNumDistSyms];
    uint32_t num_litlen;
    uint32_t num_dist;
};

/* Read a dynamic block's code-length vectors, entered immediately after BTYPE
 * has been popped off lane 0. Returns false with the schedule failed for every
 * shape this decoder refuses, and `out` is not to be read in that case.
 *
 * NOTHING HERE IS SIZED FROM THE STREAM. HLIT and HDIST are five-bit fields
 * and HCLEN is four, so the three counts are bounded by the field widths
 * rather than by a check: the static assertions below say that the capacities
 * this file declares are exactly the values those fields reach, which is the
 * form that fails at compile time if a capacity is ever changed alone. */
CUDEC_HOST_DEVICE inline bool GDeflateReadCodeLengths(
    GDeflateSchedule& s, const unsigned char* page, GDeflateCodeLengths& out) {
    static_assert(kGDeflateNumLitLenSyms == 257 + ((1u << 5) - 1u),
                  "HLIT is five bits above 257, so the literal/length "
                  "capacity must be the highest value that field reaches");
    static_assert(kGDeflateNumDistSyms == 1 + ((1u << 5) - 1u),
                  "HDIST is five bits above 1, so the distance capacity must "
                  "be the highest value that field reaches");
    static_assert(kGDeflateNumPrecodeSyms == 4 + ((1u << 4) - 1u),
                  "HCLEN is four bits above 4, so the precode alphabet must "
                  "be the highest value that field reaches");

    /* The reference's ENSURE_BITS after BTYPE. Idempotent - a lane at or above
     * the watermark is left alone - so a caller that ensured already is not
     * charged a second word for calling this. */
    GDeflateEnsure(s, page);

    const uint32_t num_litlen = GDeflatePop(s, 5) + 257u;
    const uint32_t num_dist = GDeflatePop(s, 5) + 1u;
    const uint32_t num_explicit = GDeflatePop(s, 4) + 4u;
    if (s.failed) {
        return false;
    }
    GDeflateEnsure(s, page);

    /* Every precode length the header does not state is zero: HCLEN says how
     * many of the nineteen are present, and an absent one is absent from the
     * code rather than undefined. */
    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    for (uint32_t i = 0; i < kGDeflateNumPrecodeSyms; i++) {
        precode_lens[i] = 0;
    }
    for (uint32_t i = 0; i < num_explicit; i++) {
        precode_lens[GDeflatePrecodeOrder(i)] =
            static_cast<unsigned char>(GDeflatePop(s, 3));
        GDeflateAdvance(s, page);
    }
    if (s.failed) {
        return false;
    }

    GDeflatePrecodeTable precode;
    if (!GDeflateBuildTable(precode_lens, kGDeflateNumPrecodeSyms, precode)) {
        return false;
    }

    /* The expansion starts at a reset exactly as the header did, so the first
     * length rides lane 0. */
    GDeflateReset(s);
    const uint32_t total = num_litlen + num_dist;
    /* The increment lives in the body because a round writes one length or a
     * run of them, and which it was is not known until the symbol is decoded.
     * It terminates on every input all the same: each pass either writes at
     * least one length or returns, and the run bound below is what keeps the
     * write inside the alphabet. */
    for (uint32_t i = 0; i < total;) {
        const uint32_t presym = GDeflateDecodeSymbol(s, precode);
        if (presym == kGDeflateNoSymbol) {
            return false;
        }
        if (presym < kGDeflateRepeatPrev) {
            out.lens[i++] = static_cast<unsigned char>(presym);
            GDeflateAdvance(s, page);
            continue;
        }

        unsigned char rep_val = 0;
        uint32_t rep_count = 0;
        if (presym == kGDeflateRepeatPrev) {
            if (i == 0) {
                /* Nothing to repeat. The reference refuses this as well, and
                 * it is the one repeat-code refusal the two decoders share. */
                s.failed = true;
                return false;
            }
            rep_val = out.lens[i - 1];
            rep_count = 3u + GDeflatePop(s, 2);
        } else if (presym == kGDeflateRepeatZeroShort) {
            rep_count = 3u + GDeflatePop(s, 3);
        } else {
            rep_count = 11u + GDeflatePop(s, 7);
        }
        if (s.failed) {
            return false;
        }
        if (rep_count > total - i) {
            /* See the block comment above: refused rather than absorbed into
             * slack this alphabet does not carry. Written as a subtraction on
             * the side that cannot overflow - `i` is below `total` here, so
             * `total - i` is what is left rather than a sum that could wrap. */
            s.failed = true;
            return false;
        }
        for (uint32_t n = 0; n < rep_count; n++) {
            out.lens[i++] = rep_val;
        }
        GDeflateAdvance(s, page);
    }
    if (s.failed) {
        return false;
    }
    out.num_litlen = num_litlen;
    out.num_dist = num_dist;
    return true;
}

/* The rung above: the two vectors read, then the two tables built from them.
 * Split from the read so the code-length round is provable on its own - a
 * vector recovered wrongly and a vector that merely fails to build a table are
 * different failures, and one entry point would report them as one. */
CUDEC_HOST_DEVICE inline bool GDeflateReadDynamicTables(
    GDeflateSchedule& s, const unsigned char* page, GDeflateCodeLengths& lens,
    GDeflateLitLenTable& litlen, GDeflateDistTable& dist) {
    if (!GDeflateReadCodeLengths(s, page, lens)) {
        return false;
    }
    if (!GDeflateBuildTable(lens.lens, lens.num_litlen, litlen)) {
        /* A vector that is not a code is bad input, so the schedule carries
         * the same verdict a bad round would leave: a caller that checks only
         * the flag must not read this as a page still worth decoding. */
        s.failed = true;
        return false;
    }
    if (!GDeflateBuildTable(lens.lens + lens.num_litlen, lens.num_dist, dist)) {
        s.failed = true;
        return false;
    }
    return true;
}

}  // namespace cudec_detail

#endif /* CUDEC_GDEFLATE_TABLES_H */
