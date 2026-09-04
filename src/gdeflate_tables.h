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
 * are two more each: 3144 bytes per warp for the two live tables against the
 * 3200-byte budget, where 13.1 computed 3132. What 13.1 did not cost is the
 * rest of a dynamic block's table set, which every lane reads and therefore
 * lives beside them in shared memory: the precode table, the code-length
 * vector and the precode's own lengths, 691 bytes more (src/gdeflate_decode.cuh
 * carries the set). Measured at the kernel (issue #214), 3832 bytes per warp
 * and 77 registers put six blocks on an sm_86 multiprocessor, 24 resident
 * warps against the 32 the budget was derived for; the perf pass (#204,
 * #205) is where either number moves.
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
 * compressor produces.
 *
 * THE ROUNDS ARE WRITTEN ONCE, OVER A TEAM (issue #214). The code-length rounds
 * at the bottom of this file and the block loop in src/gdeflate_block.h are
 * templates over a `Team`: the thing that says which lane is current, hands
 * out output positions in lane order, finds the first lane that flagged, and
 * refills the lanes that stepped. GDeflateHostTeam below is the sequential
 * residency, one thread walking the 32 lanes in order over a GDeflateSchedule,
 * and it is what every CPU twin runs. The warp residency in
 * src/gdeflate_decode.cuh answers the same calls with a ballot, a prefix sum
 * and a shuffle. What the rounds may assume of a team, and what a team may
 * assume of the rounds, is written at GDeflateHostTeam, because that is the
 * one that can be read without a device. */
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
 * and refusing is the answer that cannot be mistaken for a decode.
 *
 * `why` is where the rung is named. This runs on a length vector with no
 * schedule in reach - a test builds a table out of bytes it wrote itself, and
 * so does the static block - so the ladder branch goes into the caller's slot
 * and the caller holding a schedule raises it with GDeflateRefuseAs. A null
 * slot is a caller that does not want the reason, never a caller that gets a
 * table it should not have. */
template <uint32_t kCapSyms, uint32_t kRootBits, uint32_t kMaxLen>
CUDEC_HOST_DEVICE inline bool GDeflateBuildTable(
    const unsigned char* lens, uint32_t num_syms,
    GDeflateHuffTable<kCapSyms, kRootBits, kMaxLen>& t,
    GDeflateReject* why = nullptr) {
    static_assert(kRootBits <= kMaxLen,
                  "a root wider than the longest codeword would index slots "
                  "no code can reach");
    if (num_syms > kCapSyms) {
        return GDeflateTableRefuse(why, kGDeflateRejectTablePastCapacity);
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
            return GDeflateTableRefuse(why, kGDeflateRejectTableLengthPastMax);
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
        return GDeflateTableRefuse(why, kGDeflateRejectTableOverSubscribed);
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
            return GDeflateTableRefuse(why, kGDeflateRejectTableIncomplete);
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

/* Resolve one symbol out of a lane's buffer WITHOUT consuming it. `r` carries
 * how far ahead of the lane the read has gone and the rung it refused on, so a
 * caller can resolve a codeword and the extra bits behind it as one
 * speculative read and then remove `r.consumed` bits at once - or remove
 * nothing, which is how a lane that must not have consumed after an
 * end-of-block keeps the bits it entered the round with (docs/MASTERPLAN.md
 * section 13.2). The accelerator is a peek, so a miss has cost nothing and the
 * walk starts at length 1 rather than part-way through a codeword.
 *
 * EVERY EXIT THAT IS NOT A SYMBOL NAMES A RUNG INTO `r.why`. A lane that cannot
 * supply the bits a resolved codeword needs is refused by the occupancy check
 * the sequential remove uses rather than allowed to shift a buffer it does not
 * hold, and a walk that reaches the maximum length without matching has read a
 * codeword that the code does not contain, which on a complete code means the
 * bits were not produced by this table. */
template <uint32_t kCapSyms, uint32_t kRootBits, uint32_t kMaxLen>
CUDEC_HOST_DEVICE inline uint32_t GDeflateResolveSymbol(
    const GDeflateHuffTable<kCapSyms, kRootBits, kMaxLen>& t, uint64_t bitbuf,
    uint32_t bitsleft, GDeflateLaneRead& r) {
    if (r.why != kGDeflateRejectNone) {
        return kGDeflateNoSymbol;
    }
    if (t.kind == kGDeflateTableEmpty) {
        /* Stricter than the reference, deliberately: it hands back a symbol
         * the stream never encoded, and a distance built from that symbol is
         * indistinguishable from one the page asked for. */
        GDeflateTableRefuse(&r.why, kGDeflateRejectEmptyTableUsed);
        return kGDeflateNoSymbol;
    }
    if (t.kind == kGDeflateTableSingle) {
        if (!GDeflateRemoveCheck(bitsleft - r.consumed, 1u, &r.why)) {
            return kGDeflateNoSymbol;
        }
        r.consumed += 1u;
        return t.sorted[0];
    }
    /* The probe reads the buffer past the lane's occupancy, as the sequential
     * peek always did: the bits above `bitsleft` are zero by the schedule's
     * invariant, so a short lane resolves to some entry and the occupancy
     * check below is what refuses it. */
    const uint64_t window = bitbuf >> r.consumed;
    const uint32_t probe =
        static_cast<uint32_t>(window & ((1u << kRootBits) - 1u));
    const uint16_t entry = t.root[probe];
    const uint32_t hit_len = entry & kGDeflateRootLenMask;
    if (hit_len != 0) {
        if (!GDeflateRemoveCheck(bitsleft - r.consumed, hit_len, &r.why)) {
            return kGDeflateNoSymbol;
        }
        r.consumed += hit_len;
        return entry >> kGDeflateRootLenBits;
    }
    uint32_t code = 0;
    for (uint32_t len = 1; len <= kMaxLen; len++) {
        const uint32_t bit = GDeflateReadAhead(bitbuf, bitsleft, r, 1u);
        if (r.why != kGDeflateRejectNone) {
            return kGDeflateNoSymbol;
        }
        code = (code << 1) | bit;
        if (t.count[len] != 0 && code >= t.first_code[len] &&
            (code - t.first_code[len]) < t.count[len]) {
            return t.sorted[t.first_index[len] + (code - t.first_code[len])];
        }
    }
    GDeflateTableRefuse(&r.why, kGDeflateRejectCodewordNotInCode);
    return kGDeflateNoSymbol;
}

/* Decode one symbol from the current lane, consuming it: the resolve above
 * followed by the remove, with the rung the resolve named raised on the
 * schedule. This is the sequential shape the twins drive; the rounds below
 * resolve first and commit later, and go through the same resolve. */
template <class S, uint32_t kCapSyms, uint32_t kRootBits, uint32_t kMaxLen>
CUDEC_HOST_DEVICE inline uint32_t GDeflateDecodeSymbol(
    S& s, const GDeflateHuffTable<kCapSyms, kRootBits, kMaxLen>& t) {
    if (s.failed) {
        return kGDeflateNoSymbol;
    }
    GDeflateLaneRead r = {0, kGDeflateRejectNone};
    const uint32_t sym = GDeflateResolveSymbol(t, s.Buf(), s.Left(), r);
    if (r.why != kGDeflateRejectNone) {
        GDeflateRefuseAs(s, r.why);
        return kGDeflateNoSymbol;
    }
    GDeflateRemove(s, r.consumed);
    return sym;
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
 * read before the lane steps rather than after it.
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

/* The answer FirstLane gives when no lane flagged. One past the last lane, so
 * that `lane < first` is true of every lane and `lane == first` of none. */
constexpr uint32_t kGDeflateNoLane = kGDeflateNumStreams;

/* THE SEQUENTIAL TEAM, and the contract both residencies keep.
 *
 * What a team owes the rounds: Bits() is the calling lane's residency of the
 * schedule primitives (buffer, occupancy, verdict, word count); Copy() its
 * deferred match; Lane() its index; Active() whether it takes part in the
 * round in progress. LitLen(), Dist(), Precode(), Lens() and PrecodeLens() are
 * the block's table set, one per team. Round(n, f) runs f once per lane on the
 * first n lanes and then refills every lane that STEPPED - f returns whether
 * its lane did, because a lane that took no step is not refilled by the
 * format: the code-length rounds stop mid-round when the vector is full and
 * the lanes after that point keep their bits. Lane0(f) runs f on lane 0 alone
 * with no refill, which is the block header and the stored length; EnsureLane0
 * is the reference's ENSURE_BITS on lane 0 with no rotation. Build(f) runs f
 * once for the team, which is the table construction. Reset() returns the
 * sequential residency to lane 0 and is nothing on the warp.
 *
 * What the rounds owe a team, and it is the discipline that makes the warp
 * residency sound: inside f, every collective - Claim, FirstLane, CopyBytes,
 * Serial, Broadcast - is called exactly once per lane per round, at the same
 * point, by every lane whether or not it is Active(), with a neutral argument
 * from a lane that has nothing to contribute. The sequential residency does
 * not need that, because it runs one lane at a time; the warp residency needs
 * it because a collective is a point every lane of the wave must reach.
 *
 * The collectives are PREFIX operations by definition: Claim(base, len) hands
 * this lane the position after every LOWER lane's claim and advances `base`
 * past every lane's; FirstLane(flag) names the lowest flagging lane at or below
 * this one, or kGDeflateNoLane. On the warp both are computed over the whole
 * team at once, so FirstLane may name a HIGHER lane there; every comparison
 * the rounds make against its answer - below it, at it, above it - reads the
 * same on both, which is what lets one round body serve two residencies.
 *
 * CopyBytes(active, out, dst, src, len) performs the copies of every lane that
 * asked, in lane order, and a copy reads only bytes that lie before the
 * reservation it fills, so the order is the whole of what the deferred copy
 * needs. Serial(active, f) runs f on the asking lanes in lane order with each
 * one's writes visible to the next, which is what the code-length expansion
 * needs for "repeat the previous length".
 *
 * On the sequential residency a refusal stops the round at the lane that
 * refused, exactly as the reference's loop stops; on the warp every lane
 * finishes the round and the lowest refusing lane's rung is the team's, which
 * is the same verdict because no lane's step depends on a higher lane. */
class GDeflateHostTeam {
   public:
    CUDEC_HOST_DEVICE GDeflateHostTeam(GDeflateSchedule& s,
                                       const unsigned char* page,
                                       GDeflateCodeLengths* lens,
                                       GDeflateLitLenTable* litlen,
                                       GDeflateDistTable* dist)
        : s_(s),
          page_(page),
          lens_(lens),
          litlen_(litlen),
          dist_(dist),
          first_(kGDeflateNoLane) {
        for (uint32_t n = 0; n < kGDeflateNumStreams; n++) {
            copies_[n].out_pos = 0;
            copies_[n].length = 0;
            copies_[n].pending = false;
        }
    }

    CUDEC_HOST_DEVICE GDeflateSchedule& Bits() { return s_; }
    CUDEC_HOST_DEVICE GDeflateDeferredCopy& Copy() { return copies_[s_.idx]; }
    CUDEC_HOST_DEVICE uint32_t Lane() const { return s_.idx; }
    CUDEC_HOST_DEVICE bool Active() const { return true; }
    CUDEC_HOST_DEVICE bool Failed() const { return s_.failed; }
    CUDEC_HOST_DEVICE uint64_t Cursor() const { return s_.cursor; }
    CUDEC_HOST_DEVICE uint64_t WordCount() const { return s_.word_count; }
    CUDEC_HOST_DEVICE const unsigned char* Page() const { return page_; }

    CUDEC_HOST_DEVICE GDeflateLitLenTable& LitLen() { return *litlen_; }
    CUDEC_HOST_DEVICE GDeflateDistTable& Dist() { return *dist_; }
    CUDEC_HOST_DEVICE GDeflatePrecodeTable& Precode() { return precode_; }
    CUDEC_HOST_DEVICE GDeflateCodeLengths& Lens() { return *lens_; }
    CUDEC_HOST_DEVICE unsigned char* PrecodeLens() { return precode_lens_; }

    /* The priming round is GDeflateInit's, so the sequential team primes by
     * calling it: one word into each lane in lane order, refused before any
     * lane is touched when the page cannot carry it. */
    CUDEC_HOST_DEVICE bool Prime(uint64_t page_bytes) {
        return GDeflateInit(s_, page_, page_bytes);
    }

    template <class F>
    CUDEC_HOST_DEVICE void Round(uint32_t lanes, F f) {
        first_ = kGDeflateNoLane;
        for (uint32_t l = 0; l < lanes; l++) {
            s_.idx = l;
            const bool stepped = f(*this);
            if (s_.failed) {
                return;
            }
            if (stepped) {
                GDeflateAdvance(s_, page_);
                if (s_.failed) {
                    return;
                }
            }
        }
    }

    template <class F>
    CUDEC_HOST_DEVICE void Lane0(F f) {
        s_.idx = 0;
        f(*this);
    }

    CUDEC_HOST_DEVICE void EnsureLane0() {
        s_.idx = 0;
        GDeflateEnsure(s_, page_);
    }

    CUDEC_HOST_DEVICE void Reset() { GDeflateReset(s_); }

    template <class F>
    CUDEC_HOST_DEVICE void Build(F f) {
        f(*this);
    }

    template <class T>
    CUDEC_HOST_DEVICE T Broadcast(T value) const {
        return value;
    }

    CUDEC_HOST_DEVICE uint64_t Claim(uint64_t& base, uint64_t len) {
        const uint64_t pos = base;
        base += len;
        return pos;
    }

    CUDEC_HOST_DEVICE uint32_t FirstLane(bool flag) {
        if (flag && first_ == kGDeflateNoLane) {
            first_ = s_.idx;
        }
        return first_;
    }

    /* Byte by byte, forwards, which is what an overlapping match MEANS: a
     * distance below the length is a run that repeats what this copy is
     * itself writing. The warp residency copies the same bytes through the
     * closed-form gather the LZ4 kernel uses, and the two agree byte for byte
     * because every source byte of an overlapping copy is a byte this copy
     * wrote earlier in that same order (docs/DETERMINISM.md). */
    CUDEC_HOST_DEVICE void CopyBytes(bool active, unsigned char* out,
                                     uint64_t dst, uint64_t src,
                                     uint64_t len) {
        if (!active) {
            return;
        }
        for (uint64_t n = 0; n < len; n++) {
            out[dst + n] = out[src + n];
        }
    }

    template <class F>
    CUDEC_HOST_DEVICE void Serial(bool active, F f) {
        if (active) {
            f(*this);
        }
    }

    CUDEC_HOST_DEVICE uint64_t BufferedBits() const {
        return GDeflateBufferedBits(s_);
    }

    CUDEC_HOST_DEVICE void ClearCopies() {
        for (uint32_t n = 0; n < kGDeflateNumStreams; n++) {
            copies_[n].pending = false;
            copies_[n].length = 0;
            copies_[n].out_pos = 0;
        }
    }

   private:
    GDeflateSchedule& s_;
    const unsigned char* page_;
    GDeflateCodeLengths* lens_;
    GDeflateLitLenTable* litlen_;
    GDeflateDistTable* dist_;
    GDeflatePrecodeTable precode_;
    unsigned char precode_lens_[kGDeflateNumPrecodeSyms];
    GDeflateDeferredCopy copies_[kGDeflateNumStreams];
    uint32_t first_;
};

/* Read a dynamic block's code-length vectors, entered immediately after BTYPE
 * has been popped off lane 0. Returns false with the team failed for every
 * shape this decoder refuses, and the team's Lens() is not to be read in that
 * case.
 *
 * NOTHING HERE IS SIZED FROM THE STREAM. HLIT and HDIST are five-bit fields
 * and HCLEN is four, so the three counts are bounded by the field widths
 * rather than by a check: the static assertions below say that the capacities
 * this file declares are exactly the values those fields reach, which is the
 * form that fails at compile time if a capacity is ever changed alone. */
template <class Team>
CUDEC_HOST_DEVICE inline bool GDeflateReadCodeLengthRounds(Team& t) {
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
    t.EnsureLane0();
    if (t.Failed()) {
        return false;
    }

    /* The three counts ride lane 0 and every lane needs them, so they travel
     * as one packed word: the two five-bit fields and the four-bit one fit in
     * fourteen bits, and one broadcast is what a lane-0 read costs the team. */
    uint32_t header = 0;
    t.Lane0([&](Team& lane0) {
        auto& b = lane0.Bits();
        const uint32_t num_litlen = GDeflatePop(b, 5);
        const uint32_t num_dist = GDeflatePop(b, 5);
        const uint32_t num_explicit = GDeflatePop(b, 4);
        header = num_litlen | (num_dist << 5) | (num_explicit << 10);
    });
    header = t.Broadcast(header);
    if (t.Failed()) {
        return false;
    }
    const uint32_t num_litlen = (header & 0x1Fu) + 257u;
    const uint32_t num_dist = ((header >> 5) & 0x1Fu) + 1u;
    const uint32_t num_explicit = ((header >> 10) & 0xFu) + 4u;
    t.EnsureLane0();
    if (t.Failed()) {
        return false;
    }

    /* Every precode length the header does not state is zero: HCLEN says how
     * many of the nineteen are present, and an absent one is absent from the
     * code rather than undefined. Each explicit one is its own round on its
     * own lane, lane i holding the i-th in the permuted order. */
    t.Build([&](Team& one) {
        unsigned char* precode_lens = one.PrecodeLens();
        for (uint32_t i = 0; i < kGDeflateNumPrecodeSyms; i++) {
            precode_lens[i] = 0;
        }
    });
    t.Round(num_explicit, [&](Team& lane) -> bool {
        if (!lane.Active()) {
            return false;
        }
        const uint32_t len = GDeflatePop(lane.Bits(), 3);
        if (!lane.Bits().failed) {
            lane.PrecodeLens()[GDeflatePrecodeOrder(lane.Lane())] =
                static_cast<unsigned char>(len);
        }
        return true;
    });
    if (t.Failed()) {
        return false;
    }

    t.Build([&](Team& one) {
        GDeflateReject precode_why = kGDeflateRejectNone;
        if (!GDeflateBuildTable(one.PrecodeLens(), kGDeflateNumPrecodeSyms,
                                one.Precode(), &precode_why)) {
            /* The flag, not just the return. A precode that is not a code is
             * bad input like any other refusal here, and this header's own
             * contract says a caller that checks the sticky flag once at the
             * end reads the same verdict as one that checks after every call
             * - so a refusal that left the flag clear would break that
             * contract silently. Found by fuzz/fuzz_gdeflate_tables.cpp on its
             * first seed replay. */
            GDeflateRefuseAs(one.Bits(), precode_why);
        }
    });
    if (t.Failed()) {
        return false;
    }

    /* The expansion starts at a reset exactly as the header did, so the first
     * length rides lane 0. */
    t.Reset();
    const uint32_t total = num_litlen + num_dist;
    /* One length or one run per lane per round, and a lane whose position is
     * already past the vector takes no step at all: the reference's loop
     * stops at that lane, and the lanes after it keep their bits. Every
     * round either advances `filled` by at least one or refuses, and the run
     * bound below is what keeps the write inside the alphabet; the round cap
     * is the vector's own length, since a round that fills nothing refuses. */
    uint64_t filled = 0;
    for (uint32_t round = 0; round <= total && filled < total; round++) {
        t.Round(kGDeflateNumStreams, [&](Team& lane) -> bool {
            GDeflateLaneRead r = {0, kGDeflateRejectNone};
            GDeflateReject extra_why = kGDeflateRejectNone;
            uint32_t presym = kGDeflateNoSymbol;
            uint64_t count = 0;
            unsigned char value = 0;
            bool repeat_prev = false;
            if (lane.Active()) {
                const uint64_t buf = lane.Bits().Buf();
                const uint32_t left = lane.Bits().Left();
                presym = GDeflateResolveSymbol(lane.Precode(), buf, left, r);
                if (r.why == kGDeflateRejectNone) {
                    if (presym < kGDeflateRepeatPrev) {
                        count = 1;
                        value = static_cast<unsigned char>(presym);
                    } else {
                        uint32_t extra = 0;
                        if (presym == kGDeflateRepeatPrev) {
                            repeat_prev = true;
                            extra = 3u + GDeflateReadAhead(buf, left, r, 2);
                        } else if (presym == kGDeflateRepeatZeroShort) {
                            extra = 3u + GDeflateReadAhead(buf, left, r, 3);
                        } else {
                            extra = 11u + GDeflateReadAhead(buf, left, r, 7);
                        }
                        /* The codeword resolved; a shortage here is the extra
                         * bits', which the reference refuses AFTER the
                         * nothing-to-repeat check below, so it is held apart
                         * from the codeword's rung and raised in that order. */
                        extra_why = r.why;
                        count = (extra_why == kGDeflateRejectNone) ? extra : 0;
                    }
                } else {
                    count = 0;
                }
            }
            const uint64_t start = lane.Claim(filled, count);
            const bool commit = lane.Active() && start < total;
            if (commit) {
                auto& b = lane.Bits();
                if (r.why != kGDeflateRejectNone) {
                    GDeflateRefuseAs(b, r.why);
                } else if (repeat_prev && start == 0) {
                    /* Nothing to repeat. The reference refuses this as well,
                     * and it is the one repeat-code refusal the two decoders
                     * share. */
                    GDeflateRefuse(b, kGDeflateRejectRepeatNothingBefore);
                } else if (extra_why != kGDeflateRejectNone) {
                    GDeflateRefuseAs(b, extra_why);
                } else if (count > total - start) {
                    /* See the block comment above: refused rather than
                     * absorbed into slack this alphabet does not carry.
                     * Written as a subtraction on the side that cannot
                     * overflow - `start` is below `total` here, so
                     * `total - start` is what is left rather than a sum that
                     * could wrap. */
                    GDeflateRefuse(b, kGDeflateRejectRepeatRunPastAlphabet);
                } else {
                    GDeflateRemove(b, r.consumed);
                }
            }
            const bool writes = commit && !lane.Bits().failed;
            lane.Serial(writes, [&](Team& one) {
                unsigned char* lens = one.Lens().lens;
                if (repeat_prev) {
                    value = lens[start - 1];
                }
                for (uint64_t n = 0; n < count; n++) {
                    lens[start + n] = value;
                }
            });
            return commit;
        });
        if (t.Failed()) {
            return false;
        }
    }
    t.Build([&](Team& one) {
        one.Lens().num_litlen = num_litlen;
        one.Lens().num_dist = num_dist;
    });
    return true;
}

/* The rung above: the two vectors read, then the two tables built from them.
 * Split from the read so the code-length round is provable on its own - a
 * vector recovered wrongly and a vector that merely fails to build a table are
 * different failures, and one entry point would report them as one. */
template <class Team>
CUDEC_HOST_DEVICE inline bool GDeflateReadDynamicTableRounds(Team& t) {
    if (!GDeflateReadCodeLengthRounds(t)) {
        return false;
    }
    t.Build([&](Team& one) {
        GDeflateReject why = kGDeflateRejectNone;
        const GDeflateCodeLengths& lens = one.Lens();
        if (!GDeflateBuildTable(lens.lens, lens.num_litlen, one.LitLen(),
                                &why)) {
            /* A vector that is not a code is bad input, so the schedule
             * carries the same verdict a bad round would leave: a caller that
             * checks only the flag must not read this as a page still worth
             * decoding. The rung is the one the construction named, not a
             * second one saying which of the two vectors carried it: what
             * refused is the property of the lengths, and a rung per call
             * site would be two rungs one negative could not tell apart. */
            GDeflateRefuseAs(one.Bits(), why);
            return;
        }
        if (!GDeflateBuildTable(lens.lens + lens.num_litlen, lens.num_dist,
                                one.Dist(), &why)) {
            GDeflateRefuseAs(one.Bits(), why);
        }
    });
    return !t.Failed();
}

/* The sequential entry points, which the twins and the fuzz targets drive:
 * the rounds above over the host team, on a schedule positioned immediately
 * after BTYPE has been popped off lane 0. */
inline bool GDeflateReadCodeLengths(GDeflateSchedule& s,
                                    const unsigned char* page,
                                    GDeflateCodeLengths& out) {
    GDeflateHostTeam t(s, page, &out, nullptr, nullptr);
    return GDeflateReadCodeLengthRounds(t);
}

inline bool GDeflateReadDynamicTables(GDeflateSchedule& s,
                                      const unsigned char* page,
                                      GDeflateCodeLengths& lens,
                                      GDeflateLitLenTable& litlen,
                                      GDeflateDistTable& dist) {
    GDeflateHostTeam t(s, page, &lens, &litlen, &dist);
    return GDeflateReadDynamicTableRounds(t);
}

}  // namespace cudec_detail

#endif /* CUDEC_GDEFLATE_TABLES_H */
