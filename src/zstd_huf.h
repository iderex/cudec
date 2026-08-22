/* The Zstd Huffman tree description and the decoding table it builds (issue
 * #220) - the entropy unit behind a Compressed literals section. Single-sourced
 * for host and device, the sibling of src/zstd_fse.h and src/zstd_bitstream.h.
 * Internal header, not part of the ABI.
 *
 * WHAT ARRIVES AND IN WHICH OF TWO SPELLINGS. A tree description is a header
 * byte followed by weights, one per symbol from zero upwards, and the header
 * byte chooses the spelling (RFC 8878 section 4.2.1.1). A value of 128 or more
 * means `value - 127` weights follow as 4-bit nibbles, high nibble first. A
 * value below 128 is the byte length of an FSE-compressed description whose
 * two-state interleaved run produces the weights, over a table whose accuracy
 * log may not exceed six.
 *
 * THE LAST WEIGHT IS NEVER WRITTEN DOWN AND THAT IS THE FAIL-CLOSED BRANCH.
 * Weight w costs 2^(w-1) of the tree's budget and weight zero costs nothing, so
 * the weights present sum to something below a power of two and the difference
 * is what the final symbol must be worth. The difference has to BE a power of
 * two - anything else describes a tree with a hole in it - and it has to leave
 * a legal weight. A decoder that took the nearest legal value instead of
 * refusing would build a table whose cells decode symbols the encoder never
 * wrote, which is the oversubscribed/incomplete class this unit exists to
 * refuse.
 *
 * THE TABLE IS NOT REORDERED AFTER IT IS BUILT. Symbols are placed by
 * increasing weight and, inside a weight, by increasing symbol value, and each
 * one occupies 2^(w-1) consecutive cells. That ordering IS the canonical code:
 * get it wrong and every cell still holds a legal symbol and a legal bit count,
 * so nothing downstream refuses and the output is simply wrong bytes.
 *
 * THE REFERENCE'S TABLE IS THE SAME TABLE AT A DIFFERENT SCALE, and the twin
 * test is where that is proven rather than asserted. libzstd rescales a tree
 * whose natural log is below eleven up to eleven so its decode loop has a
 * constant table size, which multiplies every entry 2^(11 - log) times and
 * leaves each entry's bit count untouched. What this header builds is the
 * natural table, because that is the one a kernel wants in shared memory: a
 * tree of log six is 64 cells here and 2048 there.
 *
 * The storage is the caller's throughout, sized by it and checked here, in the
 * shape src/zstd_fse.h uses: this header allocates nothing and a device
 * translation unit compiles it unchanged. */
#ifndef CUDEC_ZSTD_HUF_H
#define CUDEC_ZSTD_HUF_H

#include "cudec.h"
#include "zstd_bitstream.h"
#include "zstd_fse.h"

#include <stdint.h>

#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* HUF_SYMBOLVALUE_MAX in the reference's huf.h: a literals alphabet is indexed
 * by a byte, so this is the widest one the format can describe. */
constexpr unsigned kZstdHufMaxSymbolValue = 255;

/* HUF_TABLELOG_MAX in the reference's huf.h. It is the ceiling on the tree's
 * own depth, which is also the widest code the table can hold. RFC 8878 states
 * the eleven-bit limit for the streams a compressor emits; the reference reads
 * up to twelve here and refuses the twelfth one level up, so the wider value is
 * what this unit's default ceiling carries and the caller may narrow it. */
constexpr unsigned kZstdHufMaxTableLog = 12;

/* The accuracy-log ceiling on the FSE table that spells a compressed weight
 * description. The reference passes six to FSE_decompress_wksp from
 * HUF_readStats_body, and it is not the sequence tables' ceiling. */
constexpr unsigned kZstdHufWeightAccuracyLogMax = 6;

/* The header byte at or above this value selects the direct nibble spelling. */
constexpr unsigned kZstdHufDirectHeaderBase = 128;

/* The reject ladder, enumerated once, in the shape src/zstd_fse.h and
 * src/zstd_bitstream.h use: every refusal returns through one choke point
 * naming its rung, so the twin requires a negative per rung instead of counting
 * statuses that repeat. The reference collapses most of these into
 * corruption_detected, so the rung is this tree's own vocabulary. */
enum ZstdHufReject {
    kZstdHufRejectNone = 0,
    /* An argument outside what the format can express, or storage the caller
     * sized below what it asked to have decoded. A caller bug rather than a
     * stream, refused rather than clamped. */
    kZstdHufRejectBadRequest,
    kZstdHufRejectEmptyDescription,
    /* The description's own declared length reaches past the bytes supplied. */
    kZstdHufRejectDescriptionTruncated,
    /* More weights than the alphabet can hold. The implied last symbol needs a
     * slot of its own, so the written weights stop one short of the alphabet. */
    kZstdHufRejectTooManyWeights,
    /* A weight above the tree's depth ceiling. */
    kZstdHufRejectWeightTooLarge,
    /* Every written weight is zero, so the tree has no budget at all. */
    kZstdHufRejectWeightSumZero,
    /* The written weights already fill or overflow a power of two, leaving the
     * implied last symbol nothing, or the remainder is not a clean power of
     * two. Both are the same statement about the tree: it does not close. */
    kZstdHufRejectLastWeightNotClean,
    /* The depth the weights imply is past the ceiling in force. */
    kZstdHufRejectTableLogTooLarge,
    /* A tree with no leaf at its deepest rank. The reference refuses the same
     * thing in the same place.
     *
     * IT ALSO REFUSES AN ODD NUMBER OF THEM AND THIS UNIT DOES NOT, because
     * that state cannot arrive here. Every weight above one contributes an even
     * amount to the budget and the budget lands on a power of two before this
     * check is reached, so the number of weight-one symbols has the parity of
     * that power of two, which is even. A guard for the odd case would be one
     * no negative could reach, and the twin proves the reachable half instead
     * of asserting the unreachable one. */
    kZstdHufRejectDeepestRankEmpty,
    /* Build side: the cell array the caller supplied is smaller than the table
     * the depth asks for. */
    kZstdHufRejectBuildCapacity,
    /* Build side: the weights handed in do not describe a table that fills
     * exactly. Unreachable from a description this header decoded; reachable
     * from a caller that builds a weight array by hand. */
    kZstdHufRejectBuildWeightsNotClean,
    kZstdHufRejectCount
};

CUDEC_HOST_DEVICE inline cudec_status ZstdHufRefuse(ZstdHufReject rung,
                                                    cudec_status status,
                                                    ZstdHufReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* Position of the highest set bit, 0-indexed. Only ever called on a non-zero
 * value - every caller refuses zero before reaching it - so there is no "no
 * bits set" answer to define. A counted loop rather than an intrinsic, because
 * this header compiles for both the host and the device. */
CUDEC_HOST_DEVICE inline unsigned ZstdHufHighestSetBit(uint32_t value) {
    unsigned highest = 0;
    /* The bound is the accumulator's width, named rather than spelled: the
     * structural scanner refuses a bare 32 in an expression because that is
     * how a wave width gets hardcoded, and it refused this line. The constant
     * is src/zstd_fse.h's rather than a second one of the same value. */
    for (unsigned bit = 0; bit < kZstdFseWordBits; bit++) {
        if ((value >> bit) & 1u) {
            highest = bit;
        }
    }
    return highest;
}

/* One decoding-table cell: which symbol the code at this index spells and how
 * many bits of it were real. Laid out to match the reference's HUF_DEltX1 field
 * for field so the twin diffs cells rather than a summary. */
struct ZstdHufCell {
    uint8_t symbol;
    uint8_t nb_bits;
};

/* The working storage the compressed spelling needs, owned by the caller for
 * the reason every other buffer here is: this header compiles __device__
 * unchanged, and half a kilobyte of per-thread local memory declared inside a
 * function is a decision the kernel has to be able to make instead.
 *
 * THE WEIGHT ALPHABET IS THE TREE'S DEPTH AND NOT A BYTE, which is what keeps
 * this small. A weight above the depth ceiling is refused whatever spelling it
 * arrives in, so an FSE table that could emit one buys nothing; the reference
 * decodes weights over the full 256-symbol alphabet and refuses the same values
 * one step later, at its own weight check. The verdicts agree and the rungs
 * differ, which the twin pins rather than leaves to be assumed. */
struct ZstdHufWeightScratch {
    int16_t counts[kZstdHufMaxTableLog + 1];
    ZstdFseCell cells[1u << kZstdHufWeightAccuracyLogMax];
    uint16_t symbol_next[kZstdHufMaxTableLog + 1];
};

/* Decodes a Huffman tree description.
 *
 * `weights` receives one entry per symbol INCLUDING the implied last one, so a
 * caller sizes it for the alphabet it is willing to accept. `out_count` is how
 * many of them were written, which is the number of symbols in the tree, and
 * `out_table_log` is the tree's depth - the length of its shortest code is
 * `out_table_log + 1 - weight`.
 *
 * `out_consumed` is the byte count the description occupied. The literals
 * section positions its Huffman stream from it, so an off-by-one there is a
 * stream read from the wrong place rather than a refusal, which is why it is an
 * output of this function and not something a caller recomputes.
 *
 * `size` is the bytes remaining in the literals section, not the description's
 * length: the length is in the header byte for one spelling and only known
 * after the FSE run for the other.
 *
 * `max_table_log` is the ceiling the caller wants enforced on the tree's depth,
 * at most kZstdHufMaxTableLog. */
CUDEC_HOST_DEVICE inline cudec_status ZstdHufReadWeights(
    const unsigned char* src, uint64_t size, unsigned max_symbol_count,
    unsigned max_table_log, ZstdHufWeightScratch* scratch, uint8_t* weights,
    unsigned* out_count, unsigned* out_table_log, uint64_t* out_consumed,
    ZstdHufReject* reject) {
    *out_count = 0;
    *out_table_log = 0;
    *out_consumed = 0;
    if (reject != 0) {
        *reject = kZstdHufRejectNone;
    }
    if (src == 0 || weights == 0 || scratch == 0 || max_symbol_count < 2u ||
        max_symbol_count > kZstdHufMaxSymbolValue + 1 || max_table_log == 0 ||
        max_table_log > kZstdHufMaxTableLog) {
        return ZstdHufRefuse(kZstdHufRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    if (size == 0) {
        return ZstdHufRefuse(kZstdHufRejectEmptyDescription,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    const unsigned header = src[0];
    unsigned written = 0;
    uint64_t consumed = 0;

    if (header >= kZstdHufDirectHeaderBase) {
        /* The direct spelling. The count is in the header byte itself, so the
         * payload's length is known before a byte of it is read. */
        written = header - (kZstdHufDirectHeaderBase - 1);
        const uint64_t payload = (written + 1u) / 2u;
        if (payload + 1u > size) {
            return ZstdHufRefuse(kZstdHufRejectDescriptionTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        /* One slot is owed to the implied last symbol, so the written weights
         * stop one short of the caller's alphabet rather than filling it. */
        if (written + 1u > max_symbol_count) {
            return ZstdHufRefuse(kZstdHufRejectTooManyWeights,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        for (unsigned i = 0; i < written; i++) {
            const unsigned byte = src[1 + i / 2u];
            weights[i] = static_cast<uint8_t>((i % 2u) == 0 ? (byte >> 4)
                                                            : (byte & 0x0Fu));
        }
        /* An odd count leaves the final low nibble unread. The reference reads
         * it into a slot it then ignores, so a non-zero value there is accepted
         * by both and is not a rung. */
        consumed = payload + 1u;
    } else {
        /* The FSE spelling. The header byte is the compressed length, so the
         * run's input is bounded before it starts. */
        const uint64_t payload = header;
        if (payload + 1u > size) {
            return ZstdHufRefuse(kZstdHufRejectDescriptionTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        unsigned fse_max_symbol = 0;
        unsigned accuracy_log = 0;
        uint64_t description = 0;
        ZstdFseReject fse_rung = kZstdFseRejectNone;
        cudec_status status = ZstdFseReadNCount(
            src + 1, payload, kZstdHufMaxTableLog,
            kZstdHufWeightAccuracyLogMax, scratch->counts, &fse_max_symbol,
            &accuracy_log, &description, &fse_rung);
        if (status != CUDEC_OK) {
            return ZstdHufRefuse(kZstdHufRejectDescriptionTruncated, status,
                                 reject);
        }
        status = ZstdFseBuildDTable(scratch->counts, fse_max_symbol,
                                    accuracy_log, scratch->cells,
                                    static_cast<uint32_t>(1u << accuracy_log),
                                    scratch->symbol_next, &fse_rung);
        if (status != CUDEC_OK) {
            return ZstdHufRefuse(kZstdHufRejectDescriptionTruncated, status,
                                 reject);
        }
        ZstdBitReader reader;
        reader.src = src + 1 + description;
        reader.size = payload - description;
        status = reader.Start();
        if (status != CUDEC_OK) {
            return ZstdHufRefuse(kZstdHufRejectDescriptionTruncated, status,
                                 reject);
        }
        /* One short of the alphabet, for the implied last symbol's slot. The
         * two-state run refuses rather than truncates when it fills, which is
         * what turns a description claiming too many weights into a rung. */
        uint32_t produced = 0;
        status = ZstdFseDecode2State(
            &reader, scratch->cells, static_cast<uint32_t>(1u << accuracy_log),
            accuracy_log, weights, max_symbol_count - 1u, &produced, &fse_rung);
        if (status != CUDEC_OK) {
            if (fse_rung == kZstdFseRejectOutputFull) {
                return ZstdHufRefuse(kZstdHufRejectTooManyWeights,
                                     CUDEC_ERR_CORRUPT_INPUT, reject);
            }
            return ZstdHufRefuse(kZstdHufRejectDescriptionTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        written = produced;
        consumed = payload + 1u;
    }

    /* The budget the written weights spend, and the depth it implies. Weight
     * zero is a symbol the tree does not carry and costs nothing. */
    uint32_t total = 0;
    for (unsigned i = 0; i < written; i++) {
        if (weights[i] > max_table_log) {
            return ZstdHufRefuse(kZstdHufRejectWeightTooLarge,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        /* Bounded by construction rather than by hope: at most 256 weights,
         * each worth at most 2^11 here, so the sum stays inside 32 bits with
         * room to spare and no overflow check is owed. */
        total += (1u << weights[i]) >> 1;
    }
    if (total == 0) {
        return ZstdHufRefuse(kZstdHufRejectWeightSumZero,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const unsigned table_log = ZstdHufHighestSetBit(total) + 1u;
    if (table_log > max_table_log) {
        return ZstdHufRefuse(kZstdHufRejectTableLogTooLarge,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    /* The implied last weight. `total` is below 2^table_log by construction of
     * table_log, so the remainder is at least one and the subtraction cannot
     * wrap; what is not guaranteed is that it is a single power of two, and a
     * remainder that is not describes a tree with a hole in it. */
    const uint32_t rest = (1u << table_log) - total;
    const unsigned last_weight = ZstdHufHighestSetBit(rest) + 1u;
    if ((1u << (last_weight - 1u)) != rest) {
        return ZstdHufRefuse(kZstdHufRejectLastWeightNotClean,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    /* THE SLOT FOR THE IMPLIED SYMBOL IS ALREADY OWED AND THERE IS NO SECOND
     * CHECK HERE. Both spellings stop one short of the caller's alphabet: the
     * direct one refuses a count that would not leave the slot before it writes
     * a nibble, and the compressed one is given a capacity of
     * max_symbol_count - 1 and refuses rather than truncating when it fills. So
     * `written` is at most max_symbol_count - 1 on every path that reaches this
     * line, and a guard for the other case would be one no negative could
     * reach. */
    weights[written] = static_cast<uint8_t>(last_weight);

    /* The deepest rank of a tree that closes holds at least two leaves: its
     * codes pair up into the rank above, so a lone one is a tree that cannot
     * have been built by merging. See the rung for why the count's parity is
     * not checked here. */
    unsigned deepest_rank = 0;
    for (unsigned i = 0; i <= written; i++) {
        if (weights[i] == 1u) {
            deepest_rank++;
        }
    }
    if (deepest_rank < 2u) {
        return ZstdHufRefuse(kZstdHufRejectDeepestRankEmpty,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    *out_count = written + 1u;
    *out_table_log = table_log;
    *out_consumed = consumed;
    return CUDEC_OK;
}

/* Builds the decoding table a validated weight array describes.
 *
 * `cells` receives 1 << table_log entries. A symbol of weight w spells a code
 * of `table_log + 1 - w` bits and owns 2^(w-1) consecutive cells; a symbol of
 * weight zero is absent and owns none.
 *
 * The weights are re-checked here rather than trusted from the read: this entry
 * point is callable with an array no description produced, and the placement
 * walk's bound is the table size only while the weights fill it exactly. */
CUDEC_HOST_DEVICE inline cudec_status ZstdHufBuildDTable(
    const uint8_t* weights, unsigned count, unsigned table_log,
    ZstdHufCell* cells, uint32_t cell_capacity, ZstdHufReject* reject) {
    if (reject != 0) {
        *reject = kZstdHufRejectNone;
    }
    if (weights == 0 || cells == 0 || count == 0 ||
        count > kZstdHufMaxSymbolValue + 1 || table_log == 0 ||
        table_log > kZstdHufMaxTableLog) {
        return ZstdHufRefuse(kZstdHufRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    const uint32_t table_size = 1u << table_log;
    if (cell_capacity < table_size) {
        return ZstdHufRefuse(kZstdHufRejectBuildCapacity,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }

    /* How many symbols sit at each weight, and where each weight's run starts.
     * Counting first is what lets the placement walk below write every cell
     * exactly once without searching. */
    uint32_t rank_count[kZstdHufMaxTableLog + 1];
    for (unsigned w = 0; w <= kZstdHufMaxTableLog; w++) {
        rank_count[w] = 0;
    }
    uint32_t claimed = 0;
    for (unsigned i = 0; i < count; i++) {
        /* THIS IS THE BOUND ON THE COUNTER ABOVE AND NOT A RESTATEMENT OF THE
         * BUDGET CHECK BELOW. A weight is a byte and the counter array is as
         * long as the deepest tree the format admits, so an unchecked weight
         * indexes past it - and the shift that follows would be by a distance
         * wider than its own type. It carries its own rung for the same reason:
         * with the budget's rung the two would be indistinguishable, and a
         * change that deleted this check would still look refused. */
        if (weights[i] > table_log) {
            return ZstdHufRefuse(kZstdHufRejectWeightTooLarge,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        rank_count[weights[i]]++;
        claimed += (1u << weights[i]) >> 1;
    }
    if (claimed != table_size) {
        return ZstdHufRefuse(kZstdHufRejectBuildWeightsNotClean,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    /* Cells are handed out by INCREASING weight, so the LONGEST codes sit at
     * the bottom of the table and the shortest at the top. That direction is
     * the format's, not a preference: a code of n bits is read as the top n
     * bits of the index, so the codes that share a prefix have to be adjacent
     * in the order their prefix orders them. Getting it backwards leaves every
     * cell holding a legal symbol and a legal length, which is why the twin
     * diffs cells against the reference instead of checking the table is
     * well formed. */
    uint32_t rank_start[kZstdHufMaxTableLog + 1];
    rank_start[0] = 0; /* weight zero owns nothing; kept defined, not used */
    uint32_t next = 0;
    for (unsigned w = 1u; w <= table_log; w++) {
        rank_start[w] = next;
        next += rank_count[w] * ((1u << w) >> 1);
    }

    /* One pass over the symbols in increasing order, which is what makes the
     * order inside a weight the symbol order. Every cell of the run gets the
     * same symbol and the same bit count: a code of that length reaches all of
     * them, and the bits below it are the next code's, not this one's. */
    for (unsigned symbol = 0; symbol < count; symbol++) {
        const unsigned weight = weights[symbol];
        if (weight == 0) {
            continue;
        }
        const uint32_t span = (1u << weight) >> 1;
        const uint8_t nb_bits = static_cast<uint8_t>(table_log + 1u - weight);
        uint32_t at = rank_start[weight];
        for (uint32_t step = 0; step < span; step++) {
            cells[at + step].symbol = static_cast<uint8_t>(symbol);
            cells[at + step].nb_bits = nb_bits;
        }
        rank_start[weight] = at + span;
    }
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_HUF_H */
