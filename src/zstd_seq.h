/* The Zstd sequences section: the section header, the four table modes per
 * field, and the three-state interleaved decode loop that turns the block's
 * backward bitstream into (literals_length, match_length, offset_value)
 * triples. RFC 8878 sections 3.1.1.3.2.1 and 4.2. Single-sourced for host and
 * device, the sibling of src/zstd_fse.h and src/zstd_huf.h. Internal header,
 * not part of the ABI.
 *
 * WHAT THIS UNIT PRODUCES IS NOT YET AN OFFSET. `offset_value` is the format's
 * own quantity, (1 << Offset_Code) + extra bits, and the values 1, 2 and 3
 * name repeat offsets rather than distances. Resolving those against the
 * frame's three-entry repeat history is a separate unit with its own state, so
 * this one hands the raw value on and refuses to guess. A caller that treats
 * `offset_value` as a distance is off by three on every explicit offset and
 * wrong in kind on every repeated one.
 *
 * THE BASELINES ARE DERIVED, NOT TABULATED. Every Literals_Length_Code and
 * Match_Length_Code baseline is the running sum of the widths below it -
 * base(c + 1) = base(c) + (1 << extra(c)) - which is a property of the
 * format's own construction rather than a coincidence this file exploits. So
 * only the extra-bit widths are written down, and the table a reader has to
 * check against the RFC is a third of the size. The sum is recomputed per
 * lookup: this is the correctness twin, and a resident baseline table is a
 * performance shape that belongs with the kernel that needs it, exactly as the
 * FSE cell layout does.
 *
 * WHAT THE FAIL-CLOSED CASES ARE HERE. A sequence count is bounded by what the
 * block can regenerate before any bit is read, because every sequence emits at
 * least a minimum-length match and a count the block cannot hold is a demand
 * for output nobody will check later. A Repeat mode with no previously decoded
 * table is refused rather than resolved against whatever the cell array
 * happens to hold. And the bitstream must end exactly where the last sequence
 * leaves it: bits left over mean the stream and the count disagree, which is
 * the same fail-open the reference refuses with BIT_DStream_completed. */
#ifndef CUDEC_ZSTD_SEQ_H
#define CUDEC_ZSTD_SEQ_H

#include "cudec.h"
#include "zstd_bitstream.h"
#include "zstd_fse.h"

#include <stdint.h>

/* Guarded: the sibling decode headers define the same macro for the same
 * reason, and a device translation unit that decodes more than one format
 * includes more than one of them. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* The three fields, in the order their table descriptions appear in the
 * section - which is not the order their states are initialised in, and not
 * the order their extra bits are read in either. All three orders are the
 * format's; none of them is this file's choice. */
constexpr unsigned kZstdSeqFieldLitLen = 0;
constexpr unsigned kZstdSeqFieldOffset = 1;
constexpr unsigned kZstdSeqFieldMatchLen = 2;

/* Symbol_Compression_Mode, RFC 8878 section 3.1.1.3.2.1.1. */
constexpr unsigned kZstdSeqModePredefined = 0;
constexpr unsigned kZstdSeqModeRle = 1;
constexpr unsigned kZstdSeqModeCompressed = 2;
constexpr unsigned kZstdSeqModeRepeat = 3;

/* Alphabet sizes: 36 literals-length codes, 53 match-length codes, and 32
 * offset codes of which the predefined distribution reaches 29. */
constexpr unsigned kZstdSeqLitLenSymbolCount = 36;
constexpr unsigned kZstdSeqMatchLenSymbolCount = 53;
constexpr unsigned kZstdSeqOffsetSymbolCount = 32;
constexpr unsigned kZstdSeqOffsetPredefinedCount = 29;
/* The widest of the three, which is what a caller's scratch must hold. */
constexpr unsigned kZstdSeqMaxAlphabet = 53;

/* Accuracy logs of the three predefined distributions. */
constexpr unsigned kZstdSeqLitLenPredefinedLog = 6;
constexpr unsigned kZstdSeqOffsetPredefinedLog = 5;
constexpr unsigned kZstdSeqMatchLenPredefinedLog = 6;

/* The shortest match a sequence can express, which is what bounds the number
 * of sequences a block of a given regenerated size can hold. */
constexpr unsigned kZstdSeqMinMatchLength = 3;

/* Number_of_Sequences, RFC 8878 section 3.1.1.3.2.1: one byte below this, two
 * below the escape, three at it. */
constexpr unsigned kZstdSeqCountTwoByteMin = 128;
constexpr unsigned kZstdSeqCountThreeByteMark = 255;
constexpr uint32_t kZstdSeqCountThreeByteBias = 0x7F00u;

/* Codes below these carry no extra bits: a literals length is its own code up
 * to fifteen, and a match length is its code plus the minimum up to
 * thirty-one. */
constexpr unsigned kZstdSeqLitLenDirectCodes = 16;
constexpr unsigned kZstdSeqMatchLenDirectCodes = 32;

/* The reject ladder, in the shape src/zstd_fse.h uses: every refusal returns
 * through one choke point naming its rung, so the twin requires a negative per
 * rung instead of counting statuses that repeat. The reference collapses all
 * of these into corruption_detected, so the rung is this tree's vocabulary. */
enum ZstdSeqReject {
    kZstdSeqRejectNone = 0,
    /* An argument outside what the format can express, or a caller's buffer
     * that is null or too small. A caller bug rather than a stream. */
    kZstdSeqRejectBadRequest,
    /* The section ended inside the Number_of_Sequences varint. */
    kZstdSeqRejectCountTruncated,
    /* More sequences than the block's regenerated size can hold. */
    kZstdSeqRejectCountTooLarge,
    /* A count of zero followed by bytes the block has no reader for. With no
     * sequences the section is the count alone and the block's content ends
     * there, so anything after it is a byte nobody consumes - and a decoder
     * that ignored it would accept a block the reference calls corrupt while
     * producing the literals as if the stream were whole. Every other count
     * lands in the bitstream, which is required to be consumed exactly, so
     * this is the one arm where the leftover has nowhere else to be refused. */
    kZstdSeqRejectBlockNotConsumed,
    /* No Symbol_Compression_Modes byte after the count. */
    kZstdSeqRejectModesTruncated,
    /* The mode byte's low two bits are reserved and must be zero. */
    kZstdSeqRejectModesReserved,
    /* An RLE table description with no byte left to carry its symbol. */
    kZstdSeqRejectRleTruncated,
    /* A description naming a symbol outside the field's alphabet: an RLE byte
     * carrying one, or an FSE description reaching one. */
    kZstdSeqRejectSymbolPastMax,
    /* A code outside the field's alphabet coming OUT of a table. Separate from
     * the rung above because it refuses a different thing: no description this
     * unit accepts can produce one, so this is the cell array a caller built
     * elsewhere, and collapsing the two would leave whichever guard was
     * removed covered by the other one's negative. */
    kZstdSeqRejectDecodedSymbolPastMax,
    /* A Set_Compressed description this section could not decode, or a table
     * that could not be built from it. The FSE unit's own rung is where the
     * reason is; this one says which field carried it. */
    kZstdSeqRejectTableDescription,
    /* Repeat_Mode with no table decoded for that field yet. */
    kZstdSeqRejectRepeatWithoutTable,
    /* The section ended with no bytes left for the bitstream, or the
     * bitstream's final byte carries no start marker. */
    kZstdSeqRejectBitstreamMissing,
    /* The stream held fewer bits than the three initial states need. */
    kZstdSeqRejectStateInitTruncated,
    /* A state landed outside its table. Well-formed tables cannot produce one;
     * this refuses a cell array that came from somewhere else. */
    kZstdSeqRejectStateOutOfTable,
    /* The bits ran out inside a sequence: an extra-bit field or a state update
     * the stream could not pay for. */
    kZstdSeqRejectSequenceTruncated,
    /* Bits left over after the last sequence. The count and the stream
     * disagree, and which of the two is lying is not decidable here. */
    kZstdSeqRejectBitstreamNotConsumed,
    /* A table claiming more cells than its array holds. Its own rung rather
     * than the caller-argument one, because it is the bound every cell read
     * below rests on: a state is checked against `table_size` and nothing else
     * checks `table_size` against the allocation. */
    kZstdSeqRejectTableCapacity,
    kZstdSeqRejectCount
};

CUDEC_HOST_DEVICE inline cudec_status ZstdSeqRefuse(ZstdSeqReject rung,
                                                    cudec_status status,
                                                    ZstdSeqReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* The highest symbol each field's alphabet reaches. */
CUDEC_HOST_DEVICE inline unsigned ZstdSeqFieldSymbolMax(unsigned field) {
    if (field == kZstdSeqFieldLitLen) {
        return kZstdSeqLitLenSymbolCount - 1;
    }
    if (field == kZstdSeqFieldOffset) {
        return kZstdSeqOffsetSymbolCount - 1;
    }
    return kZstdSeqMatchLenSymbolCount - 1;
}

/* The per-field accuracy-log ceiling. Tighter than the FSE unit's own bound,
 * and the reference refuses a description above it one level up from where the
 * description is read - so it is checked here, where the field is known. */
CUDEC_HOST_DEVICE inline unsigned ZstdSeqFieldAccuracyLogMax(unsigned field) {
    if (field == kZstdSeqFieldLitLen) {
        return kZstdLitLenAccuracyLogMax;
    }
    if (field == kZstdSeqFieldOffset) {
        return kZstdOffsetAccuracyLogMax;
    }
    return kZstdMatchLenAccuracyLogMax;
}

/* Extra-bit widths, RFC 8878 section 3.1.1.3.2.1.1. Written as the format
 * states them: a run of codes carrying none, a short irregular middle, and a
 * tail whose width is the code's own index shifted by a constant. */
CUDEC_HOST_DEVICE inline unsigned ZstdSeqLitLenExtraBits(unsigned code) {
    constexpr unsigned kTailBias = 19;
    const unsigned middle[] = {1, 1, 1, 1, 2, 2, 3, 3, 4};
    const unsigned middle_count = sizeof(middle) / sizeof(middle[0]);
    if (code < kZstdSeqLitLenDirectCodes) {
        return 0;
    }
    if (code >= kZstdSeqLitLenDirectCodes + middle_count) {
        return code - kTailBias;
    }
    return middle[code - kZstdSeqLitLenDirectCodes];
}

CUDEC_HOST_DEVICE inline unsigned ZstdSeqMatchLenExtraBits(unsigned code) {
    constexpr unsigned kTailBias = 36;
    const unsigned middle[] = {1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5};
    const unsigned middle_count = sizeof(middle) / sizeof(middle[0]);
    if (code < kZstdSeqMatchLenDirectCodes) {
        return 0;
    }
    if (code >= kZstdSeqMatchLenDirectCodes + middle_count) {
        return code - kTailBias;
    }
    return middle[code - kZstdSeqMatchLenDirectCodes];
}

/* Baselines as the running sum of the widths below them. The loops are counted
 * over the alphabet, so their trip counts are compile-time bounded and no lane
 * can be held here by a hostile stream. */
CUDEC_HOST_DEVICE inline uint32_t ZstdSeqLitLenBaseline(unsigned code) {
    uint32_t base = 0;
    for (unsigned below = 0; below < code; below++) {
        base += 1u << ZstdSeqLitLenExtraBits(below);
    }
    return base;
}

CUDEC_HOST_DEVICE inline uint32_t ZstdSeqMatchLenBaseline(unsigned code) {
    uint32_t base = kZstdSeqMinMatchLength;
    for (unsigned below = 0; below < code; below++) {
        base += 1u << ZstdSeqMatchLenExtraBits(below);
    }
    return base;
}

/* One field's decoding table, plus the fact of it existing. `present` is what
 * Repeat_Mode reads, and it is deliberately not the same question as
 * `cells != 0`: a caller hands the same cell array to every block of a frame,
 * and a Repeat in the first block must be refused over an array that is
 * allocated and stale. */
struct ZstdSeqTable {
    ZstdFseCell* cells;
    uint32_t capacity;
    uint32_t table_size;
    /* Zero for an RLE table, which has one cell and reads no state bits. */
    unsigned accuracy_log;
    bool present;
};

/* The working memory a table description needs, the caller's for the reason
 * the FSE and Huffman units give: the kernel places it, and a header that
 * allocated would decide that placement. */
struct ZstdSeqScratch {
    int16_t counts[kZstdSeqMaxAlphabet];
    uint16_t symbol_next[kZstdSeqMaxAlphabet];
};

/* The three predefined distributions, RFC 8878 section 3.1.1.3.2.2, written
 * into the caller's count vector. Local rather than resident: they are read
 * once per Predefined field, and a namespace-scope array would have this
 * header decide where the device holds them instead of the kernel. */
CUDEC_HOST_DEVICE inline void ZstdSeqPredefinedCounts(unsigned field,
                                                      int16_t* counts,
                                                      unsigned* out_symbol_max,
                                                      unsigned* out_log) {
    if (field == kZstdSeqFieldLitLen) {
        const int16_t litlen[kZstdSeqLitLenSymbolCount] = {
            4, 3, 2, 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 1, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
        for (unsigned symbol = 0; symbol < kZstdSeqLitLenSymbolCount;
             symbol++) {
            counts[symbol] = litlen[symbol];
        }
        *out_symbol_max = kZstdSeqLitLenSymbolCount - 1;
        *out_log = kZstdSeqLitLenPredefinedLog;
        return;
    }
    if (field == kZstdSeqFieldOffset) {
        const int16_t offset[kZstdSeqOffsetPredefinedCount] = {
            1, 1, 1, 1, 1, 1, 2, 2, 2, 1,  1,  1,  1,  1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
        for (unsigned symbol = 0; symbol < kZstdSeqOffsetPredefinedCount;
             symbol++) {
            counts[symbol] = offset[symbol];
        }
        *out_symbol_max = kZstdSeqOffsetPredefinedCount - 1;
        *out_log = kZstdSeqOffsetPredefinedLog;
        return;
    }
    const int16_t matchlen[kZstdSeqMatchLenSymbolCount] = {
        1,  4,  3,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1, -1, -1, -1, -1, -1, -1};
    for (unsigned symbol = 0; symbol < kZstdSeqMatchLenSymbolCount; symbol++) {
        counts[symbol] = matchlen[symbol];
    }
    *out_symbol_max = kZstdSeqMatchLenSymbolCount - 1;
    *out_log = kZstdSeqMatchLenPredefinedLog;
}

/* What the section header declares. Where `sequence_count` is zero the section
 * ends at the count byte and the three modes are absent, so they are left at
 * Predefined and a caller must not read them. */
struct ZstdSeqSectionHeader {
    uint32_t sequence_count;
    unsigned litlen_mode;
    unsigned offset_mode;
    unsigned matchlen_mode;
};

/* Number_of_Sequences and the Symbol_Compression_Modes byte.
 *
 * `block_size_max` is the number of bytes this block may regenerate, and it is
 * the bound the declared count is checked against BEFORE any table is read.
 * Every sequence emits at least a minimum-length match, so a block of that
 * size holds at most that many sequences, and a larger count is a demand this
 * section cannot be describing. Checked here rather than downstream because
 * downstream is a per-sequence loop whose trip count this number is. */
CUDEC_HOST_DEVICE inline cudec_status ZstdParseSeqSectionHeader(
    const unsigned char* src, uint64_t size, uint64_t block_size_max,
    ZstdSeqSectionHeader* out, uint64_t* out_consumed, ZstdSeqReject* reject) {
    if (reject != 0) {
        *reject = kZstdSeqRejectNone;
    }
    if (src == 0 || out == 0 || out_consumed == 0) {
        return ZstdSeqRefuse(kZstdSeqRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    out->sequence_count = 0;
    out->litlen_mode = kZstdSeqModePredefined;
    out->offset_mode = kZstdSeqModePredefined;
    out->matchlen_mode = kZstdSeqModePredefined;
    *out_consumed = 0;
    if (size == 0) {
        return ZstdSeqRefuse(kZstdSeqRejectCountTruncated,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const unsigned first = src[0];
    uint32_t count = 0;
    uint64_t consumed = 1;
    if (first < kZstdSeqCountTwoByteMin) {
        count = first;
    } else if (first < kZstdSeqCountThreeByteMark) {
        if (size < 2) {
            return ZstdSeqRefuse(kZstdSeqRejectCountTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        count = ((static_cast<uint32_t>(first) - kZstdSeqCountTwoByteMin)
                 << kZstdBitsPerByte) +
                src[1];
        consumed = 2;
    } else {
        if (size < 3) {
            return ZstdSeqRefuse(kZstdSeqRejectCountTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        count = static_cast<uint32_t>(src[1]) +
                (static_cast<uint32_t>(src[2]) << kZstdBitsPerByte) +
                kZstdSeqCountThreeByteBias;
        consumed = 3;
    }
    /* Zero sequences ends the section: no mode byte, no tables, no bitstream.
     * The block is its literals and nothing else - and `size` is what is left
     * of the block, so it ends here too. A byte after the count has no reader:
     * the tables and the bitstream that would have consumed it do not exist in
     * this arm. Measured rather than argued from the text: the pinned
     * reference refuses such a block as corrupt, and this decoder used to
     * decode it to its literals. */
    if (count == 0) {
        if (size != consumed) {
            return ZstdSeqRefuse(kZstdSeqRejectBlockNotConsumed,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        *out_consumed = consumed;
        return CUDEC_OK;
    }
    if (count > block_size_max / kZstdSeqMinMatchLength) {
        return ZstdSeqRefuse(kZstdSeqRejectCountTooLarge,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    if (size <= consumed) {
        return ZstdSeqRefuse(kZstdSeqRejectModesTruncated,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const unsigned modes = src[consumed];
    consumed++;
    constexpr unsigned kModeMask = 3u;
    constexpr unsigned kLitLenModeShift = 6;
    constexpr unsigned kOffsetModeShift = 4;
    constexpr unsigned kMatchLenModeShift = 2;
    if ((modes & kModeMask) != 0) {
        /* Reserved, and the reference refuses rather than ignoring them: bits
         * with no meaning today are how a future field arrives, and a decoder
         * that skipped them would read tomorrow's stream as today's. */
        return ZstdSeqRefuse(kZstdSeqRejectModesReserved,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    out->sequence_count = count;
    out->litlen_mode = (modes >> kLitLenModeShift) & kModeMask;
    out->offset_mode = (modes >> kOffsetModeShift) & kModeMask;
    out->matchlen_mode = (modes >> kMatchLenModeShift) & kModeMask;
    *out_consumed = consumed;
    return CUDEC_OK;
}

/* Brings one field's table into being for this block, whichever of the four
 * modes the header named, and reports how many bytes of description it read.
 * A Repeat reads none and leaves the table exactly as the previous block left
 * it, which is why the table is in/out rather than out. */
CUDEC_HOST_DEVICE inline cudec_status ZstdSeqLoadTable(
    unsigned field, unsigned mode, const unsigned char* src, uint64_t size,
    ZstdSeqScratch* scratch, ZstdSeqTable* table, uint64_t* out_consumed,
    ZstdSeqReject* reject) {
    if (reject != 0) {
        *reject = kZstdSeqRejectNone;
    }
    if (src == 0 || scratch == 0 || table == 0 || table->cells == 0 ||
        out_consumed == 0 || field > kZstdSeqFieldMatchLen ||
        mode > kZstdSeqModeRepeat) {
        return ZstdSeqRefuse(kZstdSeqRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    *out_consumed = 0;
    const unsigned symbol_max = ZstdSeqFieldSymbolMax(field);

    if (mode == kZstdSeqModeRepeat) {
        if (!table->present) {
            return ZstdSeqRefuse(kZstdSeqRejectRepeatWithoutTable,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        return CUDEC_OK;
    }

    if (mode == kZstdSeqModeRle) {
        if (size == 0) {
            return ZstdSeqRefuse(kZstdSeqRejectRleTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        if (table->capacity == 0) {
            return ZstdSeqRefuse(kZstdSeqRejectBadRequest,
                                 CUDEC_ERR_INVALID_ARGUMENT, reject);
        }
        const unsigned symbol = src[0];
        if (symbol > symbol_max) {
            return ZstdSeqRefuse(kZstdSeqRejectSymbolPastMax,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        /* One cell, zero state bits: every sequence in the block takes this
         * symbol and the state never moves. An accuracy log of zero is what
         * makes the state initialisation below read no bits for this field. */
        table->cells[0].new_state = 0;
        table->cells[0].symbol = static_cast<uint8_t>(symbol);
        table->cells[0].nb_bits = 0;
        table->table_size = 1;
        table->accuracy_log = 0;
        table->present = true;
        *out_consumed = 1;
        return CUDEC_OK;
    }

    unsigned description_symbol_max = 0;
    unsigned accuracy_log = 0;
    if (mode == kZstdSeqModePredefined) {
        ZstdSeqPredefinedCounts(field, scratch->counts,
                                &description_symbol_max, &accuracy_log);
    } else {
        ZstdFseReject fse_reject = kZstdFseRejectNone;
        uint64_t consumed = 0;
        const cudec_status status = ZstdFseReadNCount(
            src, size, symbol_max, ZstdSeqFieldAccuracyLogMax(field),
            scratch->counts, &description_symbol_max, &accuracy_log, &consumed,
            &fse_reject);
        if (status != CUDEC_OK) {
            /* The FSE unit refuses a description reaching past the field's
             * alphabet through its own rung; reported here as the field-level
             * one, so a caller sees which field the section lost. */
            const ZstdSeqReject rung =
                fse_reject == kZstdFseRejectSymbolPastMax
                    ? kZstdSeqRejectSymbolPastMax
                    : kZstdSeqRejectTableDescription;
            return ZstdSeqRefuse(rung, status, reject);
        }
        *out_consumed = consumed;
    }

    ZstdFseReject build_reject = kZstdFseRejectNone;
    const cudec_status status = ZstdFseBuildDTable(
        scratch->counts, description_symbol_max, accuracy_log, table->cells,
        table->capacity, scratch->symbol_next, &build_reject);
    if (status != CUDEC_OK) {
        *out_consumed = 0;
        return ZstdSeqRefuse(kZstdSeqRejectTableDescription, status, reject);
    }
    table->table_size = 1u << accuracy_log;
    table->accuracy_log = accuracy_log;
    table->present = true;
    return CUDEC_OK;
}

/* One decoded sequence, in the format's own quantities. `offset_value` is not
 * a distance; see the note at the top of this file. */
struct ZstdSequence {
    uint32_t literals_length;
    uint32_t match_length;
    uint64_t offset_value;
};

/* Reads the symbol a state names, without moving it.
 *
 * ZstdFseNextSymbol is not reused here, and the reason is the loop's shape
 * rather than taste: it emits and updates in one call, and the sequences loop
 * has to read three fields' extra bits BETWEEN the emit and the update, and
 * must not update at all after the last sequence. Splitting the two halves is
 * what the format asks for; keeping them fused would mean telling the fused
 * function which half to skip. */
CUDEC_HOST_DEVICE inline cudec_status ZstdSeqPeekSymbol(
    const ZstdSeqTable* table, const ZstdFseState* state, unsigned* out_symbol,
    ZstdSeqReject* reject) {
    if (state->value >= table->table_size) {
        return ZstdSeqRefuse(kZstdSeqRejectStateOutOfTable,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    *out_symbol = table->cells[state->value].symbol;
    return CUDEC_OK;
}

/* Moves a state on by the bits its current cell asks for. */
CUDEC_HOST_DEVICE inline cudec_status ZstdSeqUpdateState(
    ZstdBitReader* reader, const ZstdSeqTable* table, ZstdFseState* state,
    ZstdSeqReject* reject) {
    if (state->value >= table->table_size) {
        return ZstdSeqRefuse(kZstdSeqRejectStateOutOfTable,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const ZstdFseCell cell = table->cells[state->value];
    uint64_t bits = 0;
    if (reader->ReadBits(cell.nb_bits, &bits) != CUDEC_OK) {
        return ZstdSeqRefuse(kZstdSeqRejectSequenceTruncated,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    state->value = cell.new_state + static_cast<uint32_t>(bits);
    return CUDEC_OK;
}

/* A sequence decode in progress: everything the three-state loop has to carry
 * from one tile of sequences to the next.
 *
 * WHY THE LOOP IS RESUMABLE AT ALL. A block may declare Block_Maximum_Size / 3
 * sequences, which is 43690 for a 128 KiB block, and the whole-set form below
 * fills an array of that many. `docs/MASTERPLAN.md` section 14.9 costs that
 * against the shared memory a fused kernel has and produces sequences 128 at a
 * time instead, so the loop needs a form whose reader and three states survive
 * the call. The whole-set form is written on top of this one rather than
 * beside it: a second copy of the three orders below is the one duplication
 * this file cannot afford.
 *
 * THE TABLES ARE HELD RATHER THAN RE-PASSED, AND THAT IS NOT A TRUST
 * DECISION. Their capacities are bounded once, when the decode starts, which
 * is exactly when the whole-set form bounds them; every cell read afterwards
 * is bounded by `table_size` at the read, which is where it always was. */
struct ZstdSeqCursor {
    ZstdBitReader reader;
    const ZstdSeqTable* litlen;
    const ZstdSeqTable* offset;
    const ZstdSeqTable* matchlen;
    ZstdFseState litlen_state;
    ZstdFseState offset_state;
    ZstdFseState matchlen_state;
    /* The block's declared count, and how many of them have been emitted. */
    uint32_t total;
    uint32_t emitted;
    /* Set by the tile that emits the last sequence, once the bitstream has
     * been held to ending exactly there. A cursor is only spent when this is
     * true; a decode that stopped early stopped on a refusal. */
    bool finished;
};

/* Starts a sequence decode: validates what the whole block's decode rests on,
 * reads the start marker, and initialises the three states.
 *
 * `src`/`size` are the bitstream alone: the section header and the table
 * descriptions have already been stepped over by the caller. */
CUDEC_HOST_DEVICE inline cudec_status ZstdSeqBegin(
    const unsigned char* src, uint64_t size, uint32_t sequence_count,
    const ZstdSeqTable* litlen, const ZstdSeqTable* offset,
    const ZstdSeqTable* matchlen, ZstdSeqCursor* cursor,
    ZstdSeqReject* reject) {
    if (reject != 0) {
        *reject = kZstdSeqRejectNone;
    }
    if (src == 0 || litlen == 0 || offset == 0 || matchlen == 0 ||
        cursor == 0 || sequence_count == 0 || litlen->cells == 0 ||
        offset->cells == 0 || matchlen->cells == 0 || !litlen->present ||
        !offset->present || !matchlen->present) {
        return ZstdSeqRefuse(kZstdSeqRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    /* Every cell read below is bounded by `table_size`, and nothing so far has
     * bounded `table_size` by the array it indexes. A table this unit built
     * cannot fail this - the FSE builder refuses a table larger than the
     * capacity it was given - so what it refuses is a table assembled
     * somewhere else. */
    if (litlen->table_size > litlen->capacity ||
        offset->table_size > offset->capacity ||
        matchlen->table_size > matchlen->capacity) {
        return ZstdSeqRefuse(kZstdSeqRejectTableCapacity,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    if (size == 0) {
        return ZstdSeqRefuse(kZstdSeqRejectBitstreamMissing,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    cursor->reader = ZstdBitReader{src, size};
    cursor->litlen = litlen;
    cursor->offset = offset;
    cursor->matchlen = matchlen;
    cursor->total = sequence_count;
    cursor->emitted = 0;
    cursor->finished = false;
    if (cursor->reader.Start() != CUDEC_OK) {
        return ZstdSeqRefuse(kZstdSeqRejectBitstreamMissing,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    ZstdFseReject fse_reject = kZstdFseRejectNone;
    if (ZstdFseInitState(&cursor->reader, litlen->accuracy_log,
                         &cursor->litlen_state, &fse_reject) != CUDEC_OK ||
        ZstdFseInitState(&cursor->reader, offset->accuracy_log,
                         &cursor->offset_state, &fse_reject) != CUDEC_OK ||
        ZstdFseInitState(&cursor->reader, matchlen->accuracy_log,
                         &cursor->matchlen_state, &fse_reject) != CUDEC_OK) {
        return ZstdSeqRefuse(kZstdSeqRejectStateInitTruncated,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    return CUDEC_OK;
}

/* Decodes the next `want` sequences of a started decode, or however many are
 * left if that is fewer.
 *
 * Three orders, all the format's and no two of them the same: the states are
 * initialised literals-length, offset, match-length; the extra bits are read
 * offset, match-length, literals-length; and the states are updated
 * literals-length, match-length, offset. An implementation that gets any one
 * of them wrong still decodes the first field of the first sequence
 * correctly, which is why the corpus sweep asserts exact consumption rather
 * than only the count.
 *
 * THE END CHECK BELONGS TO THE TILE THAT EMITS THE LAST SEQUENCE and not to
 * whoever calls last. The stream must end exactly where the final sequence
 * leaves it, so the check runs the moment `emitted` reaches `total` - a caller
 * that stops there and never calls again has still had the stream held to its
 * count. */
CUDEC_HOST_DEVICE inline cudec_status ZstdSeqDecodeTile(
    ZstdSeqCursor* cursor, ZstdSequence* out, uint32_t out_capacity,
    uint32_t want, uint32_t* out_count, ZstdSeqReject* reject) {
    if (reject != 0) {
        *reject = kZstdSeqRejectNone;
    }
    if (out_count != 0) {
        *out_count = 0;
    }
    if (cursor == 0 || out == 0 || out_count == 0 || want == 0 ||
        out_capacity < want || cursor->litlen == 0 || cursor->offset == 0 ||
        cursor->matchlen == 0 || cursor->finished ||
        cursor->emitted >= cursor->total) {
        return ZstdSeqRefuse(kZstdSeqRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }

    const uint32_t remaining = cursor->total - cursor->emitted;
    const uint32_t count = want < remaining ? want : remaining;
    for (uint32_t index = 0; index < count; index++) {
        unsigned litlen_code = 0;
        unsigned offset_code = 0;
        unsigned matchlen_code = 0;
        cudec_status status = ZstdSeqPeekSymbol(
            cursor->litlen, &cursor->litlen_state, &litlen_code, reject);
        if (status != CUDEC_OK) {
            return status;
        }
        status = ZstdSeqPeekSymbol(cursor->offset, &cursor->offset_state,
                                   &offset_code, reject);
        if (status != CUDEC_OK) {
            return status;
        }
        status = ZstdSeqPeekSymbol(cursor->matchlen, &cursor->matchlen_state,
                                   &matchlen_code, reject);
        if (status != CUDEC_OK) {
            return status;
        }
        /* A code past the alphabet can only arrive from a table built
         * elsewhere - every description this unit accepts is bounded by the
         * field's maximum - and the reads below are indexed by it, so it is
         * refused before it is used. */
        if (litlen_code > ZstdSeqFieldSymbolMax(kZstdSeqFieldLitLen) ||
            offset_code > ZstdSeqFieldSymbolMax(kZstdSeqFieldOffset) ||
            matchlen_code > ZstdSeqFieldSymbolMax(kZstdSeqFieldMatchLen)) {
            return ZstdSeqRefuse(kZstdSeqRejectDecodedSymbolPastMax,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }

        uint64_t offset_extra = 0;
        uint64_t matchlen_extra = 0;
        uint64_t litlen_extra = 0;
        if (cursor->reader.ReadBits(offset_code, &offset_extra) != CUDEC_OK ||
            cursor->reader.ReadBits(ZstdSeqMatchLenExtraBits(matchlen_code),
                                    &matchlen_extra) != CUDEC_OK ||
            cursor->reader.ReadBits(ZstdSeqLitLenExtraBits(litlen_code),
                                    &litlen_extra) != CUDEC_OK) {
            return ZstdSeqRefuse(kZstdSeqRejectSequenceTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }

        out[index].offset_value = (1ull << offset_code) + offset_extra;
        out[index].match_length = ZstdSeqMatchLenBaseline(matchlen_code) +
                                  static_cast<uint32_t>(matchlen_extra);
        out[index].literals_length = ZstdSeqLitLenBaseline(litlen_code) +
                                     static_cast<uint32_t>(litlen_extra);
        cursor->emitted++;

        /* The last sequence consumes its states without updating them, which
         * is what makes the stream end exactly here. */
        if (cursor->emitted == cursor->total) {
            continue;
        }
        status = ZstdSeqUpdateState(&cursor->reader, cursor->litlen,
                                    &cursor->litlen_state, reject);
        if (status != CUDEC_OK) {
            return status;
        }
        status = ZstdSeqUpdateState(&cursor->reader, cursor->matchlen,
                                    &cursor->matchlen_state, reject);
        if (status != CUDEC_OK) {
            return status;
        }
        status = ZstdSeqUpdateState(&cursor->reader, cursor->offset,
                                    &cursor->offset_state, reject);
        if (status != CUDEC_OK) {
            return status;
        }
    }

    if (cursor->emitted == cursor->total) {
        if (!cursor->reader.AtEnd()) {
            return ZstdSeqRefuse(kZstdSeqRejectBitstreamNotConsumed,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        cursor->finished = true;
    }
    *out_count = count;
    return CUDEC_OK;
}

/* The whole block's sequences in one call: the shape the host twins and the
 * frame decoder use, and one tile the size of the block. */
CUDEC_HOST_DEVICE inline cudec_status ZstdDecodeSequences(
    const unsigned char* src, uint64_t size, uint32_t sequence_count,
    const ZstdSeqTable* litlen, const ZstdSeqTable* offset,
    const ZstdSeqTable* matchlen, ZstdSequence* out, uint32_t out_capacity,
    ZstdSeqReject* reject) {
    if (reject != 0) {
        *reject = kZstdSeqRejectNone;
    }
    /* Bounded here as well as in the tile, because the tile is handed `want`
     * rather than the block's count: a capacity short of the whole block is a
     * caller bug this form has always refused, and refusing it before the
     * bitstream is touched keeps that answer independent of the stream. */
    if (out == 0 || sequence_count == 0 || out_capacity < sequence_count) {
        return ZstdSeqRefuse(kZstdSeqRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    ZstdSeqCursor cursor;
    const cudec_status started = ZstdSeqBegin(
        src, size, sequence_count, litlen, offset, matchlen, &cursor, reject);
    if (started != CUDEC_OK) {
        return started;
    }
    uint32_t decoded = 0;
    return ZstdSeqDecodeTile(&cursor, out, out_capacity, sequence_count,
                             &decoded, reject);
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_SEQ_H */
