/* Zstd FSE table descriptions: the decode of a Set_Compressed distribution
 * and the construction of the decoding table it describes. Single-sourced for
 * host and device, the sibling of src/lz4_block.h, src/snappy_block.h and
 * src/zstd_bitstream.h. Internal header, not part of the ABI.
 *
 * The two halves ship together because the build has no reject path that a
 * validated description can reach: it is a pure function of counts that
 * already summed to the table size, and the only honest proof of it is a
 * cell-by-cell diff against the reference. Every refusal that a stream can
 * cause lives in the description decode. The build's own refusals exist for
 * the caller that hands it a vector no description produced, and the twin
 * reaches each of them by calling it directly.
 *
 * THE BIT ORDER IS NOT THE ONE src/zstd_bitstream.h READS, and the issue this
 * lands under assumed it was, so it is stated where a reader will trip over
 * it. A table description is written FORWARD and read FORWARD, low bit of the
 * low byte first; the reader in src/zstd_bitstream.h walks a Zstd bitstream
 * BACKWARD from a start marker in its final byte, high bit first. Those are
 * different serialisations of different objects. RFC 8878 section 4.1.1 gives
 * the forward one for the distribution header, and the reference implements
 * it in lib/common/entropy_common.c, FSE_readNCount_body, which reads a
 * 32-bit little-endian window and shifts up from bit 0. The backward reader
 * is what the FSE state cores and the literal streams run over, one layer up.
 * A description decoded with the backward reader would produce plausible
 * symbols out of the wrong bits, which is the failure this note exists to
 * prevent.
 *
 * WHERE THIS IS STRICTER THAN THE REFERENCE, deliberately. The reference
 * anchors its 32-bit window at four bytes before the end once the walk gets
 * close, so bits beyond the description read as zeros and a late refusal
 * catches the case afterwards. This reader never yields a bit it did not
 * read: a field whose width reaches past the buffer is refused on the spot,
 * before the value is used. Fail-closed is allowed to be stricter than the
 * oracle and the twin asserts the direction rather than assuming it.
 *
 * NO ALLOCATION. Counts, cells and the per-symbol next-state workspace are
 * all caller-provided and their capacities are arguments, so a kernel can
 * point them at shared memory and a hostile accuracy log cannot make this
 * header ask for a byte the caller did not size. */
#ifndef CUDEC_ZSTD_FSE_H
#define CUDEC_ZSTD_FSE_H

#include "cudec.h"
#include "zstd_bitstream.h"

#include <stdint.h>

/* Guarded: src/lz4_block.h, src/snappy_block.h and src/zstd_bitstream.h
 * define the same macro for the same reason, and a device translation unit
 * that decodes more than one format includes more than one header. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIP__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

constexpr unsigned kZstdFseBitsPerByte = 8;

/* FSE_MIN_TABLELOG and FSE_TABLELOG_ABSOLUTE_MAX in the reference's fse.h.
 * The four-bit accuracy-log field is stored biased by the minimum, so the
 * widest log the field can express is the minimum plus fifteen. */
constexpr unsigned kZstdFseMinAccuracyLog = 5;
constexpr unsigned kZstdFseMaxAccuracyLog = 15;

/* FSE_MAX_SYMBOL_VALUE. A count vector is indexed by an unsigned char, so
 * this is the widest alphabet the format can describe at all. */
constexpr unsigned kZstdFseMaxSymbolValue = 255;

/* The per-field bounds of the three sequence tables, RFC 8878 section
 * 3.1.1.3.2.2. They are the caller's to pass in - this header enforces
 * whatever bound it is given and does not know which field it is decoding -
 * but they live here because they are format constants and a caller that
 * spells one out itself is a caller that can spell it wrong.
 *
 * The reference carries the same six as LLFSELog/OffFSELog/MLFSELog and
 * MaxLL/MaxOff/MaxML in lib/decompress/zstd_decompress_block.c. */
constexpr unsigned kZstdLitLenAccuracyLogMax = 9;
constexpr unsigned kZstdOffsetAccuracyLogMax = 8;
constexpr unsigned kZstdMatchLenAccuracyLogMax = 9;
constexpr unsigned kZstdLitLenSymbolMax = 35;
constexpr unsigned kZstdOffsetSymbolMax = 31;
constexpr unsigned kZstdMatchLenSymbolMax = 52;

/* The odd term in the spread stride, FSE_TABLESTEP in the reference's fse.h.
 * It is what makes the stride odd, and an odd stride is coprime with a
 * power-of-two table size, which is why the walk covers every cell. */
constexpr unsigned kZstdFseSpreadStepBias = 3;

/* The reject ladder, enumerated once, in the shape src/snappy_block.h and
 * src/zstd_bitstream.h use: every refusal returns through one choke point
 * naming its rung, so the twin can require a negative per rung instead of
 * counting statuses that repeat. The oracle collapses all of these into
 * corruption_detected, so the rung is this tree's own vocabulary and the twin
 * proves each one is reachable rather than reading it back out of libzstd. */
enum ZstdFseReject {
    kZstdFseRejectNone = 0,
    /* An argument outside what the format can express: an alphabet past 255,
     * an accuracy-log bound outside [5, 15], or a null buffer. A caller bug
     * rather than a stream, refused rather than clamped. */
    kZstdFseRejectBadRequest,
    kZstdFseRejectEmptyDescription,
    kZstdFseRejectAccuracyLogTooLarge,
    /* A field whose width reaches past the buffer the caller supplied. */
    kZstdFseRejectDescriptionTruncated,
    /* The description ended with probability still unallocated: the alphabet
     * ran out, or a repeat run walked past its last symbol.
     *
     * THERE IS NO SEPARATE OVERSHOOT RUNG AND THE ISSUE THIS LANDS UNDER
     * ASKED FOR ONE. Probabilities summing PAST the table size is not a
     * reachable state here, and it is a property of the encoding rather than
     * of this code. Each field is read against a limit derived from what is
     * left: small_limit is 2*threshold-1 minus the budget, the widest value
     * the full-width form can carry is 2*threshold-1, and subtracting the
     * limit from it leaves exactly the budget - so a decoded probability is
     * never larger than the slots remaining, whatever bits the stream
     * carries. The budget therefore reaches one or stays above it, and the
     * only way a description fails to complete is this rung. A guard for the
     * other direction would be one no negative could reach. */
    kZstdFseRejectSymbolPastMax,
    /* Build side: the cell array the caller supplied is smaller than the
     * table the accuracy log asks for. */
    kZstdFseRejectBuildCapacity,
    /* Build side: a count vector whose magnitudes do not sum to the table
     * size, or which carries a probability below the "less than one" code.
     * Unreachable from a description this header decoded; reachable, and
     * reached by the twin, from a caller that builds a vector by hand. */
    kZstdFseRejectBuildCountsNotNormalized,
    /* Core side: the stream held fewer bits than the two initial states need,
     * or a zero-bit cell was reached through a reader that never started. */
    kZstdFseRejectStateInitTruncated,
    /* Core side: a state landed outside the table. Well-formed tables cannot
     * produce one - the successor of every cell is inside - so this refuses a
     * cell array that came from somewhere else. */
    kZstdFseRejectStateOutOfTable,
    /* Core side: the caller's capacity filled before the bits ran out. */
    kZstdFseRejectOutputFull,
    kZstdFseRejectCount
};

CUDEC_HOST_DEVICE inline cudec_status ZstdFseRefuse(ZstdFseReject rung,
                                                    cudec_status status,
                                                    ZstdFseReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* Position of the highest set bit, 0-indexed. Only ever called on a non-zero
 * value - the caller reaches it with a remaining budget of at least two - so
 * there is no "no bits set" answer to define. A counted loop over a
 * compile-time bound rather than an intrinsic, because this header compiles
 * for both the host and the device. */
constexpr unsigned kZstdFseWordBits = 32;

CUDEC_HOST_DEVICE inline unsigned ZstdFseHighestSetBit(uint32_t value) {
    unsigned highest = 0;
    for (unsigned bit = 0; bit < kZstdFseWordBits; bit++) {
        if ((value >> bit) & 1u) {
            highest = bit;
        }
    }
    return highest;
}

/* One decoding-table cell. Field for field the reference's FSE_decode_t in
 * lib/common/fse.h - newState, symbol, nbBits - which is what makes the twin's
 * cell-by-cell diff a comparison of the same three quantities rather than of
 * a layout this tree invented. The layout itself is not load-bearing here;
 * the kernel's shared-memory form is settled with the kernel. */
struct ZstdFseCell {
    uint16_t new_state;
    uint8_t symbol;
    uint8_t nb_bits;
};

/* The forward, low-bit-first reader the table description is written in.
 *
 * Position is carried as a byte index plus a bit offset inside that byte
 * rather than as a bit count, so no bound comparison needs the total bit
 * count of the buffer and there is no multiplication to overflow on a size
 * the caller chose. Peek fills bits at or past the end with zero and the
 * caller must not use a value it has not proved available - which is what
 * Available is for, and every call site checks it before Advance. */
struct ZstdFseDescReader {
    const unsigned char* src;
    uint64_t size;
    uint64_t byte_pos = 0;
    unsigned bit_pos = 0;

    /* True when `count` more bits sit inside the buffer. The widest count any
     * call site passes is sixteen - an accuracy log of fifteen makes the
     * probability field that wide and nothing here reads more - so the
     * arithmetic below stays far inside its type. */
    CUDEC_HOST_DEVICE bool Available(unsigned count) const {
        const unsigned needed_bits = bit_pos + count;
        const uint64_t needed_bytes =
            (needed_bits + kZstdFseBitsPerByte - 1) / kZstdFseBitsPerByte;
        /* byte_pos <= size is an invariant: Advance is only called after
         * Available said yes. */
        return needed_bytes <= size - byte_pos;
    }

    /* `count` bits from the current position, low bit first, as the low bits
     * of the result. Bits at or past the end read as zero, which is only ever
     * observed by a caller that then refuses. */
    CUDEC_HOST_DEVICE uint32_t Peek(unsigned count) const {
        uint32_t value = 0;
        uint64_t byte = byte_pos;
        unsigned bit = bit_pos;
        for (unsigned i = 0; i < count; i++) {
            unsigned taken = 0;
            if (byte < size) {
                taken = (src[byte] >> bit) & 1u;
            }
            value |= static_cast<uint32_t>(taken) << i;
            bit++;
            if (bit == kZstdFseBitsPerByte) {
                bit = 0;
                byte++;
            }
        }
        return value;
    }

    CUDEC_HOST_DEVICE void Advance(unsigned count) {
        const unsigned total = bit_pos + count;
        byte_pos += total / kZstdFseBitsPerByte;
        bit_pos = total % kZstdFseBitsPerByte;
    }

    /* Whole bytes the description occupied. A description that ends mid-byte
     * still owns that byte: the field after it starts on the next one. */
    CUDEC_HOST_DEVICE uint64_t BytesConsumed() const {
        return byte_pos + (bit_pos != 0 ? 1u : 0u);
    }
};

/* Decodes a Set_Compressed table description.
 *
 * `counts` receives max_symbol_value + 1 entries and every symbol the
 * description does not mention is left at zero, so a caller never reads a
 * stale probability for an absent symbol. `out_max_symbol` is the highest
 * symbol the description actually reached, which is at most the bound passed
 * in and is what the table build is then given.
 *
 * `out_consumed` is the byte count the description occupied, and it is the
 * value the sequences section and the Huffman weight path position
 * themselves from. An off-by-one there is silent corruption downstream, which
 * is why it is an output of this function rather than something a caller
 * recomputes.
 *
 * `size` is the bytes remaining in the block, not the description's length -
 * the length is not known until the description has been decoded. */
CUDEC_HOST_DEVICE inline cudec_status ZstdFseReadNCount(
    const unsigned char* src, uint64_t size, unsigned max_symbol_value,
    unsigned max_accuracy_log, int16_t* counts, unsigned* out_max_symbol,
    unsigned* out_accuracy_log, uint64_t* out_consumed,
    ZstdFseReject* reject) {
    *out_max_symbol = 0;
    *out_accuracy_log = 0;
    *out_consumed = 0;
    if (reject != 0) {
        *reject = kZstdFseRejectNone;
    }
    if (src == 0 || counts == 0 || max_symbol_value > kZstdFseMaxSymbolValue ||
        max_accuracy_log < kZstdFseMinAccuracyLog ||
        max_accuracy_log > kZstdFseMaxAccuracyLog) {
        return ZstdFseRefuse(kZstdFseRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    if (size == 0) {
        return ZstdFseRefuse(kZstdFseRejectEmptyDescription,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const unsigned max_symbol_count = max_symbol_value + 1;
    for (unsigned symbol = 0; symbol < max_symbol_count; symbol++) {
        counts[symbol] = 0;
    }

    ZstdFseDescReader reader{src, size};
    constexpr unsigned kAccuracyLogFieldBits = 4;
    if (!reader.Available(kAccuracyLogFieldBits)) {
        return ZstdFseRefuse(kZstdFseRejectDescriptionTruncated,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const unsigned accuracy_log =
        reader.Peek(kAccuracyLogFieldBits) + kZstdFseMinAccuracyLog;
    reader.Advance(kAccuracyLogFieldBits);
    if (accuracy_log > max_accuracy_log) {
        return ZstdFseRefuse(kZstdFseRejectAccuracyLogTooLarge,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    /* The budget, in table slots, plus one: the format's own accounting, and
     * the reason a probability is stored one higher than it is worth. */
    int32_t remaining = static_cast<int32_t>(1u << accuracy_log) + 1;
    int32_t threshold = static_cast<int32_t>(1u << accuracy_log);
    unsigned field_bits = accuracy_log + 1;
    unsigned symbol_count = 0;
    bool previous_zero = false;

    /* Counted, and the bound is the alphabet: every iteration stores exactly
     * one probability, so a description cannot describe more symbols than the
     * field admits however it is spelled. Falling out of the loop rather than
     * breaking leaves `remaining` above one, which the check below refuses -
     * the exit is a reject either way, never a silent truncation. */
    for (unsigned iteration = 0; iteration < max_symbol_count; iteration++) {
        if (previous_zero) {
            /* Runs of absent symbols: a two-bit code of three means three
             * more absent symbols and another code follows. Counted by the
             * alphabet for the same reason as the outer loop - each code
             * that continues the run adds three symbols, so the alphabet
             * bounds the run length. */
            bool run_open = true;
            const unsigned run_limit = max_symbol_count;
            for (unsigned step = 0; step < run_limit && run_open; step++) {
                constexpr unsigned kRepeatFieldBits = 2;
                constexpr unsigned kRepeatContinues = 3;
                if (!reader.Available(kRepeatFieldBits)) {
                    return ZstdFseRefuse(kZstdFseRejectDescriptionTruncated,
                                         CUDEC_ERR_CORRUPT_INPUT, reject);
                }
                const unsigned repeat = reader.Peek(kRepeatFieldBits);
                reader.Advance(kRepeatFieldBits);
                symbol_count += repeat;
                if (repeat != kRepeatContinues) {
                    run_open = false;
                }
                if (symbol_count >= max_symbol_count) {
                    return ZstdFseRefuse(kZstdFseRejectSymbolPastMax,
                                         CUDEC_ERR_CORRUPT_INPUT, reject);
                }
            }
            if (run_open) {
                return ZstdFseRefuse(kZstdFseRejectSymbolPastMax,
                                     CUDEC_ERR_CORRUPT_INPUT, reject);
            }
            previous_zero = false;
        }

        /* The variable-width probability. The low `field_bits - 1` bits are
         * enough for the values below `small_limit`; anything at or above it
         * needs the full width. The peek is taken at the full width and the
         * narrower consumption is what is charged, so a value decided by bits
         * inside the buffer is never refused for bits outside it. */
        const int32_t small_limit = (2 * threshold - 1) - remaining;
        const uint32_t peeked = reader.Peek(field_bits);
        const int32_t narrow =
            static_cast<int32_t>(peeked & static_cast<uint32_t>(threshold - 1));
        int32_t probability = 0;
        unsigned width = 0;
        if (narrow < small_limit) {
            probability = narrow;
            width = field_bits - 1;
        } else {
            probability = static_cast<int32_t>(
                peeked & static_cast<uint32_t>(2 * threshold - 1));
            if (probability >= threshold) {
                probability -= small_limit;
            }
            width = field_bits;
        }
        if (!reader.Available(width)) {
            return ZstdFseRefuse(kZstdFseRejectDescriptionTruncated,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        reader.Advance(width);

        /* Stored one high so that "less than one" has an encoding: the value
         * minus one is the probability, and minus one itself is the symbol
         * that occupies a single slot at the top of the table. Either way one
         * slot per unit of magnitude comes out of the budget. */
        probability--;
        remaining -= probability < 0 ? -probability : probability;
        counts[symbol_count] = static_cast<int16_t>(probability);
        symbol_count++;
        previous_zero = probability == 0;

        if (remaining < threshold) {
            if (remaining <= 1) {
                break;
            }
            field_bits =
                ZstdFseHighestSetBit(static_cast<uint32_t>(remaining)) + 1;
            threshold = static_cast<int32_t>(1u << (field_bits - 1));
        }
        if (symbol_count >= max_symbol_count) {
            break;
        }
    }

    if (remaining != 1) {
        /* The exact-one termination is the format's own end condition, and it
         * is the only accept. See the rung's own note for why the budget
         * cannot be below one here. */
        return ZstdFseRefuse(kZstdFseRejectSymbolPastMax,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    *out_max_symbol = symbol_count - 1;
    *out_accuracy_log = accuracy_log;
    *out_consumed = reader.BytesConsumed();
    return CUDEC_OK;
}

/* Builds the decoding table a validated count vector describes.
 *
 * `cells` receives 1 << accuracy_log entries and `symbol_next` is a
 * max_symbol_value + 1 entry scratch the build uses to hand out successive
 * states per symbol. Both are the caller's, sized by it and checked here.
 *
 * The vector is validated before a cell is written rather than after: the
 * spread walk's bound is the table size only while the magnitudes sum to it,
 * and the "less than one" symbols are placed from the top of the table
 * downwards, which walks off the bottom of the array if there are more of
 * them than the table has cells. Both are refusals rather than assumptions
 * because this entry point is callable with a vector no description
 * produced. */
CUDEC_HOST_DEVICE inline cudec_status ZstdFseBuildDTable(
    const int16_t* counts, unsigned max_symbol_value, unsigned accuracy_log,
    ZstdFseCell* cells, uint32_t cell_capacity, uint16_t* symbol_next,
    ZstdFseReject* reject) {
    if (reject != 0) {
        *reject = kZstdFseRejectNone;
    }
    if (counts == 0 || cells == 0 || symbol_next == 0 ||
        max_symbol_value > kZstdFseMaxSymbolValue ||
        accuracy_log < kZstdFseMinAccuracyLog ||
        accuracy_log > kZstdFseMaxAccuracyLog) {
        return ZstdFseRefuse(kZstdFseRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    const uint32_t table_size = 1u << accuracy_log;
    if (cell_capacity < table_size) {
        return ZstdFseRefuse(kZstdFseRejectBuildCapacity,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    const unsigned max_symbol_count = max_symbol_value + 1;

    /* The magnitudes must sum to exactly the table size. Checked as it goes
     * so the accumulator cannot run away, and checked before anything is
     * written so no bound below depends on an unvalidated number. */
    uint32_t claimed = 0;
    for (unsigned symbol = 0; symbol < max_symbol_count; symbol++) {
        const int16_t probability = counts[symbol];
        if (probability < -1) {
            return ZstdFseRefuse(kZstdFseRejectBuildCountsNotNormalized,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        const uint32_t magnitude = probability < 0
                                       ? 1u
                                       : static_cast<uint32_t>(probability);
        if (magnitude > table_size - claimed) {
            return ZstdFseRefuse(kZstdFseRejectBuildCountsNotNormalized,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        claimed += magnitude;
    }
    if (claimed != table_size) {
        return ZstdFseRefuse(kZstdFseRejectBuildCountsNotNormalized,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    /* The "less than one" symbols occupy the top of the table, one each,
     * highest cell first. high_threshold ends as the last cell the spread
     * walk below may touch. The sum check above bounds their number by the
     * table size, so the index cannot walk below zero. */
    uint32_t high_threshold = table_size - 1;
    for (unsigned symbol = 0; symbol < max_symbol_count; symbol++) {
        if (counts[symbol] < 0) {
            cells[high_threshold].symbol = static_cast<uint8_t>(symbol);
            high_threshold--;
            symbol_next[symbol] = 1;
        } else {
            symbol_next[symbol] = static_cast<uint16_t>(counts[symbol]);
        }
    }

    /* The spread. Each symbol's slots are laid down at a fixed stride that is
     * odd and therefore coprime with the power-of-two table size, so the walk
     * visits every cell exactly once - which is what makes the count vector's
     * sum the whole of the bound. */
    const uint32_t table_mask = table_size - 1;
    const uint32_t step =
        (table_size >> 1) + (table_size >> 3) + kZstdFseSpreadStepBias;
    uint32_t position = 0;
    for (unsigned symbol = 0; symbol < max_symbol_count; symbol++) {
        const int16_t probability = counts[symbol];
        for (int16_t slot = 0; slot < probability; slot++) {
            cells[position].symbol = static_cast<uint8_t>(symbol);
            position = (position + step) & table_mask;
            /* The reserved top of the table is skipped rather than written
             * over. fuel: the stride's full cycle is the table size, so no
             * more than that many probes separate two cells outside the
             * reserved area; a vector that could exhaust it is one the sum
             * check above already refused. */
            uint32_t fuel = table_size;
            while (position > high_threshold) {
                if (fuel == 0) {
                    return ZstdFseRefuse(
                        kZstdFseRejectBuildCountsNotNormalized,
                        CUDEC_ERR_CORRUPT_INPUT, reject);
                }
                fuel--;
                position = (position + step) & table_mask;
            }
        }
    }

    /* The states. A cell's symbol has already been laid down; what is left is
     * how many bits the decoder reads at that cell and which state it lands
     * in, both derived from how many of that symbol's slots have been handed
     * out so far. */
    for (uint32_t cell = 0; cell < table_size; cell++) {
        const uint8_t symbol = cells[cell].symbol;
        const uint16_t next_state = symbol_next[symbol];
        symbol_next[symbol] = static_cast<uint16_t>(next_state + 1);
        const unsigned bits =
            accuracy_log - ZstdFseHighestSetBit(next_state);
        cells[cell].nb_bits = static_cast<uint8_t>(bits);
        cells[cell].new_state =
            static_cast<uint16_t>((next_state << bits) - table_size);
    }
    return CUDEC_OK;
}

/* The decode cores that walk a built table over a backward bitstream.
 *
 * A state is an index into the table. Initialising it reads accuracy_log
 * bits; decoding a symbol reads the cell's own bit count and lands on the
 * cell's successor. The emit precedes the update, which is the whole of the
 * ordering: a decoder that updates first emits the symbol of a cell the
 * encoder never wrote.
 *
 * WHERE THE STREAM ENDS IS DECIDED BY THE BITS, not by a length. An FSE
 * stream carries no symbol count: the encoder wrote exactly the bits the
 * decoder needs for every update except the last, so the run ends when an
 * update would need more bits than are left. The reference reaches the same
 * point from the other side, by noticing after the fact that its read pointer
 * passed the start of the buffer (FSE_decompress_usingDTable_generic in
 * lib/common/fse_decompress.c), and emitting one final symbol from the other
 * state. This unit refuses to read the bits instead of reading them and
 * noticing, which is the same stopping point and no fabricated bit.
 *
 * THE OUTPUT BOUND IS NOT DECORATION. A table whose cells all read zero bits
 * is well formed - one symbol holding the whole table produces exactly that -
 * and over such a table the bit budget never runs down. The caller's capacity
 * is what ends the run there, and reaching it is a refusal rather than a
 * truncated answer. */
struct ZstdFseState {
    uint32_t value;
};

CUDEC_HOST_DEVICE inline cudec_status ZstdFseInitState(ZstdBitReader* reader,
                                                       unsigned accuracy_log,
                                                       ZstdFseState* state,
                                                       ZstdFseReject* reject) {
    uint64_t bits = 0;
    if (reader->ReadBits(accuracy_log, &bits) != CUDEC_OK) {
        return ZstdFseRefuse(kZstdFseRejectStateInitTruncated,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    state->value = static_cast<uint32_t>(bits);
    return CUDEC_OK;
}

/* One symbol. `spent` comes back true when the update would have needed more
 * bits than the stream holds: the symbol just emitted is real and the state
 * is left where it was, and the caller stops after taking one more symbol
 * from its other state.
 *
 * `table_size` is checked against the state on every call rather than trusted
 * from the build. A cell array is the one thing here a caller can hand in
 * from anywhere, and an index into it that came out of a stream is exactly
 * the read this project does not allow to go unbounded. */
CUDEC_HOST_DEVICE inline cudec_status ZstdFseNextSymbol(
    ZstdBitReader* reader, const ZstdFseCell* cells, uint32_t table_size,
    ZstdFseState* state, uint8_t* symbol, bool* spent,
    ZstdFseReject* reject) {
    *spent = false;
    if (state->value >= table_size) {
        return ZstdFseRefuse(kZstdFseRejectStateOutOfTable,
                             CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const ZstdFseCell cell = cells[state->value];
    *symbol = cell.symbol;
    if (cell.nb_bits > reader->BitsRemaining()) {
        *spent = true;
        return CUDEC_OK;
    }
    uint64_t bits = 0;
    const cudec_status status = reader->ReadBits(cell.nb_bits, &bits);
    if (status != CUDEC_OK) {
        /* Reachable: a reader whose Start never succeeded reports no bits
         * remaining and refuses a zero-width read, so the check above lets a
         * zero-bit cell through to here. */
        return ZstdFseRefuse(kZstdFseRejectStateInitTruncated, status, reject);
    }
    state->value = cell.new_state + static_cast<uint32_t>(bits);
    return CUDEC_OK;
}

/* The two-state interleaved run, which is the shape the Huffman weight
 * description is written in: two states initialised in order from one stream,
 * symbols taken from them alternately, and the pair of final symbols taken
 * once the bits run out.
 *
 * The reader must already have been started. Symbols land in `out` and the
 * count comes back in `out_count`. */
CUDEC_HOST_DEVICE inline cudec_status ZstdFseDecode2State(
    ZstdBitReader* reader, const ZstdFseCell* cells, uint32_t table_size,
    unsigned accuracy_log, uint8_t* out, uint32_t out_capacity,
    uint32_t* out_count, ZstdFseReject* reject) {
    *out_count = 0;
    if (reject != 0) {
        *reject = kZstdFseRejectNone;
    }
    if (cells == 0 || out == 0 || accuracy_log < kZstdFseMinAccuracyLog ||
        accuracy_log > kZstdFseMaxAccuracyLog ||
        table_size != (1u << accuracy_log)) {
        return ZstdFseRefuse(kZstdFseRejectBadRequest,
                             CUDEC_ERR_INVALID_ARGUMENT, reject);
    }
    ZstdFseState first;
    ZstdFseState second;
    cudec_status status =
        ZstdFseInitState(reader, accuracy_log, &first, reject);
    if (status != CUDEC_OK) {
        return status;
    }
    status = ZstdFseInitState(reader, accuracy_log, &second, reject);
    if (status != CUDEC_OK) {
        return status;
    }

    uint32_t produced = 0;
    /* Counted, and the bound is the caller's capacity: every iteration writes
     * one symbol, so a table over which the bit budget never runs down ends
     * here rather than spinning. */
    for (uint32_t step = 0; step < out_capacity; step++) {
        ZstdFseState* active = (step % 2u) == 0 ? &first : &second;
        ZstdFseState* other = (step % 2u) == 0 ? &second : &first;
        uint8_t symbol = 0;
        bool spent = false;
        status = ZstdFseNextSymbol(reader, cells, table_size, active, &symbol,
                                   &spent, reject);
        if (status != CUDEC_OK) {
            return status;
        }
        out[produced] = symbol;
        produced++;
        if (!spent) {
            continue;
        }
        /* The last update could not be paid for, so the run ends with one
         * more symbol from the other state - the reference's own tail. */
        if (produced == out_capacity) {
            return ZstdFseRefuse(kZstdFseRejectOutputFull,
                                 CUDEC_ERR_OUTPUT_TOO_SMALL, reject);
        }
        if (other->value >= table_size) {
            return ZstdFseRefuse(kZstdFseRejectStateOutOfTable,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        out[produced] = cells[other->value].symbol;
        produced++;
        *out_count = produced;
        return CUDEC_OK;
    }
    return ZstdFseRefuse(kZstdFseRejectOutputFull, CUDEC_ERR_OUTPUT_TOO_SMALL,
                         reject);
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_FSE_H */
