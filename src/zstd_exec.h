/* Zstd sequence execution: the prefix sum that names where every sequence
 * writes, and the literal and match copies that put the bytes there. RFC 8878
 * section 3.1.1.3.2. Single-sourced for host and device, the sibling of
 * src/zstd_seq.h, which produces the tuples, and of src/zstd_repcode.h, which
 * turns an Offset_Value into a distance and says in its own header that
 * bounding that distance belongs here.
 *
 * WHY THE PREFIX SUM IS ITS OWN FUNCTION AND NOT A CURSOR. A serial decoder
 * would carry a running output position and never need one, and that is
 * exactly why a serial decoder proves nothing about the copies a warp makes.
 * Every sequence's destination is a pure function of the lengths before it, so
 * the sum is computed once, up front, and every copy afterwards is independent
 * of every other. On the device that sum is a scan and the copies fan out; on
 * the host it is the loop below and the copies run in order. The two produce
 * the same array, and the array is what the copies read - so an off-by-one in
 * the scan cannot hide behind a cursor that happened to be right anyway.
 *
 * THE EXECUTION DOES NOT TRUST THE ARRAY IT IS HANDED. A caller that computes
 * the destinations somewhere else - a warp scan, a second kernel - hands in an
 * array this unit has no way to have produced. Trusting it would put every
 * bound this unit checks behind an unchecked number, which is the fail-open
 * shape the ladder exists to prevent, so each copy re-derives its own
 * successor from its lengths and refuses a destination that disagrees. The
 * check is one comparison per sequence and it is per-lane parallel, so it
 * costs the device shape nothing it would not have paid anyway.
 *
 * TWO BOUNDS ON AN OFFSET, AND THEY ARE DIFFERENT STATEMENTS. A match may not
 * reach further back than the window, which is what the format promises a
 * decoder it will never have to hold; and it may not reach before the first
 * byte the frame has produced, which is what makes the read defined at all. A
 * decoder checking only the window reads outside its own output on a frame
 * whose window is larger than what it has decoded so far, which is every frame
 * near its start. Both are checked, separately, and each carries its own rung.
 *
 * THE OFFSET IS BOUNDED AGAINST THE FRAME AND NEVER AGAINST THE BLOCK. A match
 * legitimately reaches back across a block boundary into what an earlier block
 * produced - that is what makes the blocks of a frame sequential - so the
 * bytes-produced-so-far this unit compares against are the frame's, and the
 * destination buffer it copies inside is the frame's whole output. A unit
 * bounding against the block would decode the first block of every frame
 * correctly and refuse legal matches in every block after it.
 *
 * AN OVERLAPPING MATCH REPLICATES A PATTERN AND DOES NOT COPY A RANGE. Where
 * the resolved offset is smaller than the match length the source runs into
 * the bytes the match is itself writing, so the copy is byte by byte in
 * increasing order - the same semantics src/lz4_block.h and src/snappy_block.h
 * carry, and the run-fill an offset of one produces is the extreme case. The
 * device form of the same statement is the closed-form modular gather
 * src/lz4_decode.cuh already uses, dst[to + i] = dst[to - offset + (i mod
 * offset)], which is defined for every i because an offset of zero is refused
 * below. A vector copy of the range would produce different bytes, not a
 * faster version of the same ones.
 *
 * WHAT THIS UNIT DOES NOT DO. It does not decode a block, loop over the blocks
 * of a frame, or resolve an Offset_Value: the tuples, the literals and the
 * resolved distances all arrive already produced, each from the unit that owns
 * it. Where a block's regenerated size comes from is the caller's too - a
 * compressed block does not declare one, so what this unit is given is the
 * ceiling that block may not pass, and what it reports back is the size the
 * sequences and the leftover literals actually add up to.
 *
 * Internal header, not part of the ABI. */
#ifndef CUDEC_ZSTD_EXEC_H
#define CUDEC_ZSTD_EXEC_H

#include "cudec.h"
#include "zstd_frame.h"
#include "zstd_seq.h"

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

/* The reject ladder, enumerated once, in the shape src/zstd_seq.h and
 * src/zstd_repcode.h use: every refusal returns through one choke point naming
 * its rung, so the twin requires a negative per rung instead of counting
 * statuses that repeat. */
enum ZstdExecReject {
    kZstdExecRejectNone = 0,
    /* Storage or a count the caller did not supply, or a destinations array
     * with no room for the one-past-the-end entry. Not a stream: every one of
     * these is a caller bug refused rather than decoded around. */
    kZstdExecRejectBadRequest,
    /* The sequences ask for more literal bytes than the literals section
     * produced. The leftover tail is what is left after the last sequence, so
     * this is the other side of the same accounting: over-consumption here and
     * a shorter tail there. */
    kZstdExecRejectLiteralsExhausted,
    /* The block regenerates more than Block_Maximum_Size allows. A compressed
     * block declares no regenerated size, so this ceiling is the only thing
     * that bounds the sum before the copies run. */
    kZstdExecRejectBlockTooLarge,
    /* A destination that disagrees with the lengths of the sequence that
     * writes there. Reached by a scan that is wrong, never by a stream. */
    kZstdExecRejectPlanInconsistent,
    /* A resolved offset of zero. src/zstd_repcode.h refuses one already, so
     * this catches an offset array assembled some other way; zero is refused
     * rather than treated as a copy of nothing, because it names a source that
     * is the destination and no byte of it is defined. */
    kZstdExecRejectOffsetZero,
    /* A match reaching further back than the frame's window. */
    kZstdExecRejectOffsetPastWindow,
    /* A match reaching before the first byte the frame has produced. */
    kZstdExecRejectOffsetBeforeOutput,
    /* The frame's output plus this block does not fit the destination. Its
     * status is OUTPUT_TOO_SMALL and not CORRUPT_INPUT: the bytes may be
     * perfectly good and the buffer merely short, and a caller that cannot
     * tell those apart cannot retry. */
    kZstdExecRejectDestinationTooSmall,
    kZstdExecRejectCount
};

CUDEC_HOST_DEVICE inline cudec_status ZstdExecRefuse(ZstdExecReject rung,
                                                     cudec_status status,
                                                     ZstdExecReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* What the prefix sum found out about one block, which is what the execution
 * needs and the caller wants back. */
struct ZstdExecPlan {
    /* Bytes this block regenerates: everything the sequences write plus the
     * literals left after the last one. */
    uint64_t block_size;
    /* Bytes of the literals buffer the sequences consume. The tail is
     * `literals_size - literals_used` and starts exactly there. */
    uint64_t literals_used;
};

/* The prefix sum over (Literal_Length + Match_Length), block-relative.
 *
 * `out_destinations` receives `sequence_count + 1` entries: entry `i` is where
 * sequence `i` writes its literals, and the last entry is where the leftover
 * literals begin. The one-past-the-end entry is not a convenience - it is what
 * the execution compares each sequence's own arithmetic against, and it is
 * what tells the tail where it starts without a second sum.
 *
 * `block_maximum` is Block_Maximum_Size for this frame (RFC 8878 section
 * 3.1.1.2.4), the ceiling a compressed block's regenerated size may not pass.
 * It is bounded here, before any copy, rather than discovered by a destination
 * that overflows: the copies are independent of each other and one of them
 * running past the end is not something the others would notice. */
CUDEC_HOST_DEVICE inline cudec_status ZstdExecPrefixSum(
    const ZstdSequence* sequences, uint32_t sequence_count,
    uint64_t literals_size, uint64_t block_maximum,
    uint64_t* out_destinations, uint32_t destinations_capacity,
    ZstdExecPlan* out_plan, ZstdExecReject* reject) {
    if (reject != 0) {
        *reject = kZstdExecRejectNone;
    }
    if (out_plan != 0) {
        out_plan->block_size = 0;
        out_plan->literals_used = 0;
    }
    /* THE CEILING IS BOUNDED BEFORE IT IS USED, AND THAT IS WHAT MAKES EVERY
     * SUM BELOW SAFE RATHER THAN LUCKY. Both of these are the caller's own
     * quantities and both are already bounded where they are produced - the
     * ceiling by section 3.1.1.2.4, the literals by the section that decoded
     * them - so refusing them here costs a legal frame nothing. What it buys
     * is that the running sum is compared against a ceiling of at most 128 KiB
     * at every step and can never leave that range, so no addition in this
     * file can wrap and no reject rung is needed for one. A rung for an
     * overflow that 32-bit length fields cannot reach would be a guard no
     * negative could ever red, which is worse than no guard: it reads as
     * coverage. */
    if (out_destinations == 0 || out_plan == 0 ||
        (sequences == 0 && sequence_count != 0) ||
        destinations_capacity < 1 ||
        destinations_capacity - 1 < sequence_count ||
        block_maximum > kZstdBlockSizeCeiling ||
        literals_size > kZstdBlockSizeCeiling) {
        return ZstdExecRefuse(kZstdExecRejectBadRequest,
                              CUDEC_ERR_INVALID_ARGUMENT, reject);
    }

    uint64_t at = 0;
    uint64_t literals_used = 0;
    for (uint32_t index = 0; index < sequence_count; index++) {
        out_destinations[index] = at;
        const uint64_t literals_length = sequences[index].literals_length;
        const uint64_t match_length = sequences[index].match_length;
        /* Checked as a subtraction against what is left of the ceiling, which
         * never wraps, and refused at the sequence that passes it rather than
         * after the whole sum: a block that has already exceeded its maximum
         * does not become legal by what follows. */
        const uint64_t span = literals_length + match_length;
        if (span > block_maximum - at) {
            return ZstdExecRefuse(kZstdExecRejectBlockTooLarge,
                                  CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        at += span;
        if (literals_length > literals_size - literals_used) {
            return ZstdExecRefuse(kZstdExecRejectLiteralsExhausted,
                                  CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        literals_used += literals_length;
    }
    out_destinations[sequence_count] = at;

    const uint64_t tail = literals_size - literals_used;
    if (tail > block_maximum - at) {
        return ZstdExecRefuse(kZstdExecRejectBlockTooLarge,
                              CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const uint64_t block_size = at + tail;
    out_plan->block_size = block_size;
    out_plan->literals_used = literals_used;
    return CUDEC_OK;
}

/* Executes one block's sequences and its leftover literals into the frame's
 * output.
 *
 * `offsets` holds one resolved distance per sequence, in the frame's own
 * terms, as src/zstd_repcode.h produced them. Resolving is a serial chain over
 * the frame's repeat-offset history and is that unit's business; what arrives
 * here is the finished array, which is what lets every copy below be
 * independent of every other.
 *
 * `dst` is the frame's whole output and `*produced` the bytes of it earlier
 * blocks left, in and out: on success it advances by the block's size, and on
 * any refusal it is left exactly where it was. Bytes may have been written
 * past it before a later sequence refused - a partial write is not a partial
 * success, and a caller that presented it as one would be reporting output it
 * has no reason to believe. */
CUDEC_HOST_DEVICE inline cudec_status ZstdExecuteBlock(
    const ZstdSequence* sequences, uint32_t sequence_count,
    const uint64_t* destinations, const uint64_t* offsets,
    const unsigned char* literals, uint64_t literals_size,
    const ZstdExecPlan* plan, uint64_t window_size, unsigned char* dst,
    uint64_t dst_capacity, uint64_t* produced, ZstdExecReject* reject) {
    if (reject != 0) {
        *reject = kZstdExecRejectNone;
    }
    if (destinations == 0 || plan == 0 || produced == 0 || dst == 0 ||
        (sequences == 0 && sequence_count != 0) ||
        (offsets == 0 && sequence_count != 0) ||
        (literals == 0 && literals_size != 0)) {
        return ZstdExecRefuse(kZstdExecRejectBadRequest,
                              CUDEC_ERR_INVALID_ARGUMENT, reject);
    }

    const uint64_t base = *produced;
    if (base > dst_capacity || plan->block_size > dst_capacity - base) {
        return ZstdExecRefuse(kZstdExecRejectDestinationTooSmall,
                              CUDEC_ERR_OUTPUT_TOO_SMALL, reject);
    }

    uint64_t literal_at = 0;
    for (uint32_t index = 0; index < sequence_count; index++) {
        const uint64_t literals_length = sequences[index].literals_length;
        const uint64_t match_length = sequences[index].match_length;
        const uint64_t at = destinations[index];
        /* The array is re-derived rather than believed: this sequence's own
         * lengths have to carry the destination to the next one's, and the
         * last one's to where the tail begins. A scan that dropped a sequence,
         * doubled one, or ran in the wrong order fails here rather than
         * writing over a neighbour's bytes. */
        if (at > plan->block_size ||
            literals_length + match_length > plan->block_size - at ||
            at + literals_length + match_length != destinations[index + 1]) {
            return ZstdExecRefuse(kZstdExecRejectPlanInconsistent,
                                  CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        if (literals_length > literals_size - literal_at) {
            return ZstdExecRefuse(kZstdExecRejectLiteralsExhausted,
                                  CUDEC_ERR_CORRUPT_INPUT, reject);
        }

        uint64_t to = base + at;
        for (uint64_t i = 0; i < literals_length; i++) {
            dst[to + i] = literals[literal_at + i];
        }
        literal_at += literals_length;
        to += literals_length;

        const uint64_t offset = offsets[index];
        if (offset == 0) {
            return ZstdExecRefuse(kZstdExecRejectOffsetZero,
                                  CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        if (offset > window_size) {
            return ZstdExecRefuse(kZstdExecRejectOffsetPastWindow,
                                  CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        /* `to` is where this match starts, so the bytes produced before it are
         * everything the frame holds up to that point - the earlier blocks,
         * this block's earlier sequences, and this sequence's own literals.
         * Comparing against the block's own start instead would refuse the
         * cross-block match the format allows. */
        if (offset > to) {
            return ZstdExecRefuse(kZstdExecRejectOffsetBeforeOutput,
                                  CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        const uint64_t from = to - offset;
        for (uint64_t i = 0; i < match_length; i++) {
            dst[to + i] = dst[from + i];
        }
    }

    /* Where the tail begins, and the last thing the plan is held to.
     *
     * The per-sequence check above compares each destination against the SUM
     * of a sequence's two lengths, so it cannot tell a literal byte from a
     * match byte, and with no sequences at all it does not run. Three things
     * therefore reach here unchecked: the tail's start, the literals it is
     * made of, and whether the block size the destination bound was taken from
     * accounts for it. All three are one equation, and it is stated once here
     * rather than trusted three times. Without it a block size of zero beside
     * a literals section of any length writes past the bound this function
     * already passed. */
    const uint64_t tail_at = destinations[sequence_count];
    if (literal_at != plan->literals_used || tail_at > plan->block_size ||
        literals_size - literal_at != plan->block_size - tail_at) {
        return ZstdExecRefuse(kZstdExecRejectPlanInconsistent,
                              CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    /* The literals after the last sequence, copied verbatim. They are part of
     * what the block regenerates rather than a remainder to discard, which is
     * what makes a literals-only block - no sequences at all - decode to its
     * literals. */
    for (uint64_t i = 0; i + literal_at < literals_size; i++) {
        dst[base + tail_at + i] = literals[literal_at + i];
    }

    *produced = base + plan->block_size;
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_EXEC_H */
