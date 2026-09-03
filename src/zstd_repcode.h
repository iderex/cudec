/* Zstd repeat-offset resolution: the three-slot history a frame carries, and
 * the rules that turn a sequence's Offset_Value into a distance. RFC 8878
 * section 3.1.1.5. Single-sourced for host and device, the sibling of
 * src/zstd_seq.h, which hands the raw Offset_Value on and refuses to guess.
 * Internal header, not part of the ABI.
 *
 * THE HISTORY IS THE FRAME'S AND NOT THE BLOCK'S. It is initialised to
 * {1, 4, 8} once, at frame start, and every block after the first inherits
 * what the one before it left. A decoder that reset it per block would decode
 * the first block of every frame correctly and then produce wrong bytes for
 * the rest, which is the failure the twin's cross-block rows exist to catch.
 *
 * ONE VALUE MEANS TWO DIFFERENT SLOTS AND THE LITERALS LENGTH IS WHAT DECIDES.
 * With a non-zero literals length, Offset_Value 1, 2 and 3 name the most
 * recent, second and third offsets. With a literals length of ZERO the meaning
 * shifts by one - 1 names the second, 2 the third, and 3 means "the most
 * recent, minus one". The shift exists because a repeat of the most recent
 * offset with no literals between the two matches would be one longer match
 * rather than two sequences, so that encoding is free for something else. An
 * implementation that misses the shift decodes every sequence with literals
 * correctly and silently corrupts the ones without.
 *
 * THE MINUS-ONE CASE IS THE ONLY WAY A REPEAT CAN REACH ZERO, AND ZERO IS NOT
 * AN OFFSET. Every other resolution is a slot value or an explicit offset, and
 * both are at least one: an explicit Offset_Value is above three and the three
 * is subtracted, and a slot only ever receives a value that already passed
 * this check. So the refusal is written once, over the resolved value, and it
 * catches both the corrupt stream and a caller that handed in a history it
 * never initialised. The reference reaches the same verdict differently - it
 * forces the zero to an offset no output can satisfy and lets the copy refuse
 * - so the two agree on the frame and differ on where they say so, which is
 * what the twin asserts rather than a shared error code.
 *
 * WHAT THIS UNIT DOES NOT DO. It does not bound the resolved offset against
 * the window or against the bytes produced so far: that bound belongs with
 * the copy that uses it, and stating it here would give a caller two places to
 * believe it was checked. */
#ifndef CUDEC_ZSTD_REPCODE_H
#define CUDEC_ZSTD_REPCODE_H

#include "cudec.h"

#include <stdint.h>

/* Guarded: the sibling decode headers define the same macro for the same
 * reason, and a device translation unit that decodes more than one format
 * includes more than one of them. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIP__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* Section 3.1.1.5: three slots, and the Offset_Values 1, 2 and 3 that name
 * them. The same three is what an explicit Offset_Value carries above its
 * distance, because those three encodings are spent on the repeats. */
constexpr unsigned kZstdRepcodeSlots = 3;
constexpr uint64_t kZstdRepcodeMaxSlotValue = 3;

/* The values the history starts a frame with, section 3.1.1.5. */
constexpr uint64_t kZstdRepcodeInitial0 = 1;
constexpr uint64_t kZstdRepcodeInitial1 = 4;
constexpr uint64_t kZstdRepcodeInitial2 = 8;

/* The reject ladder, enumerated once, in the shape src/zstd_seq.h and
 * src/zstd_huf.h use: every refusal returns through one choke point naming
 * its rung, so the twin requires a negative per rung instead of counting
 * statuses that repeat. */
enum ZstdRepcodeReject {
    kZstdRepcodeRejectNone = 0,
    /* An Offset_Value of zero, or storage the caller did not supply. Neither
     * is a stream: src/zstd_seq.h produces a value of at least one for every
     * Offset_Code the format has, so a zero arriving here is a caller bug
     * refused rather than resolved. */
    kZstdRepcodeRejectBadRequest,
    /* A repeat that resolved to zero. Reached by the minus-one rule at a
     * point where the most recent offset is one, and by a history that was
     * never initialised. */
    kZstdRepcodeRejectResolvedToZero,
    kZstdRepcodeRejectCount
};

/* The seven ways an Offset_Value can resolve, which is the same case split
 * ZstdRepcodeResolve makes below and is named here so a caller can ask which
 * of them a stream reached. That question is not decoration: a corpus that
 * decodes byte-identically proves nothing about a rule it never took, and the
 * seventh rule - value 3 with no literals, the most recent offset minus one -
 * is both the rarest and the one whose arithmetic can wrap.
 *
 * Derived from the value and the literals length alone, exactly as the
 * resolution is, so the two cannot disagree about which rule ran. */
enum ZstdRepcodePath {
    /* Offset_Value above three: an explicit distance. */
    kZstdRepcodePathExplicit = 0,
    /* Literals present, value 1, 2, 3: the most recent offset, the second,
     * the third. */
    kZstdRepcodePathSlot0,
    kZstdRepcodePathSlot1,
    kZstdRepcodePathSlot2,
    /* No literals, value 1 and 2: the second offset and the third. */
    kZstdRepcodePathShiftedSlot1,
    kZstdRepcodePathShiftedSlot2,
    /* No literals, value 3: the most recent offset, minus one. */
    kZstdRepcodePathMinusOne,
    kZstdRepcodePathCount
};

CUDEC_HOST_DEVICE inline ZstdRepcodePath ZstdRepcodeClassify(
    uint64_t offset_value, uint32_t literals_length) {
    if (offset_value > kZstdRepcodeMaxSlotValue || offset_value == 0) {
        return kZstdRepcodePathExplicit;
    }
    if (literals_length != 0) {
        return static_cast<ZstdRepcodePath>(
            kZstdRepcodePathSlot0 + static_cast<unsigned>(offset_value) - 1u);
    }
    return static_cast<ZstdRepcodePath>(kZstdRepcodePathShiftedSlot1 +
                                        static_cast<unsigned>(offset_value) -
                                        1u);
}

CUDEC_HOST_DEVICE inline cudec_status ZstdRepcodeRefuse(
    ZstdRepcodeReject rung, cudec_status status, ZstdRepcodeReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* The three most recent offsets, most recent first. The caller owns it and
 * carries it across the blocks of one frame; a kernel decides where it
 * lives, which is why this header neither allocates it nor hides it. */
struct ZstdRepcodeHistory {
    uint64_t slot[kZstdRepcodeSlots];
};

/* Frame start, and the only place the history is ever reset. */
CUDEC_HOST_DEVICE inline void ZstdRepcodeInit(ZstdRepcodeHistory* history) {
    history->slot[0] = kZstdRepcodeInitial0;
    history->slot[1] = kZstdRepcodeInitial1;
    history->slot[2] = kZstdRepcodeInitial2;
}

/* Resolves one sequence's Offset_Value into a distance and moves the history
 * on. Both, in one call: the recency rotation is part of the resolution
 * rather than a step a caller could forget, and a caller that forgot it would
 * decode the first repeat of a frame correctly and every later one wrongly.
 *
 * `offset_value` is src/zstd_seq.h's own quantity, (1 << Offset_Code) plus
 * the code's extra bits, so 1, 2 and 3 name slots and anything above three is
 * an explicit distance three larger than itself.
 *
 * `literals_length` is this sequence's, and only whether it is zero
 * matters. */
CUDEC_HOST_DEVICE inline cudec_status ZstdRepcodeResolve(
    ZstdRepcodeHistory* history, uint64_t offset_value,
    uint32_t literals_length, uint64_t* out_offset,
    ZstdRepcodeReject* reject) {
    if (reject != 0) {
        *reject = kZstdRepcodeRejectNone;
    }
    if (out_offset != 0) {
        *out_offset = 0;
    }
    if (history == 0 || out_offset == 0 || offset_value == 0) {
        return ZstdRepcodeRefuse(kZstdRepcodeRejectBadRequest,
                                 CUDEC_ERR_INVALID_ARGUMENT, reject);
    }

    uint64_t resolved = 0;
    /* Which slot the value took, or kZstdRepcodeSlots for a value that took
     * none - an explicit offset or the minus-one rule. That distinction is
     * what the rotation below reads, and it is not the same as the value. */
    unsigned taken = kZstdRepcodeSlots;

    if (offset_value > kZstdRepcodeMaxSlotValue) {
        resolved = offset_value - kZstdRepcodeMaxSlotValue;
    } else if (literals_length != 0) {
        taken = static_cast<unsigned>(offset_value) - 1u;
        resolved = history->slot[taken];
    } else if (offset_value < kZstdRepcodeMaxSlotValue) {
        /* The shift: 1 names the second slot and 2 the third. */
        taken = static_cast<unsigned>(offset_value);
        resolved = history->slot[taken];
    } else {
        /* Offset_Value 3 with no literals: the most recent offset, minus one.
         * READ FIRST AND SUBTRACT AFTER, so that a slot of zero leaves the
         * resolved value at zero and is refused by the one check below. The
         * other order wraps a zero slot into the widest offset there is, and
         * an unsigned subtraction that wraps is a fail-open however loudly
         * whatever runs next refuses it. A slot is zero only where a caller
         * handed in a history it never initialised, which is exactly the case
         * that has nothing else guarding it. */
        resolved = history->slot[0];
        if (resolved != 0) {
            resolved -= 1;
        }
    }

    if (resolved == 0) {
        return ZstdRepcodeRefuse(kZstdRepcodeRejectResolvedToZero,
                                 CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    /* The recency rotation, and the same three lines serve every case that
     * moves anything. Taking the most recent offset moves nothing, which is
     * why slot 0 is the one case with no update at all; taking the second
     * swaps the top two and leaves the third alone; everything else pushes
     * the resolved value onto the front and drops the oldest. */
    if (taken != 0) {
        if (taken != 1) {
            history->slot[2] = history->slot[1];
        }
        history->slot[1] = history->slot[0];
        history->slot[0] = resolved;
    }
    *out_offset = resolved;
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_REPCODE_H */
