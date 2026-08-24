/* The Snappy block sequence parser and its validation ladder - the
 * fail-closed core of the Snappy decoder, single-sourced for host and
 * device, the sibling of src/lz4_block.h. The parser owns every check;
 * callers own the copies: the CPU twin test executes them sequentially, the
 * M3 kernel fans them out across a warp. Internal header, not part of the
 * ABI.
 *
 * Ladder contract: every value decoded from the stream is validated before
 * first use as an address, length, or offset. The declared uncompressed
 * length is attacker-controlled and the destination capacity is the
 * caller's truth, so the declared length is checked against the capacity
 * before the first element is parsed and nothing is ever sized from the
 * stream. Success is reported ONLY at exact source consumption AND exact
 * production of the declared length, which is what snappy's own
 * `decompressor->eof() && writer->CheckLength()` enforces on both ends.
 *
 * Two differences from LZ4 shape the code. An element is a literal OR a
 * copy rather than a literal run followed by a match, so exactly one half
 * of the shared DecodeSequence is filled and the other is left empty - the
 * copy engine runs both halves either way and needs no kind flag to branch
 * on (src/decode_sequence.h). And a length never continues across an
 * unbounded chain of extension bytes: an element's header is the tag plus
 * at most four trailing bytes, which is snappy's kMaximumTagLength, so the
 * per-element termination argument needs no fuel counter.
 *
 * Widths, for the kernel that consumes this: the declared length is a
 * varint32, so every output position, every copy offset and every element
 * length provably fits in 32 bits. The shared element's fields are 64-bit
 * anyway, because the caller's capacity is a size_t and mixing the two
 * widths in the bound comparisons is where an overflow would hide. Whether
 * the DEVICE side should carry 32-bit copies of them is a measured
 * kernel-shape question (issue #292) and is not settled here.
 *
 * Edge semantics are settled empirically by oracle parity - whenever
 * snappy rejects, this parser must reject (tests/snappy_parser_twin.cpp).
 * There is one deliberate exception, at the literal length arithmetic
 * below, and it is documented and pinned there. */
#ifndef CUDEC_SNAPPY_BLOCK_H
#define CUDEC_SNAPPY_BLOCK_H

#include "cudec.h"
#include "decode_sequence.h"

#include <stdint.h>

namespace cudec_detail {

/* The preamble is a varint32, read the way snappy's own preamble readers
 * read it - Varint::Parse32WithLimit is the one GetUncompressedLength goes
 * through: five unrolled groups of seven bits, stopping at the first byte
 * whose continuation bit is clear. The fifth group contributes only four
 * bits before a 32-bit accumulator would overflow, which is why that
 * reference refuses a fifth byte of 16 or more - and why a non-minimal
 * encoding, which nothing there rejects, must stay accepted here. */
constexpr uint64_t kSnappyVarintGroups = 5;
constexpr unsigned kSnappyVarintBits = 7;
constexpr unsigned char kSnappyVarintContinue = 0x80;
constexpr unsigned char kSnappyVarintPayload = 0x7f;
constexpr unsigned char kSnappyVarintFinalMax = 16;

/* The tag byte: the low two bits select the element kind, the upper six
 * carry a length (literals and the two wide copy forms) or a length and
 * the offset's high bits (the narrow copy form). */
constexpr unsigned char kSnappyTagKindMask = 0x3;
constexpr unsigned kSnappyTagKindLiteral = 0;
constexpr unsigned kSnappyTagKindCopy1 = 1;
constexpr unsigned kSnappyTagKindCopy2 = 2;
constexpr unsigned kSnappyTagKindCopy4 = 3;
constexpr unsigned kSnappyTagLengthShift = 2;
constexpr unsigned kSnappyBitsPerByte = 8;

/* Every element header begins with the tag byte itself. */
constexpr uint64_t kSnappyTagBytes = 1;

/* Literals: the six tag bits hold length-1 up to and including this value.
 * Above it they select how many little-endian bytes carry length-1
 * instead, which is at most four - so an element header never exceeds the
 * five bytes snappy calls kMaximumTagLength. */
constexpr uint64_t kSnappyLiteralInlineMax = 59;

/* The narrow copy form: three bits of length biased by four, and three tag
 * bits supplying the offset's high byte. It cannot express lengths 1 to 3
 * at all; the wide forms can, and snappy accepts them (tests/snappy_probes.cpp
 * probe 5), so no length floor may be imposed here. */
constexpr unsigned char kSnappyCopy1LengthMask = 0x7;
constexpr unsigned kSnappyCopy1OffsetShift = 5;
constexpr uint64_t kSnappyCopy1MinLength = 4;

/* Offset widths, one per copy form. */
constexpr uint64_t kSnappyCopy1OffsetBytes = 1;
constexpr uint64_t kSnappyCopy2OffsetBytes = 2;
constexpr uint64_t kSnappyCopy4OffsetBytes = 4;

/* The reject ladder, enumerated once (issue #173).
 *
 * Every refusal in this header names its branch here and returns through
 * SnappyRefuse, so the branch set is DERIVED from the code rather than
 * restated beside it. Two gates read that: tests/CMakeLists.txt refuses a
 * refusal that does not go through the choke point, an enumerator with no
 * site, and a site with no enumerator; tests/snappy_parser_twin.cpp refuses
 * a branch no negative reaches. Adding a branch without a negative therefore
 * reds the suite, and deleting a negative reds it too.
 *
 * kSnappyRejectPreambleUnreachable is the compiler-required exit of the
 * varint loop, argued unreachable where it sits. It is declared unreachable
 * in the twin instead of carrying a negative, and that declaration is
 * fail-closed in both directions: the twin reds if it is ever reached. */
enum SnappyReject {
    kSnappyRejectNone = 0,
    kSnappyRejectPreambleByteMissing,
    kSnappyRejectPreambleFinalGroup,
    kSnappyRejectPreambleUnreachable,
    kSnappyRejectDeclaredOverCapacity,
    kSnappyRejectPastTerminal,
    kSnappyRejectUnderProduction,
    kSnappyRejectLiteralHeaderTruncated,
    kSnappyRejectLiteralPayloadTruncated,
    kSnappyRejectLiteralOverProduction,
    kSnappyRejectCopyHeaderTruncated,
    kSnappyRejectCopyOffsetZero,
    kSnappyRejectCopyOffsetBeforeStart,
    kSnappyRejectCopyOverProduction,
    kSnappyRejectCount
};

/* The one choke point. It records which branch refused and returns that
 * branch's status; `out` is null only where a caller has no interest in the
 * reason, and the ladder itself always supplies one. */
CUDEC_HOST_DEVICE inline cudec_status SnappyRefuse(SnappyReject branch,
                                                   cudec_status status,
                                                   SnappyReject* out) {
    if (out != 0) {
        *out = branch;
    }
    return status;
}

/* An element is reported in the shared vocabulary of
 * src/decode_sequence.h, which the chunk decoder's copy engine is the sole
 * consumer of. Snappy's element is a literal OR a copy, never both, so
 * exactly one of the two halves is non-empty and the other has length zero.
 *
 * A LITERAL fills the literals half: literals_len bytes from
 * src[literals_src ..) to dst[literals_dst ..). The two positions index
 * different buffers and no inequality relates them. match_len is zero.
 *
 * A COPY fills the match half: match_len bytes replicated inside dst,
 * reading from dst[match_src ..). The read may run into bytes this element
 * is itself writing, which is the pattern-repeating semantics an
 * overlapping Snappy copy has, so a sequential executor must copy byte by
 * byte in increasing order and a warp gather reduces by
 * offset = match_dst - match_src. match_src < match_dst always holds: an
 * offset of zero is refused. literals_len is zero.
 *
 * The TERMINAL element, the one handed back with *done set, has both
 * lengths zero and every position zero. Executing it copies nothing. It is
 * spelled out rather than left undefined because a caller that executes
 * unconditionally before testing *done reads every field.
 *
 * What the parser has already proved when it hands one back: for a
 * literal, literals_src + literals_len <= the source size; for a copy,
 * match_src < match_dst and match_dst + match_len <= the declared output
 * length; and in both cases the write end is at or below the declared
 * length, which is itself at or below the caller's capacity. */

/* The preamble alone, for a caller that must BOUND its destination before
 * any element exists - the batch entry point checks the declared length
 * against the chunk's capacity, and the CPU twin allocates from it after
 * that check. Nothing here licenses sizing an allocation from `*declared`
 * without comparing it against a capacity the caller already owns: the
 * value is attacker-controlled and reaches 2^32-1.
 *
 * SnappyParser::Begin reads the preamble through this same function, so
 * there is one varint reader and not two. On success `*declared` is the
 * declared uncompressed length and `*preamble_size` the bytes it
 * occupied. */
CUDEC_HOST_DEVICE inline cudec_status SnappyDeclaredLength(
    const unsigned char* src, uint64_t src_size, uint64_t* declared,
    uint64_t* preamble_size, SnappyReject* reject = 0) {
    uint64_t result = 0;
    for (uint64_t group = 0; group < kSnappyVarintGroups; group++) {
        if (group >= src_size) {
            return SnappyRefuse(kSnappyRejectPreambleByteMissing,
                                CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        const unsigned char byte = src[group];
        /* The final group is refused above 15 rather than truncated: a
         * decoder that masked the excess away would accept a declared
         * length the reference rejects outright. */
        if (group + 1 == kSnappyVarintGroups &&
            byte >= kSnappyVarintFinalMax) {
            return SnappyRefuse(kSnappyRejectPreambleFinalGroup,
                                CUDEC_ERR_CORRUPT_INPUT, reject);
        }
        result |= static_cast<uint64_t>(byte & kSnappyVarintPayload)
                  << (kSnappyVarintBits * group);
        if ((byte & kSnappyVarintContinue) == 0) {
            *declared = result;
            *preamble_size = group + 1;
            return CUDEC_OK;
        }
    }
    /* Unreachable: a fifth byte below 16 has its continuation bit clear and
     * returned above, and one at or above 16 was refused. The exit exists
     * because the compiler requires one, and it is the reason this branch
     * carries no crafted negative - it is declared unreachable in the twin
     * instead, which reds if it is ever reached. */
    return SnappyRefuse(kSnappyRejectPreambleUnreachable,
                        CUDEC_ERR_CORRUPT_INPUT, reject);
}

struct SnappyParser {
    const unsigned char* src;
    uint64_t src_size;
    uint64_t dst_capacity;
    uint64_t src_pos = 0;
    uint64_t dst_pos = 0;
    /* Valid once Begin has succeeded. */
    uint64_t declared = 0;
    bool begun = false;
    bool finished = false;
    /* Which ladder branch refused, valid after any call that returned a
     * reject status. Diagnostic only: no decode decision reads it, and the
     * verdict a caller acts on stays the returned cudec_status. */
    SnappyReject reject = kSnappyRejectNone;

    /* Reads the preamble and checks the declaration against the caller's
     * capacity. Idempotent, and Next calls it, so a caller that needs the
     * declared length up front pays for the varint once rather than twice. */
    CUDEC_HOST_DEVICE cudec_status Begin() {
        if (begun) {
            return CUDEC_OK;
        }
        uint64_t preamble_size = 0;
        const cudec_status status = SnappyDeclaredLength(
            src, src_size, &declared, &preamble_size, &reject);
        if (status != CUDEC_OK) {
            return status;
        }
        /* Against the caller's capacity, never the other way round: this is
         * the one place a hostile 4 GiB declaration is stopped from sizing
         * anything. */
        if (declared > dst_capacity) {
            return SnappyRefuse(kSnappyRejectDeclaredOverCapacity,
                                CUDEC_ERR_OUTPUT_TOO_SMALL, &reject);
        }
        src_pos = preamble_size;
        begun = true;
        return CUDEC_OK;
    }

    /* Parses the next element and advances both cursors past it. Exactly
     * one of three outcomes: CUDEC_OK with *element filled (execute it,
     * then call again); CUDEC_OK with *done set (the ONLY success exit,
     * reached when the source is consumed exactly and the declared length
     * was produced exactly); or a reject status.
     *
     * Liveness, stated as the caller needs it rather than as a slogan.
     * Every call that hands back a non-terminal element consumes at least
     * the tag byte, and the first call consumes the preamble as well. The
     * TERMINAL call consumes nothing - there is nothing left to consume -
     * so it is the parser and not the caller that must stop the loop: a
     * further call after it is refused. A driver that ignores *done
     * therefore terminates on a reject instead of spinning, which is what
     * a warp lane needs, because a lane that never leaves its element loop
     * holds the whole launch. src/lz4_block.h gets the same property from
     * its terminal run consuming the rest of the source; this format's
     * terminal element consumes nothing, so the flag does it instead.
     *
     * Nothing here loops on a stream-derived count. Three loops read the
     * stream - the varint groups, a literal's length bytes and a copy's
     * offset bytes - and each is a counted for whose trip count is bounded
     * by a compile-time maximum of five, four and four; the tag selects a
     * value inside that bound and cannot raise it. */
    CUDEC_HOST_DEVICE cudec_status Next(DecodeSequence* element, bool* done) {
        *done = false;
        if (finished) {
            /* Called past the terminal element. */
            return SnappyRefuse(kSnappyRejectPastTerminal,
                                CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        const cudec_status started = Begin();
        if (started != CUDEC_OK) {
            return started;
        }

        if (src_pos >= src_size) {
            /* Under-production rejects here; over-production cannot reach
             * this point, because every element's length was bounded by the
             * declared output remaining. Trailing bytes do not reach it
             * either: with the declared length already produced, the next
             * element finds zero output remaining and every element
             * produces at least one byte. */
            if (dst_pos != declared) {
                return SnappyRefuse(kSnappyRejectUnderProduction,
                                    CUDEC_ERR_CORRUPT_INPUT, &reject);
            }
            *element = DecodeSequence{};
            finished = true;
            *done = true;
            return CUDEC_OK;
        }

        /* src_pos < src_size here, so every `src_size - src_pos` below is at
         * least 1 and no header check underflows. dst_pos <= declared holds
         * on entry as the loop invariant the two output bounds maintain, so
         * `declared - dst_pos` does not underflow either. */
        const unsigned char tag = src[src_pos];
        const unsigned kind = tag & kSnappyTagKindMask;

        if (kind == kSnappyTagKindLiteral) {
            uint64_t length = tag >> kSnappyTagLengthShift;
            uint64_t header = kSnappyTagBytes;
            if (length > kSnappyLiteralInlineMax) {
                const uint64_t extra = length - kSnappyLiteralInlineMax;
                header += extra;
                if (header > src_size - src_pos) {
                    /* Header past the end. */
                    return SnappyRefuse(kSnappyRejectLiteralHeaderTruncated,
                                        CUDEC_ERR_CORRUPT_INPUT, &reject);
                }
                length = 0;
                for (uint64_t i = 0; i < extra; i++) {
                    length |= static_cast<uint64_t>(
                                  src[src_pos + kSnappyTagBytes + i])
                              << (kSnappyBitsPerByte * i);
                }
            }
            /* The encoded value is length-1, and this accumulator is 64-bit,
             * so the addition below cannot wrap. THE REFERENCE'S DOES: it
             * adds in 32 bits, so the four-byte class spelling 0xFFFFFFFF
             * becomes a zero-length literal there and the stream is
             * accepted. cudec refuses it - a length that only exists
             * because an accumulator wrapped is not a length, and
             * fail-closed on spec-nonsense outranks bug-for-bug parity
             * (prime directive 1), exactly as it does for LZ4's tolerated
             * offset == 0. This is the ONE point where cudec's accept set
             * is a strict subset of snappy's; the twin pins it as a
             * divergence rather than letting it hide inside a parity
             * count. Never widen the accumulator's wrap back in to regain
             * parity. */
            length += 1;
            src_pos += header;
            if (length > src_size - src_pos) {
                /* Literal bytes missing. */
                return SnappyRefuse(kSnappyRejectLiteralPayloadTruncated,
                                    CUDEC_ERR_CORRUPT_INPUT, &reject);
            }
            if (length > declared - dst_pos) {
                /* Over-production. */
                return SnappyRefuse(kSnappyRejectLiteralOverProduction,
                                    CUDEC_ERR_CORRUPT_INPUT, &reject);
            }
            *element = DecodeSequence{};
            element->literals_src = src_pos;
            element->literals_dst = dst_pos;
            element->literals_len = length;
            src_pos += length;
            dst_pos += length;
            return CUDEC_OK;
        }

        uint64_t offset_bytes = kSnappyCopy1OffsetBytes;
        if (kind == kSnappyTagKindCopy2) {
            offset_bytes = kSnappyCopy2OffsetBytes;
        } else if (kind == kSnappyTagKindCopy4) {
            offset_bytes = kSnappyCopy4OffsetBytes;
        }
        const uint64_t header = kSnappyTagBytes + offset_bytes;
        if (header > src_size - src_pos) {
            /* Header past the end. */
            return SnappyRefuse(kSnappyRejectCopyHeaderTruncated,
                                CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        uint64_t offset = 0;
        for (uint64_t i = 0; i < offset_bytes; i++) {
            offset |= static_cast<uint64_t>(src[src_pos + kSnappyTagBytes + i])
                      << (kSnappyBitsPerByte * i);
        }
        uint64_t length = 0;
        if (kind == kSnappyTagKindCopy1) {
            /* The narrow form splits its offset: three tag bits carry the
             * high byte, the trailing byte the low one. */
            offset |= static_cast<uint64_t>(tag >> kSnappyCopy1OffsetShift)
                      << kSnappyBitsPerByte;
            length = kSnappyCopy1MinLength +
                     ((tag >> kSnappyTagLengthShift) & kSnappyCopy1LengthMask);
        } else {
            length = (tag >> kSnappyTagLengthShift) + 1;
        }
        src_pos += header;
        /* offset == 0 has no meaning: there is no byte to replicate. snappy
         * refuses it in all three copy forms, with and without output
         * behind it, which is pinned by crafted negatives in the twin
         * rather than read off the reference's source. It is also
         * load-bearing for the kernel exactly as it is for LZ4 - a
         * closed-form gather reduces by offset, so a zero reaching the
         * device copy engine would be a modulo by zero. */
        if (offset == 0) {
            return SnappyRefuse(kSnappyRejectCopyOffsetZero,
                                CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        if (offset > dst_pos) {
            /* Reads before the output start. */
            return SnappyRefuse(kSnappyRejectCopyOffsetBeforeStart,
                                CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        if (length > declared - dst_pos) {
            /* Over-production. */
            return SnappyRefuse(kSnappyRejectCopyOverProduction,
                                CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        *element = DecodeSequence{};
        element->match_src = dst_pos - offset;
        element->match_dst = dst_pos;
        element->match_len = length;
        dst_pos += length;
        return CUDEC_OK;
    }
};

}  // namespace cudec_detail

#endif /* CUDEC_SNAPPY_BLOCK_H */
