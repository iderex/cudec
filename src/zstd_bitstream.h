/* The Zstd backward bitstream reader - the one bit source every serial core
 * in M5 runs over: the FSE table-description decode, the three-state sequence
 * loop, and each of the four literal streams. Single-sourced for host and
 * device, the sibling of src/lz4_block.h and src/snappy_block.h. Internal
 * header, not part of the ABI.
 *
 * A Zstd bitstream is written FORWARD by the encoder and read BACKWARD by the
 * decoder. The decoder starts at the last byte of the stream, which carries a
 * start marker: that byte must be non-zero, its highest set bit is the marker
 * and is consumed, and every bit below it is live data. Decoding then walks
 * down through the remaining bytes, each contributing its eight bits from bit
 * 7 to bit 0. RFC 8878 section 4.2.1.1 states the rule and the reference
 * implements it in lib/common/bitstream.h, BIT_initDStream: `lastByte == 0`
 * is an error there, not a value to clamp.
 *
 * The fail-open this unit exists to prevent is READING ZEROS PAST THE START.
 * A reader that runs out of bits and returns zeros hands its caller a symbol
 * that was never in the stream, and every ladder above it then validates a
 * value the attacker chose by truncating. So exhaustion is sticky and
 * observable: the first read that cannot be satisfied refuses, and every read
 * after it refuses too, without the caller having to remember.
 *
 * SHAPE, stated because the alternative is the obvious one. Bits are located
 * by a closed-form position rather than by a refilled 64-bit container. The
 * container is a performance shape and it belongs with the kernel that needs
 * it; what this unit owes is a contract - which value, and which reject -
 * that a faster reader can be twin-tested against. A closed form has no
 * refill boundary, which is where an off-by-one in a backward walk hides.
 *
 * Widths: a stream is bounded by the block it sits in, so a bit index fits in
 * 64 bits with room to spare. The size is a uint64_t because the caller's is,
 * and mixing widths in a bound comparison is where an overflow would hide. */
#ifndef CUDEC_ZSTD_BITSTREAM_H
#define CUDEC_ZSTD_BITSTREAM_H

#include "cudec.h"

#include <stdint.h>

/* Guarded: src/lz4_block.h and src/snappy_block.h define the same macro for
 * the same reason, and a device translation unit that decodes more than one
 * format includes more than one header. The definitions are identical, so an
 * unguarded second one is legal rather than an error - which is exactly why
 * it would go unnoticed if they ever stopped being identical. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

constexpr unsigned kZstdBitsPerByte = 8;

/* The widest single read this reader accepts. The reference's own limit is
 * the same shape and for the same reason: a value assembled in a 64-bit
 * accumulator leaves room for a partial byte at each end, so the usable width
 * stops short of the accumulator's. Nothing in the Zstd format needs more -
 * the widest field is an FSE accuracy-log-bounded state at 9 bits and an
 * offset code's extra bits at 31. */
constexpr unsigned kZstdBitReadMax = 57;

/* The reject ladder, enumerated once, in the shape src/snappy_block.h uses:
 * every refusal returns through one choke point naming its rung, so the twin
 * can require a negative per rung instead of counting statuses that repeat. */
enum ZstdBitReject {
    kZstdBitRejectNone = 0,
    kZstdBitRejectEmptyStream,
    kZstdBitRejectNoStartMarker,
    kZstdBitRejectNotStarted,
    kZstdBitRejectReadTooWide,
    kZstdBitRejectPastStart,
    kZstdBitRejectAlreadyOverrun,
    kZstdBitRejectCount
};

CUDEC_HOST_DEVICE inline cudec_status ZstdBitRefuse(ZstdBitReject rung,
                                                    cudec_status status,
                                                    ZstdBitReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* Position of the highest set bit, 0-indexed from bit 0. Only ever called on
 * a non-zero byte - the caller refuses a zero final byte before reaching it -
 * so there is no "no bits set" answer to define. A counted loop over a
 * compile-time bound rather than an intrinsic, because this header compiles
 * for both the host and the device and __clz is not on both. */
CUDEC_HOST_DEVICE inline unsigned ZstdHighestSetBit(unsigned char byte) {
    unsigned highest = 0;
    for (unsigned bit = 0; bit < kZstdBitsPerByte; bit++) {
        if ((byte >> bit) & 1u) {
            highest = bit;
        }
    }
    return highest;
}

struct ZstdBitReader {
    const unsigned char* src;
    uint64_t size;
    /* Live bits: the ones below the start marker in the last byte, plus every
     * bit of every byte before it. Valid once Start has succeeded. */
    uint64_t bits_total = 0;
    uint64_t bits_read = 0;
    /* Bits of the final byte that sit below the start marker. Kept because
     * every bit position below is measured from it. */
    unsigned marker_bits = 0;
    bool started = false;
    /* Sticky. Once a read has been refused for want of bits, every later read
     * is refused too: a caller that misses one return value must not be able
     * to walk back into a stream that already ended. */
    bool overrun = false;
    ZstdBitReject reject = kZstdBitRejectNone;

    /* Reads the start marker and fixes the live-bit count. Idempotent. */
    CUDEC_HOST_DEVICE cudec_status Start() {
        if (started) {
            return CUDEC_OK;
        }
        if (size == 0) {
            return ZstdBitRefuse(kZstdBitRejectEmptyStream,
                                 CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        const unsigned char last = src[size - 1];
        if (last == 0) {
            /* No marker, so where the data starts is undefined. The reference
             * refuses this rather than assuming a width, and so does this: a
             * decoder that picked one would be decoding a bit sequence the
             * encoder never wrote. */
            return ZstdBitRefuse(kZstdBitRejectNoStartMarker,
                                 CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        marker_bits = ZstdHighestSetBit(last);
        /* (size - 1) whole bytes plus the bits below the marker. size >= 1
         * here, so the subtraction does not underflow, and the multiplication
         * cannot overflow for any size a block can hold. */
        bits_total = (size - 1) * kZstdBitsPerByte + marker_bits;
        bits_read = 0;
        started = true;
        return CUDEC_OK;
    }

    /* Bits not yet consumed. Valid only after Start has succeeded; a caller
     * asserting exact consumption compares this against zero. */
    CUDEC_HOST_DEVICE uint64_t BitsRemaining() const {
        return bits_total - bits_read;
    }

    CUDEC_HOST_DEVICE bool AtEnd() const { return bits_read == bits_total; }

    /* The value of one live bit, by index in consumption order. Closed form,
     * no cursor: index 0 is the bit just below the start marker in the final
     * byte, and the walk moves down through the bytes from there.
     *
     * The caller has already proved index < bits_total, which is what makes
     * every subtraction below non-negative: for index >= marker_bits, the
     * byte index is size - 2 - (index - marker_bits) / 8, and index <
     * bits_total = (size - 1) * 8 + marker_bits bounds that quotient by
     * size - 2. */
    CUDEC_HOST_DEVICE unsigned BitAt(uint64_t index) const {
        if (index < marker_bits) {
            const unsigned shift =
                static_cast<unsigned>(marker_bits - 1 - index);
            return (src[size - 1] >> shift) & 1u;
        }
        const uint64_t below = index - marker_bits;
        const uint64_t byte = size - 2 - below / kZstdBitsPerByte;
        const unsigned shift = static_cast<unsigned>(
            kZstdBitsPerByte - 1 - below % kZstdBitsPerByte);
        return (src[byte] >> shift) & 1u;
    }

    /* Reads `count` bits, most significant first in consumption order, into
     * *value. A count of zero yields zero and moves no cursor - the callers
     * that ask for a field of width zero are the FSE and Huffman decoders,
     * where a zero-width field is a legal encoding rather than a mistake.
     *
     * On any refusal *value is set to zero AND a reject status is returned.
     * Both, deliberately: the zero is so a caller that ignores the status
     * cannot read stale stack, and the status is so it cannot mistake the
     * zero for data. */
    CUDEC_HOST_DEVICE cudec_status ReadBits(unsigned count, uint64_t* value) {
        *value = 0;
        if (overrun) {
            return ZstdBitRefuse(kZstdBitRejectAlreadyOverrun,
                                 CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        if (!started) {
            /* Start is not called implicitly here. A reader whose Start
             * failed must not become usable by calling ReadBits, and a
             * caller that never called it has a bug this reports rather
             * than papers over. */
            return ZstdBitRefuse(kZstdBitRejectNotStarted,
                                 CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        if (count > kZstdBitReadMax) {
            return ZstdBitRefuse(kZstdBitRejectReadTooWide,
                                 CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        if (count == 0) {
            return CUDEC_OK;
        }
        if (count > BitsRemaining()) {
            /* The fail-open this unit exists to prevent. Sticky from here. */
            overrun = true;
            return ZstdBitRefuse(kZstdBitRejectPastStart,
                                 CUDEC_ERR_CORRUPT_INPUT, &reject);
        }
        uint64_t accumulated = 0;
        /* Counted, with a compile-time bound: count <= kZstdBitReadMax was
         * checked above, so this loop's trip count is not stream-derived and
         * no lane can be held here by a hostile stream. */
        for (unsigned i = 0; i < count; i++) {
            accumulated = (accumulated << 1) | BitAt(bits_read + i);
        }
        bits_read += count;
        *value = accumulated;
        return CUDEC_OK;
    }
};

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_BITSTREAM_H */
