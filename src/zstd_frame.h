/* The Zstd frame and block header parser, and the v1 subset gate that sits
 * on top of it - the first thing any Zstd decode path in this project reads,
 * and the only place a frame is judged admissible. Single-sourced for host
 * and device, the sibling of src/lz4_block.h, src/snappy_block.h and
 * src/zstd_bitstream.h. Internal header, not part of the ABI.
 *
 * Every field layout, every derivation and every refusal below carries the
 * RFC 8878 section it was read from; the subset itself is
 * docs/MASTERPLAN.md section 12, which is where the decision to accept less
 * than the format offers is argued rather than restated here.
 *
 * TWO REJECT CLASSES, AND THE LINE BETWEEN THEM IS THE CONTRACT.
 * CUDEC_ERR_CORRUPT_INPUT means no conforming encoder produced these bytes.
 * CUDEC_ERR_UNSUPPORTED means one did and cudec does not decode that part of
 * the format. A caller that would fall back to a CPU decoder needs to tell
 * them apart, so a legal frame outside the subset must never be reported as
 * corrupt and a corrupt one must never be reported as merely unsupported.
 * Section 12.3 sets the line; this header is where it is executed.
 *
 * NOTHING IS EVER SIZED FROM THE STREAM BEFORE IT IS BOUNDED. The frame
 * header's own length depends on flag bits inside it, so the length is
 * derived first and checked against the available bytes before any field is
 * read; a block's declared size is checked against both the block maximum
 * and the bytes actually present before it is used as a step. The declared
 * content size is attacker-controlled and is never used to size anything
 * here at all - it is reported, and the caller's capacity remains the truth.
 *
 * Widths: sizes and offsets are uint64_t because the caller's are, and
 * because Frame_Content_Size is a 64-bit field a hostile stream can set to
 * anything. Mixing a 32-bit intermediate into a bound comparison is where an
 * overflow would hide, so there are none. */
#ifndef CUDEC_ZSTD_FRAME_H
#define CUDEC_ZSTD_FRAME_H

#include "cudec.h"

#include <stdint.h>

/* Guarded: src/lz4_block.h, src/snappy_block.h and src/zstd_bitstream.h
 * define the same macro for the same reason, and a device translation unit
 * that decodes more than one format includes more than one header. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* RFC 8878 section 3.1.1: the frame's own magic, little-endian. */
constexpr uint32_t kZstdMagic = 0xFD2FB528u;

/* Section 3.1.2: the skippable range. These frames are legal and trivially
 * steppable; refusing them is a scope line rather than a difficulty, and
 * section 12.4 is where that line is argued. */
constexpr uint32_t kZstdSkippableMagicMin = 0x184D2A50u;
constexpr uint32_t kZstdSkippableMagicMax = 0x184D2A5Fu;

/* Section 3.1.1.1.2 recommends decoders support windows up to 8 MB and, in
 * the same section, allows a decoder to refuse a frame demanding more than
 * its authorized range. The number is the spec's, not this project's. */
constexpr uint64_t kZstdMaxWindowSize = 8ull * 1024ull * 1024ull;

/* Section 3.1.1.2.4: Block_Maximum_Size is the smaller of Window_Size and
 * 128 KB. */
constexpr uint64_t kZstdBlockSizeCeiling = 128ull * 1024ull;

/* Section 3.1.1.2.2: block types. 3 is "not a block ... considered to be
 * corrupt data", which is the spec assigning the rejection class itself. */
constexpr uint8_t kZstdBlockTypeRaw = 0;
constexpr uint8_t kZstdBlockTypeRle = 1;
constexpr uint8_t kZstdBlockTypeCompressed = 2;
constexpr uint8_t kZstdBlockTypeReserved = 3;

/* The reject ladder, enumerated once, in the shape the Snappy parser and the
 * bitstream reader use: every refusal returns through one choke point naming
 * its rung, so the twin can require a negative per rung instead of counting
 * statuses that repeat. Ten of these return CUDEC_ERR_CORRUPT_INPUT and four
 * return CUDEC_ERR_UNSUPPORTED, so the status alone identifies nothing. */
enum ZstdFrameReject {
    kZstdFrameRejectNone = 0,
    kZstdFrameRejectMagicTruncated,
    kZstdFrameRejectMagicWrong,
    kZstdFrameRejectSkippableFrame,
    kZstdFrameRejectDescriptorTruncated,
    kZstdFrameRejectReservedBitSet,
    kZstdFrameRejectDictionaryId,
    kZstdFrameRejectContentSizeAbsent,
    kZstdFrameRejectHeaderTruncated,
    kZstdFrameRejectWindowTooLarge,
    kZstdFrameRejectBlockHeaderTruncated,
    kZstdFrameRejectBlockTypeReserved,
    kZstdFrameRejectBlockTooLarge,
    kZstdFrameRejectBlockBodyTruncated,
    kZstdFrameRejectCount
};

/* The one choke point. It records which rung refused and returns that rung's
 * status; `out` is null only where a caller has no interest in the reason,
 * and the ladder itself always supplies one. */
CUDEC_HOST_DEVICE inline cudec_status ZstdFrameRefuse(ZstdFrameReject rung,
                                                      cudec_status status,
                                                      ZstdFrameReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* What a parsed frame header says, in the terms the decode path needs.
 *
 * `header_size` counts from the magic through the last header byte, so a
 * caller steps by it to reach the first block header and never re-derives
 * the layout. `frame_content_size` is reported and never used as a size
 * here; `window_size` is the bound back-references are held to and, with the
 * 128 KB ceiling, fixes the block maximum. */
struct ZstdFrameHeader {
    uint64_t frame_content_size;
    uint64_t window_size;
    uint32_t header_size;
    bool single_segment;
    bool content_checksum;
};

/* Little-endian reads, byte by byte. The stream's byte order is fixed by the
 * format and the host's is not, and a cast to a wider type would also be an
 * unaligned access on a pointer the caller chose. */
CUDEC_HOST_DEVICE inline uint32_t ZstdRead32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

CUDEC_HOST_DEVICE inline uint64_t ZstdReadLe(const unsigned char* p,
                                             unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; i++) {
        value |= static_cast<uint64_t>(p[i]) << (8u * i);
    }
    return value;
}

/* Section 3.1.1.1.2. Window_Size is a base of 2^(10 + Exponent) plus
 * Mantissa eighths of that base. The exponent is five bits, so windowLog
 * reaches 41 and the whole value fits in 64 bits with room left; that is why
 * this returns a uint64_t rather than clamping, and why the caller compares
 * against the 8 MB line afterwards rather than this function deciding. */
CUDEC_HOST_DEVICE inline uint64_t ZstdWindowSize(unsigned char descriptor) {
    const uint64_t exponent = static_cast<uint64_t>(descriptor >> 3);
    const uint64_t mantissa = static_cast<uint64_t>(descriptor & 0x07u);
    const uint64_t base = static_cast<uint64_t>(1) << (10 + exponent);
    return base + (base / 8) * mantissa;
}

/* Section 3.1.1.1.1, the Frame_Header_Descriptor's bit assignment:
 *
 *   7-6  Frame_Content_Size_flag
 *   5    Single_Segment_flag
 *   4    Unused_bit
 *   3    Reserved_bit
 *   2    Content_Checksum_flag
 *   1-0  Dictionary_ID_flag
 *
 * The Reserved_bit "must be zero" (3.1.1.1.1.3) and a set one is corrupt.
 * The Unused_bit is a different thing wearing a similar name: 3.1.1.1.1.4
 * says a decoder compliant with this version "shall not interpret" it, so
 * refusing on it would refuse frames a future encoder may legally emit.
 * These two are one bit apart and swapping them is the mistake this comment
 * exists to stop. */
CUDEC_HOST_DEVICE inline unsigned ZstdFcsFieldSize(unsigned flag,
                                                   bool single_segment) {
    /* Section 3.1.1.1.1.1's table. Flag 0 is the only entry that depends on
     * Single_Segment_flag, and it is the one that decides whether the frame
     * declares its content size at all. */
    if (flag == 0) {
        return single_segment ? 1u : 0u;
    }
    return 1u << flag; /* 1 -> 2, 2 -> 4, 3 -> 8 */
}

/* Section 3.1.1.1.3's table: 0, 1, 2 or 4 bytes. Only flag 0 survives the
 * subset gate, so the other three sizes are never read; the function exists
 * so the header-length arithmetic is written once and stays honest if the
 * dictionary rung is ever taken (section 12.5 keeps it permanently out). */
CUDEC_HOST_DEVICE inline unsigned ZstdDidFieldSize(unsigned flag) {
    return flag == 0 ? 0u : (1u << (flag - 1u));
}

/* Parse and gate a frame header.
 *
 * `size` is the bytes available from `src`, not the frame's declared length:
 * a frame is bounded by the chunk it arrives in, and a header that reaches
 * past that is truncated whatever it claims. The refusals are ordered so
 * that each field is bounded before it is read, which is why the header
 * length is derived from the descriptor before the descriptor's dependents
 * are touched. */
CUDEC_HOST_DEVICE inline cudec_status ZstdParseFrameHeader(
    const unsigned char* src, uint64_t size, ZstdFrameHeader* out,
    ZstdFrameReject* reject) {
    if (reject != 0) {
        *reject = kZstdFrameRejectNone;
    }
    if (size < 4) {
        return ZstdFrameRefuse(kZstdFrameRejectMagicTruncated,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const uint32_t magic = ZstdRead32(src);
    if (magic != kZstdMagic) {
        /* The skippable range is separated from every other wrong magic
         * BEFORE the corrupt verdict, because those frames are legal. A
         * caller told "corrupt" about a valid skippable frame would have no
         * reason to try a CPU decoder, which is the confusion 12.3 exists to
         * prevent. */
        if (magic >= kZstdSkippableMagicMin && magic <= kZstdSkippableMagicMax) {
            return ZstdFrameRefuse(kZstdFrameRejectSkippableFrame,
                                   CUDEC_ERR_UNSUPPORTED, reject);
        }
        return ZstdFrameRefuse(kZstdFrameRejectMagicWrong,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    if (size < 5) {
        return ZstdFrameRefuse(kZstdFrameRejectDescriptorTruncated,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    const unsigned descriptor = src[4];
    const unsigned fcs_flag = descriptor >> 6;
    const bool single_segment = (descriptor & 0x20u) != 0;
    const bool reserved_bit = (descriptor & 0x08u) != 0;
    const bool content_checksum = (descriptor & 0x04u) != 0;
    const unsigned did_flag = descriptor & 0x03u;

    if (reserved_bit) {
        return ZstdFrameRefuse(kZstdFrameRejectReservedBitSet,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    /* Both of the next two are legal frames outside the subset, so both are
     * UNSUPPORTED and both are decided before the header length is used to
     * read anything: a dictionary id changes the layout, and a frame with no
     * declared content size cannot be placed in a caller's buffer before it
     * is decoded, which is the property the batch model rests on. */
    if (did_flag != 0) {
        return ZstdFrameRefuse(kZstdFrameRejectDictionaryId,
                               CUDEC_ERR_UNSUPPORTED, reject);
    }
    const unsigned fcs_size = ZstdFcsFieldSize(fcs_flag, single_segment);
    if (fcs_size == 0) {
        return ZstdFrameRefuse(kZstdFrameRejectContentSizeAbsent,
                               CUDEC_ERR_UNSUPPORTED, reject);
    }

    /* 4 magic + 1 descriptor + the window descriptor when Single_Segment is
     * clear (3.1.1.1.2) + the dictionary id, which the gate above has just
     * fixed at zero bytes + the content size. Every term is a small constant,
     * so the sum cannot overflow the 32 bits it is kept in. */
    const unsigned window_size_bytes = single_segment ? 0u : 1u;
    const unsigned did_bytes = ZstdDidFieldSize(did_flag);
    const uint32_t header_size =
        5u + window_size_bytes + did_bytes + fcs_size;
    if (size < header_size) {
        return ZstdFrameRefuse(kZstdFrameRejectHeaderTruncated,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    /* Section 3.1.1.1.1.1: the two-byte encoding carries the value minus
     * 256, so the whole 16-bit range maps to 256..65791 rather than wasting
     * the sizes a one-byte field already covers. */
    uint64_t content_size = ZstdReadLe(
        src + 5u + window_size_bytes + did_bytes, fcs_size);
    if (fcs_size == 2) {
        content_size += 256;
    }

    /* Section 3.1.1.1.1.2: with Single_Segment set the content must be
     * regenerated in one segment, so the window IS the content size and the
     * Window_Descriptor byte is absent. Section 12.2 records that this shape
     * is accepted without a window bound: the memory the caller must supply
     * is its own capacity check, and refusing here would refuse frames the
     * subset table says are in. */
    const uint64_t window_size =
        single_segment ? content_size : ZstdWindowSize(src[5]);
    if (!single_segment && window_size > kZstdMaxWindowSize) {
        return ZstdFrameRefuse(kZstdFrameRejectWindowTooLarge,
                               CUDEC_ERR_UNSUPPORTED, reject);
    }

    if (out != 0) {
        out->frame_content_size = content_size;
        out->window_size = window_size;
        out->header_size = header_size;
        out->single_segment = single_segment;
        out->content_checksum = content_checksum;
    }
    return CUDEC_OK;
}

/* What a parsed block header says. `body_size` is what the block occupies in
 * the stream and `block_size` is what the field declared: they differ for an
 * RLE block, where the declaration is the REGENERATED length and the body is
 * the single byte to repeat (3.1.1.2.3). Stepping by the declared size
 * across an RLE block is an over-read of up to 128 KB, which is why the two
 * are separate fields rather than one that means different things. */
struct ZstdBlockHeader {
    uint32_t block_size;
    uint32_t body_size;
    uint8_t block_type;
    bool last_block;
};

/* Parse and gate one block header plus the presence of its body.
 *
 * `src` points at the block header and `available` is the bytes left in the
 * frame from there, so the caller does no arithmetic this function then has
 * to trust. `window_size` comes from the frame header and, with the 128 KB
 * ceiling, is the block maximum of 3.1.1.2.4. */
CUDEC_HOST_DEVICE inline cudec_status ZstdParseBlockHeader(
    const unsigned char* src, uint64_t available, uint64_t window_size,
    ZstdBlockHeader* out, ZstdFrameReject* reject) {
    if (reject != 0) {
        *reject = kZstdFrameRejectNone;
    }
    if (available < 3) {
        return ZstdFrameRefuse(kZstdFrameRejectBlockHeaderTruncated,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    /* Section 3.1.1.2: three bytes, little-endian, holding Last_Block in bit
     * 0, Block_Type in bits 1-2 and Block_Size in the remaining 21. */
    const uint32_t raw = static_cast<uint32_t>(src[0]) |
                         (static_cast<uint32_t>(src[1]) << 8) |
                         (static_cast<uint32_t>(src[2]) << 16);
    const bool last_block = (raw & 0x01u) != 0;
    const uint8_t block_type = static_cast<uint8_t>((raw >> 1) & 0x03u);
    const uint32_t block_size = raw >> 3;

    if (block_type == kZstdBlockTypeReserved) {
        return ZstdFrameRefuse(kZstdFrameRejectBlockTypeReserved,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const uint64_t block_max = window_size < kZstdBlockSizeCeiling
                                   ? window_size
                                   : kZstdBlockSizeCeiling;
    if (static_cast<uint64_t>(block_size) > block_max) {
        return ZstdFrameRefuse(kZstdFrameRejectBlockTooLarge,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }
    const uint32_t body_size =
        block_type == kZstdBlockTypeRle ? 1u : block_size;
    /* 3 + body_size in 64-bit, against the bytes actually present. body_size
     * is bounded by block_max above, so the sum cannot wrap; the width is
     * 64-bit anyway because `available` is, and a narrower comparison is
     * where the wrap would have hidden. */
    if (available < static_cast<uint64_t>(3) + body_size) {
        return ZstdFrameRefuse(kZstdFrameRejectBlockBodyTruncated,
                               CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    if (out != 0) {
        out->block_size = block_size;
        out->body_size = body_size;
        out->block_type = block_type;
        out->last_block = last_block;
    }
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_FRAME_H */
