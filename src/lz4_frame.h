/* The LZ4 frame (.lz4 container) envelope parser (issue #141): a candidate
 * byte range becomes a validated frame descriptor and block table - one
 * offset and compressed length per block - or it is refused with a defined
 * status. It decodes no block payload and touches no device. Internal
 * header, not part of the ABI, the sibling of src/lz4_block.h,
 * src/snappy_block.h, src/tilestream.h and src/zstd_frame.h. Frame spec
 * (public): lz4_Frame_format.md.
 *
 * WHY THE ENVELOPE MOVED OUT OF src/frame.cpp. The walk it holds is the one
 * LZ4 surface arbitrary bytes reach before anything else, and it is the gate
 * that decides which blocks the device decoder is asked about at all - so a
 * fail-open here admits a stream to the kernel rather than only to this rung.
 * src/frame.cpp reaches the device runtime through src/vendor_rt.h and joins
 * the vendor runtime library, which
 * puts it out of reach of the one runner that could drive arbitrary bytes
 * into it: fuzz/ links no cudec archive and builds with Clang on a machine
 * with no CUDA at all. Header-only and CUDA-free is what lets the parser and
 * the fuzz target be the same code rather than two readings of one
 * specification.
 *
 * THE BLOCK TABLE GROWS AND IS NOT A CALLER-PROVIDED ARRAY, which is the one
 * place this header departs from src/tilestream.h's shape. That container
 * declares its tile count in a uint16 field, so its caller can size an array
 * before the walk. A frame declares no block count anywhere: the table ends
 * at an end mark the walk has to reach, so the only bound available up front
 * is the frame's own length, and an array sized from it would be orders of
 * magnitude too large on a real frame. The growth is the caller's allocator,
 * and cudec_lz4f_decompress already catches the std::bad_alloc a hostile
 * length can drive.
 *
 * NOTHING IS READ BEFORE ITS BYTES ARE PROVEN PRESENT. Every field read below
 * is preceded by a length check written against the frame size rather than
 * against a running remainder, and in every such check both terms of the sum
 * are already bounded by the frame size, so the sum cannot wrap. The
 * block-length field is masked to 31 bits before it is used in any address
 * arithmetic. */
#ifndef CUDEC_LZ4_FRAME_H
#define CUDEC_LZ4_FRAME_H

#include "cudec.h"
#include "xxhash32.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cudec_detail {

constexpr uint32_t kLz4FrameMagic = 0x184D2204u;

/* The skippable-frame magic family, frame spec section "Skippable Frames":
 * the low nibble is the frame's own index and all sixteen spellings name a
 * legal member of the container. Held as its own range for the reason
 * src/zstd_frame.h holds the identical one - the two container formats share
 * these bytes and must not disagree about what they mean. */
constexpr uint32_t kLz4SkippableMagicMin = 0x184D2A50u;
constexpr uint32_t kLz4SkippableMagicMax = 0x184D2A5Fu;

/* Magic (4) + FLG (1) + BD (1) + HC (1) is the smallest legal header. */
constexpr size_t kLz4FrameMinHeaderBytes = 7;

/* The frame format's smallest block-max, 64 KiB, doubled twice per BD step.
 * Named for the reason issue #211 gives: digits in an expression cannot be
 * told from a lane count by anything that reads this tree. */
constexpr size_t kLz4FrameBlockMaxKiB = 64;

/* One parsed data block: an offset/length into the frame, and whether it is
 * stored uncompressed (a straight copy) or LZ4-compressed (GPU-decoded). */
struct Lz4FrameBlock {
    size_t src_off;
    size_t src_len;
    bool uncompressed;
};

/* The validated frame descriptor. Every field is derived from bytes this
 * parser has already bounds-checked and header-checksum-checked; none is
 * copied out of the frame unexamined. `body_off` is where the block table
 * starts, which is the one thing the walk below needs from the header. */
struct Lz4FrameDescriptor {
    bool block_checksum;
    bool content_size;
    bool content_checksum;
    size_t block_max;
    uint64_t declared_content_size;
    size_t body_off;
};

inline uint32_t Lz4FrameRead32LE(const unsigned char* p) {
    uint32_t v;
    std::memcpy(&v, p, 4); /* the frame format and the target hosts are LE */
    return v;
}

inline uint64_t Lz4FrameRead64LE(const unsigned char* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

/* Rung 1: the magic number and the frame descriptor, through the header
 * checksum. A malformed descriptor is CORRUPT_INPUT; a legal one this decoder
 * declines - a skippable frame, linked blocks, a dictionary id - is
 * UNSUPPORTED, so a caller can tell "these bytes are not a frame" from "this
 * is a frame I do not decode". Only the second is worth a CPU fallback. */
inline cudec_status Lz4ParseFrameDescriptor(const unsigned char* f,
                                            size_t frame_size,
                                            Lz4FrameDescriptor* out) {
    if (frame_size < 4) {
        return CUDEC_ERR_CORRUPT_INPUT; /* no magic to read */
    }
    const uint32_t magic = Lz4FrameRead32LE(f);
    /* The skippable range is separated from every other wrong magic BEFORE
     * the corrupt verdict, because those frames are legal: liblz4's frame API
     * steps over one and reports a complete frame. A caller told "corrupt"
     * about a valid skippable frame would have no reason to reach for a CPU
     * decoder, which is the confusion the two classes exist to prevent.
     * Stepping over one is refused as a scope line rather than as a
     * difficulty; MASTERPLAN section 17 is where that line is argued. */
    if (magic >= kLz4SkippableMagicMin && magic <= kLz4SkippableMagicMax) {
        return CUDEC_ERR_UNSUPPORTED;
    }
    if (frame_size < kLz4FrameMinHeaderBytes || magic != kLz4FrameMagic) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const unsigned flg = f[4];
    const unsigned bd = f[5];
    if (((flg >> 6) & 3) != 1 || (flg & 2) != 0) {
        return CUDEC_ERR_CORRUPT_INPUT; /* version / reserved bit */
    }
    const bool block_independent = (flg >> 5) & 1;
    const bool block_checksum = (flg >> 4) & 1;
    const bool content_size = (flg >> 3) & 1;
    const bool content_checksum = (flg >> 2) & 1;
    const bool dict_id = flg & 1;
    if ((bd & 0x8F) != 0) {
        return CUDEC_ERR_CORRUPT_INPUT; /* BD reserved bits */
    }
    const unsigned bmax = (bd >> 4) & 7;
    if (bmax < 4 || bmax > 7) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const size_t block_max = (kLz4FrameBlockMaxKiB << 10) << ((bmax - 4) * 2);
    if (!block_independent || dict_id) {
        return CUDEC_ERR_UNSUPPORTED; /* linked blocks / dictionaries */
    }

    /* Header checksum covers FLG..end-of-descriptor. */
    size_t pos = 6;
    uint64_t declared_content_size = 0;
    if (content_size) {
        if (frame_size < pos + 8) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        declared_content_size = Lz4FrameRead64LE(f + pos);
        pos += 8;
    }
    if (frame_size < pos + 1) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const unsigned hc = f[pos];
    if (((xxhash32(f + 4, pos - 4) >> 8) & 0xFF) != hc) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    pos += 1;

    out->block_checksum = block_checksum;
    out->content_size = content_size;
    out->content_checksum = content_checksum;
    out->block_max = block_max;
    out->declared_content_size = declared_content_size;
    out->body_off = pos;
    return CUDEC_OK;
}

/* Rung 2: the block table, from `desc.body_off` to the end mark. On success
 * `*tail_off` is the offset just past the end mark, which is where the
 * optional content checksum lives. Every accepted entry lies inside the
 * frame and is no larger than the descriptor's own block maximum.
 *
 * ON A REFUSAL `*tail_off` IS THE OFFSET THE WALK STOPPED AT - the block
 * header that ended it, or the position where the next one would have begun.
 * It is written on every path so a caller never reads a stale value, and it
 * exists because a caller that has to identify WHICH refusal happened would
 * otherwise re-derive it by scanning: fuzz/fuzz_lz4_frame.cpp exempts one
 * pinned strictness from its stricter-direction check, and a scan for the
 * byte pattern anywhere in the stream would excuse a real divergence that
 * merely happened to contain it. */
inline cudec_status Lz4WalkFrameBlocks(const unsigned char* f,
                                       size_t frame_size,
                                       const Lz4FrameDescriptor& desc,
                                       std::vector<Lz4FrameBlock>* blocks,
                                       size_t* tail_off) {
    size_t pos = desc.body_off;
    /* Fuel: every step consumes at least the 4 block-size bytes, so a frame
     * of frame_size bytes admits at most frame_size / 4 + 1 steps - a budget
     * no frame the guards below admit can reach. It makes a future bounds
     * bug a rejected frame instead of a spinning host thread. */
    uint64_t fuel = frame_size / 4 + 1;
    bool end_mark = false;
    while (fuel-- != 0) {
        const size_t header_off = pos;
        if (frame_size < pos + 4) {
            *tail_off = header_off;
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        const uint32_t bs = Lz4FrameRead32LE(f + pos);
        pos += 4;
        if (bs == 0) {
            end_mark = true;
            break;
        }
        const bool uncompressed = (bs >> 31) & 1;
        const size_t blen = bs & 0x7FFFFFFFu;
        if (blen == 0 || blen > desc.block_max || frame_size < pos + blen) {
            *tail_off = header_off;
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        if (desc.block_checksum) {
            if (frame_size < pos + blen + 4) {
                *tail_off = header_off;
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            if (xxhash32(f + pos, blen) != Lz4FrameRead32LE(f + pos + blen)) {
                *tail_off = header_off;
                return CUDEC_ERR_CORRUPT_INPUT;
            }
        }
        blocks->push_back({pos, blen, uncompressed});
        pos += blen + (desc.block_checksum ? 4 : 0);
    }
    if (!end_mark) {
        *tail_off = pos;
        return CUDEC_ERR_CORRUPT_INPUT; /* fuel exhausted: no end mark */
    }
    *tail_off = pos;
    return CUDEC_OK;
}

/* Rung 3: the two claims a frame makes about its own output, checked once the
 * output exists. A declared content size (FLG bit 3) must equal the produced
 * size - liblz4 rejects a frame whose declared size does not match, too large
 * or too small; cudec rejects it too (oracle parity). */
inline cudec_status Lz4VerifyFrameTail(const unsigned char* f,
                                       size_t frame_size,
                                       const Lz4FrameDescriptor& desc,
                                       size_t tail_off,
                                       const unsigned char* out,
                                       size_t produced) {
    if (desc.content_size && desc.declared_content_size != produced) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    if (desc.content_checksum) {
        if (frame_size < tail_off + 4) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        if (xxhash32(out, produced) != Lz4FrameRead32LE(f + tail_off)) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
    }
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_LZ4_FRAME_H */
