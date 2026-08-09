/* The GDeflate TileStream envelope parser (issue #174): a candidate byte
 * range becomes a validated tile table - one offset and compressed size per
 * tile, plus the total uncompressed size - or it is refused with a defined
 * status. It decodes nothing, reads no tile payload, and touches no device.
 * Internal header, not part of the ABI, the sibling of src/lz4_block.h,
 * src/snappy_block.h and src/zstd_frame.h.
 *
 * WHY THE ENVELOPE IS A SEPARATE THING AT ALL. The page bitstream carries no
 * uncompressed size and no checksum of any kind (MASTERPLAN section 11.4), so
 * every bound a decoder has for a GDeflate tile comes from outside the bits.
 * This container is where those bounds come from, which makes it the whole
 * structural defence: a tile size this parser gets wrong is a bound the kernel
 * then enforces against the wrong number. Nothing downstream can re-derive it.
 *
 * NOTHING IS SIZED FROM THE STREAM BEFORE IT IS BOUNDED. The table of contents
 * length is derived from a declared tile count, so the count is bounded and
 * the declared table is checked against the bytes actually present before a
 * single table entry is read. Every derived size is computed in uint64_t: the
 * largest total this container can describe is 65535 * 65536 bytes, which is
 * inside a uint32_t by 65536 bytes and would be one careless addition away
 * from wrapping there.
 *
 * A SINGLE-SOURCED HEADER RATHER THAN THE .h/.cpp PAIR THE ISSUE NAMED. Every
 * format parser in this tree is a header (lz4_block, snappy_block,
 * zstd_frame); the .cpp files under src/ are host orchestration that the CUDA
 * build adds to the library. A tilestream.cpp would have to join that list to
 * be compiled at all, which is the one build where this parser is not wanted -
 * it is host-only by design so the GPU-less runner and the sanitizer gate both
 * reach it. Header-only keeps it in every translation unit that asks for it
 * and adds no library membership question. The ABI entry that will link it is
 * issue #177.
 *
 * FIELD LAYOUT PROVENANCE, STATED RATHER THAN IMPLIED. The header and table
 * layout below is issue #174's statement of the DirectStorage TileStream
 * container, not a quotation from a specification this repository holds - the
 * dossier records that enveloping is external and does not pin this container
 * (MASTERPLAN section 11.4). Pinning these facts against the oracle and the
 * real DirectStorage vectors is issues #170 and #169; until those land, this
 * parser is self-consistent and unconfirmed, and a divergence they find is a
 * defect here rather than in them. */
#ifndef CUDEC_TILESTREAM_H
#define CUDEC_TILESTREAM_H

#include "cudec.h"

#include <stdint.h>

namespace cudec_detail {

/* The one tile size this container may declare. tileSizeIdx is a 2-bit field
 * with four spellings and exactly one of them is decodable here; the other
 * three are refused rather than mapped onto a default. */
constexpr uint64_t kTileStreamTileSize = 65536;
constexpr uint32_t kTileStreamTileSizeIdx64K = 1;

/* 8 bytes: id, magic, a 16-bit tile count, and one packed 32-bit word. */
constexpr uint64_t kTileStreamHeaderBytes = 8;
constexpr uint64_t kTileStreamTocEntryBytes = 4;

/* The declared tile count is a uint16, so this is the field's own ceiling and
 * not a policy this parser applies. It is asserted rather than re-checked at
 * run time: a runtime branch on it could never be reached, and a guard no
 * input can reach is a guard nothing proves. What the total-size arithmetic
 * actually rests on is this equality, so this is where it is pinned. */
constexpr uint64_t kTileStreamMaxTiles = 65535;
static_assert(kTileStreamMaxTiles == UINT16_MAX,
              "the tile-count ceiling is the declared field's width; the "
              "total-size arithmetic below is overflow-free only because of "
              "it");

/* One validated tile: where its compressed bytes start in the stream, how
 * many there are, and how many bytes it must produce. Every field is derived
 * and checked; none is copied out of the stream unexamined. */
struct TileStreamTile {
    uint64_t src_off;
    uint64_t src_size;
    uint64_t dst_size;
};

/* The validated envelope. `tiles` is written by the caller's array, which is
 * why the parse takes a capacity and refuses rather than allocating. */
struct TileStreamInfo {
    uint64_t tile_count;
    uint64_t total_uncompressed;
};

inline uint16_t TileStreamRead16LE(const unsigned char* p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                                 (static_cast<uint16_t>(p[1]) << 8));
}

inline uint32_t TileStreamRead32LE(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

/* Parses and validates the envelope at `stream[0 .. stream_size)`.
 *
 * On CUDEC_OK: *info holds the tile count and the total uncompressed size,
 * and tiles[0 .. tile_count) hold the per-tile ranges, each one wholly inside
 * the stream. On any other status nothing about *info or tiles[] is promised
 * beyond that no read outside the stream was performed to produce it.
 *
 * CUDEC_ERR_INVALID_ARGUMENT is for the caller's own mistake - a null pointer,
 * or a tile array too small for the count the stream declares. Everything the
 * bytes get wrong is CUDEC_ERR_CORRUPT_INPUT. The container carries no
 * optional features, so nothing here returns CUDEC_ERR_UNSUPPORTED: a
 * tileSizeIdx this build cannot decode is refused as corrupt because the
 * decision that only 64 KiB tiles exist is this parser's, and re-labelling it
 * UNSUPPORTED would promise a caller a fallback path that has no meaning yet.
 * That line moves the day a second tile size is real, and issue #177 is where
 * it would move. */
inline cudec_status TileStreamParse(const unsigned char* stream,
                                    uint64_t stream_size,
                                    TileStreamTile* tiles,
                                    uint64_t tiles_capacity,
                                    TileStreamInfo* info) {
    if (stream == nullptr || tiles == nullptr || info == nullptr) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }

    /* The header is read only after the bytes for it are known to be there.
     * A stream shorter than 8 bytes has no tile count to trust, so there is
     * nothing to derive a further bound from. */
    if (stream_size < kTileStreamHeaderBytes) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }

    const unsigned char id = stream[0];
    const unsigned char magic = stream[1];
    /* The container's only self-consistency check. It is one byte of
     * redundancy and catches a range that was never a TileStream at all;
     * it is not integrity, and nothing below treats it as such. */
    if (id != static_cast<unsigned char>(magic ^ 0xFFu)) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }

    const uint64_t tile_count = TileStreamRead16LE(stream + 2);
    if (tile_count == 0) {
        return CUDEC_ERR_CORRUPT_INPUT; /* an envelope describing nothing */
    }

    /* One 32-bit word, unpacked by shift and mask rather than read through a
     * bitfield struct: bitfield allocation order is implementation-defined,
     * so a struct would make this parser's acceptance depend on the compiler
     * that built it. */
    const uint32_t packed = TileStreamRead32LE(stream + 4);
    const uint32_t tile_size_idx = packed & 0x3u;
    const uint64_t last_tile_size = (packed >> 2) & 0x3FFFFu;
    const uint32_t reserved = (packed >> 20) & 0xFFFu;

    if (tile_size_idx != kTileStreamTileSizeIdx64K) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    /* Reserved bits are refused rather than ignored. A future revision that
     * gives them meaning must not decode here as if it had not: this parser
     * would produce bytes under rules it has never read. */
    if (reserved != 0) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    /* The field is 18 bits and a tile is 65536 bytes, so the field can express
     * sizes no tile can hold. Zero is the full-tile spelling. */
    if (last_tile_size >= kTileStreamTileSize) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }

    if (tiles_capacity < tile_count) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }

    /* Now the table's own length is derivable, and it is checked against the
     * bytes present before any entry is read. Both operands are bounded above
     * (8 and 4 * 65535), so the sum cannot wrap. */
    const uint64_t toc_bytes = tile_count * kTileStreamTocEntryBytes;
    const uint64_t toc_end = kTileStreamHeaderBytes + toc_bytes;
    if (stream_size < toc_end) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }

    /* Entry 0 is the LAST tile's compressed size, not an offset - the table
     * spends the slot tile 0 would need on the one size no following offset
     * could imply. Tile 0's data therefore begins immediately after the table.
     */
    const uint64_t last_tile_src_size =
        TileStreamRead32LE(stream + kTileStreamHeaderBytes);
    /* A zero-byte tile is refused, here and by the strict monotonicity below.
     * Every tile must produce at least one byte, and no bitstream produces a
     * byte from no bytes; accepting one would hand the kernel a range it can
     * only fail on later, further from the evidence. This is stricter than the
     * container's own rules require, deliberately, and issue #179 is where it
     * meets the real vectors. */
    if (last_tile_src_size == 0) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }

    /* Offsets first, as a chain: each one past the table and strictly above
     * its predecessor. Strictness is what makes every derived size non-zero
     * without a second subtraction to check.
     *
     * THERE IS DELIBERATELY NO PER-OFFSET `off < stream_size` CHECK HERE, and
     * the reason is that it could not reject anything the end-of-stream check
     * below does not. On an accepting path the last tile satisfies
     * `off_last + src_size_last <= stream_size` with `src_size_last >= 1`, so
     * `off_last < stream_size`; strict monotonicity puts every earlier offset
     * below `off_last`. A per-offset check would therefore be a branch no
     * input can reach as the sole cause of a refusal - unprovable by deleting
     * it, which is the one thing a guard here has to be. The property it would
     * have named is pinned by fixtures instead, in
     * tests/tilestream_host_negative.cpp: an offset at the end of the stream
     * and an offset at 0xFFFFFFFF are both refused with the branch absent.
     *
     * Nothing in this loop reads at `off`. The only reads are of the table,
     * whose whole extent was bounded above. */
    uint64_t prev_off = toc_end;
    for (uint64_t i = 1; i < tile_count; i++) {
        const uint64_t off = TileStreamRead32LE(
            stream + kTileStreamHeaderBytes + i * kTileStreamTocEntryBytes);
        if (off <= prev_off) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        tiles[i].src_off = off;
        prev_off = off;
    }

    /* Sizes, derived from the chain. Every subtraction below is on a pair the
     * loop above already ordered, so none of them can wrap. */
    tiles[0].src_off = toc_end;
    for (uint64_t i = 0; i + 1 < tile_count; i++) {
        tiles[i].src_size = tiles[i + 1].src_off - tiles[i].src_off;
    }
    tiles[tile_count - 1].src_size = last_tile_src_size;

    /* The last tile is the only one whose end is not implied by a following
     * offset, so it is the only one that can run off the end of the stream.
     * Checked with the addition on the left rather than a subtraction, and in
     * uint64_t, so a 32-bit-sized size and a 32-bit-sized offset cannot meet
     * above 4 GiB and wrap into a passing comparison. */
    const uint64_t last_end =
        tiles[tile_count - 1].src_off + tiles[tile_count - 1].src_size;
    if (last_end > stream_size) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }

    /* Output sizes: every tile is a full tile except possibly the last, and
     * zero is the last one's full-tile spelling. */
    for (uint64_t i = 0; i + 1 < tile_count; i++) {
        tiles[i].dst_size = kTileStreamTileSize;
    }
    tiles[tile_count - 1].dst_size =
        (last_tile_size == 0) ? kTileStreamTileSize : last_tile_size;

    uint64_t total = tile_count * kTileStreamTileSize;
    if (last_tile_size != 0) {
        total -= (kTileStreamTileSize - last_tile_size);
    }

    info->tile_count = tile_count;
    info->total_uncompressed = total;
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_TILESTREAM_H */
