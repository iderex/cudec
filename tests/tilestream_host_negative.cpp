/* Every reject branch in src/tilestream.h, one crafted fixture each, plus a
 * single-byte mutation sweep over the whole envelope of a valid stream
 * (issue #174).
 *
 * No GPU and no CUDA call: the parser is pure host arithmetic over a byte
 * range, so this runs on the GPU-less CI runner and, in the sanitizer build,
 * puts ASan and UBSan over the pointer arithmetic that a hostile envelope
 * drives. That is the point of splitting the envelope out of the kernel work -
 * the container is where every bound the decoder later enforces comes from,
 * and it is testable long before a tile can be decoded.
 *
 * It is a fail-closed test and not an oracle-parity one. Every crafted stream
 * here is malformed by construction and its expected status is pinned; whether
 * this parser agrees with the real DirectStorage container on well-formed
 * input is issues #169 and #179, which have the vectors this does not.
 *
 * The sweep's property is the one a per-branch fixture cannot state: over
 * every single-byte flip of a valid envelope's header and table, an ACCEPT is
 * only ever allowed to describe tiles that lie inside the stream. A mutation
 * that lands on a byte no rule reads may legitimately still parse, so the
 * sweep asserts the invariant rather than the verdict.
 *
 * WHAT THIS FILE PROVES BY VERDICT AND WHAT IT LEAVES TO THE SANITIZER. Most
 * of the parser's refusals change the answer when they are removed, so a
 * fixture here reds without one. Three do not, and they are the three whose
 * job is a BOUND rather than a policy: the 8-byte header length, the declared
 * table length, and the non-zero tile count. Delete any of those and the next
 * thing the parser does is read or index outside what it was given, which is
 * not a verdict at all - it is undefined behaviour that may well still produce
 * a refusal. The only mechanism that separates them is the ASan/UBSan build,
 * so every stream in those three cases is handed over in an allocation exactly
 * as long as the length the parser is told about. Nothing else in this file
 * makes that leg load-bearing, and a longer buffer would quietly take it back
 * out. */
#include "require.h"
#include "tilestream.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using cudec_detail::kTileStreamHeaderBytes;
using cudec_detail::kTileStreamMaxTiles;
using cudec_detail::kTileStreamTileSize;
using cudec_detail::TileStreamInfo;
using cudec_detail::TileStreamParse;
using cudec_detail::TileStreamTile;

using Bytes = std::vector<unsigned char>;

void Put16(Bytes* b, uint16_t v) {
    b->push_back(static_cast<unsigned char>(v & 0xFF));
    b->push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
}

void Put32(Bytes* b, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        b->push_back(static_cast<unsigned char>((v >> (i * 8)) & 0xFF));
    }
}

void Write32(Bytes* b, size_t at, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        (*b)[at + static_cast<size_t>(i)] =
            static_cast<unsigned char>((v >> (i * 8)) & 0xFF);
    }
}

/* The packed word, assembled the way the parser unpacks it: 2 bits of tile
 * size index, 18 of last-tile size, 12 reserved. */
uint32_t Packed(uint32_t tile_size_idx, uint32_t last_tile_size,
                uint32_t reserved) {
    return (tile_size_idx & 0x3u) | ((last_tile_size & 0x3FFFFu) << 2) |
           ((reserved & 0xFFFu) << 20);
}

constexpr unsigned char kMagic = 0xFB;
constexpr unsigned char kId = static_cast<unsigned char>(kMagic ^ 0xFF);

/* A well-formed envelope over `sizes`, one compressed size per tile, with the
 * declared last-tile output size. The payload bytes are filler: this parser
 * never reads them, and a test that made them meaningful would be asserting
 * something it cannot check. */
Bytes MakeEnvelope(const std::vector<uint32_t>& sizes,
                   uint32_t last_tile_size) {
    const uint16_t n = static_cast<uint16_t>(sizes.size());
    Bytes b;
    b.push_back(kId);
    b.push_back(kMagic);
    Put16(&b, n);
    Put32(&b, Packed(1, last_tile_size, 0));

    const uint32_t toc_end =
        static_cast<uint32_t>(kTileStreamHeaderBytes) + 4u * n;
    /* Entry 0 is the last tile's compressed size; entries 1..n-1 are offsets,
     * accumulated from the end of the table. */
    Put32(&b, sizes[sizes.size() - 1]);
    uint32_t off = toc_end;
    for (size_t i = 1; i < sizes.size(); i++) {
        off += sizes[i - 1];
        Put32(&b, off);
    }
    uint32_t payload = 0;
    for (size_t i = 0; i < sizes.size(); i++) {
        payload += sizes[i];
    }
    b.resize(toc_end + payload, 0xAB);
    return b;
}

/* Offsets of the fields, for the crafted mutations below. */
constexpr size_t kOffId = 0;
constexpr size_t kOffPacked = 4;
constexpr size_t kOffTileCount = 2;

/* A copy of the first `n` bytes in an allocation that is exactly `n` bytes
 * long, so a parser that read past the length it was handed reads past the
 * ALLOCATION too and the sanitizer build says so.
 *
 * This matters more than it looks. Handing the parser a pointer into a longer
 * buffer and a short length makes an over-read land on bytes that are still
 * validly allocated: neither the verdict nor ASan can then distinguish a
 * parser that respects the length from one that ignores it, and both
 * length-derived bounds in src/tilestream.h - the 8-byte header and the
 * declared table - are exactly the kind that can only be caught that way.
 *
 * A zero-length case still needs a non-null pointer, because a null one is the
 * INVALID_ARGUMENT branch rather than the truncation branch. One spare byte is
 * allocated for that case only, and the length passed is still zero. */
class ExactBuffer {
  public:
    ExactBuffer(const Bytes& src, size_t n)
        : bytes_(n == 0 ? 1 : n), size_(n) {
        for (size_t i = 0; i < n; i++) {
            bytes_[i] = src[i];
        }
    }
    const unsigned char* data() const { return bytes_.data(); }
    size_t size() const { return size_; }

  private:
    Bytes bytes_;
    size_t size_;
};

cudec_status Parse(const Bytes& b, TileStreamInfo* info,
                   std::vector<TileStreamTile>* tiles) {
    tiles->assign(70000, TileStreamTile{0, 0, 0});
    return TileStreamParse(b.data(), b.size(), tiles->data(), tiles->size(),
                           info);
}

/* Every tile the parse reported must lie wholly inside the stream, and its
 * output size must be a size a 64 KiB tile can hold. The sweep's invariant,
 * factored out because the positive cases assert it too. */
bool TilesAreInBounds(const std::vector<TileStreamTile>& tiles, uint64_t count,
                      uint64_t stream_size) {
    for (uint64_t i = 0; i < count; i++) {
        if (tiles[i].src_size == 0) {
            return false;
        }
        if (tiles[i].src_off > stream_size) {
            return false;
        }
        if (tiles[i].src_size > stream_size - tiles[i].src_off) {
            return false;
        }
        if (tiles[i].dst_size == 0 || tiles[i].dst_size > kTileStreamTileSize) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    TileStreamInfo info{};
    std::vector<TileStreamTile> tiles;

    /* ---- The positive cases first: a reject test whose "valid" stream is not
     * actually valid proves nothing, because every negative below would then
     * be rejected for the wrong reason. */
    {
        const Bytes one = MakeEnvelope({40}, 0);
        REQUIRE(Parse(one, &info, &tiles) == CUDEC_OK);
        REQUIRE(info.tile_count == 1);
        REQUIRE(info.total_uncompressed == kTileStreamTileSize);
        REQUIRE(tiles[0].src_off == kTileStreamHeaderBytes + 4);
        REQUIRE(tiles[0].src_size == 40);
        REQUIRE(tiles[0].dst_size == kTileStreamTileSize);
        REQUIRE(TilesAreInBounds(tiles, info.tile_count, one.size()));
    }
    {
        /* Three tiles with a short last one: the shape where every derivation
         * differs from its neighbours - tile 0's offset is implicit, tile 1's
         * size comes from the following offset, tile 2's from entry 0. */
        const Bytes three = MakeEnvelope({10, 20, 30}, 1234);
        REQUIRE(Parse(three, &info, &tiles) == CUDEC_OK);
        REQUIRE(info.tile_count == 3);
        REQUIRE(info.total_uncompressed == 2 * kTileStreamTileSize + 1234);
        const uint64_t toc_end = kTileStreamHeaderBytes + 12;
        REQUIRE(tiles[0].src_off == toc_end);
        REQUIRE(tiles[0].src_size == 10);
        REQUIRE(tiles[1].src_off == toc_end + 10);
        REQUIRE(tiles[1].src_size == 20);
        REQUIRE(tiles[2].src_off == toc_end + 30);
        REQUIRE(tiles[2].src_size == 30);
        REQUIRE(tiles[0].dst_size == kTileStreamTileSize);
        REQUIRE(tiles[1].dst_size == kTileStreamTileSize);
        REQUIRE(tiles[2].dst_size == 1234);
        REQUIRE(TilesAreInBounds(tiles, info.tile_count, three.size()));
    }
    {
        /* Trailing bytes after the last tile are not this parser's to refuse:
         * the container says where tiles are, not that it ends with one. */
        Bytes slack = MakeEnvelope({16}, 0);
        slack.push_back(0x00);
        REQUIRE(Parse(slack, &info, &tiles) == CUDEC_OK);
        REQUIRE(TilesAreInBounds(tiles, info.tile_count, slack.size()));
    }
    {
        /* The largest envelope the fields can describe, which is where the
         * total-output arithmetic is closest to the edge the header argues it
         * stays inside: every tile full, at the tile count's own ceiling. The
         * number is written out rather than recomputed from the same
         * expression the parser uses, because a test that restates the
         * derivation agrees with it however wrong both are.
         *
         * Pinned here rather than reached by fuzzing (issue #184). A table
         * this size is a quarter of a megabyte of table alone, so no fuzz
         * input under any workable length bound carries a tile count near the
         * ceiling; building the stream is the only route to this branch. */
        const uint64_t ceiling = 4294901760u;
        Bytes full = MakeEnvelope(std::vector<uint32_t>(static_cast<size_t>(kTileStreamMaxTiles), 1),
                                  0);
        REQUIRE(Parse(full, &info, &tiles) == CUDEC_OK);
        REQUIRE(info.tile_count == kTileStreamMaxTiles);
        REQUIRE(info.total_uncompressed == ceiling);
        REQUIRE(info.total_uncompressed < UINT32_MAX);
        REQUIRE(TilesAreInBounds(tiles, info.tile_count, full.size()));

        /* And one byte short of full on the last tile, the other end of the
         * same arithmetic: the subtraction that shortens the total must take
         * exactly the shortfall and not a tile. */
        Bytes almost = MakeEnvelope(
            std::vector<uint32_t>(static_cast<size_t>(kTileStreamMaxTiles), 1), 65535);
        REQUIRE(Parse(almost, &info, &tiles) == CUDEC_OK);
        REQUIRE(info.total_uncompressed == ceiling - 1);
    }

    /* ---- The caller's own mistakes: INVALID_ARGUMENT, not CORRUPT_INPUT.
     * A caller cannot tell "your buffer is too small" from "these bytes are
     * hostile" if both arrive as the same status. */
    {
        const Bytes v = MakeEnvelope({16}, 0);
        REQUIRE(TileStreamParse(nullptr, v.size(), tiles.data(), tiles.size(),
                                &info) == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(TileStreamParse(v.data(), v.size(), nullptr, 4, &info) ==
                CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(TileStreamParse(v.data(), v.size(), tiles.data(), tiles.size(),
                                nullptr) == CUDEC_ERR_INVALID_ARGUMENT);
        TileStreamTile one_slot{0, 0, 0};
        const Bytes two = MakeEnvelope({8, 8}, 0);
        REQUIRE(TileStreamParse(two.data(), two.size(), &one_slot, 1, &info) ==
                CUDEC_ERR_INVALID_ARGUMENT);
    }

    /* ---- One crafted fixture per reject branch, each pinned to its status. */
    {
        /* Shorter than the 8-byte header, at every length below it, each in an
         * allocation of exactly that length - see ExactBuffer for why the
         * short-length-into-a-long-buffer spelling proves nothing here. */
        const Bytes v = MakeEnvelope({16}, 0);
        for (size_t n = 0; n < kTileStreamHeaderBytes; n++) {
            const ExactBuffer buf(v, n);
            REQUIRE_CTX(TileStreamParse(buf.data(), buf.size(), tiles.data(),
                                        tiles.size(),
                                        &info) == CUDEC_ERR_CORRUPT_INPUT,
                        "header truncated to %zu bytes", n);
        }
    }
    {
        /* id != magic ^ 0xff. */
        Bytes v = MakeEnvelope({16}, 0);
        v[kOffId] = static_cast<unsigned char>(v[kOffId] ^ 0x01u);
        REQUIRE(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
    }
    {
        /* numTiles == 0. */
        Bytes v = MakeEnvelope({16}, 0);
        v[kOffTileCount] = 0;
        v[kOffTileCount + 1] = 0;
        REQUIRE(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
    }
    {
        /* tileSizeIdx != 1, all three other spellings. */
        for (uint32_t idx = 0; idx < 4; idx++) {
            if (idx == 1) {
                continue;
            }
            Bytes v = MakeEnvelope({16}, 0);
            Write32(&v, kOffPacked, Packed(idx, 0, 0));
            REQUIRE_CTX(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT,
                        "tileSizeIdx=%u", idx);
        }
    }
    {
        /* Any reserved bit set. All twelve, one at a time: a mask that had
         * lost a bit would still pass a single-bit fixture. */
        for (uint32_t bit = 0; bit < 12; bit++) {
            Bytes v = MakeEnvelope({16}, 0);
            Write32(&v, kOffPacked, Packed(1, 0, 1u << bit));
            REQUIRE_CTX(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT,
                        "reserved bit %u", bit);
        }
    }
    {
        /* lastTileSize at and above the tile size, up to the field's width. */
        const uint32_t cases[] = {65536, 65537, 0x3FFFF};
        for (uint32_t last : cases) {
            Bytes v = MakeEnvelope({16}, 0);
            Write32(&v, kOffPacked, Packed(1, last, 0));
            REQUIRE_CTX(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT,
                        "lastTileSize=%u", last);
        }
    }
    {
        /* The declared table runs past the end of the stream. Exactly sized,
         * for the reason ExactBuffer gives: a vector's allocation may be
         * larger than its length, and an over-read into that slack is neither
         * a wrong verdict nor a sanitizer report. */
        Bytes v = MakeEnvelope({16}, 0);
        v[kOffTileCount] = 0x10; /* 4112 tiles, table far beyond the bytes */
        v[kOffTileCount + 1] = 0x10;
        const ExactBuffer buf(v, v.size());
        REQUIRE(TileStreamParse(buf.data(), buf.size(), tiles.data(),
                                tiles.size(),
                                &info) == CUDEC_ERR_CORRUPT_INPUT);
    }
    {
        /* The last tile's compressed size is zero. */
        Bytes v = MakeEnvelope({16}, 0);
        Write32(&v, kTileStreamHeaderBytes, 0);
        REQUIRE(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
    }
    {
        /* The last tile's size runs off the end of the stream. */
        Bytes v = MakeEnvelope({16}, 0);
        Write32(&v, kTileStreamHeaderBytes, 17);
        REQUIRE(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
        /* And by exactly one byte, which is the boundary the check owns. */
        Bytes exact = MakeEnvelope({16}, 0);
        Write32(&exact, kTileStreamHeaderBytes, 16);
        REQUIRE(Parse(exact, &info, &tiles) == CUDEC_OK);
    }
    {
        /* An offset pointing inside the table itself. */
        Bytes v = MakeEnvelope({16, 16}, 0);
        Write32(&v, kTileStreamHeaderBytes + 4, 12);
        REQUIRE(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
        /* And landing exactly on the table's end, which would make tile 0
         * zero bytes long. */
        Bytes at_end = MakeEnvelope({16, 16}, 0);
        Write32(&at_end, kTileStreamHeaderBytes + 4,
                static_cast<uint32_t>(kTileStreamHeaderBytes) + 8);
        REQUIRE(Parse(at_end, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
    }
    {
        /* Offsets that do not increase: equal, and going backwards. */
        Bytes equal = MakeEnvelope({16, 16, 16}, 0);
        const uint32_t off1 =
            static_cast<uint32_t>(kTileStreamHeaderBytes) + 12 + 16;
        Write32(&equal, kTileStreamHeaderBytes + 8, off1);
        REQUIRE(Parse(equal, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);

        Bytes backwards = MakeEnvelope({16, 16, 16}, 0);
        Write32(&backwards, kTileStreamHeaderBytes + 8, off1 - 1);
        REQUIRE(Parse(backwards, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
    }
    {
        /* An offset at or beyond the end of the stream. There is no branch in
         * the parser that names this case - src/tilestream.h argues that one
         * could never be the sole cause of a refusal - so these two pin the
         * PROPERTY, and they are what would red if that argument were ever
         * wrong. */
        Bytes v = MakeEnvelope({16, 16}, 0);
        Write32(&v, kTileStreamHeaderBytes + 4,
                static_cast<uint32_t>(v.size()));
        REQUIRE(Parse(v, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);

        Bytes far = MakeEnvelope({16, 16}, 0);
        Write32(&far, kTileStreamHeaderBytes + 4, 0xFFFFFFFFu);
        REQUIRE(Parse(far, &info, &tiles) == CUDEC_ERR_CORRUPT_INPUT);
    }

    /* ---- The mutation sweep. Every single-byte flip of every bit across the
     * header and the whole table of a three-tile envelope. The verdict is not
     * asserted - a flip in a byte no rule reads is allowed to parse - but an
     * accepted parse must never describe a tile outside the stream. */
    {
        const Bytes base = MakeEnvelope({10, 20, 30}, 1234);
        const size_t envelope_bytes = kTileStreamHeaderBytes + 12;
        size_t accepted = 0;
        size_t rejected = 0;
        for (size_t byte = 0; byte < envelope_bytes; byte++) {
            for (int bit = 0; bit < 8; bit++) {
                Bytes v = base;
                v[byte] = static_cast<unsigned char>(v[byte] ^ (1u << bit));
                const cudec_status st = Parse(v, &info, &tiles);
                if (st == CUDEC_OK) {
                    accepted++;
                    REQUIRE_CTX(
                        TilesAreInBounds(tiles, info.tile_count, v.size()),
                        "accepted mutant byte %zu bit %d describes a tile "
                        "outside its %zu-byte stream",
                        byte, bit, v.size());
                    REQUIRE_CTX(info.tile_count != 0, "byte %zu bit %d", byte,
                                bit);
                } else {
                    REQUIRE_CTX(st == CUDEC_ERR_CORRUPT_INPUT ||
                                    st == CUDEC_ERR_INVALID_ARGUMENT,
                                "byte %zu bit %d gave status %d", byte, bit,
                                static_cast<int>(st));
                    rejected++;
                }
            }
        }
        /* Both arms have to be non-empty, or the sweep proved nothing: all
         * rejects would mean the invariant was never evaluated on an accept,
         * and all accepts would mean no mutation reached a rule. */
        REQUIRE(accepted > 0);
        REQUIRE(rejected > 0);
        std::printf("mutation sweep: %zu accepted, %zu rejected\n", accepted,
                    rejected);
    }

    /* ---- Truncation sweep: the same valid stream at every length. Nothing
     * may be accepted whose tiles reach past the length it was given. */
    {
        const Bytes base = MakeEnvelope({10, 20, 30}, 1234);
        for (size_t n = 0; n <= base.size(); n++) {
            const ExactBuffer buf(base, n);
            const cudec_status st = TileStreamParse(
                buf.data(), buf.size(), tiles.data(), tiles.size(), &info);
            if (st == CUDEC_OK) {
                REQUIRE_CTX(TilesAreInBounds(tiles, info.tile_count, n),
                            "accepted at length %zu describes a tile outside "
                            "it",
                            n);
            } else {
                REQUIRE_CTX(st == CUDEC_ERR_CORRUPT_INPUT,
                            "length %zu gave status %d", n,
                            static_cast<int>(st));
            }
        }
    }

    std::printf("PASS\n");
    return 0;
}
