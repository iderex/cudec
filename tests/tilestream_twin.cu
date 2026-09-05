/* The whole-TileStream ABI entry against the reference (issue #177).
 *
 * The reference's compressor produces the pages, this test wraps them in a
 * DirectStorage-shaped envelope, and cudec_gdeflate_tilestream_decompress
 * decodes the container end to end. What is asserted is what the entry adds
 * over the page batch decoder, so the page decode itself is not re-litigated
 * here - tests/gdeflate_device.cu owns that:
 *
 *  - the envelope's declaration is read correctly: tile count and total
 *    uncompressed size, from the host entry, on host bytes;
 *  - the decoded container equals the SOURCE and equals what the reference's
 *    own decompressor produces from the same pages, byte for byte, in both
 *    memory spaces;
 *  - a poisoned destination keeps its poison past bytes_written;
 *  - the same context decodes the same stream twice to identical bytes;
 *  - ONE corrupted tile fails alone: its own defined status with
 *    bytes_written 0, its region of the destination left as it was, every
 *    sibling still byte-exact, and the whole-stream answer non-OK with
 *    *bytes_written 0 - the frame precedent, not a partial file.
 *
 * The corpus is deliberately uneven: three full 64 KiB tiles and a short
 * trailing one, so the last-tile-size field of the envelope carries a value
 * and the tail is the case a fixed 64 KiB assumption would get wrong. */
#include "cudec.h"
#include "tilestream.h"

#include "require.h"
#include "vendor_rt_test.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::kTileStreamHeaderBytes;
using cudec_detail::kTileStreamTileSize;

constexpr unsigned char kDstPoison = 0xA5;
constexpr unsigned char kMagic = 0xFB;
constexpr unsigned char kId = static_cast<unsigned char>(kMagic ^ 0xFF);

typedef std::vector<unsigned char> Bytes;

/* The same named recurrence the block twin and the device test draw their
 * corpora from, so the stream is identical on every machine and a failure
 * reproduces. */
class Lcg {
   public:
    explicit Lcg(unsigned seed) : state_(seed) {}
    unsigned Next() {
        state_ = state_ * 1103515245u + 12345u;
        return (state_ >> 16) & 0xFFFFu;
    }

   private:
    unsigned state_;
};

Bytes MixedEntropy(unsigned seed, size_t n) {
    Lcg lcg(seed);
    Bytes v(n);
    for (size_t i = 0; i < n; i++) {
        const unsigned r = lcg.Next();
        v[i] = static_cast<unsigned char>((r % 4u == 0u) ? (r & 0xFFu)
                                                         : ('a' + (r % 5u)));
    }
    return v;
}

void Put16(Bytes* b, uint16_t v) {
    b->push_back(static_cast<unsigned char>(v & 0xFF));
    b->push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
}

void Put32(Bytes* b, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        b->push_back(static_cast<unsigned char>((v >> (i * 8)) & 0xFF));
    }
}

/* The packed word, assembled the way src/tilestream.h unpacks it: 2 bits of
 * tile size index, 18 of last-tile size, 12 reserved. */
uint32_t Packed(uint32_t tile_size_idx, uint32_t last_tile_size) {
    return (tile_size_idx & 0x3u) | ((last_tile_size & 0x3FFFFu) << 2);
}

/* Splits `in` into 64 KiB tiles and compresses each one as its own GDeflate
 * page, then lays the pages out under a well-formed envelope.
 *
 * ONE PAGE PER COMPRESSOR CALL, deliberately, rather than one call over the
 * whole input. The container's contract is that a tile is an independent page;
 * compressing tile by tile is what makes the payload actually satisfy it, and
 * it keeps the page boundaries this test asserts about under this test's
 * control rather than the reference's paging. */
bool BuildStream(int level, const Bytes& in, Bytes* stream,
                 std::vector<Bytes>* pages) {
    const size_t tile = static_cast<size_t>(kTileStreamTileSize);
    const size_t n = (in.size() + tile - 1u) / tile;
    if (n == 0 || n > 65535u) {
        return false;
    }
    pages->clear();
    for (size_t i = 0; i < n; i++) {
        const size_t off = i * tile;
        const size_t len = (in.size() - off < tile) ? (in.size() - off) : tile;
        libdeflate_gdeflate_compressor* c =
            libdeflate_alloc_gdeflate_compressor(level);
        if (c == nullptr) {
            return false;
        }
        size_t npages = 0;
        const size_t bound = libdeflate_gdeflate_compress_bound(c, len,
                                                                &npages);
        if (bound == 0 || npages != 1) {
            libdeflate_free_gdeflate_compressor(c);
            return false;
        }
        Bytes pool(bound, 0);
        libdeflate_gdeflate_out_page out;
        out.data = pool.data();
        out.nbytes = pool.size();
        const size_t total =
            libdeflate_gdeflate_compress(c, in.data() + off, len, &out, 1);
        libdeflate_free_gdeflate_compressor(c);
        if (total == 0 || out.nbytes == 0) {
            return false;
        }
        const unsigned char* p = static_cast<const unsigned char*>(out.data);
        pages->push_back(Bytes(p, p + out.nbytes));
    }

    const size_t last_len = in.size() - (n - 1u) * tile;
    stream->clear();
    stream->push_back(kId);
    stream->push_back(kMagic);
    Put16(stream, static_cast<uint16_t>(n));
    /* Zero is the full-tile spelling of the last tile's size. */
    Put32(stream, Packed(1u, (last_len == tile)
                                 ? 0u
                                 : static_cast<uint32_t>(last_len)));

    /* Entry 0 is the LAST tile's compressed size; entries 1..n-1 are the
     * offsets of tiles 1..n-1, accumulated from the end of the table. */
    const uint32_t toc_end =
        static_cast<uint32_t>(kTileStreamHeaderBytes) + 4u * static_cast<uint32_t>(n);
    Put32(stream, static_cast<uint32_t>((*pages)[n - 1u].size()));
    uint32_t off = toc_end;
    for (size_t i = 1; i < n; i++) {
        off += static_cast<uint32_t>((*pages)[i - 1u].size());
        Put32(stream, off);
    }
    for (size_t i = 0; i < n; i++) {
        stream->insert(stream->end(), (*pages)[i].begin(), (*pages)[i].end());
    }
    return true;
}

/* Where tile `i`'s compressed bytes begin inside the assembled stream. Derived
 * the same way the envelope declares them, so a mutation aimed at a tile lands
 * inside that tile and nowhere else. */
size_t TileOffset(const std::vector<Bytes>& pages, size_t i) {
    size_t off = static_cast<size_t>(kTileStreamHeaderBytes) + 4u * pages.size();
    for (size_t k = 0; k < i; k++) {
        off += pages[k].size();
    }
    return off;
}

/* What the reference's own decompressor makes of the same pages. The entry
 * under test must equal THIS, not merely equal the source: a decoder that
 * agreed with the source while disagreeing with the reference would be a
 * divergence this project reports rather than a pass. */
bool OracleDecode(const std::vector<Bytes>& pages, size_t out_size,
                  Bytes* out) {
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        return false;
    }
    out->assign(out_size, 0);
    const size_t tile = static_cast<size_t>(kTileStreamTileSize);
    bool ok = true;
    for (size_t i = 0; i < pages.size() && ok; i++) {
        libdeflate_gdeflate_in_page in;
        in.data = pages[i].data();
        in.nbytes = pages[i].size();
        const size_t off = i * tile;
        const size_t len = (out_size - off < tile) ? (out_size - off) : tile;
        size_t produced = 0;
        ok = libdeflate_gdeflate_decompress(d, &in, 1, out->data() + off, len,
                                            &produced) == LIBDEFLATE_SUCCESS &&
             produced == len;
    }
    libdeflate_free_gdeflate_decompressor(d);
    return ok;
}

/* One decode through the shipped entry, host-output, into a poisoned buffer
 * larger than the stream needs so the untouched tail is checkable. */
int DecodeHost(cudec_stream_ctx* ctx, const Bytes& stream, size_t out_size,
               size_t slack, Bytes* out, std::vector<cudec_chunk_result>* res,
               size_t* written, cudec_status* st) {
    out->assign(out_size + slack, kDstPoison);
    res->assign(res->size(), cudec_chunk_result());
    for (size_t i = 0; i < res->size(); i++) {
        (*res)[i].status = static_cast<int32_t>(CUDEC_ERR_NOT_IMPLEMENTED);
        (*res)[i].reserved = 0xFFFFFFFFu;
        (*res)[i].bytes_written = 0xFFFFFFFFFFFFFFFFull;
    }
    *written = 0xFFFFFFFFu;
    *st = cudec_gdeflate_tilestream_decompress_ctx(
        ctx, stream.data(), stream.size(), out->data(), out->size(),
        CUDEC_MEM_HOST, res->data(), res->size(), written);
    return 0;
}

}  // namespace

int main(void) {
    /* Three full tiles and a short trailing one. */
    const size_t tile = static_cast<size_t>(kTileStreamTileSize);
    const size_t tail = 12345u;
    const Bytes source = MixedEntropy(0x5EED, 3u * tile + tail);
    const size_t n_tiles = 4u;

    Bytes stream;
    std::vector<Bytes> pages;
    REQUIRE(BuildStream(6, source, &stream, &pages));
    REQUIRE(pages.size() == n_tiles);

    /* ---- The envelope reads as declared, on the host, with no device. ---- */
    size_t tiles = 0;
    size_t total = 0;
    REQUIRE(cudec_gdeflate_tilestream_info(stream.data(), stream.size(),
                                           &tiles, &total) == CUDEC_OK);
    REQUIRE(tiles == n_tiles);
    REQUIRE(total == source.size());

    Bytes oracle;
    REQUIRE(OracleDecode(pages, source.size(), &oracle));
    REQUIRE(oracle.size() == source.size());
    REQUIRE(std::memcmp(oracle.data(), source.data(), source.size()) == 0);

    cudec_stream_ctx* ctx = nullptr;
    REQUIRE(cudec_stream_ctx_create(&ctx) == CUDEC_OK);

    /* ---- Host output: byte-exact, poison beyond it intact. ---- */
    const size_t slack = 4096u;
    std::vector<cudec_chunk_result> res(n_tiles);
    Bytes out;
    size_t written = 0;
    cudec_status st = CUDEC_ERR_NOT_IMPLEMENTED;
    DecodeHost(ctx, stream, source.size(), slack, &out, &res, &written, &st);
    REQUIRE(st == CUDEC_OK);
    REQUIRE(written == source.size());
    REQUIRE(std::memcmp(out.data(), oracle.data(), oracle.size()) == 0);
    for (size_t i = 0; i < slack; i++) {
        REQUIRE_CTX(out[source.size() + i] == kDstPoison, "slack byte %zu", i);
    }
    for (size_t i = 0; i < n_tiles; i++) {
        const size_t expect = (i + 1u == n_tiles) ? tail : tile;
        REQUIRE_CTX(res[i].status == CUDEC_OK, "tile %zu status %d", i,
                    static_cast<int>(res[i].status));
        REQUIRE_CTX(res[i].bytes_written == expect, "tile %zu wrote %llu", i,
                    static_cast<unsigned long long>(res[i].bytes_written));
        REQUIRE(res[i].reserved == 0);
    }

    /* ---- The same context, the same stream, twice: identical bytes. ---- */
    {
        Bytes again;
        std::vector<cudec_chunk_result> res2(n_tiles);
        size_t written2 = 0;
        cudec_status st2 = CUDEC_ERR_NOT_IMPLEMENTED;
        DecodeHost(ctx, stream, source.size(), slack, &again, &res2, &written2,
                   &st2);
        REQUIRE(st2 == CUDEC_OK);
        REQUIRE(written2 == written);
        REQUIRE(again.size() == out.size());
        REQUIRE(std::memcmp(again.data(), out.data(), out.size()) == 0);
    }

    /* ---- Device output: the same bytes, decoded straight into VRAM. ---- */
    {
        void* d_out = nullptr;
        REQUIRE_RT(cudec_rt::device_malloc(&d_out, source.size()));
        REQUIRE_RT(cudec_rt::device_memset(d_out, kDstPoison, source.size()));
        std::vector<cudec_chunk_result> dres(n_tiles);
        size_t dwritten = 0;
        const cudec_status dst_st = cudec_gdeflate_tilestream_decompress_ctx(
            ctx, stream.data(), stream.size(), d_out, source.size(),
            CUDEC_MEM_DEVICE, dres.data(), dres.size(), &dwritten);
        REQUIRE(dst_st == CUDEC_OK);
        REQUIRE(dwritten == source.size());
        Bytes back(source.size(), 0);
        REQUIRE_RT(cudec_rt::memcpy(back.data(), d_out, source.size(),
                                    cudec_rt::memcpy_d2h));
        REQUIRE(std::memcmp(back.data(), oracle.data(), oracle.size()) == 0);
        REQUIRE_RT(cudec_rt::device_free(d_out));
    }

    /* ---- One corrupted tile fails alone. ----
     *
     * The mutation lands inside tile 1's compressed bytes, past its first
     * byte so the page header still parses and the refusal comes out of the
     * bitstream rather than out of a rejected header. If a mutation happens to
     * decode anyway, that is a corpus fact rather than a defect, so the tile is
     * only asserted about once the entry has actually refused it - and the
     * search below is bounded, so a stream in which nothing refuses reds
     * rather than passing quietly. */
    {
        const size_t base = TileOffset(pages, 1u);
        const size_t span = pages[1].size();
        REQUIRE(span > 8u);
        bool refused = false;
        for (size_t k = 4u; k < span && !refused; k++) {
            Bytes bad = stream;
            bad[base + k] = static_cast<unsigned char>(bad[base + k] ^ 0xFFu);
            std::vector<cudec_chunk_result> bres(n_tiles);
            Bytes bout;
            size_t bwritten = 0;
            cudec_status bst = CUDEC_OK;
            DecodeHost(ctx, bad, source.size(), slack, &bout, &bres, &bwritten,
                       &bst);
            if (bst == CUDEC_OK) {
                continue; /* this flip still decoded; try the next byte */
            }
            refused = true;

            /* The whole-stream answer is the frame precedent: no partial
             * file, and the failing tile's own status. */
            REQUIRE(bwritten == 0);
            REQUIRE(bst == static_cast<cudec_status>(bres[1].status));
            REQUIRE(bres[1].status != CUDEC_OK);
            REQUIRE(bres[1].bytes_written == 0);
            REQUIRE(bres[1].reserved == 0);

            /* The failing tile is isolated: every sibling decoded, reported
             * its own byte count, and put the right bytes in its own region -
             * and tile 1's region of the destination still holds the poison it
             * was filled with. */
            for (size_t i = 0; i < n_tiles; i++) {
                if (i == 1u) {
                    continue;
                }
                const size_t expect = (i + 1u == n_tiles) ? tail : tile;
                REQUIRE_CTX(bres[i].status == CUDEC_OK, "sibling %zu status %d",
                            i, static_cast<int>(bres[i].status));
                REQUIRE_CTX(bres[i].bytes_written == expect,
                            "sibling %zu wrote %llu", i,
                            static_cast<unsigned long long>(
                                bres[i].bytes_written));
                REQUIRE_CTX(std::memcmp(bout.data() + i * tile,
                                        oracle.data() + i * tile, expect) == 0,
                            "sibling %zu bytes", i);
            }
            for (size_t i = 0; i < tile; i++) {
                REQUIRE_CTX(bout[tile + i] == kDstPoison,
                            "failed tile byte %zu", i);
            }
            std::printf(
                "tilestream_twin: tile 1 refused at payload byte %zu with "
                "status %d, three siblings byte-exact\n",
                k, static_cast<int>(bres[1].status));
        }
        REQUIRE(refused);
    }

    cudec_stream_ctx_destroy(ctx);
    std::printf(
        "tilestream_twin: %zu-tile stream (%zu bytes) decoded through the "
        "TileStream entry in both memory spaces, equal to the reference\n",
        n_tiles, source.size());
    return 0;
}
