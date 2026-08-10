/* The LZ4 frame decoder held to liblz4's frame API (#23). Frames are
 * generated with LZ4F_compressFrame in block-independent mode (cudec's
 * supported subset), decoded on the GPU via cudec_lz4f_decompress, and
 * compared byte-exact to the original and to LZ4F_decompress. Also pins:
 * the library's XXH32 against liblz4's, the uncompressed-block path
 * (random data), and every reject branch (linked frame, dictionary,
 * corrupt magic/checksum, truncation, undersized output). */
#include "cudec.h"
#include "require.h"
#include "xxhash32.h"

extern "C" {
#include "lz4frame.h"
#include "xxhash.h"
}

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/* Deterministic corpus generator (own PRNG - reproducible across runs). */
std::vector<unsigned char> MakeData(size_t n, uint64_t seed, int mode) {
    std::vector<unsigned char> out(n);
    uint64_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        if (mode == 0) {
            out[i] = static_cast<unsigned char>('A' + (i / 37) % 26);
        } else if (mode == 1) {
            out[i] = static_cast<unsigned char>(s >> 56); /* incompressible */
        } else {
            out[i] = static_cast<unsigned char>((i / 8) % 2 ? (s >> 56) : 0);
        }
    }
    return out;
}

std::vector<unsigned char> CompressFrame(const std::vector<unsigned char>& in,
                                         bool independent, bool content_ck,
                                         bool block_ck,
                                         bool content_size = false) {
    LZ4F_preferences_t prefs;
    std::memset(&prefs, 0, sizeof(prefs));
    prefs.frameInfo.blockMode =
        independent ? LZ4F_blockIndependent : LZ4F_blockLinked;
    prefs.frameInfo.contentChecksumFlag =
        content_ck ? LZ4F_contentChecksumEnabled : LZ4F_noContentChecksum;
    prefs.frameInfo.blockChecksumFlag =
        block_ck ? LZ4F_blockChecksumEnabled : LZ4F_noBlockChecksum;
    /* Setting contentSize makes LZ4F emit the 8-byte declared-size field
     * (FLG bit 3), which the decoder must validate against the real size. */
    prefs.frameInfo.contentSize = content_size ? in.size() : 0;
    const size_t bound = LZ4F_compressFrameBound(in.size(), &prefs);
    std::vector<unsigned char> frame(bound);
    const size_t r = LZ4F_compressFrame(frame.data(), bound, in.data(),
                                        in.size(), &prefs);
    if (LZ4F_isError(r)) {
        std::fprintf(stderr, "LZ4F_compressFrame failed\n");
        std::abort();
    }
    frame.resize(r);
    return frame;
}

/* Decode a whole frame through liblz4's frame API (the oracle). Returns
 * true and fills *out with the decoded bytes on success; false if liblz4
 * rejects the frame. */
bool Lz4fDecode(const std::vector<unsigned char>& frame,
                std::vector<unsigned char>* out) {
    LZ4F_dctx* dctx = nullptr;
    if (LZ4F_isError(LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION))) {
        return false;
    }
    size_t src_off = 0, dst_off = 0;
    bool ok = true;
    size_t hint = 1;
    while (src_off < frame.size()) {
        size_t src_left = frame.size() - src_off;
        size_t dst_left = out->size() - dst_off;
        hint = LZ4F_decompress(dctx, out->data() + dst_off, &dst_left,
                               frame.data() + src_off, &src_left, nullptr);
        if (LZ4F_isError(hint)) {
            ok = false;
            break;
        }
        src_off += src_left;
        dst_off += dst_left;
        if (hint == 0) {
            break; /* frame complete */
        }
        if (src_left == 0 && dst_left == 0) {
            ok = false; /* no progress: truncated / undersized out */
            break;
        }
    }
    LZ4F_freeDecompressionContext(dctx);
    if (ok && hint != 0) {
        ok = false; /* input exhausted mid-frame */
    }
    if (ok) {
        out->resize(dst_off);
    }
    return ok;
}

/* Decodes a frame through cudec and requires byte-exact equality with the
 * original. */
int CheckFrameOk(const char* ctx, const std::vector<unsigned char>& frame,
                 const std::vector<unsigned char>& original) {
    std::vector<unsigned char> out(original.size() + 16, 0xEE);
    size_t written = 12345;
    REQUIRE_CTX(cudec_lz4f_decompress(frame.data(), frame.size(), out.data(),
                                      out.size(), &written) == CUDEC_OK,
                "%s", ctx);
    REQUIRE_CTX(written == original.size(), "%s size", ctx);
    REQUIRE_CTX(equal_bytes(out.data(), original.data(), original.size()),
                "%s bytes", ctx);
    return 0;
}

int CheckReject(const char* ctx, const std::vector<unsigned char>& frame,
                cudec_status expected) {
    std::vector<unsigned char> out(1 << 20, 0);
    size_t written = 999;
    REQUIRE_CTX(cudec_lz4f_decompress(frame.data(), frame.size(), out.data(),
                                      out.size(), &written) == expected,
                "%s", ctx);
    REQUIRE_CTX(written == 0, "%s bytes_written must be 0", ctx);
    return 0;
}

}  // namespace

int main() {
    /* 1. The library's XXH32 must match liblz4's, byte for byte. */
    for (size_t n : {size_t{0}, size_t{1}, size_t{15}, size_t{16},
                     size_t{17}, size_t{1000}, size_t{65537}}) {
        const auto d = MakeData(n, 0x51 + n, 1);
        REQUIRE_CTX(cudec_detail::xxhash32(d.data(), n) ==
                        XXH32(d.data(), n, 0),
                    "xxhash32 parity at n=%zu", n);
    }

    /* 2. Independent frames of every flag combination and size class decode
     * byte-exact. Sizes span single-block, multi-block, and the block
     * boundary; modes span compressible, incompressible (uncompressed
     * blocks), and mixed. */
    const size_t sizes[] = {0, 1, 100, 65535, 65536, 65537, 200000};
    for (size_t n : sizes) {
        for (int mode = 0; mode < 3; mode++) {
            const auto data = MakeData(n, 0x100 + n + mode, mode);
            for (int flags = 0; flags < 4; flags++) {
                const bool content_ck = flags & 1;
                const bool block_ck = flags & 2;
                const auto frame =
                    CompressFrame(data, true, content_ck, block_ck);
                char ctx[64];
                std::snprintf(ctx, sizeof(ctx), "n=%zu mode=%d flags=%d", n,
                              mode, flags);
                REQUIRE(CheckFrameOk(ctx, frame, data) == 0);
            }
        }
    }

    /* 2b. Content-size frames (FLG bit 3): the 8-byte declared size is
     * emitted and must be validated. A correct frame decodes byte-exact; a
     * frame whose declared size is corrupted must reject in parity with
     * liblz4 (which returns frameSize_wrong). */
    for (size_t n : {size_t{0}, size_t{100}, size_t{65537}, size_t{200000}}) {
        const auto data = MakeData(n, 0x777 + n, 0);
        const auto frame = CompressFrame(data, true, false, false, true);
        char ctx[48];
        std::snprintf(ctx, sizeof(ctx), "content-size n=%zu", n);
        REQUIRE(CheckFrameOk(ctx, frame, data) == 0);
    }
    {
        const auto data = MakeData(100000, 0x778, 0);
        auto frame = CompressFrame(data, true, false, false, true);
        /* Descriptor: FLG(4) BD(5) ContentSize(6..13) HC(14). Corrupt the
         * declared size and repair HC so the size-mismatch path (not the
         * header-checksum path) is what fires. liblz4 must also reject. */
        frame[6] ^= 0xFF;
        frame[14] =
            static_cast<unsigned char>((XXH32(frame.data() + 4, 10, 0) >> 8) &
                                       0xFF);
        std::vector<unsigned char> oout(data.size() + 4096);
        REQUIRE(Lz4fDecode(frame, &oout) == false);
        REQUIRE(CheckReject("content-size-mismatch", frame,
                            CUDEC_ERR_CORRUPT_INPUT) == 0);
    }

    /* 2c. On-device block reject - the frame.cpp<->kernel status seam. Use a
     * single-block compressed frame and mutate ONLY bytes inside the block
     * payload, never the 4-byte block-size field or the header. The host
     * parser validates block size and checksums but not block content, so a
     * payload-only mutation always reaches the GPU decoder: any cudec reject
     * here is a kernel-status-seam reject by construction, not a host-side
     * one. Also held to the oracle direction: whenever liblz4 rejects, cudec
     * rejects; when both accept, the bytes match (cudec may be stricter). */
    {
        const auto data = MakeData(40000, 0x321, 0); /* fits one 64 KiB block */
        const auto base = CompressFrame(data, true, false, false);
        /* Layout with no checksums / no content size: magic(4) FLG(5) BD(6)
         * HC(7) | block size(4) | block data(blen) | end mark(4). */
        const uint32_t bs = static_cast<uint32_t>(base[7]) |
                            (static_cast<uint32_t>(base[8]) << 8) |
                            (static_cast<uint32_t>(base[9]) << 16) |
                            (static_cast<uint32_t>(base[10]) << 24);
        REQUIRE((bs >> 31) == 0); /* compressible input -> a compressed block */
        const size_t blen = bs & 0x7FFFFFFFu;
        const size_t data_begin = 11;
        const size_t data_end = data_begin + blen;
        REQUIRE(data_end + 4 == base.size()); /* exactly one block + end mark */

        /* Output capacity >= the 64 KiB block max, so a mutation that still
         * decodes cleanly (to at most block_max bytes) can never trip the
         * assembly-stage OUTPUT_TOO_SMALL. Every cudec reject in this sweep
         * is then a kernel-status-seam reject, with nothing host-side left
         * to count. */
        const size_t kSeamCap = (size_t{64} << 10) + 4096;
        const size_t step = blen > 96 ? blen / 96 : 1;
        size_t seam_rejects = 0;
        for (size_t p = data_begin; p < data_end; p += step) {
            auto frame = base;
            frame[p] ^= 0x5A;

            std::vector<unsigned char> oout(kSeamCap);
            const bool oracle_ok = Lz4fDecode(frame, &oout);

            std::vector<unsigned char> cout(kSeamCap, 0xCC);
            size_t w = 1;
            const cudec_status s = cudec_lz4f_decompress(
                frame.data(), frame.size(), cout.data(), cout.size(), &w);

            if (s != CUDEC_OK) {
                /* Payload-only mutation, so this reject came from the GPU. */
                REQUIRE_CTX(w == 0, "seam mut@%zu: reject wrote", p);
                seam_rejects++;
            }
            if (!oracle_ok) {
                REQUIRE_CTX(s != CUDEC_OK, "seam mut@%zu: oracle rejects, "
                                           "cudec accepted",
                            p);
            } else if (s == CUDEC_OK) {
                REQUIRE_CTX(w == oout.size(), "seam mut@%zu: size vs oracle", p);
                REQUIRE_CTX(equal_bytes(cout.data(), oout.data(), w),
                            "seam mut@%zu: bytes vs oracle", p);
            }
        }
        REQUIRE(seam_rejects > 0); /* the kernel-status seam is actually hit */
    }

    /* 2d. A stored block BETWEEN two compressed ones.
     *
     * Assembly merges decoded blocks that are adjacent in device memory into
     * one copy, and only COMPRESSED blocks occupy device slots. So this is the
     * frame where the two sides of that adjacency disagree: blocks 0 and 2 sit
     * back to back in device memory and 64 KiB apart in the output, because
     * the stored block 1 lands between them there and nowhere else. A merge
     * that checked only the device side would write block 2 over block 1's
     * bytes. Nothing above builds this shape - every case there is all
     * compressed or all stored - so without it that half of the check is
     * unproven, and it was: deleting it left this suite green.
     *
     * The block table is asserted rather than assumed, because the fixture is
     * only worth its name while liblz4 still stores the middle block. LZ4F
     * stores a block it cannot shrink, and the top bit of the 4-byte block
     * size is what says it did. */
    {
        const size_t bm = size_t{64} << 10;
        std::vector<unsigned char> mixed;
        for (int part = 0; part < 3; part++) {
            /* Part 1 is PRNG bytes, which LZ4F cannot shrink; 0 and 2 are the
             * repeating-text mode and compress. */
            const auto p = MakeData(bm, 0x900 + part, part == 1 ? 1 : 0);
            mixed.insert(mixed.end(), p.begin(), p.end());
        }
        const auto frame = CompressFrame(mixed, true, false, false);

        size_t pos = 7; /* magic(4) FLG(1) BD(1) HC(1), then the block table */
        int compressed_before = 0;
        int stored = 0;
        int compressed_after = 0;
        for (int b = 0; b < 3; b++) {
            REQUIRE_CTX(pos + 4 <= frame.size(), "mixed table block %d", b);
            const uint32_t bs = static_cast<uint32_t>(frame[pos]) |
                                (static_cast<uint32_t>(frame[pos + 1]) << 8) |
                                (static_cast<uint32_t>(frame[pos + 2]) << 16) |
                                (static_cast<uint32_t>(frame[pos + 3]) << 24);
            if ((bs >> 31) != 0) {
                stored++;
            } else if (stored == 0) {
                compressed_before++;
            } else {
                compressed_after++;
            }
            pos += 4 + (bs & 0x7FFFFFFFu);
        }
        REQUIRE_CTX(compressed_before == 1 && stored == 1 &&
                        compressed_after == 1,
                    "stored-between-compressed fixture is %d/%d/%d, not 1/1/1",
                    compressed_before, stored, compressed_after);

        REQUIRE(CheckFrameOk("stored-between-compressed", frame, mixed) == 0);
    }

    /* 2e. A SHORT compressed block followed by a full one.
     *
     * The other half of the same adjacency test. Device slots are strided by
     * the frame's block max whatever a block actually decodes to, so a block
     * that decodes short leaves a hole before the next slot, and merging
     * across it would copy the hole into the output. LZ4F never emits a short
     * block anywhere but last, so no compressor-built frame reaches this; the
     * frame format permits it and liblz4 accepts it, which makes it exactly
     * the crafted-input case this decoder is supposed to survive. Built by
     * splicing one frame's block record in front of another's, both frames
     * carrying the same descriptor, so the header stays the one LZ4F wrote.
     *
     * Deleting the device-side test leaves this red and the rest of the suite
     * green. */
    {
        const auto short_data = MakeData(1000, 0xA01, 0);
        const auto full_data = MakeData(size_t{64} << 10, 0xA02, 0);
        const auto fa = CompressFrame(short_data, true, false, false);
        const auto fb = CompressFrame(full_data, true, false, false);

        /* magic(4) FLG(1) BD(1) HC(1) | size(4) payload | end mark(4). */
        auto record = [](const std::vector<unsigned char>& f) {
            const uint32_t bs = static_cast<uint32_t>(f[7]) |
                                (static_cast<uint32_t>(f[8]) << 8) |
                                (static_cast<uint32_t>(f[9]) << 16) |
                                (static_cast<uint32_t>(f[10]) << 24);
            return std::vector<unsigned char>(
                f.begin() + 7, f.begin() + 7 + 4 + (bs & 0x7FFFFFFFu));
        };
        REQUIRE((static_cast<uint32_t>(fa[10]) >> 7) == 0); /* both compressed */
        REQUIRE((static_cast<uint32_t>(fb[10]) >> 7) == 0);
        REQUIRE(fa[5] == fb[5] && fa[6] == fb[6]); /* same FLG and BD */

        std::vector<unsigned char> spliced(fa.begin(), fa.begin() + 7);
        const auto ra = record(fa);
        const auto rb = record(fb);
        spliced.insert(spliced.end(), ra.begin(), ra.end());
        spliced.insert(spliced.end(), rb.begin(), rb.end());
        spliced.insert(spliced.end(), 4, 0); /* end mark */

        std::vector<unsigned char> expect = short_data;
        expect.insert(expect.end(), full_data.begin(), full_data.end());

        /* The oracle has to accept it, or the fixture is proving a reject
         * rather than a placement. */
        std::vector<unsigned char> oout(expect.size() + 4096);
        REQUIRE(Lz4fDecode(spliced, &oout) == true);
        REQUIRE(oout.size() == expect.size());
        REQUIRE(equal_bytes(oout.data(), expect.data(), expect.size()));

        REQUIRE(CheckFrameOk("short-then-full", spliced, expect) == 0);
    }

    /* 3. Reject branches. */
    const auto data = MakeData(200000, 0x999, 2);

    /* Block-linked (liblz4's default) is unsupported, not corrupt. */
    REQUIRE(CheckReject("linked", CompressFrame(data, false, false, false),
                        CUDEC_ERR_UNSUPPORTED) == 0);

    /* A dictionary-id frame is unsupported: set the DictID FLG bit on an
     * otherwise valid independent frame and repair the header checksum so
     * the UNSUPPORTED path (not the checksum path) is what fires. */
    {
        auto frame = CompressFrame(data, true, false, false);
        frame[4] |= 0x01; /* FLG DictID bit */
        /* HC covers FLG..end-of-descriptor; no content size here, so the
         * descriptor is FLG,BD and HC is at offset 6. Recompute it. */
        frame[6] = static_cast<unsigned char>(
            (XXH32(frame.data() + 4, 2, 0) >> 8) & 0xFF);
        /* The DictID bit implies 4 more header bytes liblz4 would read, but
         * cudec rejects on the bit before parsing them - which is the
         * point. */
        REQUIRE(CheckReject("dictid", frame, CUDEC_ERR_UNSUPPORTED) == 0);
    }

    /* Corrupt magic. */
    {
        auto frame = CompressFrame(data, true, false, false);
        frame[0] ^= 0xFF;
        REQUIRE(CheckReject("bad-magic", frame, CUDEC_ERR_CORRUPT_INPUT) == 0);
    }

    /* Corrupt header checksum. */
    {
        auto frame = CompressFrame(data, true, false, false);
        frame[6] ^= 0xFF;
        REQUIRE(CheckReject("bad-hc", frame, CUDEC_ERR_CORRUPT_INPUT) == 0);
    }

    /* Corrupt content checksum (last 4 bytes of a content-checksummed
     * frame). */
    {
        auto frame = CompressFrame(data, true, true, false);
        frame[frame.size() - 1] ^= 0xFF;
        REQUIRE(CheckReject("bad-content-ck", frame,
                            CUDEC_ERR_CORRUPT_INPUT) == 0);
    }

    /* Corrupt block checksum. */
    {
        auto frame = CompressFrame(data, true, false, true);
        /* first block checksum sits after the header + first block data;
         * flip a byte late in the frame that is not the end mark / content
         * checksum - the middle of the frame is block data or a block
         * checksum, both of which must fail closed. */
        frame[frame.size() / 2] ^= 0xFF;
        std::vector<unsigned char> out(1 << 20, 0);
        size_t w = 1;
        const cudec_status s = cudec_lz4f_decompress(
            frame.data(), frame.size(), out.data(), out.size(), &w);
        REQUIRE(s == CUDEC_ERR_CORRUPT_INPUT);
        REQUIRE(w == 0);
    }

    /* Truncation: any prefix of a valid frame must reject, never over-read
     * or falsely succeed. */
    {
        const auto frame = CompressFrame(data, true, true, true);
        for (size_t cut : {size_t{3}, size_t{6}, size_t{10}, frame.size() / 2,
                           frame.size() - 1}) {
            std::vector<unsigned char> t(frame.begin(),
                                         frame.begin() +
                                             static_cast<long>(cut));
            std::vector<unsigned char> out(1 << 20, 0);
            size_t w = 1;
            const cudec_status s = cudec_lz4f_decompress(
                t.data(), t.size(), out.data(), out.size(), &w);
            REQUIRE_CTX(s != CUDEC_OK, "truncation to %zu accepted", cut);
            REQUIRE_CTX(w == 0, "truncation to %zu wrote", cut);
        }
    }

    /* Undersized output buffer. */
    {
        const auto frame = CompressFrame(data, true, false, false);
        std::vector<unsigned char> out(data.size() / 2, 0);
        size_t w = 1;
        REQUIRE(cudec_lz4f_decompress(frame.data(), frame.size(), out.data(),
                                      out.size(), &w) ==
                CUDEC_ERR_OUTPUT_TOO_SMALL);
        REQUIRE(w == 0);
    }

    std::printf("PASS: frame decode in liblz4 parity across sizes/flags/"
                "modes; xxhash32 verified; linked/dictid/corrupt/truncation/"
                "undersized all reject fail-closed\n");
    return 0;
}
