/* The LZ4 frame parse's reject branches and its host assembly path, driven
 * with NO GPU (issue #42). Every case here rejects, or decodes an
 * all-uncompressed frame, BEFORE the first CUDA call: the frame descriptor and
 * block-table walk in src/frame.cpp reject on the host, and an all-uncompressed
 * frame assembles entirely on the host (DecodeAndAssemble makes no CUDA call
 * when there are no compressed blocks). So this runs on the GPU-less CI runner
 * and, under the sanitizer gate, gives frame.cpp's hostile-input pointer
 * arithmetic the ASan/UBSan coverage the GPU-labeled frame_twin cannot get in
 * CI. It is a memory-safety and fail-closed test, not an oracle-parity one:
 * every crafted frame is malformed by construction and its expected status is
 * pinned. Header checksums are computed with the library's own xxhash32.h so
 * the deeper reject branches (past the header-checksum gate) are reachable. */
#include "cudec.h"
#include "require.h"
#include "xxhash32.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

void PutMagic(Bytes* f) {
    /* 0x184D2204, little-endian. */
    f->push_back(0x04);
    f->push_back(0x22);
    f->push_back(0x4D);
    f->push_back(0x18);
}

void Put32(Bytes* f, uint32_t v) {
    f->push_back(static_cast<unsigned char>(v & 0xFF));
    f->push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
    f->push_back(static_cast<unsigned char>((v >> 16) & 0xFF));
    f->push_back(static_cast<unsigned char>((v >> 24) & 0xFF));
}

void Put64(Bytes* f, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        f->push_back(static_cast<unsigned char>((v >> (i * 8)) & 0xFF));
    }
}

/* The frame header checksum byte: (XXH32(descriptor) >> 8) & 0xFF over the
 * `desc_len` descriptor bytes that start at offset 4 (FLG onward). */
unsigned char Hc(const Bytes& f, size_t desc_len) {
    return static_cast<unsigned char>(
        (cudec_detail::xxhash32(f.data() + 4, desc_len) >> 8) & 0xFF);
}

/* Builds a frame of UNCOMPRESSED blocks with a correct header checksum. BD is
 * fixed at 0x40 (bmax 4 -> 64 KiB block max) unless a caller overrides it. FLG
 * bit 4 (block checksum) and bit 2 (content checksum) are honored so those
 * branches are reachable; every block is stored uncompressed, so a well-formed
 * frame decodes entirely on the host. `content_size` is the optional declared
 * content-size field (FLG bit 3): nullptr omits it (the descriptor is the
 * 2-byte FLG,BD); non-null inserts the 8-byte little-endian size after BD and
 * extends the header-checksum descriptor to 10 bytes. */
Bytes BuildFrame(unsigned char flg, unsigned char bd,
                 const std::vector<Bytes>& blocks,
                 const uint64_t* content_size = nullptr) {
    const bool block_ck = (flg >> 4) & 1;
    const bool content_ck = (flg >> 2) & 1;
    Bytes f;
    PutMagic(&f);
    f.push_back(flg);
    f.push_back(bd);
    size_t desc_len = 2; /* FLG,BD when there is no content size */
    if (content_size != nullptr) {
        Put64(&f, *content_size);
        desc_len = 10; /* FLG,BD,8-byte content size */
    }
    f.push_back(Hc(f, desc_len));
    Bytes out; /* the assembled content, for the content checksum */
    for (const auto& b : blocks) {
        Put32(&f, 0x80000000u | static_cast<uint32_t>(b.size())); /* uncompressed */
        f.insert(f.end(), b.begin(), b.end());
        if (block_ck) {
            Put32(&f, cudec_detail::xxhash32(b.data(), b.size()));
        }
        out.insert(out.end(), b.begin(), b.end());
    }
    Put32(&f, 0); /* end mark */
    if (content_ck) {
        Put32(&f, cudec_detail::xxhash32(out.data(), out.size()));
    }
    return f;
}

/* Thin wrapper: BuildFrame with the optional declared content-size field
 * present. Lets the content-size match/mismatch branch in src/frame.cpp be
 * driven in both directions. The caller passes an FLG with bit 3 set (0x68). */
Bytes BuildFrameContentSize(unsigned char flg, unsigned char bd,
                            const std::vector<Bytes>& blocks,
                            uint64_t declared) {
    return BuildFrame(flg, bd, blocks, &declared);
}

/* Copies the crafted frame into an EXACTLY-sized heap allocation and decodes
 * from THAT pointer. The crafted frames are assembled with std::vector, whose
 * capacity is rounded up past the logical size, so a 1-4-byte over-read past
 * frame_size would read valid slack and leave ASan green - hiding exactly the
 * truncation/off-by-one class this gate exists to catch. A tight allocation
 * puts an ASan redzone immediately after the last frame byte, so any such
 * over-read reds the sanitizer build. */
cudec_status DecodeTight(const Bytes& frame, unsigned char* dst,
                         size_t dst_capacity, size_t* written) {
    const size_t n = frame.size();
    auto tight = std::make_unique<unsigned char[]>(n);
    if (n != 0) {
        std::memcpy(tight.get(), frame.data(), n);
    }
    return cudec_lz4f_decompress(tight.get(), n, dst, dst_capacity, written);
}

/* A frame that must be rejected: pins the exact status AND that no output byte
 * is presented (bytes_written stays 0). The output buffer is generous so a
 * reject is never masked by an OUTPUT_TOO_SMALL from a small buffer. */
int ExpectReject(const char* name, const Bytes& frame, cudec_status expected) {
    Bytes out(1u << 16, 0xCC);
    size_t written = 777;
    const cudec_status s = DecodeTight(frame, out.data(), out.size(), &written);
    REQUIRE_CTX(s == expected, "%s: status %d (want %d)", name,
                static_cast<int>(s), static_cast<int>(expected));
    REQUIRE_CTX(written == 0, "%s: bytes_written %zu (want 0)", name, written);
    return 0;
}

/* A frame that must decode, host-only, to exactly `expected`. */
int ExpectDecode(const char* name, const Bytes& frame, const Bytes& expected) {
    Bytes out(expected.size() + 16, 0xEE); /* padding proves we stop at size */
    size_t written = 777;
    const cudec_status s = DecodeTight(frame, out.data(), out.size(), &written);
    REQUIRE_CTX(s == CUDEC_OK, "%s: status %d (want OK)", name,
                static_cast<int>(s));
    REQUIRE_CTX(written == expected.size(), "%s: size %zu (want %zu)", name,
                written, expected.size());
    REQUIRE_CTX(equal_bytes(out.data(), expected.data(), written), "%s bytes",
                name);
    return 0;
}

}  // namespace

int main() {
    const Bytes b8 = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'};
    const Bytes b3 = {'x', 'y', 'z'};

    /* --- Well-formed all-uncompressed frames: the host assembly + capacity
     * path, exercised under the sanitizer with no GPU call. --- */
    REQUIRE(ExpectDecode("empty", BuildFrame(0x60, 0x40, {}), {}) == 0);
    REQUIRE(ExpectDecode("one-uncompressed", BuildFrame(0x60, 0x40, {b8}), b8) ==
            0);
    {
        Bytes concat = b8;
        concat.insert(concat.end(), b3.begin(), b3.end());
        REQUIRE(ExpectDecode("two-uncompressed",
                             BuildFrame(0x60, 0x40, {b8, b3}), concat) == 0);
    }
    /* Content and block checksums verified on the success side too. */
    REQUIRE(ExpectDecode("content-ck-ok", BuildFrame(0x64, 0x40, {b8}), b8) ==
            0);
    REQUIRE(ExpectDecode("block-ck-ok", BuildFrame(0x70, 0x40, {b8}), b8) == 0);

    /* A NULL destination with a zero capacity is valid for a zero-byte decode:
     * the assembly must never dereference it. */
    {
        const Bytes f = BuildFrame(0x60, 0x40, {});
        size_t written = 777;
        REQUIRE(DecodeTight(f, nullptr, 0, &written) == CUDEC_OK);
        REQUIRE(written == 0);
    }

    /* --- C-ABI argument validation: the extern "C" guard rejects before any
     * parse with nothing written. Each case pins CUDEC_ERR_INVALID_ARGUMENT
     * and, where bytes_written is observable, written == 0. --- */
    {
        const Bytes f = BuildFrame(0x60, 0x40, {b8});
        Bytes out(1u << 16, 0xCC);
        size_t written = 777;
        /* NULL frame. */
        REQUIRE(cudec_lz4f_decompress(nullptr, f.size(), out.data(),
                                      out.size(), &written) ==
                CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(written == 0);
        /* NULL bytes_written (no output parameter left to observe). */
        REQUIRE(cudec_lz4f_decompress(f.data(), f.size(), out.data(),
                                      out.size(), nullptr) ==
                CUDEC_ERR_INVALID_ARGUMENT);
        /* NULL destination with a non-zero capacity. */
        written = 777;
        REQUIRE(cudec_lz4f_decompress(f.data(), f.size(), nullptr, out.size(),
                                      &written) == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(written == 0);
    }

    /* --- Declared content size (FLG bit 3): the size-match branch driven in
     * both directions. --- */
    /* Valid: the declared size equals the produced size. */
    REQUIRE(ExpectDecode("content-size-ok",
                         BuildFrameContentSize(0x68, 0x40, {b8}, b8.size()),
                         b8) == 0);
    /* Declared too large. */
    REQUIRE(ExpectReject(
                "content-size-too-large",
                BuildFrameContentSize(0x68, 0x40, {b8}, b8.size() + 1),
                CUDEC_ERR_CORRUPT_INPUT) == 0);
    /* Declared too small. */
    REQUIRE(ExpectReject(
                "content-size-too-small",
                BuildFrameContentSize(0x68, 0x40, {b8}, b8.size() - 1),
                CUDEC_ERR_CORRUPT_INPUT) == 0);

    /* --- Header reject branches (all fire before the block-table walk). --- */

    /* Under the 7-byte minimum header: the magic must NOT be read (short
     * frames short-circuit), so ASan would catch an over-read here. */
    REQUIRE(ExpectReject("too-short-3", Bytes{0x04, 0x22, 0x4D},
                         CUDEC_ERR_CORRUPT_INPUT) == 0);
    REQUIRE(ExpectReject("too-short-6", Bytes{0x04, 0x22, 0x4D, 0x18, 0x60,
                                              0x40},
                         CUDEC_ERR_CORRUPT_INPUT) == 0);

    {
        Bytes f = BuildFrame(0x60, 0x40, {b8});
        f[0] ^= 0xFF;
        REQUIRE(ExpectReject("bad-magic", f, CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* Version field != 01 (FLG bits 7-6). */
    REQUIRE(ExpectReject("bad-version", BuildFrame(0x20, 0x40, {b8}),
                         CUDEC_ERR_CORRUPT_INPUT) == 0);
    /* Reserved FLG bit 1 set. */
    REQUIRE(ExpectReject("reserved-flg", BuildFrame(0x62, 0x40, {b8}),
                         CUDEC_ERR_CORRUPT_INPUT) == 0);
    /* Reserved BD bit (bit 0) set. */
    REQUIRE(ExpectReject("bd-reserved", BuildFrame(0x60, 0x41, {b8}),
                         CUDEC_ERR_CORRUPT_INPUT) == 0);
    /* Reserved BD bit 7 set (the high reserved bit; bmax field still 4). */
    REQUIRE(ExpectReject("bd-reserved-bit7", BuildFrame(0x60, 0xC0, {b8}),
                         CUDEC_ERR_CORRUPT_INPUT) == 0);
    /* Block-max code out of the 4..7 range. */
    REQUIRE(ExpectReject("bad-bmax", BuildFrame(0x60, 0x30, {b8}),
                         CUDEC_ERR_CORRUPT_INPUT) == 0);
    /* Block-linked mode and a dictionary id are valid-but-unsupported. */
    REQUIRE(ExpectReject("linked", BuildFrame(0x40, 0x40, {b8}),
                         CUDEC_ERR_UNSUPPORTED) == 0);
    REQUIRE(ExpectReject("dictid", BuildFrame(0x61, 0x40, {b8}),
                         CUDEC_ERR_UNSUPPORTED) == 0);
    /* Corrupt header checksum (the HC byte at offset 6). */
    {
        Bytes f = BuildFrame(0x60, 0x40, {b8});
        f[6] ^= 0xFF;
        REQUIRE(ExpectReject("bad-hc", f, CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* Content-size flag set but the 8-byte field is truncated. */
    {
        Bytes f;
        PutMagic(&f);
        f.push_back(0x68); /* FLG: version 01, independent, content-size bit 3 */
        f.push_back(0x40);
        f.push_back(0);
        f.push_back(0);
        f.push_back(0); /* frame_size 9 < 6 + 8 */
        REQUIRE(ExpectReject("content-size-truncated", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* Content-size field present but the frame ends before the HC byte. */
    {
        Bytes f;
        PutMagic(&f);
        f.push_back(0x68);
        f.push_back(0x40);
        for (int i = 0; i < 8; i++) {
            f.push_back(0); /* frame_size 14: past the size field, no HC */
        }
        REQUIRE(ExpectReject("content-size-no-hc", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }

    /* --- Block-table walk reject branches (host pointer arithmetic on the
     * hostile block sizes - the surface issue #42 gap 2 names). --- */

    /* No block-size field at all, and a partial one. */
    {
        Bytes f = BuildFrame(0x60, 0x40, {b8});
        f.resize(7);
        REQUIRE(ExpectReject("no-block-table", f, CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    {
        Bytes f = BuildFrame(0x60, 0x40, {b8});
        f.resize(9); /* header + 2 of the 4 block-size bytes */
        REQUIRE(ExpectReject("block-size-truncated", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* A non-end-mark block whose declared length is zero. */
    {
        Bytes f;
        PutMagic(&f);
        f.push_back(0x60);
        f.push_back(0x40);
        f.push_back(Hc(f, 2));
        Put32(&f, 0x80000000u); /* uncompressed, blen 0 (not the 0 end mark) */
        REQUIRE(ExpectReject("block-blen-zero", f, CUDEC_ERR_CORRUPT_INPUT) ==
                0);
    }
    /* A block whose declared length exceeds the block max. */
    {
        Bytes f;
        PutMagic(&f);
        f.push_back(0x60);
        f.push_back(0x40);
        f.push_back(Hc(f, 2));
        Put32(&f, 0x80000000u | 65537u); /* > 64 KiB block max */
        REQUIRE(ExpectReject("block-blen-over-max", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* A block whose declared length runs past the end of the frame. */
    {
        Bytes f;
        PutMagic(&f);
        f.push_back(0x60);
        f.push_back(0x40);
        f.push_back(Hc(f, 2));
        Put32(&f, 0x80000000u | 100u);
        for (int i = 0; i < 10; i++) {
            f.push_back(0xAB); /* only 10 of the 100 declared bytes */
        }
        REQUIRE(ExpectReject("block-blen-truncated", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* A block checksum that does not match its payload (payload corrupted after
     * the stored checksum was computed). */
    {
        Bytes f = BuildFrame(0x70, 0x40, {b8}); /* FLG bit 4: block checksum */
        f[11] ^= 0xFF; /* first payload byte: header 7 + block-size 4 */
        REQUIRE(ExpectReject("block-checksum-mismatch", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* A content checksum that does not match the assembled output. This one
     * runs the whole host assembly (n == 0, no GPU) and then the content-
     * checksum branch. */
    {
        Bytes f = BuildFrame(0x64, 0x40, {b8}); /* FLG bit 2: content checksum */
        f.back() ^= 0xFF;
        REQUIRE(ExpectReject("content-checksum-mismatch", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* Block-checksum flag set (FLG bit 4) but the 4-byte block checksum is
     * truncated: frame_size < pos + blen + 4, so the checksum read must not
     * happen. */
    {
        Bytes f;
        PutMagic(&f);
        f.push_back(0x70); /* FLG bit 4: block checksum */
        f.push_back(0x40);
        f.push_back(Hc(f, 2));
        Put32(&f, 0x80000000u | 8u); /* uncompressed, 8-byte block */
        for (int i = 0; i < 8; i++) {
            f.push_back(0xAB); /* the 8 payload bytes */
        }
        f.push_back(0x00);
        f.push_back(0x00); /* only 2 of the 4 block-checksum bytes */
        REQUIRE(ExpectReject("block-checksum-truncated", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }
    /* Content-checksum flag set (FLG bit 2) but the 4-byte content checksum is
     * truncated: frame_size < pos + 4 after the end mark. */
    {
        Bytes f;
        PutMagic(&f);
        f.push_back(0x64); /* FLG bit 2: content checksum */
        f.push_back(0x40);
        f.push_back(Hc(f, 2));
        Put32(&f, 0x80000000u | 8u); /* uncompressed, 8-byte block */
        for (int i = 0; i < 8; i++) {
            f.push_back(0xAB);
        }
        Put32(&f, 0); /* end mark */
        f.push_back(0x00);
        f.push_back(0x00); /* only 2 of the 4 content-checksum bytes */
        REQUIRE(ExpectReject("content-checksum-truncated", f,
                             CUDEC_ERR_CORRUPT_INPUT) == 0);
    }

    /* An output buffer too small for an all-uncompressed frame: the host
     * assembly checks capacity before writing and rejects with nothing
     * written. */
    {
        const Bytes f = BuildFrame(0x60, 0x40, {b8});
        Bytes out(4, 0xEE);
        size_t written = 777;
        REQUIRE(DecodeTight(f, out.data(), out.size(), &written) ==
                CUDEC_ERR_OUTPUT_TOO_SMALL);
        REQUIRE(written == 0);
    }

    std::printf("PASS: frame header + block-table reject branches and the "
                "host assembly path exercised GPU-less; fail-closed on every "
                "malformed frame\n");
    return 0;
}
