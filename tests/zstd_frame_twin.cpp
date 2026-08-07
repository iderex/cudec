/* The Zstd frame and block header parser from src/zstd_frame.h, executed on
 * the host and held to libzstd (issue #200). Host-side and GPU-less on
 * purpose: the gate that decides which frames cudec will decode at all runs
 * on the runner with no device, and the sanitizer build reaches it there.
 *
 * TWO DIRECTIONS, WHICH IS THE WHOLE POINT OF THE LADDER. Every negative
 * below carries the class docs/MASTERPLAN.md section 12 assigns it, and the
 * oracle is asked about each one:
 *
 *   CORRUPT_INPUT rows: libzstd must also refuse the bytes. A row where the
 *   reference decodes and cudec calls the data corrupt is a fail-closed
 *   error in the other direction, and it would be invisible to a test that
 *   only checked cudec's own verdict.
 *
 *   UNSUPPORTED rows: libzstd must ACCEPT the header. That is what makes the
 *   refusal a scope decision rather than a rejection - the frame is legal,
 *   cudec declines it, and section 12.3 says a caller must be able to tell
 *   those apart because one of them is worth a CPU fallback.
 *
 * The frames are hand-built byte by byte, for the reason the Snappy and Zstd
 * probes give: a shape the compressor never emits is exactly the one a
 * hostile stream will carry. The compressor is driven once, for a real frame
 * the hand-built ones are checked against.
 *
 * ZSTD_getFrameHeader is the field-for-field oracle for the positive cases.
 * It is in libzstd's static API, hence ZSTD_STATIC_LINKING_ONLY in the
 * target's compile definitions, and it means this test grows no second
 * frame-header parser to check the first one against. */
#include "require.h"
#include "zstd_frame.h"

#include <zstd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* Which reject rungs a declared negative reached. Same discipline as the
 * bitstream twin: the enumeration lives once, in the header, and main()
 * requires every rung to have been reached by a negative written for it. */
bool g_reject_covered[cudec_detail::kZstdFrameRejectCount] = {false};

void CoverRung(cudec_detail::ZstdFrameReject rung) {
    if (rung != cudec_detail::kZstdFrameRejectNone) {
        g_reject_covered[rung] = true;
    }
}

void AppendLe(Bytes* out, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i++) {
        out->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xFFu));
    }
}

/* The frame magic, then whatever the caller wants after it. */
Bytes Magic(uint32_t magic) {
    Bytes out;
    AppendLe(&out, magic, 4);
    return out;
}

/* One block header in the 3-byte little-endian shape of RFC 8878 section
 * 3.1.1.2: Last_Block in bit 0, Block_Type in bits 1-2, Block_Size above. */
void AppendBlockHeader(Bytes* out, uint32_t size, uint8_t type, bool last) {
    const uint32_t raw = (size << 3) | (static_cast<uint32_t>(type) << 1) |
                         (last ? 1u : 0u);
    AppendLe(out, raw, 3);
}

/* Does the reference decode these bytes at all? The frames here are small,
 * so one fixed output buffer covers every row; a row that would need more is
 * one this test should not be building. */
bool OracleDecodes(const Bytes& frame, std::string* decoded) {
    std::vector<char> out(1 << 16);
    const size_t rc =
        ZSTD_decompress(out.data(), out.size(), frame.data(), frame.size());
    if (ZSTD_isError(rc)) {
        return false;
    }
    if (decoded != 0) {
        decoded->assign(out.data(), out.data() + rc);
    }
    return true;
}

/* Does the reference read the header as a legal one? Zero means it parsed
 * the whole header; a positive return means it wanted more bytes, which for
 * these fixtures is a defect in the fixture rather than a verdict. */
bool OracleHeaderIsLegal(const Bytes& frame, ZSTD_frameHeader* header) {
    return ZSTD_getFrameHeader(header, frame.data(), frame.size()) == 0;
}

struct Negative {
    const char* name;
    Bytes bytes;
    cudec_status expect_status;
    cudec_detail::ZstdFrameReject expect_rung;
    /* True where the bytes are a legal frame cudec declines, false where no
     * conforming encoder produced them. Drives which way the oracle is
     * asked. */
    bool legal_frame;
    /* Block-level rows carry a frame header the parser must accept first,
     * so they are driven through both calls rather than the first alone. */
    bool block_level;
};

int RunNegative(const Negative& row) {
    cudec_detail::ZstdFrameReject rung = cudec_detail::kZstdFrameRejectNone;
    cudec_detail::ZstdFrameHeader header;
    std::memset(&header, 0, sizeof(header));
    cudec_status status = cudec_detail::ZstdParseFrameHeader(
        row.bytes.data(), row.bytes.size(), &header, &rung);

    if (row.block_level) {
        /* The header of a block-level negative is in the subset, so a
         * refusal here means the fixture broke something it did not mean
         * to. */
        REQUIRE_CTX(status == CUDEC_OK, "%s: frame header refused with %d",
                    row.name, static_cast<int>(status));
        const uint64_t offset = header.header_size;
        REQUIRE_CTX(offset <= row.bytes.size(), "%s: header past the end",
                    row.name);
        cudec_detail::ZstdBlockHeader block;
        std::memset(&block, 0, sizeof(block));
        status = cudec_detail::ZstdParseBlockHeader(
            row.bytes.data() + offset, row.bytes.size() - offset,
            header.window_size, &block, &rung);
    }

    REQUIRE_CTX(status == row.expect_status, "%s: status %d, wanted %d",
                row.name, static_cast<int>(status),
                static_cast<int>(row.expect_status));
    REQUIRE_CTX(rung == row.expect_rung, "%s: rung %d, wanted %d", row.name,
                static_cast<int>(rung), static_cast<int>(row.expect_rung));
    CoverRung(rung);

    if (row.legal_frame) {
        ZSTD_frameHeader oracle;
        std::memset(&oracle, 0, sizeof(oracle));
        REQUIRE_CTX(OracleHeaderIsLegal(row.bytes, &oracle),
                    "%s: cudec calls this unsupported but libzstd does not "
                    "read it as a legal header - the refusal would be a "
                    "corrupt verdict wearing the wrong class",
                    row.name);
        REQUIRE_CTX(row.expect_status == CUDEC_ERR_UNSUPPORTED,
                    "%s: a legal frame must be refused as unsupported",
                    row.name);
    } else {
        REQUIRE_CTX(!OracleDecodes(row.bytes, 0),
                    "%s: libzstd decodes these bytes and cudec calls them "
                    "corrupt - reject parity runs in both directions",
                    row.name);
        REQUIRE_CTX(row.expect_status == CUDEC_ERR_CORRUPT_INPUT,
                    "%s: bytes no encoder produced must be refused as "
                    "corrupt",
                    row.name);
    }
    return 0;
}

}  // namespace

int main() {
    /* ---- Positives, held to the reference field for field ---------------
     *
     * A single-segment frame with a one-byte content size and one raw
     * block, the smallest thing the subset accepts. Section 3.1.1.1.1.1:
     * Frame_Content_Size_flag 0 with Single_Segment set is a one-byte
     * field. */
    {
        Bytes frame = Magic(cudec_detail::kZstdMagic);
        frame.push_back(0x20); /* Single_Segment, FCS flag 0, no dict, no
                                  checksum, reserved clear */
        frame.push_back(0x03); /* Frame_Content_Size = 3 */
        AppendBlockHeader(&frame, 3, cudec_detail::kZstdBlockTypeRaw, true);
        frame.push_back('a');
        frame.push_back('b');
        frame.push_back('c');

        cudec_detail::ZstdFrameHeader header;
        std::memset(&header, 0, sizeof(header));
        cudec_detail::ZstdFrameReject rung =
            cudec_detail::kZstdFrameRejectNone;
        REQUIRE(cudec_detail::ZstdParseFrameHeader(
                    frame.data(), frame.size(), &header, &rung) == CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdFrameRejectNone);

        ZSTD_frameHeader oracle;
        std::memset(&oracle, 0, sizeof(oracle));
        REQUIRE(OracleHeaderIsLegal(frame, &oracle));
        REQUIRE(header.header_size == oracle.headerSize);
        REQUIRE(header.frame_content_size == oracle.frameContentSize);
        REQUIRE(header.window_size == oracle.windowSize);
        REQUIRE(header.content_checksum == (oracle.checksumFlag != 0));
        REQUIRE(header.single_segment);

        cudec_detail::ZstdBlockHeader block;
        std::memset(&block, 0, sizeof(block));
        REQUIRE(cudec_detail::ZstdParseBlockHeader(
                    frame.data() + header.header_size,
                    frame.size() - header.header_size, header.window_size,
                    &block, &rung) == CUDEC_OK);
        REQUIRE(block.block_type == cudec_detail::kZstdBlockTypeRaw);
        REQUIRE(block.block_size == 3);
        REQUIRE(block.body_size == 3);
        REQUIRE(block.last_block);

        std::string decoded;
        REQUIRE(OracleDecodes(frame, &decoded));
        REQUIRE(decoded == "abc");
    }

    /* An RLE block, which is the one place the declared size and the bytes
     * on the wire differ: section 3.1.1.2.3 makes Block_Size the REGENERATED
     * length and the body a single byte. A walker that stepped by the
     * declared size here would step 1000 bytes past a 1-byte body, so the
     * distinction is asserted against a frame the reference agrees decodes
     * to 1000 bytes. */
    {
        const uint32_t regenerated = 1000;
        Bytes frame = Magic(cudec_detail::kZstdMagic);
        frame.push_back(0x60); /* Single_Segment, FCS flag 1 -> 2 bytes */
        AppendLe(&frame, regenerated - 256, 2);
        AppendBlockHeader(&frame, regenerated,
                          cudec_detail::kZstdBlockTypeRle, true);
        frame.push_back('A');

        cudec_detail::ZstdFrameHeader header;
        std::memset(&header, 0, sizeof(header));
        cudec_detail::ZstdFrameReject rung =
            cudec_detail::kZstdFrameRejectNone;
        REQUIRE(cudec_detail::ZstdParseFrameHeader(
                    frame.data(), frame.size(), &header, &rung) == CUDEC_OK);
        REQUIRE(header.frame_content_size == regenerated);
        REQUIRE(header.window_size == regenerated);

        ZSTD_frameHeader oracle;
        std::memset(&oracle, 0, sizeof(oracle));
        REQUIRE(OracleHeaderIsLegal(frame, &oracle));
        REQUIRE(header.frame_content_size == oracle.frameContentSize);
        REQUIRE(header.window_size == oracle.windowSize);
        REQUIRE(header.header_size == oracle.headerSize);

        cudec_detail::ZstdBlockHeader block;
        std::memset(&block, 0, sizeof(block));
        REQUIRE(cudec_detail::ZstdParseBlockHeader(
                    frame.data() + header.header_size,
                    frame.size() - header.header_size, header.window_size,
                    &block, &rung) == CUDEC_OK);
        REQUIRE(block.block_size == regenerated);
        REQUIRE(block.body_size == 1);

        std::string decoded;
        REQUIRE(OracleDecodes(frame, &decoded));
        REQUIRE(decoded.size() == regenerated);
        REQUIRE(decoded[0] == 'A');
        REQUIRE(decoded[regenerated - 1] == 'A');
    }

    /* A frame the compressor actually produced, so the hand-built shapes
     * above are not the only ones the parser has ever seen. Default
     * settings: content size present, no dictionary, no checksum. */
    {
        const std::string source(4096, 'z');
        Bytes frame(ZSTD_compressBound(source.size()));
        const size_t written = ZSTD_compress(frame.data(), frame.size(),
                                             source.data(), source.size(), 3);
        REQUIRE(!ZSTD_isError(written));
        frame.resize(written);

        cudec_detail::ZstdFrameHeader header;
        std::memset(&header, 0, sizeof(header));
        cudec_detail::ZstdFrameReject rung =
            cudec_detail::kZstdFrameRejectNone;
        REQUIRE(cudec_detail::ZstdParseFrameHeader(
                    frame.data(), frame.size(), &header, &rung) == CUDEC_OK);

        ZSTD_frameHeader oracle;
        std::memset(&oracle, 0, sizeof(oracle));
        REQUIRE(OracleHeaderIsLegal(frame, &oracle));
        REQUIRE(header.header_size == oracle.headerSize);
        REQUIRE(header.frame_content_size == oracle.frameContentSize);
        REQUIRE(header.window_size == oracle.windowSize);
        REQUIRE(header.content_checksum == (oracle.checksumFlag != 0));
        REQUIRE(header.frame_content_size == source.size());

        /* Block_Maximum_Size, section 3.1.1.2.4, is what the parser holds a
         * declared block size to; the reference computes the same number and
         * publishes it, so the bound is checked rather than assumed. */
        const uint64_t block_max =
            header.window_size < cudec_detail::kZstdBlockSizeCeiling
                ? header.window_size
                : cudec_detail::kZstdBlockSizeCeiling;
        REQUIRE(block_max == oracle.blockSizeMax);

        /* Walk the block chain to the end of the frame. Nothing here decodes
         * anything; it is the stepping arithmetic that has to be right, and
         * a wrong body_size for any block would land the last step somewhere
         * other than the frame's end. */
        uint64_t offset = header.header_size;
        unsigned blocks = 0;
        bool reached_last = false;
        /* Bounded by the frame's own length: a block costs at least its
         * three header bytes, so no frame can hold more blocks than it has
         * bytes. The walk terminates on the count even if the last-block
         * flag never arrives. */
        for (uint64_t step = 0; step < frame.size() && !reached_last; step++) {
            cudec_detail::ZstdBlockHeader block;
            std::memset(&block, 0, sizeof(block));
            REQUIRE(cudec_detail::ZstdParseBlockHeader(
                        frame.data() + offset, frame.size() - offset,
                        header.window_size, &block, &rung) == CUDEC_OK);
            offset += 3u + block.body_size;
            blocks++;
            reached_last = block.last_block;
        }
        REQUIRE(reached_last);
        REQUIRE(blocks >= 1);
        REQUIRE(offset == frame.size());
    }

    /* ---- The negative ladder, one row per rung -------------------------
     *
     * The rows are the table in docs/MASTERPLAN.md section 12.2 turned into
     * bytes: every property that table names, in the class it assigns. */
    std::vector<Negative> rows;

    /* Magic number, refused as corrupt. Three bytes cannot even carry it. */
    {
        Bytes bytes = {0x28, 0xB5, 0x2F};
        rows.push_back({"magic truncated", bytes, CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectMagicTruncated, false,
                        false});
    }
    {
        Bytes bytes = {0xDE, 0xAD, 0xBE, 0xEF, 0x20, 0x03};
        rows.push_back({"magic wrong", bytes, CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectMagicWrong, false,
                        false});
    }
    /* Skippable frames are legal and refused as a scope line, section 12.4.
     * The frame is complete: the magic and a four-byte Frame_Size of zero. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdSkippableMagicMin);
        AppendLe(&bytes, 0, 4);
        rows.push_back({"skippable frame", bytes, CUDEC_ERR_UNSUPPORTED,
                        cudec_detail::kZstdFrameRejectSkippableFrame, true,
                        false});
    }
    /* The magic alone: the descriptor byte the layout needs next is not
     * there. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        rows.push_back({"descriptor truncated", bytes,
                        CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectDescriptorTruncated,
                        false, false});
    }
    /* Section 3.1.1.1.1.3: the reserved bit must be zero. Bit 3, one below
     * the Unused_bit a decoder of this version must NOT interpret. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x28); /* Single_Segment + Reserved_bit */
        bytes.push_back(0x03);
        rows.push_back({"reserved bit set", bytes, CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectReservedBitSet, false,
                        false});
    }
    /* A dictionary id is a legal frame outside the subset: dictionaries
     * break the frame independence the batch unit rests on (section 12.5). */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x21); /* Single_Segment, Dictionary_ID_flag 1 */
        bytes.push_back(0x07); /* the one-byte dictionary id */
        bytes.push_back(0x03); /* Frame_Content_Size */
        rows.push_back({"dictionary id", bytes, CUDEC_ERR_UNSUPPORTED,
                        cudec_detail::kZstdFrameRejectDictionaryId, true,
                        false});
    }
    /* No declared content size, which is legal and which the batch model
     * cannot place output for (section 12.5, rung 2). */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x00); /* FCS flag 0, Single_Segment clear */
        bytes.push_back(0x00); /* Window_Descriptor: the smallest window */
        rows.push_back({"content size absent", bytes, CUDEC_ERR_UNSUPPORTED,
                        cudec_detail::kZstdFrameRejectContentSizeAbsent, true,
                        false});
    }
    /* A header that declares an eight-byte content size and stops short of
     * it. The length is derived from the descriptor and checked before any
     * dependent field is read, which is the rung this reaches. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0xE0); /* FCS flag 3 -> 8 bytes, Single_Segment */
        AppendLe(&bytes, 0, 3);
        rows.push_back({"header truncated", bytes, CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectHeaderTruncated, false,
                        false});
    }
    /* Section 3.1.1.1.2 allows refusing a window beyond the decoder's
     * authorized range, and section 12.2 puts that line at 8 MB. Exponent 13
     * with mantissa 1 is 8 MB plus an eighth, the smallest step over it that
     * the encoding can express at that exponent. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x40); /* FCS flag 1 -> 2 bytes, Single_Segment
                                  clear */
        bytes.push_back(static_cast<unsigned char>((13u << 3) | 1u));
        AppendLe(&bytes, 0, 2); /* content size 256 */
        rows.push_back({"window too large", bytes, CUDEC_ERR_UNSUPPORTED,
                        cudec_detail::kZstdFrameRejectWindowTooLarge, true,
                        false});
    }
    /* Block-level rows. Each carries a header the subset accepts, so the
     * refusal is the block's and the frame header's acceptance is proven on
     * the way past it. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x20);
        bytes.push_back(0x03);
        bytes.push_back(0x19); /* two bytes of a three-byte block header */
        bytes.push_back(0x00);
        rows.push_back({"block header truncated", bytes,
                        CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectBlockHeaderTruncated,
                        false, true});
    }
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x20);
        bytes.push_back(0x03);
        AppendBlockHeader(&bytes, 3, cudec_detail::kZstdBlockTypeReserved,
                          true);
        AppendLe(&bytes, 0, 3);
        rows.push_back({"block type reserved", bytes,
                        CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectBlockTypeReserved,
                        false, true});
    }
    /* Block_Maximum_Size is min(Window_Size, 128 KB), and here the window is
     * the 1024-byte smallest one, so 2000 is over the line without needing a
     * 128 KB fixture to prove it. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x40); /* Single_Segment clear, FCS 2 bytes */
        bytes.push_back(0x00); /* Window_Descriptor 0 -> 1024 */
        AppendLe(&bytes, 0, 2);
        AppendBlockHeader(&bytes, 2000, cudec_detail::kZstdBlockTypeRaw,
                          true);
        bytes.resize(bytes.size() + 2000, 0);
        rows.push_back({"block over the maximum", bytes,
                        CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectBlockTooLarge, false,
                        true});
    }
    /* A block that declares more bytes than the frame holds. The declared
     * size is inside the maximum, so this is the presence check rather than
     * the bound check, and the two are separate rungs for that reason. */
    {
        Bytes bytes = Magic(cudec_detail::kZstdMagic);
        bytes.push_back(0x20);
        bytes.push_back(0x64); /* content size 100 */
        AppendBlockHeader(&bytes, 100, cudec_detail::kZstdBlockTypeRaw, true);
        bytes.resize(bytes.size() + 10, 0);
        rows.push_back({"block body truncated", bytes,
                        CUDEC_ERR_CORRUPT_INPUT,
                        cudec_detail::kZstdFrameRejectBlockBodyTruncated,
                        false, true});
    }

    for (size_t i = 0; i < rows.size(); i++) {
        const int rc = RunNegative(rows[i]);
        if (rc != 0) {
            return rc;
        }
    }

    /* Every rung named by a negative written to reach it, walked rather than
     * restated: a rung added to the parser with none behind it lands here as
     * a hole with its number on it. */
    {
        size_t rungs_covered = 0;
        for (int rung = cudec_detail::kZstdFrameRejectNone + 1;
             rung < cudec_detail::kZstdFrameRejectCount; rung++) {
            REQUIRE_CTX(g_reject_covered[rung],
                        "reject rung %d has no declared negative that reaches "
                        "it - add one, or the rung is untested",
                        rung);
            rungs_covered++;
        }
        REQUIRE(rungs_covered ==
                static_cast<size_t>(cudec_detail::kZstdFrameRejectCount) - 1);
    }

    std::printf("PASS: 3 accepted frames checked field for field against "
                "ZSTD_getFrameHeader, %zu negatives over %d reject rungs, "
                "each held to libzstd in the direction its class implies\n",
                rows.size(),
                static_cast<int>(cudec_detail::kZstdFrameRejectCount) - 1);
    return 0;
}
