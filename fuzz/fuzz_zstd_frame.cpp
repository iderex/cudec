/* Differential fuzz target over the Zstd frame and block header walk (issue
 * #180), the fourth target in fuzz/ and the first one over the M5 surface.
 *
 * THIS IS THE ONE ZSTD SURFACE RAW BYTES ACTUALLY REACH, which is why it needs
 * no structure-aware layer while the entropy targets do. Everything below the
 * envelope sits behind a four-byte magic number; the envelope itself is what a
 * fuzzer can hit from nothing, and it is also the gate that decides which
 * frames cudec will attempt at all - so a fail-open here admits a stream to
 * every later stage rather than only to this one.
 *
 * TWO DIRECTIONS, AND ONLY ONE OF THEM IS ASSERTED. Whenever the twin ACCEPTS,
 * libzstd must have parsed the same header and agreed field for field. The
 * other direction is not asserted, and the reason is not caution: cudec
 * decodes a declared SUBSET of Zstd (docs/MASTERPLAN.md section 12), so a
 * frame with a window above the supported bound, a dictionary id, an absent
 * content size, or a skippable magic is refused here and accepted by libzstd
 * on purpose. Trapping on those would report a scope decision as a defect.
 *
 * WHAT UNSUPPORTED MEANS HERE AND WHY IT IS CHECKED SEPARATELY. Section 12.3
 * puts those refusals in their own status precisely so a caller can tell "this
 * is not a legal frame" from "this is a legal frame I decline", because only
 * the second is worth a CPU fallback. So the target asserts that the twin
 * never answers UNSUPPORTED about bytes the reference calls CORRUPT: a scope
 * refusal over corrupt input sends a caller down a fallback that cannot work
 * either, and it is the one confusion between the two classes that a verdict
 * comparison can see. Which of the reference's errors count as corrupt is
 * narrower than "it returned an error" and is argued where the set is built.
 *
 * THE ENVELOPE WALK IS COMPARED AGAINST A REFERENCE THAT ALSO ONLY WALKS.
 * ZSTD_findFrameCompressedSize reads the frame's block headers and reports
 * where the frame ends without decompressing a byte, which is exactly what the
 * twin's walk does. So on a frame both accept, the two lengths must agree -
 * and a divergence there is a block-envelope defect that no header-field
 * comparison would have shown.
 */
#include "cudec.h"
#include "zstd_frame.h"

/* ZSTD_STATIC_LINKING_ONLY is defined by the build rather than here, which is
 * how tests/zstd_frame_twin.cpp reaches the same symbol. Defining it in both
 * places is a redefinition the strict-warning build refuses, and it refused
 * this file on its first CI run. */
#include <zstd.h>
#include <zstd_errors.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

using cudec_detail::kZstdBlockTypeReserved;
using cudec_detail::kZstdBlockTypeRle;
using cudec_detail::ZstdBlockHeader;
using cudec_detail::ZstdFrameHeader;
using cudec_detail::ZstdFrameReject;
using cudec_detail::ZstdParseBlockHeader;
using cudec_detail::ZstdParseFrameHeader;

/* Bounded so libFuzzer explores the envelope rather than the allocator. A
 * frame's blocks are at most 128 KiB each, but nothing here decodes a body:
 * the walk only steps over them, so a long input buys coverage of one loop. */
constexpr size_t kMaxStream = 1u << 14;

/* A frame cannot hold more blocks than it has room for three-byte headers, so
 * this cap is unreachable on any input the target is given. It exists because
 * a walk whose termination depends on the bytes it is walking is exactly the
 * shape that hangs a fuzzer instead of failing it. */
constexpr uint64_t kMaxBlocks = kMaxStream;

void Trap(const char* what, size_t size) {
    std::fprintf(stderr, "DIVERGENCE: %s; stream=%zu\n", what, size);
    __builtin_trap();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    size_t stream_size = size;
    if (stream_size > kMaxStream) {
        stream_size = kMaxStream;
    }

    /* libFuzzer hands out a slice of a buffer sized to -max_len rather than to
     * this input, so a read past the stream would land in that slack and stay
     * green. Both sides run over one exactly-sized copy instead. */
    auto stream = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(stream.get(), data, stream_size);
    }

    ZstdFrameHeader header;
    std::memset(&header, 0, sizeof(header));
    ZstdFrameReject reject = cudec_detail::kZstdFrameRejectNone;
    const cudec_status twin =
        ZstdParseFrameHeader(stream.get(), stream_size, &header, &reject);

    /* The reference's own verdict on the same bytes. Zero means a complete
     * header was parsed; a positive return means it wanted more bytes, which
     * is a truncation rather than an acceptance; an error code is a refusal. */
    ZSTD_frameHeader zfh;
    std::memset(&zfh, 0, sizeof(zfh));
    const size_t rc = ZSTD_getFrameHeader(&zfh, stream.get(), stream_size);
    const bool oracle_ok = (rc == 0);

    /* WHICH OF THE REFERENCE'S ERRORS MEAN "THESE BYTES ARE NOT A FRAME", AND
     * WHY THE DISTINCTION IS NOT PEDANTIC. libzstd reports refusals of two
     * different kinds through one error channel. prefix_unknown and
     * corruption_detected say the bytes are wrong. frameParameter_unsupported
     * and frameParameter_windowTooLarge say the bytes are a legal frame this
     * build will not decode - which is the same thing cudec's UNSUPPORTED
     * says, so treating them as corruption would make the check below fire on
     * exactly the cases where the two implementations AGREE.
     *
     * Measured rather than assumed: a frame whose window exponent is raised
     * past the supported bound comes back as error 16
     * (frameParameter_windowTooLarge), and cudec answers UNSUPPORTED about the
     * same bytes. An unnarrowed check traps there, which was the first thing
     * this target did. */
    const ZSTD_ErrorCode oracle_error =
        ZSTD_isError(rc) ? ZSTD_getErrorCode(rc) : ZSTD_error_no_error;
    const bool oracle_says_corrupt =
        oracle_error == ZSTD_error_prefix_unknown ||
        oracle_error == ZSTD_error_corruption_detected;

    if (twin != CUDEC_OK && twin != CUDEC_ERR_CORRUPT_INPUT &&
        twin != CUDEC_ERR_UNSUPPORTED) {
        Trap("a frame-header verdict outside the documented set", stream_size);
    }

    /* The class confusion, in the direction a verdict comparison can see. An
     * UNSUPPORTED answer claims the bytes are a legal frame this build
     * declines; if the reference refuses them outright, that claim is wrong
     * and it points a caller at a fallback that cannot work either. */
    if (twin == CUDEC_ERR_UNSUPPORTED && oracle_says_corrupt) {
        Trap("scope refusal (UNSUPPORTED) over bytes libzstd refuses outright",
             stream_size);
    }

    if (twin != CUDEC_OK) {
        return 0; /* the stricter direction is the declared subset */
    }

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to prove the comparisons below are live
     * without waiting for a real divergence: a second binary built with this
     * defined perturbs the accepted reference header, so a harness that had
     * silently stopped comparing passes where this one traps. Never define it
     * in a build whose findings are being believed. */
    zfh.windowSize ^= 1u;
#endif

    if (!oracle_ok) {
        Trap("FAIL-OPEN: the twin parsed a frame header libzstd did not",
             stream_size);
    }

    /* Field for field on the accepted header. A verdict that agrees while a
     * field does not is the shape that produces a wrong bound rather than a
     * wrong answer, and the window is the field the whole back-reference
     * bound rests on. */
    if (header.window_size != zfh.windowSize) {
        std::fprintf(stderr, "twin window=%llu oracle window=%llu\n",
                     static_cast<unsigned long long>(header.window_size),
                     static_cast<unsigned long long>(zfh.windowSize));
        Trap("window size divergence on an accepted header", stream_size);
    }
    if (header.frame_content_size != zfh.frameContentSize) {
        Trap("content size divergence on an accepted header", stream_size);
    }
    if (header.content_checksum != (zfh.checksumFlag != 0)) {
        Trap("checksum flag divergence on an accepted header", stream_size);
    }
    if (header.header_size != zfh.headerSize) {
        Trap("header size divergence on an accepted header", stream_size);
    }
    /* The single-segment flag has no field of its own in the reference's
     * report, so this is the twin against itself rather than against libzstd:
     * a single-segment frame carries no window descriptor and its window IS
     * its content size, so the two must agree or the flag was read from the
     * wrong bit. Stated as the weaker check it is, next to the ones that are
     * parity. */
    if (header.single_segment &&
        header.window_size != header.frame_content_size) {
        Trap("a single-segment frame whose window is not its content size",
             stream_size);
    }

    /* The block-envelope walk. Nothing is decoded: each header is parsed, its
     * body stepped over, and the walk stops at the last block or at the first
     * refusal. */
    uint64_t cursor = header.header_size;
    uint64_t blocks = 0;
    bool reached_last = false;
    while (cursor <= stream_size && blocks < kMaxBlocks) {
        ZstdBlockHeader block;
        std::memset(&block, 0, sizeof(block));
        const cudec_status st =
            ZstdParseBlockHeader(stream.get() + cursor, stream_size - cursor,
                                 header.window_size, &block, &reject);
        if (st != CUDEC_OK) {
            if (st != CUDEC_ERR_CORRUPT_INPUT && st != CUDEC_ERR_UNSUPPORTED) {
                Trap("a block-header verdict outside the documented set",
                     stream_size);
            }
            break;
        }
        ++blocks;

        /* THE INVARIANT OVER AN ACCEPTED BLOCK HEADER, ASSERTED HERE RATHER
         * THAN LEFT TO THE ENVELOPE COMPARISON BELOW. That comparison only
         * runs when the walk reaches a last block, so a fail-open that lets a
         * malformed block through and then stalls the walk would never reach
         * it - which is exactly what the relaxation sweep found: removing the
         * reserved-block-type refusal changed nothing a parity check could
         * see. These four are properties of the block header alone and they
         * hold whatever the rest of the frame does. */
        if (block.block_type == kZstdBlockTypeReserved) {
            Trap("an accepted block header carrying the reserved block type",
                 stream_size);
        }
        const uint64_t block_max =
            header.window_size < cudec_detail::kZstdBlockSizeCeiling
                ? header.window_size
                : cudec_detail::kZstdBlockSizeCeiling;
        if (block.block_size > block_max) {
            Trap("an accepted block above the frame's own block maximum",
                 stream_size);
        }
        const uint64_t span = 3u + block.body_size;
        if (span > stream_size - cursor) {
            Trap("an accepted block whose body runs past the frame",
                 stream_size);
        }
        if (block.block_type == kZstdBlockTypeRle) {
            if (block.body_size != 1) {
                Trap("an RLE block whose body is not one byte", stream_size);
            }
        } else if (block.body_size != block.block_size) {
            Trap("a non-RLE block whose body is not its declared size",
                 stream_size);
        }
        cursor += span;
        if (block.last_block) {
            reached_last = true;
            break;
        }
    }
    if (blocks >= kMaxBlocks) {
        Trap("the block walk did not terminate inside the input's own bound",
             stream_size);
    }
    if (!reached_last) {
        return 0; /* a frame that ends before its last block says nothing more */
    }

    /* Both sides walked the same envelope, so both must say the frame ends in
     * the same place. The reference reads block headers and steps over bodies
     * exactly as the loop above does - it decompresses nothing - so this is a
     * comparison of two walks and not of two decoders. */
    const uint64_t frame_end = cursor + (header.content_checksum ? 4u : 0u);
    if (frame_end > stream_size) {
        return 0; /* the trailer is not present; the twin's checksum rung owns
                   * that and it is not this walk's claim */
    }
    const size_t oracle_frame_size =
        ZSTD_findFrameCompressedSize(stream.get(), stream_size);
    if (ZSTD_isError(oracle_frame_size)) {
        Trap("FAIL-OPEN: the twin walked a whole frame envelope libzstd "
             "refuses",
             stream_size);
    }
    if (oracle_frame_size != frame_end) {
        std::fprintf(stderr, "twin end=%llu oracle end=%llu\n",
                     static_cast<unsigned long long>(frame_end),
                     static_cast<unsigned long long>(oracle_frame_size));
        Trap("the two envelope walks disagree about where the frame ends",
             stream_size);
    }
    return 0;
}
