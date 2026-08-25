/* Differential fuzz target over the LZ4 frame envelope (issue #141), the
 * second LZ4 target here and the container above the one fuzz_lz4_block
 * already drives.
 *
 * WHAT IT IS OVER, AND WHY THAT IS THE WHOLE FRAME SURFACE A FUZZER OWNS.
 * The block parser has its own target and its own reference. What no block
 * target can see is the layer that decides which blocks exist at all: the
 * descriptor (FLG/BD), the optional declared content size, the header
 * checksum, the block-size envelope, the per-block and content xxHash32
 * fields, and the block-table walk to the end mark. A fail-open there hands
 * the device decoder a block table derived from bytes nobody validated, so it
 * is the rung a hostile .lz4 file reaches first.
 *
 * THE HOST-ONLY CHOICE THE ISSUE ASKS FOR, AND WHY IT IS THE ENVELOPE ONE.
 * fuzz/ links no cudec archive and builds with Clang on a runner with no CUDA
 * at all, so the compressed-block decode - which is a device batch in the
 * shipped path (src/frame.cpp) - is not reachable from here. Of the two
 * options the issue names, this target takes the envelope: the byte
 * comparison runs on frames whose blocks are ALL stored uncompressed, which is
 * exactly the set src/frame.cpp finishes without issuing a CUDA call, and a
 * frame carrying a compressed block gets the envelope-level assertions and no
 * output comparison. Nothing is lost by that split, because the compressed
 * half is what fuzz_lz4_block is for: the two targets meet at the block
 * boundary rather than overlapping across it. What it costs is stated rather
 * than implied - no fuzz coverage of the assembly and capacity arithmetic that
 * runs after a device decode, which is `tests/frame_host_negative.cpp`'s
 * OUTPUT_TOO_SMALL rungs and `tests/frame_twin.cu`'s job.
 *
 * BOTH DIRECTIONS ARE ASSERTED HERE, WHICH IS UNUSUAL IN THIS DIRECTORY.
 * The fail-open direction is the primary one: where the twin accepts an
 * all-uncompressed frame, liblz4's own frame API must accept the same bytes
 * and produce the identical output and size. The other direction is
 * assertable too, and only because the twin and the reference are given the
 * same question here - both walk one .lz4 frame from its magic number - so a
 * frame liblz4 decodes and cudec calls CORRUPT_INPUT is cudec refusing a valid
 * container rather than cudec being carefully strict. Two strictnesses are
 * declared and identified rather than guessed at, and each is pinned by a
 * negative test in the tree:
 *
 *   - A skippable frame (magic 0x184D2A50..5F). liblz4 skips it and reports a
 *     complete frame; cudec decodes the one frame type its ABI documents.
 *     Pinned as `skippable-frame`. That cudec answers CORRUPT_INPUT about it
 *     where src/zstd_frame.h answers UNSUPPORTED about the identical magic
 *     range is a scope question rather than a defect in this walk, and it is
 *     issue #379.
 *   - A block header whose 31-bit length masks to zero with the uncompressed
 *     bit set. liblz4 masks first and reads it as the end mark; cudec reads it
 *     as a zero-length data block and refuses. Pinned as `block-blen-zero`.
 *
 * Both pins are rows in tests/frame_host_negative.cpp, so an exemption this
 * target grants is covered by the non-fuzz gate on every pull request rather
 * than living only in the sentence above.
 *
 * WHAT IT DOES NOT CHECK, SAID PLAINLY. The class confusion fuzz_zstd_frame
 * asserts - a scope refusal (UNSUPPORTED) over bytes the reference refuses
 * outright - is NOT asserted here. cudec reads the FLG bits that decide the
 * declared subset before it verifies the header checksum, so a linked-block
 * frame with a corrupt checksum is UNSUPPORTED here and corrupt to liblz4, and
 * asserting the property would report that ordering as a defect on almost
 * every input. What is asserted instead is the weaker twin-against-itself
 * form: an UNSUPPORTED verdict must be about a frame that really does declare
 * linked blocks or a dictionary id, so the status cannot leak out of any other
 * branch. */
#include "cudec.h"
#include "lz4_frame.h"

#include "lz4frame.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using cudec_detail::Lz4FrameBlock;
using cudec_detail::Lz4FrameDescriptor;

/* Bounded so libFuzzer explores the envelope rather than the allocator. The
 * block table is what a long input buys here, and a frame's worth of table is
 * reached long before this ceiling. */
constexpr size_t kMaxStream = 1u << 14;

/* The reference's output ceiling. A block may declare up to the descriptor's
 * block maximum, and an LZ4 block expands by at most roughly 255x, so a stream
 * bounded above cannot ask for more than this. It is the REFERENCE's buffer,
 * not the code under test's, so it is a loose one-time allocation rather than
 * a tight per-run one; the twin's buffer below is tight for the usual reason. */
constexpr size_t kMaxOracleOut = 8u << 20;

/* The skippable-frame magic family: the low nibble is the frame's own index
 * and any of the sixteen spellings is a skippable frame to liblz4. */
constexpr uint32_t kLz4SkippableMagicBase = 0x184D2A50u;
constexpr uint32_t kLz4SkippableMagicMask = 0xFFFFFFF0u;

void Trap(const char* what, size_t size) {
    std::fprintf(stderr, "DIVERGENCE: %s; stream=%zu\n", what, size);
    __builtin_trap();
}

enum class OracleVerdict {
    kAccepted,  /* a whole frame was decoded */
    kRefused,   /* the reference called the bytes an error */
    kIncomplete /* the reference wanted more input, or more room */
};

/* One pass of liblz4's frame API over the same bytes. `produced` is only
 * meaningful on kAccepted. */
OracleVerdict OracleDecodeFrame(const unsigned char* stream, size_t stream_size,
                                unsigned char* out, size_t out_capacity,
                                size_t* produced) {
    *produced = 0;
    if (stream_size == 0) {
        return OracleVerdict::kRefused; /* never hand liblz4 a null source */
    }
    LZ4F_dctx* dctx = nullptr;
    if (LZ4F_isError(LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION))) {
        return OracleVerdict::kRefused;
    }
    OracleVerdict verdict = OracleVerdict::kIncomplete;
    size_t src_pos = 0;
    size_t dst_pos = 0;
    for (;;) {
        size_t src_taken = stream_size - src_pos;
        size_t dst_written = out_capacity - dst_pos;
        if (src_taken == 0) {
            break; /* the frame did not end inside the input */
        }
        const size_t hint =
            LZ4F_decompress(dctx, out + dst_pos, &dst_written, stream + src_pos,
                            &src_taken, nullptr);
        if (LZ4F_isError(hint)) {
            verdict = OracleVerdict::kRefused;
            break;
        }
        src_pos += src_taken;
        dst_pos += dst_written;
        if (hint == 0) {
            verdict = OracleVerdict::kAccepted;
            break;
        }
        /* A call that consumed nothing and wrote nothing cannot make progress
         * on the next one either - the output buffer is full. Bail rather than
         * spin: a loop whose termination depends on the bytes it is walking is
         * the shape that hangs a fuzzer instead of failing it. */
        if (src_taken == 0 && dst_written == 0) {
            break;
        }
    }
    LZ4F_freeDecompressionContext(dctx);
    *produced = dst_pos;
    return verdict;
}

/* The twin's whole verdict on an all-uncompressed frame: the envelope walk,
 * the concatenation src/frame.cpp performs for a stored block, and the tail
 * the frame declares about its own output. */
struct TwinResult {
    cudec_status status;
    bool has_compressed_block;
    /* Where the block walk stopped. Meaningful only once the descriptor was
     * accepted, which `walked` records; it is what identifies WHICH refusal a
     * CORRUPT_INPUT verdict was, without a second walk and without a scan. */
    bool walked;
    size_t stop_off;
    size_t size;
    std::unique_ptr<unsigned char[]> bytes; /* tight: exactly `size` */
};

TwinResult TwinDecodeFrame(const unsigned char* stream, size_t stream_size) {
    TwinResult result{CUDEC_OK, false, false, 0, 0, nullptr};

    Lz4FrameDescriptor desc;
    std::memset(&desc, 0, sizeof(desc));
    result.status =
        cudec_detail::Lz4ParseFrameDescriptor(stream, stream_size, &desc);
    if (result.status != CUDEC_OK) {
        return result;
    }

    std::vector<Lz4FrameBlock> blocks;
    size_t tail_off = 0;
    result.status = cudec_detail::Lz4WalkFrameBlocks(stream, stream_size, desc,
                                                     &blocks, &tail_off);
    result.walked = true;
    result.stop_off = tail_off;
    if (result.status != CUDEC_OK) {
        return result;
    }

    /* THE INVARIANTS OVER AN ACCEPTED TABLE, asserted here rather than left to
     * the output comparison below. That comparison only runs on a frame both
     * sides accept, so a table entry that runs past the frame on a stream the
     * reference refuses would never reach it - and a bad entry is a bound the
     * device decoder would then be handed. */
    if (tail_off > stream_size) {
        Trap("an accepted walk ended past the end of the frame", stream_size);
    }
    size_t previous_end = desc.body_off;
    for (const Lz4FrameBlock& b : blocks) {
        if (b.src_len == 0 || b.src_len > desc.block_max) {
            Trap("an accepted block outside the descriptor's block maximum",
                 stream_size);
        }
        if (b.src_off < previous_end || b.src_off > stream_size ||
            b.src_len > stream_size - b.src_off) {
            Trap("an accepted block that does not lie inside the frame",
                 stream_size);
        }
        previous_end = b.src_off + b.src_len;
        if (!b.uncompressed) {
            result.has_compressed_block = true;
        }
    }
    if (result.has_compressed_block) {
        return result; /* the device half; fuzz_lz4_block owns those bytes */
    }

    size_t total = 0;
    for (const Lz4FrameBlock& b : blocks) {
        total += b.src_len; /* bounded by the frame: the walk proved each span */
    }
    result.bytes = std::make_unique<unsigned char[]>(total == 0 ? 1 : total);
    size_t off = 0;
    for (const Lz4FrameBlock& b : blocks) {
        std::memcpy(result.bytes.get() + off, stream + b.src_off, b.src_len);
        off += b.src_len;
    }
    result.status = cudec_detail::Lz4VerifyFrameTail(
        stream, stream_size, desc, tail_off, result.bytes.get(), total);
    result.size = total;
    return result;
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
    auto stream = std::make_unique<unsigned char[]>(
        stream_size == 0 ? 1 : stream_size);
    if (stream_size != 0) {
        std::memcpy(stream.get(), data, stream_size);
    }

    const TwinResult twin = TwinDecodeFrame(stream.get(), stream_size);

    if (twin.status != CUDEC_OK && twin.status != CUDEC_ERR_CORRUPT_INPUT &&
        twin.status != CUDEC_ERR_UNSUPPORTED) {
        Trap("a frame verdict outside the documented set", stream_size);
    }

    /* An UNSUPPORTED answer claims the bytes are a legal frame this build
     * declines, so the two declared declines must be visible in the bytes it
     * answered about. The twin against itself, not against liblz4 - the
     * comment at the top says why the parity form is not asserted. */
    if (twin.status == CUDEC_ERR_UNSUPPORTED) {
        if (stream_size < cudec_detail::kLz4FrameMinHeaderBytes) {
            Trap("a scope refusal about bytes too short to hold a descriptor",
                 stream_size);
        }
        const unsigned flg = stream[4];
        const bool block_independent = (flg >> 5) & 1;
        const bool dict_id = flg & 1;
        if (block_independent && !dict_id) {
            Trap("a scope refusal about a frame declaring neither linked "
                 "blocks nor a dictionary id",
                 stream_size);
        }
    }

    static std::unique_ptr<unsigned char[]> oracle_out(
        new unsigned char[kMaxOracleOut]);
    size_t oracle_size = 0;
    const OracleVerdict oracle = OracleDecodeFrame(
        stream.get(), stream_size, oracle_out.get(), kMaxOracleOut,
        &oracle_size);

    /* THE STRICTER DIRECTION, AND THE TWO PLACES IT IS NOT A DEFECT. Both
     * sides were asked about one .lz4 frame starting at byte zero, so a frame
     * the reference decodes end to end and the twin calls corrupt is the twin
     * refusing a valid container. The two exemptions are identified from the
     * bytes rather than inferred from the verdict. */
    if (oracle == OracleVerdict::kAccepted &&
        twin.status == CUDEC_ERR_CORRUPT_INPUT) {
        bool declared_strictness = false;
        if (stream_size >= 4) {
            const uint32_t magic = cudec_detail::Lz4FrameRead32LE(stream.get());
            declared_strictness |=
                (magic & kLz4SkippableMagicMask) == kLz4SkippableMagicBase;
        }
        /* The masks-to-zero block header, read at the exact offset the walk
         * stopped at rather than searched for. A scan of the whole stream
         * would excuse any divergence whose bytes happened to contain the
         * pattern somewhere, which over millions of random inputs is most of
         * them - an exemption that wide is a check that has stopped running. */
        if (!declared_strictness && twin.walked &&
            twin.stop_off + 4 <= stream_size) {
            const uint32_t bs =
                cudec_detail::Lz4FrameRead32LE(stream.get() + twin.stop_off);
            declared_strictness = (bs != 0) && ((bs & 0x7FFFFFFFu) == 0);
        }
        if (!declared_strictness) {
            Trap("FAIL-CLOSED-TOO-FAR: the twin refused a frame liblz4 decoded",
                 stream_size);
        }
    }

    if (twin.status != CUDEC_OK || twin.has_compressed_block) {
        return 0; /* nothing left that this target may compare */
    }

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to prove the comparison below is live
     * without waiting for a real divergence: a second binary built with this
     * defined perturbs the accepted reference result, so a harness that had
     * silently stopped comparing passes where this one traps. Never define it
     * in a build whose findings are being believed. */
    if (oracle_size != 0) {
        oracle_out[0] ^= 1;
    } else {
        oracle_size = 1;
    }
#endif

    if (oracle != OracleVerdict::kAccepted) {
        Trap("FAIL-OPEN: the twin decoded a frame liblz4's frame API did not",
             stream_size);
    }
    if (oracle_size != twin.size) {
        std::fprintf(stderr, "twin size=%zu oracle size=%zu\n", twin.size,
                     oracle_size);
        Trap("size divergence on a frame both sides accepted", stream_size);
    }
    if (twin.size != 0 &&
        std::memcmp(twin.bytes.get(), oracle_out.get(), twin.size) != 0) {
        Trap("byte divergence on a frame both sides accepted", stream_size);
    }
    return 0;
}
