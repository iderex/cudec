/* The twin over the block loop and the per-frame state it carries (issue
 * #201): src/zstd_blocks.h against the pinned libzstd, on the GPU-less
 * runner.
 *
 * WHAT THIS ASKS THAT THE SIBLING TWINS CANNOT. Each unit's twin proves one
 * unit against the reference entry point that matches it, and
 * tests/zstd_entropy_twin.cpp drives whole frames for the entropy stage. What
 * neither can ask is whether the state that must survive a block boundary
 * actually survives it, and whether the state that must NOT survive a frame
 * boundary actually dies. Those two are the loop's whole contract and they
 * fail in the same silent way: the first block of every frame decodes
 * correctly and the bytes after it are wrong, on exactly the streams a
 * compressor chose to encode with a carry.
 *
 * SO THE CARRY IS PROVEN BY BREAKING IT RATHER THAN BY DECODING WITH IT. A
 * frame whose first block asks for a Huffman table it has not been given is
 * refused at the literals rung that says so; the SAME bytes, handed to a
 * state that was not re-initialised after an earlier frame, walk past that
 * rung. The refusal disappearing is what says ZstdFrameStateInit is
 * load-bearing, and a decode that merely succeeds would say nothing.
 *
 * THE CONTENT-SIZE NEGATIVES ARE HAND-BUILT, AND THE BUILDER IS PROVEN FIRST.
 * A frame that regenerates other than it declared cannot be produced by the
 * pinned compressor, so the bytes are written here - and a hand-built frame is
 * worth nothing until the reference agrees it is a frame, so the same builder
 * emits a legal frame first and libzstd decodes it before any mutation of it
 * is trusted.
 *
 * EVERY RUNG OF THE LOOP'S OWN LADDER IS REACHED, and the run says so at the
 * end. A rung with no negative behind it is a refusal nobody has seen fire. */
#include "require.h"
#include "zstd_blocks.h"
#include "zstd_corpus.h"
#include "zstd_twin_driver.h"

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using cudec_twin::Bytes;
using cudec_twin::DecodeFrame;
using cudec_twin::Run;
using cudec_detail::ZstdBlocksReject;

/* Above anything the corpus regenerates, so a refusal is never about room
 * except where the test asked for one. */
constexpr uint64_t kCapacity = 8ull * 1024ull * 1024ull;

size_t g_rung_seen[cudec_detail::kZstdBlocksRejectCount] = {0};
size_t g_carry_treeless = 0;
size_t g_carry_repeat[3] = {0, 0, 0};
size_t g_carry_raw_then_compressed = 0;
size_t g_carry_single_rle = 0;
size_t g_multi_block_frames = 0;
size_t g_fixture_bytes = 0;

void CoverRung(ZstdBlocksReject rung) {
    if (rung > cudec_detail::kZstdBlocksRejectNone &&
        rung < cudec_detail::kZstdBlocksRejectCount) {
        g_rung_seen[rung]++;
    }
}

/* ---- The harness --------------------------------------------------------
 *
 * The loop asks the caller to place every buffer it writes, which is what
 * lets a test hand it storage that is deliberately too small. The driver in
 * tests/zstd_twin_driver.h sizes everything to the block maximum and is the
 * shape a real caller uses; this one is the same thing with the sizes turned
 * into parameters. */
struct Harness {
    std::vector<cudec_detail::ZstdHufCell> huf_cells;
    std::vector<cudec_detail::ZstdFseCell> litlen_cells;
    std::vector<cudec_detail::ZstdFseCell> offset_cells;
    std::vector<cudec_detail::ZstdFseCell> matchlen_cells;
    cudec_detail::ZstdLiteralsScratch literals_scratch;
    cudec_detail::ZstdSeqScratch seq_scratch;
    std::vector<unsigned char> literals;
    std::vector<cudec_detail::ZstdSequence> sequences;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> destinations;
    cudec_detail::ZstdFrameState state;

    Harness(uint64_t literals_capacity, uint32_t sequences_capacity)
        : huf_cells(1u << cudec_detail::kZstdLiteralsMaxTableLog),
          litlen_cells(1u << cudec_detail::kZstdLitLenAccuracyLogMax),
          offset_cells(1u << cudec_detail::kZstdOffsetAccuracyLogMax),
          matchlen_cells(1u << cudec_detail::kZstdMatchLenAccuracyLogMax),
          literals_scratch(),
          seq_scratch(),
          literals(static_cast<size_t>(literals_capacity) + 1),
          sequences(static_cast<size_t>(sequences_capacity) + 1),
          offsets(static_cast<size_t>(sequences_capacity) + 1),
          destinations(static_cast<size_t>(sequences_capacity) + 2),
          state() {
        state.literals_table.cells = huf_cells.data();
        state.literals_table.capacity =
            static_cast<uint32_t>(huf_cells.size());
        state.litlen.cells = litlen_cells.data();
        state.litlen.capacity = static_cast<uint32_t>(litlen_cells.size());
        state.offset.cells = offset_cells.data();
        state.offset.capacity = static_cast<uint32_t>(offset_cells.size());
        state.matchlen.cells = matchlen_cells.data();
        state.matchlen.capacity = static_cast<uint32_t>(matchlen_cells.size());
        state.literals_scratch = &literals_scratch;
        state.seq_scratch = &seq_scratch;
        state.literals = literals.data();
        state.literals_capacity = literals_capacity;
        state.sequences = sequences.data();
        state.sequences_capacity = sequences_capacity;
        state.offsets = offsets.data();
        state.offsets_capacity = sequences_capacity;
        state.destinations = destinations.data();
        state.destinations_capacity = sequences_capacity + 1;
        cudec_detail::ZstdFrameStateInit(&state);
    }
};

constexpr uint64_t kBlockMaximum = cudec_detail::kZstdBlockSizeCeiling;

/* ---- Hand-built frames --------------------------------------------------
 *
 * Single_Segment, a four-byte content size and no checksum, so the header is
 * nine bytes and the declared size is written where a test can choose it. The
 * window is the content size for a single-segment frame (section 3.1.1.1.1.2),
 * which is what keeps these frames inside the subset. */
Bytes FrameHeader(uint32_t content_size) {
    Bytes frame;
    frame.push_back(0x28);
    frame.push_back(0xB5);
    frame.push_back(0x2F);
    frame.push_back(0xFD);
    /* Frame_Content_Size_flag 2 (a four-byte field), Single_Segment set. */
    frame.push_back(0xA0);
    for (unsigned shift = 0; shift < 32; shift += 8) {
        frame.push_back(static_cast<unsigned char>(content_size >> shift));
    }
    return frame;
}

void AppendBlockHeader(Bytes* frame, uint32_t block_size, unsigned type,
                       bool last) {
    const uint32_t raw = (block_size << 3) | (type << 1) | (last ? 1u : 0u);
    frame->push_back(static_cast<unsigned char>(raw));
    frame->push_back(static_cast<unsigned char>(raw >> 8));
    frame->push_back(static_cast<unsigned char>(raw >> 16));
}

Bytes RawFrame(uint32_t declared, const Bytes& content, bool last) {
    Bytes frame = FrameHeader(declared);
    AppendBlockHeader(&frame, static_cast<uint32_t>(content.size()), 0, last);
    frame.insert(frame.end(), content.begin(), content.end());
    return frame;
}

/* Two raw blocks rather than one, which is the only way to build a frame that
 * regenerates past its own declaration. A single-segment frame's window IS
 * its declared size (section 3.1.1.1.2), so one oversized block is refused by
 * the block header for passing the block maximum and never reaches the
 * loop's own bound - the two rungs would be indistinguishable. Split across
 * two blocks, each one is inside the maximum and only their sum is not. */
Bytes TwoRawFrame(uint32_t declared, const Bytes& first, const Bytes& second) {
    Bytes frame = FrameHeader(declared);
    AppendBlockHeader(&frame, static_cast<uint32_t>(first.size()), 0, false);
    frame.insert(frame.end(), first.begin(), first.end());
    AppendBlockHeader(&frame, static_cast<uint32_t>(second.size()), 0, true);
    frame.insert(frame.end(), second.begin(), second.end());
    return frame;
}

Bytes RleFrame(uint32_t declared, unsigned char value, uint32_t run) {
    Bytes frame = FrameHeader(declared);
    AppendBlockHeader(&frame, run, 1, true);
    frame.push_back(value);
    return frame;
}

/* Two RLE blocks, for the reason TwoRawFrame gives, and because an RLE
 * block's declared size is a REGENERATED length while its body is one byte -
 * the one place where stepping by the declaration would over-read. */
Bytes TwoRleFrame(uint32_t declared, unsigned char value, uint32_t run) {
    Bytes frame = FrameHeader(declared);
    AppendBlockHeader(&frame, run, 1, false);
    frame.push_back(value);
    AppendBlockHeader(&frame, run, 1, true);
    frame.push_back(value);
    return frame;
}

bool ReferenceAccepts(const Bytes& frame, Bytes* out) {
    Bytes buffer(kCapacity, 0);
    const size_t result = ZSTD_decompress(buffer.data(), buffer.size(),
                                          frame.data(), frame.size());
    if (ZSTD_isError(result) != 0) {
        return false;
    }
    out->assign(buffer.begin(), buffer.begin() + static_cast<long>(result));
    return true;
}

/* ---- The negatives over the loop's own ladder --------------------------- */

int NegativeFrame(const char* name, const Bytes& frame, uint64_t capacity,
                  ZstdBlocksReject want_rung, cudec_status want_status) {
    Harness harness(kBlockMaximum, static_cast<uint32_t>(kBlockMaximum));
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject frame_rung =
        cudec_detail::kZstdFrameRejectNone;
    REQUIRE_CTX(cudec_detail::ZstdParseFrameHeader(frame.data(), frame.size(),
                                                   &header,
                                                   &frame_rung) == CUDEC_OK,
                "%s: the frame header itself was refused", name);

    Bytes out(static_cast<size_t>(capacity) + 1, 0);
    uint64_t produced = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdBlocksReport report;
    ZstdBlocksReject rung = cudec_detail::kZstdBlocksRejectNone;
    const cudec_status status = cudec_detail::ZstdDecodeBlocks(
        frame.data() + header.header_size, frame.size() - header.header_size,
        &header, &harness.state, out.data(), capacity, &produced, &consumed,
        &report, &rung);
    REQUIRE_CTX(status == want_status, "%s: status %d, want %d", name,
                static_cast<int>(status), static_cast<int>(want_status));
    REQUIRE_CTX(rung == want_rung, "%s: rung %d, want %d", name,
                static_cast<int>(rung), static_cast<int>(want_rung));
    REQUIRE_CTX(produced == 0, "%s: reported %llu bytes on a refusal", name,
                static_cast<unsigned long long>(produced));
    CoverRung(rung);
    return 0;
}

int RunHandBuiltFrames() {
    /* The builder proven before anything built with it is trusted. */
    Bytes content;
    for (unsigned index = 0; index < 64; index++) {
        content.push_back(static_cast<unsigned char>(index * 7u + 1u));
    }
    const Bytes legal = RawFrame(64, content, true);
    Bytes reference;
    REQUIRE(ReferenceAccepts(legal, &reference));
    REQUIRE(reference.size() == content.size());
    REQUIRE(equal_bytes(reference.data(), content.data(), content.size()));

    Run run = DecodeFrame(legal, kCapacity);
    REQUIRE_CTX(run.ok, "the hand-built raw frame: %s", run.why.c_str());
    REQUIRE(run.output.size() == content.size());
    REQUIRE(equal_bytes(run.output.data(), content.data(), content.size()));
    REQUIRE(run.blocks == 1);

    /* A frame of a single RLE block, which is one of the carries this issue
     * names and the one shape the corpus cannot be relied on to hold. */
    const Bytes rle = RleFrame(300, 0x5A, 300);
    REQUIRE(ReferenceAccepts(rle, &reference));
    REQUIRE(reference.size() == 300);
    run = DecodeFrame(rle, kCapacity);
    REQUIRE_CTX(run.ok, "the hand-built RLE frame: %s", run.why.c_str());
    REQUIRE(run.output.size() == 300);
    REQUIRE(equal_bytes(run.output.data(), reference.data(), 300));
    REQUIRE(run.blocks == 1);
    g_carry_single_rle++;

    /* Produced less than declared. The block is legal and the declaration is
     * one byte above what it regenerates, which is the shape that would
     * otherwise return a short buffer as a success. */
    const Bytes short_frame = RawFrame(65, content, true);
    REQUIRE(!ReferenceAccepts(short_frame, &reference));
    int rc = NegativeFrame("declared one byte more than produced", short_frame,
                           kCapacity,
                           cudec_detail::kZstdBlocksRejectContentSizeMismatch,
                           CUDEC_ERR_CORRUPT_INPUT);
    if (rc != 0) {
        return rc;
    }

    /* Produced more than declared, refused as it happens rather than
     * afterwards: at "afterwards" the bytes are already written. */
    const Bytes half(content.begin(), content.begin() + 32);
    const Bytes long_frame = TwoRawFrame(40, half, half);
    REQUIRE(!ReferenceAccepts(long_frame, &reference));
    rc = NegativeFrame("a raw block past the declared size", long_frame,
                       kCapacity,
                       cudec_detail::kZstdBlocksRejectBlockPastCapacity,
                       CUDEC_ERR_OUTPUT_TOO_SMALL);
    if (rc != 0) {
        return rc;
    }

    /* The same wall for an RLE block, whose declared size is a regenerated
     * length rather than a body length - the one place those two differ. */
    const Bytes long_rle = TwoRleFrame(400, 0x5A, 250);
    REQUIRE(!ReferenceAccepts(long_rle, &reference));
    rc = NegativeFrame("an RLE block past the declared size", long_rle,
                       kCapacity,
                       cudec_detail::kZstdBlocksRejectBlockPastCapacity,
                       CUDEC_ERR_OUTPUT_TOO_SMALL);
    if (rc != 0) {
        return rc;
    }

    /* The frame declares more than the caller has room for. Refused before a
     * byte is written, which is why the capacity handed in is below the
     * declaration rather than below what the blocks turn out to produce. */
    rc = NegativeFrame("declared past the destination", legal, 32,
                       cudec_detail::kZstdBlocksRejectContentPastCapacity,
                       CUDEC_ERR_OUTPUT_TOO_SMALL);
    if (rc != 0) {
        return rc;
    }

    /* The input runs out before any block sets the last-block flag. */
    const Bytes unterminated = RawFrame(64, content, false);
    REQUIRE(!ReferenceAccepts(unterminated, &reference));
    run = DecodeFrame(unterminated, kCapacity);
    REQUIRE(!run.ok);
    REQUIRE_CTX(run.stage == cudec_twin::kStageBlockHeader,
                "no last block: stopped at %s", cudec_twin::kStageNames[run.stage]);
    REQUIRE(run.rung ==
            static_cast<int>(cudec_detail::kZstdFrameRejectBlockHeaderTruncated));

    /* A block claiming a size past the end of the frame. The block's declared
     * size is raised by one and nothing else moves, so the only thing wrong
     * with the frame is that the body it names is not there. The frame's
     * content size is set well above the block for this one: a single-segment
     * frame's window is its declared content size, so over a declaration of
     * 64 the raised block would pass the block maximum first and refuse for
     * being too large instead of for being absent. */
    Bytes overrun = RawFrame(128, content, true);
    const uint32_t raw = static_cast<uint32_t>(overrun[9]) |
                         (static_cast<uint32_t>(overrun[10]) << 8) |
                         (static_cast<uint32_t>(overrun[11]) << 16);
    const uint32_t bigger = ((raw >> 3) + 1u) << 3 | (raw & 0x07u);
    overrun[9] = static_cast<unsigned char>(bigger);
    overrun[10] = static_cast<unsigned char>(bigger >> 8);
    overrun[11] = static_cast<unsigned char>(bigger >> 16);
    REQUIRE(!ReferenceAccepts(overrun, &reference));
    run = DecodeFrame(overrun, kCapacity);
    REQUIRE(!run.ok);
    REQUIRE_CTX(run.stage == cudec_twin::kStageBlockHeader,
                "block past the frame: stopped at %s",
                cudec_twin::kStageNames[run.stage]);
    REQUIRE(run.rung ==
            static_cast<int>(cudec_detail::kZstdFrameRejectBlockBodyTruncated));
    return 0;
}

/* ---- The storage rungs --------------------------------------------------
 *
 * Reachable only from a caller that placed the buffers, which is every caller
 * the loop has: the two below are what a kernel sizing a per-warp slab gets
 * wrong, and they are separated from the bad-request rung because they depend
 * on the frame rather than on the call. */
int RunStorageRungs(const Bytes& frame_with_sequences) {
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject frame_rung =
        cudec_detail::kZstdFrameRejectNone;
    REQUIRE(cudec_detail::ZstdParseFrameHeader(frame_with_sequences.data(),
                                               frame_with_sequences.size(),
                                               &header,
                                               &frame_rung) == CUDEC_OK);
    const uint64_t block_max =
        cudec_detail::ZstdLiteralsBlockMaximum(header.window_size);
    Bytes out(static_cast<size_t>(kCapacity), 0);
    uint64_t produced = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdBlocksReport report;

    {
        /* One byte of literals room below the block maximum this frame's
         * window implies. Checked against the window rather than against any
         * section, so the refusal cannot depend on bytes an attacker chose. */
        Harness harness(block_max - 1, static_cast<uint32_t>(block_max));
        ZstdBlocksReject rung = cudec_detail::kZstdBlocksRejectNone;
        const cudec_status status = cudec_detail::ZstdDecodeBlocks(
            frame_with_sequences.data() + header.header_size,
            frame_with_sequences.size() - header.header_size, &header,
            &harness.state, out.data(), kCapacity, &produced, &consumed,
            &report, &rung);
        REQUIRE(status == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(rung == cudec_detail::kZstdBlocksRejectLiteralsStorageTooSmall);
        CoverRung(rung);
    }

    {
        /* Room for no sequences at all, against a frame that has some. */
        Harness harness(block_max, 0);
        ZstdBlocksReject rung = cudec_detail::kZstdBlocksRejectNone;
        const cudec_status status = cudec_detail::ZstdDecodeBlocks(
            frame_with_sequences.data() + header.header_size,
            frame_with_sequences.size() - header.header_size, &header,
            &harness.state, out.data(), kCapacity, &produced, &consumed,
            &report, &rung);
        REQUIRE(status == CUDEC_ERR_OUTPUT_TOO_SMALL);
        REQUIRE(rung == cudec_detail::kZstdBlocksRejectSequenceStorageTooSmall);
        REQUIRE(report.stage == cudec_detail::kZstdBlocksStageSequences);
        CoverRung(rung);
    }

    {
        /* The destination array one entry short of what the sequence array
         * would need. A caller bug rather than a stream, so it is refused
         * before the frame is looked at. */
        Harness harness(block_max, static_cast<uint32_t>(block_max));
        harness.state.destinations_capacity =
            harness.state.sequences_capacity;
        ZstdBlocksReject rung = cudec_detail::kZstdBlocksRejectNone;
        const cudec_status status = cudec_detail::ZstdDecodeBlocks(
            frame_with_sequences.data() + header.header_size,
            frame_with_sequences.size() - header.header_size, &header,
            &harness.state, out.data(), kCapacity, &produced, &consumed,
            &report, &rung);
        REQUIRE(status == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(rung == cudec_detail::kZstdBlocksRejectBadRequest);
        CoverRung(rung);
    }
    return 0;
}

/* ---- The reset, proven by removing it ----------------------------------- */

int RunResetProof(const Bytes& compressed_literals_frame) {
    /* The first compressed block's literals section begins three bytes after
     * the frame header - the block header is those three bytes - and its type
     * is the low two bits of the section's first byte. Turning Compressed
     * into Treeless leaves a section that asks for a table nobody gave it. */
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject frame_rung =
        cudec_detail::kZstdFrameRejectNone;
    REQUIRE(cudec_detail::ZstdParseFrameHeader(
                compressed_literals_frame.data(),
                compressed_literals_frame.size(), &header,
                &frame_rung) == CUDEC_OK);
    const size_t section = header.header_size + 3;
    REQUIRE(section < compressed_literals_frame.size());

    Bytes treeless_first = compressed_literals_frame;
    treeless_first[section] = static_cast<unsigned char>(
        (treeless_first[section] & ~0x03u) |
        cudec_detail::kZstdLiteralsTypeTreeless);

    /* A fresh frame state is the first block of a frame, and this is what the
     * reset buys: the table is absent and the section is refused for it. */
    Run fresh = DecodeFrame(treeless_first, kCapacity);
    REQUIRE(!fresh.ok);
    REQUIRE_CTX(fresh.stage == cudec_twin::kStageLiterals,
                "treeless first block: stopped at %s",
                cudec_twin::kStageNames[fresh.stage]);
    REQUIRE_CTX(fresh.rung == static_cast<int>(
                                  cudec_detail::kZstdLiteralsRejectNoPreviousTable),
                "treeless first block: rung %d", fresh.rung);

    /* The same bytes over a state that decoded a frame and was NOT
     * re-initialised. If the reset were not load-bearing this would refuse
     * identically; it does not, because the previous frame's tree is still
     * marked present, so the refusal above is the reset doing its work. */
    Harness harness(kBlockMaximum, static_cast<uint32_t>(kBlockMaximum));
    Bytes out(static_cast<size_t>(kCapacity), 0);
    uint64_t produced = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdBlocksReport report;
    ZstdBlocksReject rung = cudec_detail::kZstdBlocksRejectNone;
    REQUIRE(cudec_detail::ZstdDecodeBlocks(
                compressed_literals_frame.data() + header.header_size,
                compressed_literals_frame.size() - header.header_size, &header,
                &harness.state, out.data(), kCapacity, &produced, &consumed,
                &report, &rung) == CUDEC_OK);

    uint64_t second_produced = 0;
    const cudec_status carried = cudec_detail::ZstdDecodeBlocks(
        treeless_first.data() + header.header_size,
        treeless_first.size() - header.header_size, &header, &harness.state,
        out.data(), kCapacity, &second_produced, &consumed, &report, &rung);
    const bool refused_for_the_missing_table =
        carried != CUDEC_OK &&
        report.stage == cudec_detail::kZstdBlocksStageLiterals &&
        report.rung == static_cast<int>(
                           cudec_detail::kZstdLiteralsRejectNoPreviousTable);
    REQUIRE_CTX(!refused_for_the_missing_table,
                "the state was not reset and the table was still reported "
                "missing, so the reset above proved nothing");
    return 0;
}

/* ---- A frame whose first block is Raw and whose second is Compressed -----
 *
 * The pinned compressor does not emit that mixture: it chooses Raw only where
 * a block does not compress, and those blocks arrive together rather than in
 * front of a compressed one. So the mixture is grafted - a raw block written
 * here in front of a real compressed body, under a header this file writes -
 * and libzstd decodes the result before the twin's answer is compared to it,
 * which is what makes the graft a frame rather than a guess.
 *
 * The graft is sound because everything the second block can reach still
 * exists. Its match offsets never point behind the start of the frame it came
 * from, the window of the new single-segment frame is the whole declared size
 * and so is at least as large as the old one's reach, and the repeat-offset
 * history starts a frame the same way whatever the first block was. */
size_t FrameHeaderSize(const Bytes& frame) {
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject rung = cudec_detail::kZstdFrameRejectNone;
    if (cudec_detail::ZstdParseFrameHeader(frame.data(), frame.size(), &header,
                                           &rung) != CUDEC_OK) {
        return 0;
    }
    return header.header_size;
}

int RunRawThenCompressed(const Bytes& body, const Bytes& original) {
    Bytes prefix;
    for (unsigned index = 0; index < 40; index++) {
        prefix.push_back(static_cast<unsigned char>(0xC0u + (index & 0x0Fu)));
    }
    Bytes frame = FrameHeader(
        static_cast<uint32_t>(prefix.size() + original.size()));
    AppendBlockHeader(&frame, static_cast<uint32_t>(prefix.size()), 0, false);
    frame.insert(frame.end(), prefix.begin(), prefix.end());
    frame.insert(frame.end(), body.begin(), body.end());

    Bytes reference;
    REQUIRE(ReferenceAccepts(frame, &reference));
    REQUIRE(reference.size() == prefix.size() + original.size());
    REQUIRE(equal_bytes(reference.data(), prefix.data(), prefix.size()));
    REQUIRE(equal_bytes(reference.data() + prefix.size(), original.data(),
                        original.size()));

    const Run run = DecodeFrame(frame, kCapacity);
    REQUIRE_CTX(run.ok, "raw then compressed: %s", run.why.c_str());
    REQUIRE(run.output.size() == reference.size());
    REQUIRE(equal_bytes(run.output.data(), reference.data(),
                        reference.size()));
    REQUIRE(run.blocks >= 2);
    g_carry_raw_then_compressed++;
    return 0;
}

/* ---- The corpus, and which carries it actually holds -------------------- */

int RunCorpus() {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());

    Bytes compressed_literals_frame;
    Bytes frame_with_sequences;
    Bytes graft_body;
    Bytes graft_original;

    for (size_t index = 0; index < fixtures.size(); index++) {
        const ZstdFixture& fixture = fixtures[index];
        const Bytes frame(fixture.compressed.begin(),
                          fixture.compressed.end());
        const Run run = DecodeFrame(frame, kCapacity);
        if (!run.ok) {
            /* A fixture outside the v1 subset is declined rather than
             * refused, and declining is not this file's subject. */
            REQUIRE_CTX(run.status == CUDEC_ERR_UNSUPPORTED,
                        "%s: %s at %s rung %d", fixture.name.c_str(),
                        run.why.c_str(), cudec_twin::kStageNames[run.stage],
                        run.rung);
            continue;
        }
        REQUIRE_CTX(run.output.size() == fixture.original.size(),
                    "%s: %zu bytes, want %zu", fixture.name.c_str(),
                    run.output.size(), fixture.original.size());
        REQUIRE_CTX(equal_bytes(run.output.data(), fixture.original.data(),
                                fixture.original.size()),
                    "%s", fixture.name.c_str());
        g_fixture_bytes += fixture.original.size();

        ZstdFrameShape shape;
        std::string why;
        REQUIRE_CTX(ParseZstdFrameShape(frame, &shape, &why), "%s: %s",
                    fixture.name.c_str(), why.c_str());
        if (shape.blocks.size() > 1) {
            g_multi_block_frames++;
        }
        bool seen_compressed_literals = false;
        bool seen_raw_block = false;
        for (size_t b = 0; b < shape.blocks.size(); b++) {
            const ZstdBlockShape& block = shape.blocks[b];
            if (block.block_type == kZstdBlockRaw) {
                seen_raw_block = true;
            }
            if (block.block_type != kZstdBlockCompressed) {
                continue;
            }
            if (seen_raw_block) {
                g_carry_raw_then_compressed++;
                seen_raw_block = false;
            }
            if (block.literals_type == kZstdLiteralsTreeless) {
                /* Only a carry can satisfy it: a Treeless section reuses the
                 * tree an earlier block described. */
                REQUIRE_CTX(seen_compressed_literals,
                            "%s: a treeless block with no compressed literals "
                            "block before it", fixture.name.c_str());
                g_carry_treeless++;
            }
            if (block.literals_type == kZstdLiteralsCompressed) {
                seen_compressed_literals = true;
                if (compressed_literals_frame.empty() && b == 0) {
                    compressed_literals_frame = frame;
                    graft_body.assign(
                        frame.begin() +
                            static_cast<long>(FrameHeaderSize(frame)),
                        frame.end() -
                            (shape.checksum_present ? 4 : 0));
                    graft_original.assign(fixture.original.begin(),
                                          fixture.original.end());
                }
            }
            if (block.sequence_count > 0) {
                if (frame_with_sequences.empty()) {
                    frame_with_sequences = frame;
                }
                if (block.ll_mode == kZstdTableRepeat) {
                    g_carry_repeat[0]++;
                }
                if (block.of_mode == kZstdTableRepeat) {
                    g_carry_repeat[1]++;
                }
                if (block.ml_mode == kZstdTableRepeat) {
                    g_carry_repeat[2]++;
                }
            }
        }
    }

    /* The carries this issue names, each present in the corpus that was just
     * decoded byte-exactly. A corpus that stopped holding one of them would
     * leave this file passing over a carry it no longer exercises, which is
     * the failure the counts exist to refuse. */
    REQUIRE(g_carry_treeless > 0);
    REQUIRE(g_carry_repeat[0] > 0);
    REQUIRE(g_carry_repeat[1] > 0);
    REQUIRE(g_carry_repeat[2] > 0);
    REQUIRE(g_multi_block_frames > 0);
    REQUIRE(!compressed_literals_frame.empty());
    REQUIRE(!frame_with_sequences.empty());
    REQUIRE(!graft_body.empty());

    const int mixed = RunRawThenCompressed(graft_body, graft_original);
    if (mixed != 0) {
        return mixed;
    }

    const int storage = RunStorageRungs(frame_with_sequences);
    if (storage != 0) {
        return storage;
    }
    return RunResetProof(compressed_literals_frame);
}

}  // namespace

int main() {
    const int hand_built = RunHandBuiltFrames();
    if (hand_built != 0) {
        return hand_built;
    }
    const int corpus = RunCorpus();
    if (corpus != 0) {
        return corpus;
    }

    /* Every rung of the loop's own ladder reached by a negative above. A rung
     * with nothing behind it is a refusal nobody has seen fire, and the whole
     * point of enumerating them is that each one can be aimed at. */
    for (int rung = cudec_detail::kZstdBlocksRejectNone + 1;
         rung < cudec_detail::kZstdBlocksRejectCount; rung++) {
        REQUIRE_CTX(g_rung_seen[rung] > 0, "block-loop rung %d never fired",
                    rung);
    }

    std::printf(
        "PASS: block loop - %zu fixture bytes decoded byte-exact, %zu "
        "multi-block frames, carries seen: treeless %zu, repeat LL/OF/ML "
        "%zu/%zu/%zu, raw-then-compressed %zu, single RLE %zu; %d reject "
        "rungs each reached\n",
        g_fixture_bytes, g_multi_block_frames, g_carry_treeless,
        g_carry_repeat[0], g_carry_repeat[1], g_carry_repeat[2],
        g_carry_raw_then_compressed, g_carry_single_rle,
        cudec_detail::kZstdBlocksRejectCount - 1);
    return 0;
}
