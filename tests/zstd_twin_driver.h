/* The host driver that decodes a whole Zstd frame through the single-sourced
 * units in src/, so a claim about one of them becomes bytes the pinned
 * reference has an opinion about.
 *
 * WHY THIS IS A TEST FILE AND NOT A UNIT, AND WHAT IS LEFT OF THAT. Every
 * stage it calls is shipped, and since issue #201 the loop over a frame's
 * blocks is shipped too: it is `ZstdDecodeBlocks` in src/zstd_blocks.h, with
 * the per-frame entropy state it carries. What is left here is the frame
 * envelope around that loop - the header parse, the content checksum, the
 * refusal of bytes after the frame - and the storage the loop asks the caller
 * to place, which on the host is three std::vectors and on a device will be
 * whatever the kernel decides. That is the least code that turns the units
 * into one byte string, so that "the offset resolved to four" and "this
 * literals section decodes" become "the frame decodes to exactly what libzstd
 * decodes it to".
 *
 * THE STATE THAT SURVIVES A BLOCK BOUNDARY IS NOT THIS FILE'S ANY MORE. The
 * Huffman table a Treeless literals section reuses, the three FSE tables a
 * Repeat mode reuses, and the repeat-offset history are reset by
 * ZstdFrameStateInit once per frame and carried by the shipped loop, which is
 * where the argument for that lives. A driver that reset any of them per
 * block would decode the first block of every frame correctly and then
 * produce wrong bytes, and the corpus rows exist to catch exactly that.
 *
 * WHERE A RUN STOPPED IS REPORTED, NOT JUST THAT IT DID. Reject parity over a
 * mutation corpus is only worth something if a refusal can be attributed, so
 * the stage is carried alongside the verdict and a caller that cares asserts
 * on it. The stages the loop walks are the loop's own enumeration and are
 * spelled here by name rather than by number, so the two cannot drift. */
#ifndef CUDEC_TESTS_ZSTD_TWIN_DRIVER_H
#define CUDEC_TESTS_ZSTD_TWIN_DRIVER_H

#include "zstd_blocks.h"
#include "zstd_exec.h"
#include "zstd_frame.h"
#include "zstd_literals.h"
#include "zstd_repcode.h"
#include "zstd_seq.h"
#include "xxhash64.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cudec_twin {

using Bytes = std::vector<unsigned char>;

/* Cell storage for the three sequence fields, sized by each one's
 * accuracy-log ceiling. */
constexpr unsigned kLitLenCells = 1u
                                  << cudec_detail::kZstdLitLenAccuracyLogMax;
constexpr unsigned kOffsetCells = 1u
                                  << cudec_detail::kZstdOffsetAccuracyLogMax;
constexpr unsigned kMatchLenCells =
    1u << cudec_detail::kZstdMatchLenAccuracyLogMax;

/* What the per-block working buffers are sized to. The block maximum is the
 * smaller of the window and the 128 KB ceiling (section 3.1.1.2.4), and the
 * largest window the subset admits is 8 MB, so the ceiling is the answer for
 * every frame this driver can be handed. Sizing to the frame's own window
 * instead would make the driver refuse a legal frame whenever the window
 * grew, and the loop's storage rungs would then be firing on the harness
 * rather than on the stream. */
constexpr uint64_t kDriverBlockMaximum = cudec_detail::kZstdBlockSizeCeiling;
static_assert(cudec_detail::kZstdMaxWindowSize >=
                  cudec_detail::kZstdBlockSizeCeiling,
              "the ceiling is only the block maximum while the largest "
              "admitted window is at least as large as it");

/* Everything the shipped loop asks the caller to place. Heap-backed here
 * because a host test can afford it; the reason the loop does not allocate is
 * that a kernel decides where this lives. */
struct Storage {
    std::vector<cudec_detail::ZstdHufCell> huf_cells;
    cudec_detail::ZstdFseCell litlen_cells[kLitLenCells];
    cudec_detail::ZstdFseCell offset_cells[kOffsetCells];
    cudec_detail::ZstdFseCell matchlen_cells[kMatchLenCells];
    cudec_detail::ZstdLiteralsScratch literals_scratch;
    cudec_detail::ZstdSeqScratch seq_scratch;
    std::vector<unsigned char> literals;
    std::vector<cudec_detail::ZstdSequence> sequences;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> destinations;
    cudec_detail::ZstdFrameState state;

    Storage()
        : huf_cells(1u << cudec_detail::kZstdLiteralsMaxTableLog),
          literals_scratch(),
          seq_scratch(),
          literals(static_cast<size_t>(kDriverBlockMaximum)),
          /* A block regenerating N bytes carries at most N sequences, which
           * is the bound ZstdParseSeqSectionHeader refuses past, so this size
           * never refuses a legal frame. */
          sequences(static_cast<size_t>(kDriverBlockMaximum)),
          offsets(static_cast<size_t>(kDriverBlockMaximum)),
          destinations(static_cast<size_t>(kDriverBlockMaximum) + 1),
          state() {
        state.literals_table.cells = huf_cells.data();
        state.literals_table.capacity =
            static_cast<uint32_t>(huf_cells.size());
        state.litlen.cells = litlen_cells;
        state.litlen.capacity = kLitLenCells;
        state.offset.cells = offset_cells;
        state.offset.capacity = kOffsetCells;
        state.matchlen.cells = matchlen_cells;
        state.matchlen.capacity = kMatchLenCells;
        state.literals_scratch = &literals_scratch;
        state.seq_scratch = &seq_scratch;
        state.literals = literals.data();
        state.literals_capacity = literals.size();
        state.sequences = sequences.data();
        state.sequences_capacity = static_cast<uint32_t>(sequences.size());
        state.offsets = offsets.data();
        state.offsets_capacity = static_cast<uint32_t>(offsets.size());
        state.destinations = destinations.data();
        state.destinations_capacity =
            static_cast<uint32_t>(destinations.size());
        cudec_detail::ZstdFrameStateInit(&state);
    }
};

/* Where a run stopped. The stages the block loop walks keep the loop's own
 * numbering, spelled by name so a stage added there cannot silently become a
 * different one here; the three the envelope owns follow after them. */
enum DriverStage {
    kStageNone = cudec_detail::kZstdBlocksStageNone,
    kStageBlockHeader = cudec_detail::kZstdBlocksStageBlockHeader,
    kStageLiterals = cudec_detail::kZstdBlocksStageLiterals,
    kStageSequenceHeader = cudec_detail::kZstdBlocksStageSequenceHeader,
    kStageSequenceTable = cudec_detail::kZstdBlocksStageSequenceTable,
    kStageSequences = cudec_detail::kZstdBlocksStageSequences,
    kStageRepcode = cudec_detail::kZstdBlocksStageRepcode,
    kStageExecute = cudec_detail::kZstdBlocksStageExecute,
    kStageContentSize = cudec_detail::kZstdBlocksStageContentSize,
    kStageFrameHeader = cudec_detail::kZstdBlocksStageCount,
    kStageChecksum,
    kStageTrailingBytes,
    kStageCount
};

const char* const kStageNames[kStageCount] = {
    "none",     "block-header",    "literals",      "sequence-header",
    "sequence-table", "sequences", "repcode",       "execute",
    "content-size",   "frame-header", "checksum",   "trailing-bytes"};

/* The seven ways an Offset_Value can resolve, counted as the loop goes. A
 * corpus that decodes byte-identically proves nothing about a rule it never
 * reached, and this is what says which rules it reached. The split itself is
 * ZstdRepcodeClassify's, in src/zstd_repcode.h, so the names below label the
 * shipped enumeration rather than a second copy of it. */
enum RulePath {
    kPathExplicit = cudec_detail::kZstdRepcodePathExplicit,
    kPathSlot0 = cudec_detail::kZstdRepcodePathSlot0,
    kPathSlot1 = cudec_detail::kZstdRepcodePathSlot1,
    kPathSlot2 = cudec_detail::kZstdRepcodePathSlot2,
    kPathShiftedSlot1 = cudec_detail::kZstdRepcodePathShiftedSlot1,
    kPathShiftedSlot2 = cudec_detail::kZstdRepcodePathShiftedSlot2,
    kPathMinusOne = cudec_detail::kZstdRepcodePathMinusOne,
    kPathCount = cudec_detail::kZstdRepcodePathCount
};

const char* const kPathNames[kPathCount] = {
    "explicit", "rep1", "rep2", "rep3", "rep1-shifted", "rep2-shifted",
    "rep1-minus-one"};

/* What a driver run produced, or where and why it stopped. */
struct Run {
    bool ok = true;
    /* The status the stage that stopped returned, which is not the same
     * question as the stage: a frame this decoder DECLINES as outside the v1
     * subset comes back CUDEC_ERR_UNSUPPORTED, and a corpus-level reject
     * parity run has to separate that from a refusal. */
    cudec_status status = CUDEC_OK;
    DriverStage stage = kStageNone;
    /* The rung the stage that stopped reported, as an int because each stage
     * has its own enumeration and the caller that cares knows which. */
    int rung = 0;
    std::string why;
    Bytes output;
    size_t blocks = 0;
    size_t sequences = 0;
    size_t repeats = 0;
    size_t path[kPathCount] = {0};
};

inline void Stop(Run* run, DriverStage stage, int rung, cudec_status status,
                 const char* why) {
    run->ok = false;
    run->stage = stage;
    run->rung = rung;
    run->status = status;
    run->why = why;
}

/* Decodes one whole frame inside the v1 subset (#102).
 *
 * The frame-level enforcement is here or in the loop rather than left out,
 * because a run that skipped it would ACCEPT frames the reference refuses and
 * a reject parity claim over a mutation corpus would be measuring the gap
 * instead of the decoder: the declared content size, the content checksum
 * when the descriptor sets it, the window bound on a match, and bytes after
 * the frame.
 *
 * `dst_capacity` bounds what the run will produce; a frame declaring more is
 * refused rather than truncated. */
inline Run DecodeFrame(const Bytes& frame, uint64_t dst_capacity) {
    Run run;
    Storage storage;
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject frame_rung =
        cudec_detail::kZstdFrameRejectNone;
    const cudec_status header_status = cudec_detail::ZstdParseFrameHeader(
        frame.data(), frame.size(), &header, &frame_rung);
    if (header_status != CUDEC_OK) {
        Stop(&run, kStageFrameHeader, static_cast<int>(frame_rung),
             header_status, "frame header refused");
        return run;
    }

    /* The frame's whole output, sized to what it declares, which the subset
     * guarantees exists and which the loop bounds against this capacity
     * before it writes anything. One byte of slack so the buffer has an
     * address even for a frame declaring nothing; the capacity handed to the
     * loop is the caller's number and never the slack. */
    const uint64_t capacity = header.frame_content_size < dst_capacity
                                  ? header.frame_content_size
                                  : dst_capacity;
    run.output.assign(static_cast<size_t>(capacity) + 1, 0);
    /* The loop is handed the buffer's own size and never the caller's larger
     * number, so a frame declaring more than the caller allowed is refused by
     * the loop's capacity rung rather than by a write nobody bounded. Where
     * the declaration fits, the two numbers are the same. */

    uint64_t produced = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdBlocksReport report;
    cudec_detail::ZstdBlocksReject blocks_rung =
        cudec_detail::kZstdBlocksRejectNone;
    const cudec_status blocks_status = cudec_detail::ZstdDecodeBlocks(
        frame.data() + header.header_size, frame.size() - header.header_size,
        &header, &storage.state, run.output.data(), capacity, &produced,
        &consumed, &report, &blocks_rung);
    run.blocks = static_cast<size_t>(report.blocks);
    run.sequences = static_cast<size_t>(report.sequences);
    run.repeats = static_cast<size_t>(report.repeats);
    for (unsigned index = 0; index < kPathCount; index++) {
        run.path[index] = static_cast<size_t>(report.path[index]);
    }
    if (blocks_status != CUDEC_OK) {
        /* The loop's stage numbering IS this enumeration's for the stages it
         * walks, which the enum above states rather than converts. Where the
         * refusal is the loop's own rather than a unit's, the loop's rung is
         * what a caller can assert on, and the report's rung is zero. */
        const DriverStage stage = static_cast<DriverStage>(report.stage);
        const int rung = report.stage == cudec_detail::kZstdBlocksStageContentSize
                             ? static_cast<int>(blocks_rung)
                             : report.rung;
        Stop(&run, stage, rung, blocks_status, kStageNames[stage]);
        return run;
    }
    run.output.resize(static_cast<size_t>(produced));

    uint64_t pos = header.header_size + consumed;
    if (header.content_checksum) {
        const uint64_t digest =
            cudec_detail::Xxh64(run.output.data(), run.output.size());
        if (cudec_detail::ZstdVerifyContentChecksum(
                frame.data() + pos, frame.size() - pos, digest,
                &frame_rung) != CUDEC_OK) {
            Stop(&run, kStageChecksum, static_cast<int>(frame_rung),
                 CUDEC_ERR_CORRUPT_INPUT, "content checksum refused");
            return run;
        }
        pos += 4;
    }

    /* A chunk holding more than one frame is outside the subset (section
     * 12.4), so bytes after the frame are refused rather than ignored. */
    if (pos != frame.size()) {
        Stop(&run, kStageTrailingBytes, 0, CUDEC_ERR_CORRUPT_INPUT,
             "bytes follow the frame");
        return run;
    }
    return run;
}

}  // namespace cudec_twin

#endif /* CUDEC_TESTS_ZSTD_TWIN_DRIVER_H */
