/* The host driver that decodes a whole Zstd frame through the single-sourced
 * units in src/, so a claim about one of them becomes bytes the pinned
 * reference has an opinion about.
 *
 * WHY THIS IS A TEST FILE AND NOT A UNIT. Every stage it calls is shipped -
 * the frame and block headers, the literals section, the FSE tables, the
 * sequences loop, the repeat-offset history, the sequence execution, the
 * content checksum - and the one thing it adds is not: the loop over a
 * frame's blocks. That is its own issue with its own contract, and a shipped
 * decoder will make different choices about where the per-frame state lives.
 * What is here is the least code that turns the stages into one byte string,
 * so that "the offset resolved to four" and "this literals section decodes"
 * become "the frame decodes to exactly what libzstd decodes it to".
 *
 * THE COPY THAT EXECUTES A SEQUENCE USED TO BE HERE AND IS NOW SHIPPED. It
 * moved to src/zstd_exec.h under issue #198, which is what makes the corpus
 * runs below a statement about the decoder rather than about a copy loop that
 * only tests ever compile. The driver still owns the block loop and the
 * per-frame state it carries, because that is the sub-issue that has not
 * landed.
 *
 * ALL OF THE STATE IS PER-FRAME AND NONE OF IT IS PER-BLOCK. The Huffman
 * table a Treeless literals section reuses, the three FSE tables a Repeat
 * mode reuses, and the repeat-offset history are exactly what survives a
 * block boundary. A driver that reset any of them would decode the first
 * block of every frame correctly and then produce wrong bytes, which is a
 * failure mode the corpus rows exist to catch.
 *
 * WHERE A RUN STOPPED IS REPORTED, NOT JUST THAT IT DID. Reject parity over a
 * mutation corpus is only worth something if a refusal can be attributed, so
 * the stage is carried out alongside the verdict and a caller that cares
 * asserts on it. */
#ifndef CUDEC_TESTS_ZSTD_TWIN_DRIVER_H
#define CUDEC_TESTS_ZSTD_TWIN_DRIVER_H

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

/* Everything one frame's decode needs, in the shape each unit asks for. */
struct Driver {
    std::vector<cudec_detail::ZstdHufCell> huf_cells;
    cudec_detail::ZstdLiteralsScratch literals_scratch;
    cudec_detail::ZstdLiteralsTable literals_table;
    cudec_detail::ZstdFseCell litlen_cells[kLitLenCells];
    cudec_detail::ZstdFseCell offset_cells[kOffsetCells];
    cudec_detail::ZstdFseCell matchlen_cells[kMatchLenCells];
    cudec_detail::ZstdSeqTable litlen;
    cudec_detail::ZstdSeqTable offset;
    cudec_detail::ZstdSeqTable matchlen;
    cudec_detail::ZstdSeqScratch seq_scratch;
    cudec_detail::ZstdRepcodeHistory repcodes;

    Driver()
        : huf_cells(1u << cudec_detail::kZstdLiteralsMaxTableLog),
          literals_scratch(), literals_table(), litlen(), offset(), matchlen(),
          seq_scratch(), repcodes() {
        literals_table.cells = huf_cells.data();
        literals_table.capacity = static_cast<uint32_t>(huf_cells.size());
        literals_table.table_log = 0;
        literals_table.present = false;
        litlen.cells = litlen_cells;
        litlen.capacity = kLitLenCells;
        offset.cells = offset_cells;
        offset.capacity = kOffsetCells;
        matchlen.cells = matchlen_cells;
        matchlen.capacity = kMatchLenCells;
        cudec_detail::ZstdRepcodeInit(&repcodes);
    }
};

/* Where a run stopped. Named per STAGE rather than per rung: each stage's
 * rungs are the business of that unit's own twin, and what a corpus-level run
 * needs is which stage refused. */
enum DriverStage {
    kStageNone = 0,
    kStageFrameHeader,
    kStageBlockHeader,
    kStageLiterals,
    kStageSequenceHeader,
    kStageSequenceTable,
    kStageSequences,
    kStageRepcode,
    kStageExecute,
    kStageContentSize,
    kStageChecksum,
    kStageTrailingBytes,
    kStageCount
};

const char* const kStageNames[kStageCount] = {
    "none",           "frame-header",  "block-header",   "literals",
    "sequence-header", "sequence-table", "sequences",     "repcode",
    "execute",        "content-size",  "checksum",       "trailing-bytes"};

/* The seven ways an Offset_Value can resolve, counted as the driver goes. A
 * corpus that decodes byte-identically proves nothing about a rule it never
 * reached, and this is what says which rules it reached. */
enum RulePath {
    kPathExplicit = 0, /* Offset_Value above three: an explicit distance */
    kPathSlot0,        /* literals, value 1: the most recent offset */
    kPathSlot1,        /* literals, value 2: the second */
    kPathSlot2,        /* literals, value 3: the third */
    kPathShiftedSlot1, /* no literals, value 1: the second */
    kPathShiftedSlot2, /* no literals, value 2: the third */
    /* no literals, value 3: the most recent offset, minus one */
    kPathMinusOne,
    kPathCount
};

const char* const kPathNames[kPathCount] = {
    "explicit", "rep1", "rep2", "rep3", "rep1-shifted", "rep2-shifted",
    "rep1-minus-one"};

inline RulePath ClassifyPath(uint64_t offset_value,
                             uint32_t literals_length) {
    if (offset_value > cudec_detail::kZstdRepcodeMaxSlotValue) {
        return kPathExplicit;
    }
    if (literals_length != 0) {
        return static_cast<RulePath>(kPathSlot0 + (offset_value - 1));
    }
    return static_cast<RulePath>(kPathShiftedSlot1 + (offset_value - 1));
}

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
 * The frame-level enforcement is here rather than left out, because a run
 * that skipped it would ACCEPT frames the reference refuses and a reject
 * parity claim over a mutation corpus would be measuring the gap instead of
 * the decoder: the declared content size, the content checksum when the
 * descriptor sets it, the window bound on a match, and bytes after the frame.
 *
 * `dst_capacity` bounds what the run will produce; a frame declaring more is
 * refused rather than truncated. */
inline Run DecodeFrame(const Bytes& frame, uint64_t dst_capacity) {
    Run run;
    Driver driver;
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
    if (header.frame_content_size > dst_capacity) {
        Stop(&run, kStageContentSize, 0, CUDEC_ERR_OUTPUT_TOO_SMALL,
             "the declared content size is past the destination");
        return run;
    }

    /* The frame's whole output, sized to what it declares, which the subset
     * guarantees exists and the check above bounded. One byte of slack so the
     * buffer has an address even for a frame declaring nothing; the capacity
     * handed to the units is the declared size and never the slack. */
    run.output.assign(static_cast<size_t>(header.frame_content_size) + 1, 0);
    const uint64_t capacity = header.frame_content_size;
    uint64_t produced = 0;

    uint64_t pos = header.header_size;
    Bytes literals;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> destinations;
    for (;;) {
        cudec_detail::ZstdBlockHeader block;
        const cudec_status block_status = cudec_detail::ZstdParseBlockHeader(
            frame.data() + pos, frame.size() - pos, header.window_size, &block,
            &frame_rung);
        if (block_status != CUDEC_OK) {
            Stop(&run, kStageBlockHeader, static_cast<int>(frame_rung),
                 block_status, "block header refused");
            return run;
        }
        run.blocks++;
        const unsigned char* body = frame.data() + pos + 3;
        if (block.block_type == cudec_detail::kZstdBlockTypeRaw) {
            if (block.body_size > capacity - produced) {
                Stop(&run, kStageContentSize, 0, CUDEC_ERR_OUTPUT_TOO_SMALL,
                     "the frame regenerated more than the destination");
                return run;
            }
            for (uint64_t i = 0; i < block.body_size; i++) {
                run.output[static_cast<size_t>(produced + i)] = body[i];
            }
            produced += block.body_size;
        } else if (block.block_type == cudec_detail::kZstdBlockTypeRle) {
            if (block.block_size > capacity - produced) {
                Stop(&run, kStageContentSize, 0, CUDEC_ERR_OUTPUT_TOO_SMALL,
                     "the frame regenerated more than the destination");
                return run;
            }
            for (uint64_t i = 0; i < block.block_size; i++) {
                run.output[static_cast<size_t>(produced + i)] = body[0];
            }
            produced += block.block_size;
        } else {
            /* A block may regenerate at most the block maximum, which is what
             * sizes the literals buffer: the section's own bound is checked
             * inside the unit against the same window. */
            const uint64_t block_max =
                cudec_detail::ZstdLiteralsBlockMaximum(header.window_size);
            literals.assign(static_cast<size_t>(block_max), 0);
            uint64_t literals_produced = 0;
            uint64_t consumed = 0;
            cudec_detail::ZstdLiteralsReject literals_rung =
                cudec_detail::kZstdLiteralsRejectNone;
            if (cudec_detail::ZstdDecodeLiterals(
                    body, block.body_size, header.window_size,
                    &driver.literals_table, &driver.literals_scratch,
                    literals.data(), literals.size(), &literals_produced,
                    &consumed, &literals_rung) != CUDEC_OK) {
                Stop(&run, kStageLiterals, static_cast<int>(literals_rung),
                     CUDEC_ERR_CORRUPT_INPUT, "literals section refused");
                return run;
            }
            literals.resize(static_cast<size_t>(literals_produced));

            const unsigned char* section = body + consumed;
            uint64_t remaining = block.body_size - consumed;
            cudec_detail::ZstdSeqSectionHeader seq_header;
            uint64_t seq_consumed = 0;
            cudec_detail::ZstdSeqReject seq_rung =
                cudec_detail::kZstdSeqRejectNone;
            if (cudec_detail::ZstdParseSeqSectionHeader(
                    section, remaining, block_max, &seq_header, &seq_consumed,
                    &seq_rung) != CUDEC_OK) {
                Stop(&run, kStageSequenceHeader, static_cast<int>(seq_rung),
                     CUDEC_ERR_CORRUPT_INPUT, "sequences header refused");
                return run;
            }
            section += seq_consumed;
            remaining -= seq_consumed;

            std::vector<cudec_detail::ZstdSequence> sequences(
                seq_header.sequence_count);
            if (seq_header.sequence_count != 0) {
                const unsigned fields[] = {
                    cudec_detail::kZstdSeqFieldLitLen,
                    cudec_detail::kZstdSeqFieldOffset,
                    cudec_detail::kZstdSeqFieldMatchLen};
                const unsigned modes[] = {seq_header.litlen_mode,
                                          seq_header.offset_mode,
                                          seq_header.matchlen_mode};
                cudec_detail::ZstdSeqTable* targets[] = {
                    &driver.litlen, &driver.offset, &driver.matchlen};
                for (unsigned index = 0; index < 3; index++) {
                    uint64_t table_consumed = 0;
                    if (cudec_detail::ZstdSeqLoadTable(
                            fields[index], modes[index], section, remaining,
                            &driver.seq_scratch, targets[index],
                            &table_consumed, &seq_rung) != CUDEC_OK) {
                        Stop(&run, kStageSequenceTable,
                             static_cast<int>(seq_rung),
                             CUDEC_ERR_CORRUPT_INPUT,
                             "sequence table refused");
                        return run;
                    }
                    section += table_consumed;
                    remaining -= table_consumed;
                }
                if (cudec_detail::ZstdDecodeSequences(
                        section, remaining, seq_header.sequence_count,
                        &driver.litlen, &driver.offset, &driver.matchlen,
                        sequences.data(),
                        static_cast<uint32_t>(sequences.size()),
                        &seq_rung) != CUDEC_OK) {
                    Stop(&run, kStageSequences, static_cast<int>(seq_rung),
                         CUDEC_ERR_CORRUPT_INPUT, "sequences refused");
                    return run;
                }
            }

            /* The repeat-offset history is a serial chain over the frame and
             * is resolved for every sequence before any byte moves, which is
             * the shape the execution below asks for: with the distances in
             * hand each copy is independent of every other. The rungs stay
             * this stage's, so a twin can still say which of the two refused.
             */
            offsets.assign(sequences.size(), 0);
            for (size_t s = 0; s < sequences.size(); s++) {
                const cudec_detail::ZstdSequence& sequence = sequences[s];
                cudec_detail::ZstdRepcodeReject repcode_rung =
                    cudec_detail::kZstdRepcodeRejectNone;
                if (cudec_detail::ZstdRepcodeResolve(
                        &driver.repcodes, sequence.offset_value,
                        sequence.literals_length, &offsets[s],
                        &repcode_rung) != CUDEC_OK) {
                    Stop(&run, kStageRepcode, static_cast<int>(repcode_rung),
                         CUDEC_ERR_CORRUPT_INPUT, "repeat offset refused");
                    return run;
                }
                run.sequences++;
                run.path[ClassifyPath(sequence.offset_value,
                                      sequence.literals_length)]++;
                if (sequence.offset_value <=
                    cudec_detail::kZstdRepcodeMaxSlotValue) {
                    run.repeats++;
                }
            }

            destinations.assign(sequences.size() + 1, 0);
            cudec_detail::ZstdExecPlan plan;
            cudec_detail::ZstdExecReject exec_rung =
                cudec_detail::kZstdExecRejectNone;
            cudec_status exec_status = cudec_detail::ZstdExecPrefixSum(
                sequences.empty() ? 0 : sequences.data(),
                static_cast<uint32_t>(sequences.size()), literals.size(),
                block_max, destinations.data(),
                static_cast<uint32_t>(destinations.size()), &plan, &exec_rung);
            if (exec_status != CUDEC_OK) {
                Stop(&run, kStageExecute, static_cast<int>(exec_rung),
                     exec_status, "sequence destinations refused");
                return run;
            }
            exec_status = cudec_detail::ZstdExecuteBlock(
                sequences.empty() ? 0 : sequences.data(),
                static_cast<uint32_t>(sequences.size()), destinations.data(),
                offsets.empty() ? 0 : offsets.data(),
                literals.empty() ? 0 : literals.data(), literals.size(), &plan,
                header.window_size, run.output.data(), capacity, &produced,
                &exec_rung);
            if (exec_status != CUDEC_OK) {
                Stop(&run, kStageExecute, static_cast<int>(exec_rung),
                     exec_status, "sequence execution refused");
                return run;
            }
        }
        pos += 3 + block.body_size;
        if (block.last_block) {
            break;
        }
    }

    /* Section 3.1.1.1.1: where a content size is declared it is exact, and a
     * frame in this subset always declares one. The buffer was sized to that
     * number, so a frame producing MORE has already been refused for want of
     * room; what is left for this check is the frame that produced less. */
    if (produced != header.frame_content_size) {
        Stop(&run, kStageContentSize, 0, CUDEC_ERR_CORRUPT_INPUT,
             "the frame produced other than its declared content size");
        return run;
    }
    run.output.resize(static_cast<size_t>(produced));

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
