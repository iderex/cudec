/* The loop over one Zstd frame's blocks, and the per-frame state it carries
 * across their boundaries. Single-sourced for host and device, the sibling of
 * src/zstd_frame.h, src/zstd_literals.h, src/zstd_seq.h, src/zstd_repcode.h
 * and src/zstd_exec.h - it composes them and owns nothing they own. Internal
 * header, not part of the ABI.
 *
 * WHAT IS PER-FRAME AND WHAT IS PER-BLOCK IS THE WHOLE CONTRACT. The Huffman
 * table a Treeless literals section reuses, the three FSE tables a Repeat
 * mode reuses, and the repeat-offset history are exactly what survives a
 * block boundary; the literals buffer, the sequences and their destinations
 * are exactly what does not. A loop that reset the first group at a block
 * start would decode the first block of every frame correctly and then
 * produce wrong bytes, silently and only on streams the encoder chose to
 * compress that way. So the reset lives in one place, ZstdFrameStateInit, and
 * this loop never calls it.
 *
 * THE STATE IS THE CALLER'S, ALL OF IT. Every buffer this loop writes is
 * handed in with its capacity, for the reason each unit below gives: a kernel
 * decides where a frame's working memory lives - registers, shared memory, a
 * per-warp slab - and a header that allocated would decide that for it. A
 * capacity too small for the frame in hand is refused before anything is
 * written, never grown.
 *
 * THE DECLARED CONTENT SIZE IS ENFORCED EXACTLY AND IS NEVER USED TO SIZE
 * ANYTHING. Section 3.1.1.1.1: where a content size is declared it is exact,
 * and the v1 subset requires one (docs/MASTERPLAN.md section 12). It is
 * attacker-controlled, so it is compared against the caller's capacity before
 * the loop runs and against what the frame actually produced after it, and
 * the caller's capacity remains the truth in between. A frame producing fewer
 * bytes than it declared is corrupt, and so is one producing more - the
 * second is caught as it happens, because a copy that would pass the
 * declaration is refused rather than measured afterwards.
 *
 * WHERE A RUN STOPPED IS REPORTED, NOT JUST THAT IT DID. Every unit below has
 * its own reject ladder and every rung is that unit's vocabulary, so a stop
 * carries the STAGE alongside the rung: reject parity over a mutation corpus
 * is only worth something if a refusal can be attributed, and a caller
 * comparing two decoders needs to know they refused the same thing rather
 * than merely that both refused. */
#ifndef CUDEC_ZSTD_BLOCKS_H
#define CUDEC_ZSTD_BLOCKS_H

#include "cudec.h"
#include "zstd_exec.h"
#include "zstd_frame.h"
#include "zstd_literals.h"
#include "zstd_repcode.h"
#include "zstd_seq.h"

#include <stdint.h>

/* Guarded: every single-sourced header in this tree defines the same macro
 * for the same reason, and a device translation unit that decodes more than
 * one format includes more than one of them. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* Which stage stopped, named per STAGE rather than per rung: each stage's
 * rungs are the business of that unit's own twin, and what a caller needs
 * from the loop is which stage refused. The content-size stage is this loop's
 * own and has no unit behind it, so its rung is always zero. */
enum ZstdBlocksStage {
    kZstdBlocksStageNone = 0,
    kZstdBlocksStageBlockHeader,
    kZstdBlocksStageLiterals,
    kZstdBlocksStageSequenceHeader,
    kZstdBlocksStageSequenceTable,
    kZstdBlocksStageSequences,
    kZstdBlocksStageRepcode,
    kZstdBlocksStageExecute,
    kZstdBlocksStageContentSize,
    kZstdBlocksStageCount
};

/* This loop's own reject ladder - the refusals no unit below makes, so a
 * negative can name one instead of counting a status that repeats. Every
 * refusal returns through ZstdBlocksRefuse, the way every sibling header
 * routes its own. */
enum ZstdBlocksReject {
    kZstdBlocksRejectNone = 0,
    /* Storage the caller did not supply, or capacities that disagree with
     * each other. A caller bug, refused rather than worked around. */
    kZstdBlocksRejectBadRequest,
    /* The frame declares more content than the destination holds. Refused
     * before a byte is written rather than discovered by a copy, because the
     * declaration is attacker-controlled. */
    kZstdBlocksRejectContentPastCapacity,
    /* A Raw or RLE block would regenerate past what the declared content size
     * leaves. The compressed path reaches the same wall inside
     * ZstdExecuteBlock, which is why only those two are named here. */
    kZstdBlocksRejectBlockPastCapacity,
    /* The frame ended having produced other than its declared content size.
     * Only the short direction reaches this: the long one is refused as it
     * happens, by the rung above and by the execution's own. */
    kZstdBlocksRejectContentSizeMismatch,
    /* More sequences than the caller's sequence storage holds. Separate from
     * BadRequest because it depends on the frame rather than on the call: the
     * same storage is right for one stream and short for the next, and a
     * caller sizing it wants to know which of the two it hit. */
    kZstdBlocksRejectSequenceStorageTooSmall,
    /* Literals storage below the block maximum this frame's window implies.
     * Checked against the window rather than against any section, so the
     * refusal cannot depend on bytes an attacker chose. */
    kZstdBlocksRejectLiteralsStorageTooSmall,
    kZstdBlocksRejectCount
};

CUDEC_HOST_DEVICE inline cudec_status ZstdBlocksRefuse(
    ZstdBlocksReject rung, cudec_status status, ZstdBlocksReject* out) {
    if (out != 0) {
        *out = rung;
    }
    return status;
}

/* Everything one frame's decode carries, in the shape each unit asks for.
 *
 * The first group survives a block boundary and is reset once per frame by
 * ZstdFrameStateInit. The second is per-block working memory: this loop
 * overwrites it every block and reads nothing out of it that a previous block
 * wrote, which is what makes its sizes a function of the block maximum rather
 * than of the frame.
 *
 * `sequences_capacity` is the one a caller has to think about. A block may
 * declare as many sequences as it has room to regenerate, so the size that
 * never refuses a legal frame is the block maximum; a smaller array is legal
 * and refuses a frame needing more, by its own rung, rather than truncating
 * one. */
struct ZstdFrameState {
    /* Carried across blocks. */
    ZstdLiteralsTable literals_table;
    ZstdSeqTable litlen;
    ZstdSeqTable offset;
    ZstdSeqTable matchlen;
    ZstdRepcodeHistory repcodes;

    /* Per-block working memory, all of it the caller's. */
    ZstdLiteralsScratch* literals_scratch;
    ZstdSeqScratch* seq_scratch;
    unsigned char* literals;
    uint64_t literals_capacity;
    ZstdSequence* sequences;
    uint32_t sequences_capacity;
    uint64_t* offsets;
    uint32_t offsets_capacity;
    /* One entry per sequence plus one: ZstdExecPrefixSum's one-past-the-end
     * destination is what the tail of the literals is found through. */
    uint64_t* destinations;
    uint32_t destinations_capacity;
};

/* Frame start, and the only place any of the carried state is reset.
 *
 * The cell arrays and the working buffers are the caller's and are left
 * exactly as handed in; what is reset is the four `present` flags and the
 * repeat-offset history, which is the whole of what "reset at frame start"
 * means. Marking a table absent rather than clearing its cells is deliberate:
 * a Treeless section in a frame's first block must be refused over an array
 * that is allocated and holds the PREVIOUS frame's table, and only the flag
 * can tell those two apart. */
CUDEC_HOST_DEVICE inline void ZstdFrameStateInit(ZstdFrameState* state) {
    if (state == 0) {
        return;
    }
    state->literals_table.table_log = 0;
    state->literals_table.present = false;
    state->litlen.table_size = 0;
    state->litlen.accuracy_log = 0;
    state->litlen.present = false;
    state->offset.table_size = 0;
    state->offset.accuracy_log = 0;
    state->offset.present = false;
    state->matchlen.table_size = 0;
    state->matchlen.accuracy_log = 0;
    state->matchlen.present = false;
    ZstdRepcodeInit(&state->repcodes);
}

/* Where a run stopped, and what it counted on the way.
 *
 * The tallies are not decoration. A corpus that decodes byte-identically
 * proves nothing about a rule it never reached, so a caller measuring
 * coverage needs the loop to say which of the repeat-offset rules a stream
 * actually exercised; nothing downstream can recover it, because the
 * sequences are overwritten block by block. A caller that does not care
 * passes no report and the loop writes nothing. */
struct ZstdBlocksReport {
    ZstdBlocksStage stage;
    /* The stage's own rung, as an int because each stage enumerates its own
     * and the caller that cares knows which. */
    int rung;
    uint64_t blocks;
    uint64_t sequences;
    uint64_t repeats;
    uint64_t path[kZstdRepcodePathCount];
};

CUDEC_HOST_DEVICE inline void ZstdBlocksReportInit(ZstdBlocksReport* report) {
    if (report == 0) {
        return;
    }
    report->stage = kZstdBlocksStageNone;
    report->rung = 0;
    report->blocks = 0;
    report->sequences = 0;
    report->repeats = 0;
    for (unsigned index = 0; index < kZstdRepcodePathCount; index++) {
        report->path[index] = 0;
    }
}

CUDEC_HOST_DEVICE inline void ZstdBlocksStop(ZstdBlocksReport* report,
                                             ZstdBlocksStage stage, int rung) {
    if (report == 0) {
        return;
    }
    report->stage = stage;
    report->rung = rung;
}

/* One Compressed block: its literals section, its sequences section, the
 * repeat-offset resolution over them and the copies that execute them.
 *
 * Split out of the loop below only because the loop is a walk and this is a
 * decode; the two have different reasons to be read. `produced` is in and
 * out, the frame's running total, and it is advanced only by a copy that
 * completed - ZstdExecuteBlock leaves it exactly where it was on any refusal,
 * so a partially written block is never reported as a partial success.
 *
 * `capacity` is the frame's declared content size rather than the caller's
 * buffer, which is larger or equal. Handing the smaller of the two down is
 * what makes a block that regenerates past the declaration refuse here rather
 * than at the end of the frame, and the difference matters: at the end, the
 * bytes are already written. */
CUDEC_HOST_DEVICE inline cudec_status ZstdDecodeCompressedBlock(
    const unsigned char* body, uint64_t body_size,
    const ZstdFrameHeader* header, ZstdFrameState* state, uint64_t block_max,
    unsigned char* dst, uint64_t capacity, uint64_t* produced,
    ZstdBlocksReport* report, ZstdBlocksReject* reject) {
    uint64_t literals_size = 0;
    uint64_t consumed = 0;
    ZstdLiteralsReject literals_rung = kZstdLiteralsRejectNone;
    const cudec_status literals_status = ZstdDecodeLiterals(
        body, body_size, header->window_size, &state->literals_table,
        state->literals_scratch, state->literals, state->literals_capacity,
        &literals_size, &consumed, &literals_rung);
    if (literals_status != CUDEC_OK) {
        ZstdBlocksStop(report, kZstdBlocksStageLiterals,
                       static_cast<int>(literals_rung));
        return literals_status;
    }

    const unsigned char* section = body + consumed;
    uint64_t remaining = body_size - consumed;
    ZstdSeqSectionHeader seq_header;
    uint64_t seq_consumed = 0;
    ZstdSeqReject seq_rung = kZstdSeqRejectNone;
    cudec_status status = ZstdParseSeqSectionHeader(
        section, remaining, block_max, &seq_header, &seq_consumed, &seq_rung);
    if (status != CUDEC_OK) {
        ZstdBlocksStop(report, kZstdBlocksStageSequenceHeader,
                       static_cast<int>(seq_rung));
        return status;
    }
    section += seq_consumed;
    remaining -= seq_consumed;

    /* Checked before the tables are read rather than before the loop that
     * fills the array, because reading three table descriptions into a frame
     * whose sequences will not fit is work spent on a refusal already
     * decided. */
    if (seq_header.sequence_count > state->sequences_capacity) {
        ZstdBlocksStop(report, kZstdBlocksStageSequences, 0);
        return ZstdBlocksRefuse(kZstdBlocksRejectSequenceStorageTooSmall,
                                CUDEC_ERR_OUTPUT_TOO_SMALL, reject);
    }

    if (seq_header.sequence_count != 0) {
        /* The three fields in the order the section writes their
         * descriptions, section 3.1.1.3.2.1. Repeat_Mode reads the table this
         * state carried from an earlier block and is refused where none was
         * carried; that refusal is ZstdSeqLoadTable's, over the `present`
         * flag ZstdFrameStateInit cleared at frame start. */
        const unsigned fields[3] = {kZstdSeqFieldLitLen, kZstdSeqFieldOffset,
                                    kZstdSeqFieldMatchLen};
        const unsigned modes[3] = {seq_header.litlen_mode,
                                   seq_header.offset_mode,
                                   seq_header.matchlen_mode};
        ZstdSeqTable* targets[3] = {&state->litlen, &state->offset,
                                    &state->matchlen};
        for (unsigned index = 0; index < 3; index++) {
            uint64_t table_consumed = 0;
            status = ZstdSeqLoadTable(fields[index], modes[index], section,
                                      remaining, state->seq_scratch,
                                      targets[index], &table_consumed,
                                      &seq_rung);
            if (status != CUDEC_OK) {
                ZstdBlocksStop(report, kZstdBlocksStageSequenceTable,
                               static_cast<int>(seq_rung));
                return status;
            }
            section += table_consumed;
            remaining -= table_consumed;
        }
        status = ZstdDecodeSequences(section, remaining,
                                     seq_header.sequence_count, &state->litlen,
                                     &state->offset, &state->matchlen,
                                     state->sequences,
                                     state->sequences_capacity, &seq_rung);
        if (status != CUDEC_OK) {
            ZstdBlocksStop(report, kZstdBlocksStageSequences,
                           static_cast<int>(seq_rung));
            return status;
        }
    }

    /* The repeat-offset history is a serial chain over the whole frame and is
     * resolved for every sequence before any byte moves, which is the shape
     * the execution below asks for: with the distances in hand each copy is
     * independent of every other. */
    for (uint32_t index = 0; index < seq_header.sequence_count; index++) {
        const ZstdSequence sequence = state->sequences[index];
        ZstdRepcodeReject repcode_rung = kZstdRepcodeRejectNone;
        status = ZstdRepcodeResolve(&state->repcodes, sequence.offset_value,
                                    sequence.literals_length,
                                    &state->offsets[index], &repcode_rung);
        if (status != CUDEC_OK) {
            ZstdBlocksStop(report, kZstdBlocksStageRepcode,
                           static_cast<int>(repcode_rung));
            return status;
        }
        if (report != 0) {
            report->sequences++;
            report->path[ZstdRepcodeClassify(sequence.offset_value,
                                             sequence.literals_length)]++;
            if (sequence.offset_value <= kZstdRepcodeMaxSlotValue) {
                report->repeats++;
            }
        }
    }

    ZstdExecPlan plan;
    ZstdExecReject exec_rung = kZstdExecRejectNone;
    status = ZstdExecPrefixSum(state->sequences, seq_header.sequence_count,
                               literals_size, block_max, state->destinations,
                               seq_header.sequence_count + 1, &plan,
                               &exec_rung);
    if (status != CUDEC_OK) {
        ZstdBlocksStop(report, kZstdBlocksStageExecute,
                       static_cast<int>(exec_rung));
        return status;
    }
    status = ZstdExecuteBlock(state->sequences, seq_header.sequence_count,
                              state->destinations, state->offsets,
                              state->literals, literals_size, &plan,
                              header->window_size, dst, capacity, produced,
                              &exec_rung);
    if (status != CUDEC_OK) {
        ZstdBlocksStop(report, kZstdBlocksStageExecute,
                       static_cast<int>(exec_rung));
        return status;
    }
    return CUDEC_OK;
}

/* Decodes one frame's blocks, first to last, into `dst`.
 *
 * `src` points at the first block header - the frame header is
 * ZstdParseFrameHeader's and is already parsed into `header` - and `size` is
 * the bytes left in the frame from there, any trailer included.
 * `out_consumed` is what the blocks occupied, which is where a content
 * checksum trailer begins; the trailer itself is
 * ZstdVerifyContentChecksum's.
 *
 * `state` carries the entropy tables and the repeat-offset history across the
 * blocks and must have been through ZstdFrameStateInit for THIS frame. The
 * loop never resets it, which is the contract, and cannot check that it was
 * reset: a caller handing over a previous frame's state gets that frame's
 * tables, which is why the reset has exactly one home.
 *
 * `out_produced` is the bytes written, and on every success it equals the
 * declared content size. */
CUDEC_HOST_DEVICE inline cudec_status ZstdDecodeBlocks(
    const unsigned char* src, uint64_t size, const ZstdFrameHeader* header,
    ZstdFrameState* state, unsigned char* dst, uint64_t dst_capacity,
    uint64_t* out_produced, uint64_t* out_consumed, ZstdBlocksReport* report,
    ZstdBlocksReject* reject) {
    if (reject != 0) {
        *reject = kZstdBlocksRejectNone;
    }
    ZstdBlocksReportInit(report);
    if (out_produced != 0) {
        *out_produced = 0;
    }
    if (out_consumed != 0) {
        *out_consumed = 0;
    }
    if (src == 0 || header == 0 || state == 0 || dst == 0 ||
        out_produced == 0 || out_consumed == 0 ||
        state->literals_scratch == 0 || state->seq_scratch == 0 ||
        state->literals == 0 || state->destinations == 0 ||
        state->destinations_capacity == 0 ||
        state->offsets_capacity < state->sequences_capacity ||
        state->destinations_capacity - 1u < state->sequences_capacity ||
        (state->sequences == 0 && state->sequences_capacity != 0) ||
        (state->offsets == 0 && state->offsets_capacity != 0)) {
        return ZstdBlocksRefuse(kZstdBlocksRejectBadRequest,
                                CUDEC_ERR_INVALID_ARGUMENT, reject);
    }

    /* The block maximum this frame's window implies, section 3.1.1.2.4. It is
     * the ceiling on everything below and is derived from the header rather
     * than from any block's own declaration, so a hostile block cannot move
     * it. */
    const uint64_t block_max = ZstdLiteralsBlockMaximum(header->window_size);
    if (state->literals_capacity < block_max) {
        return ZstdBlocksRefuse(kZstdBlocksRejectLiteralsStorageTooSmall,
                                CUDEC_ERR_INVALID_ARGUMENT, reject);
    }

    /* THE DECLARED SIZE IS CHECKED AGAINST THE CAPACITY, NEVER USED AS ONE.
     * The frame says how much it will regenerate and the caller says how much
     * room there is; the smaller is not silently taken, because a frame
     * declaring more than fits is refused rather than truncated. */
    const uint64_t declared = header->frame_content_size;
    if (declared > dst_capacity) {
        ZstdBlocksStop(report, kZstdBlocksStageContentSize, 0);
        return ZstdBlocksRefuse(kZstdBlocksRejectContentPastCapacity,
                                CUDEC_ERR_OUTPUT_TOO_SMALL, reject);
    }

    uint64_t produced = 0;
    uint64_t pos = 0;
    for (;;) {
        ZstdBlockHeader block;
        ZstdFrameReject frame_rung = kZstdFrameRejectNone;
        const cudec_status block_status = ZstdParseBlockHeader(
            src + pos, size - pos, header->window_size, &block, &frame_rung);
        if (block_status != CUDEC_OK) {
            ZstdBlocksStop(report, kZstdBlocksStageBlockHeader,
                           static_cast<int>(frame_rung));
            return block_status;
        }
        if (report != 0) {
            report->blocks++;
        }
        const unsigned char* body = src + pos + 3;

        if (block.block_type == kZstdBlockTypeRaw ||
            block.block_type == kZstdBlockTypeRle) {
            /* Both regenerate their declared size and differ only in where
             * the bytes come from; ZstdParseBlockHeader already separated
             * that size from the one byte an RLE block occupies. Bounded in
             * the subtraction direction against what the declaration leaves,
             * so nothing can wrap. */
            const uint64_t regenerated = block.block_size;
            if (regenerated > declared - produced) {
                ZstdBlocksStop(report, kZstdBlocksStageContentSize, 0);
                return ZstdBlocksRefuse(kZstdBlocksRejectBlockPastCapacity,
                                        CUDEC_ERR_OUTPUT_TOO_SMALL, reject);
            }
            const bool raw = block.block_type == kZstdBlockTypeRaw;
            for (uint64_t index = 0; index < regenerated; index++) {
                dst[produced + index] = raw ? body[index] : body[0];
            }
            produced += regenerated;
        } else {
            const cudec_status status = ZstdDecodeCompressedBlock(
                body, block.body_size, header, state, block_max, dst, declared,
                &produced, report, reject);
            if (status != CUDEC_OK) {
                return status;
            }
        }

        pos += static_cast<uint64_t>(3) + block.body_size;
        if (block.last_block) {
            break;
        }
    }

    /* Section 3.1.1.1.1 from the other side. Every path that would have
     * produced MORE than the declaration was refused where it happened, so
     * what is left for this check is the frame that produced less - the one
     * that would otherwise return a short buffer as a success. */
    if (produced != declared) {
        ZstdBlocksStop(report, kZstdBlocksStageContentSize, 0);
        return ZstdBlocksRefuse(kZstdBlocksRejectContentSizeMismatch,
                                CUDEC_ERR_CORRUPT_INPUT, reject);
    }

    *out_produced = produced;
    *out_consumed = pos;
    return CUDEC_OK;
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_BLOCKS_H */
