/* The block-per-frame Zstd decoder (issue #203): one 128-thread block per
 * frame over a grid-stride loop, one entropy table set per frame in shared
 * memory, the literals staged in the tail of the frame's own remaining
 * destination, and the sequences produced and consumed a tile at a time so
 * the frame needs no workspace beyond the caller's buffer
 * (docs/MASTERPLAN.md sections 14.1 to 14.4). Internal header, not part of
 * the ABI; src/batch.cu instantiates it.
 *
 * NOTHING IN THIS FILE DECODES A BIT. Every value is produced by a unit under
 * src/zstd_*.h that the CPU twins already run, compiled __device__ unchanged:
 * the frame and block headers, the literals section, the Huffman and FSE
 * machinery, the resumable sequence cursor, the tiled prefix sum, the
 * per-sequence execution and the content checksum. What this file owns is the
 * placement - which memory each unit is handed - and the order they run in,
 * which is the order src/zstd_blocks.h runs them in on the host. So a frame
 * that decodes to different bytes on the two residencies is a defect here,
 * and the units cannot be that defect: they are the same code.
 *
 * WHERE THE LITERALS LIVE, AND WHY THAT IS THE WHOLE REASON THERE IS NO
 * WORKSPACE. Section 14.3: with R bytes of declared output left at the start
 * of a block and L literal bytes regenerated, the literals are decoded into
 * [P + R - L, P + R) and every sequence's literal run then copies toward
 * LOWER addresses, so run i ends at or before run i+1's source begins. The
 * host twin decodes the same literals into a buffer of its own; the bytes are
 * identical and only the address differs.
 *
 * WHAT THIS LANDING DOES NOT TAKE FROM THE LANE MAP. Sections 14.7 to 14.11
 * assign the literal streams to four lanes of warp 0 and one sequence of a
 * tile to each of the 128 threads. Neither is taken here and both are
 * recorded rather than silently dropped:
 *
 *   - The four literal streams are one call of ZstdDecodeLiterals, because
 *     the four spans are derived inside that unit. Splitting them across
 *     lanes needs the span derivation extracted into a function the host twin
 *     still runs, which is 14.9's own pattern and is not this rung's work.
 *   - The sequences of a tile execute IN ORDER on one lane. The ruling of
 *     2026-09-04 on #203 forbids fanning a tile's match copies across lanes -
 *     over the #185 corpus 60616 of 124212 matches read bytes a lower-numbered
 *     sequence of their own tile writes, the nearest at an offset of two. What
 *     the ruling permits, a parallel literal phase ahead of a serial match
 *     phase, would need ZstdExecuteSequence split in two, and the split
 *     reorders the reject ladder: all the literal rungs of a tile would fire
 *     ahead of the match rung of a lower-numbered sequence, so the device
 *     would report a different refusal from the host twin on the same bytes.
 *
 * So the parallelism this kernel takes is the copies that are copies rather
 * than format units: a Raw or RLE block's body and a block's leftover literal
 * tail move across all 128 threads. The rest of a frame runs on thread 0, and
 * what that costs is a measurement rather than a claim - #231 records the
 * first baselines and #235 to #239 are the levers that would widen it.
 *
 * EVERY THREAD-0 SECTION IS PRECEDED BY A BARRIER, AND THAT IS A RULE RATHER
 * THAN A HABIT. The block's control flow is decided on flags in shared memory
 * that thread 0 writes and all 128 threads read, so a flag read after the
 * barrier that published it is still racing the NEXT write of it: thread 0 is
 * free to run ahead the moment a barrier releases, and a lagging warp then
 * reads the following block's `last_block` or the following call's `status`.
 * What that produces is not a wrong byte but a SPLIT BLOCK - some warps break
 * out of the loop or return while the others go on - and from there the
 * barriers below are reached by a subset of the block. Measured on this
 * kernel before the rule was applied: a legal two-block frame answered
 * CUDEC_OK with bytes_written set and 380 of its 1024 bytes never written by
 * anybody, because three of four warps had left. The barrier in front of each
 * thread-0 section is what stops the writer from moving until every reader is
 * done with the previous value, and tests/CMakeLists.txt refuses a section in
 * this file that does not have one.
 *
 * FAIL-CLOSED, WITH THE BOUNDS THE UNITS ALREADY CARRY. Every write is
 * bounded before it happens by the frame's declared content size, which is
 * bounded by the caller's capacity before the first block runs. On any
 * refusal the frame's result is a non-OK status with bytes_written zero, and
 * the destination is unspecified and never presented as a decode. */
#ifndef CUDEC_ZSTD_DECODE_CUH
#define CUDEC_ZSTD_DECODE_CUH

#include "cudec.h"
#include "xxhash64.h"
#include "zstd_blocks.h"

#include "vendor_rt.h"

namespace cudec_detail {

/* Section 14.2: the smallest block at which the table set stops being the
 * resource that decides how many frames an SM holds. It is a thread count and
 * not a wave count, so it does not move with the wave width; what the width
 * decides is how many waves that is, which the static assertion in the kernel
 * holds to a whole number. */
constexpr unsigned kZstdBlockThreads = 128;

/* The table set of 14.1, sized by the format's own ceilings and by nothing a
 * frame says. */
constexpr uint32_t kZstdLitLenCells = 1u << kZstdLitLenAccuracyLogMax;
constexpr uint32_t kZstdMatchLenCells = 1u << kZstdMatchLenAccuracyLogMax;
constexpr uint32_t kZstdOffsetCells = 1u << kZstdOffsetAccuracyLogMax;
constexpr uint32_t kZstdHufTableCells = 1u << kZstdLiteralsMaxTableLog;

/* Sequences per tile.
 *
 * SECTION 14.10 BUDGETS 128 AND THIS LANDING TAKES THE HALVING THAT SECTION
 * ITSELF NAMES AS THE FIRST MOVE. Its arithmetic costs a tile record at five
 * 32-bit fields, twenty bytes; the record here is what the units actually
 * read and write - a ZstdSequence, a uint64 destination, a uint64 literal
 * cursor and a uint64 resolved offset - which is forty. Packing them to the
 * budgeted twenty would mean a narrowed copy live beside the arrays the units
 * fill, which costs more than it saves rather than less. At sixty-four
 * records the shared footprint is under 12.4 KiB, so the eight resident
 * blocks 14.2 derives still fit inside the roughly 100 KiB an sm_86 SM
 * offers, with room left for whatever the driver reserves. */
constexpr uint32_t kZstdTileSequences = 64;

/* The per-block working memory of a table description. The literals section
 * finishes before the first sequence table is read - that is the order
 * src/zstd_blocks.h runs them in - so one region serves both and the union
 * says so rather than two regions sitting idle in turn. */
union ZstdBuildScratch {
    ZstdLiteralsScratch literals;
    ZstdSeqScratch seq;
};

/* One tile of sequences and everything the execution needs per sequence.
 * `offsets` holds the resolved distance the repeat-offset chain produced,
 * which is what lets each copy be independent of the chain that made it. */
struct ZstdTile {
    ZstdSequence sequences[kZstdTileSequences];
    uint64_t destinations[kZstdTileSequences];
    uint64_t literal_cursors[kZstdTileSequences];
    uint64_t offsets[kZstdTileSequences];
};

/* Everything one block holds for the frame it is decoding.
 *
 * The scratch and the tile overlay each other for the reason 14.10 gives: the
 * tile holds nothing until the sequences of a block start arriving, and by
 * then every description that needed scratch has been read. */
struct ZstdFrameShared {
    ZstdFseCell litlen_cells[kZstdLitLenCells];
    ZstdFseCell matchlen_cells[kZstdMatchLenCells];
    ZstdFseCell offset_cells[kZstdOffsetCells];
    ZstdHufCell huf_cells[kZstdHufTableCells];
    union {
        ZstdBuildScratch build;
        ZstdTile tile;
    } work;

    ZstdFrameState state;
    ZstdFrameHeader header;

    /* The walk, kept where every thread reads it: the barriers below are what
     * make thread 0's writes visible to the rest, and a value recomputed per
     * thread instead would be a second spelling of the walk. */
    cudec_status status;
    uint64_t produced;
    uint64_t pos;
    /* What the block header declared, which is how the walk steps over the
     * block; never the count of the copy below, which is a different
     * quantity and once shared this field. */
    uint64_t body_size;
    /* The one copy the whole block makes for the block being decoded: a Raw
     * or RLE body, or the leftover literal tail of a Compressed one. */
    uint64_t copy_count;
    const unsigned char* copy_src;
    unsigned char* copy_dst;
    bool copy_is_rle;
    bool compressed;
    bool last_block;
};

/* One block per frame, capped where the chunk decoder caps its own grid and
 * for the same reason: a grid larger than the device will ever run resident
 * buys nothing, and the loop inside the kernel is what covers the rest. */
inline unsigned zstd_grid_blocks(size_t chunk_count) {
    constexpr size_t kMaxBlocks = 8192;
    const size_t blocks = chunk_count == 0 ? 1 : chunk_count;
    return static_cast<unsigned>(blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

/* A forward copy of `count` bytes whose destination is at or below its
 * source, spread over the whole block.
 *
 * WHY THE BARRIER IS INSIDE THE LOOP. With the destination below the source
 * the regions may overlap, and a thread writing dst[i] can be clobbering the
 * byte another thread has not read yet. Reading a chunk into registers,
 * meeting at a barrier and only then writing it is what removes that race:
 * inside a chunk every read precedes every write, and across chunks the
 * writes of chunk k end at or below the reads of chunk k+1 because the
 * destination is at or below the source. Serial byte order would also be
 * safe and is what the host twin does; this is the same move, held to the
 * same rule, at 128 bytes a step. */
__device__ inline void ZstdMoveDown(unsigned char* dst,
                                    const unsigned char* src, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(kZstdBlockThreads);
    const uint64_t steps = (count + stride - 1) / stride;
    for (uint64_t step = 0; step < steps; step++) {
        const uint64_t at = step * stride + threadIdx.x;
        unsigned char byte = 0;
        const bool live = at < count;
        if (live) {
            byte = src[at];
        }
        __syncthreads();
        if (live) {
            dst[at] = byte;
        }
        __syncthreads();
    }
}

/* A Raw or RLE block's body, spread over the whole block. No overlap is
 * possible here - the source is the compressed frame and the destination is
 * the caller's output - so this needs no barrier of its own. */
__device__ inline void ZstdFillBlock(unsigned char* dst,
                                     const unsigned char* src, uint64_t count,
                                     bool rle) {
    const uint64_t stride = static_cast<uint64_t>(kZstdBlockThreads);
    for (uint64_t at = threadIdx.x; at < count; at += stride) {
        dst[at] = rle ? src[0] : src[at];
    }
}

/* One Compressed block, on thread 0, in the order src/zstd_blocks.h runs the
 * same units in: the literals section, the sequences section header, the
 * three table descriptions, then the sequences a tile at a time.
 *
 * `remaining` is what the frame has left to declare, so every bound below is
 * the caller's capacity narrowed by what earlier blocks produced.
 *
 * THE EXECUTION IS HANDED `remaining` WHERE THE HOST HANDS IT THE BLOCK'S OWN
 * SIZE, AND WHAT MAKES THAT SOUND IS THE CHECK IN THE TILE LOOP. A block's
 * size is the running total plus the literals its sequences do not consume,
 * so it is known only once the last tile has been summed - and the tiles have
 * to execute as they are produced, because holding them all is the workspace
 * 14.3 refuses. What the loop does instead is hold the size-so-far to
 * `remaining` after each tile's prefix sum and BEFORE that tile's copies, so
 * by the time a sequence runs, the bound the host would have given it is
 * already known to be at least as tight as `remaining`. On a block that
 * decodes the two are the same number; on one that does not, the refusal
 * comes from that comparison with the class the host gives, rather than from
 * a rung inside the execution unit that no stream is supposed to reach. */
__device__ inline cudec_status ZstdDecodeCompressedBlockDevice(
    ZstdFrameShared* frame, const unsigned char* body, uint64_t body_size,
    unsigned char* dst, uint64_t remaining) {
    ZstdFrameState* state = &frame->state;
    const uint64_t block_max = ZstdLiteralsBlockMaximum(
        frame->header.window_size);

    /* Where the literals go, which is what the whole no-workspace shape rests
     * on. The regenerated size is read from the header before the section is
     * decoded so the tail address is known before a byte is written; the
     * header is then read a second time by the decode itself, which is a pure
     * function of the same bytes and cannot disagree with the first reading. */
    ZstdLiteralsHeader literals_header;
    ZstdLiteralsReject literals_rung = kZstdLiteralsRejectNone;
    cudec_status status = ZstdParseLiteralsHeader(body, body_size,
                                                  &literals_header,
                                                  &literals_rung);
    if (status != CUDEC_OK) {
        return status;
    }
    const uint64_t literals_size = literals_header.regenerated_size;
    if (literals_size > remaining) {
        /* A block regenerates at least its own literals, so a literals
         * section larger than the frame has left to declare is a block whose
         * size passes the declaration. Refused before the address below is
         * formed, which is what keeps the subtraction from wrapping.
         *
         * THE STATUS IS THE HOST'S. The host reaches the same bytes through
         * ZstdExecuteBlock's block-past-declaration rung, and
         * `plan.block_size >= literals_size` is what makes the two rungs the
         * same statement: a frame whose blocks pass its own declaration is
         * malformed, not short of room (section 12.3, #460). A section past
         * the block maximum the window implies is refused by
         * ZstdDecodeLiterals below in the same class, exactly where the host
         * refuses it, so there is no order between the two bounds for this
         * function to keep - which is why it no longer compares the block
         * maximum itself. */
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    unsigned char* literals = dst + frame->produced + remaining - literals_size;

    uint64_t consumed = 0;
    uint64_t produced_literals = 0;
    status = ZstdDecodeLiterals(body, body_size, frame->header.window_size,
                                &state->literals_table,
                                &frame->work.build.literals, literals,
                                literals_size, &produced_literals, &consumed,
                                &literals_rung);
    if (status != CUDEC_OK) {
        return status;
    }

    const unsigned char* section = body + consumed;
    uint64_t section_size = body_size - consumed;
    ZstdSeqSectionHeader seq_header;
    uint64_t seq_consumed = 0;
    ZstdSeqReject seq_rung = kZstdSeqRejectNone;
    status = ZstdParseSeqSectionHeader(section, section_size, block_max,
                                       &seq_header, &seq_consumed, &seq_rung);
    if (status != CUDEC_OK) {
        return status;
    }
    section += seq_consumed;
    section_size -= seq_consumed;

    /* The sequence count is NOT compared against a capacity here, and its
     * absence is the tile. The host loop refuses a count larger than the
     * array it materialises; this one holds sixty-four sequences whatever the
     * block declares, so the only bound the count needs is the one
     * ZstdParseSeqSectionHeader already applied against the block maximum. */
    if (seq_header.sequence_count != 0) {
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
                                      section_size, &frame->work.build.seq,
                                      targets[index], &table_consumed,
                                      &seq_rung);
            if (status != CUDEC_OK) {
                return status;
            }
            section += table_consumed;
            section_size -= table_consumed;
        }
    }

    ZstdExecCarry carry;
    carry.at = 0;
    carry.literals_used = 0;
    ZstdTile* tile = &frame->work.tile;

    if (seq_header.sequence_count != 0) {
        ZstdSeqCursor cursor;
        status = ZstdSeqBegin(section, section_size, seq_header.sequence_count,
                              &state->litlen, &state->offset, &state->matchlen,
                              &cursor, &seq_rung);
        if (status != CUDEC_OK) {
            return status;
        }
        /* The trip count is derived from the declared sequence count before
         * the first tile runs, so the walk cannot be extended by anything the
         * bitstream says. */
        const uint32_t tiles =
            (seq_header.sequence_count + kZstdTileSequences - 1) /
            kZstdTileSequences;
        uint32_t done = 0;
        for (uint32_t tile_index = 0; tile_index < tiles; tile_index++) {
            uint32_t want = seq_header.sequence_count - done;
            if (want > kZstdTileSequences) {
                want = kZstdTileSequences;
            }
            uint32_t got = 0;
            status = ZstdSeqDecodeTile(&cursor, tile->sequences,
                                       kZstdTileSequences, want, &got,
                                       &seq_rung);
            if (status != CUDEC_OK) {
                return status;
            }

            /* The repeat-offset chain is serial over the whole frame and is
             * resolved for the tile before any of its bytes move, which is
             * what makes each copy below independent of the chain. */
            for (uint32_t index = 0; index < got; index++) {
                ZstdRepcodeReject repcode_rung = kZstdRepcodeRejectNone;
                status = ZstdRepcodeResolve(&state->repcodes,
                                            tile->sequences[index].offset_value,
                                            tile->sequences[index].literals_length,
                                            &tile->offsets[index],
                                            &repcode_rung);
                if (status != CUDEC_OK) {
                    return status;
                }
            }

            ZstdExecReject exec_rung = kZstdExecRejectNone;
            status = ZstdExecPrefixSumTile(tile->sequences, got, literals_size,
                                           block_max, &carry,
                                           tile->destinations,
                                           tile->literal_cursors,
                                           kZstdTileSequences, &exec_rung);
            if (status != CUDEC_OK) {
                return status;
            }

            /* THE ONE CHECK THAT MAKES THE WHOLE NO-WORKSPACE SHAPE SAFE,
             * AND IT RUNS BEFORE THE COPIES RATHER THAN AFTER THEM.
             *
             * `carry.at + (literals_size - carry.literals_used)` is the
             * block's size as far as the sum has got: the bytes the
             * sequences so far produce, plus every literal they have not
             * consumed. It is the same quantity ZstdExecPrefixSumFinish
             * returns as `block_size`, taken at a tile boundary, and it only
             * grows - each further sequence adds its match bytes and moves
             * literals from the tail into the runs.
             *
             * Held to `remaining`, it establishes 14.3's placement invariant
             * for every sequence of this tile before one byte moves. The
             * literals sit at [P + R - L, P + R) and sequence i's run copies
             * from `literals + literal_at_i` to `base + at_i`, so the copy
             * moves toward lower addresses exactly when the match bytes
             * before it fit in `R - L` - which is what this comparison says.
             * Checked after the tile's copies instead, a block that violated
             * it would overwrite literals it has not read yet, and only the
             * refusal at the end of the block would stop that garbage being
             * reported. It stays inside the caller's buffer either way, but
             * "inside the buffer and refused afterwards" is not the property
             * this shape claims.
             *
             * It is also what keeps ZstdExecuteSequence's plan-consistency
             * rung out of reach of a stream. That rung is the unit's
             * caller-bug rung, documented as reachable by a wrong scan and
             * never by a stream, and handing the execution `remaining`
             * instead of the block's own size would otherwise let a hostile
             * frame fire it - mis-filing a stream rejection as a decoder bug
             * in anything that triages by rung. The class is the host's
             * block-past-declaration answer, CORRUPT_INPUT (section 12.3):
             * the bound being passed is the frame's own declaration, which
             * no larger destination repairs. */
            if (carry.at > remaining ||
                literals_size - carry.literals_used > remaining - carry.at) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }

            for (uint32_t index = 0; index < got; index++) {
                /* The successor of the tile's last sequence is where the sum
                 * has got to, which is the destination the next tile's first
                 * sequence will be given. */
                const uint64_t next = index + 1 < got
                                          ? tile->destinations[index + 1]
                                          : carry.at;
                status = ZstdExecuteSequence(
                    &tile->sequences[index], tile->destinations[index], next,
                    tile->offsets[index], literals, literals_size,
                    tile->literal_cursors[index], remaining,
                    frame->header.window_size, dst, frame->produced,
                    &exec_rung);
                if (status != CUDEC_OK) {
                    return status;
                }
            }
            done += got;
        }
    }

    ZstdExecPlan plan;
    ZstdExecReject exec_rung = kZstdExecRejectNone;
    uint64_t tail_at = 0;
    status = ZstdExecPrefixSumFinish(&carry, literals_size, block_max,
                                     &tail_at, &plan, &exec_rung);
    if (status != CUDEC_OK) {
        return status;
    }
    /* The same comparison the tile loop makes, at the close. It is not a
     * second guard for the tiles - each of those was already held to it - it
     * is the one for a block with no sequences at all, whose whole size is
     * its literals and which never enters the loop above. */
    if (plan.block_size > remaining) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    frame->copy_count = literals_size - plan.literals_used;
    frame->copy_src = literals + plan.literals_used;
    frame->copy_dst = dst + frame->produced + tail_at;
    frame->produced += plan.block_size;
    return CUDEC_OK;
}

/* One frame, by the whole block. Thread 0 walks it and the rest meet it at
 * the barriers; every value the walk produces that another thread reads goes
 * through shared memory, so there is one walk and not 128 of them. */
__device__ inline void ZstdDecodeFrameDevice(ZstdFrameShared* frame,
                                             const unsigned char* src,
                                             uint64_t size, unsigned char* dst,
                                             uint64_t capacity) {
    __syncthreads();
    if (threadIdx.x == 0) {
        ZstdFrameReject frame_rung = kZstdFrameRejectNone;
        frame->produced = 0;
        frame->pos = 0;
        frame->last_block = false;
        frame->status = ZstdParseFrameHeader(src, size, &frame->header,
                                             &frame_rung);
        if (frame->status == CUDEC_OK &&
            frame->header.frame_content_size > capacity) {
            /* THE DECLARED SIZE IS CHECKED AGAINST THE CAPACITY, NEVER USED
             * AS ONE - src/zstd_blocks.h's rule, applied here because this
             * kernel runs the block loop itself. */
            frame->status = CUDEC_ERR_OUTPUT_TOO_SMALL;
        }
        if (frame->status == CUDEC_OK) {
            frame->pos = frame->header.header_size;
            ZstdFrameStateInit(&frame->state);
        }
    }
    __syncthreads();
    if (frame->status != CUDEC_OK) {
        return;
    }

    const uint64_t declared = frame->header.frame_content_size;
    /* The termination fuel of src/zstd_blocks.h, for the reason it gives:
     * every block consumes at least its three header bytes, so a frame of
     * `size` bytes admits at most that many steps and the step after the last
     * one a legal frame could take finds no three bytes to read. A bounds
     * defect becomes a rejected frame rather than a block that never
     * retires. */
    const uint64_t fuel = size / 3 + 1;
    for (uint64_t step = 0; step < fuel; step++) {
        __syncthreads();
        if (threadIdx.x == 0) {
            ZstdBlockHeader block;
            ZstdFrameReject frame_rung = kZstdFrameRejectNone;
            frame->status = ZstdParseBlockHeader(src + frame->pos,
                                                 size - frame->pos,
                                                 frame->header.window_size,
                                                 &block, &frame_rung);
            frame->copy_count = 0;
            frame->copy_is_rle = false;
            frame->copy_src = src;
            frame->copy_dst = dst;
            frame->compressed = false;
            if (frame->status == CUDEC_OK) {
                frame->body_size = block.body_size;
                frame->last_block = block.last_block;
                frame->compressed =
                    block.block_type == kZstdBlockTypeCompressed;
                if (!frame->compressed) {
                    /* Raw and RLE both regenerate their declared size and
                     * differ only in where the bytes come from. Bounded in
                     * the subtraction direction against what the declaration
                     * leaves, so nothing can wrap. The class is the host
                     * loop's: a block past the frame's own declaration is
                     * CORRUPT_INPUT, and the capacity answer was given once
                     * above, from the header (section 12.3). */
                    const uint64_t regenerated = block.block_size;
                    if (regenerated > declared - frame->produced) {
                        frame->status = CUDEC_ERR_CORRUPT_INPUT;
                    } else {
                        frame->copy_count = regenerated;
                        frame->copy_is_rle =
                            block.block_type == kZstdBlockTypeRle;
                        frame->copy_src = src + frame->pos + 3;
                        frame->copy_dst = dst + frame->produced;
                        frame->produced += regenerated;
                    }
                }
            }
        }
        __syncthreads();
        if (frame->status != CUDEC_OK) {
            return;
        }

        if (frame->compressed) {
            __syncthreads();
            if (threadIdx.x == 0) {
                frame->status = ZstdDecodeCompressedBlockDevice(
                    frame, src + frame->pos + 3, frame->body_size, dst,
                    declared - frame->produced);
            }
            __syncthreads();
            if (frame->status != CUDEC_OK) {
                return;
            }
            /* The leftover literals: the bytes after the final sequence's
             * literal run, moved down out of the staging tail into the place
             * the plan located. */
            ZstdMoveDown(frame->copy_dst, frame->copy_src, frame->copy_count);
        } else {
            ZstdFillBlock(frame->copy_dst, frame->copy_src, frame->copy_count,
                          frame->copy_is_rle);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            frame->pos += static_cast<uint64_t>(3) + frame->body_size;
        }
        __syncthreads();
        if (frame->last_block) {
            break;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        if (!frame->last_block) {
            /* The fuel ran out, which the bound above puts beyond every
             * input: the block header refuses for want of its three bytes
             * first, and that is the refusal reported here. */
            frame->status = CUDEC_ERR_CORRUPT_INPUT;
        } else if (frame->produced != declared) {
            /* Every path that would have produced MORE than the declaration
             * was refused where it happened; what is left is the frame that
             * produced less. */
            frame->status = CUDEC_ERR_CORRUPT_INPUT;
        } else if (frame->header.content_checksum) {
            const uint64_t digest = Xxh64(dst, frame->produced);
            ZstdFrameReject frame_rung = kZstdFrameRejectNone;
            frame->status = ZstdVerifyContentChecksum(src + frame->pos,
                                                      size - frame->pos,
                                                      digest, &frame_rung);
            frame->pos += 4;
        }
        if (frame->status == CUDEC_OK && frame->pos != size) {
            /* A chunk holding more than one frame is outside the subset
             * (section 12.4), so bytes after the frame are refused rather
             * than ignored. */
            frame->status = CUDEC_ERR_CORRUPT_INPUT;
        }
    }
    __syncthreads();
}

/* One block per frame over a grid-stride loop. A grid with no blocks in it
 * cannot arise from the launcher, which derives the grid from the chunk
 * count; a block of the wrong width can only arise from a launch this file
 * did not shape, and it is refused rather than decoded, because the shared
 * region below is sized for that shape. */
template <int WaveSize>
__global__ void __launch_bounds__(kZstdBlockThreads)
    zstd_decode_batch(const void* const* src_ptrs, const size_t* src_sizes,
                      void* const* dst_ptrs, const size_t* dst_caps,
                      size_t chunk_count, cudec_chunk_result* results) {
    /* The block is a whole number of waves at either width, which is what
     * lets every collective in this file be a block barrier rather than a
     * wave one. */
    static_assert(kZstdBlockThreads % static_cast<unsigned>(WaveSize) == 0,
                  "the Zstd block must be a whole number of waves");
    __shared__ ZstdFrameShared frame;

    if (blockDim.x != kZstdBlockThreads || gridDim.x == 0) {
        return;
    }
    for (size_t chunk = blockIdx.x; chunk < chunk_count; chunk += gridDim.x) {
        const unsigned char* src =
            static_cast<const unsigned char*>(src_ptrs[chunk]);
        unsigned char* dst = static_cast<unsigned char*>(dst_ptrs[chunk]);
        __syncthreads();
        if (threadIdx.x == 0) {
            frame.state.literals_table.cells = frame.huf_cells;
            frame.state.literals_table.capacity = kZstdHufTableCells;
            frame.state.litlen.cells = frame.litlen_cells;
            frame.state.litlen.capacity = kZstdLitLenCells;
            frame.state.offset.cells = frame.offset_cells;
            frame.state.offset.capacity = kZstdOffsetCells;
            frame.state.matchlen.cells = frame.matchlen_cells;
            frame.state.matchlen.capacity = kZstdMatchLenCells;
            frame.body_size = 0;
            frame.copy_count = 0;
            frame.copy_is_rle = false;
            frame.compressed = false;
            frame.copy_src = src;
            frame.copy_dst = dst;
        }
        __syncthreads();

        ZstdDecodeFrameDevice(&frame, src, src_sizes[chunk], dst,
                              dst_caps[chunk]);

        __syncthreads();
        if (threadIdx.x == 0) {
            results[chunk].status = frame.status;
            results[chunk].reserved = 0;
            results[chunk].bytes_written =
                frame.status == CUDEC_OK ? frame.produced : 0;
        }
        /* The next frame's table build must not overtake a thread still
         * reading this one's tables. */
        __syncthreads();
    }
}

}  // namespace cudec_detail

#endif /* CUDEC_ZSTD_DECODE_CUH */
