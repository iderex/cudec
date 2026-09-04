/* What a tile of 128 sequences carries across its own lanes, measured on the
 * corpus (issue #203).
 *
 * WHY THIS EXISTS BEFORE THE KERNEL DOES. `docs/MASTERPLAN.md` section 14.8
 * stage D3 gives thread `t` sequence `t` of a tile and has it run "its literal
 * run and its match copy". Section 14.9 states the property that would make
 * that legal: `ZstdExecuteBlock`'s "loop body is already independent per
 * sequence in everything but the `literal_at` cursor". Two of the reads a
 * sequence makes are not independent of its neighbours, and neither section
 * measures them:
 *
 *  - A MATCH READS THE OUTPUT. `ZstdExecuteSequence` copies from
 *    `dst[to - offset]`, and for an offset smaller than the bytes the tile has
 *    produced before it, that address is inside the range a LOWER-numbered
 *    sequence of the SAME tile writes. Run serially that is the format working
 *    as designed; run one sequence per lane it is a read of a write that may
 *    not have happened.
 *  - A LITERAL READS THE OUTPUT TOO, once 14.3's placement is taken. The fused
 *    shape has no literals buffer: a block's literals are decoded into the tail
 *    of the frame's remaining destination region, at `[P + R - L, P + R)`.
 *    14.3's safety argument is explicitly about "executing them in increasing
 *    order", which is the serial claim; whether the same ranges are disjoint
 *    across the lanes of one tile is a different question and is the second
 *    number below.
 *
 * WHAT THIS FILE DOES AND DOES NOT CLAIM. It counts, on the #185 corpus, how
 * often each of those two reads lands inside the same tile's own output. It
 * does not decide what the kernel should do about it - that is the map's
 * business and the map is section 14.7 to 14.11. What it removes is the option
 * of finding out from a wrong-bytes report on a device: a count above zero says
 * a lane-parallel D3 has a read-after-write on streams the corpus already
 * contains, and a count of zero would say the corpus cannot see the question.
 *
 * IT IS SELF-CHECKING, WHICH IS WHY ITS NUMBERS ARE WORTH ANYTHING. The walk
 * below drives the shipped units in their shipped whole-block forms and then
 * diffs its own output against the pinned libzstd for every fixture. A walk
 * that mis-derived a destination or dropped a sequence would produce different
 * bytes long before it produced a different count, so the diff is what stands
 * behind the arithmetic rather than a second reading of it.
 *
 * THE TILE IS 128 BECAUSE THE MAP SAYS 128. Section 14.8 D3 fixes one sequence
 * per thread and 14.2 fixes 128 threads; the constant is named here from those
 * two rather than chosen, and 14.10 records that it halves to 64 if the
 * driver's per-block shared reservation demands it - which only makes a tile
 * smaller and can never introduce a dependency a larger tile did not have. */
#include "require.h"
#include "zstd_blocks.h"
#include "zstd_corpus.h"
#include "zstd_frame.h"
#include "zstd_twin_driver.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* Section 14.8 D3: one sequence per thread, 128 threads per block (14.2). */
constexpr uint32_t kTileSequences = 128;

struct Counts {
    size_t frames = 0;
    size_t declined = 0;
    size_t compressed_blocks = 0;
    size_t sequences = 0;
    /* Tiles that hold more than one sequence. A tile of one has no lane to
     * collide with and is counted apart so the denominator below is the
     * population the question is about. */
    size_t multi_lane_tiles = 0;
    /* A sequence whose match source range intersects the output range the
     * lower-numbered sequences of its own tile write. */
    size_t match_reads_same_tile = 0;
    /* The same question for the literal source range, under 14.3's placement
     * of the literals in the tail of the frame's remaining destination. */
    size_t literal_reads_same_tile = 0;
    /* The smallest offset seen on a hazardous match, which says how near the
     * collision is rather than only that there is one. */
    uint64_t smallest_hazard_offset = 0;
};

/* Whether [a_begin, a_end) and [b_begin, b_end) share a byte. Half-open on
 * both sides, so touching ranges do not count: a read that starts exactly
 * where another lane's write ends reads none of its bytes.
 *
 * AN EMPTY RANGE INTERSECTS NOTHING, AND SAYING SO COSTS A LINE THAT LOOKS
 * REDUNDANT AND IS NOT. The first sequence of a tile has no lower lane, so the
 * range it is tested against is empty, and the two comparisons below are both
 * satisfiable across an empty range - which reported every tile's first
 * sequence as a collision with nobody until the one-lane control said so. */
bool Intersects(uint64_t a_begin, uint64_t a_end, uint64_t b_begin,
                uint64_t b_end) {
    if (a_begin >= a_end || b_begin >= b_end) {
        return false;
    }
    return a_begin < b_end && b_begin < a_end;
}

/* One compressed block, driven through the shipped whole-block units, with the
 * tile partition measured over the arrays they produce.
 *
 * `produced` is the bytes earlier blocks of this frame left, so every address
 * below is the frame's and not the block's - which is the frame the offset
 * bounds are taken in and the frame a lane would actually be writing. */
cudec_status MeasureCompressedBlock(
    const unsigned char* body, uint64_t body_size,
    const cudec_detail::ZstdFrameHeader* header,
    cudec_detail::ZstdFrameState* state, uint64_t block_max,
    unsigned char* dst, uint64_t capacity, uint64_t produced,
    uint64_t* out_block_size, Counts* counts, std::string* why) {
    uint64_t literals_size = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdLiteralsReject literals_rung =
        cudec_detail::kZstdLiteralsRejectNone;
    cudec_status status = cudec_detail::ZstdDecodeLiterals(
        body, body_size, header->window_size, &state->literals_table,
        state->literals_scratch, state->literals, state->literals_capacity,
        &literals_size, &consumed, &literals_rung);
    if (status != CUDEC_OK) {
        *why = "literals section";
        return status;
    }

    const unsigned char* section = body + consumed;
    uint64_t remaining = body_size - consumed;
    cudec_detail::ZstdSeqSectionHeader seq_header;
    uint64_t seq_consumed = 0;
    cudec_detail::ZstdSeqReject seq_rung = cudec_detail::kZstdSeqRejectNone;
    status = cudec_detail::ZstdParseSeqSectionHeader(
        section, remaining, block_max, &seq_header, &seq_consumed, &seq_rung);
    if (status != CUDEC_OK) {
        *why = "sequence section header";
        return status;
    }
    section += seq_consumed;
    remaining -= seq_consumed;

    if (seq_header.sequence_count > state->sequences_capacity) {
        *why = "sequence storage too small";
        return CUDEC_ERR_OUTPUT_TOO_SMALL;
    }

    if (seq_header.sequence_count != 0) {
        const unsigned fields[3] = {cudec_detail::kZstdSeqFieldLitLen,
                                    cudec_detail::kZstdSeqFieldOffset,
                                    cudec_detail::kZstdSeqFieldMatchLen};
        const unsigned modes[3] = {seq_header.litlen_mode,
                                   seq_header.offset_mode,
                                   seq_header.matchlen_mode};
        cudec_detail::ZstdSeqTable* targets[3] = {&state->litlen,
                                                  &state->offset,
                                                  &state->matchlen};
        for (unsigned index = 0; index < 3; index++) {
            uint64_t table_consumed = 0;
            status = cudec_detail::ZstdSeqLoadTable(
                fields[index], modes[index], section, remaining,
                state->seq_scratch, targets[index], &table_consumed,
                &seq_rung);
            if (status != CUDEC_OK) {
                *why = "sequence table";
                return status;
            }
            section += table_consumed;
            remaining -= table_consumed;
        }
        status = cudec_detail::ZstdDecodeSequences(
            section, remaining, seq_header.sequence_count, &state->litlen,
            &state->offset, &state->matchlen, state->sequences,
            state->sequences_capacity, &seq_rung);
        if (status != CUDEC_OK) {
            *why = "sequence decode";
            return status;
        }
    }

    for (uint32_t index = 0; index < seq_header.sequence_count; index++) {
        cudec_detail::ZstdRepcodeReject repcode_rung =
            cudec_detail::kZstdRepcodeRejectNone;
        status = cudec_detail::ZstdRepcodeResolve(
            &state->repcodes, state->sequences[index].offset_value,
            state->sequences[index].literals_length, &state->offsets[index],
            &repcode_rung);
        if (status != CUDEC_OK) {
            *why = "repeat-offset resolution";
            return status;
        }
    }

    cudec_detail::ZstdExecPlan plan;
    cudec_detail::ZstdExecReject exec_rung = cudec_detail::kZstdExecRejectNone;
    status = cudec_detail::ZstdExecPrefixSum(
        state->sequences, seq_header.sequence_count, literals_size, block_max,
        state->destinations, seq_header.sequence_count + 1, &plan, &exec_rung);
    if (status != CUDEC_OK) {
        *why = "prefix sum";
        return status;
    }

    /* WHERE THE LITERALS WOULD BE ON THE DEVICE, WHICH IS NOT WHERE THEY ARE
     * HERE. Section 14.3 places a block's literals at `[P + R - L, P + R)`
     * with `P` the bytes produced so far, `R` the declared output still to come
     * and `L` the literals this block regenerated. `P + R` is the frame's
     * declared content size for every block, so the tail begins at
     * `frame_content_size - L` and nothing about it depends on the block. The
     * host walk keeps its own literals buffer, so this address is arithmetic
     * over the placement rather than a pointer - which is what a measurement
     * taken before the kernel exists can be. */
    const uint64_t literal_base = header->frame_content_size - literals_size;

    uint64_t literal_cursor = 0;
    for (uint32_t index = 0; index < seq_header.sequence_count; index++) {
        /* A tile holds min(kTileSequences, what is left), and only a tile
         * with a second lane can have a collision at all. */
        if (index % kTileSequences == 0 && kTileSequences > 1 &&
            seq_header.sequence_count - index > 1) {
            counts->multi_lane_tiles++;
        }
        const uint32_t tile_start = index - (index % kTileSequences);
        const cudec_detail::ZstdSequence sequence = state->sequences[index];
        const uint64_t at = produced + state->destinations[index];
        /* What the lanes below this one in the same tile write: from the
         * tile's first destination up to this sequence's own. */
        const uint64_t earlier_begin =
            produced + state->destinations[tile_start];
        const uint64_t earlier_end = at;

        const uint64_t match_to = at + sequence.literals_length;
        const uint64_t offset = state->offsets[index];
        if (sequence.match_length != 0 && offset <= match_to) {
            const uint64_t from = match_to - offset;
            if (Intersects(from, from + sequence.match_length, earlier_begin,
                           earlier_end)) {
                counts->match_reads_same_tile++;
                if (counts->smallest_hazard_offset == 0 ||
                    offset < counts->smallest_hazard_offset) {
                    counts->smallest_hazard_offset = offset;
                }
            }
        }

        if (sequence.literals_length != 0) {
            const uint64_t literal_from = literal_base + literal_cursor;
            if (Intersects(literal_from,
                           literal_from + sequence.literals_length,
                           earlier_begin, earlier_end)) {
                counts->literal_reads_same_tile++;
            }
        }
        literal_cursor += sequence.literals_length;
        counts->sequences++;
    }

    uint64_t block_produced = produced;
    status = cudec_detail::ZstdExecuteBlock(
        state->sequences, seq_header.sequence_count, state->destinations,
        state->offsets, state->literals, literals_size, &plan,
        header->window_size, dst, capacity, &block_produced, &exec_rung);
    if (status != CUDEC_OK) {
        *why = "block execution";
        return status;
    }
    *out_block_size = block_produced - produced;
    counts->compressed_blocks++;
    return CUDEC_OK;
}

/* One whole frame, block by block, with the compressed blocks measured.
 *
 * The raw and RLE arms are here because a frame that mixes them is one the
 * corpus emits, and skipping them would leave `produced` wrong for every
 * compressed block after one - which the byte diff would catch, but only after
 * the counts had already been taken over the wrong addresses. */
cudec_status MeasureFrame(const Bytes& frame, Bytes* out, Counts* counts,
                          std::string* why) {
    cudec_twin::Storage storage;
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject frame_rung =
        cudec_detail::kZstdFrameRejectNone;
    cudec_status status = cudec_detail::ZstdParseFrameHeader(
        frame.data(), frame.size(), &header, &frame_rung);
    if (status != CUDEC_OK) {
        *why = "frame header";
        return status;
    }

    const uint64_t capacity = header.frame_content_size;
    out->assign(static_cast<size_t>(capacity) + 1, 0);
    const uint64_t block_max =
        cudec_detail::ZstdLiteralsBlockMaximum(header.window_size);

    uint64_t pos = header.header_size;
    uint64_t produced = 0;
    uint64_t fuel = frame.size() / 3 + 1;
    bool last_seen = false;
    while (fuel-- != 0) {
        cudec_detail::ZstdBlockHeader block;
        status = cudec_detail::ZstdParseBlockHeader(
            frame.data() + pos, frame.size() - pos, header.window_size, &block,
            &frame_rung);
        if (status != CUDEC_OK) {
            *why = "block header";
            return status;
        }
        const unsigned char* body = frame.data() + pos + 3;
        if (block.block_type == cudec_detail::kZstdBlockTypeRaw ||
            block.block_type == cudec_detail::kZstdBlockTypeRle) {
            if (block.block_size > capacity - produced) {
                *why = "block regenerates past the declared content size";
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            for (uint32_t i = 0; i < block.block_size; i++) {
                (*out)[static_cast<size_t>(produced) + i] =
                    block.block_type == cudec_detail::kZstdBlockTypeRaw
                        ? body[i]
                        : body[0];
            }
            produced += block.block_size;
        } else {
            uint64_t block_size = 0;
            status = MeasureCompressedBlock(body, block.body_size, &header,
                                            &storage.state, block_max,
                                            out->data(), capacity, produced,
                                            &block_size, counts, why);
            if (status != CUDEC_OK) {
                return status;
            }
            produced += block_size;
        }
        pos += 3 + block.body_size;
        if (block.last_block) {
            last_seen = true;
            break;
        }
    }
    if (!last_seen) {
        *why = "no last block";
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    out->resize(static_cast<size_t>(produced));
    counts->frames++;
    return CUDEC_OK;
}

}  // namespace

int main() {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());

    Counts counts;
    for (size_t index = 0; index < fixtures.size(); index++) {
        const ZstdFixture& fixture = fixtures[index];
        Bytes decoded;
        std::string why;
        const cudec_status status =
            MeasureFrame(fixture.compressed, &decoded, &counts, &why);
        if (status != CUDEC_OK) {
            /* A fixture outside the v1 subset is declined rather than refused,
             * and declining is not this file's subject. Anything else is. */
            REQUIRE_CTX(status == CUDEC_ERR_UNSUPPORTED, "%s: %s refused (%d)",
                        fixture.name.c_str(), why.c_str(),
                        static_cast<int>(status));
            counts.declined++;
            continue;
        }
        /* The walk is only worth its counts if it is the decoder. The
         * reference is what says so, per fixture rather than in aggregate. */
        Bytes reference;
        REQUIRE_CTX(ZstdOracleDecodes(fixture.compressed, &reference),
                    "fixture %s: the reference declined its own frame",
                    fixture.name.c_str());
        REQUIRE_CTX(decoded.size() == reference.size(),
                    "fixture %s: %zu bytes, reference %zu",
                    fixture.name.c_str(), decoded.size(), reference.size());
        REQUIRE_CTX(
            equal_bytes(decoded.data(), reference.data(), reference.size()),
            "fixture %s", fixture.name.c_str());
    }

    /* THE NUMBERS ARE PRINTED BEFORE THEY ARE JUDGED. Both assertions below
     * are about the corpus rather than about a decode, so a run that fails one
     * of them is a run whose counts are the thing to read; printing after the
     * assertion would take the measurement away exactly when it is wanted. */
    std::printf(
        "tile dependency - %zu frames walked (%zu declined outside the "
        "subset), %zu compressed blocks, %zu sequences, %zu multi-lane tiles "
        "of %u; %zu matches and %zu literal runs read inside their own tile's "
        "output; smallest hazardous offset %llu\n",
        counts.frames, counts.declined, counts.compressed_blocks,
        counts.sequences, counts.multi_lane_tiles, kTileSequences,
        counts.match_reads_same_tile, counts.literal_reads_same_tile,
        static_cast<unsigned long long>(counts.smallest_hazard_offset));

    /* THE CORPUS REACHES THE QUESTION. A tile with one lane cannot collide
     * with anything, so a corpus of nothing but short blocks would report zero
     * hazards while saying nothing at all about the map. This is the
     * denominator, asserted so a later corpus change that lost it is a failure
     * here rather than a silently weaker number above. */
    REQUIRE(counts.multi_lane_tiles > 0);

    /* THE MEASUREMENT SECTION 14.9's INDEPENDENCE CLAIM IS ABOUT. A match
     * whose source lies in the output the same tile's lower lanes write is a
     * read-after-write the moment those lanes run concurrently. The assertion
     * is deliberately in this direction: it locks that the corpus CONTAINS the
     * case, so no implementation of D3 can be certified against a corpus that
     * never posed the question. What to do about the dependency is the map's
     * decision and is not taken here. */
    REQUIRE(counts.match_reads_same_tile > 0);

    std::printf("PASS: the corpus reaches the tile question\n");
    return 0;
}
