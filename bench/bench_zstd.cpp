/* The Zstd worst-case bench corpus (issue #229): a hand-constructed frame
 * family that is adversarial on all three axes the M5 design named, gated by
 * libzstd for validity and locked on shape, plus the libzstd CPU decode of
 * that corpus.
 *
 * WHY IT IS HAND-CONSTRUCTED. The reference compressor never emits this
 * shape. Its match finder extends a run into one long match, picks whatever
 * table mode is cheapest, and reuses a table across blocks the moment that
 * costs less - all three of which are the opposite of what a worst case
 * needs. So the sequences are written here and handed to the reference's own
 * sequence-compression entry point, which turns them into a frame without
 * being allowed to choose them.
 *
 * THE THREE AXES, and each one has an assertion under it so a future edit
 * that produces a valid-but-easy frame reds CI instead of leaving a report
 * that still says "worst case":
 *
 *   Maximum sequence density. Literal length 0 and match length 3 (the
 *   format's minimum) is one sequence per three decoded bytes, the most a
 *   valid frame can carry, so the serial sequence stage runs the most symbols
 *   per byte of output. This is the Zstd analog of the LZ4 offset-1 minmatch
 *   block in bench_lz4.cpp.
 *
 *   Maximum table rebuilds. Blocks alternate between three sequence shapes
 *   whose symbol statistics disagree, so the previous block's tables are
 *   never worth reusing and Repeat mode is never chosen. Every block boundary
 *   therefore pays a table build. The assertion is over the emitted frame:
 *   no field in any block may carry Repeat.
 *
 *   Adversarial repeat offsets. Every sequence carries literals_length == 0,
 *   which is the arm where the offset code selects a SHIFTED entry of the
 *   repeat-offset history, and the offsets are derived from the history so
 *   that all three arms - including the offset_value 3 / repeat_offset_1 - 1
 *   case - fire densely rather than occasionally.
 *
 * WHERE THE SHAPE LOCK STOPS, and this is a bound rather than a detail. The
 * frame walker reads headers, so the density and table-mode locks are over
 * the emitted bytes. The repeat-offset lock is not: offset codes live inside
 * the sequence bitstream, and reading them needs a sequence decoder this tree
 * does not have yet. That axis is therefore locked over the CONSTRUCTED
 * sequences - the resolution below is the format's own rule, run over the
 * list before the compressor sees it - together with the compressor parameter
 * that makes the reference resolve repcodes at all. When the sequence decoder
 * lands, this lock is the one to move onto the frame.
 *
 * Bench-only. Nothing here is compiled into the library. */
#include "bench_stats.h"

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include "zstd_corpus.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using cudec_bench::GbpsFromMs;
using cudec_bench::Percentile;

/* One frame's decoded size. The same 64 KiB rung the LZ4 rows use, so a
 * throughput number here is read against those without a unit conversion. */
constexpr size_t kFrameBytes = 65536;

/* ~210 MB, the scale of the recorded Silesia rows. */
constexpr size_t kWorstFrames = 3200;

/* The CI rot check runs the identical construction over a handful of frames
 * so it stays fast on the GPU-less runner. The frame is the same either
 * way - only the replica count moves. */
constexpr size_t kWorstSelfcheckFrames = 4;

/* Every byte of the source is this one value, which is what lets any offset
 * be a legal match source: the repeat-offset walk below chases
 * repeat_offset_1 - 1, and that value is not a multiple of anything. */
constexpr unsigned char kSeed = 0xA5;

/* The floors. Each one is a measured property of the intended construction
 * with margin, not a round number: weakening the axis it guards drops the
 * measurement far below it rather than just under it. */
constexpr double kSequenceDensityFloor = 0.30;  /* measured 0.307 */
constexpr double kRepcodeFractionFloor = 0.90;  /* measured 0.938 */
constexpr double kRepcodeArmFloor = 0.05;       /* measured 0.29 - 0.33 each */
constexpr size_t kMinBlocks = 4;
constexpr size_t kMinTableModes = 3;

/* Symbol_Compression_Mode, RFC 8878 section 3.1.1.3.2.1.1. Repeat is the one
 * this corpus exists to deny. */
constexpr unsigned kModeRepeat = 3;

struct ZstdCorpus {
    std::string name;
    std::string provenance;
    std::vector<std::vector<unsigned char>> originals;
    std::vector<std::vector<unsigned char>> compressed;
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
};

/* What the construction turned out to be, counted while it is built. The
 * repeat-offset half of the shape lock reads this. */
struct SequenceCensus {
    size_t sequences = 0;
    size_t repcoded = 0;
    size_t arm[4] = {0, 0, 0, 0}; /* index 0 is an explicit offset */
    size_t decoded_bytes = 0;
};

/* The repeat-offset history, RFC 8878 section 3.1.1.5. Initialised to
 * {1, 4, 8} at frame start and carried across blocks. */
struct RepHistory {
    unsigned r[3] = {1, 4, 8};
};

/* The format's own offset-code resolution, run over the constructed
 * sequences. Returns the offset code (1, 2 or 3) a conforming encoder must
 * emit for this offset, or 0 where no repeat entry matches and the offset
 * goes out explicitly. Updates the history either way, because the recency
 * rotation happens on both paths.
 *
 * The literals_length == 0 arm is the one worth reading twice: the index
 * shifts, so code 1 selects the SECOND entry, code 2 the third, and code 3
 * means repeat_offset_1 - 1. */
int ResolveOffsetCode(RepHistory* h, unsigned offset, unsigned litlen) {
    unsigned* r = h->r;
    int code = 0;
    if (litlen == 0) {
        if (offset == r[1]) {
            code = 1;
        } else if (offset == r[2]) {
            code = 2;
        } else if (r[0] > 1 && offset == r[0] - 1) {
            code = 3;
        }
    } else {
        if (offset == r[0]) {
            code = 1;
        } else if (offset == r[1]) {
            code = 2;
        } else if (offset == r[2]) {
            code = 3;
        }
    }

    if (code == 0) {
        r[2] = r[1];
        r[1] = r[0];
        r[0] = offset;
        return code;
    }
    if (litlen != 0 && code == 1) {
        return code; /* repeat_offset_1 selected: the history does not move */
    }
    if (litlen == 0 && code == 3) {
        const unsigned derived = r[0] - 1;
        r[2] = r[1];
        r[1] = r[0];
        r[0] = derived;
        return code;
    }
    /* The selected entry moves to the front and the ones above it shift
     * down. With the shift applied, code c selects index c for a zero
     * literal length and index c - 1 otherwise. */
    const unsigned idx = (litlen == 0) ? static_cast<unsigned>(code)
                                       : static_cast<unsigned>(code) - 1u;
    const unsigned selected = r[idx];
    if (idx == 2) {
        r[2] = r[1];
        r[1] = r[0];
    } else if (idx == 1) {
        r[1] = r[0];
    }
    r[0] = selected;
    return code;
}

/* Picks the offset for the next sequence from the history, so that the
 * sequence is a repeat offset by construction and the requested arm is the
 * one that fires. Falls back through the other two arms, and returns 0 when
 * none of them names an offset the frame can legally reach yet. */
unsigned OffsetForArm(const RepHistory& h, int want, unsigned litlen,
                      size_t reachable) {
    for (int step = 0; step < 3; step++) {
        const int arm = 1 + ((want - 1 + step) % 3);
        unsigned candidate = 0;
        if (litlen == 0) {
            candidate = (arm == 1)   ? h.r[1]
                        : (arm == 2) ? h.r[2]
                                     : (h.r[0] > 1 ? h.r[0] - 1u : 0u);
        } else {
            candidate = h.r[arm - 1];
        }
        if (candidate >= 1 && candidate <= reachable) {
            return candidate;
        }
    }
    return 0;
}

/* Builds one adversarial frame of `out_bytes` decoded bytes, and reports what
 * it built. The frame is produced by the reference's sequence-compression
 * entry point from the sequence list assembled here: the bytes are the
 * reference's, the sequences are not.
 *
 * Block shapes cycle so that consecutive blocks disagree on their symbol
 * statistics:
 *   shape 0 - match length fixed at 3, the densest possible
 *   shape 1 - match length alternating 3 and 4
 *   shape 2 - a short block with match lengths spread over 3..10
 * The reference then has nothing to reuse at a block boundary, which is the
 * table-rebuild axis, and it also picks a different mode for each shape,
 * which is what puts more than one mode in the frame. */
bool BuildWorstZstdBlock(size_t out_bytes, std::vector<unsigned char>* original,
                         std::vector<unsigned char>* frame,
                         SequenceCensus* census) {
    /* The first sequence carries a literal prologue, and nothing below may
     * produce fewer bytes than that plus one minimum match. Reject a size
     * that would wrap the loop bound instead of looping on a huge one. */
    constexpr unsigned kPrologueLiterals = 32;
    constexpr size_t kBlockReserve = 64;
    constexpr size_t kMinBytes = 4096;
    if (out_bytes < kMinBytes) {
        std::fprintf(stderr,
                     "worst-zstd frame needs at least %zu decoded bytes, "
                     "got %zu\n",
                     kMinBytes, out_bytes);
        return false;
    }

    /* Shape 2 is deliberately short: a block with few sequences is where the
     * reference stops paying for a table description, which is the third
     * mode in the frame. */
    constexpr size_t kLongBlockBytes = 8192;
    constexpr size_t kShortBlockBytes = 256;

    original->clear();
    original->reserve(out_bytes);
    std::vector<ZSTD_Sequence> seqs;
    RepHistory history;
    SequenceCensus counted;

    size_t produced = 0;
    size_t block = 0;
    bool first_sequence = true;
    while (produced < out_bytes) {
        const int shape = static_cast<int>(block % 3);
        size_t want = (shape == 2) ? kShortBlockBytes : kLongBlockBytes;
        if (produced + want > out_bytes) {
            want = out_bytes - produced;
        }
        size_t in_block = 0;
        size_t step = 0;
        while (in_block + kBlockReserve <= want) {
            unsigned litlen = 0;
            unsigned matchlen = 3;
            if (shape == 1) {
                matchlen = (step % 2 == 0) ? 4u : 3u;
            } else if (shape == 2) {
                matchlen = 3u + static_cast<unsigned>(step % 8);
            }
            if (first_sequence) {
                /* The frame's only literals: the history every later match
                 * reads from. Later blocks need none of their own, because
                 * the window reaches back across a block boundary. */
                litlen = kPrologueLiterals;
            }

            /* Every sixteenth sequence goes out on an explicit offset drawn
             * from a spread of distinct values. Without it the three history
             * slots converge on neighbouring small numbers, repeat_offset_1
             * minus one lands on a value another slot already holds, and the
             * resolution classifies it as the first arm instead - which
             * silently drops the third arm to nothing. The floor below is
             * what caught that. */
            static const unsigned kSpread[] = {64, 97, 131, 48};
            unsigned offset = 0;
            if (step % 16 == 15) {
                const unsigned candidate = kSpread[(step / 16) % 4];
                if (candidate <= produced) {
                    offset = candidate;
                }
            }
            if (offset == 0) {
                const int want_arm = 1 + static_cast<int>(step % 3);
                offset =
                    OffsetForArm(history, want_arm, litlen, produced + litlen);
            }
            if (offset == 0) {
                offset = 1;
            }

            original->insert(original->end(),
                             static_cast<size_t>(litlen) + matchlen, kSeed);
            ZSTD_Sequence seq;
            std::memset(&seq, 0, sizeof(seq));
            seq.offset = offset;
            seq.litLength = litlen;
            seq.matchLength = matchlen;
            seqs.push_back(seq);

            const int arm = ResolveOffsetCode(&history, offset, litlen);
            counted.arm[arm]++;
            counted.sequences++;
            if (arm != 0) {
                counted.repcoded++;
            }
            in_block += static_cast<size_t>(litlen) + matchlen;
            produced += static_cast<size_t>(litlen) + matchlen;
            first_sequence = false;
            step++;
        }
        if (in_block == 0) {
            /* What is left is smaller than one sequence. Stop here rather
             * than delimiting an empty block: the frame ends a few bytes
             * short of the requested size, and every size reported downstream
             * is the size actually built. */
            break;
        }
        /* An all-zero sequence is the block delimiter this API documents. */
        ZSTD_Sequence delimiter;
        std::memset(&delimiter, 0, sizeof(delimiter));
        seqs.push_back(delimiter);
        block++;
    }
    counted.decoded_bytes = produced;

    if (block < kMinBlocks) {
        std::fprintf(stderr,
                     "worst-zstd frame has %zu blocks, below the %zu-block "
                     "floor the table-rebuild axis needs\n",
                     block, kMinBlocks);
        return false;
    }

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (cctx == nullptr) {
        std::fprintf(stderr, "ZSTD_createCCtx failed\n");
        return false;
    }
    struct Param {
        ZSTD_cParameter id;
        int value;
        const char* name;
    };
    /* Explicit block delimiters because the block boundaries are the
     * table-rebuild axis and must not be chosen for us; minMatch 3 because
     * the density axis emits matches of exactly 3; repcode resolution
     * enabled because the third axis is the whole point and it is off by
     * default below level 10; validation on so an unreachable offset is a
     * refusal here rather than a frame nothing can decode. */
    const Param params[] = {
        {ZSTD_c_compressionLevel, 3, "compressionLevel"},
        {ZSTD_c_minMatch, 3, "minMatch"},
        {ZSTD_c_blockDelimiters, ZSTD_sf_explicitBlockDelimiters,
         "blockDelimiters"},
        {ZSTD_c_validateSequences, 1, "validateSequences"},
        {ZSTD_c_repcodeResolution, ZSTD_ps_enable, "repcodeResolution"},
        {ZSTD_c_contentSizeFlag, 1, "contentSizeFlag"},
        {ZSTD_c_checksumFlag, 1, "checksumFlag"},
    };
    for (const Param& p : params) {
        const size_t rc = ZSTD_CCtx_setParameter(cctx, p.id, p.value);
        if (ZSTD_isError(rc)) {
            std::fprintf(stderr, "ZSTD_CCtx_setParameter(%s) refused\n",
                         p.name);
            ZSTD_freeCCtx(cctx);
            return false;
        }
    }

    frame->assign(ZSTD_compressBound(original->size()) + 4096, 0);
    const size_t written =
        ZSTD_compressSequences(cctx, frame->data(), frame->size(), seqs.data(),
                               seqs.size(), original->data(),
                               original->size());
    ZSTD_freeCCtx(cctx);
    if (ZSTD_isError(written)) {
        std::fprintf(stderr,
                     "ZSTD_compressSequences refused the constructed "
                     "sequences\n");
        return false;
    }
    frame->resize(written);
    *census = counted;
    return true;
}

/* The repeat-offset half of the shape lock, over the constructed sequences.
 * See the header comment for why this half cannot be read out of the frame
 * yet. */
bool RepcodeAxisHolds(const SequenceCensus& census) {
    if (census.sequences == 0) {
        std::fprintf(stderr, "worst-zstd frame carries no sequences\n");
        return false;
    }
    const double total = static_cast<double>(census.sequences);
    const double repcoded = static_cast<double>(census.repcoded) / total;
    if (repcoded < kRepcodeFractionFloor) {
        std::fprintf(stderr,
                     "worst-zstd is not adversarial on repeat offsets: "
                     "%.4f of sequences resolve to one, below the %.2f "
                     "floor\n",
                     repcoded, kRepcodeFractionFloor);
        return false;
    }
    for (int arm = 1; arm <= 3; arm++) {
        const double share =
            static_cast<double>(census.arm[arm]) / total;
        if (share < kRepcodeArmFloor) {
            std::fprintf(stderr,
                         "worst-zstd leaves offset code %d nearly unused: "
                         "%.4f of sequences, below the %.2f floor\n",
                         arm, share, kRepcodeArmFloor);
            return false;
        }
    }
    return true;
}

/* The table-rebuild axis on its own, so the selfcheck can prove this refusal
 * bites: the shape it refuses does not occur in this corpus by construction,
 * and a guard nothing can trip is a guard nobody has checked. */
bool NoBlockReusesATable(const ZstdFrameShape& shape) {
    for (const ZstdBlockShape& b : shape.blocks) {
        if (b.sequence_count == 0) {
            continue;
        }
        const unsigned modes[3] = {b.ll_mode, b.of_mode, b.ml_mode};
        for (unsigned mode : modes) {
            if (mode == kModeRepeat) {
                return false;
            }
        }
    }
    return true;
}

/* The density and table-rebuild halves, over the emitted frame. */
bool FrameAxesHold(const std::vector<unsigned char>& frame,
                   size_t decoded_bytes) {
    ZstdFrameShape shape;
    std::string why;
    if (!ParseZstdFrameShape(frame, &shape, &why)) {
        std::fprintf(stderr, "worst-zstd frame did not walk: %s\n",
                     why.c_str());
        return false;
    }
    if (shape.blocks.size() < kMinBlocks) {
        std::fprintf(stderr,
                     "worst-zstd frame has %zu blocks, below the %zu-block "
                     "floor\n",
                     shape.blocks.size(), kMinBlocks);
        return false;
    }
    if (!NoBlockReusesATable(shape)) {
        std::fprintf(stderr,
                     "worst-zstd frame reuses a table: a block carries "
                     "Repeat mode, so that boundary pays no table build\n");
        return false;
    }

    size_t sequences = 0;
    bool mode_seen[4] = {false, false, false, false};
    for (const ZstdBlockShape& b : shape.blocks) {
        sequences += b.sequence_count;
        if (b.sequence_count == 0) {
            continue;
        }
        mode_seen[b.ll_mode] = true;
        mode_seen[b.of_mode] = true;
        mode_seen[b.ml_mode] = true;
    }
    size_t distinct = 0;
    for (unsigned mode = 0; mode < kModeRepeat; mode++) {
        if (mode_seen[mode]) {
            distinct++;
        }
    }
    if (distinct < kMinTableModes) {
        std::fprintf(stderr,
                     "worst-zstd frame carries %zu distinct table modes, "
                     "below the %zu the alternation is supposed to force\n",
                     distinct, kMinTableModes);
        return false;
    }

    const double density =
        static_cast<double>(sequences) / static_cast<double>(decoded_bytes);
    if (density < kSequenceDensityFloor) {
        std::fprintf(stderr,
                     "worst-zstd frame is not adversarial on sequence "
                     "density: %.4f sequences per decoded byte, below the "
                     "%.2f floor\n",
                     density, kSequenceDensityFloor);
        return false;
    }
    return true;
}

bool OracleDecodes(const std::vector<unsigned char>& frame, size_t expect,
                   std::vector<unsigned char>* out) {
    out->assign(expect, 0);
    const size_t rc =
        ZSTD_decompress(out->data(), out->size(), frame.data(), frame.size());
    if (ZSTD_isError(rc)) {
        return false;
    }
    out->resize(rc);
    return true;
}

/* Replicates the adversarial frame, and refuses to hand anything on that the
 * oracle does not accept or that is not adversarial. The oracle is the sole
 * authority on validity; the floors above are the authority on whether the
 * corpus still deserves its name. */
bool BuildWorstZstdCorpus(ZstdCorpus* corpus, size_t frames,
                          SequenceCensus* census) {
    std::vector<unsigned char> original;
    std::vector<unsigned char> frame;
    if (!BuildWorstZstdBlock(kFrameBytes, &original, &frame, census)) {
        return false;
    }

    std::vector<unsigned char> decoded;
    if (!OracleDecodes(frame, original.size() + 16, &decoded) ||
        decoded.size() != original.size() ||
        std::memcmp(decoded.data(), original.data(), decoded.size()) != 0) {
        std::fprintf(stderr,
                     "worst-zstd construction rejected by the oracle - "
                     "refusing to time an invalid frame\n");
        return false;
    }

    /* The oracle gate is only worth having if it bites. One flipped byte in
     * the middle of the frame, and the same call must refuse it - otherwise
     * the check above is passing on something other than the bytes. */
    std::vector<unsigned char> corrupted = frame;
    corrupted[corrupted.size() / 2] ^= 0xFFu;
    std::vector<unsigned char> ignored;
    if (OracleDecodes(corrupted, original.size() + 16, &ignored) &&
        ignored.size() == original.size() &&
        std::memcmp(ignored.data(), original.data(), ignored.size()) == 0) {
        std::fprintf(stderr,
                     "the oracle gate does not bite: a frame with a flipped "
                     "byte still round-tripped\n");
        return false;
    }

    if (!FrameAxesHold(frame, original.size()) || !RepcodeAxisHolds(*census)) {
        return false;
    }

    corpus->name = "worst-zstd";
    corpus->originals.assign(frames, original);
    corpus->compressed.assign(frames, frame);
    corpus->original_bytes = original.size() * frames;
    corpus->compressed_bytes = frame.size() * frames;
    corpus->provenance =
        "hand-constructed maximum-density / no-table-reuse / repeat-offset "
        "worst case, emitted through ZSTD_compressSequences and "
        "oracle-validated; the reference compressor never chooses this shape";
    return true;
}

/* The table-rebuild refusal is the one assertion whose subject never occurs
 * in this corpus, so nothing here would notice if it stopped refusing. The
 * shape it is about does occur elsewhere: the forced-mode corpus carries a
 * fixture built to demand Repeat mode, and running the refusal over that
 * fixture is what proves it still bites. Selfcheck only - the timed path has
 * no reason to build the fixture set. */
bool TableReuseRefusalBites() {
    for (const ZstdFixture& fixture : MakeZstdFixtures()) {
        ZstdFrameShape shape;
        std::string why;
        if (!ParseZstdFrameShape(fixture.compressed, &shape, &why)) {
            continue;
        }
        if (!NoBlockReusesATable(shape)) {
            return true;
        }
    }
    std::fprintf(stderr,
                 "the table-reuse refusal did not bite on any forced-mode "
                 "fixture - it can no longer tell a reused table from a "
                 "rebuilt one\n");
    return false;
}

double DecodeAllSeconds(const ZstdCorpus& corpus, unsigned char* scratch,
                        size_t scratch_size) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        const size_t rc =
            ZSTD_decompress(scratch, scratch_size, corpus.compressed[i].data(),
                            corpus.compressed[i].size());
        if (ZSTD_isError(rc)) {
            std::fprintf(stderr, "decode failed inside the timed region\n");
            std::exit(1);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void PrintReport(const ZstdCorpus& corpus, const SequenceCensus& census,
                 const std::vector<double>& sorted, size_t warmup,
                 size_t runs) {
    const double gb = static_cast<double>(corpus.original_bytes) / 1e9;
    std::printf("## bench_zstd report\n");
    std::printf("- decoder: CPU oracle, ZSTD_decompress (libzstd %s), single "
                "thread\n",
                ZSTD_versionString());
    std::printf("- corpus: %s, %zu frames, %.2f MB original, %.2f MB "
                "compressed (ratio %.4f), %s\n",
                corpus.name.c_str(), corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) /
                    static_cast<double>(corpus.original_bytes),
                corpus.provenance.c_str());
    std::printf("- shape: %.4f sequences per decoded byte, %.4f of them "
                "repeat offsets (codes 1/2/3 at %.4f / %.4f / %.4f of the "
                "sequences), no block reuses a table\n",
                static_cast<double>(census.sequences) /
                    static_cast<double>(census.decoded_bytes),
                static_cast<double>(census.repcoded) /
                    static_cast<double>(census.sequences),
                static_cast<double>(census.arm[1]) /
                    static_cast<double>(census.sequences),
                static_cast<double>(census.arm[2]) /
                    static_cast<double>(census.sequences),
                static_cast<double>(census.arm[3]) /
                    static_cast<double>(census.sequences));
    std::printf("- method: %zu warmup + %zu measured runs, wall clock per "
                "whole-corpus decode; the timed region is ZSTD_decompress "
                "only; output byte-verified before timing; percentiles are "
                "nearest-rank\n",
                warmup, runs);
    const double p50 = Percentile(sorted, 50);
    const double p90 = Percentile(sorted, 90);
    const double p99 = Percentile(sorted, 99);
    std::printf("- wall per run: p50 %.3f ms / p90 %.3f ms / p99 %.3f ms\n",
                p50 * 1e3, p90 * 1e3, p99 * 1e3);
    std::printf("- decode throughput: p50 %.3f GB/s / p90 %.3f GB/s / p99 "
                "%.3f GB/s\n",
                GbpsFromMs(gb, p50 * 1e3), GbpsFromMs(gb, p90 * 1e3),
                GbpsFromMs(gb, p99 * 1e3));
}

} // namespace

int main(int argc, char** argv) {
    bool worst = false;
    bool selfcheck = false;
    size_t warmup = 3;
    size_t runs = 30;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--worst") {
            worst = true;
        } else if (arg == "--selfcheck") {
            selfcheck = true;
        } else {
            std::fprintf(stderr, "usage: %s --worst [--selfcheck]\n", argv[0]);
            return 1;
        }
    }
    if (!worst) {
        std::fprintf(stderr, "usage: %s --worst [--selfcheck]\n", argv[0]);
        return 1;
    }
    if (selfcheck) {
        warmup = 0;
        runs = 1;
    }

    ZstdCorpus corpus;
    SequenceCensus census;
    if (!BuildWorstZstdCorpus(
            &corpus, selfcheck ? kWorstSelfcheckFrames : kWorstFrames,
            &census)) {
        return 1;
    }

    /* Byte-verify once, outside the timed region. */
    std::vector<unsigned char> scratch;
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        if (!OracleDecodes(corpus.compressed[i],
                           corpus.originals[i].size() + 16, &scratch) ||
            scratch.size() != corpus.originals[i].size() ||
            std::memcmp(scratch.data(), corpus.originals[i].data(),
                        scratch.size()) != 0) {
            std::fprintf(stderr, "verification failed at frame %zu\n", i);
            return 1;
        }
    }

    std::vector<unsigned char> timed_scratch(kFrameBytes);
    for (size_t i = 0; i < warmup; i++) {
        (void)DecodeAllSeconds(corpus, timed_scratch.data(),
                               timed_scratch.size());
    }
    std::vector<double> times;
    for (size_t i = 0; i < runs; i++) {
        times.push_back(
            DecodeAllSeconds(corpus, timed_scratch.data(),
                             timed_scratch.size()));
    }
    std::sort(times.begin(), times.end());
    PrintReport(corpus, census, times, warmup, runs);

    if (selfcheck && !TableReuseRefusalBites()) {
        return 1;
    }

    if (selfcheck) {
        std::printf("PASS: selfcheck complete\n");
    }
    return 0;
}
