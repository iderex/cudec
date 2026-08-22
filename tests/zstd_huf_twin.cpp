/* The CPU twin of the Zstd Huffman unit (src/zstd_huf.h): the tree description
 * decode in both of its spellings and the decoding table built from it (issue
 * #220). The sibling of tests/zstd_fse_twin.cpp and tests/zstd_bitstream_twin.
 * cpp: the single-source unit executed on the host, on the GPU-less CI runner,
 * and held to the pinned reference's verdicts.
 *
 * FOUR PROOFS, AND THEY ARE DIFFERENT IN KIND.
 *
 * THE FSE SPELLING IS TAKEN FROM REAL FRAMES rather than written here. Every
 * Compressed literals section of every #185 fixture is located by the frame
 * walker in tests/zstd_corpus.h and handed to both sides, which compares the
 * weights, the symbol count, the tree depth and the consumed byte count, then
 * diffs the built table cell by cell. A compressor writes that spelling and
 * nothing in this file could write it as convincingly.
 *
 * THE DIRECT SPELLING IS WRITTEN HERE, with weights computed by hand, because
 * the compressor does not emit it at these sizes and the format admits it. The
 * writer is four lines and its output is proven by the reference reading it
 * back, so the negatives below are malformed versions of bytes the reference
 * itself accepts.
 *
 * THE TABLE IS DIFFED AGAINST A TABLE BUILT AT A DIFFERENT SCALE, and this is
 * the thing to read before the comparison. libzstd rescales a tree whose
 * natural depth is below eleven UP to eleven so its decode loop has a constant
 * table size (HUF_rescaleStats in lib/decompress/huf_decompress.c), which
 * multiplies every entry 2^(11 - depth) times and leaves each entry's bit count
 * alone: a symbol's code length is depth + 1 - weight before the rescale and
 * targetDepth + 1 - (weight + scale) after it, which is the same number. So the
 * reference's cell i and the twin's cell i >> scale must agree exactly, and
 * that relation is asserted rather than the sizes being made to match. cudec
 * builds the natural table because that is the one a kernel puts in shared
 * memory - a tree of depth six is 64 cells here and 2048 there.
 *
 * THE NEGATIVES are one per reject rung, with the reference's verdict asserted
 * beside the twin's wherever the reference has one at this layer. Two rungs it
 * does not: the caller-argument rung and the build-side capacity rung are not
 * streams at all. The reference's refusal codes do not discriminate - unrelated
 * malformed descriptions all come back corruption_detected - so a negative here
 * asserts that the reference refused and that the twin refused through the rung
 * the negative was written for. */
#include "require.h"
#include "zstd_corpus.h"
#include "zstd_huf.h"

/* The reference's Huffman entry points are internal to libzstd and its huf.h
 * carries no C++ linkage guard of its own, so the include is wrapped here
 * rather than its declarations restated. HUF_STATIC_LINKING_ONLY is what
 * exposes HUF_readStats_wksp, HUF_readDTableX1_wksp and the sizing macros; it
 * is defined by the build for the reason issue #180 landed, and defining it in
 * both places is a redefinition the strict-warning build refuses. */
extern "C" {
#include <common/huf.h>
}

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;
using Weights = std::vector<uint8_t>;

using cudec_detail::kZstdHufMaxSymbolValue;
using cudec_detail::kZstdHufMaxTableLog;
using cudec_detail::ZstdHufBuildDTable;
using cudec_detail::ZstdHufCell;
using cudec_detail::ZstdHufReadWeights;
using cudec_detail::ZstdHufReject;
using cudec_detail::ZstdHufWeightScratch;

/* Which reject rungs a declared negative reached. Same discipline as the FSE
 * and bitstream twins: the enumeration lives once, in the header, and main()
 * requires every rung to have been named by a negative written to reach it. A
 * rung added to the unit with no negative behind it reds this test. */
bool g_reject_covered[cudec_detail::kZstdHufRejectCount] = {false};

void CoverRung(ZstdHufReject rung) {
    if (rung != cudec_detail::kZstdHufRejectNone) {
        g_reject_covered[rung] = true;
    }
}

/* Writes the direct spelling of a weight vector: the header byte carries the
 * count biased by 127, then one 4-bit nibble per weight, high nibble first. An
 * odd count leaves the final low nibble unused; it is written as zero here and
 * one negative below sets it to prove neither side reads it. */
Bytes EncodeDirect(const Weights& weights) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(127u + weights.size()));
    for (size_t i = 0; i < weights.size(); i += 2) {
        unsigned char byte = static_cast<unsigned char>(weights[i] << 4);
        if (i + 1 < weights.size()) {
            byte = static_cast<unsigned char>(byte | (weights[i + 1] & 0x0Fu));
        }
        out.push_back(byte);
    }
    return out;
}

/* The reference's own read of a tree description. `alphabet` is the hwSize it
 * is given, which is the same bound the twin is given, so the two agree about
 * how many weights they are willing to accept. */
struct OracleStats {
    bool ok = false;
    unsigned table_log = 0;
    unsigned nb_symbols = 0;
    size_t consumed = 0;
    Bytes weights;
};

OracleStats ReadStatsWithOracle(const unsigned char* src, size_t size,
                                size_t alphabet) {
    OracleStats out;
    out.weights.assign(alphabet, 0);
    std::vector<unsigned> rank(HUF_TABLELOG_MAX + 1, 0);
    std::vector<unsigned> wksp(HUF_DECOMPRESS_WORKSPACE_SIZE_U32, 0);
    unsigned nb = 0;
    unsigned log = 0;
    const size_t rc = HUF_readStats_wksp(
        out.weights.data(), alphabet, rank.data(), &nb, &log, src, size,
        wksp.data(), wksp.size() * sizeof(unsigned), 0);
    if (HUF_isError(rc)) {
        out.weights.assign(alphabet, 0);
        return out;
    }
    out.ok = true;
    out.table_log = log;
    out.nb_symbols = nb;
    out.consumed = rc;
    return out;
}

/* The reference's decoding table for the same description, flattened to the two
 * fields per cell the twin also produces, plus the depth it was built at.
 *
 * The descriptor word is the one zstd's own decoder installs before decoding a
 * block (ZSTD_HUFFDTABLE_CAPACITY_LOG in lib/decompress/zstd_decompress_
 * internal.h), so the reference is asked exactly what a real decode asks it
 * rather than something more permissive invented here. */
bool BuildWithOracle(const unsigned char* src, size_t size,
                     std::vector<ZstdHufCell>* out, unsigned* out_log) {
    std::vector<HUF_DTable> dtable(HUF_DTABLE_SIZE(12), 0);
    dtable[0] = static_cast<HUF_DTable>(11) * 0x01000001u;
    std::vector<unsigned> wksp(HUF_DECOMPRESS_WORKSPACE_SIZE_U32, 0);
    const size_t rc =
        HUF_readDTableX1_wksp(dtable.data(), src, size, wksp.data(),
                              wksp.size() * sizeof(unsigned), 0);
    if (HUF_isError(rc)) {
        return false;
    }
    unsigned descriptor = 0;
    std::memcpy(&descriptor, dtable.data(), sizeof(descriptor));
    /* DTableDesc is { maxTableLog, tableType, tableLog, reserved }. */
    const unsigned log = (descriptor >> 16) & 0xFFu;
    *out_log = log;
    /* HUF_DEltX1 is { nbBits, byte } in the pinned release, and reading the two
     * the other way round produces bit counts near a hundred rather than a
     * mismatch - which is how this was caught. */
    struct Elt {
        unsigned char nb_bits;
        unsigned char byte;
    };
    const Elt* cells = reinterpret_cast<const Elt*>(dtable.data() + 1);
    out->resize(static_cast<size_t>(1u) << log);
    for (size_t i = 0; i < out->size(); i++) {
        (*out)[i].nb_bits = cells[i].nb_bits;
        (*out)[i].symbol = cells[i].byte;
    }
    return true;
}

size_t g_descriptions = 0;
size_t g_cells = 0;

/* Both sides over one description, every quantity compared. Returns false on
 * the first divergence and says which one it was. */
bool ParityHolds(const unsigned char* src, size_t size, const char* where) {
    ZstdHufWeightScratch scratch;
    Weights weights(kZstdHufMaxSymbolValue + 1, 0);
    unsigned count = 0;
    unsigned log = 0;
    uint64_t consumed = 0;
    ZstdHufReject rung = cudec_detail::kZstdHufRejectNone;
    const cudec_status twin = ZstdHufReadWeights(
        src, size, kZstdHufMaxSymbolValue + 1, kZstdHufMaxTableLog, &scratch,
        weights.data(), &count, &log, &consumed, &rung);

    const OracleStats oracle =
        ReadStatsWithOracle(src, size, kZstdHufMaxSymbolValue + 1);
    if (!oracle.ok) {
        std::fprintf(stderr,
                     "%s: the reference refused a description the sweep "
                     "expects it to accept\n",
                     where);
        return false;
    }
    if (twin != CUDEC_OK) {
        std::fprintf(stderr, "%s: twin refused (rung %d) where the reference "
                             "read %u symbols at depth %u\n",
                     where, static_cast<int>(rung), oracle.nb_symbols,
                     oracle.table_log);
        return false;
    }
    if (count != oracle.nb_symbols) {
        std::fprintf(stderr, "%s: symbol count %u vs %u\n", where, count,
                     oracle.nb_symbols);
        return false;
    }
    if (log != oracle.table_log) {
        std::fprintf(stderr, "%s: tree depth %u vs %u\n", where, log,
                     oracle.table_log);
        return false;
    }
    if (consumed != oracle.consumed) {
        std::fprintf(stderr, "%s: consumed %llu vs %zu bytes\n", where,
                     static_cast<unsigned long long>(consumed),
                     oracle.consumed);
        return false;
    }
    for (unsigned i = 0; i < count; i++) {
        if (weights[i] != oracle.weights[i]) {
            std::fprintf(stderr, "%s: weight[%u] %u vs %u\n", where, i,
                         weights[i], oracle.weights[i]);
            return false;
        }
    }

    std::vector<ZstdHufCell> reference;
    unsigned reference_log = 0;
    if (!BuildWithOracle(src, size, &reference, &reference_log)) {
        std::fprintf(stderr, "%s: the reference refused to build the table\n",
                     where);
        return false;
    }
    std::vector<ZstdHufCell> cells(static_cast<size_t>(1u) << log);
    const cudec_status build =
        ZstdHufBuildDTable(weights.data(), count, log, cells.data(),
                           static_cast<uint32_t>(cells.size()), &rung);
    if (build != CUDEC_OK) {
        std::fprintf(stderr, "%s: twin refused to build (rung %d)\n", where,
                     static_cast<int>(rung));
        return false;
    }
    if (reference_log < log) {
        std::fprintf(stderr, "%s: the reference built at depth %u, below the "
                             "natural %u - the rescale only ever goes up\n",
                     where, reference_log, log);
        return false;
    }
    const unsigned scale = reference_log - log;
    for (size_t i = 0; i < reference.size(); i++) {
        const ZstdHufCell& want = reference[i];
        const ZstdHufCell& have = cells[i >> scale];
        if (want.symbol != have.symbol || want.nb_bits != have.nb_bits) {
            std::fprintf(stderr,
                         "%s: cell %zu (scale %u) is (symbol %u, nbBits %u) "
                         "and the reference has (symbol %u, nbBits %u)\n",
                         where, i, scale, have.symbol, have.nb_bits,
                         want.symbol, want.nb_bits);
            return false;
        }
    }
    g_descriptions++;
    g_cells += reference.size();
    return true;
}

/* One malformed description: what the twin must answer, and whether the
 * reference has a verdict at this layer to assert beside it. */
struct Negative {
    const char* name;
    Bytes stream;
    ZstdHufReject rung;
    bool oracle_refuses;
};

}  // namespace

int main() {
    /* ---- Step 1: the direct spelling, written here and read back by the
     * reference, so the negatives below are mutations of bytes it accepts. */
    {
        /* Two symbols of weight two and one implied symbol of weight three:
         * 2 + 2 makes four, the tree closes at depth three, and the remainder
         * of four is the implied last symbol's 2^(3-1). Hand-computed so the
         * expectation does not come from the thing under test. */
        const Weights base = {2, 2};
        const Bytes stream = EncodeDirect(base);
        REQUIRE(stream.size() == 2);
        REQUIRE(stream[0] == 129);
        REQUIRE(stream[1] == 0x22);

        ZstdHufWeightScratch scratch;
        Weights weights(kZstdHufMaxSymbolValue + 1, 0);
        unsigned count = 0;
        unsigned log = 0;
        uint64_t consumed = 0;
        ZstdHufReject rung = cudec_detail::kZstdHufRejectNone;
        /* The written pair alone leaves the deepest rank empty, which is its
         * own rung; the accepted vector below adds the two weight-one symbols
         * that close it. */
        REQUIRE(ZstdHufReadWeights(stream.data(), stream.size(),
                                   kZstdHufMaxSymbolValue + 1,
                                   kZstdHufMaxTableLog, &scratch,
                                   weights.data(), &count, &log, &consumed,
                                   &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdHufRejectDeepestRankEmpty);
        CoverRung(rung);
        REQUIRE(!ReadStatsWithOracle(stream.data(), stream.size(),
                                     kZstdHufMaxSymbolValue + 1)
                     .ok);
    }

    /* A four-symbol tree that closes: weights 2, 1, 1 written, implied last
     * weight 1. Budget 2 + 1 + 1 = 4, depth 3... computed the long way: the
     * written weights spend 2^(2-1) + 2^(1-1) + 2^(1-1) = 2 + 1 + 1 = 4, the
     * depth is highbit(4) + 1 = 3, the remainder is 8 - 4 = 4 and the implied
     * weight is highbit(4) + 1 = 3. So the tree is {2, 1, 1, 3} at depth 3,
     * with code lengths {2, 3, 3, 1}. */
    const Weights accepted = {2, 1, 1};
    {
        const Bytes stream = EncodeDirect(accepted);
        REQUIRE(ParityHolds(stream.data(), stream.size(), "direct-four-symbol"));

        ZstdHufWeightScratch scratch;
        Weights weights(kZstdHufMaxSymbolValue + 1, 0);
        unsigned count = 0;
        unsigned log = 0;
        uint64_t consumed = 0;
        ZstdHufReject rung = cudec_detail::kZstdHufRejectNone;
        REQUIRE(ZstdHufReadWeights(stream.data(), stream.size(),
                                   kZstdHufMaxSymbolValue + 1,
                                   kZstdHufMaxTableLog, &scratch,
                                   weights.data(), &count, &log, &consumed,
                                   &rung) == CUDEC_OK);
        /* The hand-computed answers, asserted rather than only diffed: a
         * comparison against the reference agrees when both are wrong in the
         * same way, and nothing else in this file would notice. */
        REQUIRE(count == 4);
        REQUIRE(log == 3);
        REQUIRE(weights[3] == 3);
        REQUIRE(consumed == 3);

        std::vector<ZstdHufCell> cells(8);
        REQUIRE(ZstdHufBuildDTable(weights.data(), count, log, cells.data(), 8,
                                   &rung) == CUDEC_OK);
        /* Placement, spelled out. Weight one is the deepest rank and goes
         * first: symbols 1 and 2, one cell each, three bits. Then weight two,
         * symbol 0, two cells, two bits. Then weight three, symbol 3, four
         * cells, one bit. */
        const uint8_t want_symbol[8] = {1, 2, 0, 0, 3, 3, 3, 3};
        const uint8_t want_bits[8] = {3, 3, 2, 2, 1, 1, 1, 1};
        for (unsigned i = 0; i < 8; i++) {
            REQUIRE_CTX(cells[i].symbol == want_symbol[i],
                        "cell %u symbol %u", i, cells[i].symbol);
            REQUIRE_CTX(cells[i].nb_bits == want_bits[i], "cell %u nbBits %u",
                        i, cells[i].nb_bits);
        }
    }

    /* An odd weight count, so the final low nibble is written and unread. Both
     * sides must accept whatever is in it, which is what makes the nibble a
     * non-rung rather than an unstated assumption. */
    {
        /* Five written weights spend 1 + 1 + 2 + 4 + 4 = 12 of sixteen, so the
         * depth is four and the implied last weight is three. Odd count, so
         * the third byte's low nibble is written and never read. */
        const Weights odd = {1, 1, 2, 3, 3};
        Bytes clean = EncodeDirect(odd);
        REQUIRE(clean.size() == 4);
        Bytes dirty = clean;
        dirty[3] = static_cast<unsigned char>(dirty[3] | 0x0Fu);
        REQUIRE(dirty != clean);
        REQUIRE(ParityHolds(clean.data(), clean.size(), "direct-odd-clean"));
        REQUIRE(ParityHolds(dirty.data(), dirty.size(), "direct-odd-dirty"));
    }

    /* ---- Step 2: the FSE spelling, over every Compressed literals section of
     * every corpus fixture. This is what the issue's first Done bullet names,
     * and it is the only place a compressor-written weight description
     * appears. */
    size_t sections = 0;
    {
        const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
        REQUIRE(!fixtures.empty());
        for (size_t f = 0; f < fixtures.size(); f++) {
            const ZstdFixture& fixture = fixtures[f];
            ZstdFrameShape shape;
            std::string why;
            REQUIRE_CTX(ParseZstdFrameShape(fixture.compressed, &shape, &why),
                        "%s: %s", fixture.name.c_str(), why.c_str());
            for (size_t b = 0; b < shape.blocks.size(); b++) {
                const ZstdBlockShape& block = shape.blocks[b];
                if (block.literals_type != kZstdLiteralsCompressed) {
                    continue;
                }
                sections++;
                char where[192];
                std::snprintf(where, sizeof(where), "%s block %zu",
                              fixture.name.c_str(), b);
                REQUIRE(ParityHolds(
                    fixture.compressed.data() + block.literals_payload_offset,
                    block.literals_payload_size, where));
            }
        }
        /* A sweep that found nothing would pass every assertion above, which
         * is the shape this project has already been bitten by twice. */
        REQUIRE(sections > 0);
    }

    /* ---- Step 3: the negatives, one per reachable stream rung. */
    {
        std::vector<Negative> negatives;

        /* Nothing at all. The reference reports srcSize_wrong, which is a
         * refusal like any other at this layer. */
        negatives.push_back({"empty", Bytes(),
                             cudec_detail::kZstdHufRejectEmptyDescription,
                             true});

        /* A direct header claiming eight weights with only two bytes of
         * nibbles behind it. */
        {
            Bytes stream = EncodeDirect({2, 1, 1, 4, 4, 4, 4, 4});
            REQUIRE(stream.size() == 5);
            stream.resize(3);
            negatives.push_back(
                {"direct-truncated", stream,
                 cudec_detail::kZstdHufRejectDescriptionTruncated, true});
        }

        /* An FSE header byte claiming more compressed bytes than are there. */
        {
            Bytes stream;
            stream.push_back(40);
            stream.push_back(0x20);
            stream.push_back(0x00);
            negatives.push_back(
                {"fse-truncated", stream,
                 cudec_detail::kZstdHufRejectDescriptionTruncated, true});
        }

        /* A weight above the tree's depth ceiling. Fifteen is the widest a
         * nibble can carry and the ceiling is twelve. */
        negatives.push_back({"weight-past-ceiling", EncodeDirect({15, 1, 1}),
                             cudec_detail::kZstdHufRejectWeightTooLarge, true});

        /* Every written weight zero, so the tree has no budget at all. */
        negatives.push_back({"weight-sum-zero", EncodeDirect({0, 0, 0, 0}),
                             cudec_detail::kZstdHufRejectWeightSumZero, true});

        /* A budget whose remainder is not a clean power of two: 4 + 1 spends
         * five of eight, and three is not a weight. */
        negatives.push_back({"remainder-not-a-power-of-two",
                             EncodeDirect({3, 1}),
                             cudec_detail::kZstdHufRejectLastWeightNotClean,
                             true});

        /* Three symbols at the ceiling spend 3 * 2^11, which asks for a depth
         * of thirteen. */
        negatives.push_back({"depth-past-ceiling", EncodeDirect({12, 12, 12}),
                             cudec_detail::kZstdHufRejectTableLogTooLarge,
                             true});

        /* A tree with nothing at its deepest rank: two symbols of weight two
         * and an implied third of weight three close at depth three with no
         * weight-one leaf anywhere. */
        negatives.push_back({"deepest-rank-empty", EncodeDirect({2, 2}),
                             cudec_detail::kZstdHufRejectDeepestRankEmpty,
                             true});

        for (size_t n = 0; n < negatives.size(); n++) {
            const Negative& negative = negatives[n];
            ZstdHufWeightScratch scratch;
            Weights weights(kZstdHufMaxSymbolValue + 1, 0);
            unsigned count = 0;
            unsigned log = 0;
            uint64_t consumed = 0;
            ZstdHufReject rung = cudec_detail::kZstdHufRejectNone;
            const unsigned char* data =
                negative.stream.empty() ? weights.data() : negative.stream.data();
            const cudec_status status = ZstdHufReadWeights(
                data, negative.stream.size(), kZstdHufMaxSymbolValue + 1,
                kZstdHufMaxTableLog, &scratch, weights.data(), &count, &log,
                &consumed, &rung);
            REQUIRE_CTX(status != CUDEC_OK, "%s was accepted", negative.name);
            REQUIRE_CTX(rung == negative.rung, "%s reached rung %d, not %d",
                        negative.name, static_cast<int>(rung),
                        static_cast<int>(negative.rung));
            CoverRung(rung);
            if (negative.oracle_refuses) {
                REQUIRE_CTX(!ReadStatsWithOracle(data, negative.stream.size(),
                                                 kZstdHufMaxSymbolValue + 1)
                                 .ok,
                            "%s: the reference accepted it", negative.name);
            }
        }
    }

    /* The alphabet bound, which both sides carry and which is not a property of
     * the bytes: a description of eight weights read against an alphabet of
     * four. The reference takes the same bound as its hwSize and refuses for
     * the same reason, so this negative has a verdict beside it. */
    {
        const Bytes stream = EncodeDirect({2, 1, 1, 4, 4, 4, 4, 4});
        ZstdHufWeightScratch scratch;
        Weights weights(kZstdHufMaxSymbolValue + 1, 0);
        unsigned count = 0;
        unsigned log = 0;
        uint64_t consumed = 0;
        ZstdHufReject rung = cudec_detail::kZstdHufRejectNone;
        REQUIRE(ZstdHufReadWeights(stream.data(), stream.size(), 4,
                                   kZstdHufMaxTableLog, &scratch,
                                   weights.data(), &count, &log, &consumed,
                                   &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdHufRejectTooManyWeights);
        CoverRung(rung);
        REQUIRE(!ReadStatsWithOracle(stream.data(), stream.size(), 4).ok);
    }

    /* An argument the format cannot express. The reference has no verdict on a
     * caller's argument and this rung says so where it is written. */
    {
        ZstdHufWeightScratch scratch;
        Weights weights(kZstdHufMaxSymbolValue + 1, 0);
        unsigned count = 0;
        unsigned log = 0;
        uint64_t consumed = 0;
        ZstdHufReject rung = cudec_detail::kZstdHufRejectNone;
        const Bytes stream = EncodeDirect(accepted);
        REQUIRE(ZstdHufReadWeights(stream.data(), stream.size(),
                                   kZstdHufMaxSymbolValue + 1,
                                   kZstdHufMaxTableLog + 1, &scratch,
                                   weights.data(), &count, &log, &consumed,
                                   &rung) == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(rung == cudec_detail::kZstdHufRejectBadRequest);
        CoverRung(rung);
    }

    /* ---- Step 4: the build side, reachable only from a caller that hands in
     * a weight array no description produced. */
    {
        ZstdHufReject rung = cudec_detail::kZstdHufRejectNone;
        const uint8_t weights[4] = {2, 1, 1, 3};
        /* Sized for the widest depth any case below asks for, so a capacity
         * refusal only ever fires where a case asks for one. The capacity
         * passed in is what the unit reads, not this. */
        std::vector<ZstdHufCell> cells(16);

        /* A cell array below the depth's own table size. */
        REQUIRE(ZstdHufBuildDTable(weights, 4, 3, cells.data(), 7, &rung) ==
                CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(rung == cudec_detail::kZstdHufRejectBuildCapacity);
        CoverRung(rung);

        /* Weights that do not fill the table they claim: the same four at a
         * depth of four spend eight of sixteen cells. */
        REQUIRE(ZstdHufBuildDTable(weights, 4, 4, cells.data(),
                                   static_cast<uint32_t>(cells.size()),
                                   &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdHufRejectBuildWeightsNotClean);
        CoverRung(rung);

        /* A weight past the depth it is being built at. Its own rung, because
         * a weight is a byte and the rank counters are only as long as the
         * deepest tree the format admits: this is the bound on that array, and
         * sharing the budget's rung would make its removal invisible. Two
         * cases, one just past the depth and one far past it, because the
         * counter is what the second would land in. */
        const uint8_t deep[3] = {5, 1, 1};
        REQUIRE(ZstdHufBuildDTable(deep, 3, 3, cells.data(),
                                   static_cast<uint32_t>(cells.size()),
                                   &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdHufRejectWeightTooLarge);
        CoverRung(rung);

        const uint8_t wild[3] = {200, 1, 1};
        REQUIRE(ZstdHufBuildDTable(wild, 3, 3, cells.data(),
                                   static_cast<uint32_t>(cells.size()),
                                   &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdHufRejectWeightTooLarge);
        CoverRung(rung);

        /* And the build-side caller-argument rung. */
        REQUIRE(ZstdHufBuildDTable(weights, 0, 3, cells.data(),
                                   static_cast<uint32_t>(cells.size()),
                                   &rung) == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(rung == cudec_detail::kZstdHufRejectBadRequest);
        CoverRung(rung);
    }

    /* Every rung the unit can return has a negative written to reach it. A rung
     * added later with nothing behind it reds here rather than shipping as
     * untested refusal. */
    for (int rung = cudec_detail::kZstdHufRejectNone + 1;
         rung < cudec_detail::kZstdHufRejectCount; rung++) {
        REQUIRE_CTX(g_reject_covered[rung], "reject rung %d has no negative",
                    rung);
    }

    std::printf("PASS: %zu descriptions compared against HUF_readStats "
                "(%zu from compressed literals sections), %zu table cells "
                "diffed against HUF_readDTableX1, %d reject rungs covered\n",
                g_descriptions, sections, g_cells,
                cudec_detail::kZstdHufRejectCount - 1);
    return 0;
}
