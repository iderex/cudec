/* The Zstd corpus proving itself before any Zstd decoder exists, the
 * counterpart of oracle_lz4.cpp and snappy_corpus.cpp (issue #185).
 *
 * Four properties, and the third is the one this corpus was built for:
 *
 *  - the pairs are real: the pinned oracle decodes every frame back to the
 *    bytes it was made from;
 *  - generation is deterministic: two passes agree byte for byte, and a
 *    pinned digest catches a drift that both passes share;
 *  - every fixture GOT the mode it asked for, read out of the emitted
 *    headers. A compressor that quietly declines a forced parameter fails
 *    here instead of costing coverage in silence;
 *  - the coverage set is fixed in place, so deleting a fixture reds this
 *    test rather than shrinking the corpus unnoticed.
 *
 * The machinery that judges the third property is itself checked against
 * cases it must refuse - a walker that accepted anything, or a demand check
 * that passed everything, would make the whole file vacuous.
 *
 * docs/ZSTD-CORPUS.md is the coverage matrix in prose and records why the
 * Silesia rung is not here. */
#include "require.h"
#include "zstd_corpus.h"

#include <cstdio>
#include <set>
#include <string>

namespace {

/* Corpus drift tripwire, the instrument oracle_lz4.cpp and
 * snappy_corpus.cpp both carry: a zstd bump that moves the compressor's
 * output moves every fixture, and the bump should have to update this number
 * on purpose instead of watching a changed corpus pass. FNV-1a, not crypto:
 * a tripwire against drift, not a defence. */
constexpr uint64_t kExpectedZstdCorpusDigest = 0xd104a6f6fef0e98bull;

uint64_t Fnv1a64(uint64_t hash, const void* data, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

/* The cells this corpus claims, one string per (dimension, value). Written
 * out rather than derived from the fixtures, because a set derived from the
 * fixtures would shrink with them and prove nothing. Cells covered
 * elsewhere in the tree are named in docs/ZSTD-CORPUS.md, not here. */
const char* const kExpectedCoverage[] = {
    "block:compressed",   "block:raw",           "block:rle",
    "checksum:0",         "checksum:1",          "content-size:0",
    "content-size:1",     "literals:compressed", "literals:raw",
    "literals:rle",       "literals:treeless",   "ll-mode:basic",
    "ll-mode:compressed", "ll-mode:repeat",      "ml-mode:basic",
    "ml-mode:compressed", "ml-mode:repeat",      "ml-mode:rle",
    "of-mode:basic",      "of-mode:compressed",  "of-mode:repeat",
    "of-mode:rle",        "single-segment:0",    "single-segment:1",
    "streams:1",          "streams:4",
};

const char* const kBlockNames[] = {"raw", "rle", "compressed"};
const char* const kLiteralsNames[] = {"raw", "rle", "compressed", "treeless"};
const char* const kModeNames[] = {"basic", "rle", "compressed", "repeat"};

void CollectCoverage(const ZstdDemand& d, std::set<std::string>* into) {
    if (d.single_segment >= 0) {
        into->insert("single-segment:" + std::to_string(d.single_segment));
    }
    if (d.content_size >= 0) {
        into->insert("content-size:" + std::to_string(d.content_size));
    }
    if (d.checksum >= 0) {
        into->insert("checksum:" + std::to_string(d.checksum));
    }
    if (d.block_type >= 0) {
        into->insert(std::string("block:") + kBlockNames[d.block_type]);
    }
    if (d.literals_type >= 0) {
        into->insert(std::string("literals:") + kLiteralsNames[d.literals_type]);
    }
    if (d.literals_streams >= 0) {
        into->insert("streams:" + std::to_string(d.literals_streams));
    }
    if (d.ll_mode >= 0) {
        into->insert(std::string("ll-mode:") + kModeNames[d.ll_mode]);
    }
    if (d.of_mode >= 0) {
        into->insert(std::string("of-mode:") + kModeNames[d.of_mode]);
    }
    if (d.ml_mode >= 0) {
        into->insert(std::string("ml-mode:") + kModeNames[d.ml_mode]);
    }
}

}  // namespace

int main() {
    const auto fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());
    const auto again = MakeZstdFixtures();
    REQUIRE(again.size() == fixtures.size());

    uint64_t digest = 14695981039346656037ull;
    std::set<std::string> covered;
    size_t block_total = 0;

    for (size_t i = 0; i < fixtures.size(); i++) {
        const ZstdFixture& f = fixtures[i];
        const char* name = f.name.c_str();

        /* Determinism first: everything below is a statement about one of two
         * runs, and it is only worth making if the two runs agree. */
        REQUIRE_CTX(again[i].name == f.name, "fixture %zu order moved", i);
        REQUIRE_CTX(again[i].original == f.original, "fixture %s source", name);
        REQUIRE_CTX(again[i].compressed == f.compressed, "fixture %s frame",
                    name);

        digest = Fnv1a64(digest, f.name.data(), f.name.size());
        digest = Fnv1a64(digest, f.compressed.data(), f.compressed.size());

        std::vector<unsigned char> decoded;
        REQUIRE_CTX(ZstdOracleDecodes(f.compressed, &decoded),
                    "fixture %s: the oracle refused a frame the corpus made",
                    name);
        REQUIRE_CTX(decoded.size() == f.original.size(),
                    "fixture %s: %zu bytes back, %zu expected", name,
                    decoded.size(), f.original.size());
        REQUIRE_CTX(
            equal_bytes(decoded.data(), f.original.data(), decoded.size()),
            "fixture %s", name);

        ZstdFrameShape shape;
        std::string why;
        REQUIRE_CTX(ParseZstdFrameShape(f.compressed, &shape, &why),
                    "fixture %s: the walker could not read the frame it "
                    "generated: %s",
                    name, why.c_str());
        REQUIRE_CTX(!shape.blocks.empty(), "fixture %s has no blocks", name);
        REQUIRE_CTX(ZstdShapeSatisfies(shape, f.demand, &why),
                    "fixture %s asked the compressor for a mode it did not "
                    "emit (%s) - coverage lost, not a tolerable outcome",
                    name, why.c_str());
        block_total += shape.blocks.size();
        CollectCoverage(f.demand, &covered);
    }

    REQUIRE_CTX(digest == kExpectedZstdCorpusDigest,
                "corpus digest is 0x%016llx - an oracle or generator change "
                "moved the fixtures; verify deliberately, then update the pin",
                static_cast<unsigned long long>(digest));

    /* Every claimed cell present, and no cell claimed that nothing covers:
     * both directions, so the list cannot drift away from the corpus in
     * either one. */
    const size_t expected_cells =
        sizeof(kExpectedCoverage) / sizeof(kExpectedCoverage[0]);
    for (size_t i = 0; i < expected_cells; i++) {
        REQUIRE_CTX(covered.count(kExpectedCoverage[i]) == 1,
                    "no fixture demands %s - a corpus family was removed or "
                    "its demand weakened",
                    kExpectedCoverage[i]);
    }
    REQUIRE_CTX(covered.size() == expected_cells,
                "%zu cells demanded, %zu listed - a new family arrived "
                "without joining the recorded matrix",
                covered.size(), expected_cells);

    /* The batch rung: the geometry cudec_zstd_decompress_batch consumes is
     * independent frames over one source, so the proof is that the frames
     * decode independently and reassemble to the source exactly. */
    const std::vector<unsigned char>& source = fixtures[0].original;
    const size_t chunk = 1024;
    const auto frames = MakeZstdBatchFrames(source, chunk, 3);
    REQUIRE(!frames.empty());
    REQUIRE(frames.size() == (source.size() + chunk - 1) / chunk);
    std::vector<unsigned char> rejoined;
    for (size_t i = 0; i < frames.size(); i++) {
        std::vector<unsigned char> out;
        REQUIRE_CTX(ZstdOracleDecodes(frames[i], &out), "batch frame %zu", i);
        rejoined.insert(rejoined.end(), out.begin(), out.end());
    }
    REQUIRE(rejoined.size() == source.size());
    REQUIRE(equal_bytes(rejoined.data(), source.data(), rejoined.size()));

    /* The judging machinery against cases it must refuse. Without these three
     * every claim above could be produced by a walker that returned true and
     * a demand check that agreed with everything. */
    {
        ZstdFrameShape shape;
        std::string why;
        std::vector<unsigned char> truncated = fixtures[0].compressed;
        truncated.resize(truncated.size() - 1);
        REQUIRE(!ParseZstdFrameShape(truncated, &shape, &why));

        /* A frame that is well formed in every other respect, with one magic
         * byte changed. Short junk would be refused for being short, which
         * proves nothing about the magic check; this input is refused only
         * because the magic is read. */
        std::vector<unsigned char> wrong_magic = fixtures[0].compressed;
        wrong_magic[0] = 0x27;
        REQUIRE(!ParseZstdFrameShape(wrong_magic, &shape, &why));

        REQUIRE(ParseZstdFrameShape(fixtures[0].compressed, &shape, &why));
        ZstdDemand impossible;
        impossible.block_type = kZstdBlockRaw;
        impossible.literals_type = kZstdLiteralsTreeless;
        REQUIRE(!ZstdShapeSatisfies(shape, impossible, &why));
    }

    std::printf("PASS: %zu zstd fixtures round-trip over %zu blocks; %zu "
                "coverage cells demanded and met; %zu batch frames rejoin; "
                "generation deterministic\n",
                fixtures.size(), block_total, covered.size(), frames.size());
    return 0;
}
