/* The GDeflate benchmark harness: the CPU denominator for M4 (issue #224).
 *
 * There is no GDeflate kernel yet, so this measures the vendored reference
 * alone and says so in its own report rather than leaving a reader to infer
 * it from the absence of a GPU row (docs/MASTERPLAN.md section 5, honest
 * numbers). When the kernel lands, the device rows are read against these.
 *
 * THE UNIT IS THE PAGE, and that is the format's decision rather than this
 * harness's. GDeflate fixes a 64 KiB page, no match crosses one, and the
 * batch surface cudec will expose takes raw pages with caller-supplied sizes
 * (issue #216). So the source is cut into 64 KiB pieces, each piece is
 * compressed on its own - which is what makes every page independently
 * decodable - and the timed loop decodes them one at a time. A corpus built
 * by handing the whole file to the compressor would produce the same bytes
 * but would measure a decode this project's batch model never performs.
 *
 * THE LEVEL SET IS SECTION 11.8's, executed rather than re-taken: 0, 1, 6 and
 * 12, and each is there to reach something. Level 0 emits uncompressed blocks
 * by construction, which is the only block-type guarantee the reference's own
 * header gives; 1 is the fast end of the search; 6 is the default and
 * therefore the shape most data arrives in; 12 is the densest table
 * description and the most aggressively chosen block boundaries.
 *
 * WHAT THE LEVEL SWEEP DOES NOT CLAIM. Section 11.8 says the level list is
 * the plan for coverage and not the coverage argument: a compressor chooses
 * block types for its own reasons, and block type is not a pure function of
 * level because incompressible input forces stored blocks at any level. So
 * the four-level rows below claim no block type at all.
 *
 * WHAT --blocktypes ADDS (issue #225). A second corpus path whose families
 * are level CROSSED WITH INPUT CHARACTER, each one asserting the block type
 * it actually reached by walking the emitted page's first block header with
 * the substream schedule in src/gdeflate_schedule.h. A family that did not
 * reach its target type fails rather than quietly costing coverage, which is
 * 11.8's rule executed rather than restated. Those rows are decode-path
 * coverage and never headline numbers: they carry a ratio and they are not
 * the milestone's throughput figures.
 *
 * WHAT THE COMPOSITION LOCK CANNOT SEE, said before anyone reads it as more.
 * It reads the FIRST block of each page and stops. A GDeflate block boundary
 * is not findable by scanning - dossier 11.3 D5, and the schedule header says
 * the same - so reaching a page's second block means decoding to it, and
 * there is no decoder in the tree yet (#176, #182). So a family asserting
 * `static` proves every page OPENS with a static block, and says nothing
 * about what follows in that page.
 *
 * Ratio is printed beside throughput on every row. For a DEFLATE-class format
 * the ratio is half the pitch, and a throughput-only table would misreport
 * what the format is for.
 *
 * No CUDA in this binary at all: the reference decodes on the host, and there
 * is no cudec GDeflate path to launch.
 *
 * Bench-only. Nothing here is compiled into the library. */
#include "assetlike_source.h"
#include "bench_stats.h"

#include <libdeflate.h>

#include "gdeflate_schedule.h"
#include "xxhash64.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using cudec_bench::GbpsFromMs;
using cudec_bench::Percentile;

/* 64 KiB, the GDeflate page, fixed by the format rather than by this harness
 * (docs/MASTERPLAN.md section 11.2). */
constexpr size_t kPageBytes = 65536;

/* The levels section 11.8 names, and the order the report rows follow. */
constexpr int kLevels[] = {0, 1, 6, 12};
constexpr size_t kLevelCount = sizeof(kLevels) / sizeof(kLevels[0]);

/* The asset-like corpus replicates one generated block to this many pages:
 * the same ~210 MB scale as the recorded Silesia rows, so the two throughput
 * numbers are directly comparable. */
constexpr size_t kAssetlikePages = 3200;

/* The CI rot checks run the identical construction over a handful of pages so
 * they stay fast on the GPU-less runner. What is built is the same either
 * way - only the page count moves. */
constexpr size_t kSelfcheckPages = 4;

/* The selfcheck source for the file path: several pages long, compressible
 * enough that the pages are not all stored blocks, and from a fixed PRNG so a
 * failure reproduces byte for byte. */
constexpr size_t kSelfcheckBytes = kPageBytes * kSelfcheckPages;

constexpr size_t kMaxRuns = 1000000;

/* The generator's block and this harness's page are one granularity. A
 * granularity that moved on either side must fail to compile rather than
 * quietly benchmark a differently-tiled block. */
static_assert(cudec_bench::kAssetlikeBlockBytes == kPageBytes,
              "the asset-like generator's block is the GDeflate page");

struct Corpus {
    std::string name;
    std::vector<std::vector<unsigned char>> originals;
    std::vector<std::vector<unsigned char>> compressed;
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    /* Printed verbatim in the methodology block, so it must stay true for
     * whichever corpus ran. */
    std::string provenance;
    /* What the block-type walk asserted about this corpus, or the sentence
     * saying it asserted nothing. Printed rather than derived at the print
     * site, so a corpus path that skips the walk cannot inherit a claim a
     * different path earned. */
    std::string composition;
};

/* The RFC 1951 block-type values the format keeps (dossier 11.3: D3 changes
 * what a stored block CONTAINS, not the two bits that name it). Same three
 * constants tests/gdeflate_probes.cpp reads; this binary links no test
 * object, so they are stated rather than shared. */
constexpr uint32_t kBlockStored = 0;
constexpr uint32_t kBlockStatic = 1;
constexpr uint32_t kBlockDynamic = 2;

/* The most bytes one stored block can carry, from the 16-bit length field
 * tests/gdeflate_probes.cpp measures against the reference rather than
 * carrying over from RFC 1951 (probe 7, dossier D3/D4). It is one byte short
 * of a page, which is why a page of stored data is never one block. */
constexpr size_t kStoredBlockMaxBytes = 0xFFFF;
static_assert(kStoredBlockMaxBytes < kPageBytes,
              "a full page of stored data would fit in one stored block, and "
              "the composition claim below assumes it cannot");

/* What a corpus path that did not walk the pages says about itself. A
 * sentence rather than an empty string: an empty composition field printed
 * as an empty line reads as a claim nobody made. */
const char* const kCompositionNotAsserted =
    "not asserted on this path; the four-level sweep claims no block type, "
    "which is section 11.8's rule that a level list is a plan for coverage "
    "and not the coverage argument";

const char* BlockTypeName(uint32_t type) {
    switch (type) {
        case kBlockStored:
            return "stored";
        case kBlockStatic:
            return "static";
        case kBlockDynamic:
            return "dynamic";
        default:
            return "reserved";
    }
}

/* The first block header of a page: prime the 32 lanes, reset to lane 0
 * (every block header rides lane 0, dossier 11.2), then BFINAL and BTYPE.
 * This is the reading tests/gdeflate_probes.cpp pins, driven here over a
 * corpus rather than over one crafted page.
 *
 * Returns false when the page is too short to prime or the schedule went
 * sticky, which is a corpus this harness must not go on to time rather than
 * a block type to report. */
bool FirstBlockHeader(const std::vector<unsigned char>& page, uint32_t* type,
                      bool* is_final) {
    cudec_detail::GDeflateSchedule s;
    if (!cudec_detail::GDeflateInit(s, page.data(), page.size())) {
        return false;
    }
    cudec_detail::GDeflateReset(s);
    const uint32_t bfinal = cudec_detail::GDeflatePop(s, 1);
    const uint32_t block_type = cudec_detail::GDeflatePop(s, 2);
    if (s.failed) {
        return false;
    }
    *type = block_type;
    *is_final = bfinal != 0;
    return true;
}

/* 11.8's rule, executed: a family that did not reach its target block type
 * fails rather than quietly costing coverage. Every page is walked and the
 * first disagreement names the page and what was read there, because "some
 * page in this corpus is wrong" is not a thing anyone can act on. */
bool AssertComposition(Corpus* corpus, uint32_t want) {
    const size_t pages = corpus->compressed.size();
    size_t whole_page_blocks = 0;
    for (size_t i = 0; i < pages; i++) {
        uint32_t got = 0;
        bool is_final = false;
        if (!FirstBlockHeader(corpus->compressed[i], &got, &is_final)) {
            std::fprintf(stderr,
                         "page %zu of %s carries no readable block header - "
                         "the schedule refused it before any type was read\n",
                         i, corpus->name.c_str());
            return false;
        }
        if (got != want) {
            std::fprintf(stderr,
                         "page %zu of %s opens with a %s block, not the %s "
                         "block this family exists to reach - the corpus no "
                         "longer covers the decode path it is named for\n",
                         i, corpus->name.c_str(), BlockTypeName(got),
                         BlockTypeName(want));
            return false;
        }
        /* A stored block carries at most kStoredBlockMaxBytes, so a page
         * holding more than that cannot be one stored block and its first
         * block cannot be final. Reading BFINAL set there would mean the
         * length field is wider than probe 7 measured or that the walk is
         * reading the wrong bit, and both are ladder facts rather than corpus
         * accidents - so it is a refusal rather than a smaller claim. */
        if (want == kBlockStored && is_final &&
            corpus->originals[i].size() > kStoredBlockMaxBytes) {
            std::fprintf(stderr,
                         "page %zu of %s holds %zu bytes and its first stored "
                         "block is final, so one stored block would have to "
                         "carry more than the %zu bytes its length field can "
                         "hold\n",
                         i, corpus->name.c_str(), corpus->originals[i].size(),
                         kStoredBlockMaxBytes);
            return false;
        }
        if (is_final) {
            whole_page_blocks++;
        }
    }
    /* What the walk covered, not merely what it looked at. BFINAL on the
     * first block says the page has no second block, so for those pages the
     * opening block IS the page and the claim is complete; for the rest it is
     * a lower bound, because reaching a page's second block means decoding to
     * it. Reporting the split is what keeps a lower bound from being read as
     * a census (issues #206, #225).
     *
     * ONE SIDE OF THIS IS A REFUSAL AND THE OTHER IS A REPORT, and the
     * asymmetry is deliberate rather than unfinished. A walk that read BFINAL
     * as set where it is clear reds on the stored refusal above. A walk that
     * read it as clear everywhere - a dead read, the usual failure - turns
     * every whole-page claim below into a lower bound, which is the harness
     * claiming LESS than it covered. There is no structural fact that decides
     * how many blocks a table-using page holds, so nothing here can refuse
     * that direction, and a check that goes dead towards a weaker claim is
     * not a fail-open. */
    /* The parenthetical belongs only where a later block can exist. Carrying
     * it onto a page proved whole would deny a claim the walk has just
     * earned. */
    const char* const kUnreached =
        " (a later block in the same page is neither asserted nor denied, "
        "because reaching one means decoding to it)";
    std::string covered;
    if (whole_page_blocks == pages) {
        covered = "and BFINAL is set on it in every page, so each page is that "
                  "one block and this is the page's whole composition rather "
                  "than its opening";
    } else if (whole_page_blocks == 0) {
        covered = std::string(
                      "and BFINAL is clear on it in every page, so every page "
                      "carries at least one more block that this walk does "
                      "not reach - a lower bound on the count, never a "
                      "census") +
                  kUnreached;
    } else {
        covered = "and BFINAL is set on it in " +
                  std::to_string(whole_page_blocks) + " of " +
                  std::to_string(pages) +
                  " pages, which are whole; the rest carry at least one more "
                  "block that this walk does not reach" +
                  kUnreached;
    }
    corpus->composition = std::string("every page opens with a ") +
                          BlockTypeName(want) + " block, " + covered;
    return true;
}

/* One page compressed on its own. The page count is read back from the
 * reference rather than assumed: the page split is the reference's to decide,
 * and a pin whose geometry moved must fail here rather than pass quietly. */
bool CompressPage(libdeflate_gdeflate_compressor* c,
                  const std::vector<unsigned char>& in,
                  std::vector<unsigned char>* out) {
    size_t npages = 0;
    const size_t bound =
        libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    if (bound == 0 || npages != 1) {
        std::fprintf(stderr,
                     "the reference splits %zu bytes into %zu pages; this "
                     "harness compresses one page at a time\n",
                     in.size(), npages);
        return false;
    }
    std::vector<unsigned char> buffer(bound, 0);
    libdeflate_gdeflate_out_page page{};
    page.data = buffer.data();
    page.nbytes = bound;
    if (libdeflate_gdeflate_compress(c, in.data(), in.size(), &page, 1) == 0) {
        std::fprintf(stderr,
                     "the reference refused to compress a %zu-byte page\n",
                     in.size());
        return false;
    }
    /* What the compressor WROTE, not what it was offered: page.nbytes is an
     * out-parameter here, and timing the offered bound would time bytes the
     * page does not contain. */
    buffer.resize(page.nbytes);
    *out = std::move(buffer);
    return true;
}

/* Decodes one page through the reference into `out`, which the caller has
 * already sized to the page's decoded length.
 *
 * SUCCESS ALONE IS NOT THE CHECK, and this is the first thing
 * tests/oracle_gdeflate.cpp pins about this reference: it hands back an
 * actual output size and its own header says it may be used only where the
 * uncompressed size is known, so a status read without the size has a hole
 * exactly where a wrong stream lands. Every caller here compares the size,
 * and the round-trip pass compares the bytes. */
bool DecodePage(libdeflate_gdeflate_decompressor* d,
                const std::vector<unsigned char>& page,
                std::vector<unsigned char>* out) {
    libdeflate_gdeflate_in_page in{};
    in.data = page.data();
    in.nbytes = page.size();
    size_t got = 0;
    const libdeflate_result r =
        libdeflate_gdeflate_decompress(d, &in, 1, out->data(), out->size(),
                                       &got);
    return r == LIBDEFLATE_SUCCESS && got == out->size();
}

/* Cuts `source` into pages and compresses each one at `level`. */
bool BuildCorpus(const std::vector<unsigned char>& source, int level,
                 Corpus* corpus) {
    libdeflate_gdeflate_compressor* c =
        libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        std::fprintf(stderr, "cannot allocate a level-%d compressor\n", level);
        return false;
    }
    bool ok = true;
    for (size_t at = 0; at < source.size() && ok; at += kPageBytes) {
        const size_t n = std::min(kPageBytes, source.size() - at);
        std::vector<unsigned char> original(
            source.begin() + static_cast<std::ptrdiff_t>(at),
            source.begin() + static_cast<std::ptrdiff_t>(at + n));
        std::vector<unsigned char> page;
        ok = CompressPage(c, original, &page);
        if (ok) {
            corpus->originals.push_back(std::move(original));
            corpus->compressed.push_back(std::move(page));
        }
    }
    libdeflate_free_gdeflate_compressor(c);
    if (!ok) {
        return false;
    }
    if (corpus->originals.empty()) {
        std::fprintf(stderr, "the corpus is empty at level %d\n", level);
        return false;
    }
    corpus->original_bytes = 0;
    corpus->compressed_bytes = 0;
    for (size_t i = 0; i < corpus->originals.size(); i++) {
        corpus->original_bytes += corpus->originals[i].size();
        corpus->compressed_bytes += corpus->compressed[i].size();
    }
    return true;
}

/* Every page round-trips through the reference before anything is timed, and
 * the decoded pages are compared against the source they were cut from. A
 * number taken on a page set that dropped, reordered or truncated part of the
 * corpus is a number about a different corpus, and the ratio it reports would
 * still look plausible. */
bool CorpusRoundTrips(const std::vector<unsigned char>& source,
                      const Corpus& corpus) {
    if (corpus.original_bytes != source.size()) {
        std::fprintf(stderr,
                     "the pages hold %zu bytes but the source is %zu - the "
                     "corpus is not the one the report would name\n",
                     corpus.original_bytes, source.size());
        return false;
    }
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        std::fprintf(stderr, "cannot allocate a decompressor\n");
        return false;
    }
    bool ok = true;
    size_t at = 0;
    for (size_t i = 0; i < corpus.compressed.size() && ok; i++) {
        std::vector<unsigned char> back(corpus.originals[i].size(), 0);
        if (!DecodePage(d, corpus.compressed[i], &back)) {
            std::fprintf(stderr, "the reference refuses page %zu of %s\n", i,
                         corpus.name.c_str());
            ok = false;
            break;
        }
        if (std::memcmp(back.data(), source.data() + at, back.size()) != 0) {
            std::fprintf(stderr, "page %zu does not round-trip\n", i);
            ok = false;
            break;
        }
        at += back.size();
    }
    libdeflate_free_gdeflate_decompressor(d);
    return ok;
}

/* Destinations allocated outside the timed region, so the number is the
 * decoder's and not the allocator's. */
std::vector<std::vector<unsigned char>> MakeBuffers(const Corpus& corpus) {
    std::vector<std::vector<unsigned char>> buffers;
    buffers.reserve(corpus.originals.size());
    for (const auto& original : corpus.originals) {
        buffers.push_back(std::vector<unsigned char>(original.size()));
    }
    return buffers;
}

/* One timed pass over the whole page set. The timed region is the decompress
 * call alone. Returns a negative duration if any page fails, so a broken
 * decode can never be reported as a fast one. */
double DecodeAllSeconds(libdeflate_gdeflate_decompressor* d,
                        const Corpus& corpus,
                        std::vector<std::vector<unsigned char>>* buffers) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        if (!DecodePage(d, corpus.compressed[i], &(*buffers)[i])) {
            return -1.0;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

/* The corpus lock, on the model bench_snappy.cpp and bench_zstd.cpp record in
 * full: a fold of each page's length and XXH64 in corpus order, hashed again.
 * What it has to catch is drift - an oracle pin that moved, a page split that
 * changed, an input file that is not the one the report names - and all three
 * change the produced pages, so the digest runs over those. The fetched
 * inputs are already pinned by the manifest bench/get-corpora.sh writes, and a
 * second input lock here would be two authorities on one fact.
 *
 * XXH64 and not SHA-256, said plainly so nobody reads more into it: this is a
 * drift detector over data the harness just built, not a defence against a
 * chosen collision, and it is the hash already in the tree. */
void AppendLe64(uint64_t value, std::vector<unsigned char>* out) {
    for (unsigned i = 0; i < 8; i++) {
        out->push_back(static_cast<unsigned char>(value >> (i * 8)));
    }
}

uint64_t CorpusDigest(const Corpus& corpus) {
    std::vector<unsigned char> fold;
    fold.reserve(corpus.compressed.size() * 16);
    for (const auto& page : corpus.compressed) {
        AppendLe64(page.size(), &fold);
        AppendLe64(cudec_detail::Xxh64(page.data(), page.size()), &fold);
    }
    return cudec_detail::Xxh64(fold.data(), fold.size());
}

/* Reads one corpus file onto the end of `source`. Fails closed on I/O trouble
 * and on a file that contributed nothing, per FILE rather than over the
 * accumulated source: the accumulated test goes vacuous from the second
 * argument on, and a file that contributed no bytes must never end up
 * attested in the methodology block. */
bool AppendFile(const std::string& path, std::vector<unsigned char>* source) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open corpus file: %s\n", path.c_str());
        return false;
    }
    const size_t before = source->size();
    char buffer[1 << 16];
    while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
        source->insert(source->end(), buffer, buffer + in.gcount());
    }
    if (in.bad()) {
        std::fprintf(stderr, "read error in corpus file: %s\n", path.c_str());
        return false;
    }
    if (source->size() == before) {
        std::fprintf(stderr, "corpus file contributed no data: %s\n",
                     path.c_str());
        return false;
    }
    return true;
}

/* The file path's selfcheck source. Runs of a repeating alphabet with
 * occasional noise: matches for the decoder to execute, without collapsing
 * into one long match. */
std::vector<unsigned char> MakeSelfcheckSource(size_t bytes) {
    std::vector<unsigned char> out(bytes);
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < bytes; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        out[i] = (i % 61 == 0)
                     ? static_cast<unsigned char>(state >> 56)
                     : static_cast<unsigned char>('a' + (i / 7) % 26);
    }
    return out;
}

/* THE TWO FORCED SOURCES, and why each one forces what it does.
 *
 * Neither is a plausible workload and neither is meant to be: these corpora
 * exist to reach a decode path, and the report says so on their rows.
 *
 * Incompressible: a xorshift stream. LZ77 finds nothing and Huffman coding a
 * uniform byte distribution costs more than it saves, so the reference emits
 * a stored block at every level. This is the half of 11.8's sentence that
 * says block type is not a pure function of level.
 *
 * Low-entropy: a short repeating alphabet with no noise at all. Everything
 * after the first few bytes is one long match, the symbol set is tiny, and a
 * dynamic table description costs more than the fixed one it would replace -
 * so the reference takes the static block. */
std::vector<unsigned char> MakeIncompressibleSource(size_t bytes) {
    std::vector<unsigned char> out(bytes);
    uint64_t x = 88172645463325252ull;
    for (size_t i = 0; i < bytes; i++) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out[i] = static_cast<unsigned char>(x);
    }
    return out;
}

std::vector<unsigned char> MakeLowEntropySource(size_t bytes) {
    std::vector<unsigned char> out(bytes);
    for (size_t i = 0; i < bytes; i++) {
        out[i] = static_cast<unsigned char>('a' + (i % 7));
    }
    return out;
}

/* The asset-like source, replicated to `pages` identical pages. The block is
 * generated in bench/assetlike_source.h, which bench_lz4 reads too: the
 * recorded numbers are attested against those bytes, and two harnesses
 * generating them separately would be two corpora under one name. */
std::vector<unsigned char> MakeAssetlikeSource(size_t pages) {
    std::vector<unsigned char> block;
    cudec_bench::BuildAssetlikeSource(&block);
    std::vector<unsigned char> out;
    out.reserve(block.size() * pages);
    for (size_t i = 0; i < pages; i++) {
        out.insert(out.end(), block.begin(), block.end());
    }
    return out;
}

void PrintReport(const Corpus& corpus, int level,
                 const std::vector<double>& sorted, size_t warmup,
                 size_t runs) {
    std::vector<size_t> sizes;
    sizes.reserve(corpus.originals.size());
    for (const auto& original : corpus.originals) {
        sizes.push_back(original.size());
    }
    std::sort(sizes.begin(), sizes.end());
    const double gb = static_cast<double>(corpus.original_bytes) / 1e9;

    std::printf("## bench_gdeflate report\n");
    std::printf("- decoder: CPU oracle, libdeflate_gdeflate_decompress (the "
                "pinned NVIDIA/libdeflate gdeflate fork, commit %s), single "
                "thread. cudec has no GDeflate kernel yet, so this report is "
                "the denominator and carries no cudec number\n",
                CUDEC_GDEFLATE_COMMIT);
    std::printf("- host CPU: %s\n", cudec_bench::HostCpuName().c_str());
    std::printf("- corpus: %s, %zu pages, %.2f MB original, %.2f MB "
                "compressed (ratio %.4f), %s\n",
                corpus.name.c_str(), corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) /
                    static_cast<double>(corpus.original_bytes),
                corpus.provenance.c_str());
    std::printf("- granularity: %zu KiB pages, compression level %d\n",
                kPageBytes / 1024, level);
    std::printf("- corpus digest: %016llx (XXH64 over per-page length and "
                "XXH64, little-endian, in corpus order)\n",
                static_cast<unsigned long long>(CorpusDigest(corpus)));
    std::printf("- page sizes: min %zu / median %zu / max %zu bytes "
                "uncompressed\n",
                sizes.front(), sizes[sizes.size() / 2], sizes.back());
    std::printf("- method: %zu warmup + %zu measured runs, wall clock per "
                "whole-batch decode; the timed region is "
                "libdeflate_gdeflate_decompress only (destinations allocated "
                "outside it); every page round-trip-verified against the "
                "source once before timing; percentiles are nearest-rank\n",
                warmup, runs);
    std::printf("- block-type composition: %s\n",
                corpus.composition.c_str());
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

/* Reports the digest of the corpus it built through `digest_out` rather than
 * judging it here, so the caller can walk every level and name all of the
 * ones that moved. Stopping at the first would report one drifted level and
 * leave the rest unexamined, which reads as "only that one moved". */
bool RunLevel(const std::vector<unsigned char>& source, const std::string& name,
              const std::string& provenance, int level, int want_block_type,
              size_t warmup, size_t runs, uint64_t* digest_out) {
    Corpus corpus;
    corpus.name = name;
    corpus.provenance = provenance;
    corpus.composition = kCompositionNotAsserted;
    if (!BuildCorpus(source, level, &corpus)) {
        return false;
    }
    if (!CorpusRoundTrips(source, corpus)) {
        return false;
    }
    /* Before the digest and before anything is timed: a family that missed
     * its block type is not a corpus to record a number about, and a digest
     * pinned over it would pin the miss. */
    if (want_block_type >= 0 &&
        !AssertComposition(&corpus,
                           static_cast<uint32_t>(want_block_type))) {
        return false;
    }
    *digest_out = CorpusDigest(corpus);

    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        std::fprintf(stderr, "cannot allocate a decompressor\n");
        return false;
    }
    std::vector<std::vector<unsigned char>> buffers = MakeBuffers(corpus);
    bool ok = true;
    for (size_t i = 0; i < warmup && ok; i++) {
        if (DecodeAllSeconds(d, corpus, &buffers) < 0) {
            std::fprintf(stderr, "a warmup decode failed\n");
            ok = false;
        }
    }
    std::vector<double> times;
    for (size_t i = 0; i < runs && ok; i++) {
        const double seconds = DecodeAllSeconds(d, corpus, &buffers);
        if (seconds < 0) {
            std::fprintf(stderr, "a measured decode failed\n");
            ok = false;
            break;
        }
        times.push_back(seconds);
    }
    libdeflate_free_gdeflate_decompressor(d);
    if (!ok || times.empty()) {
        return false;
    }
    std::sort(times.begin(), times.end());
    PrintReport(corpus, level, times, warmup, runs);
    return true;
}

/* The selfcheck corpora are generated from a fixed source and compressed by
 * the pinned reference, so every page set is reproducible byte for byte and
 * its digest is a constant. Asserting them is what makes this a rot check
 * rather than a run: an oracle pin that moved, a page split that changed, or
 * a generator that lost a region all move a digest, and none of them would
 * stop the corpus round-tripping - which is why the round trip alone is not
 * the check.
 *
 * One digest per level, in the order kLevels lists them. */
constexpr uint64_t kSelfcheckDigests[kLevelCount] = {
    0xd46934fd0e71f055ull, 0x9ce5496f4c7cfc2eull, 0x808fbdf79a44154cull,
    0x0a8fa51b3c17129dull};
constexpr uint64_t kAssetlikeSelfcheckDigests[kLevelCount] = {
    0xc27a1a3637c14945ull, 0x3c7582779571120full, 0x22c785b441c4cb03ull,
    0xc8dda2e72b80524cull};

/* The level list and the two pin tables are indexed by one loop variable, so
 * a level added to one and not the others would read off the end. Refused at
 * compile time rather than left to the run, because the read would be in the
 * selfcheck - the one place a wrong answer looks like a verdict. */
static_assert(sizeof(kSelfcheckDigests) / sizeof(kSelfcheckDigests[0]) ==
                  kLevelCount,
              "one selfcheck digest per level");
static_assert(sizeof(kAssetlikeSelfcheckDigests) /
                      sizeof(kAssetlikeSelfcheckDigests[0]) ==
                  kLevelCount,
              "one asset-like selfcheck digest per level");

/* THE FORCED-BLOCK-TYPE FAMILIES (issue #225), one row each.
 *
 * Section 11.8 says the sweep is level CROSSED WITH INPUT CHARACTER, so each
 * family names both and the two stored rows differ in which of the two did
 * the forcing. Dropping either would leave the corpus proving less than it
 * looks like it proves: `stored-by-level` alone would let a reader conclude
 * that only level 0 reaches a stored block, which is the sentence 11.8
 * exists to refuse.
 *
 * MEASURED RATHER THAN ASSUMED, and the measurement is what decided that
 * nothing here is hand-constructed. The issue's scope allows hand-building a
 * stream where a block type cannot be forced out of the compressor; the
 * probe over levels 0, 1, 6 and 12 crossed with incompressible, low-entropy
 * and mixed input found both target types reachable, so no hand-built page
 * is in this tree and none is owed. `dynamic` is not a family here: it is
 * what the four-level sweep already produces on ordinary input, and a corpus
 * forcing the common case would carry cost for no coverage. */
struct ForcedFamily {
    const char* name;
    int level;
    uint32_t want_type;
    std::vector<unsigned char> (*source)(size_t);
    const char* provenance;
    /* The digest of the four-page selfcheck corpus this family builds. */
    uint64_t selfcheck_digest;
};

/* Bigger than the selfcheck's four pages so the row's timing is not noise,
 * and far smaller than the level sweep's corpora because these rows are
 * decode-path coverage rather than a throughput figure anyone quotes. */
constexpr size_t kBlocktypePages = 64;

const ForcedFamily kForcedFamilies[] = {
    {"stored-by-level", 0, kBlockStored, MakeSelfcheckSource,
     "generated in-harness from a fixed PRNG and compressed at level 0, "
     "which the reference's own header says emits uncompressed blocks by "
     "construction; DECODE-PATH COVERAGE, not a throughput figure",
     0xd46934fd0e71f055ull},
    {"stored-by-input", 6, kBlockStored, MakeIncompressibleSource,
     "generated in-harness from a xorshift PRNG, incompressible by "
     "construction, compressed at the DEFAULT level 6 - the stored block "
     "here is forced by the input and not by the level; DECODE-PATH "
     "COVERAGE, not a throughput figure",
     0x28488bd60cba9ce9ull},
    {"static-low-entropy", 1, kBlockStatic, MakeLowEntropySource,
     "generated in-harness as a seven-byte repeating alphabet with no noise, "
     "compressed at level 1, where a dynamic table description costs more "
     "than the fixed code it would replace; DECODE-PATH COVERAGE, not a "
     "throughput figure",
     0x8dfa68cd98743adeull},
};
constexpr size_t kForcedFamilyCount =
    sizeof(kForcedFamilies) / sizeof(kForcedFamilies[0]);

/* Both stored families and at least one static family must be present, or
 * this path has stopped being the thing its issue asked for. Refused at
 * compile time because a family deleted from the table above would otherwise
 * leave a green selfcheck that covers one block type. */
static_assert(kForcedFamilyCount >= 3,
              "the forced set must keep both stored forcings and the static "
              "one; a smaller table covers less than #225 asks for");

bool CheckDigest(uint64_t actual, const std::string& name, int level,
                 uint64_t expected) {
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr,
                 "the selfcheck corpus digest moved on %s at level %d: "
                 "expected %016llx, built %016llx - the corpus this harness "
                 "constructs is not the one its numbers were recorded on\n",
                 name.c_str(), level,
                 static_cast<unsigned long long>(expected),
                 static_cast<unsigned long long>(actual));
    return false;
}

bool ParseCount(const char* text, size_t lo, size_t hi, size_t* out) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || v < lo || v > hi) {
        return false;
    }
    *out = static_cast<size_t>(v);
    return true;
}

void Usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--warmup N] [--runs N] [--assetlike] "
                 "[--blocktypes] [--selfcheck] [corpus files...]\n",
                 argv0);
}

/* The forced-block-type path. Each family is its own corpus at its own
 * level, so this walks the family table instead of the level list the
 * other paths sweep. */
int RunBlocktypes(size_t warmup, size_t runs, bool selfcheck) {
    const size_t pages = selfcheck ? kSelfcheckPages : kBlocktypePages;
    bool ok = true;
    for (size_t i = 0; i < kForcedFamilyCount; i++) {
        const ForcedFamily& f = kForcedFamilies[i];
        const std::vector<unsigned char> source = f.source(pages * kPageBytes);
        uint64_t digest = 0;
        if (!RunLevel(source, f.name, f.provenance, f.level,
                      static_cast<int>(f.want_type), warmup, runs,
                      &digest)) {
            return 1;
        }
        if (selfcheck &&
            !CheckDigest(digest, f.name, f.level, f.selfcheck_digest)) {
            ok = false;
        }
        std::printf("\n");
    }
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    size_t warmup = 3;
    size_t runs = 30;
    bool selfcheck = false;
    bool assetlike = false;
    bool blocktypes = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--assetlike") {
            assetlike = true;
        } else if (arg == "--blocktypes") {
            blocktypes = true;
        } else if (arg == "--selfcheck") {
            selfcheck = true;
        } else if (arg == "--runs" && i + 1 < argc) {
            if (!ParseCount(argv[++i], 1, kMaxRuns, &runs)) {
                Usage(argv[0]);
                return 2;
            }
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!ParseCount(argv[++i], 0, kMaxRuns, &warmup)) {
                Usage(argv[0]);
                return 2;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            Usage(argv[0]);
            return 2;
        } else {
            files.push_back(arg);
        }
    }

    if (assetlike && !files.empty()) {
        std::fprintf(stderr, "--assetlike builds its own corpus; do not pass "
                             "corpus files with it\n");
        return 2;
    }

    if (blocktypes && (assetlike || !files.empty())) {
        std::fprintf(stderr,
                     "--blocktypes builds its own corpora, one per forced "
                     "block type; do not pass corpus files or --assetlike "
                     "with it\n");
        return 2;
    }

    if (selfcheck) {
        /* Short and fixed, so the rot check stays fast on the GPU-less runner
         * and its verdict is a digest rather than a timing. */
        warmup = 0;
        runs = 1;
    }

    if (blocktypes) {
        return RunBlocktypes(warmup, runs, selfcheck);
    }

    std::vector<unsigned char> source;
    std::string name;
    std::string provenance;
    if (assetlike) {
        source =
            MakeAssetlikeSource(selfcheck ? kSelfcheckPages : kAssetlikePages);
        name = "asset-like";
        provenance =
            "generated in-harness, a MODEL of a game asset package "
            "(bench/assetlike_source.h, issue #139) and not a measurement on "
            "real game data; cut into 64 KiB pages and each page compressed "
            "on its own by the pinned gdeflate fork";
    } else if (selfcheck) {
        source = MakeSelfcheckSource(kSelfcheckBytes);
        name = "selfcheck";
        provenance = "generated in-harness from a fixed PRNG; cut into 64 KiB "
                     "pages and each page compressed on its own by the pinned "
                     "gdeflate fork";
    } else {
        if (files.empty()) {
            Usage(argv[0]);
            std::fprintf(stderr, "no corpus files given (the recorded run "
                                 "uses bench/corpora/silesia/*)\n");
            return 2;
        }
        for (size_t i = 0; i < files.size(); i++) {
            if (!AppendFile(files[i], &source)) {
                return 1;
            }
            const size_t slash = files[i].find_last_of("/\\");
            const std::string base = slash == std::string::npos
                                         ? files[i]
                                         : files[i].substr(slash + 1);
            name += (i == 0 ? "" : "+") + base;
        }
        provenance = "cut into 64 KiB pages and each page compressed on its "
                     "own by the pinned gdeflate fork; every page decoded "
                     "back by the reference and compared against the source "
                     "before timing";
    }

    const uint64_t* expected =
        assetlike ? kAssetlikeSelfcheckDigests : kSelfcheckDigests;
    bool ok = true;
    for (size_t i = 0; i < kLevelCount; i++) {
        uint64_t digest = 0;
        if (!RunLevel(source, name, provenance, kLevels[i],
                      /*want_block_type=*/-1, warmup, runs, &digest)) {
            return 1;
        }
        if (selfcheck && !CheckDigest(digest, name, kLevels[i], expected[i])) {
            ok = false;
        }
        std::printf("\n");
    }
    return ok ? 0 : 1;
}
