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
 * the same - so reaching a page's second block means decoding to it, and this
 * walk decodes nothing. So a family asserting `static` proves every page OPENS
 * with a static block, and says nothing about what follows in that page. THAT
 * SENTENCE USED TO GIVE THE ABSENCE OF A DECODER AS THE REASON, naming #176
 * and #182; both landed, `--blockmix` censuses whole pages through
 * src/gdeflate_block.h, and the bound above is a property of THIS walk rather
 * than of the tree.
 *
 * Ratio is printed beside throughput on every row. For a DEFLATE-class format
 * the ratio is half the pitch, and a throughput-only table would misreport
 * what the format is for.
 *
 * THIS PARAGRAPH SAID THERE WAS NO CUDA IN THIS BINARY AT ALL, because no
 * cudec GDeflate path existed to launch. One landed under #214 and `--gpu`
 * times it (issue #228): the reference still decodes on the host and is still
 * the denominator, the device rows are printed under the same methodology
 * block so the two cannot be quoted apart, and without the flag this binary
 * is what it always was. The device rows go through
 * `cudec_gdeflate_decompress_batch` and through nothing else.
 *
 * Bench-only. Nothing here is compiled into the library. */
#include "assetlike_source.h"
#include "bench_stats.h"
#include "cudec.h"
#include "gpu_bench.h"

#include <libdeflate.h>

#include "gdeflate_block.h"
#include "gdeflate_page_writer.h"
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
    /* Bytes inside `compressed` that no decoder reads: the zero tail a
     * hand-emitted page carries for the reference's unchecked refill. Held
     * apart from compressed_bytes because a ratio that silently counted
     * padding would attest a stream that is not there. Zero for every corpus
     * the compressor produced. */
    size_t padding_bytes = 0;
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

/* THE WORST-CASE CORPORA (issues #226 and #430), and what they lock.
 *
 * WHY THEY ARE NOT COMPRESSED BY THE REFERENCE. Every other corpus in this
 * file is what the pinned compressor chose to emit; a compressor picks the
 * code that costs the FEWEST bits, which is the opposite of what these rows
 * are for. The security-posture question is what a page an attacker fully
 * controls can force out of a decoder, so the page is emitted here and the
 * reference is kept as the authority on whether it is valid at all.
 *
 * WHAT THE SHAPE IS, against the four ingredients MASTERPLAN section 13.5
 * names. One generator builds all four, and the two rows it reports are the
 * two ends of the last of them:
 *
 *  - CODE LENGTHS LONG ENOUGH THAT THE ROOT ACCELERATOR MISSES. Every symbol
 *    this page emits - literal, length and distance alike - sits at DEFLATE's
 *    maximum codeword length. The spine symbols that make the code complete
 *    are never emitted.
 *  - EVERY LANE CONSUMING NEAR ITS WATERMARK ON EVERY ROUND. A length round
 *    spends a maximum codeword and sixteen extra bits, a distance round a
 *    maximum codeword and the widest extra-bit field the output so far
 *    reaches back over, and a literal round the codeword alone - so the
 *    average is the density the row prints and not a claim about every round
 *    separately.
 *  - COPIES SPLIT ACROSS TWO ROUNDS ON THE SAME LANE. The body is minimum
 *    matches: the length round reserves and the distance round on the same
 *    lane retires, so `GDeflateDoCopy` runs once per three decoded bytes and
 *    the deferred-copy machinery is exercised rather than skipped. Length
 *    symbol 285 is DEFLATE64's base 3 with SIXTEEN extra bits, which is the
 *    most expensive way the format has of asking for three bytes.
 *  - FREQUENT BLOCK HEADERS, which is what separates the two rows.
 *    `--worstrounds` (#226) puts one final dynamic block on a page and claims
 *    nothing about this ingredient; `--worstheaders` (#430) cuts the same page
 *    into one dynamic block per group of 32 matches - a header every 96
 *    decoded bytes, which is the most this construction admits, since a block
 *    carries whole groups. A header is bits that produce no output at all and
 *    it rides lane 0 while the other 31 lanes do nothing, so the many-block
 *    page is strictly the harder one: measured, 0.7026 refills per decoded
 *    byte against 0.6152, on pages that decode to the same bytes.
 *
 * THE QUANTITY IS REFILLS PER DECODED BYTE, AND IT IS NOT THE ONE THE PLAN
 * FIRST NAMED. Section 13.5 fixed rounds per decoded byte. That quantity does
 * not separate an easy page from this one in the direction a floor can use: an
 * all-literal dynamic block and a stored block both sit at exactly 1/32, so a
 * floor between them is impossible, and a match-carrying page sits BELOW 1/32
 * (two rounds per three bytes at the minimum length), so 1/32 is the maximum
 * of that quantity rather than a value nothing falls under. A refill is one
 * 32-bit word handed to one lane, so it counts the bits a page spends, and it
 * ranks all three cases the way the cost does. 13.5 carries the same reasoning
 * and was amended in the change that built this.
 *
 * WHAT REFILLS ARE AND ARE NOT, because the issue's scope line offers the
 * compressed/original ratio as a substitute and the honest answer is not a
 * clean refusal. For a page whose every emitted word is consumed - which this
 * one is, exactly, and MeasureRefillDensity refuses a page that is not - the
 * two are affine: density equals the emitted stream's ratio divided by 32/8.
 * What
 * refills buy is that they are defined by decoder work rather than by page
 * size, so they do not move with bytes no decoder reads, and they are read out
 * of a decode that had to reproduce the source. The ratio proxy's real defect
 * is over COMPRESSOR output, where it points the wrong way: the M4 denominator
 * table on #224 measures level 0 - the highest ratio the format can produce -
 * as the FASTEST family on both corpora. */

/* DEFLATE's maximum code length, from the header that builds against it rather
 * than as a second copy of 15. Nothing about this generator is adversarial
 * except that every emitted symbol sits at this length. */
constexpr uint32_t kWorstCodeBits = cudec_detail::kGDeflateMaxCodeLen;

/* The literal/length vector runs 0..285, so the length symbol below is inside
 * it and HLIT is 29. */
constexpr uint32_t kWorstLengthSym = 285;
constexpr size_t kWorstLitLenSyms = kWorstLengthSym + 1;

/* The unary spine: symbols 0..kWorstSpineSyms-1 at lengths 1..kWorstSpineSyms,
 * never emitted, present so the deep symbols can sit at the maximum length in
 * a code that spends its codespace exactly. */
constexpr uint32_t kWorstLitLenSpine = 10;
constexpr uint32_t kWorstLitLenDeep = 1u << (kWorstCodeBits - kWorstLitLenSpine);
/* Two of the deep slots are the end-of-block symbol and the length symbol; the
 * rest are literals, and they are the last ones before end-of-block. */
constexpr uint32_t kWorstDeepLiterals = kWorstLitLenDeep - 2u;
constexpr uint32_t kWorstFirstDeepLiteral =
    cudec_test::kEndOfBlock - kWorstDeepLiterals;

/* WHAT THESE ASSERTIONS DO AND DO NOT DO. The Kraft identity itself is NOT
 * among them: `kWorstLitLenDeep` is DEFINED as 2^(15 - spine), so an assertion
 * comparing the two is a tautology that passes whatever the constants say.
 * What the code being complete actually depends on is that the vector the
 * generator builds assigns exactly that many symbols the maximum length, and
 * that is read off the built vector by WorstCodeIsComplete below rather than
 * argued here.
 *
 * These are the bounds instead, and they are written so that no subtraction
 * they depend on can wrap before they are evaluated: an unsigned underflow
 * turns a spine that outgrew its alphabet into a huge first index, which the
 * writes in WorstLitLenLengths would take past the end of the vector. */
static_assert(kWorstLitLenSpine < kWorstCodeBits,
              "a spine at or past the maximum length spends the codespace "
              "before the deep symbols get any");
static_assert(kWorstLitLenDeep + kWorstLitLenSpine < cudec_test::kEndOfBlock,
              "the spine and the deep literals must fit below end-of-block "
              "without the first-deep-literal subtraction wrapping");
static_assert(kWorstDeepLiterals < cudec_test::kEndOfBlock &&
                  kWorstFirstDeepLiteral > kWorstLitLenSpine &&
                  kWorstFirstDeepLiteral + kWorstDeepLiterals <=
                      cudec_test::kEndOfBlock,
              "every deep literal index must land inside the vector and clear "
              "the spine");
static_assert(kWorstLitLenSyms > cudec_test::kEndOfBlock &&
                  kWorstLitLenSyms > kWorstLengthSym,
              "every symbol this generator writes must be inside the vector it "
              "writes into");
static_assert(kWorstLitLenSyms >= 257 && kWorstLitLenSyms <= 288,
              "HLIT is five bits over a 257-symbol floor");

/* The distance vector is the whole alphabet, with its own spine and its own
 * deep set. The deep set is the TOP of the alphabet because extra-bit count
 * grows with the distance base, and an extra-bit field is bits the lane spends
 * for no symbol: symbol 31 carries fourteen of them. */
constexpr uint32_t kWorstDistSyms = cudec_detail::kGDeflateNumDistSyms;
constexpr uint32_t kWorstDistSpine = 11;
constexpr uint32_t kWorstDistDeep = 1u << (kWorstCodeBits - kWorstDistSpine);
constexpr uint32_t kWorstFirstDeepDist = kWorstDistSyms - kWorstDistDeep;
static_assert(kWorstDistSpine < kWorstCodeBits,
              "a distance spine at or past the maximum length spends the "
              "codespace before the deep symbols get any");
static_assert(kWorstDistDeep + kWorstDistSpine < kWorstDistSyms,
              "the distance spine and the deep distance symbols must fit in "
              "the alphabet without the first-deep-distance subtraction "
              "wrapping");
static_assert(kWorstFirstDeepDist > kWorstDistSpine &&
                  kWorstFirstDeepDist + kWorstDistDeep <= kWorstDistSyms,
              "every deep distance index must land inside the vector and clear "
              "the spine");

/* The match this page is made of: the minimum length the format admits, asked
 * for in the most expensive way it can be asked for. */
constexpr uint32_t kWorstMatchLen = cudec_detail::kGDeflateMinMatchLen;
constexpr uint32_t kWorstLengthIndex =
    kWorstLengthSym - 257u; /* into GDeflateLengthBase/Extra */

/* One group is one round of reservations across all 32 lanes, followed by the
 * round of retirements that pays for them. */
constexpr uint32_t kWorstGroupMatches = cudec_detail::kGDeflateNumStreams;
constexpr uint32_t kWorstGroupBytes = kWorstGroupMatches * kWorstMatchLen;

/* The literal run the page opens with, and the group count that follows it.
 * The run exists because a match may not reach before the page's own output,
 * so the cheapest deep distance symbol's base is the floor on how much output
 * must exist before the first match; the run is the smallest whole number of
 * groups plus the page's own remainder that clears that base. Derived from the
 * table rather than written down, so a base that moved moves the run with it.
 */
uint32_t WorstLeadBytes() {
    uint32_t lead = kPageBytes % kWorstGroupBytes;
    const uint32_t need = cudec_detail::GDeflateDistBase(kWorstFirstDeepDist);
    while (lead < need) {
        lead += kWorstGroupBytes;
    }
    return lead;
}

uint32_t WorstGroups() {
    return static_cast<uint32_t>((kPageBytes - WorstLeadBytes()) /
                                 kWorstGroupBytes);
}

/* Big enough that the timed run is not measuring the harness, far smaller than
 * the level sweep's corpora because every page here is emitted a bit at a time
 * as well as decoded. The row therefore is NOT scale-comparable with the
 * level-sweep rows, and the provenance string says so where the number is
 * printed. */
constexpr size_t kWorstRoundsPages = 512;

/* The floor the SINGLE-BLOCK corpus is required to clear, in refills per
 * decoded byte, and what sits under it. The many-block row carries its own,
 * beside its own digests, at kWorstHeadersRefillFloor below.
 *
 * A refill hands a lane 32 bits, so a page spending b bits per decoded byte
 * sits at b/32; a stored block spends 8 and lands at 0.250. This corpus
 * spends a maximum-length length symbol, sixteen extra bits, a maximum-length
 * distance symbol and up to fourteen more extra bits for every three decoded
 * bytes, and measures 0.6152. Weakened by hand and watched being refused: the
 * same generator with both codes balanced instead of maximal measures 0.4483,
 * and with every codeword capped at twelve bits instead of fifteen it
 * measures 0.5523. The floor sits above the second of those, so it refuses a
 * page set that kept the matches and dropped the codeword depth - which the
 * 0.55 it was first set to did not.
 *
 * A FLOOR ALONE CANNOT CARRY THE DEPTH CLAIM AND IS NOT ASKED TO. Most of a
 * match's cost is its extra-bit fields, which are the same width whatever the
 * codewords are, so the density separates the two shapes by a tenth rather
 * than by a mile. What refuses a shallow code outright is
 * WorstCodeIsMaximal, which reads the depth of every symbol the page emits
 * off the vector it emitted it from. */
constexpr double kWorstRefillFloor = 0.60;

/* A canonical code is decodable only if its lengths spend the codespace
 * exactly. Read off the vector the generator built rather than argued from the
 * constants it was built from: the two are the same thing only while nobody
 * has edited one of them. Integer arithmetic, so "exactly" is exact. */
bool WorstCodeIsComplete(const std::vector<unsigned char>& lens) {
    uint64_t spent = 0;
    for (size_t i = 0; i < lens.size(); i++) {
        if (lens[i] == 0) {
            continue;
        }
        if (lens[i] > kWorstCodeBits) {
            return false;
        }
        spent += static_cast<uint64_t>(1) << (kWorstCodeBits - lens[i]);
    }
    return spent == (static_cast<uint64_t>(1) << kWorstCodeBits);
}

/* THE DEPTH CLAIM, AS A REFUSAL. The provenance line this corpus prints says
 * every symbol in it sits at the maximum codeword length, and the density
 * floor cannot carry that sentence: extra-bit fields are most of a match's
 * cost and they do not move with the codewords, so a twelve-bit code loses
 * only a tenth of the density. This reads the length of every symbol the body
 * actually emits out of the vector it will be encoded from, so a generator
 * that shallowed its codes is refused rather than printing a claim about
 * bytes it did not produce. */
bool WorstCodeIsMaximal(const std::vector<cudec_test::BodyToken>& body,
                        const std::vector<unsigned char>& litlen_lens,
                        const std::vector<unsigned char>& dist_lens) {
    for (size_t i = 0; i < body.size(); i++) {
        const std::vector<unsigned char>& lens =
            body[i].from_dist ? dist_lens : litlen_lens;
        if (body[i].sym >= lens.size() || lens[body[i].sym] != kWorstCodeBits) {
            std::fprintf(stderr,
                         "the worst-rounds body emits %s symbol %u at %u bits "
                         "and this corpus reports every symbol at the maximum "
                         "%u; the page would be valid and the report would be "
                         "false%s",
                         body[i].from_dist ? "distance" : "literal/length",
                         body[i].sym,
                         body[i].sym >= lens.size()
                             ? 0u
                             : static_cast<uint32_t>(lens[body[i].sym]),
                         kWorstCodeBits, "\n");
            return false;
        }
    }
    return true;
}

/* The literal/length code lengths described above. */
std::vector<unsigned char> WorstLitLenLengths() {
    std::vector<unsigned char> lens(kWorstLitLenSyms, 0);
    for (uint32_t i = 0; i < kWorstLitLenSpine; i++) {
        lens[i] = static_cast<unsigned char>(i + 1u);
    }
    for (uint32_t i = 0; i < kWorstDeepLiterals; i++) {
        lens[kWorstFirstDeepLiteral + i] =
            static_cast<unsigned char>(kWorstCodeBits);
    }
    lens[cudec_test::kEndOfBlock] = static_cast<unsigned char>(kWorstCodeBits);
    lens[kWorstLengthSym] = static_cast<unsigned char>(kWorstCodeBits);
    return lens;
}

/* The distance code lengths described above. */
std::vector<unsigned char> WorstDistLengths() {
    std::vector<unsigned char> lens(kWorstDistSyms, 0);
    for (uint32_t i = 0; i < kWorstDistSpine; i++) {
        lens[i] = static_cast<unsigned char>(i + 1u);
    }
    for (uint32_t i = 0; i < kWorstDistDeep; i++) {
        lens[kWorstFirstDeepDist + i] =
            static_cast<unsigned char>(kWorstCodeBits);
    }
    return lens;
}

/* The most expensive distance symbol whose match can legally be taken at this
 * output position. Every deep symbol costs a maximum codeword, so what
 * separates them is the extra-bit field, which grows with the base - and the
 * base is exactly what the output has to be long enough to reach back over
 * (src/gdeflate_block.h refuses a match that begins before the page's own
 * output). Walking down from the top is therefore walking down from the most
 * expensive. */
bool WorstDistSymFor(uint64_t out_pos, uint32_t* sym) {
    for (uint32_t s = kWorstDistSyms; s-- > kWorstFirstDeepDist;) {
        if (cudec_detail::GDeflateDistBase(s) <= out_pos) {
            *sym = s;
            return true;
        }
    }
    return false;
}

/* How many dynamic blocks the header row cuts a page into, DERIVED from the
 * page's own shape rather than chosen: a block has to carry whole groups, or
 * a lane would hold a reservation across a block boundary that the drain
 * cannot retire, so one group is the smallest a block can be and the group
 * count is therefore the most block headers a page of this construction can
 * hold. The frequency is the maximum for the same reason the codewords are the
 * maximum - this is the adversarial extreme of the ingredient, not a sample
 * of it. */
uint32_t WorstHeaderBlocks() { return WorstGroups(); }

/* One page: an opening literal run, then `WorstGroups()` groups of 32 minimum
 * matches, then end-of-block - cut into `blocks` dynamic blocks, each carrying
 * whole groups and stating its own end.
 *
 * `blocks` IS THE ONE INGREDIENT THAT VARIES, and the two rows this file
 * reports are the two ends of it: 1 is #226's single final block per page, and
 * WorstHeaderBlocks() is #430's page of many. Everything else - the code
 * vectors, the literal run, the group construction, the distance choice - is
 * shared, because two generators would be two things to keep adversarial and
 * only one of them would be read when the other drifted.
 *
 * A BLOCK BOUNDARY COSTS OUTPUT NOTHING AND COSTS BITS A HEADER, which is why
 * cutting the page changes the density without changing the bytes: the source
 * a page decodes to does not depend on `blocks` at all, and the two rows
 * therefore differ in exactly the ingredient they are being compared on.
 *
 * THE GROUP SHAPE IS THE DECODER'S AND NOT A CHOICE. A body token at position
 * p rides lane p mod 32, and the distance a reserved match needs is read on
 * that lane's NEXT round - position p + 32. So 32 reservations followed by 32
 * retirements is the only shape in which every lane holds a match for exactly
 * one round, which is what "copies split across two rounds on the same lane"
 * means when every lane does it at once.
 *
 * `page_index` seeds the literal run, so the pages of a corpus differ from one
 * another in their bytes as well as in their index. */
bool BuildWorstPage(size_t page_index, uint32_t blocks,
                    std::vector<unsigned char>* original,
                    std::vector<unsigned char>* page) {
    const std::vector<unsigned char> litlen_lens = WorstLitLenLengths();
    const std::vector<unsigned char> dist_lens = WorstDistLengths();

    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<cudec_test::LenToken> toks = cudec_test::BuildTokens(all);
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    if (!cudec_test::PlanPrecode(toks, /*forced_explicit=*/0, precode_lens,
                                 &num_explicit)) {
        std::fprintf(stderr,
                     "no precode covers the worst-rounds length vector\n");
        return false;
    }

    const uint32_t lead = WorstLeadBytes();
    const uint32_t groups = WorstGroups();
    if (blocks == 0 || blocks > groups) {
        std::fprintf(stderr,
                     "a worst-case page of %u groups cannot be cut into %u "
                     "blocks: a block carries whole groups, so one group is "
                     "the smallest a block can be\n",
                     groups, blocks);
        return false;
    }
    original->assign(kPageBytes, 0);

    /* Every block states the same header. The vectors and the token stream are
     * built once above and copied into each, so a page of many blocks pays the
     * header's bits many times over - which is the cost this row exists to
     * measure - without this generator holding two descriptions of one code. */
    cudec_test::PageBlock proto;
    proto.litlen_lens = litlen_lens;
    proto.dist_lens = dist_lens;
    proto.toks = toks;
    std::memcpy(proto.precode_lens, precode_lens, sizeof(proto.precode_lens));
    proto.num_explicit = num_explicit;

    std::vector<cudec_test::PageBlock> page_blocks;
    page_blocks.reserve(blocks);

    /* The opening run. A fixed LCG seeded by the page index, so every page of
     * a corpus is different bytes and every corpus is reproducible. */
    uint64_t state = 0x9E3779B97F4A7C15ull + page_index;
    uint64_t out_pos = 0;
    uint32_t groups_placed = 0;
    for (uint32_t b = 0; b < blocks; b++) {
        cudec_test::PageBlock blk = proto;
        /* The groups spread as evenly as they divide, remainder to the front.
         * Every block therefore holds at least one, which is what the bound
         * above refused a `blocks` too large for. */
        const uint32_t take =
            groups / blocks + (b < groups % blocks ? 1u : 0u);
        if (b == 0) {
            blk.body.reserve(lead + 2u * take * kWorstGroupMatches + 1u);
            for (uint32_t n = 0; n < lead; n++) {
                state = state * 6364136223846793005ull +
                        1442695040888963407ull;
                const uint32_t sym =
                    kWorstFirstDeepLiteral +
                    static_cast<uint32_t>((state >> 33) % kWorstDeepLiterals);
                blk.body.push_back(cudec_test::BodyToken{sym, false, 0, 0});
                (*original)[out_pos] = static_cast<unsigned char>(sym);
                out_pos++;
            }
        } else {
            blk.body.reserve(2u * take * kWorstGroupMatches + 1u);
        }

        for (uint32_t g = 0; g < take; g++) {
            uint64_t reserved[kWorstGroupMatches];
            uint32_t dist_sym[kWorstGroupMatches];
            for (uint32_t m = 0; m < kWorstGroupMatches; m++) {
                if (!WorstDistSymFor(out_pos, &dist_sym[m])) {
                    std::fprintf(stderr,
                                 "no deep distance symbol reaches back over "
                                 "%llu bytes of output\n",
                                 static_cast<unsigned long long>(out_pos));
                    return false;
                }
                reserved[m] = out_pos;
                out_pos += kWorstMatchLen;
                blk.body.push_back(cudec_test::BodyToken{
                    kWorstLengthSym, false,
                    cudec_detail::GDeflateLengthExtra(kWorstLengthIndex), 0});
            }
            for (uint32_t m = 0; m < kWorstGroupMatches; m++) {
                const uint32_t s = dist_sym[m];
                const uint64_t offset = cudec_detail::GDeflateDistBase(s);
                /* Mirrors GDeflateDoCopy exactly, in the same order the
                 * decoder retires the group, so the expected bytes are what a
                 * decode produces rather than what this generator meant. */
                for (uint32_t i = 0; i < kWorstMatchLen; i++) {
                    (*original)[reserved[m] + i] =
                        (*original)[reserved[m] - offset + i];
                }
                blk.body.push_back(cudec_test::BodyToken{
                    s, true, cudec_detail::GDeflateDistExtra(s), 0});
            }
        }
        groups_placed += take;
        /* Every block states its own end. A group is 32 reservations followed
         * by the 32 retirements that pay for them, so no lane holds a
         * reservation across this boundary and the drain that follows it
         * consumes nothing - which is what makes a block boundary placeable
         * here at all. */
        blk.body.push_back(
            cudec_test::BodyToken{cudec_test::kEndOfBlock, false, 0, 0});
        if (!WorstCodeIsMaximal(blk.body, litlen_lens, dist_lens)) {
            return false;
        }
        page_blocks.push_back(std::move(blk));
    }

    if (groups_placed != groups || out_pos != kPageBytes) {
        std::fprintf(stderr,
                     "the worst-case page places %u of %u groups and produces "
                     "%llu bytes, and the page is %zu\n",
                     groups_placed, groups,
                     static_cast<unsigned long long>(out_pos), kPageBytes);
        return false;
    }
    if (!WorstCodeIsComplete(litlen_lens) ||
        !WorstCodeIsComplete(dist_lens)) {
        std::fprintf(stderr,
                     "the worst-rounds code lengths do not spend the codespace "
                     "exactly, so the page would carry a code no decoder can "
                     "rebuild%s",
                     "\n");
        return false;
    }
    if (!cudec_test::EmitPageBlocks(page_blocks, page)) {
        std::fprintf(stderr, "the page writer refused page %zu\n", page_index);
        return false;
    }
    /* The writer's zero tail is kept at the size the writer declares. It is
     * not slack this corpus may trade for a tidier ratio: the reference
     * refills without a bound check, so the tail is what turns an emission
     * that drifted from the schedule into a read of zeros instead of a read
     * past the end of this vector - and no sanitizer build covers bench/, so
     * nothing else would see it. What the ratio owes instead is disclosure,
     * which is `Corpus::padding_bytes` below. */
    return true;
}

/* The whole corpus, plus the source it is required to decode back to. The
 * source is assembled here rather than handed in, because these pages are not
 * a cut of anything - the bytes exist because the page says them. */
bool BuildWorstCorpus(size_t pages, uint32_t blocks,
                      std::vector<unsigned char>* source, Corpus* corpus) {
    for (size_t i = 0; i < pages; i++) {
        std::vector<unsigned char> original;
        std::vector<unsigned char> page;
        if (!BuildWorstPage(i, blocks, &original, &page)) {
            return false;
        }
        source->insert(source->end(), original.begin(), original.end());
        corpus->originals.push_back(std::move(original));
        corpus->compressed.push_back(std::move(page));
    }
    corpus->original_bytes = 0;
    corpus->compressed_bytes = 0;
    for (size_t i = 0; i < corpus->originals.size(); i++) {
        corpus->original_bytes += corpus->originals[i].size();
        corpus->compressed_bytes += corpus->compressed[i].size();
    }
    corpus->padding_bytes =
        corpus->originals.size() * cudec_test::kGDeflateWriterTailWords * 4u;
    return true;
}

/* What the density lock reads. */
struct RefillDensity {
    uint64_t refills = 0;
    uint64_t out_bytes = 0;
    /* The worst single page, which is what the lock is taken on: an aggregate
     * that clears a floor can hold a page set where most pages are trivial. */
    double worst_page = 0.0;
    uint64_t copies = 0;
    /* How many blocks the pages turned out to hold, and how many of those were
     * dynamic. READ OUT OF THE DECODE (GDeflatePageState::blocks and its
     * per-type census) rather than out of the construction, because a block
     * boundary inside a page is not findable by scanning and only a decode
     * that walked to a block can count it (issue #206). The extremes rather
     * than a mean: a corpus whose pages disagree about their block count is
     * the thing a mean would hide, and the headers row below refuses on
     * exactly that. */
    uint32_t min_blocks = 0;
    uint32_t max_blocks = 0;
    uint64_t blocks = 0;
    uint64_t dynamic_blocks = 0;
};

double PerDecodedByte(const RefillDensity& d) {
    return d.out_bytes == 0 ? 0.0
                            : static_cast<double>(d.refills) /
                                  static_cast<double>(d.out_bytes);
}

/* Walk every page with cudec's own schedule and count the words it hands to
 * lanes. `GDeflateSchedule::cursor` IS that count: it advances by exactly one
 * per refill, in GDeflateEnsure, and by nothing else.
 *
 * The decode is required to produce the source bytes, for the reason
 * CensusCorpus gives about its own walk: a count read off a decode that did
 * not follow this page is a number about something else. It is also required
 * to produce a WHOLE PAGE, which is the constraint the density is maximal
 * under - the fixed per-page cost of the priming round, the header and the
 * drain divides by the output, so a page emitting fewer bytes clears any floor
 * without carrying a single long codeword.
 *
 * The source bound is asserted here rather than inherited from whichever gate
 * ran first, so the gate order stays a matter of cost and not of safety. */
bool MeasureRefillDensity(const std::vector<unsigned char>& source,
                          const Corpus& corpus, RefillDensity* density) {
    size_t offset = 0;
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        const size_t want = corpus.originals[i].size();
        if (want != kPageBytes) {
            std::fprintf(stderr,
                         "page %zu of %s decodes to %zu bytes and this corpus "
                         "is locked per decoded byte, so a page short of the "
                         "%zu-byte page would clear any floor on the fixed "
                         "per-page cost alone\n",
                         i, corpus.name.c_str(), want, kPageBytes);
            return false;
        }
        if (offset + want > source.size()) {
            std::fprintf(stderr,
                         "page %zu of %s runs past the source it is compared "
                         "against\n",
                         i, corpus.name.c_str());
            return false;
        }
        std::vector<unsigned char> out(want);
        cudec_detail::GDeflatePageState st;
        uint64_t produced = 0;
        if (!cudec_detail::GDeflateDecodePage(st, corpus.compressed[i].data(),
                                              corpus.compressed[i].size(),
                                              out.data(), want, &produced)) {
            std::fprintf(stderr,
                         "page %zu of %s was refused by the page decoder, so "
                         "no refill count can be read out of it\n",
                         i, corpus.name.c_str());
            return false;
        }
        if (produced != want ||
            std::memcmp(out.data(), source.data() + offset, want) != 0) {
            std::fprintf(stderr,
                         "page %zu of %s decoded to bytes that are not its "
                         "source, so the walk behind the refill count did not "
                         "follow this page\n",
                         i, corpus.name.c_str());
            return false;
        }
        const double page_density = static_cast<double>(st.s.cursor) /
                                    static_cast<double>(produced);
        if (i == 0 || page_density < density->worst_page) {
            density->worst_page = page_density;
        }
        if (i == 0 || st.blocks < density->min_blocks) {
            density->min_blocks = st.blocks;
        }
        if (i == 0 || st.blocks > density->max_blocks) {
            density->max_blocks = st.blocks;
        }
        density->blocks += st.blocks;
        density->dynamic_blocks +=
            st.type_blocks[cudec_detail::kGDeflateBlockDynamic];
        /* EVERY EMITTED WORD IS CONSUMED, and this is where that stops
         * being a sentence. The page carries the writer's declared zero tail
         * past the words it handed out; the cursor says how far a mirror of
         * the reference walked, so a cursor short of the emitted words means
         * the page is carrying stream nothing reads, and a cursor past them
         * means the emission has drifted from the schedule and the tail is
         * load-bearing rather than padding. Neither is this corpus. */
        const uint64_t page_words = corpus.compressed[i].size() / 4u;
        const uint64_t tail_words = cudec_test::kGDeflateWriterTailWords;
        const uint64_t emitted_words =
            page_words < tail_words ? 0 : page_words - tail_words;
        if (st.s.cursor != emitted_words) {
            std::fprintf(stderr,
                         "page %zu of %s consumed %llu of its %llu emitted "
                         "words; the density this corpus locks is only the "
                         "stream's expansion when every emitted word is "
                         "read%s",
                         i, corpus.name.c_str(),
                         static_cast<unsigned long long>(st.s.cursor),
                         static_cast<unsigned long long>(emitted_words),
                         "\n");
            return false;
        }
        density->refills += st.s.cursor;
        density->out_bytes += produced;
        /* ARITHMETIC OVER THE SHAPE, NOT A COUNT THE DECODE REPORTS.
         * GDeflatePageState carries the live copy slots and the per-type
         * census and no retired-copy counter, so nothing here observes a
         * retirement. Every three bytes past the opening run is one
         * minimum-length match by construction, and this is that construction
         * divided out - printed because it says how much of the page is the
         * deferred-copy path, and labelled so nobody reads it as a
         * measurement. */
        density->copies += (produced - WorstLeadBytes()) / kWorstMatchLen;
        offset += want;
    }
    return true;
}

/* The lock. Validity alone is not enough: a valid-but-easy page set
 * round-trips and leaves CI green while the report still says worst case,
 * which is what this floor exists to refuse. It is taken on the WORST page and
 * not on the mean, because a mean clears the floor while part of the corpus is
 * the easiest page this generator can build: with one page of the 512 built
 * all-literal, the run prints a mean of 0.6149 - above the floor - beside a
 * worst page of 0.4697. */
bool CheckRefillFloor(const char* name, const RefillDensity& density,
                      double floor) {
    if (density.worst_page >= floor) {
        return true;
    }
    std::fprintf(stderr,
                 "the worst page of the %s corpus forces %.4f "
                 "refills per decoded byte, under the %.4f floor this corpus "
                 "exists to hold (the corpus mean is %.4f): it is still a "
                 "valid page set and it is no longer the adversarial one the "
                 "report names\n",
                 name, density.worst_page, floor, PerDecodedByte(density));
    return false;
}

/* THE BLOCK-COUNT CLAIM, AS A REFUSAL, and it is the headers row's equivalent
 * of WorstCodeIsMaximal above. That row's whole reason to exist is that its
 * pages carry many dynamic block headers, and the refill floor cannot carry
 * that sentence on its own: a header is a few hundred bits against a page of
 * half a million, so a generator that collapsed back to one block per page
 * loses a tenth of the density rather than most of it, and a floor low enough
 * to be robust would not notice. This reads the count out of the decode's own
 * census and requires EVERY page to hold exactly what the construction says,
 * so a corpus that stopped emitting the ingredient is named by what it stopped
 * being rather than by a number that drifted. */
bool CheckBlocksPerPage(const char* name, const RefillDensity& density,
                        uint32_t want) {
    if (density.min_blocks == want && density.max_blocks == want) {
        return true;
    }
    std::fprintf(stderr,
                 "the %s corpus holds between %u and %u blocks per page and "
                 "its construction emits %u; the frequent-block-header "
                 "ingredient this row is named for is not in the pages it "
                 "built\n",
                 name, density.min_blocks, density.max_blocks, want);
    return false;
}

/* `level` is negative for a corpus no compressor produced, and `density` is
 * null for every corpus that locks something other than refills. Neither is a
 * default a caller may forget: printing a compression level beside a
 * hand-emitted page would attest a compressor that never ran, and printing a
 * density line for a corpus that measured none would attest a lock that does
 * not exist. */
void PrintReport(const Corpus& corpus, int level,
                 const std::vector<double>& sorted, size_t warmup, size_t runs,
                 const RefillDensity* density, double floor, bool gpu) {
    std::vector<size_t> sizes;
    sizes.reserve(corpus.originals.size());
    for (const auto& original : corpus.originals) {
        sizes.push_back(original.size());
    }
    std::sort(sizes.begin(), sizes.end());
    const double gb = static_cast<double>(corpus.original_bytes) / 1e9;

    char device[256];
    (void)cudec_bench_gpu_device_line(device, sizeof(device));

    std::printf("## bench_gdeflate report\n");
    std::printf("- decoder: CPU oracle, libdeflate_gdeflate_decompress (the "
                "pinned NVIDIA/libdeflate gdeflate fork, commit %s), single "
                "thread%s\n",
                CUDEC_GDEFLATE_COMMIT,
                gpu ? ". The GPU rows below time cudec's own decoder through "
                      "cudec_gdeflate_decompress_batch, and the CPU rows are "
                      "the denominator they are read against"
                    : ". This run timed no cudec kernel - it is the "
                      "denominator alone, and --gpu is what adds the device "
                      "rows");
    std::printf("- host CPU: %s\n", cudec_bench::HostCpuName().c_str());
    std::printf("- CUDA device: %s\n", device);
    std::printf("- cudec: %d\n", cudec_version());
    /* The ratio is taken over the STREAM and never over the buffer: a
     * hand-emitted page carries a zero tail no decoder reads, and counting it
     * would attest bytes that are not stream. It is disclosed on its own line
     * rather than dropped. */
    const size_t stream_bytes = corpus.compressed_bytes - corpus.padding_bytes;
    std::printf("- corpus: %s, %zu pages, %.2f MB original, %.2f MB "
                "compressed (ratio %.4f), %s\n",
                corpus.name.c_str(), corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(stream_bytes) / 1e6,
                static_cast<double>(stream_bytes) /
                    static_cast<double>(corpus.original_bytes),
                corpus.provenance.c_str());
    if (corpus.padding_bytes != 0) {
        std::printf("- padding: %.2f MB on top of the compressed figure "
                    "above, a zero tail per page for the reference's "
                    "unchecked refill; no decoder reads it and the ratio "
                    "does not count it\n",
                    static_cast<double>(corpus.padding_bytes) / 1e6);
    }
    if (level >= 0) {
        std::printf("- granularity: %zu KiB pages, compression level %d\n",
                    kPageBytes / 1024, level);
    } else {
        std::printf("- granularity: %zu KiB pages, emitted by this harness - "
                    "no compressor and therefore no compression level\n",
                    kPageBytes / 1024);
    }
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
    if (density != nullptr) {
        std::printf("- refill density: %.4f refills per decoded byte over "
                    "%llu refills and %llu decoded bytes, worst page "
                    "%.4f, floor %.4f taken on the worst page rather "
                    "than on the mean, counted by cudec's own schedule "
                    "(src/gdeflate_schedule.h) on a decode required to "
                    "reproduce the source and to fill a whole page; this "
                    "is the quantity the corpus is locked on and not a "
                    "timing\n",
                    PerDecodedByte(*density),
                    static_cast<unsigned long long>(density->refills),
                    static_cast<unsigned long long>(density->out_bytes),
                    density->worst_page, floor);
        /* Read out of the decode and not out of the generator, for the reason
         * --blockmix exists: a block boundary inside a page is not findable by
         * scanning, so a walk that stopped at the opening header could report
         * one block per page and be believed. */
        std::printf("- blocks per page: min %u / max %u, %llu blocks over the "
                    "corpus of which %llu dynamic, counted by cudec's own page "
                    "decode (src/gdeflate_block.h) rather than by the "
                    "construction that emitted them\n",
                    density->min_blocks, density->max_blocks,
                    static_cast<unsigned long long>(density->blocks),
                    static_cast<unsigned long long>(density->dynamic_blocks));
        std::printf("- deferred copies: %llu, one per %u decoded bytes past "
                    "the opening literal run - the length round reserves and "
                    "the distance round on the same lane retires. ARITHMETIC "
                    "over the corpus's construction and not a count read out "
                    "of the decode: no decoder in this tree reports a "
                    "retirement\n",
                    static_cast<unsigned long long>(density->copies),
                    kWorstMatchLen);
    }
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

/* The warmup and measured passes, sorted. One decompressor for the whole
 * sequence and one set of destinations, both allocated outside the timed
 * region. A failed decode anywhere ends the sequence rather than shortening
 * it: a corpus that stopped decoding must not be reported as a fast one. */
bool TimeCorpus(const Corpus& corpus, size_t warmup, size_t runs,
                std::vector<double>* times) {
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
    for (size_t i = 0; i < runs && ok; i++) {
        const double seconds = DecodeAllSeconds(d, corpus, &buffers);
        if (seconds < 0) {
            std::fprintf(stderr, "a measured decode failed\n");
            ok = false;
            break;
        }
        times->push_back(seconds);
    }
    libdeflate_free_gdeflate_decompressor(d);
    if (!ok || times->empty()) {
        return false;
    }
    std::sort(times->begin(), times->end());
    return true;
}

/* The device rows for one corpus, printed under the same methodology block as
 * the CPU rows above them so the two cannot be quoted apart. The GPU decode is
 * device-resident (H2D and D2H excluded) and CUDA-event timed, and
 * cudec_bench_gpu_gdeflate refuses to time a batch whose pages did not all
 * decode to their original size - so a number here is never from an unverified
 * decode.
 *
 * THE PAGE HANDED TO THE KERNEL IS THE ONE IN THE CORPUS, TAIL AND ALL. A
 * hand-emitted corpus carries a zero tail per page for the reference's
 * unchecked refill, and the ratio line above holds that apart from the stream.
 * The kernel is handed the whole buffer with its own size, and its refills are
 * bounded by that size rather than by a convention, so the tail is words it
 * never reads. What it does affect is nothing in the number below: throughput
 * here is per DECODED byte, which the tail does not touch.
 *
 * NO PARSE-ONLY ROW, AND THE REASON RATHER THAN A SILENT OMISSION. #228 asks
 * for the split variant if the design admits one. It does not: the two chunk
 * formats get a parse-only ceiling from a template flag in
 * src/chunk_decode.cuh that elides the copies while running the identical
 * parse, and the GDeflate round loop's next round depends on bytes the
 * previous round's copies produced, so eliding them would decode different
 * symbols and ceiling nothing. */
bool PrintGpuRows(const Corpus& corpus, double cpu_p50_seconds, size_t warmup,
                  size_t runs) {
    std::vector<const unsigned char*> comp_ptrs(corpus.compressed.size());
    std::vector<size_t> comp_sizes(corpus.compressed.size());
    std::vector<size_t> orig_sizes(corpus.originals.size());
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        comp_ptrs[i] = corpus.compressed[i].data();
        comp_sizes[i] = corpus.compressed[i].size();
        orig_sizes[i] = corpus.originals[i].size();
    }
    cudec_gpu_result g;
    if (!cudec_bench_gpu_gdeflate(comp_ptrs.data(), comp_sizes.data(),
                                  orig_sizes.data(), corpus.originals.size(),
                                  static_cast<int>(warmup),
                                  static_cast<int>(runs), &g)) {
        std::fprintf(stderr, "GPU bench failed\n");
        return false;
    }
    std::printf("- GPU decode (device-resident, CUDA-event timed, %zu warmup "
                "+ %zu runs, %zu pages, every page verified to its original "
                "size before timing): p50 %.3f ms, %.3f GB/s\n",
                warmup, runs, g.chunks, g.full_ms_p50, g.full_gbps_p50);
    std::printf("- GPU parse-only ceiling: not instantiable for this format - "
                "the round loop's next round reads bytes the previous round's "
                "copies produced, so a variant with the copies elided decodes "
                "different symbols and ceilings nothing. The row is absent "
                "rather than filled with the full decode timed twice\n");
    /* The ratio the milestone is read on: the device number against this
     * report's own CPU denominator, computed from the two rows above rather
     * than from a number quoted out of another run. */
    std::printf("- GPU vs the CPU denominator in this report: %.2fx (CPU p50 "
                "%.3f ms, GPU p50 %.3f ms)\n",
                cpu_p50_seconds * 1e3 / g.full_ms_p50, cpu_p50_seconds * 1e3,
                g.full_ms_p50);
    return true;
}

/* Reports the digest of the corpus it built through `digest_out` rather than
 * judging it here, so the caller can walk every level and name all of the
 * ones that moved. Stopping at the first would report one drifted level and
 * leave the rest unexamined, which reads as "only that one moved". */
bool RunLevel(const std::vector<unsigned char>& source, const std::string& name,
              const std::string& provenance, int level, int want_block_type,
              size_t warmup, size_t runs, bool gpu, uint64_t* digest_out) {
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

    std::vector<double> times;
    if (!TimeCorpus(corpus, warmup, runs, &times)) {
        return false;
    }
    PrintReport(corpus, level, times, warmup, runs, /*density=*/nullptr,
                /*floor=*/0.0, gpu);
    if (gpu && !PrintGpuRows(corpus, cudec_bench::Percentile(times, 50),
                             warmup, runs)) {
        return false;
    }
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

/* `level` is negative on a corpus no compressor produced, and the message says
 * so rather than printing a level that never existed. */
bool CheckDigest(uint64_t actual, const std::string& name, int level,
                 uint64_t expected) {
    if (actual == expected) {
        return true;
    }
    char where[64];
    if (level >= 0) {
        std::snprintf(where, sizeof(where), " at level %d", level);
    } else {
        std::snprintf(where, sizeof(where), " (no compression level)");
    }
    std::fprintf(stderr,
                 "the corpus digest moved on %s%s: "
                 "expected %016llx, built %016llx - the corpus this harness "
                 "constructs is not the one its numbers were recorded on\n",
                 name.c_str(), where,
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
                 "usage: %s [--gpu] [--warmup N] [--runs N] [--assetlike] "
                 "[--blocktypes] [--blockmix] [--worstrounds] "
                 "[--worstheaders] [--selfcheck] "
                 "[corpus files...]\n",
                 argv0);
}

/* The forced-block-type path. Each family is its own corpus at its own
 * level, so this walks the family table instead of the level list the
 * other paths sweep. */
int RunBlocktypes(size_t warmup, size_t runs, bool selfcheck, bool gpu) {
    const size_t pages = selfcheck ? kSelfcheckPages : kBlocktypePages;
    bool ok = true;
    for (size_t i = 0; i < kForcedFamilyCount; i++) {
        const ForcedFamily& f = kForcedFamilies[i];
        const std::vector<unsigned char> source = f.source(pages * kPageBytes);
        uint64_t digest = 0;
        if (!RunLevel(source, f.name, f.provenance, f.level,
                      static_cast<int>(f.want_type), warmup, runs, gpu,
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

/* THE BLOCK-TYPE MIX (issue #206), and what makes it a census rather than a
 * histogram of openings.
 *
 * The specialisation lever this feeds asks whether a second decode path for
 * stored and static blocks pays, and the evidence it rests on is how many
 * blocks of each type the reference actually emits and how many bytes each
 * type produces. A block boundary inside a page is not findable by scanning
 * (masterplan 11.3), so the walk over first block headers that --blocktypes
 * performs stops after one block per page and cannot answer that. Worse, it
 * is biased AGAINST the type this lever cares about most: a stored block's
 * length field is sixteen bits, 65535 is one byte short of a page, so a page
 * of stored data is at least two blocks every time while a page that uses a
 * table can be one. Counting openings undercounts stored blocks by
 * construction.
 *
 * What reaches the second block is a decode, so this walks every page with
 * cudec's own page decoder and reads the census it produces. That census is
 * only worth what the decode is worth, so every page is required to decode
 * to exactly its source bytes here, on top of the reference round trip the
 * corpus already passed - a walk that drifted off the block structure would
 * produce different bytes long before it produced a wrong count.
 *
 * IT REPORTS AND LOCKS NOTHING ABOUT ADVERSARIALITY. This is a measurement of
 * what a compressor emits, not a corpus with a shape to defend, so it carries
 * no density floor and no throughput number. */
struct BlockMix {
    size_t pages = 0;
    uint64_t blocks = 0;
    uint64_t out_bytes = 0;
    uint64_t type_blocks[cudec_detail::kGDeflateBlockTypeCount] = {0, 0, 0};
    uint64_t type_bytes[cudec_detail::kGDeflateBlockTypeCount] = {0, 0, 0};
};

/* Decode every page and fold its census in. Three things are required of each
 * page, and each of them is a different way for the walk to be wrong:
 *
 *  - it decodes to exactly the source bytes, so the walk followed the real
 *    block structure and not a plausible-looking wrong one;
 *  - the per-type counts sum to the page's block count, so no block escaped
 *    the accounting or was counted twice;
 *  - the per-type byte counts sum to the page's output, so every produced
 *    byte is attributed to exactly one block type.
 *
 * The last two are what separate a census from a tally that happens to add
 * up to something. */
bool CensusCorpus(const std::vector<unsigned char>& source,
                  const Corpus& corpus, BlockMix* mix) {
    size_t offset = 0;
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        const std::vector<unsigned char>& page = corpus.compressed[i];
        const size_t want = corpus.originals[i].size();
        std::vector<unsigned char> out(want == 0 ? 1 : want);
        cudec_detail::GDeflatePageState st;
        uint64_t produced = 0;
        if (!cudec_detail::GDeflateDecodePage(st, page.data(), page.size(),
                                              out.data(), want, &produced)) {
            std::fprintf(stderr,
                         "page %zu of %s was refused by the page decoder, so "
                         "no block census can be read out of it\n",
                         i, corpus.name.c_str());
            return false;
        }
        if (produced != want ||
            std::memcmp(out.data(), source.data() + offset, want) != 0) {
            std::fprintf(stderr,
                         "page %zu of %s decoded to bytes that are not its "
                         "source, so the block walk behind the census did not "
                         "follow this page\n",
                         i, corpus.name.c_str());
            return false;
        }
        uint64_t blocks_seen = 0;
        uint64_t bytes_seen = 0;
        for (uint32_t t = 0; t < cudec_detail::kGDeflateBlockTypeCount; t++) {
            blocks_seen += st.type_blocks[t];
            bytes_seen += st.type_bytes[t];
            mix->type_blocks[t] += st.type_blocks[t];
            mix->type_bytes[t] += st.type_bytes[t];
        }
        if (blocks_seen != st.blocks || bytes_seen != produced) {
            std::fprintf(stderr,
                         "page %zu of %s censused %llu blocks and %llu bytes "
                         "against %u blocks and %llu bytes decoded - the "
                         "attribution does not cover the page\n",
                         i, corpus.name.c_str(),
                         static_cast<unsigned long long>(blocks_seen),
                         static_cast<unsigned long long>(bytes_seen),
                         st.blocks,
                         static_cast<unsigned long long>(produced));
            return false;
        }
        mix->blocks += st.blocks;
        mix->out_bytes += produced;
        mix->pages++;
        offset += want;
    }
    return true;
}

double Share(uint64_t part, uint64_t whole) {
    return whole == 0 ? 0.0
                      : 100.0 * static_cast<double>(part) /
                            static_cast<double>(whole);
}

void PrintBlockMix(const Corpus& corpus, int level, const BlockMix& mix) {
    std::printf("## bench_gdeflate block-type mix\n");
    std::printf("- corpus: %s, %zu pages, %.2f MB original, ratio %.4f, %s\n",
                corpus.name.c_str(), corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) /
                    static_cast<double>(corpus.original_bytes),
                corpus.provenance.c_str());
    std::printf("- compressor: the pinned NVIDIA/libdeflate gdeflate fork, "
                "commit %s, compression level %d\n",
                CUDEC_GDEFLATE_COMMIT, level);
    std::printf("- corpus digest: %016llx\n",
                static_cast<unsigned long long>(CorpusDigest(corpus)));
    std::printf("- read by: cudec's own page decoder "
                "(src/gdeflate_block.h), every page required to decode to its "
                "source bytes and every block and byte attributed to a type; "
                "this is a whole-page census and not a walk over openings\n");
    std::printf("- blocks: %llu over %zu pages (%.3f per page), %llu output "
                "bytes\n",
                static_cast<unsigned long long>(mix.blocks), mix.pages,
                mix.pages == 0 ? 0.0
                               : static_cast<double>(mix.blocks) /
                                     static_cast<double>(mix.pages),
                static_cast<unsigned long long>(mix.out_bytes));
    std::printf("| type    | blocks | share of blocks | output bytes | share "
                "of bytes |\n");
    std::printf("| ------- | ------ | --------------- | ------------ | "
                "-------------- |\n");
    for (uint32_t t = 0; t < cudec_detail::kGDeflateBlockTypeCount; t++) {
        std::printf("| %-7s | %6llu | %14.4f%% | %12llu | %13.4f%% |\n",
                    BlockTypeName(t),
                    static_cast<unsigned long long>(mix.type_blocks[t]),
                    Share(mix.type_blocks[t], mix.blocks),
                    static_cast<unsigned long long>(mix.type_bytes[t]),
                    Share(mix.type_bytes[t], mix.out_bytes));
    }
}

/* The selfcheck's two anchors, and they are anchors rather than restatements
 * of what the census just produced.
 *
 * The first is the only block-type guarantee the reference's own header
 * gives: level 0 emits uncompressed blocks by construction. A census that
 * misattributed a type would have to misattribute it here too, where the
 * answer is known from outside this harness.
 *
 * The second is what proves the walk REACHES PAST THE FIRST BLOCK, which is
 * the whole reason this path exists beside --blocktypes. A stored block's
 * length field is sixteen bits and its maximum is one byte short of a page,
 * so a full page of stored data is at least two stored blocks, always. A
 * census that stopped at the opening would report exactly one block per page
 * and fail here. */
bool CheckLevelZeroAnchors(const Corpus& corpus, const BlockMix& mix) {
    const uint64_t stored = mix.type_blocks[cudec_detail::kGDeflateBlockStored];
    if (stored != mix.blocks ||
        mix.type_bytes[cudec_detail::kGDeflateBlockStored] != mix.out_bytes) {
        std::fprintf(stderr,
                     "level 0 censused %llu of %llu blocks as stored and %llu "
                     "of %llu bytes - the reference emits uncompressed blocks "
                     "at level 0 by construction, so the census disagrees "
                     "with the compressor rather than with an expectation\n",
                     static_cast<unsigned long long>(stored),
                     static_cast<unsigned long long>(mix.blocks),
                     static_cast<unsigned long long>(
                         mix.type_bytes[cudec_detail::kGDeflateBlockStored]),
                     static_cast<unsigned long long>(mix.out_bytes));
        return false;
    }
    size_t full_pages = 0;
    for (size_t i = 0; i < corpus.originals.size(); i++) {
        if (corpus.originals[i].size() > kStoredBlockMaxBytes) {
            full_pages++;
        }
    }
    if (mix.blocks < 2 * static_cast<uint64_t>(full_pages)) {
        std::fprintf(stderr,
                     "level 0 censused %llu blocks over %zu pages that each "
                     "hold more than the %zu bytes a stored block's length "
                     "field can express - such a page cannot be one block, so "
                     "this walk is not reaching past the first one\n",
                     static_cast<unsigned long long>(mix.blocks), full_pages,
                     kStoredBlockMaxBytes);
        return false;
    }
    return true;
}

/* The census path. It is handed the corpus source the level sweep would have
 * used, so the two paths cannot disagree about what a corpus is, and it times
 * nothing: what it produces is a table, and a timing beside it would invite
 * the table to be read as a throughput result. */
int RunBlockmix(const std::vector<unsigned char>& source,
                const std::string& name, const std::string& provenance,
                bool selfcheck, const uint64_t* expected) {
    bool ok = true;
    for (size_t i = 0; i < kLevelCount; i++) {
        Corpus corpus;
        corpus.name = name;
        corpus.provenance = provenance;
        corpus.composition = kCompositionNotAsserted;
        if (!BuildCorpus(source, kLevels[i], &corpus)) {
            return 1;
        }
        if (!CorpusRoundTrips(source, corpus)) {
            return 1;
        }
        if (selfcheck &&
            !CheckDigest(CorpusDigest(corpus), name, kLevels[i], expected[i])) {
            ok = false;
        }
        BlockMix mix;
        if (!CensusCorpus(source, corpus, &mix)) {
            return 1;
        }
        if (selfcheck && kLevels[i] == 0 &&
            !CheckLevelZeroAnchors(corpus, mix)) {
            ok = false;
        }
        PrintBlockMix(corpus, kLevels[i], mix);
        std::printf("\n");
    }
    return ok ? 0 : 1;
}

/* The digests of the four worst-case corpora - each row's four-page selfcheck
 * corpus and the full one its reported row is taken on. They are
 * constants for the reason the level sweep's digests are: this page set is
 * emitted from fixed constants by code in this tree, so a change to the
 * generator, to either length vector or to the page writer's round order moves
 * one of them, and none of those would stop the pages round-tripping.
 *
 * EACH IS CHECKED ON EVERY RUN, and that is a departure from the sibling
 * paths, which check a digest only under --selfcheck. It is deliberate: the
 * full corpus is the one a reported number is taken on, and a reporting path
 * whose corpus nothing pins is a number attesting a page set nobody fixed. */
constexpr uint64_t kWorstRoundsSelfcheckDigest = 0x925326a21c48a952ull;
constexpr uint64_t kWorstRoundsFullDigest = 0x6e2f2850f76891bbull;
constexpr uint64_t kWorstHeadersSelfcheckDigest = 0xdda9f7867d7bd574ull;
constexpr uint64_t kWorstHeadersFullDigest = 0xc79e0a872dba2ca3ull;

/* The floor the many-block row is required to clear, and why it is its own
 * number rather than the single-block row's.
 *
 * A BLOCK HEADER IS BITS THAT PRODUCE NO OUTPUT AT ALL, so cutting a page into
 * many blocks raises refills per decoded byte without moving the bytes the
 * page decodes to. That makes the two rows comparable in exactly one
 * ingredient, and it makes the single-block row's measurement the thing this
 * floor has to sit above: 0.6152 is what the same generator reaches with
 * `blocks` set to 1, measured rather than argued, and a floor under it would
 * be cleared by a generator that had lost its headers entirely. This floor is
 * therefore set between that reading and the one the many-block shape reaches,
 * so the weakening the row is most likely to suffer is refused by the number
 * as well as by CheckBlocksPerPage above. */
constexpr double kWorstHeadersRefillFloor = 0.66;
static_assert(kWorstHeadersRefillFloor > kWorstRefillFloor,
              "a many-block page spends strictly more bits per decoded byte "
              "than the single-block page of the same construction, so a "
              "headers floor at or under the rounds floor would refuse "
              "nothing the rounds row does not already refuse");

/* The worst-case rows. Five gates, all of them before anything is timed and
 * before anything is printed:
 *
 *  - the reference accepts every page and it round-trips to the source. It is
 *    the sole authority on validity and it runs first, which is what #226
 *    asks for. It is safe to hand it a hand-emitted page only because the
 *    page carries the writer's declared zero tail: the reference refills
 *    without a bound check, and that tail is what turns an emission that
 *    drifted into a read of zeros;
 *  - cudec's own decode reproduces the source, fills a whole page, consumes
 *    exactly the emitted words, and yields the refill count;
 *  - every page holds exactly the blocks the construction says, counted out of
 *    that same decode. It runs BEFORE the floor because it names the
 *    ingredient a drifted corpus stopped carrying, where the floor names a
 *    number that moved;
 *  - the refill density clears its floor, on the worst page and not the mean;
 *  - the digest matches the pin for the corpus size that was built.
 *
 * A corpus that drifted is therefore named by what it stopped being before it
 * is named by a number that moved, and a corpus that failed any of the five
 * prints no report at all - a throughput row beside a refused gate is the one
 * thing this path may not emit.
 *
 * ONE FUNCTION FOR BOTH ROWS. They differ in the block count, the floor, the
 * digests and the sentence they print about themselves, and in nothing else;
 * a second copy of the five gates would be the place one of them silently
 * stopped running. */
int RunWorstShape(const char* name, const std::string& provenance,
                  uint32_t blocks, double floor, uint64_t selfcheck_digest,
                  uint64_t full_digest, size_t warmup, size_t runs,
                  bool selfcheck, bool gpu) {
    const size_t pages = selfcheck ? kSelfcheckPages : kWorstRoundsPages;
    Corpus corpus;
    corpus.name = name;
    corpus.provenance = provenance;
    corpus.composition = kCompositionNotAsserted;

    std::vector<unsigned char> source;
    if (!BuildWorstCorpus(pages, blocks, &source, &corpus)) {
        return 1;
    }
    if (!CorpusRoundTrips(source, corpus)) {
        return 1;
    }
    RefillDensity density;
    if (!MeasureRefillDensity(source, corpus, &density)) {
        return 1;
    }
    if (!CheckBlocksPerPage(name, density, blocks)) {
        return 1;
    }
    if (!CheckRefillFloor(name, density, floor)) {
        return 1;
    }
    /* Every page opens with the dynamic block this generator emits. That is
     * asserted rather than narrated, so a generator that started emitting
     * something else is caught here and not in the prose. What the walk can
     * say beyond the opening is decided by BFINAL and printed by
     * AssertComposition itself: on the single-block row the opening block IS
     * the page, and on the many-block row the walk reports a lower bound while
     * the census above carries the count.  */
    if (!AssertComposition(&corpus, kBlockDynamic)) {
        return 1;
    }
    if (!CheckDigest(CorpusDigest(corpus), corpus.name, /*level=*/-1,
                     selfcheck ? selfcheck_digest : full_digest)) {
        return 1;
    }
    std::vector<double> times;
    if (!TimeCorpus(corpus, warmup, runs, &times)) {
        return 1;
    }
    PrintReport(corpus, /*level=*/-1, times, warmup, runs, &density, floor,
                gpu);
    if (gpu && !PrintGpuRows(corpus, cudec_bench::Percentile(times, 50),
                             warmup, runs)) {
        return 1;
    }
    return 0;
}

/* The three ingredients of MASTERPLAN section 13.5 that do not need a block
 * boundary, in a page that is one final dynamic block (issue #226). */
int RunWorstRounds(size_t warmup, size_t runs, bool selfcheck, bool gpu) {
    return RunWorstShape(
        "worst-rounds",
        "HAND-CONSTRUCTED by this harness and validated by the pinned "
        "gdeflate fork, not produced by any compressor: one final dynamic "
        "block per page, every symbol in it at DEFLATE's maximum codeword "
        "length, and a body of minimum-length matches asked for through the "
        "sixteen-extra-bit length symbol, so the deferred-copy path runs once "
        "per three decoded bytes; ADVERSARIAL, the M4 security-posture row "
        "and never a headline throughput figure, and NOT scale-comparable "
        "with this harness's other corpora, which are recorded on their own "
        "invocations and are several times this one in pages",
        /*blocks=*/1, kWorstRefillFloor, kWorstRoundsSelfcheckDigest,
        kWorstRoundsFullDigest, warmup, runs, selfcheck, gpu);
}

/* The same page with the fourth ingredient added (issue #430): the same
 * output, the same symbols and the same matches, cut into as many dynamic
 * blocks as whole groups of matches allow, so every group pays a block header
 * and the one-active-lane header rounds are a real share of the total. */
int RunWorstHeaders(size_t warmup, size_t runs, bool selfcheck, bool gpu) {
    return RunWorstShape(
        "worst-headers",
        "HAND-CONSTRUCTED by this harness and validated by the pinned "
        "gdeflate fork, not produced by any compressor: the worst-rounds page "
        "cut into one dynamic block per group of 32 minimum-length matches, "
        "so it carries that row's maximum-length codewords and its "
        "deferred-copy-per-three-bytes body AND a dynamic block header every "
        "96 decoded bytes; ADVERSARIAL, the M4 security-posture row and never "
        "a headline throughput figure, and NOT scale-comparable with this "
        "harness's other corpora, which are recorded on their own invocations "
        "and are several times this one in pages",
        WorstHeaderBlocks(), kWorstHeadersRefillFloor,
        kWorstHeadersSelfcheckDigest, kWorstHeadersFullDigest, warmup, runs,
        selfcheck, gpu);
}

}  // namespace

int main(int argc, char** argv) {
    size_t warmup = 3;
    size_t runs = 30;
    bool selfcheck = false;
    bool assetlike = false;
    bool blocktypes = false;
    bool blockmix = false;
    bool worstrounds = false;
    bool worstheaders = false;
    bool gpu = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--gpu") {
            gpu = true;
        } else if (arg == "--assetlike") {
            assetlike = true;
        } else if (arg == "--blocktypes") {
            blocktypes = true;
        } else if (arg == "--blockmix") {
            blockmix = true;
        } else if (arg == "--worstrounds") {
            worstrounds = true;
        } else if (arg == "--worstheaders") {
            worstheaders = true;
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

    if ((worstrounds || worstheaders) &&
        (assetlike || blocktypes || blockmix || !files.empty())) {
        std::fprintf(stderr,
                     "--worstrounds and --worstheaders emit their own corpora "
                     "and no compressor runs on those paths; do not pass "
                     "corpus files or any of the compressor-driven modes with "
                     "them\n");
        return 2;
    }

    /* Two shapes of one construction, differing in the ingredient each is
     * named for, so a run that asked for both would print two rows and leave
     * the reader to work out which quantity each locked. They are asked for
     * one at a time. */
    if (worstrounds && worstheaders) {
        std::fprintf(stderr,
                     "--worstrounds and --worstheaders are the same generator "
                     "at one block per page and at one block per group of "
                     "matches; each locks its own refill floor and its own "
                     "digests, so they are run separately\n");
        return 2;
    }

    if (blockmix && blocktypes) {
        std::fprintf(stderr,
                     "--blockmix censuses the level sweep's corpora and "
                     "--blocktypes builds forced-type corpora of its own; "
                     "they answer different questions and cannot be asked "
                     "for at once\n");
        return 2;
    }

    /* The selfcheck is the rot check the GPU-less CI runner runs, so it stays
     * CPU-only by refusal rather than by convention: a --gpu --selfcheck that
     * quietly dropped the device rows would report green on a runner where
     * nothing device-side ran, which is the shape of a check that has stopped
     * checking. */
    if (selfcheck && gpu) {
        std::fprintf(stderr, "--gpu and --selfcheck are exclusive: the "
                             "selfcheck runs on the GPU-less runner\n");
        return 2;
    }

    /* --blockmix censuses the block types of a corpus and times no decode, so
     * it has no row a device number would go beside. Refused rather than
     * ignored, for the reason above: a flag that is accepted and does nothing
     * is read afterwards as a measurement that was taken. */
    if (blockmix && gpu) {
        std::fprintf(stderr, "--gpu and --blockmix are exclusive: the census "
                             "times no decode, so there is no row for a "
                             "device number to go beside\n");
        return 2;
    }

    if (selfcheck) {
        /* Short and fixed, so the rot check stays fast on the GPU-less runner
         * and its verdict is a digest rather than a timing. */
        warmup = 0;
        runs = 1;
    }

    if (blocktypes) {
        return RunBlocktypes(warmup, runs, selfcheck, gpu);
    }

    if (worstrounds) {
        return RunWorstRounds(warmup, runs, selfcheck, gpu);
    }

    if (worstheaders) {
        return RunWorstHeaders(warmup, runs, selfcheck, gpu);
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

    if (blockmix) {
        return RunBlockmix(source, name, provenance, selfcheck, expected);
    }

    bool ok = true;
    for (size_t i = 0; i < kLevelCount; i++) {
        uint64_t digest = 0;
        if (!RunLevel(source, name, provenance, kLevels[i],
                      /*want_block_type=*/-1, warmup, runs, gpu, &digest)) {
            return 1;
        }
        if (selfcheck && !CheckDigest(digest, name, kLevels[i], expected[i])) {
            ok = false;
        }
        std::printf("\n");
    }
    return ok ? 0 : 1;
}
