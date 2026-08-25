/* The Snappy benchmark harness: the CPU denominator every GPU number is read
 * against, and - under --gpu (issue #167) - the device rows themselves. The
 * default run measures the reference decoder alone and says so in its own
 * report rather than leaving a reader to infer it (docs/MASTERPLAN.md
 * section 5, honest numbers).
 *
 * Two corpus shapes, because the batch API and the format disagree about
 * what a unit is. Snappy's own framing is a whole stream per file; the
 * shape this library decodes is a page, which the columnar formats emit at
 * 64 KiB. A denominator taken on one shape does not transfer to the other,
 * so both are built from the same bytes and reported separately.
 *
 * Every corpus carries a digest of what was actually built, so a run that
 * silently compressed something else cannot be mistaken for a comparable
 * one. */
#include "bench_stats.h"
#include "cudec.h"
#include "fixtures.h"
#include "gpu_bench.h"
#include "literal_hist.h"
#include "snappy_block.h"
#include "xxhash64.h"

#include <snappy.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

/* The page size the columnar formats emit and the batch API targets. */
constexpr size_t kChunkBytes = 65536;

constexpr size_t kMaxRuns = 1000000;

/* The selfcheck source. Compressible enough that the streams are not all
 * literals, several chunks long at the chunked shape, and from a fixed PRNG
 * so a failure reproduces. */
constexpr size_t kSelfcheckBytes = 3u << 20;

/* The maximum-element-density corpus (issue #166). Same 64 KiB unit and the
 * same ~200 MB scale as the LZ4 worst-4Bmatch rows, so the two adversarial
 * numbers are directly comparable once a Snappy kernel exists. */
constexpr size_t kWorstChunks = 3200;
constexpr size_t kWorstSelfcheckChunks = 4;

/* Two floors, because the corpus can stop being adversarial in two ways and
 * neither of them stops it decoding.
 *
 * A generator whose copies grew longer still emits one element per two
 * compressed bytes, so the element rate alone would not notice; what moves is
 * the compressed share, from a half toward a fifth. A generator that switched
 * to a costlier element form keeps the compressed share and loses the element
 * rate. The ideal stream sits at 0.500 on both, and the slack below is there
 * for the preamble and the tail literal rather than for drift. */
constexpr double kWorstMinCompressedShare = 0.49;
constexpr double kWorstMinElementsPerByte = 0.49;

/* The parse-bound corpus (issue #119). It decodes to the same block as the
 * one above and encodes it the expensive way: every output byte is its own
 * length-1 literal element, so the element rate per COMPRESSED byte is
 * identical at one per two and the element rate per DECODED byte is four
 * times higher. Throughput here is reported per decoded byte, which is why
 * the two shapes have to be locked separately - the copy corpus says nothing
 * about the margin this one measures.
 *
 * Two locks again, and they are not the copy corpus's two. The compressed
 * share is useless here because this stream expands rather than compresses;
 * what pins the shape is the element rate against the compressed bytes and
 * the decoded bytes each element produces. A generator whose literals grew
 * longer keeps producing one decoded byte per compressed pair on average but
 * costs three compressed bytes for two output bytes, so the element rate
 * falls to a third; a generator that reverted to copies keeps the element
 * rate at a half and produces four decoded bytes each. The ideal stream sits
 * at 0.500 and 1.000 exactly, and the slack is for the varint preamble. */
constexpr double kWorstLitMinElementsPerByte = 0.49;
constexpr double kWorstLitMaxBytesPerElement = 1.02;

/* The minimum-cost Snappy element: a one-byte tag plus one offset byte, four
 * output bytes. Tag 0x01 is kind 01 (copy, 1-byte offset) with the length
 * field zero, which the format reads as 4, and the three high bits zero, so
 * the offset is the following byte alone. */
constexpr unsigned char kMinCopyTag = 0x01;
constexpr unsigned char kMinCopyOffset = 0x01;
constexpr size_t kMinCopyOutput = 4;

/* The costliest way to carry one output byte: kind 00 (literal) with the
 * length field zero, which the format reads as a length of 1, followed by the
 * byte itself. Two compressed bytes, one decoded byte, one element. */
constexpr unsigned char kLen1LiteralTag = 0x00;

/* Both constructed corpora decode to a block of this byte, so the copy corpus
 * is one long run and the literal corpus is the same bytes spelled out. */
constexpr unsigned char kConstructedSeed = 0xA5;

enum class Shape {
    /* One Snappy raw stream per input file. */
    kWhole,
    /* One Snappy raw stream per 64 KiB of input. */
    kChunked,
};

struct Corpus {
    std::string name;
    Shape shape = Shape::kWhole;
    std::vector<std::vector<unsigned char>> originals;
    std::vector<std::vector<unsigned char>> compressed;
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    /* Printed verbatim in the methodology block, so it must stay true for
     * whichever corpus ran. */
    std::string provenance =
        "compressed in-harness by the pinned snappy oracle";
};

const char* ShapeName(Shape shape) {
    return shape == Shape::kWhole ? "whole-file streams"
                                  : "64 KiB-chunked streams";
}

/* The corpus lock.
 *
 * What has to be caught is drift: a compressor pin that moved, a chunking
 * rule that changed, an input file that is not the one the report names.
 * All three change the produced streams, so the digest runs over those and
 * not over the inputs - the inputs are already pinned by the manifest
 * bench/get-corpora.sh writes, and adding a second fetch path or a second
 * input lock here would be two authorities on one fact.
 *
 * The digest is a fold rather than one hash over the concatenation, so it
 * costs 16 bytes per stream instead of a second copy of the corpus: each
 * stream contributes its length and its own XXH64, little-endian, in
 * corpus order, and the reported digest is the XXH64 of that array.
 * Reordering, retruncating or recompressing any stream moves it.
 *
 * XXH64 and not SHA-256, stated so nobody reads more into it than is
 * there: this is a drift detector over data the harness just built, not a
 * defence against a chosen collision, and it is the hash already in the
 * tree (src/xxhash64.h) rather than a new dependency for a bench. */
void AppendLe64(uint64_t value, std::vector<unsigned char>* out) {
    for (unsigned i = 0; i < 8; i++) {
        out->push_back(static_cast<unsigned char>(value >> (i * 8)));
    }
}

uint64_t CorpusDigest(const Corpus& corpus) {
    std::vector<unsigned char> fold;
    fold.reserve(corpus.compressed.size() * 16);
    for (const auto& stream : corpus.compressed) {
        AppendLe64(stream.size(), &fold);
        AppendLe64(cudec_detail::Xxh64(stream.data(), stream.size()), &fold);
    }
    return cudec_detail::Xxh64(fold.data(), fold.size());
}

bool AppendFile(const std::string& path, Shape shape, Corpus* corpus) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open corpus file: %s\n", path.c_str());
        return false;
    }
    const size_t before = corpus->originals.size();
    if (shape == Shape::kChunked) {
        while (true) {
            std::vector<unsigned char> chunk(kChunkBytes);
            in.read(reinterpret_cast<char*>(chunk.data()),
                    static_cast<std::streamsize>(kChunkBytes));
            const std::streamsize got = in.gcount();
            if (got <= 0) {
                break;
            }
            chunk.resize(static_cast<size_t>(got));
            corpus->originals.push_back(std::move(chunk));
        }
    } else {
        std::vector<unsigned char> whole;
        char buffer[1 << 16];
        while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
            whole.insert(whole.end(), buffer, buffer + in.gcount());
        }
        if (!whole.empty()) {
            corpus->originals.push_back(std::move(whole));
        }
    }
    /* Fail closed on I/O trouble and on zero contribution, per FILE rather
     * than over the accumulated corpus: the accumulated test goes vacuous
     * from the second argument on, and a file that contributed nothing must
     * never end up attested in the methodology block. */
    if (in.bad()) {
        std::fprintf(stderr, "read error in corpus file: %s\n", path.c_str());
        return false;
    }
    if (corpus->originals.size() == before) {
        std::fprintf(stderr, "corpus file contributed no data: %s\n",
                     path.c_str());
        return false;
    }
    return true;
}

void CompressAll(Corpus* corpus) {
    for (const auto& original : corpus->originals) {
        corpus->compressed.push_back(SnappyCompressBlock(original));
    }
}

/* The byte counts every report line divides by, taken from the streams that
 * are actually there. Separate from CompressAll because one corpus is not
 * compressed by the reference at all: the adversarial stream below is
 * constructed byte by byte, and a tally that only ran inside the compressor
 * would report zeroes for it. */
void Tally(Corpus* corpus) {
    corpus->original_bytes = 0;
    corpus->compressed_bytes = 0;
    for (size_t i = 0; i < corpus->originals.size(); i++) {
        corpus->original_bytes += corpus->originals[i].size();
        corpus->compressed_bytes += corpus->compressed[i].size();
    }
}

void AppendVarint32(size_t value, std::vector<unsigned char>* out) {
    while (value >= 0x80) {
        out->push_back(static_cast<unsigned char>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    out->push_back(static_cast<unsigned char>(value));
}

/* Builds one adversarial-but-valid Snappy stream: back-to-back minimum-cost
 * copies at offset 1. That is the maximum element density the format can
 * carry, one element per two compressed bytes and one per four decoded
 * bytes, and it is where a per-element decoder floors.
 *
 * The reference compressor never emits it - it extends an offset-1 run into
 * one long copy, which is the best case rather than the worst - so the stream
 * is constructed here and the oracle passes verdict on it before anything is
 * timed.
 *
 * Wire layout: the varint preamble, one literal byte to give the first copy
 * something to read, then {tag, offset} pairs, then a literal tail of
 * whatever the four-byte copies could not reach. Every byte of the output is
 * the same seed, so the whole block is one run and the decode is bounded by
 * element count rather than by bytes moved. */
bool BuildWorstDensityBlock(size_t out_bytes,
                            std::vector<unsigned char>* original,
                            std::vector<unsigned char>* compressed,
                            size_t* elements) {
    constexpr unsigned char kSeed = kConstructedSeed;
    /* The literal seed plus one copy is the smallest stream that is still the
     * shape this corpus is named for. Below it the tail would be the whole
     * block, so refuse loudly rather than return something that is not
     * adversarial. */
    constexpr size_t kMinBytes = 256;
    /* A literal run of at most this many bytes is encoded by the one-byte
     * literal tag alone, which is what the tail below relies on. */
    constexpr size_t kMaxTailLiteral = 60;

    if (out_bytes < kMinBytes) {
        std::fprintf(stderr, "the max-density block needs at least %zu output "
                             "bytes, got %zu\n",
                     kMinBytes, out_bytes);
        return false;
    }

    original->assign(out_bytes, kSeed);

    std::vector<unsigned char>& c = *compressed;
    c.clear();
    AppendVarint32(out_bytes, &c);
    c.push_back(0x00); /* literal, length 1 */
    c.push_back(kSeed);
    size_t produced = 1;
    *elements = 1;

    while (produced + kMinCopyOutput <= out_bytes) {
        c.push_back(kMinCopyTag);
        c.push_back(kMinCopyOffset);
        produced += kMinCopyOutput;
        (*elements)++;
    }

    const size_t tail = out_bytes - produced;
    if (tail > kMaxTailLiteral) {
        std::fprintf(stderr, "the max-density tail is %zu bytes, which this "
                             "construction does not encode\n",
                     tail);
        return false;
    }
    if (tail != 0) {
        c.push_back(static_cast<unsigned char>((tail - 1) << 2));
        c.insert(c.end(), tail, kSeed);
        (*elements)++;
    }
    return true;
}

/* Builds one adversarial-but-valid Snappy stream at the same element density
 * as the copy chain above and a quarter of its decoded bytes: the declared
 * length, then one length-1 literal element per output byte.
 *
 * The reference compressor never emits it either, and for the opposite
 * reason - a block of one repeated byte is exactly what it collapses into a
 * copy - so this stream is constructed here too and the oracle passes verdict
 * on it before anything is timed.
 *
 * Wire layout: the varint preamble, then {tag, byte} pairs, and no tail,
 * because every element carries exactly one byte and the block divides
 * evenly. */
bool BuildWorstLiteralBlock(size_t out_bytes,
                            std::vector<unsigned char>* original,
                            std::vector<unsigned char>* compressed,
                            size_t* elements) {
    /* The same floor the copy chain refuses below, for the same reason: under
     * it the varint preamble stops being negligible against the element
     * stream and the ratios this corpus is locked on drift on arithmetic
     * rather than on shape. */
    constexpr size_t kMinBytes = 256;

    if (out_bytes < kMinBytes) {
        std::fprintf(stderr, "the parse-bound block needs at least %zu output "
                             "bytes, got %zu\n",
                     kMinBytes, out_bytes);
        return false;
    }

    original->assign(out_bytes, kConstructedSeed);

    std::vector<unsigned char>& c = *compressed;
    c.clear();
    AppendVarint32(out_bytes, &c);
    c.reserve(c.size() + 2 * out_bytes);
    for (size_t i = 0; i < out_bytes; i++) {
        c.push_back(kLen1LiteralTag);
        c.push_back(kConstructedSeed);
    }
    *elements = out_bytes;
    return true;
}

/* Refuses the corpus if it stopped being adversarial, in the two directions
 * the literal chain can drift. Validity is the oracle's job and happens
 * separately; this is the other half, for the same reason the copy corpus
 * gives - a valid-but-easy stream round-trips and would leave the report
 * claiming a worst case it no longer measures. */
bool CheckLiteralDensity(const Corpus& corpus, size_t elements) {
    const double per_compressed = static_cast<double>(elements) /
                                  static_cast<double>(corpus.compressed_bytes);
    const double per_element = static_cast<double>(corpus.original_bytes) /
                               static_cast<double>(elements);
    if (per_compressed < kWorstLitMinElementsPerByte) {
        std::fprintf(stderr,
                     "the parse-bound corpus carries %.4f elements per "
                     "compressed byte, below the %.2f floor: its elements "
                     "cost more than the two-byte length-1 literal\n",
                     per_compressed, kWorstLitMinElementsPerByte);
        return false;
    }
    if (per_element > kWorstLitMaxBytesPerElement) {
        std::fprintf(stderr,
                     "the parse-bound corpus produces %.4f decoded bytes per "
                     "element, above the %.2f ceiling: its elements are no "
                     "longer one output byte each, so the parse work per "
                     "reported byte is not the worst case it reports\n",
                     per_element, kWorstLitMaxBytesPerElement);
        return false;
    }
    return true;
}

/* Refuses the corpus if it stopped being adversarial. Validity is the
 * oracle's job and happens separately; this is the other half, because a
 * valid-but-easy stream round-trips and would leave the report claiming a
 * worst case it no longer measures. */
bool CheckDensity(const Corpus& corpus, size_t elements) {
    const double share = static_cast<double>(corpus.compressed_bytes) /
                         static_cast<double>(corpus.original_bytes);
    const double per_byte = static_cast<double>(elements) /
                            static_cast<double>(corpus.compressed_bytes);
    if (share < kWorstMinCompressedShare) {
        std::fprintf(stderr,
                     "the max-density corpus compressed to %.4f of its "
                     "output, below the %.2f floor: its elements are "
                     "producing more bytes each than the minimum copy, so "
                     "this is no longer the worst case it reports\n",
                     share, kWorstMinCompressedShare);
        return false;
    }
    if (per_byte < kWorstMinElementsPerByte) {
        std::fprintf(stderr,
                     "the max-density corpus carries %.4f elements per "
                     "compressed byte, below the %.2f floor: its elements "
                     "cost more than the minimum copy encoding\n",
                     per_byte, kWorstMinElementsPerByte);
        return false;
    }
    return true;
}

/* Replicates an adversarial block to `chunks` identical chunks. The block is
 * the same whatever the replica count, so the selfcheck exercises the
 * identical construction on a handful of them. Shared by both constructed
 * corpora: they differ in how one block is built and in nothing else, and a
 * second replicator would be a second place for the chunk unit to drift. */
using BlockBuilder = bool (*)(size_t, std::vector<unsigned char>*,
                              std::vector<unsigned char>*, size_t*);

bool BuildConstructedCorpus(BlockBuilder build, size_t chunks, Corpus* corpus,
                            size_t* elements) {
    std::vector<unsigned char> original, compressed;
    size_t per_block = 0;
    if (!build(kChunkBytes, &original, &compressed, &per_block)) {
        return false;
    }
    for (size_t i = 0; i < chunks; i++) {
        corpus->originals.push_back(original);
        corpus->compressed.push_back(compressed);
    }
    *elements = per_block * chunks;
    return true;
}

/* The oracle is the sole authority on validity, and it says so before any
 * timing: a number taken on a stream the reference refuses, or on one that
 * does not round-trip, is a number about nothing. */
bool VerifyCorpus(const Corpus& corpus) {
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        std::vector<unsigned char> decoded;
        if (!SnappyOracleDecodes(corpus.compressed[i], &decoded)) {
            std::fprintf(stderr, "the oracle refuses stream %zu of %s\n", i,
                         corpus.name.c_str());
            return false;
        }
        if (decoded != corpus.originals[i]) {
            std::fprintf(stderr, "stream %zu of %s does not round-trip\n", i,
                         corpus.name.c_str());
            return false;
        }
    }
    return true;
}

/* One timed pass over the whole corpus. The timed region is
 * snappy::RawUncompress alone: the destination buffers are allocated and
 * the declared lengths read outside it, so the number is the decoder's and
 * not the allocator's.
 *
 * Returns a negative duration if any stream fails, so a broken decode can
 * never be reported as a fast one. */
double DecodeAllSeconds(const Corpus& corpus,
                        std::vector<std::vector<char>>* buffers) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        const auto& stream = corpus.compressed[i];
        if (!snappy::RawUncompress(reinterpret_cast<const char*>(stream.data()),
                                   stream.size(), (*buffers)[i].data())) {
            return -1.0;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

bool MakeBuffers(const Corpus& corpus,
                 std::vector<std::vector<char>>* buffers) {
    buffers->clear();
    for (const auto& stream : corpus.compressed) {
        size_t declared = 0;
        if (!snappy::GetUncompressedLength(
                reinterpret_cast<const char*>(stream.data()), stream.size(),
                &declared)) {
            std::fprintf(stderr, "no declared length in a %s stream\n",
                         corpus.name.c_str());
            return false;
        }
        buffers->push_back(std::vector<char>(declared));
    }
    return true;
}

void PrintReport(const Corpus& corpus, const std::vector<double>& sorted,
                 size_t warmup, size_t runs, bool gpu) {
    std::vector<size_t> sizes;
    for (const auto& original : corpus.originals) {
        sizes.push_back(original.size());
    }
    std::sort(sizes.begin(), sizes.end());
    const double to_gbps = static_cast<double>(corpus.original_bytes) / 1e9;

    char device[256];
    (void)cudec_bench_gpu_device_line(device, sizeof(device));

    std::printf("## bench_snappy report\n");
    std::printf("- decoder: CPU oracle, snappy::RawUncompress (google/snappy "
                "%d.%d.%d), single thread%s\n",
                SNAPPY_MAJOR, SNAPPY_MINOR, SNAPPY_PATCHLEVEL,
                gpu ? ". The GPU rows below time cudec's own decoder through "
                      "cudec_snappy_decompress_batch, and the CPU rows are "
                      "the denominator they are read against"
                    : ". This run timed no cudec kernel - it is the "
                      "denominator alone, and --gpu is what adds the device "
                      "rows");
    std::printf("- host CPU: %s\n", cudec_bench::HostCpuName().c_str());
    std::printf("- CUDA device: %s\n", device);
    std::printf("- cudec: %d\n", cudec_version());
    std::printf("- corpus: %s, %s, %zu streams, %.2f MB original, %.2f MB "
                "compressed (ratio %.3f), %s\n",
                corpus.name.c_str(), ShapeName(corpus.shape),
                corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) /
                    static_cast<double>(corpus.original_bytes),
                corpus.provenance.c_str());
    std::printf("- corpus digest: %016llx (XXH64 over per-stream length and "
                "XXH64, little-endian, in corpus order)\n",
                static_cast<unsigned long long>(CorpusDigest(corpus)));
    std::printf("- stream sizes: min %zu / median %zu / max %zu bytes "
                "uncompressed\n",
                sizes.front(), sizes[sizes.size() / 2], sizes.back());
    std::printf("- method: %zu warmup + %zu measured runs, wall clock per "
                "whole-corpus decode; the timed region is "
                "snappy::RawUncompress only (no allocation, no length "
                "parse); every stream round-trip-verified against the "
                "original once before timing; percentiles are nearest-rank\n",
                warmup, runs);
    std::printf("- wall per run: p50 %.3f ms / p90 %.3f ms / p99 %.3f ms\n",
                cudec_bench::Percentile(sorted, 50) * 1e3,
                cudec_bench::Percentile(sorted, 90) * 1e3,
                cudec_bench::Percentile(sorted, 99) * 1e3);
    std::printf("- decode throughput: p50 %.3f GB/s / p90 %.3f GB/s / p99 "
                "%.3f GB/s\n",
                to_gbps / cudec_bench::Percentile(sorted, 50),
                to_gbps / cudec_bench::Percentile(sorted, 90),
                to_gbps / cudec_bench::Percentile(sorted, 99));
}

/* The device rows for one corpus, printed under the same methodology block
 * as the CPU rows above them so the two cannot be quoted apart. The GPU
 * decode is device-resident (H2D/D2H excluded) and CUDA-event timed, and
 * cudec_bench_gpu_snappy refuses to time a batch whose chunks did not all
 * decode to their original size, so a number here is never from an
 * unverified decode.
 *
 * Three decimals on GB/s rather than the one bench_lz4 prints: the
 * whole-file shape puts one warp on each of twelve streams, so its
 * throughput lands two orders of magnitude below the chunked shape's and
 * one decimal rounds the whole row to a single digit. */
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
    if (!cudec_bench_gpu_snappy(comp_ptrs.data(), comp_sizes.data(),
                                orig_sizes.data(), corpus.originals.size(),
                                static_cast<int>(warmup),
                                static_cast<int>(runs), &g)) {
        std::fprintf(stderr, "GPU bench failed\n");
        return false;
    }
    std::printf("- GPU decode (device-resident, CUDA-event timed, %zu warmup "
                "+ %zu runs, %zu chunks, every chunk verified to its original "
                "size before timing): p50 %.3f ms, %.3f GB/s\n",
                warmup, runs, g.chunks, g.full_ms_p50, g.full_gbps_p50);
    std::printf("- GPU parse-only ceiling (copies elided, the identical "
                "lockstep parse through the chunk-decoder template seam): p50 "
                "%.3f ms, %.3f GB/s\n",
                g.parse_only_ms_p50, g.parse_only_gbps_p50);
    /* The ratio the milestone is read on: the device number against this
     * report's own CPU denominator, computed from the two rows above rather
     * than from a number quoted out of another run. */
    std::printf("- GPU vs the CPU denominator in this report: %.2fx (CPU p50 "
                "%.3f ms, GPU p50 %.3f ms)\n",
                cpu_p50_seconds * 1e3 / g.full_ms_p50, cpu_p50_seconds * 1e3,
                g.full_ms_p50);
    return true;
}

/* The literal-length distribution of one corpus (issue #165), walked with
 * the shipped parser so the elements counted are the elements the decoder
 * executes. A stream this harness just built and round-trip-verified must
 * parse, so a refusal is a defect and is reported as one rather than
 * skipped. */
bool PrintLiteralDistribution(const Corpus& corpus) {
    cudec_bench::LiteralHistogram hist;
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        if (!cudec_bench::AccumulateLiteralLengths<cudec_detail::SnappyParser>(
                corpus.compressed[i].data(), corpus.compressed[i].size(),
                corpus.originals[i].size(), &hist)) {
            std::fprintf(stderr,
                         "the parser refused stream %zu of a corpus this "
                         "harness built and verified\n",
                         i);
            return false;
        }
    }
    cudec_bench::PrintLiteralHistogram(hist);
    return true;
}

bool RunCorpus(Corpus* corpus, size_t warmup, size_t runs, bool gpu,
               bool literals) {
    Tally(corpus);
    if (corpus->original_bytes == 0) {
        std::fprintf(stderr, "corpus is empty - nothing to benchmark\n");
        return false;
    }
    if (!VerifyCorpus(*corpus)) {
        return false;
    }
    std::vector<std::vector<char>> buffers;
    if (!MakeBuffers(*corpus, &buffers)) {
        return false;
    }
    for (size_t i = 0; i < warmup; i++) {
        if (DecodeAllSeconds(*corpus, &buffers) < 0) {
            std::fprintf(stderr, "a warmup decode failed\n");
            return false;
        }
    }
    std::vector<double> times;
    for (size_t i = 0; i < runs; i++) {
        const double seconds = DecodeAllSeconds(*corpus, &buffers);
        if (seconds < 0) {
            std::fprintf(stderr, "a measured decode failed\n");
            return false;
        }
        times.push_back(seconds);
    }
    std::sort(times.begin(), times.end());
    PrintReport(*corpus, times, warmup, runs, gpu);
    if (literals && !PrintLiteralDistribution(*corpus)) {
        return false;
    }
    if (gpu && !PrintGpuRows(*corpus, cudec_bench::Percentile(times, 50),
                             warmup, runs)) {
        return false;
    }
    return true;
}

std::vector<unsigned char> MakeSelfcheckSource(size_t bytes) {
    std::vector<unsigned char> out(bytes);
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < bytes; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        /* Runs of a repeating alphabet with occasional noise: copies for the
         * decoder to execute, without collapsing to one long copy. */
        out[i] = (i % 61 == 0) ? static_cast<unsigned char>(state >> 56)
                               : static_cast<unsigned char>('a' + (i / 7) % 26);
    }
    return out;
}

/* The selfcheck's corpus is generated from a fixed PRNG and compressed by
 * the pinned oracle, so both shapes are reproducible byte for byte and
 * their digests are constants. Asserting them is what makes this a rot
 * check rather than a run: a compressor pin that moved, a chunk size that
 * changed, or a generator that lost its noise source all move a digest,
 * and none of them would stop the corpus round-tripping. */
constexpr uint64_t kSelfcheckWholeDigest = 0x82b0f9596c1b94e8ull;
constexpr uint64_t kSelfcheckChunkedDigest = 0xe7c9cffbc38ed2e4ull;

bool CheckDigest(const Corpus& corpus, uint64_t expected) {
    const uint64_t actual = CorpusDigest(corpus);
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr,
                 "the %s selfcheck corpus digest moved: expected %016llx, "
                 "built %016llx - the corpus this harness constructs is not "
                 "the one its numbers were recorded on\n",
                 ShapeName(corpus.shape),
                 static_cast<unsigned long long>(expected),
                 static_cast<unsigned long long>(actual));
    return false;
}

bool ParseCount(const char* text, size_t low, size_t high, size_t* out) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value < low || value > high) {
        return false;
    }
    *out = static_cast<size_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    size_t runs = 30;
    size_t warmup = 3;
    bool selfcheck = false;
    bool whole = false;
    bool chunked = false;
    bool worst = false;
    bool worstlit = false;
    bool gpu = false;
    bool literals = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--selfcheck") {
            selfcheck = true;
        } else if (arg == "--whole") {
            whole = true;
        } else if (arg == "--chunked") {
            chunked = true;
        } else if (arg == "--worst") {
            worst = true;
        } else if (arg == "--worstlit") {
            worstlit = true;
        } else if (arg == "--gpu") {
            gpu = true;
        } else if (arg == "--literals") {
            literals = true;
        } else if (arg == "--runs" && i + 1 < argc) {
            if (!ParseCount(argv[++i], 1, kMaxRuns, &runs)) {
                std::fprintf(stderr, "--runs must be in [1, %zu]\n", kMaxRuns);
                return 2;
            }
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!ParseCount(argv[++i], 0, kMaxRuns, &warmup)) {
                std::fprintf(stderr, "--warmup must be in [0, %zu]\n",
                             kMaxRuns);
                return 2;
            }
        } else if (arg == "--runs" || arg == "--warmup") {
            std::fprintf(stderr, "%s needs a value\n", arg.c_str());
            return 2;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "usage: bench_snappy [--runs N] [--warmup N] "
                                 "[--whole] [--chunked] [--worst] "
                                 "[--worstlit] [--gpu] [--literals] "
                                 "[--selfcheck] "
                                 "[corpus files...]\n");
            return 2;
        } else {
            files.push_back(arg);
        }
    }
    if (selfcheck) {
        warmup = 1;
        runs = 3;
    }

    /* The selfcheck is the rot check the GPU-less CI runner runs, so it stays
     * CPU-only by construction rather than by whoever invokes it remembering
     * to. Refused rather than silently ignored: a run that was asked for
     * device rows and printed none reads as a machine with no device. */
    if (gpu && selfcheck) {
        std::fprintf(stderr, "--gpu and --selfcheck are exclusive: the "
                             "selfcheck runs on the GPU-less runner\n");
        return 2;
    }

    /* The adversarial corpora are constructed rather than compressed, and
     * their shape is the whole point of them, so each takes neither corpus
     * files nor a shape flag and runs on its own. They are also two separate
     * measurements of two separate regimes, so asking for both in one run
     * would print two reports under one methodology block. */
    if (worst && worstlit) {
        std::fprintf(stderr, "--worst and --worstlit are separate corpora and "
                             "separate reports; run one at a time\n");
        return 2;
    }
    if (worst || worstlit) {
        if (!files.empty() || whole || chunked) {
            std::fprintf(stderr, "%s builds its own corpus and has one shape; "
                                 "it takes no files and no shape flag\n",
                         worst ? "--worst" : "--worstlit");
            return 2;
        }
        Corpus corpus;
        corpus.shape = Shape::kChunked;
        if (worst) {
            corpus.name = "max element density (constructed)";
            corpus.provenance =
                "hand-constructed in-harness as back-to-back minimum-cost "
                "copies at offset 1, which the reference compressor never "
                "emits; every stream validated by the pinned snappy oracle "
                "before timing";
        } else {
            corpus.name = "max parse work per output byte (constructed)";
            corpus.provenance =
                "hand-constructed in-harness as one length-1 literal element "
                "per output byte, which the reference compressor never emits; "
                "every stream validated by the pinned snappy oracle before "
                "timing";
        }
        size_t elements = 0;
        if (!BuildConstructedCorpus(
                worst ? &BuildWorstDensityBlock : &BuildWorstLiteralBlock,
                selfcheck ? kWorstSelfcheckChunks : kWorstChunks, &corpus,
                &elements)) {
            return 1;
        }
        Tally(&corpus);
        /* Before the report and before any timing: an easy corpus must not
         * reach a run that would print "worst case" over it. */
        if (!(worst ? CheckDensity(corpus, elements)
                    : CheckLiteralDensity(corpus, elements))) {
            return 1;
        }
        if (!RunCorpus(&corpus, warmup, runs, gpu, literals)) {
            return 1;
        }
        std::printf("- element density: %zu elements, %.4f per compressed "
                    "byte, %.4f decoded bytes each\n",
                    elements,
                    static_cast<double>(elements) /
                        static_cast<double>(corpus.compressed_bytes),
                    static_cast<double>(corpus.original_bytes) /
                        static_cast<double>(elements));
        return 0;
    }

    /* Neither shape named means both, which is the recorded run. */
    if (!whole && !chunked) {
        whole = true;
        chunked = true;
    }

    if (files.empty() && !selfcheck) {
        std::fprintf(stderr, "bench_snappy needs corpus files (the recorded "
                             "run uses bench/corpora/silesia/*); --selfcheck "
                             "runs it on a generated source instead\n");
        return 2;
    }

    const std::vector<unsigned char> generated =
        selfcheck ? MakeSelfcheckSource(kSelfcheckBytes)
                  : std::vector<unsigned char>();

    const Shape shapes[] = {Shape::kWhole, Shape::kChunked};
    for (Shape shape : shapes) {
        if (shape == Shape::kWhole && !whole) {
            continue;
        }
        if (shape == Shape::kChunked && !chunked) {
            continue;
        }
        Corpus corpus;
        corpus.shape = shape;
        if (selfcheck) {
            corpus.name = "generated (selfcheck)";
            corpus.provenance = "generated in-harness from a fixed PRNG, "
                                "compressed by the pinned snappy oracle";
            if (shape == Shape::kChunked) {
                for (size_t off = 0; off < generated.size();
                     off += kChunkBytes) {
                    const size_t take =
                        std::min(kChunkBytes, generated.size() - off);
                    const auto first =
                        generated.begin() + static_cast<std::ptrdiff_t>(off);
                    corpus.originals.push_back(std::vector<unsigned char>(
                        first, first + static_cast<std::ptrdiff_t>(take)));
                }
            } else {
                corpus.originals.push_back(generated);
            }
        } else {
            for (const auto& path : files) {
                if (!AppendFile(path, shape, &corpus)) {
                    return 1;
                }
                const size_t slash = path.find_last_of("/\\");
                corpus.name +=
                    (corpus.name.empty() ? "" : "+") +
                    path.substr(slash == std::string::npos ? 0 : slash + 1);
            }
        }
        CompressAll(&corpus);
        if (!RunCorpus(&corpus, warmup, runs, gpu, literals)) {
            return 1;
        }
        if (selfcheck && !CheckDigest(corpus, shape == Shape::kWhole
                                                  ? kSelfcheckWholeDigest
                                                  : kSelfcheckChunkedDigest)) {
            return 1;
        }
        std::printf("\n");
    }
    return 0;
}
