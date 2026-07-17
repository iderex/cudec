/* The benchmark harness (M0 skeleton): times batch LZ4 block decode
 * through the CPU oracle - the only decoder that exists until M1, which
 * plugs in as a second timed path in this same harness. A report cannot
 * be produced without its methodology block; that is the point
 * (docs/MASTERPLAN.md section 5, honest numbers). */
#include "cudec.h"
#include "fixtures.h"

#include <cuda_runtime.h>
#include <lz4.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

/* The LZ4 block-format ceiling used across the project. */
constexpr size_t kChunkBytes = 65536;

constexpr size_t kMaxRuns = 1000000;

struct Corpus {
    std::string name;
    std::vector<std::vector<unsigned char>> originals;
    std::vector<std::vector<unsigned char>> compressed;
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
};

bool AppendFileChunked(const std::string& path, Corpus* corpus) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open corpus file: %s\n", path.c_str());
        return false;
    }
    const size_t chunks_before = corpus->originals.size();
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
    /* Fail closed on I/O trouble and on zero contribution: a file that
     * adds no chunks (empty, unreadable, a directory) must never end up
     * attested in the methodology block. Per-FILE contribution, not the
     * accumulated corpus - the accumulated check goes vacuous from the
     * second argument on. */
    if (in.bad()) {
        std::fprintf(stderr, "read error in corpus file: %s\n", path.c_str());
        return false;
    }
    if (corpus->originals.size() == chunks_before) {
        std::fprintf(stderr, "corpus file contributed no data: %s\n",
                     path.c_str());
        return false;
    }
    return true;
}

void CompressAll(Corpus* corpus) {
    for (const auto& original : corpus->originals) {
        corpus->compressed.push_back(Lz4CompressBlock(original));
        corpus->original_bytes += original.size();
        corpus->compressed_bytes += corpus->compressed.back().size();
    }
}

/* One measured repetition: decode the whole batch, wall clock. The timed
 * region contains ONLY LZ4_decompress_safe calls into a pre-sized scratch
 * buffer - no buffer clears, no allocation - so the label on the number
 * is the number (OracleDecodes zero-fills its output and is therefore
 * used on the untimed verify pass only). Exits on any decode failure:
 * numbers for a broken decoder are not numbers. */
double DecodeAllSeconds(const Corpus& corpus, unsigned char* scratch) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        const int written = LZ4_decompress_safe(
            reinterpret_cast<const char*>(corpus.compressed[i].data()),
            reinterpret_cast<char*>(scratch),
            static_cast<int>(corpus.compressed[i].size()),
            static_cast<int>(corpus.originals[i].size()));
        if (written < 0 ||
            static_cast<size_t>(written) != corpus.originals[i].size()) {
            std::fprintf(stderr, "chunk %zu failed to decode - refusing to "
                                 "time a broken decoder\n",
                         i);
            std::exit(1);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

std::string HostCpuName() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                return line.substr(colon + 2);
            }
        }
    }
    return "unknown host CPU";
}

std::string CudaDeviceLine() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        return "none visible to this process (the GPU decode path arrives "
               "with M1)";
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return "device query failed";
    }
    int driver = 0;
    int runtime = 0;
    (void)cudaDriverGetVersion(&driver);
    (void)cudaRuntimeGetVersion(&runtime);
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
           std::to_string(prop.minor) + "), driver " +
           std::to_string(driver / 1000) + "." +
           std::to_string((driver % 100) / 10) + ", runtime " +
           std::to_string(runtime / 1000) + "." +
           std::to_string((runtime % 100) / 10);
}

/* Strict count parsing: whole-string decimal in [min, max] - rejects
 * strtoul's silent garbage-to-0 and negative wraparound alike. */
bool ParseCount(const char* text, size_t min, size_t max, size_t* out) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value < min || value > max) {
        return false;
    }
    *out = value;
    return true;
}

/* Nearest-rank with ceiling: never flatters - at 30 runs, p99 is the
 * slowest run, not the second-slowest a floor index would pick. */
double Percentile(const std::vector<double>& sorted, int pct) {
    const size_t rank =
        (sorted.size() * static_cast<size_t>(pct) + 99) / 100;
    return sorted[(rank == 0 ? 1 : rank) - 1];
}

void PrintReport(const Corpus& corpus, const std::vector<double>& sorted,
                 size_t warmup, size_t runs) {
    std::vector<size_t> sizes;
    for (const auto& original : corpus.originals) {
        sizes.push_back(original.size());
    }
    std::sort(sizes.begin(), sizes.end());
    const double to_gbps = static_cast<double>(corpus.original_bytes) / 1e9;

    std::printf("## bench_lz4 report\n");
    std::printf("- decoder: CPU oracle, LZ4_decompress_safe (liblz4 %s), "
                "single thread\n",
                LZ4_versionString());
    std::printf("- host CPU: %s\n", HostCpuName().c_str());
    std::printf("- CUDA device: %s\n", CudaDeviceLine().c_str());
    std::printf("- cudec: %d (stub era - the library decodes nothing yet)\n",
                cudec_version());
    std::printf("- corpus: %s, %zu chunks, %.2f MB original, %.2f MB "
                "compressed (ratio %.3f), compressed in-harness via "
                "LZ4_compress_default\n",
                corpus.name.c_str(), corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) /
                    static_cast<double>(corpus.original_bytes));
    std::printf("- chunk sizes: min %zu / median %zu / max %zu bytes\n",
                sizes.front(), sizes[sizes.size() / 2], sizes.back());
    std::printf("- method: %zu warmup + %zu measured runs, wall clock per "
                "whole-batch decode; the timed region is "
                "LZ4_decompress_safe only (no clears, no allocation); "
                "output byte-verified once before timing; percentiles are "
                "nearest-rank\n",
                warmup, runs);
    std::printf("- wall per run: p50 %.3f ms / p90 %.3f ms / p99 %.3f ms\n",
                Percentile(sorted, 50) * 1e3, Percentile(sorted, 90) * 1e3,
                Percentile(sorted, 99) * 1e3);
    std::printf("- decode throughput: p50 %.3f GB/s / p90 %.3f GB/s / p99 "
                "%.3f GB/s\n",
                to_gbps / Percentile(sorted, 50),
                to_gbps / Percentile(sorted, 90),
                to_gbps / Percentile(sorted, 99));
}

}  // namespace

int main(int argc, char** argv) {
    size_t runs = 30;
    size_t warmup = 3;
    bool selfcheck = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--selfcheck") {
            selfcheck = true;
        } else if (arg == "--runs" && i + 1 < argc) {
            if (!ParseCount(argv[++i], 1, kMaxRuns, &runs)) {
                std::fprintf(stderr, "--runs must be in [1, %zu]\n",
                             kMaxRuns);
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
            std::fprintf(stderr, "usage: bench_lz4 [--runs N] [--warmup N] "
                                 "[--selfcheck] [corpus files...]\n");
            return 2;
        } else {
            files.push_back(arg);
        }
    }
    if (selfcheck) {
        warmup = 1;
        runs = 3;
    }

    Corpus corpus;
    if (files.empty()) {
        corpus.name = "builtin";
        for (auto& fixture : MakeLz4BlockFixtures()) {
            corpus.originals.push_back(std::move(fixture.original));
        }
    } else {
        for (const auto& path : files) {
            if (!AppendFileChunked(path, &corpus)) {
                return 1;
            }
            const size_t slash = path.find_last_of("/\\");
            corpus.name += (corpus.name.empty() ? "" : "+") +
                           path.substr(slash == std::string::npos
                                           ? 0
                                           : slash + 1);
        }
    }
    /* An empty corpus has nothing to attest: refuse instead of emitting a
     * report over zero bytes (and indexing empty vectors). */
    if (corpus.originals.empty()) {
        std::fprintf(stderr, "corpus is empty - nothing to benchmark\n");
        return 1;
    }
    /* The timed scratch buffer is sized to kChunkBytes; enforce the
     * chunking invariant here instead of trusting fixture growth in
     * another directory to keep it (a larger chunk would otherwise hand
     * LZ4_decompress_safe an overstated capacity). */
    for (const auto& original : corpus.originals) {
        if (original.size() > kChunkBytes) {
            std::fprintf(stderr, "chunk of %zu bytes exceeds the %zu-byte "
                                 "scratch invariant\n",
                         original.size(), kChunkBytes);
            return 1;
        }
    }
    CompressAll(&corpus);

    /* Byte-verify every chunk once, outside the timed region. */
    std::vector<unsigned char> scratch;
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        if (!OracleDecodes(corpus.compressed[i], corpus.originals[i].size(),
                           &scratch) ||
            scratch.size() != corpus.originals[i].size() ||
            std::memcmp(scratch.data(), corpus.originals[i].data(),
                        scratch.size()) != 0) {
            std::fprintf(stderr, "verification failed at chunk %zu\n", i);
            return 1;
        }
    }

    /* One pre-sized buffer for the timed loops; every chunk fits by the
     * kChunkBytes chunking invariant. */
    std::vector<unsigned char> timed_scratch(kChunkBytes);
    for (size_t i = 0; i < warmup; i++) {
        (void)DecodeAllSeconds(corpus, timed_scratch.data());
    }
    std::vector<double> times;
    for (size_t i = 0; i < runs; i++) {
        times.push_back(DecodeAllSeconds(corpus, timed_scratch.data()));
    }
    std::sort(times.begin(), times.end());

    PrintReport(corpus, times, warmup, runs);
    if (selfcheck) {
        std::printf("PASS: selfcheck complete\n");
    }
    return 0;
}
