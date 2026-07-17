/* The benchmark harness (M0 skeleton): times batch LZ4 block decode
 * through the CPU oracle - the only decoder that exists until M1, which
 * plugs in as a second timed path in this same harness. A report cannot
 * be produced without its methodology block; that is the point
 * (docs/MASTERPLAN.md section 5, honest numbers). */
#include "cudec.h"
#include "fixtures.h"
#include "gpu_bench.h"

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

/* The worst-case sweep replicates one adversarial block to this many chunks:
 * enough 64 KB warps to saturate the RTX 3080 (68 SMs, >= 32 warps/SM) and
 * the same ~200 MB scale as the Silesia GPU row, so the two throughput
 * numbers are directly comparable. */
constexpr size_t kWorst4bChunks = 3200;

/* The CI rot check (--worst4b --selfcheck) exercises the identical
 * construction on a handful of chunks so it stays fast on the GPU-less
 * runner; the block itself is the same regardless of the replica count. */
constexpr size_t kWorst4bSelfcheckChunks = 4;

struct Corpus {
    std::string name;
    std::vector<std::vector<unsigned char>> originals;
    std::vector<std::vector<unsigned char>> compressed;
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    /* How the compressed streams were produced - printed verbatim in the
     * methodology block, so it must stay true for whichever corpus ran. */
    std::string provenance = "compressed in-harness via LZ4_compress_default";
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

/* Builds one adversarial-but-valid LZ4 block: back-to-back minimum matches
 * (match length 4, offset 1). This is the maximum sequence density a valid
 * block can carry - one parsed sequence per 4 decoded bytes - and it drives
 * the kernel's per-byte closed-form modular gather on every match byte, so
 * the redundant lockstep parse floors here. LZ4_compress_default never emits
 * it (it extends any offset-1 run into a single long match, the best case),
 * so the stream is constructed directly and proven valid by the oracle
 * before timing.
 *
 * The block decodes to `out_bytes` copies of one seed byte. Wire layout:
 *   token 0x10, 1 seed literal, offset 0x0001      (1 literal + 4 match)
 *   token 0x00, offset 0x0001               x M    (4 match bytes each)
 *   token (literal-length tail), >= 12 trailing literals
 * The trailing literal run keeps the last match clear of the block end,
 * satisfying LZ4's parsing restrictions (LASTLITERALS = 5, last match >= 12
 * bytes before the end); the oracle is the sole authority and confirms the
 * verdict in BuildWorst4bCorpus before any timing. Assumes out_bytes is a
 * full chunk (kChunkBytes) - comfortably above the tail minimum. */
void BuildWorst4bBlock(size_t out_bytes, std::vector<unsigned char>* original,
                       std::vector<unsigned char>* compressed) {
    constexpr unsigned char kSeed = 0xA5;
    constexpr size_t kMinTail = 12; /* >= LZ4's last-match distance rule */

    original->assign(out_bytes, kSeed);

    std::vector<unsigned char>& c = *compressed;
    c.clear();
    /* Seed sequence: one real literal, then a length-4 offset-1 match that
     * copies it forward (offset 1 = run-length; the match reads the byte
     * just written). */
    c.push_back(0x10);  /* literal length 1, match length 4 */
    c.push_back(kSeed); /* the one literal byte */
    c.push_back(0x01);  /* offset low byte (offset = 1) */
    c.push_back(0x00);  /* offset high byte */
    size_t produced = 5; /* 1 literal + 4 match bytes */

    /* Minimum matches until only the tail (>= kMinTail literals) remains. */
    while (produced + 4 <= out_bytes - kMinTail) {
        c.push_back(0x00); /* literal length 0, match length 4 */
        c.push_back(0x01);
        c.push_back(0x00);
        produced += 4;
    }

    /* Literals-only tail sequence (no offset/match follows the literals; the
     * decoder detects end-of-block when the input is exhausted). */
    const size_t tail = out_bytes - produced;
    if (tail < 15) {
        c.push_back(static_cast<unsigned char>(tail << 4));
    } else {
        c.push_back(0xF0); /* literal length >= 15: read extension bytes */
        size_t rem = tail - 15;
        while (rem >= 255) {
            c.push_back(255);
            rem -= 255;
        }
        c.push_back(static_cast<unsigned char>(rem));
    }
    c.insert(c.end(), tail, kSeed);
}

/* Replicates the worst-case block to `chunks` identical chunks. Rejects an
 * invalid construction before any timing: the oracle (liblz4) is the sole
 * authority on validity, and honest numbers require a stream that actually
 * decodes (docs/MASTERPLAN.md, "the oracles decide"). */
bool BuildWorst4bCorpus(Corpus* corpus, size_t chunks) {
    std::vector<unsigned char> original;
    std::vector<unsigned char> compressed;
    BuildWorst4bBlock(kChunkBytes, &original, &compressed);

    std::vector<unsigned char> decoded;
    if (!OracleDecodes(compressed, original.size(), &decoded) ||
        decoded.size() != original.size() ||
        std::memcmp(decoded.data(), original.data(), decoded.size()) != 0) {
        std::fprintf(stderr, "worst-4Bmatch construction rejected by the "
                             "oracle - refusing to time an invalid stream\n");
        return false;
    }

    corpus->name = "worst-4Bmatch";
    corpus->originals.assign(chunks, original);
    corpus->compressed.assign(chunks, compressed);
    corpus->original_bytes = original.size() * chunks;
    corpus->compressed_bytes = compressed.size() * chunks;
    corpus->provenance = "hand-constructed offset-1 minmatch worst case "
                         "(oracle-validated; LZ4_compress_default never emits "
                         "it)";
    return true;
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
    std::printf("- cudec: %d (the CPU rows time the liblz4 oracle baseline; "
                "the GPU rows below, when --gpu is set, time cudec's "
                "decoder)\n",
                cudec_version());
    std::printf("- corpus: %s, %zu chunks, %.2f MB original, %.2f MB "
                "compressed (ratio %.3f), %s\n",
                corpus.name.c_str(), corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) /
                    static_cast<double>(corpus.original_bytes),
                corpus.provenance.c_str());
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
    bool gpu = false;
    bool gpu_stream = false;
    bool worst4b = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--selfcheck") {
            selfcheck = true;
        } else if (arg == "--gpu") {
            gpu = true;
        } else if (arg == "--gpu-stream") {
            gpu_stream = true;
        } else if (arg == "--worst4b") {
            worst4b = true;
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
            std::fprintf(stderr,
                         "usage: bench_lz4 [--runs N] [--warmup N] [--gpu] "
                         "[--gpu-stream] [--worst4b] [--selfcheck] "
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

    Corpus corpus;
    if (worst4b) {
        /* The worst-case corpus is generated, not read: it carries its own
         * hand-built compressed streams, so it must not also take files. */
        if (!files.empty()) {
            std::fprintf(stderr, "--worst4b builds its own corpus; do not "
                                 "also pass corpus files\n");
            return 2;
        }
        if (!BuildWorst4bCorpus(&corpus, selfcheck ? kWorst4bSelfcheckChunks
                                                   : kWorst4bChunks)) {
            return 1;
        }
    } else if (files.empty()) {
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
    /* The worst-case corpus already carries its hand-built streams; the
     * standard compressor would replace them with a single long match (the
     * best case), defeating the point. Every other corpus is compressed by
     * the oracle here. */
    if (!worst4b) {
        CompressAll(&corpus);
    }

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

    if (gpu) {
        std::vector<const unsigned char*> comp_ptrs(corpus.compressed.size());
        std::vector<size_t> comp_sizes(corpus.compressed.size());
        std::vector<size_t> orig_sizes(corpus.originals.size());
        for (size_t i = 0; i < corpus.compressed.size(); i++) {
            comp_ptrs[i] = corpus.compressed[i].data();
            comp_sizes[i] = corpus.compressed[i].size();
            orig_sizes[i] = corpus.originals[i].size();
        }
        cudec_gpu_result g;
        if (!cudec_bench_gpu(comp_ptrs.data(), comp_sizes.data(),
                             orig_sizes.data(), corpus.originals.size(),
                             static_cast<int>(warmup), static_cast<int>(runs),
                             &g)) {
            std::fprintf(stderr, "GPU bench failed\n");
            return 1;
        }
        std::printf("- GPU decode (device-resident, CUDA-event timed, "
                    "%d warmup + %d runs): p50 %.3f ms, %.1f GB/s\n",
                    static_cast<int>(warmup), static_cast<int>(runs),
                    g.full_ms_p50, g.full_gbps_p50);
        std::printf("- GPU parse-only ceiling (copies elided): p50 %.3f ms, "
                    "%.1f GB/s - ceilings this design AND any two-phase "
                    "phase-1 (shared parse)\n",
                    g.parse_only_ms_p50, g.parse_only_gbps_p50);
    }

    if (gpu_stream) {
        std::vector<const unsigned char*> comp_ptrs(corpus.compressed.size());
        std::vector<size_t> comp_sizes(corpus.compressed.size());
        std::vector<size_t> orig_sizes(corpus.originals.size());
        for (size_t i = 0; i < corpus.compressed.size(); i++) {
            comp_ptrs[i] = corpus.compressed[i].data();
            comp_sizes[i] = corpus.compressed[i].size();
            orig_sizes[i] = corpus.originals[i].size();
        }
        const unsigned kStreams = 4;
        cudec_stream_result s;
        if (!cudec_bench_gpu_stream(comp_ptrs.data(), comp_sizes.data(),
                                    orig_sizes.data(), corpus.originals.size(),
                                    static_cast<int>(warmup),
                                    static_cast<int>(runs), kStreams, &s)) {
            std::fprintf(stderr, "GPU streaming bench failed\n");
            return 1;
        }
        /* The effective overlap depth: the entry caps streams to the wave
         * count (64 chunks/wave), so a small corpus runs fewer than requested.
         */
        const size_t kWaveChunks = 64;
        const size_t waves =
            (corpus.originals.size() + kWaveChunks - 1) / kWaveChunks;
        const unsigned eff_streams = static_cast<unsigned>(
            s.overlap_streams < waves ? s.overlap_streams : waves);
        /* End-to-end throughput = decoded output bytes / wall time (H2D and,
         * for host output, D2H included). The wall also includes the one-shot
         * per-call ring allocation this synchronous entry does (a reusable
         * context is deferred): the numbers below are DOMINATED by that setup,
         * not by copy or decode - compare the device wall to the pure-H2D and
         * device-resident-decode rows. For the same reason the two device rows
         * are NOT a clean overlap comparison: the higher-stream config
         * provisions proportionally more ring, so its extra time is setup, not
         * a copy/decode-overlap regression. A setup-free overlap number needs
         * the reusable context (follow-up); the overlap CAPABILITY itself is
         * locked by the stream_overlap test. */
        std::printf("- GPU streaming, end-to-end, ONE-SHOT-SETUP-DOMINATED "
                    "(host compressed in -> decoded out; wall clock around the "
                    "whole synchronous call incl. per-call ring setup; %d "
                    "warmup + %d runs):\n",
                    static_cast<int>(warmup), static_cast<int>(runs));
        std::printf("    device out: 1 stream p50 %.1f ms, %.2f GB/s ; %u "
                    "streams (of %u requested) p50 %.1f ms, %.2f GB/s\n",
                    s.device_serial_ms, s.device_serial_gbps, eff_streams,
                    s.overlap_streams, s.device_overlap_ms,
                    s.device_overlap_gbps);
        std::printf("    host out (1 internal stream; readback synchronous): "
                    "p50 %.1f ms, %.2f GB/s\n",
                    s.host_ms, s.host_gbps);
        std::printf("    context: pure contiguous H2D of %.2f MB compressed = "
                    "p50 %.3f ms (a best-case floor; the pipeline stages many "
                    "smaller per-wave copies) - dwarfed by the walls above, so "
                    "the walls are setup-bound, not PCIe-bound\n",
                    static_cast<double>(s.compressed_bytes) / 1e6, s.h2d_ms);
    }

    if (selfcheck) {
        std::printf("PASS: selfcheck complete\n");
    }
    return 0;
}
