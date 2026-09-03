/* The two pieces of arithmetic every reported number in this harness goes
 * through, defined once (issue #65).
 *
 * The percentile method was copied three times across two translation units,
 * and the report already promises that the CPU and GPU rows of one report share
 * one definition (the comment at gpu_bench.cu's timing loop). A promise kept by
 * three hand-matched copies is a promise about the copies, so here it is one
 * function that both paths call.
 *
 * Bench-only. Nothing here is compiled into the library. */
#ifndef CUDEC_BENCH_STATS_H
#define CUDEC_BENCH_STATS_H

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace cudec_bench {

/* Nearest-rank with ceiling: never flatters - at 30 runs, p99 is the slowest
 * run, not the second-slowest a floor index would pick. The rank == 0 clamp is
 * what makes pct == 0 and a one-element sample land on a real index.
 *
 * Templated because the device path times with the runtime's event timer
 * into float and the host path uses double; the index arithmetic is identical
 * and the sample's own type is returned unconverted.
 *
 * REQUIRES: `sorted` is non-empty and ascending. Callers sort immediately
 * before the call - the harness aborts on an empty sample long before here,
 * and a bench that indexed an empty vector would be reporting nothing at all. */
template <class T>
T Percentile(const std::vector<T>& sorted, int pct) {
    const size_t rank = (sorted.size() * static_cast<size_t>(pct) + 99) / 100;
    return sorted[(rank == 0 ? 1 : rank) - 1];
}

/* Throughput from a duration in milliseconds, with the sub-microsecond case
 * guarded: a tiny corpus can event-time to 0.0 ms, and a degenerate run must
 * report 0 rather than inf. */
inline double GbpsFromMs(double gb, double ms) {
    return ms > 0.0 ? gb / (ms / 1e3) : 0.0;
}

/* The host the numbers were taken on, read from the kernel rather than
 * configured, so a report cannot attest a CPU the run did not happen on.
 * Here rather than in one harness because every report block names it and a
 * second hand-matched copy is what this header exists to prevent. */
inline std::string HostCpuName() {
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

}  // namespace cudec_bench

#endif /* CUDEC_BENCH_STATS_H */
