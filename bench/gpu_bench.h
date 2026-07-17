/* GPU decode timing for the bench harness (implemented in gpu_bench.cu).
 * bench_lz4.cpp stays host-only and calls this. */
#ifndef CUDEC_BENCH_GPU_BENCH_H
#define CUDEC_BENCH_GPU_BENCH_H

#include <cstddef>

struct cudec_gpu_result {
    size_t chunks;
    size_t output_bytes;
    double full_ms_p50;
    double parse_only_ms_p50;
    double full_gbps_p50;       /* device-resident decode throughput */
    double parse_only_gbps_p50; /* the parse ceiling (copies elided) */
};

/* Times device-resident decode of the compressed batch: uploads once, then
 * event-times the full decode and the parse-only ceiling over `runs`
 * iterations after `warmup`. Returns false on any CUDA failure or if a
 * chunk fails to decode. */
bool cudec_bench_gpu(const unsigned char* const* comp,
                     const size_t* comp_sizes, const size_t* orig_sizes,
                     size_t n, int warmup, int runs, cudec_gpu_result* out);

struct cudec_stream_ctx_result {
    size_t chunks;
    size_t output_bytes;
    size_t compressed_bytes;
    /* Steady-state: the p50 of repeated decodes on ONE reused context, its
     * staging already grown - the setup-free number, the acceptance datum. */
    double device_steady_ms;
    double host_steady_ms;
    /* Cold: the p50 of the FIRST decode on a freshly created context, which
     * pays the staging grow - so (cold - steady) is the amortized setup. */
    double device_cold_ms;
    double host_cold_ms;
    double device_steady_gbps;
    double host_steady_gbps;
    double device_cold_gbps;
    double host_cold_gbps;
};

/* Times the reusable streaming context END TO END (host compressed in ->
 * decoded out, wall clock around the whole synchronous decode call, so the
 * pinned staging, H2D, decode, and D2H are all included as a caller sees them).
 * For each memory space it reports the STEADY-STATE p50 - repeated decodes on
 * one context whose staging is already grown (the setup-free datum) - and the
 * COLD p50 - the first decode on a fresh context, which pays the grow - so the
 * difference is the amortized per-call setup the reusable context removes.
 * Returns false on any CUDA failure or if a chunk fails to decode. */
bool cudec_bench_gpu_stream_ctx(const unsigned char* const* comp,
                                const size_t* comp_sizes,
                                const size_t* orig_sizes, size_t n, int warmup,
                                int runs, cudec_stream_ctx_result* out);

#endif /* CUDEC_BENCH_GPU_BENCH_H */
