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

#endif /* CUDEC_BENCH_GPU_BENCH_H */
