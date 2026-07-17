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

struct cudec_stream_result {
    size_t chunks;
    size_t output_bytes;
    size_t compressed_bytes;
    unsigned overlap_streams;   /* the N used for the overlapped device row */
    double device_serial_ms;    /* device out, streams=1 (no overlap) */
    double device_overlap_ms;   /* device out, streams=N (overlapped) */
    double host_ms;             /* host out, end to end (serial by design) */
    double h2d_ms;              /* pure H2D of the compressed bytes, context */
    double device_serial_gbps;
    double device_overlap_gbps;
    double host_gbps;
};

/* Times the streaming entry cudec_lz4_decompress_stream END TO END (host
 * compressed in -> decoded out, wall clock around the whole synchronous
 * call, so the pinned staging, H2D, decode, and D2H are all included, as a
 * caller would see them) over `runs` iterations after `warmup`. Measures the
 * device-output path at streams=1 (serial) and streams=`streams` (overlapped)
 * to expose the copy/decode overlap, plus the host-output path, plus a pure
 * H2D of the compressed bytes for context. Returns false on any CUDA failure
 * or if a chunk fails to decode. */
bool cudec_bench_gpu_stream(const unsigned char* const* comp,
                            const size_t* comp_sizes, const size_t* orig_sizes,
                            size_t n, int warmup, int runs, unsigned streams,
                            cudec_stream_result* out);

#endif /* CUDEC_BENCH_GPU_BENCH_H */
