/* GPU decode timing for the bench harness (implemented in gpu_bench.cu).
 * bench_lz4.cpp stays host-only and calls this. */
#ifndef CUDEC_BENCH_GPU_BENCH_H
#define CUDEC_BENCH_GPU_BENCH_H

#include <cstddef>

/* The one description of the device a GPU row was produced on - name, sm_
 * version, driver and runtime - written into `out` and truncated to `n`.
 * It lives here rather than in each harness because a methodology block is
 * only comparable across reports while every report names the machine the
 * same way, and two copies of this string are two ways. Returns false, and
 * writes the reason, when no device is visible to the process. */
bool cudec_bench_gpu_device_line(char* out, size_t n);

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

/* The same measurement over the Snappy batch entry (issue #167): the shipped
 * cudec_snappy_decompress_batch, and the parse-only ceiling of the same
 * parser through the chunk-decoder template seam. It reports through the
 * struct above rather than one of its own, because the two formats are timed
 * by the identical protocol and a second struct would let them drift apart. */
bool cudec_bench_gpu_snappy(const unsigned char* const* comp,
                            const size_t* comp_sizes, const size_t* orig_sizes,
                            size_t n, int warmup, int runs,
                            cudec_gpu_result* out);

/* The value the two parse-only fields carry when the format under measurement
 * has no parse-only variant to time. Negative so it cannot be mistaken for a
 * measurement and cannot be divided into: a zero would read as an
 * infinitely fast parse, which is the wrong direction for a missing number to
 * fail in. A caller prints the absence and its reason rather than the value. */
const double kCudecBenchNoParseOnly = -1.0;

/* GDeflate (issue #228): the shipped cudec_gdeflate_decompress_batch, one
 * warp per 64 KiB page, timed by the identical protocol.
 *
 * NO PARSE-ONLY CEILING, AND THE ABSENCE IS THE MEASUREMENT'S RATHER THAN
 * THIS FUNCTION'S. The two chunk formats get one because
 * src/chunk_decode.cuh is templated on a ParseOnly flag that elides the
 * copies while running the identical parse. The GDeflate kernel is not that
 * shape: its parse is a warp-cooperative round loop whose next round depends
 * on bytes the previous round's copies produced, so a variant with the copies
 * elided would not decode the same symbols and would ceiling nothing. Both
 * parse_only fields therefore come back as kCudecBenchNoParseOnly. */
bool cudec_bench_gpu_gdeflate(const unsigned char* const* comp,
                              const size_t* comp_sizes,
                              const size_t* orig_sizes, size_t n, int warmup,
                              int runs, cudec_gpu_result* out);

/* TAIL WASTE (issue #207): what a launch's geometry costs when the page count
 * does not fill the machine, measured without a hardware performance counter.
 *
 * The counter route is shut on the machine this project measures on - Nsight
 * Compute connects and is refused ERR_NVGPUCTRPERM, which is a desktop driver
 * setting and not something a session may change - so an achieved-warps figure
 * cannot come from a profiler here. What replaces it is not a substitute for
 * the same number under another name: it is the runtime's own residency answer
 * (registers and shared memory against the SM's budget), which bounds what any
 * launch of this kernel can hold, next to event-timed decodes of the same batch
 * at different grid sizes, which is what a batch actually costs. Neither is a
 * counter reading and neither is reported as one.
 */
struct cudec_gdeflate_residency {
    int sm_count;
    int block_threads;
    int warps_per_block;
    int registers_per_thread;
    size_t shared_bytes_per_block;
    size_t local_bytes_per_thread;
    /* From the runtime's occupancy answer for the shipped kernel at the
     * shipped block shape - a ceiling on residency, never a count of warps
     * that ran. */
    int max_blocks_per_sm;
    int resident_blocks;
    int resident_warps;
};

/* Fills `out` for the shipped GDeflate kernel at the shipped block shape.
 * False on any runtime failure, with `out` left alone. */
bool cudec_bench_gdeflate_residency(cudec_gdeflate_residency* out);

/* The grid the shipped batch entry launches for `pages`, from the library's
 * own arithmetic rather than from a copy of it in the harness. */
unsigned cudec_bench_gdeflate_shipped_grid(size_t pages);

/* One point of the sweep: how many of the batch leading pages to decode, and
 * at what grid width. `blocks` of 0 means the grid the shipped entry would
 * launch for `pages`, so the baseline row comes from the same arithmetic the
 * library uses rather than from a copy of it. */
struct cudec_gdeflate_sweep_point {
    size_t pages;
    unsigned blocks;
    double ms_p50; /* out */
};

/* Event-times the SHIPPED GDeflate kernel over ONE uploaded batch at every
 * point, filling each point ms_p50.
 *
 * THIS IS THE LEVER ITSELF AND NOT A MODEL OF IT. The kernel already carries a
 * grid-stride loop over pages, so a grid narrower than one warp per page is
 * multi-tile-per-warp executing, with no kernel byte changed: at
 * `resident_blocks` every warp is resident for the whole launch and drives as
 * many pages as it can pull.
 *
 * ONE UPLOAD FOR EVERY POINT, and that is a correctness property rather than a
 * saving. The device buffers this harness allocates are reclaimed at process
 * exit, so a sweep that uploaded per point would exhaust the device and would
 * time each point against a differently-fragmented heap. Every point here runs
 * over the same bytes at the same addresses; only `pages` and gridDim.x move.
 *
 * Every page is decoded and verified through the public batch entry before a
 * single run is timed, exactly as the throughput rows are. */
bool cudec_bench_gpu_gdeflate_sweep(const unsigned char* const* comp,
                                    const size_t* comp_sizes,
                                    const size_t* orig_sizes, size_t n,
                                    int warmup, int runs,
                                    cudec_gdeflate_sweep_point* points,
                                    size_t point_count);

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
