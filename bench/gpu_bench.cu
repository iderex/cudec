/* GPU decode timing for the bench harness. Owns all device work (device
 * buffers, event timing, both kernel variants) so bench_lz4.cpp stays
 * host-only, and reaches the runtime through the harness seam
 * (tests/vendor_rt_test.h) so this one file is the bench on both backends
 * (issue #432). Reports device-resident decode throughput - data already on
 * the GPU, so H2D/D2H is excluded and the number is pure kernel decode
 * throughput. */
#include "gpu_bench.h"

#include "bench_stats.h"
#include "cudec.h"
#include "chunk_decode.cuh"
#include "gdeflate_decode.cuh"
#include "lz4_block.h"
#include "snappy_block.h"
#include "vendor_rt_test.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

/* Parse-only ceiling variant: the identical redundant lockstep parse with
 * the copies elided, launched exactly like the shipped decode. It ceilings
 * both single-pass and any two-phase phase-1 (which shares this parse).
 *
 * Templated on the parser rather than written once per format, for the same
 * reason src/chunk_decode.cuh is: the launch shape, the argument validation
 * and the grid are format-independent, and a second copy of them here would
 * be a second place for the bench and the shipped path to drift apart. Only
 * the instantiations named below are emitted. */
template <class Parser>
cudec_status LaunchParseOnly(const void* const* s, const size_t* ss,
                             void* const* d, const size_t* dc, size_t n,
                             cudec_chunk_result* r, cudec_rt::stream_t stream) {
    const cudec_status valid =
        cudec_detail::validate_batch_args(s, ss, d, dc, n, r);
    if (valid != CUDEC_OK) {
        return valid;
    }
    (void)cudec_rt::get_last_error();
    cudec_detail::chunk_decode_batch<Parser, true, cudec_detail::kCudaWaveSize>
        <<<cudec_detail::decode_grid_blocks(n), cudec_detail::kBlockThreads, 0,
           stream>>>(s, ss, d, dc, n, r);
    return cudec_rt::get_last_error() == cudec_rt::success ? CUDEC_OK
                                                           : CUDEC_ERR_CUDA;
}

#define BG_RT(call)                                       \
    do {                                                  \
        if ((call) != cudec_rt::success) return false;    \
    } while (0)

/* The one launch shape every timed variant here has: the batch ABI's own
 * argument list, with the stream in the runtime's own type - the two public
 * launchers below make the ABI cast, the parse-only one launches directly.
 * Named once so the timing loop and the format entries cannot disagree
 * about it. */
using LaunchFn = cudec_status (*)(const void* const*, const size_t*,
                                  void* const*, const size_t*, size_t,
                                  cudec_chunk_result*, cudec_rt::stream_t);

/* Event-times `launch` over `runs` iterations after `warmup`, returning the
 * p50 milliseconds. */
bool TimeKernel(LaunchFn launch,
                const void** d_s, size_t* d_ss, void** d_d, size_t* d_dc,
                size_t n, cudec_chunk_result* d_r, cudec_rt::stream_t stream,
                int warmup, int runs, double* p50_ms) {
    for (int i = 0; i < warmup; i++) {
        if (launch(d_s, d_ss, d_d, d_dc, n, d_r, stream) != CUDEC_OK) {
            return false;
        }
    }
    BG_RT(cudec_rt::stream_synchronize(stream));
    cudec_rt::event_t start, stop;
    BG_RT(cudec_rt::event_create(&start));
    BG_RT(cudec_rt::event_create(&stop));
    std::vector<float> times(static_cast<size_t>(runs));
    for (int i = 0; i < runs; i++) {
        BG_RT(cudec_rt::event_record(start, stream));
        if (launch(d_s, d_ss, d_d, d_dc, n, d_r, stream) != CUDEC_OK) {
            return false;
        }
        BG_RT(cudec_rt::event_record(stop, stream));
        BG_RT(cudec_rt::event_synchronize(stop));
        BG_RT(cudec_rt::event_elapsed_ms(&times[static_cast<size_t>(i)],
                                         start, stop));
    }
    BG_RT(cudec_rt::event_destroy(start));
    BG_RT(cudec_rt::event_destroy(stop));
    std::sort(times.begin(), times.end());
    /* The same nearest-rank definition the CPU rows use, so the two rows of
     * one report mean the same thing (bench/bench_stats.h). */
    *p50_ms = cudec_bench::Percentile(times, 50);
    return true;
}

cudec_status LaunchFull(const void* const* s, const size_t* ss,
                        void* const* d, const size_t* dc, size_t n,
                        cudec_chunk_result* r, cudec_rt::stream_t stream) {
    return cudec_lz4_decompress_batch(s, ss, d, dc, n, r,
                                      cudec_rt::abi_stream(stream));
}

cudec_status LaunchFullSnappy(const void* const* s, const size_t* ss,
                              void* const* d, const size_t* dc, size_t n,
                              cudec_chunk_result* r,
                              cudec_rt::stream_t stream) {
    return cudec_snappy_decompress_batch(s, ss, d, dc, n, r,
                                         cudec_rt::abi_stream(stream));
}

cudec_status LaunchFullGDeflate(const void* const* s, const size_t* ss,
                                void* const* d, const size_t* dc, size_t n,
                                cudec_chunk_result* r,
                                cudec_rt::stream_t stream) {
    return cudec_gdeflate_decompress_batch(s, ss, d, dc, n, r,
                                           cudec_rt::abi_stream(stream));
}

double Median(std::vector<double>* t) {
    std::sort(t->begin(), t->end());
    return cudec_bench::Percentile(*t, 50);
}

/* Steady-state wall time: one reusable context, warmed (so its staging is
 * already grown), then `runs` decodes on that SAME context. The call drains
 * internally, so std::chrono around it is the honest end-to-end MINUS the
 * per-call setup the context amortizes away - the setup-free datum. */
bool TimeCtxSteady(const void* const* h_src, const size_t* h_ssz,
                   void* const* dst, const size_t* h_cap, size_t n,
                   cudec_mem_space space, int warmup, int runs,
                   double* p50_ms) {
    cudec_stream_ctx* ctx = nullptr;
    if (cudec_stream_ctx_create(&ctx) != CUDEC_OK) {
        return false;
    }
    std::vector<cudec_chunk_result> res(n);
    bool ok = true;
    /* warmup + 1: at least one decode must run to grow the staging before the
     * steady-state timing, even when warmup is 0. */
    for (int i = 0; i < warmup + 1 && ok; i++) {
        ok = cudec_lz4_decompress_stream_ctx(ctx, h_src, h_ssz, dst, h_cap, n,
                                             space, res.data()) == CUDEC_OK;
    }
    std::vector<double> t;
    if (ok) {
        t.resize(static_cast<size_t>(runs));
        for (int i = 0; i < runs && ok; i++) {
            const auto s = std::chrono::steady_clock::now();
            const cudec_status st = cudec_lz4_decompress_stream_ctx(
                ctx, h_src, h_ssz, dst, h_cap, n, space, res.data());
            const auto e = std::chrono::steady_clock::now();
            ok = (st == CUDEC_OK);
            t[static_cast<size_t>(i)] =
                std::chrono::duration<double, std::milli>(e - s).count();
        }
    }
    cudec_stream_ctx_destroy(ctx);
    if (!ok) {
        return false;
    }
    *p50_ms = Median(&t);
    return true;
}

/* Cold wall time: each iteration creates a FRESH context and times only its
 * first decode - which pays the staging grow (the pinned and the device
 * allocation) - then destroys it. So (cold - steady) is the amortized
 * per-call setup the reusable context removes. */
bool TimeCtxCold(const void* const* h_src, const size_t* h_ssz,
                 void* const* dst, const size_t* h_cap, size_t n,
                 cudec_mem_space space, int runs, double* p50_ms) {
    std::vector<cudec_chunk_result> res(n);
    std::vector<double> t(static_cast<size_t>(runs));
    for (int i = 0; i < runs; i++) {
        cudec_stream_ctx* ctx = nullptr;
        if (cudec_stream_ctx_create(&ctx) != CUDEC_OK) {
            return false;
        }
        const auto s = std::chrono::steady_clock::now();
        const cudec_status st = cudec_lz4_decompress_stream_ctx(
            ctx, h_src, h_ssz, dst, h_cap, n, space, res.data());
        const auto e = std::chrono::steady_clock::now();
        cudec_stream_ctx_destroy(ctx);
        if (st != CUDEC_OK) {
            return false;
        }
        t[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::milli>(e - s).count();
    }
    *p50_ms = Median(&t);
    return true;
}

/* The device-resident measurement itself, format-agnostic: the two launchers
 * are the only thing that differs between the formats on this seam, so the
 * upload, the correctness precondition, the timing protocol and the
 * throughput arithmetic are written once and both entries below share them.
 * A second copy per format would be a second protocol, and two numbers
 * produced by two protocols are not comparable. */
bool BenchBatch(LaunchFn full, LaunchFn parse_only,
                const unsigned char* const* comp, const size_t* comp_sizes,
                const size_t* orig_sizes, size_t n, int warmup, int runs,
                cudec_gpu_result* out) {
    std::vector<const void*> h_s(n);
    std::vector<void*> h_d(n);
    std::vector<size_t> h_ss(n), h_dc(n);
    size_t total_out = 0;
    for (size_t i = 0; i < n; i++) {
        void* ds = nullptr;
        void* dd = nullptr;
        BG_RT(cudec_rt::device_malloc(&ds, comp_sizes[i] ? comp_sizes[i] : 1));
        if (comp_sizes[i]) {
            BG_RT(cudec_rt::memcpy(ds, comp[i], comp_sizes[i],
                                   cudec_rt::memcpy_h2d));
        }
        BG_RT(cudec_rt::device_malloc(&dd, orig_sizes[i] ? orig_sizes[i] : 1));
        h_s[i] = ds;
        h_d[i] = dd;
        h_ss[i] = comp_sizes[i];
        h_dc[i] = orig_sizes[i];
        total_out += orig_sizes[i];
    }
    const void** d_s;
    void** d_d;
    size_t *d_ss, *d_dc;
    cudec_chunk_result* d_r;
    BG_RT(cudec_rt::device_malloc(&d_s, n * sizeof(*d_s)));
    BG_RT(cudec_rt::device_malloc(&d_d, n * sizeof(*d_d)));
    BG_RT(cudec_rt::device_malloc(&d_ss, n * sizeof(*d_ss)));
    BG_RT(cudec_rt::device_malloc(&d_dc, n * sizeof(*d_dc)));
    BG_RT(cudec_rt::device_malloc(&d_r, n * sizeof(*d_r)));
    BG_RT(cudec_rt::memcpy(d_s, h_s.data(), n * sizeof(*d_s),
                           cudec_rt::memcpy_h2d));
    BG_RT(cudec_rt::memcpy(d_d, h_d.data(), n * sizeof(*d_d),
                           cudec_rt::memcpy_h2d));
    BG_RT(cudec_rt::memcpy(d_ss, h_ss.data(), n * sizeof(*d_ss),
                           cudec_rt::memcpy_h2d));
    BG_RT(cudec_rt::memcpy(d_dc, h_dc.data(), n * sizeof(*d_dc),
                           cudec_rt::memcpy_h2d));

    cudec_rt::stream_t stream;
    BG_RT(cudec_rt::stream_create(&stream));

    /* Correctness precondition: every chunk must decode OK, or the numbers
     * are meaningless (honest-numbers discipline). */
    if (full(d_s, d_ss, d_d, d_dc, n, d_r, stream) != CUDEC_OK) {
        return false;
    }
    BG_RT(cudec_rt::stream_synchronize(stream));
    std::vector<cudec_chunk_result> res(n);
    BG_RT(cudec_rt::memcpy(res.data(), d_r, n * sizeof(*d_r),
                           cudec_rt::memcpy_d2h));
    for (size_t i = 0; i < n; i++) {
        if (res[i].status != CUDEC_OK ||
            res[i].bytes_written != orig_sizes[i]) {
            std::fprintf(stderr, "gpu bench: chunk %zu did not decode\n", i);
            return false;
        }
    }

    double full_ms = 0.0;
    double parse_ms = 0.0;
    if (!TimeKernel(full, d_s, d_ss, d_d, d_dc, n, d_r, stream, warmup, runs,
                    &full_ms)) {
        return false;
    }
    /* A format whose kernel has no parse-only variant passes none, and the
     * two fields carry the sentinel rather than a number nothing measured.
     * The alternative - timing the full decode twice and calling one of them
     * a ceiling - is the shape of dishonesty this harness exists against. */
    if (parse_only != nullptr &&
        !TimeKernel(parse_only, d_s, d_ss, d_d, d_dc, n, d_r, stream, warmup,
                    runs, &parse_ms)) {
        return false;
    }

    const double gb = static_cast<double>(total_out) / 1e9;
    out->chunks = n;
    out->output_bytes = total_out;
    out->full_ms_p50 = full_ms;
    out->parse_only_ms_p50 =
        parse_only != nullptr ? parse_ms : kCudecBenchNoParseOnly;
    /* GbpsFromMs carries the sub-microsecond guard (bench/bench_stats.h). */
    out->full_gbps_p50 = cudec_bench::GbpsFromMs(gb, full_ms);
    out->parse_only_gbps_p50 =
        parse_only != nullptr ? cudec_bench::GbpsFromMs(gb, parse_ms)
                              : kCudecBenchNoParseOnly;
    /* Buffers are reclaimed at process exit; the bench is a short-lived
     * one-shot, like the test harness. */
    return true;
}

}  // namespace

bool cudec_bench_gpu_device_line(char* out, size_t n) {
    if (out == nullptr || n == 0) {
        return false;
    }
    int count = 0;
    if (cudec_rt::get_device_count(&count) != cudec_rt::success ||
        count == 0) {
        std::snprintf(out, n, "none visible to this process");
        return false;
    }
    cudec_rt::device_prop_t prop{};
    if (cudec_rt::get_device_properties(&prop, 0) != cudec_rt::success) {
        std::snprintf(out, n, "device query failed");
        return false;
    }
    int driver = 0;
    int runtime = 0;
    (void)cudec_rt::driver_get_version(&driver);
    (void)cudec_rt::runtime_get_version(&runtime);
    /* The architecture token and the version split are the backend's own
     * spellings, read from the seam rather than assumed here. */
    char arch[64];
    cudec_rt::device_arch(prop, arch, sizeof(arch));
    std::snprintf(out, n, "%s (%s), driver %d.%d, runtime %d.%d", prop.name,
                  arch, cudec_rt::version_major(driver),
                  cudec_rt::version_minor(driver),
                  cudec_rt::version_major(runtime),
                  cudec_rt::version_minor(runtime));
    return true;
}

bool cudec_bench_gpu(const unsigned char* const* comp,
                     const size_t* comp_sizes, const size_t* orig_sizes,
                     size_t n, int warmup, int runs, cudec_gpu_result* out) {
    return BenchBatch(LaunchFull, LaunchParseOnly<cudec_detail::Lz4Parser>,
                      comp, comp_sizes, orig_sizes, n, warmup, runs, out);
}

bool cudec_bench_gpu_snappy(const unsigned char* const* comp,
                            const size_t* comp_sizes, const size_t* orig_sizes,
                            size_t n, int warmup, int runs,
                            cudec_gpu_result* out) {
    return BenchBatch(LaunchFullSnappy,
                      LaunchParseOnly<cudec_detail::SnappyParser>, comp,
                      comp_sizes, orig_sizes, n, warmup, runs, out);
}

bool cudec_bench_gpu_gdeflate(const unsigned char* const* comp,
                              const size_t* comp_sizes,
                              const size_t* orig_sizes, size_t n, int warmup,
                              int runs, cudec_gpu_result* out) {
    return BenchBatch(LaunchFullGDeflate, nullptr, comp, comp_sizes,
                      orig_sizes, n, warmup, runs, out);
}

bool cudec_bench_gpu_stream_ctx(const unsigned char* const* comp,
                                const size_t* comp_sizes,
                                const size_t* orig_sizes, size_t n, int warmup,
                                int runs, cudec_stream_ctx_result* out) {
    size_t total_out = 0;
    size_t total_comp = 0;
    for (size_t i = 0; i < n; i++) {
        /* Overflow-guard the arena sizes, as the library entry does its own
         * (a wrapped sum would under-allocate d_out and drive OOB slicing). */
        if (SIZE_MAX - total_out < orig_sizes[i] ||
            SIZE_MAX - total_comp < comp_sizes[i]) {
            return false;
        }
        total_out += orig_sizes[i];
        total_comp += comp_sizes[i];
    }

    /* Host input arrays: the compressed blocks stay where the caller has them
     * (host memory); the context stages them through its pinned buffer. */
    std::vector<const void*> h_src(n);
    std::vector<size_t> h_ssz(n), h_cap(n);
    for (size_t i = 0; i < n; i++) {
        h_src[i] = comp[i];
        h_ssz[i] = comp_sizes[i];
        h_cap[i] = orig_sizes[i];
    }
    std::vector<cudec_chunk_result> res(n);

    /* One device and one host output arena, sliced per chunk. */
    void* d_out = nullptr;
    BG_RT(cudec_rt::device_malloc(&d_out, total_out ? total_out : 1));
    std::vector<void*> d_dst(n);
    std::vector<unsigned char> h_out(total_out ? total_out : 1);
    std::vector<void*> h_dst(n);
    {
        size_t off = 0;
        for (size_t i = 0; i < n; i++) {
            d_dst[i] = static_cast<unsigned char*>(d_out) + off;
            h_dst[i] = h_out.data() + off;
            off += orig_sizes[i];
        }
    }

    /* Correctness precondition on both paths: every chunk returns CUDEC_OK with
     * its original decoded size, or the numbers are meaningless (honest-numbers
     * rule). Byte-for-byte correctness is gated by the stream_twin / gpu_fixture
     * oracle tests, not re-verified here. */
    for (int pass = 0; pass < 2; pass++) {
        void* const* dst = (pass == 0) ? d_dst.data() : h_dst.data();
        const cudec_mem_space sp =
            (pass == 0) ? CUDEC_MEM_DEVICE : CUDEC_MEM_HOST;
        if (cudec_lz4_decompress_stream(h_src.data(), h_ssz.data(), dst,
                                        h_cap.data(), n, sp,
                                        res.data()) != CUDEC_OK) {
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            if (res[i].status != CUDEC_OK ||
                res[i].bytes_written != orig_sizes[i]) {
                std::fprintf(stderr,
                             "stream bench: chunk %zu did not decode (%s)\n", i,
                             pass == 0 ? "device" : "host");
                return false;
            }
        }
    }

    /* Steady-state (setup-free, the acceptance datum) and cold (first call on a
     * fresh context, paying the staging grow) for both memory spaces. */
    double dev_steady = 0.0, host_steady = 0.0;
    double dev_cold = 0.0, host_cold = 0.0;
    if (!TimeCtxSteady(h_src.data(), h_ssz.data(), d_dst.data(), h_cap.data(),
                       n, CUDEC_MEM_DEVICE, warmup, runs, &dev_steady) ||
        !TimeCtxSteady(h_src.data(), h_ssz.data(), h_dst.data(), h_cap.data(),
                       n, CUDEC_MEM_HOST, warmup, runs, &host_steady) ||
        !TimeCtxCold(h_src.data(), h_ssz.data(), d_dst.data(), h_cap.data(), n,
                     CUDEC_MEM_DEVICE, runs, &dev_cold) ||
        !TimeCtxCold(h_src.data(), h_ssz.data(), h_dst.data(), h_cap.data(), n,
                     CUDEC_MEM_HOST, runs, &host_cold)) {
        return false;
    }

    const double gb = static_cast<double>(total_out) / 1e9;
    out->chunks = n;
    out->output_bytes = total_out;
    out->compressed_bytes = total_comp;
    out->device_steady_ms = dev_steady;
    out->host_steady_ms = host_steady;
    out->device_cold_ms = dev_cold;
    out->host_cold_ms = host_cold;
    out->device_steady_gbps = cudec_bench::GbpsFromMs(gb, dev_steady);
    out->host_steady_gbps = cudec_bench::GbpsFromMs(gb, host_steady);
    out->device_cold_gbps = cudec_bench::GbpsFromMs(gb, dev_cold);
    out->host_cold_gbps = cudec_bench::GbpsFromMs(gb, host_cold);
    /* The output arenas (d_out) are reclaimed at process exit; the bench is a
     * short-lived one-shot called once, like cudec_bench_gpu above. */
    return true;
}
