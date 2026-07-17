/* GPU decode timing for the bench harness. Owns all CUDA (device buffers,
 * CUDA-event timing, both kernel variants) so bench_lz4.cpp stays host-only.
 * Reports device-resident decode throughput - data already on the GPU, so
 * H2D/D2H is excluded and the number is pure kernel decode throughput. */
#include "gpu_bench.h"

#include "cudec.h"
#include "lz4_decode.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

/* Parse-only ceiling variant: the identical redundant lockstep parse with
 * the copies elided, launched exactly like the shipped decode. It ceilings
 * both single-pass and any two-phase phase-1 (which shares this parse). */
cudec_status LaunchParseOnly(const void* const* s, const size_t* ss,
                             void* const* d, const size_t* dc, size_t n,
                             cudec_chunk_result* r, cudaStream_t stream) {
    const cudec_status valid =
        cudec_detail::validate_batch_args(s, ss, d, dc, n, r);
    if (valid != CUDEC_OK) {
        return valid;
    }
    (void)cudaGetLastError();
    cudec_detail::lz4_decode_batch<true>
        <<<cudec_detail::decode_grid_blocks(n), cudec_detail::kBlockThreads, 0,
           stream>>>(s, ss, d, dc, n, r);
    return cudaGetLastError() == cudaSuccess ? CUDEC_OK : CUDEC_ERR_CUDA;
}

#define BG_CUDA(call)                            \
    do {                                         \
        if ((call) != cudaSuccess) return false; \
    } while (0)

/* Event-times `launch` over `runs` iterations after `warmup`, returning the
 * p50 milliseconds. */
bool TimeKernel(cudec_status (*launch)(const void* const*, const size_t*,
                                       void* const*, const size_t*, size_t,
                                       cudec_chunk_result*, cudaStream_t),
                const void** d_s, size_t* d_ss, void** d_d, size_t* d_dc,
                size_t n, cudec_chunk_result* d_r, cudaStream_t stream,
                int warmup, int runs, double* p50_ms) {
    for (int i = 0; i < warmup; i++) {
        if (launch(d_s, d_ss, d_d, d_dc, n, d_r, stream) != CUDEC_OK) {
            return false;
        }
    }
    BG_CUDA(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    BG_CUDA(cudaEventCreate(&start));
    BG_CUDA(cudaEventCreate(&stop));
    std::vector<float> times(static_cast<size_t>(runs));
    for (int i = 0; i < runs; i++) {
        BG_CUDA(cudaEventRecord(start, stream));
        if (launch(d_s, d_ss, d_d, d_dc, n, d_r, stream) != CUDEC_OK) {
            return false;
        }
        BG_CUDA(cudaEventRecord(stop, stream));
        BG_CUDA(cudaEventSynchronize(stop));
        BG_CUDA(cudaEventElapsedTime(&times[static_cast<size_t>(i)], start,
                                     stop));
    }
    BG_CUDA(cudaEventDestroy(start));
    BG_CUDA(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    /* Nearest-rank p50, matching the CPU path's percentile method so the
     * two rows of the same report use one definition. */
    const size_t rank = (times.size() * 50 + 99) / 100;
    *p50_ms = times[(rank == 0 ? 1 : rank) - 1];
    return true;
}

cudec_status LaunchFull(const void* const* s, const size_t* ss,
                        void* const* d, const size_t* dc, size_t n,
                        cudec_chunk_result* r, cudaStream_t stream) {
    return cudec_lz4_decompress_batch(s, ss, d, dc, n, r, stream);
}

/* Wall-clock-times the synchronous streaming entry over `runs` iterations
 * after `warmup`, returning the p50 milliseconds. The call drains internally,
 * so std::chrono around it is the honest end-to-end (pinned staging + H2D +
 * decode + [D2H] + the one-shot ring setup, as a caller sees it). */
bool TimeStream(const void* const* h_src, const size_t* h_ssz,
                void* const* h_dst, const size_t* h_cap, size_t n,
                cudec_mem_space space, unsigned streams,
                cudec_chunk_result* results, int warmup, int runs,
                double* p50_ms) {
    for (int i = 0; i < warmup; i++) {
        if (cudec_lz4_decompress_stream(h_src, h_ssz, h_dst, h_cap, n, space,
                                        streams, results) != CUDEC_OK) {
            return false;
        }
    }
    std::vector<double> t(static_cast<size_t>(runs));
    for (int i = 0; i < runs; i++) {
        const auto s = std::chrono::steady_clock::now();
        const cudec_status st = cudec_lz4_decompress_stream(
            h_src, h_ssz, h_dst, h_cap, n, space, streams, results);
        const auto e = std::chrono::steady_clock::now();
        if (st != CUDEC_OK) {
            return false;
        }
        t[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::milli>(e - s).count();
    }
    std::sort(t.begin(), t.end());
    const size_t rank = (t.size() * 50 + 99) / 100;
    *p50_ms = t[(rank == 0 ? 1 : rank) - 1];
    return true;
}

}  // namespace

bool cudec_bench_gpu(const unsigned char* const* comp,
                     const size_t* comp_sizes, const size_t* orig_sizes,
                     size_t n, int warmup, int runs, cudec_gpu_result* out) {
    std::vector<const void*> h_s(n);
    std::vector<void*> h_d(n);
    std::vector<size_t> h_ss(n), h_dc(n);
    size_t total_out = 0;
    for (size_t i = 0; i < n; i++) {
        void* ds = nullptr;
        void* dd = nullptr;
        BG_CUDA(cudaMalloc(&ds, comp_sizes[i] ? comp_sizes[i] : 1));
        if (comp_sizes[i]) {
            BG_CUDA(cudaMemcpy(ds, comp[i], comp_sizes[i],
                               cudaMemcpyHostToDevice));
        }
        BG_CUDA(cudaMalloc(&dd, orig_sizes[i] ? orig_sizes[i] : 1));
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
    BG_CUDA(cudaMalloc(&d_s, n * sizeof(*d_s)));
    BG_CUDA(cudaMalloc(&d_d, n * sizeof(*d_d)));
    BG_CUDA(cudaMalloc(&d_ss, n * sizeof(*d_ss)));
    BG_CUDA(cudaMalloc(&d_dc, n * sizeof(*d_dc)));
    BG_CUDA(cudaMalloc(&d_r, n * sizeof(*d_r)));
    BG_CUDA(cudaMemcpy(d_s, h_s.data(), n * sizeof(*d_s),
                       cudaMemcpyHostToDevice));
    BG_CUDA(cudaMemcpy(d_d, h_d.data(), n * sizeof(*d_d),
                       cudaMemcpyHostToDevice));
    BG_CUDA(cudaMemcpy(d_ss, h_ss.data(), n * sizeof(*d_ss),
                       cudaMemcpyHostToDevice));
    BG_CUDA(cudaMemcpy(d_dc, h_dc.data(), n * sizeof(*d_dc),
                       cudaMemcpyHostToDevice));

    cudaStream_t stream;
    BG_CUDA(cudaStreamCreate(&stream));

    /* Correctness precondition: every chunk must decode OK, or the numbers
     * are meaningless (honest-numbers discipline). */
    if (LaunchFull(d_s, d_ss, d_d, d_dc, n, d_r, stream) != CUDEC_OK) {
        return false;
    }
    BG_CUDA(cudaStreamSynchronize(stream));
    std::vector<cudec_chunk_result> res(n);
    BG_CUDA(cudaMemcpy(res.data(), d_r, n * sizeof(*d_r),
                       cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) {
        if (res[i].status != CUDEC_OK ||
            res[i].bytes_written != orig_sizes[i]) {
            std::fprintf(stderr, "gpu bench: chunk %zu did not decode\n", i);
            return false;
        }
    }

    double full_ms = 0.0;
    double parse_ms = 0.0;
    if (!TimeKernel(LaunchFull, d_s, d_ss, d_d, d_dc, n, d_r, stream, warmup,
                    runs, &full_ms)) {
        return false;
    }
    if (!TimeKernel(LaunchParseOnly, d_s, d_ss, d_d, d_dc, n, d_r, stream,
                    warmup, runs, &parse_ms)) {
        return false;
    }

    const double gb = static_cast<double>(total_out) / 1e9;
    out->chunks = n;
    out->output_bytes = total_out;
    out->full_ms_p50 = full_ms;
    out->parse_only_ms_p50 = parse_ms;
    /* Guard the sub-microsecond case (a tiny corpus can event-time to
     * 0.0 ms) so a degenerate run reports 0, never inf. */
    out->full_gbps_p50 = full_ms > 0.0 ? gb / (full_ms / 1e3) : 0.0;
    out->parse_only_gbps_p50 = parse_ms > 0.0 ? gb / (parse_ms / 1e3) : 0.0;
    /* Buffers are reclaimed at process exit; the bench is a short-lived
     * one-shot, like the test harness. */
    return true;
}

bool cudec_bench_gpu_stream(const unsigned char* const* comp,
                            const size_t* comp_sizes, const size_t* orig_sizes,
                            size_t n, int warmup, int runs, unsigned streams,
                            cudec_stream_result* out) {
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

    /* Host input arrays: the compressed blocks stay where the caller has
     * them (host memory); the library stages them through its pinned ring. */
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
    BG_CUDA(cudaMalloc(&d_out, total_out ? total_out : 1));
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

    /* Correctness precondition on both paths: every chunk returns CUDEC_OK
     * with its original decoded size, or the numbers are meaningless
     * (honest-numbers rule). Byte-for-byte correctness is gated by the
     * stream_twin / gpu_fixture oracle tests, not re-verified here. */
    for (int pass = 0; pass < 2; pass++) {
        void* const* dst = (pass == 0) ? d_dst.data() : h_dst.data();
        const cudec_mem_space sp =
            (pass == 0) ? CUDEC_MEM_DEVICE : CUDEC_MEM_HOST;
        if (cudec_lz4_decompress_stream(h_src.data(), h_ssz.data(), dst,
                                        h_cap.data(), n, sp, streams,
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

    /* Device output at 1 stream (no overlap) and at `streams` (overlapped);
     * host output at 1 stream (the entry forces it serial anyway, so more
     * would only provision unused ring). */
    double dev_serial = 0.0;
    double dev_overlap = 0.0;
    double host_ms = 0.0;
    if (!TimeStream(h_src.data(), h_ssz.data(), d_dst.data(), h_cap.data(), n,
                    CUDEC_MEM_DEVICE, 1, res.data(), warmup, runs,
                    &dev_serial) ||
        !TimeStream(h_src.data(), h_ssz.data(), d_dst.data(), h_cap.data(), n,
                    CUDEC_MEM_DEVICE, streams, res.data(), warmup, runs,
                    &dev_overlap) ||
        !TimeStream(h_src.data(), h_ssz.data(), h_dst.data(), h_cap.data(), n,
                    CUDEC_MEM_HOST, 1, res.data(), warmup, runs, &host_ms)) {
        return false;
    }

    /* Pure H2D of the compressed bytes, pinned->device, for context: a
     * best-case PCIe floor. Comparing it to the end-to-end wall shows whether
     * that wall is copy-bound or (as measured) dominated by per-call setup. */
    double h2d_ms = 0.0;
    {
        void* d_comp = nullptr;
        void* p_comp = nullptr;
        BG_CUDA(cudaMalloc(&d_comp, total_comp ? total_comp : 1));
        BG_CUDA(cudaHostAlloc(&p_comp, total_comp ? total_comp : 1,
                              cudaHostAllocDefault));
        size_t off = 0;
        for (size_t i = 0; i < n; i++) {
            if (comp_sizes[i]) {
                std::copy(comp[i], comp[i] + comp_sizes[i],
                          static_cast<unsigned char*>(p_comp) + off);
            }
            off += comp_sizes[i];
        }
        for (int i = 0; i < warmup; i++) {
            BG_CUDA(cudaMemcpy(d_comp, p_comp, total_comp ? total_comp : 1,
                               cudaMemcpyHostToDevice));
        }
        std::vector<double> t(static_cast<size_t>(runs));
        for (int i = 0; i < runs; i++) {
            const auto s = std::chrono::steady_clock::now();
            BG_CUDA(cudaMemcpy(d_comp, p_comp, total_comp ? total_comp : 1,
                               cudaMemcpyHostToDevice));
            const auto e = std::chrono::steady_clock::now();
            t[static_cast<size_t>(i)] =
                std::chrono::duration<double, std::milli>(e - s).count();
        }
        std::sort(t.begin(), t.end());
        const size_t rank = (t.size() * 50 + 99) / 100;
        h2d_ms = t[(rank == 0 ? 1 : rank) - 1];
    }

    const double gb = static_cast<double>(total_out) / 1e9;
    out->chunks = n;
    out->output_bytes = total_out;
    out->compressed_bytes = total_comp;
    out->overlap_streams = streams;
    out->device_serial_ms = dev_serial;
    out->device_overlap_ms = dev_overlap;
    out->host_ms = host_ms;
    out->h2d_ms = h2d_ms;
    out->device_serial_gbps = dev_serial > 0.0 ? gb / (dev_serial / 1e3) : 0.0;
    out->device_overlap_gbps =
        dev_overlap > 0.0 ? gb / (dev_overlap / 1e3) : 0.0;
    out->host_gbps = host_ms > 0.0 ? gb / (host_ms / 1e3) : 0.0;
    /* The output/context arenas (d_out, d_comp, p_comp) are reclaimed at
     * process exit; the bench is a short-lived one-shot called once, like
     * cudec_bench_gpu above. */
    return true;
}
