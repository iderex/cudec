/* GPU decode timing for the bench harness. Owns all CUDA (device buffers,
 * CUDA-event timing, both kernel variants) so bench_lz4.cpp stays host-only.
 * Reports device-resident decode throughput - data already on the GPU, so
 * H2D/D2H is excluded and the number is pure kernel decode throughput. */
#include "gpu_bench.h"

#include "cudec.h"
#include "lz4_decode.cuh"

#include <cuda_runtime.h>

#include <algorithm>
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
