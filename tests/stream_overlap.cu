/* The overlap lock for the streaming decode path (#24). The internal streams
 * of cudec_lz4_decompress_stream are not observable across the C ABI, so this
 * test cannot introspect the library's own pipeline; it locks the CAPABILITY
 * the streaming overlap is built on - that an async H2D copy and the shipped
 * decode kernel EXECUTE CONCURRENTLY on this platform.
 *
 * It measures each stage's duration ALONE, then runs both together on separate
 * non-blocking streams and measures the combined wall time from a common
 * origin. If they truly overlap, the combined time is materially less than the
 * sum of the two alone (a serialized run would take ~the sum); the test asserts
 * that at least half of the shorter stage is hidden behind the longer. This is
 * a real concurrency measurement, not an event-ordering artifact: it uses only
 * completion events (accurate) and the two alone-durations, so a run that did
 * NOT overlap (the copy and decode funneled onto one queue) fails. Both stages
 * are milliseconds - above WSL2/WDDM jitter - and the margin is generous (the
 * concurrent/serial gap is ~2x), so it is not flaky.
 *
 * Whether cudec's OWN streaming pipeline achieves this overlap is a separate,
 * currently un-gateable question (the one-shot per-call ring setup dominates
 * the end-to-end wall, masking the internal overlap - see docs/BENCHMARKS.md
 * and the reusable-context follow-up); this test locks the platform premise
 * that the overlap is even possible here. */
#include "cudec.h"
#include "fixtures.h"
#include "require.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

/* Event-times one op enqueued by `enqueue` on `stream`, alone. The op starts
 * immediately after e0 (the stream is idle), so the elapsed e0->e1 is the op's
 * true duration - no wait sits between the event and the op. */
template <typename F>
float TimeAlone(cudaStream_t stream, cudaEvent_t e0, cudaEvent_t e1, F enqueue) {
    float ms = 0.0f;
    if (cudaEventRecord(e0, stream) != cudaSuccess) return -1.0f;
    enqueue();
    if (cudaEventRecord(e1, stream) != cudaSuccess) return -1.0f;
    if (cudaEventSynchronize(e1) != cudaSuccess) return -1.0f;
    if (cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess) return -1.0f;
    return ms;
}

}  // namespace

int main() {
    /* A real compressed chunk, replicated to a batch whose decode takes
     * milliseconds (grid-stride re-entry past 8192 blocks is fine - this is a
     * timing workload, output is not checked here). */
    const auto fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());
    const Fixture* big = &fixtures[0];
    for (const auto& f : fixtures) {
        if (f.compressed.size() > big->compressed.size()) {
            big = &f;
        }
    }
    const size_t n = 60000;

    void* d_src = nullptr;
    void* d_dst = nullptr;
    REQUIRE_CUDA(cudaMalloc(&d_src, big->compressed.size()));
    REQUIRE_CUDA(cudaMemcpy(d_src, big->compressed.data(),
                            big->compressed.size(), cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMalloc(&d_dst, big->original.size()));

    std::vector<const void*> h_s(n, d_src);
    std::vector<void*> h_d(n, d_dst);
    std::vector<size_t> h_ss(n, big->compressed.size());
    std::vector<size_t> h_dc(n, big->original.size());
    const void** dd_s;
    void** dd_d;
    size_t* dd_ss;
    size_t* dd_dc;
    cudec_chunk_result* dd_r;
    REQUIRE_CUDA(cudaMalloc(&dd_s, n * sizeof(*dd_s)));
    REQUIRE_CUDA(cudaMalloc(&dd_d, n * sizeof(*dd_d)));
    REQUIRE_CUDA(cudaMalloc(&dd_ss, n * sizeof(*dd_ss)));
    REQUIRE_CUDA(cudaMalloc(&dd_dc, n * sizeof(*dd_dc)));
    REQUIRE_CUDA(cudaMalloc(&dd_r, n * sizeof(*dd_r)));
    REQUIRE_CUDA(cudaMemcpy(dd_s, h_s.data(), n * sizeof(*dd_s),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(dd_d, h_d.data(), n * sizeof(*dd_d),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(dd_ss, h_ss.data(), n * sizeof(*dd_ss),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(dd_dc, h_dc.data(), n * sizeof(*dd_dc),
                            cudaMemcpyHostToDevice));

    /* A milliseconds-scale H2D, sized to be comparable to the decode so the
     * overlap saving is a large, robust fraction of the total. Pinned so it is
     * a real async DMA. */
    const size_t kBig = size_t{128} << 20; /* 128 MiB -> ~10 ms over PCIe */
    void* p_big = nullptr;
    void* d_big = nullptr;
    REQUIRE_CUDA(cudaHostAlloc(&p_big, kBig, cudaHostAllocDefault));
    REQUIRE_CUDA(cudaMalloc(&d_big, kBig));

    cudaStream_t copy_stream, decode_stream;
    REQUIRE_CUDA(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    REQUIRE_CUDA(
        cudaStreamCreateWithFlags(&decode_stream, cudaStreamNonBlocking));

    cudaEvent_t e0, e1, origin, copy_done, decode_done;
    REQUIRE_CUDA(cudaEventCreate(&e0));
    REQUIRE_CUDA(cudaEventCreate(&e1));
    REQUIRE_CUDA(cudaEventCreate(&origin));
    REQUIRE_CUDA(cudaEventCreate(&copy_done));
    REQUIRE_CUDA(cudaEventCreate(&decode_done));

    auto do_copy = [&]() {
        (void)cudaMemcpyAsync(d_big, p_big, kBig, cudaMemcpyHostToDevice,
                              copy_stream);
    };
    auto do_decode = [&]() {
        (void)cudec_lz4_decompress_batch(dd_s, dd_ss, dd_d, dd_dc, n, dd_r,
                                         decode_stream);
    };

    /* Warm both paths (module load, first-copy page-locking), then time each
     * alone. */
    do_decode();
    do_copy();
    REQUIRE_CUDA(cudaDeviceSynchronize());
    const float copy_ms = TimeAlone(copy_stream, e0, e1, do_copy);
    const float decode_ms = TimeAlone(decode_stream, e0, e1, do_decode);
    REQUIRE(copy_ms > 0.0f && decode_ms > 0.0f);
    /* Both stages must be well above timing jitter for the margin to mean
     * something. */
    REQUIRE_CTX(copy_ms > 2.0f && decode_ms > 2.0f,
                "stages too short to time overlap: copy %.2f ms, decode %.2f ms",
                copy_ms, decode_ms);

    /* Run both concurrently from a common origin; measure the combined wall. */
    REQUIRE_CUDA(cudaEventRecord(origin, copy_stream));
    REQUIRE_CUDA(cudaStreamWaitEvent(decode_stream, origin, 0));
    do_copy();
    REQUIRE_CUDA(cudaEventRecord(copy_done, copy_stream));
    do_decode();
    REQUIRE_CUDA(cudaEventRecord(decode_done, decode_stream));
    REQUIRE_CUDA(cudaEventSynchronize(copy_done));
    REQUIRE_CUDA(cudaEventSynchronize(decode_done));

    float t_copy_end = 0.0f;
    float t_decode_end = 0.0f;
    REQUIRE_CUDA(cudaEventElapsedTime(&t_copy_end, origin, copy_done));
    REQUIRE_CUDA(cudaEventElapsedTime(&t_decode_end, origin, decode_done));
    const float concurrent_ms = std::max(t_copy_end, t_decode_end);

    /* A serialized run would take ~copy_ms + decode_ms; true overlap hides the
     * shorter stage behind the longer. Require at least half of the shorter
     * stage hidden - a wide margin around the ~2x concurrent/serial gap. */
    const float shorter = std::min(copy_ms, decode_ms);
    const float serial_ms = copy_ms + decode_ms;
    REQUIRE_CTX(concurrent_ms < serial_ms - 0.5f * shorter,
                "no overlap: concurrent %.2f ms vs serial %.2f ms (copy %.2f + "
                "decode %.2f)",
                concurrent_ms, serial_ms, copy_ms, decode_ms);

    std::printf("PASS: H2D copy (%.2f ms) and decode (%.2f ms) run concurrently "
                "- combined %.2f ms vs %.2f ms serial (overlap capability the "
                "streaming path relies on)\n",
                copy_ms, decode_ms, concurrent_ms, serial_ms);
    return 0;
}
