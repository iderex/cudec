/* Termination on the device, behind a host-side watchdog (issue #72).
 *
 * tests/termination.cpp proves the shared parser reaches a terminal state on
 * the host; this drives the identical hostile corpus through the real kernel,
 * where the failure mode is worse: a warp that never leaves its sequence loop
 * does not return a bad status, it holds the launch - and with it every other
 * chunk in the batch and the stream behind it.
 *
 * A plain cudaStreamSynchronize would inherit exactly that hang and stall the
 * suite, so the launch is fenced with an event and polled against a wall-clock
 * deadline: an expiry FAILS the test with a message instead of blocking. The
 * ctest TIMEOUT on this target is the second line of defence; the process exit
 * on failure is what releases the still-running launch (the driver tears the
 * context down - the test deliberately does not synchronise on the way out). */
#include "adversarial_blocks.h"
#include "cudec.h"
#include "fixtures.h"
#include "lz4_decode.cuh"
#include "require.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr unsigned char kDstPoison = 0xA5;
/* Generous against a loaded GPU, tiny against "never": the whole corpus is a
 * few megabytes and decodes in milliseconds. */
constexpr double kDeadlineSeconds = 30.0;

struct Chunk {
    std::string name;
    std::vector<unsigned char> src;
    size_t dst_capacity;
};

/* Everything hostile the project can generate: the crafted corpus, the
 * seeded fixture mutants, and every fixture truncated at three points -
 * a stream cut mid-sequence, mid-length-extension, and mid-offset. */
std::vector<Chunk> MakeHostileChunks(const std::vector<Fixture>& fixtures) {
    std::vector<Chunk> chunks;
    for (const AdversarialBlock& block : MakeAdversarialBlocks()) {
        chunks.push_back({block.name, block.stream, block.dst_capacity});
    }
    for (const Fixture& f : fixtures) {
        chunks.push_back({f.name + "/pristine", f.compressed,
                          f.original.size()});
        for (const Mutant& m : MutateStream(f.compressed, f.seed)) {
            chunks.push_back(
                {f.name + "/" + m.description, m.stream, f.original.size()});
        }
    }
    return chunks;
}

/* Uploads the batch into one source blob and one destination arena, plus the
 * five ABI tables. */
struct Batch {
    size_t n = 0;
    std::vector<std::string> names;
    unsigned char* d_src_blob = nullptr;
    unsigned char* d_dst_arena = nullptr;
    size_t dst_arena_size = 0;
    const void** d_srcs = nullptr;
    void** d_dsts = nullptr;
    size_t* d_sizes = nullptr;
    size_t* d_caps = nullptr;
    cudec_chunk_result* d_results = nullptr;
};

size_t RoundUp16(size_t v) { return (v + 15) & ~size_t{15}; }

int BuildBatch(const std::vector<Chunk>& chunks, Batch* batch) {
    const size_t n = chunks.size();
    std::vector<size_t> src_off(n), dst_off(n), sizes(n), caps(n);
    size_t src_total = 0;
    size_t dst_total = 0;
    for (size_t i = 0; i < n; i++) {
        src_off[i] = src_total;
        dst_off[i] = dst_total;
        sizes[i] = chunks[i].src.size();
        caps[i] = chunks[i].dst_capacity;
        src_total += RoundUp16(sizes[i] ? sizes[i] : 1);
        dst_total += RoundUp16(caps[i] ? caps[i] : 1);
    }
    std::vector<unsigned char> src_blob(src_total, 0);
    std::vector<const void*> h_srcs(n);
    std::vector<void*> h_dsts(n);

    batch->n = n;
    batch->dst_arena_size = dst_total;
    REQUIRE_CUDA(cudaMalloc(&batch->d_src_blob, src_total));
    REQUIRE_CUDA(cudaMalloc(&batch->d_dst_arena, dst_total));
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < sizes[i]; k++) {
            src_blob[src_off[i] + k] = chunks[i].src[k];
        }
        batch->names.push_back(chunks[i].name);
        h_srcs[i] = batch->d_src_blob + src_off[i];
        h_dsts[i] = batch->d_dst_arena + dst_off[i];
    }
    REQUIRE_CUDA(cudaMemcpy(batch->d_src_blob, src_blob.data(), src_total,
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemset(batch->d_dst_arena, kDstPoison, dst_total));

    REQUIRE_CUDA(cudaMalloc(&batch->d_srcs, n * sizeof(*batch->d_srcs)));
    REQUIRE_CUDA(cudaMalloc(&batch->d_dsts, n * sizeof(*batch->d_dsts)));
    REQUIRE_CUDA(cudaMalloc(&batch->d_sizes, n * sizeof(*batch->d_sizes)));
    REQUIRE_CUDA(cudaMalloc(&batch->d_caps, n * sizeof(*batch->d_caps)));
    REQUIRE_CUDA(cudaMalloc(&batch->d_results, n * sizeof(*batch->d_results)));
    REQUIRE_CUDA(cudaMemcpy(batch->d_srcs, h_srcs.data(),
                            n * sizeof(*batch->d_srcs),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(batch->d_dsts, h_dsts.data(),
                            n * sizeof(*batch->d_dsts),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(batch->d_sizes, sizes.data(),
                            n * sizeof(*batch->d_sizes),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(batch->d_caps, caps.data(),
                            n * sizeof(*batch->d_caps),
                            cudaMemcpyHostToDevice));
    /* The 0xFF sentinel is out of the status enum's range, so a chunk the
     * kernel never reached cannot be mistaken for a decoded one. */
    REQUIRE_CUDA(
        cudaMemset(batch->d_results, 0xFF, n * sizeof(*batch->d_results)));
    return 0;
}

/* Submits the batch and waits on an event, not on the stream: a launch that
 * does not finish must be REPORTED, never waited on. */
int DecodeWithWatchdog(const Batch& b, size_t begin, size_t count) {
    cudaStream_t stream;
    cudaEvent_t finished;
    REQUIRE_CUDA(cudaStreamCreate(&stream));
    REQUIRE_CUDA(cudaEventCreateWithFlags(&finished, cudaEventDisableTiming));
    REQUIRE(cudec_lz4_decompress_batch(b.d_srcs + begin, b.d_sizes + begin,
                                       b.d_dsts + begin, b.d_caps + begin,
                                       count, b.d_results + begin,
                                       stream) == CUDEC_OK);
    REQUIRE_CUDA(cudaEventRecord(finished, stream));

    const auto start = std::chrono::steady_clock::now();
    while (true) {
        const cudaError_t query = cudaEventQuery(finished);
        if (query == cudaSuccess) {
            break;
        }
        REQUIRE_CTX(query == cudaErrorNotReady, "cudaEventQuery failed: %s",
                    cudaGetErrorString(query));
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count();
        /* On expiry the launch is still resident. Returning (and exiting)
         * leaves the teardown to the driver; synchronising or destroying the
         * stream here would block on the very hang being reported. */
        REQUIRE_CTX(elapsed < kDeadlineSeconds,
                    "decode of chunks [%zu,%zu) did not complete within "
                    "%.0f s - the corpus is a few MB of bounded-parse "
                    "chunks, so this is a non-terminating decode",
                    begin, begin + count, kDeadlineSeconds);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE_CUDA(cudaEventDestroy(finished));
    REQUIRE_CUDA(cudaStreamDestroy(stream));
    return 0;
}

/* A launch geometry that does not hold one whole warp: the grid-stride loop
 * would advance by zero warps per step and spin forever, so the kernel has to
 * refuse it itself rather than trust its caller. The shipped entry point never
 * produces this geometry, which is exactly why nothing else would catch a
 * regression here. Same watchdog: a spin is reported, not waited on. */
int RequireSubWarpLaunchTerminates(const Batch& b) {
    cudaStream_t stream;
    cudaEvent_t finished;
    REQUIRE_CUDA(cudaStreamCreate(&stream));
    REQUIRE_CUDA(cudaEventCreateWithFlags(&finished, cudaEventDisableTiming));
    cudec_detail::lz4_decode_batch<false><<<1, 16, 0, stream>>>(
        b.d_srcs, b.d_sizes, b.d_dsts, b.d_caps, b.n, b.d_results);
    REQUIRE_CUDA(cudaGetLastError());
    REQUIRE_CUDA(cudaEventRecord(finished, stream));
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        const cudaError_t query = cudaEventQuery(finished);
        if (query == cudaSuccess) {
            break;
        }
        REQUIRE_CTX(query == cudaErrorNotReady, "cudaEventQuery failed: %s",
                    cudaGetErrorString(query));
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count();
        REQUIRE_CTX(elapsed < kDeadlineSeconds,
                    "a sub-warp launch geometry (<<<1, 16>>>) did not "
                    "terminate within %.0f s",
                    kDeadlineSeconds);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE_CUDA(cudaEventDestroy(finished));
    REQUIRE_CUDA(cudaStreamDestroy(stream));
    return 0;
}

int CheckResults(const Batch& b) {
    std::vector<cudec_chunk_result> results(b.n);
    REQUIRE_CUDA(cudaMemcpy(results.data(), b.d_results,
                            b.n * sizeof(*b.d_results),
                            cudaMemcpyDeviceToHost));
    size_t rejected = 0;
    for (size_t i = 0; i < b.n; i++) {
        const cudec_chunk_result& r = results[i];
        REQUIRE_CTX(r.status == CUDEC_OK ||
                        r.status == CUDEC_ERR_CORRUPT_INPUT ||
                        r.status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                    "chunk %zu (%s): undefined status %d - the 0xFF sentinel "
                    "means the kernel never reached this chunk",
                    i, b.names[i].c_str(), static_cast<int>(r.status));
        REQUIRE_CTX(r.reserved == 0, "chunk %zu (%s): reserved word not zeroed",
                    i, b.names[i].c_str());
        if (r.status != CUDEC_OK) {
            REQUIRE_CTX(r.bytes_written == 0,
                        "chunk %zu (%s): rejected chunk reports %llu bytes", i,
                        b.names[i].c_str(),
                        static_cast<unsigned long long>(r.bytes_written));
            rejected++;
        }
    }
    /* A corpus the decoder accepted wholesale would prove nothing about the
     * reject paths' termination. */
    REQUIRE(rejected != 0);
    std::printf("termination_gpu: %zu chunks decoded within the watchdog "
                "deadline (%zu rejected, all with a defined status)\n",
                b.n, rejected);
    return 0;
}

}  // namespace

int main() {
    const std::vector<Fixture> fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());
    const std::vector<Chunk> chunks = MakeHostileChunks(fixtures);
    REQUIRE(!chunks.empty());

    Batch batch;
    REQUIRE(BuildBatch(chunks, &batch) == 0);

    /* The whole corpus in one launch: a single non-terminating warp holds
     * the launch, which is exactly the failure this test must observe. */
    REQUIRE(DecodeWithWatchdog(batch, 0, batch.n) == 0);
    /* And one chunk per launch, so a hostile stream cannot hide behind a
     * batch mate that happens to keep the warp busy. */
    for (size_t i = 0; i < batch.n; i++) {
        REQUIRE(DecodeWithWatchdog(batch, i, 1) == 0);
    }
    /* After the real launches, so the results it must not touch are already
     * the ones CheckResults verifies below. */
    REQUIRE(RequireSubWarpLaunchTerminates(batch) == 0);
    REQUIRE(CheckResults(batch) == 0);
    return 0;
}
