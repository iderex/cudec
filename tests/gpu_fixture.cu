/* The fixture corpus through the real batch plumbing (device pointer
 * tables, per-chunk sizes/capacities, poisoned destinations) - exactly the
 * path the M1 decoder lands into. Today it pins the stub contract that
 * smoke.cu does not cover: every chunk reports NOT_IMPLEMENTED and the
 * destination buffers stay untouched (the header's "writes no output"
 * promise). The reject-parity implication for mutants is written and
 * executing now; M1 flips the pristine-pair expectations to CUDEC_OK plus
 * byte-equality against the oracle output. */
#include "cudec.h"
#include "fixtures.h"
#include "require.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr unsigned char kDstPoison = 0xA5;

struct Chunk {
    std::string context;
    const std::vector<unsigned char>* src;
    size_t dst_capacity;
};

/* Uploads a batch, runs it through the entry point on a created stream,
 * and returns the per-chunk results plus the downloaded (poisoned)
 * destination buffers. Plain int-returning so REQUIRE can early-abort. */
int RunBatch(const std::vector<Chunk>& chunks,
             std::vector<cudec_chunk_result>* results,
             std::vector<std::vector<unsigned char>>* dst_bytes) {
    const size_t n = chunks.size();
    std::vector<const void*> h_srcs(n);
    std::vector<void*> h_dsts(n);
    std::vector<size_t> h_sizes(n);
    std::vector<size_t> h_caps(n);
    for (size_t i = 0; i < n; i++) {
        void* d_src = nullptr;
        void* d_dst = nullptr;
        const size_t src_size = chunks[i].src->size();
        REQUIRE_CUDA(cudaMalloc(&d_src, src_size ? src_size : 1));
        if (src_size) {
            REQUIRE_CUDA(cudaMemcpy(d_src, chunks[i].src->data(), src_size,
                                    cudaMemcpyHostToDevice));
        }
        REQUIRE_CUDA(cudaMalloc(&d_dst, chunks[i].dst_capacity));
        REQUIRE_CUDA(cudaMemset(d_dst, kDstPoison, chunks[i].dst_capacity));
        h_srcs[i] = d_src;
        h_dsts[i] = d_dst;
        h_sizes[i] = src_size;
        h_caps[i] = chunks[i].dst_capacity;
    }

    const void** d_srcs;
    void** d_dsts;
    size_t* d_sizes;
    size_t* d_caps;
    cudec_chunk_result* d_results;
    REQUIRE_CUDA(cudaMalloc(&d_srcs, n * sizeof(*d_srcs)));
    REQUIRE_CUDA(cudaMalloc(&d_dsts, n * sizeof(*d_dsts)));
    REQUIRE_CUDA(cudaMalloc(&d_sizes, n * sizeof(*d_sizes)));
    REQUIRE_CUDA(cudaMalloc(&d_caps, n * sizeof(*d_caps)));
    REQUIRE_CUDA(cudaMalloc(&d_results, n * sizeof(*d_results)));
    REQUIRE_CUDA(cudaMemcpy(d_srcs, h_srcs.data(), n * sizeof(*d_srcs),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(d_dsts, h_dsts.data(), n * sizeof(*d_dsts),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(d_sizes, h_sizes.data(), n * sizeof(*d_sizes),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(d_caps, h_caps.data(), n * sizeof(*d_caps),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemset(d_results, 0xFF, n * sizeof(*d_results)));

    cudaStream_t stream;
    REQUIRE_CUDA(cudaStreamCreate(&stream));
    REQUIRE(cudec_lz4_decompress_batch(d_srcs, d_sizes, d_dsts, d_caps, n,
                                       d_results, stream) == CUDEC_OK);
    REQUIRE_CUDA(cudaStreamSynchronize(stream));
    REQUIRE_CUDA(cudaStreamDestroy(stream));

    results->assign(n, cudec_chunk_result{});
    REQUIRE_CUDA(cudaMemcpy(results->data(), d_results,
                            n * sizeof(*d_results), cudaMemcpyDeviceToHost));
    dst_bytes->assign(n, {});
    for (size_t i = 0; i < n; i++) {
        (*dst_bytes)[i].assign(h_caps[i], 0);
        REQUIRE_CUDA(cudaMemcpy((*dst_bytes)[i].data(), h_dsts[i], h_caps[i],
                                cudaMemcpyDeviceToHost));
    }
    /* Process teardown reclaims the device allocations; the harness keeps
     * no device state across tests (RUN_SERIAL on the gpu label). */
    return 0;
}

int CheckStubContract(const std::vector<Chunk>& chunks,
                      const std::vector<cudec_chunk_result>& results,
                      const std::vector<std::vector<unsigned char>>& dsts) {
    const std::vector<unsigned char> poison_row(65536, kDstPoison);
    for (size_t i = 0; i < chunks.size(); i++) {
        const char* ctx = chunks[i].context.c_str();
        REQUIRE_CTX(results[i].status == CUDEC_ERR_NOT_IMPLEMENTED, "%s", ctx);
        REQUIRE_CTX(results[i].reserved == 0, "%s", ctx);
        REQUIRE_CTX(results[i].bytes_written == 0, "%s", ctx);
        /* "Writes no output": the poison must survive verbatim. */
        REQUIRE_CTX(dsts[i].size() <= poison_row.size(), "%s", ctx);
        REQUIRE_CTX(equal_bytes(dsts[i].data(), poison_row.data(),
                                dsts[i].size()),
                    "%s", ctx);
    }
    return 0;
}

}  // namespace

int main() {
    /* Self-sufficiently non-vacuous: this test must bite on its own, not
     * by courtesy of oracle_lz4 running in the same suite. */
    const auto fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());

    /* Batch 1: the pristine pairs - the exact plumbing the M1 decoder
     * inherits; at M1 these expectations flip to CUDEC_OK + bytes_written
     * + byte-equality against the oracle output. */
    std::vector<Chunk> pairs;
    for (const auto& f : fixtures) {
        pairs.push_back(Chunk{f.name, &f.compressed, f.original.size()});
    }
    std::vector<cudec_chunk_result> results;
    std::vector<std::vector<unsigned char>> dsts;
    REQUIRE(RunBatch(pairs, &results, &dsts) == 0);
    REQUIRE(CheckStubContract(pairs, results, dsts) == 0);

    /* Batch 2: the mutant corpus with oracle verdicts. The reject-parity
     * implication (oracle rejects => cudec must not report success) is
     * armed and executing today - trivially satisfied by the stub, load-
     * bearing from M1 on. */
    std::vector<std::vector<unsigned char>> mutant_streams;
    std::vector<std::string> mutant_contexts;
    std::vector<size_t> mutant_capacities;
    std::vector<bool> oracle_rejected;
    for (const auto& f : fixtures) {
        for (auto& m : MutateStream(f.compressed, f.seed)) {
            std::vector<unsigned char> decoded;
            oracle_rejected.push_back(
                !OracleDecodes(m.stream, f.original.size(), &decoded));
            mutant_streams.push_back(std::move(m.stream));
            mutant_contexts.push_back(f.name + "/" + m.description);
            mutant_capacities.push_back(f.original.size());
        }
    }
    /* Chunk keeps pointers into mutant_streams, so it is built only after
     * the vector stopped growing (reallocation would dangle them). */
    std::vector<Chunk> mutant_chunks;
    for (size_t i = 0; i < mutant_streams.size(); i++) {
        mutant_chunks.push_back(Chunk{mutant_contexts[i], &mutant_streams[i],
                                      mutant_capacities[i]});
    }
    REQUIRE(RunBatch(mutant_chunks, &results, &dsts) == 0);
    REQUIRE(CheckStubContract(mutant_chunks, results, dsts) == 0);
    size_t rejected_count = 0;
    for (size_t i = 0; i < mutant_chunks.size(); i++) {
        if (!oracle_rejected[i]) {
            continue;
        }
        rejected_count++;
        /* M1 note: for mutants the oracle ACCEPTS, the flip must compare
         * against the oracle's own output and size (a truncation can land
         * on a valid, shorter stream) - never against f.original. */
        REQUIRE_CTX(results[i].status != CUDEC_OK,
                    "reject parity (armed for M1): %s",
                    mutant_chunks[i].context.c_str());
    }
    /* The parity arm must have teeth: at least one mutant per corpus is
     * oracle-rejected, or this whole loop was vacuous. */
    REQUIRE(rejected_count > 0);

    std::printf("PASS: %zu pairs + %zu mutants (%zu oracle-rejected) through "
                "the batch plumbing; stub contract and poison intact\n",
                pairs.size(), mutant_chunks.size(), rejected_count);
    return 0;
}
