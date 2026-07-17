/* The fixture corpus through the real batch plumbing (device pointer
 * tables, per-chunk sizes/capacities, poisoned destinations) on the GPU
 * decoder. Pristine pairs must decode to the original with byte-equality;
 * the mutant corpus is held to the same two-direction oracle-parity
 * contract as the CPU twin (where cudec accepts, liblz4 accepts and the
 * bytes match; where liblz4 rejects, cudec rejects), with the documented
 * offset==0 stricter case allowed. On success the decoder writes exactly
 * bytes_written, so the poison beyond it must survive. */
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

/* Allocates the five device pointer/size/result tables for an n-chunk
 * batch, uploads the four host tables, and primes the results buffer with
 * the 0xFF non-OK sentinel. Shared by RunBatch and the grid-stride
 * wraparound test below, which stage their host tables differently
 * (per-chunk allocation vs. one shared src/dst buffer repeated wrap_n
 * times) but need the identical device-side upload. */
int UploadBatchTables(size_t n, const std::vector<const void*>& h_srcs,
                      const std::vector<void*>& h_dsts,
                      const std::vector<size_t>& h_sizes,
                      const std::vector<size_t>& h_caps,
                      const void*** d_srcs, void*** d_dsts, size_t** d_sizes,
                      size_t** d_caps, cudec_chunk_result** d_results) {
    REQUIRE_CUDA(cudaMalloc(d_srcs, n * sizeof(**d_srcs)));
    REQUIRE_CUDA(cudaMalloc(d_dsts, n * sizeof(**d_dsts)));
    REQUIRE_CUDA(cudaMalloc(d_sizes, n * sizeof(**d_sizes)));
    REQUIRE_CUDA(cudaMalloc(d_caps, n * sizeof(**d_caps)));
    REQUIRE_CUDA(cudaMalloc(d_results, n * sizeof(**d_results)));
    REQUIRE_CUDA(cudaMemcpy(*d_srcs, h_srcs.data(), n * sizeof(**d_srcs),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(*d_dsts, h_dsts.data(), n * sizeof(**d_dsts),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(*d_sizes, h_sizes.data(), n * sizeof(**d_sizes),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(*d_caps, h_caps.data(), n * sizeof(**d_caps),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemset(*d_results, 0xFF, n * sizeof(**d_results)));
    return 0;
}

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
    REQUIRE(UploadBatchTables(n, h_srcs, h_dsts, h_sizes, h_caps, &d_srcs,
                              &d_dsts, &d_sizes, &d_caps, &d_results) == 0);

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

/* On a successful decode the output is exactly the expected bytes and the
 * poison beyond bytes_written is untouched. */
int CheckDecodedOk(const char* ctx, const cudec_chunk_result& result,
                   const std::vector<unsigned char>& expected,
                   const std::vector<unsigned char>& dst) {
    REQUIRE_CTX(result.status == CUDEC_OK, "%s status=%d", ctx,
                static_cast<int>(result.status));
    REQUIRE_CTX(result.reserved == 0, "%s", ctx);
    REQUIRE_CTX(result.bytes_written == expected.size(), "%s", ctx);
    REQUIRE_CTX(dst.size() >= expected.size(), "%s", ctx);
    REQUIRE_CTX(equal_bytes(dst.data(), expected.data(), expected.size()),
                "%s", ctx);
    for (size_t j = expected.size(); j < dst.size(); j++) {
        REQUIRE_CTX(dst[j] == kDstPoison, "%s poison at %zu", ctx, j);
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
    /* Slack capacity so the "writes exactly bytes_written, poison beyond
     * survives" guarantee is exercised on the primary success path (not
     * only via short-decoding mutants). A larger capacity is still parity-
     * faithful for a valid stream: the terminal/LASTLITERALS checks only
     * grow more lenient, and the decode still yields original.size() bytes. */
    std::vector<Chunk> pairs;
    for (const auto& f : fixtures) {
        pairs.push_back(Chunk{f.name, &f.compressed, f.original.size() + 8});
    }
    std::vector<cudec_chunk_result> results;
    std::vector<std::vector<unsigned char>> dsts;
    REQUIRE(RunBatch(pairs, &results, &dsts) == 0);
    for (size_t i = 0; i < fixtures.size(); i++) {
        REQUIRE(CheckDecodedOk(pairs[i].context.c_str(), results[i],
                               fixtures[i].original, dsts[i]) == 0);
    }

    /* Batch 2: the mutant corpus, held to the two-direction oracle-parity
     * contract - the same as the CPU twin, now on the GPU decoder. */
    std::vector<std::vector<unsigned char>> mutant_streams;
    std::vector<std::string> mutant_contexts;
    std::vector<size_t> mutant_capacities;
    std::vector<bool> oracle_accepts;
    std::vector<std::vector<unsigned char>> oracle_outputs;
    for (const auto& f : fixtures) {
        for (auto& m : MutateStream(f.compressed, f.seed)) {
            std::vector<unsigned char> decoded;
            const bool ok =
                OracleDecodes(m.stream, f.original.size(), &decoded);
            oracle_accepts.push_back(ok);
            oracle_outputs.push_back(ok ? decoded
                                        : std::vector<unsigned char>{});
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
    size_t rejected_count = 0;
    size_t stricter_count = 0;
    std::string stricter_ctx;
    for (size_t i = 0; i < mutant_chunks.size(); i++) {
        const char* ctx = mutant_chunks[i].context.c_str();
        if (results[i].status == CUDEC_OK) {
            /* cudec accepts => liblz4 accepts and the bytes match its own
             * output and size (a truncation can decode to a valid, shorter
             * stream - compare against the oracle output, never
             * f.original). */
            REQUIRE_CTX(oracle_accepts[i], "cudec accepts, liblz4 rejects: %s",
                        ctx);
            REQUIRE(CheckDecodedOk(ctx, results[i], oracle_outputs[i],
                                   dsts[i]) == 0);
        } else {
            /* Failure contract: a rejected chunk reports no output, never
             * presenting its partial dst as a valid decode. */
            REQUIRE_CTX(results[i].bytes_written == 0, "reject bw: %s", ctx);
            REQUIRE_CTX(results[i].reserved == 0, "reject reserved: %s", ctx);
            if (oracle_accepts[i]) {
                stricter_count++;
                stricter_ctx = mutant_chunks[i].context;
            } else {
                rejected_count++;
            }
        }
    }
    /* The parity arm must have teeth: at least one mutant is oracle-
     * rejected. The stricter set is pinned by IDENTITY, not just count: it
     * is exactly the one offset==0 mutant (matching the CPU twin). */
    REQUIRE(rejected_count > 0);
    REQUIRE(stricter_count == 1);
    REQUIRE(stricter_ctx == "text-256/flip-bit-at-88");

    /* Batch 3: the grid-stride re-entry path (a warp decoding more than one
     * chunk). The host caps the grid at 8192 blocks = 32768 warps, so a
     * batch beyond that forces the `chunk += total_warps` wraparound - the
     * exactly-once distribution that no other gate exercises. All chunks
     * share one empty-block src and one dst (an empty block writes nothing),
     * so 40000 chunks cost two device buffers, not 80000 allocations. A
     * skipped chunk keeps its 0xFF-poisoned result (status != OK); a
     * double-decode is idempotent - so requiring every result CUDEC_OK with
     * bytes_written 0 proves each chunk was decoded exactly once. */
    {
        const size_t wrap_n = 40000;
        unsigned char* d_src_one;
        unsigned char* d_dst_one;
        REQUIRE_CUDA(cudaMalloc(&d_src_one, 1));
        REQUIRE_CUDA(cudaMemset(d_src_one, 0, 1)); /* empty-block token */
        REQUIRE_CUDA(cudaMalloc(&d_dst_one, 1));
        std::vector<const void*> h_s(wrap_n, d_src_one);
        std::vector<void*> h_d(wrap_n, d_dst_one);
        std::vector<size_t> h_sz(wrap_n, 1);
        std::vector<size_t> h_cp(wrap_n, 1);
        const void** d_s;
        void** d_d;
        size_t* d_sz;
        size_t* d_cp;
        cudec_chunk_result* d_r;
        REQUIRE(UploadBatchTables(wrap_n, h_s, h_d, h_sz, h_cp, &d_s, &d_d,
                                  &d_sz, &d_cp, &d_r) == 0);
        REQUIRE(cudec_lz4_decompress_batch(d_s, d_sz, d_d, d_cp, wrap_n, d_r,
                                           nullptr) == CUDEC_OK);
        REQUIRE_CUDA(cudaDeviceSynchronize());
        std::vector<cudec_chunk_result> wrap_res(wrap_n);
        REQUIRE_CUDA(cudaMemcpy(wrap_res.data(), d_r, wrap_n * sizeof(*d_r),
                                cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < wrap_n; i++) {
            REQUIRE_CTX(wrap_res[i].status == CUDEC_OK, "wrap chunk %zu", i);
            REQUIRE_CTX(wrap_res[i].bytes_written == 0, "wrap chunk %zu", i);
        }
    }

    std::printf("PASS: %zu pairs decoded byte-exact + %zu mutants in oracle "
                "parity (%zu oracle-rejected, %zu offset==0 stricter) + 40000 "
                "grid-stride wraparound chunks on the GPU decoder; failure "
                "contract and poison beyond bytes_written intact\n",
                pairs.size(), mutant_chunks.size(), rejected_count,
                stricter_count);
    return 0;
}
