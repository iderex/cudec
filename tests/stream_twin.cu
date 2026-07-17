/* The streaming decode entry (cudec_lz4_decompress_stream) held to the
 * substitute-for-compute-sanitizer gate, extended to the new concurrency
 * axis: the SAME input decoded at every stream count and in both memory
 * spaces must be BIT-IDENTICAL (determinism under interleaving), and equal
 * to the CPU oracle's decode (correctness). Plus the fail-closed ladder: a
 * corrupt chunk mid-batch reports its error while neighbours decode OK and
 * the aggregate carries it; zero chunks / a NULL dst reject synchronously;
 * an undersized dst[k] fails only that chunk. #24. */
#include "cudec.h"
#include "fixtures.h"
#include "require.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr unsigned char kPoison = 0xA5;

struct SChunk {
    const std::vector<unsigned char>* src;
    size_t cap;
};

/* Runs one batch through the streaming entry for a given memory space and
 * stream count. Returns the per-chunk results, the host-visible destination
 * bytes (downloaded for device space), and the aggregate status. Device
 * allocations are reclaimed at process exit (the gpu label is RUN_SERIAL and
 * no state is kept across tests) - the same convention as gpu_fixture. */
int RunStream(const std::vector<SChunk>& chunks, cudec_mem_space space,
              unsigned streams, std::vector<cudec_chunk_result>* results,
              std::vector<std::vector<unsigned char>>* dst_bytes,
              cudec_status* aggregate) {
    const size_t n = chunks.size();
    std::vector<const void*> h_src(n);
    std::vector<size_t> h_ssz(n);
    std::vector<void*> h_dst(n);
    std::vector<size_t> h_cap(n);
    std::vector<std::vector<unsigned char>> host_dst;
    if (space == CUDEC_MEM_HOST) {
        host_dst.resize(n);
    }
    for (size_t i = 0; i < n; i++) {
        h_src[i] = chunks[i].src->data();
        h_ssz[i] = chunks[i].src->size();
        h_cap[i] = chunks[i].cap;
        if (space == CUDEC_MEM_HOST) {
            host_dst[i].assign(chunks[i].cap, kPoison);
            h_dst[i] = host_dst[i].data();
        } else {
            void* d = nullptr;
            REQUIRE_CUDA(cudaMalloc(&d, chunks[i].cap ? chunks[i].cap : 1));
            if (chunks[i].cap) {
                REQUIRE_CUDA(cudaMemset(d, kPoison, chunks[i].cap));
            }
            h_dst[i] = d;
        }
    }

    results->assign(n, cudec_chunk_result{});
    *aggregate = cudec_lz4_decompress_stream(
        h_src.data(), h_ssz.data(), h_dst.data(), h_cap.data(), n, space,
        streams, results->data());

    dst_bytes->assign(n, {});
    for (size_t i = 0; i < n; i++) {
        (*dst_bytes)[i].assign(chunks[i].cap, 0);
        if (chunks[i].cap == 0) {
            continue;
        }
        if (space == CUDEC_MEM_HOST) {
            std::memcpy((*dst_bytes)[i].data(), host_dst[i].data(),
                        chunks[i].cap);
        } else {
            REQUIRE_CUDA(cudaMemcpy((*dst_bytes)[i].data(), h_dst[i],
                                    chunks[i].cap, cudaMemcpyDeviceToHost));
        }
    }
    return 0;
}

}  // namespace

int main() {
    const auto fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());

    /* A batch spanning several 64-chunk waves so the multi-stream
     * round-robin actually engages (single-wave batches never interleave). */
    std::vector<SChunk> batch;
    std::vector<const std::vector<unsigned char>*> expect;
    for (int rep = 0; rep < 40; rep++) {
        for (const auto& f : fixtures) {
            batch.push_back(SChunk{&f.compressed, f.original.size() + 8});
            expect.push_back(&f.original);
        }
    }
    const size_t n = batch.size();
    REQUIRE(n > 128); /* > 2 waves */

    /* Reference: device space, one stream (no overlap). Every other run must
     * match its decoded bytes exactly. */
    std::vector<cudec_chunk_result> ref_res;
    std::vector<std::vector<unsigned char>> ref_dst;
    cudec_status ref_agg;
    REQUIRE(RunStream(batch, CUDEC_MEM_DEVICE, 1, &ref_res, &ref_dst,
                      &ref_agg) == 0);
    REQUIRE(ref_agg == CUDEC_OK);
    for (size_t i = 0; i < n; i++) {
        REQUIRE_CTX(ref_res[i].status == CUDEC_OK, "ref chunk %zu", i);
        REQUIRE_CTX(ref_res[i].bytes_written == expect[i]->size(),
                    "ref bw %zu", i);
        REQUIRE_CTX(equal_bytes(ref_dst[i].data(), expect[i]->data(),
                                expect[i]->size()),
                    "ref bytes %zu", i);
        /* Device space: the kernel writes exactly bytes_written and leaves
         * the poison beyond it untouched. */
        for (size_t j = expect[i]->size(); j < ref_dst[i].size(); j++) {
            REQUIRE_CTX(ref_dst[i][j] == kPoison, "ref poison %zu@%zu", i, j);
        }
    }

    /* Determinism + oracle parity across both spaces and stream counts: the
     * WHOLE caller buffer (decoded prefix + the untouched poison tail) is
     * bit-identical to the reference for every (space, streams). Comparing
     * the full buffer, not just the prefix, also locks the no-clobber
     * property - both spaces must leave the space beyond bytes_written as the
     * caller left it. This is the masterplan's substitute gate extended to
     * the concurrency axis. */
    const cudec_mem_space spaces[] = {CUDEC_MEM_DEVICE, CUDEC_MEM_HOST};
    const unsigned stream_counts[] = {1, 2, 4, 7};
    for (cudec_mem_space space : spaces) {
        for (unsigned s : stream_counts) {
            std::vector<cudec_chunk_result> res;
            std::vector<std::vector<unsigned char>> dst;
            cudec_status agg;
            REQUIRE(RunStream(batch, space, s, &res, &dst, &agg) == 0);
            REQUIRE_CTX(agg == CUDEC_OK, "agg space=%d streams=%u",
                        static_cast<int>(space), s);
            for (size_t i = 0; i < n; i++) {
                REQUIRE_CTX(res[i].status == CUDEC_OK, "ok space=%d s=%u c=%zu",
                            static_cast<int>(space), s, i);
                REQUIRE_CTX(res[i].bytes_written == ref_res[i].bytes_written,
                            "bw space=%d s=%u c=%zu", static_cast<int>(space),
                            s, i);
                REQUIRE_CTX(dst[i].size() == ref_dst[i].size(),
                            "cap space=%d s=%u c=%zu", static_cast<int>(space),
                            s, i);
                REQUIRE_CTX(equal_bytes(dst[i].data(), ref_dst[i].data(),
                                        dst[i].size()),
                            "determinism space=%d s=%u c=%zu",
                            static_cast<int>(space), s, i);
            }
        }
    }

    /* Fail-closed ladder. */

    /* A corrupt chunk between two valid ones: it reports its defined error
     * with no output, the neighbours decode OK, and the aggregate carries
     * the corrupt chunk's status. The corrupt stream is the first
     * oracle-rejected mutant (cudec rejects everything the oracle rejects). */
    std::vector<unsigned char> corrupt;
    for (const auto& f : fixtures) {
        for (auto& m : MutateStream(f.compressed, f.seed)) {
            std::vector<unsigned char> decoded;
            if (!OracleDecodes(m.stream, f.original.size(), &decoded)) {
                corrupt = std::move(m.stream);
                break;
            }
        }
        if (!corrupt.empty()) {
            break;
        }
    }
    REQUIRE(!corrupt.empty());
    for (cudec_mem_space space : spaces) {
        std::vector<SChunk> b = {
            SChunk{&fixtures[0].compressed, fixtures[0].original.size() + 8},
            SChunk{&corrupt, 1 << 16},
            SChunk{&fixtures[0].compressed, fixtures[0].original.size() + 8}};
        std::vector<cudec_chunk_result> res;
        std::vector<std::vector<unsigned char>> dst;
        cudec_status agg;
        REQUIRE(RunStream(b, space, 2, &res, &dst, &agg) == 0);
        REQUIRE_CTX(res[0].status == CUDEC_OK, "space=%d",
                    static_cast<int>(space));
        REQUIRE_CTX(res[2].status == CUDEC_OK, "space=%d",
                    static_cast<int>(space));
        REQUIRE_CTX(res[1].status != CUDEC_OK, "space=%d",
                    static_cast<int>(space));
        REQUIRE_CTX(res[1].bytes_written == 0, "space=%d",
                    static_cast<int>(space));
        REQUIRE_CTX(agg == static_cast<cudec_status>(res[1].status),
                    "space=%d", static_cast<int>(space));
    }

    /* An undersized dst fails only its chunk; the neighbour decodes OK. */
    for (cudec_mem_space space : spaces) {
        std::vector<SChunk> b = {
            SChunk{&fixtures[0].compressed, fixtures[0].original.size() + 8},
            SChunk{&fixtures[0].compressed, fixtures[0].original.size() / 2}};
        std::vector<cudec_chunk_result> res;
        std::vector<std::vector<unsigned char>> dst;
        cudec_status agg;
        REQUIRE(RunStream(b, space, 2, &res, &dst, &agg) == 0);
        REQUIRE_CTX(res[0].status == CUDEC_OK, "space=%d",
                    static_cast<int>(space));
        REQUIRE_CTX(res[1].status == CUDEC_ERR_OUTPUT_TOO_SMALL, "space=%d",
                    static_cast<int>(space));
        REQUIRE_CTX(res[1].bytes_written == 0, "space=%d",
                    static_cast<int>(space));
        REQUIRE_CTX(agg == CUDEC_ERR_OUTPUT_TOO_SMALL, "space=%d",
                    static_cast<int>(space));
    }

    /* Zero chunks and a NULL destination with a claimed capacity reject
     * synchronously. */
    {
        const void* s0 = fixtures[0].compressed.data();
        size_t sz0 = fixtures[0].compressed.size();
        void* d0 = nullptr;
        size_t cap0 = 100;
        cudec_chunk_result r0;
        REQUIRE(cudec_lz4_decompress_stream(&s0, &sz0, &d0, &cap0, 0,
                                            CUDEC_MEM_HOST, 4,
                                            &r0) == CUDEC_ERR_INVALID_ARGUMENT);
        REQUIRE(cudec_lz4_decompress_stream(&s0, &sz0, &d0, &cap0, 1,
                                            CUDEC_MEM_HOST, 4,
                                            &r0) == CUDEC_ERR_INVALID_ARGUMENT);
    }

    std::printf("PASS: streaming decode bit-identical across streams {1,2,4,7} "
                "x {host,device} in oracle parity over %zu chunks; corrupt "
                "isolation, undersized, zero-chunk, null-dst all fail-closed\n",
                n);
    return 0;
}
