/* The fixture corpus through the real batch plumbing (device pointer
 * tables, per-chunk sizes/capacities, poisoned destinations) on the GPU
 * decoder. Pristine pairs must decode to the original with byte-equality;
 * the mutant corpus is held to the same two-direction oracle-parity
 * contract as the CPU twin (where cudec accepts, liblz4 accepts and the
 * bytes match; where liblz4 rejects, cudec rejects), with the documented
 * offset==0 stricter case allowed. On success the decoder writes exactly
 * bytes_written, so the poison beyond it must survive.
 *
 * Both formats run through here, and the same plumbing drives both on
 * purpose (issue #150): the two entry points differ in one template
 * argument to one chunk decoder, so a difference this test could see would
 * be a difference in the parser and nowhere else. The Snappy half is the
 * pristine direction only - the mutant reject parity and the determinism
 * gate for that format are the device gate set (#154). */
#include "adversarial_snappy_blocks.h"
#include "cudec.h"
#include "fixtures.h"
#include "require.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifndef CUDEC_SNAPPY_TESTDATA_DIR
#error "CUDEC_SNAPPY_TESTDATA_DIR must name the pinned archive's testdata/"
#endif

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

/* The shape both public batch entries have. Taken as a parameter rather
 * than branched on, so the plumbing below cannot treat one format
 * differently from the other by accident. */
using BatchEntry = cudec_status (*)(const void* const*, const size_t*,
                                    void* const*, const size_t*, size_t,
                                    cudec_chunk_result*, cudec_stream_t);

/* Uploads a batch, runs it through the entry point on a created stream,
 * and returns the per-chunk results plus the downloaded (poisoned)
 * destination buffers. Plain int-returning so REQUIRE can early-abort. */
int RunBatch(const std::vector<Chunk>& chunks, BatchEntry entry,
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
    REQUIRE(entry(d_srcs, d_sizes, d_dsts, d_caps, n, d_results, stream) ==
            CUDEC_OK);
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

/* One file out of the pinned snappy archive's testdata/, read whole. The
 * bytes are proved by the archive's URL_HASH, so nothing is copied into
 * this tree; a missing or unreadable file is an infrastructure failure and
 * the caller reds on the empty result rather than skipping the entry. */
std::vector<unsigned char> ReadTestdata(const char* name) {
    const std::string path = std::string(CUDEC_SNAPPY_TESTDATA_DIR) + "/" + name;
    std::ifstream in(path.c_str(), std::ios::binary);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(in),
                                      std::istreambuf_iterator<char>());
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
    REQUIRE(RunBatch(pairs, cudec_lz4_decompress_batch, &results, &dsts) ==
            0);
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
    REQUIRE(RunBatch(mutant_chunks, cudec_lz4_decompress_batch, &results,
                     &dsts) == 0);
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

    /* Batch 4: the generated Snappy corpus through the Snappy
     * instantiation of the same chunk decoder, at the same slack capacity,
     * so the exactly-bytes_written guarantee is exercised on this format's
     * primary success path too. A wider capacity cannot make a valid raw
     * stream decode differently: the declaration is compared against it and
     * every later bound is taken from the declaration. */
    const auto snappy = MakeSnappyFixtures();
    REQUIRE(!snappy.empty());
    std::vector<Chunk> snappy_pairs;
    for (const auto& f : snappy) {
        snappy_pairs.push_back(
            Chunk{f.name, &f.compressed, f.original.size() + 8});
    }
    REQUIRE(RunBatch(snappy_pairs, cudec_snappy_decompress_batch, &results,
                     &dsts) == 0);
    for (size_t i = 0; i < snappy.size(); i++) {
        REQUIRE(CheckDecodedOk(snappy_pairs[i].context.c_str(), results[i],
                               snappy[i].original, dsts[i]) == 0);
    }

    /* Batch 5: the Snappy mutant corpus, held to the same two-direction
     * oracle-parity contract as the LZ4 one (issue #154). The capacity is the
     * pristine original's size: a mutant that declares more is refused at the
     * preamble, which is the one place a declared length is compared against
     * a capacity at all.
     *
     * The stricter set is pinned at zero rather than at an identity, and that
     * is a statement rather than a missing case. Snappy's decoder has one
     * documented point where cudec is stricter - the four-byte literal length
     * spelling 0xFFFFFFFF, which wraps to zero in the reference's 32-bit
     * addition - and no mutation of a compressor-produced stream constructs
     * it. The host twin pins that divergence on hand-built bytes; here the
     * aggregate is what makes the named exception readable as the whole of
     * it. */
    std::vector<std::vector<unsigned char>> snappy_mutants;
    std::vector<std::string> snappy_mutant_contexts;
    std::vector<size_t> snappy_mutant_capacities;
    std::vector<bool> snappy_oracle_accepts;
    std::vector<std::vector<unsigned char>> snappy_oracle_outputs;
    for (const auto& f : snappy) {
        for (auto& m : MutateSnappyStream(f.compressed, f.seed)) {
            std::vector<unsigned char> decoded;
            const bool ok = SnappyOracleDecodes(m.stream, &decoded);
            snappy_oracle_accepts.push_back(ok);
            snappy_oracle_outputs.push_back(
                ok ? decoded : std::vector<unsigned char>{});
            snappy_mutants.push_back(std::move(m.stream));
            snappy_mutant_contexts.push_back(f.name + "/" + m.description);
            snappy_mutant_capacities.push_back(f.original.size());
        }
    }
    std::vector<Chunk> snappy_mutant_chunks;
    for (size_t i = 0; i < snappy_mutants.size(); i++) {
        snappy_mutant_chunks.push_back(Chunk{snappy_mutant_contexts[i],
                                             &snappy_mutants[i],
                                             snappy_mutant_capacities[i]});
    }
    REQUIRE(RunBatch(snappy_mutant_chunks, cudec_snappy_decompress_batch,
                     &results, &dsts) == 0);
    size_t snappy_rejected = 0;
    size_t snappy_stricter = 0;
    for (size_t i = 0; i < snappy_mutant_chunks.size(); i++) {
        const char* ctx = snappy_mutant_chunks[i].context.c_str();
        if (results[i].status == CUDEC_OK) {
            REQUIRE_CTX(snappy_oracle_accepts[i],
                        "cudec accepts, snappy rejects: %s", ctx);
            REQUIRE(CheckDecodedOk(ctx, results[i], snappy_oracle_outputs[i],
                                   dsts[i]) == 0);
        } else {
            REQUIRE_CTX(results[i].bytes_written == 0, "reject bw: %s", ctx);
            REQUIRE_CTX(results[i].reserved == 0, "reject reserved: %s", ctx);
            if (snappy_oracle_accepts[i]) {
                snappy_stricter++;
                std::fprintf(stderr, "stricter than snappy: %s\n", ctx);
            } else {
                snappy_rejected++;
            }
        }
    }
    REQUIRE(snappy_rejected > 0);
    REQUIRE(snappy_stricter == 0);

    /* Batch 6: the whole vendored corpus, compressed by the oracle and
     * decoded on the device, at exact capacity. The generated fixtures
     * above are built against the format's own boundaries and are small;
     * these are real inputs of a size where a page of literals, the 64-byte
     * copy cap and the compressor's 65536-byte block split all occur many
     * times over, which is the only place a lane-fan-out defect that
     * survives a short stream would show. Named one by one rather than
     * globbed: a directory that quietly lost a file would otherwise shrink
     * this batch and still pass. The three baddata*.snappy entries are not
     * here - they are already decoded streams and are the negative corpus
     * of tests/snappy_upstream_negatives.cpp. */
    static const char* const kTestdata[] = {
        "alice29.txt", "asyoulik.txt",   "fireworks.jpeg", "geo.protodata",
        "html",        "html_x_4",       "kppkn.gtb",      "lcet10.txt",
        "paper-100k.pdf", "plrabn12.txt", "urls.10K"};
    constexpr size_t kTestdataCount = sizeof(kTestdata) / sizeof(*kTestdata);
    std::vector<std::vector<unsigned char>> corpus_original(kTestdataCount);
    std::vector<std::vector<unsigned char>> corpus_compressed(kTestdataCount);
    std::vector<std::string> corpus_names(kTestdataCount);
    size_t corpus_bytes = 0;
    for (size_t i = 0; i < kTestdataCount; i++) {
        corpus_original[i] = ReadTestdata(kTestdata[i]);
        REQUIRE_CTX(!corpus_original[i].empty(), "testdata %s unreadable",
                    kTestdata[i]);
        corpus_compressed[i] = SnappyCompressBlock(corpus_original[i]);
        corpus_names[i] = std::string("testdata/") + kTestdata[i];
        corpus_bytes += corpus_original[i].size();
    }
    std::vector<Chunk> corpus_chunks;
    for (size_t i = 0; i < kTestdataCount; i++) {
        corpus_chunks.push_back(Chunk{corpus_names[i], &corpus_compressed[i],
                                      corpus_original[i].size()});
    }
    REQUIRE(RunBatch(corpus_chunks, cudec_snappy_decompress_batch, &results,
                     &dsts) == 0);
    for (size_t i = 0; i < kTestdataCount; i++) {
        REQUIRE(CheckDecodedOk(corpus_names[i].c_str(), results[i],
                               corpus_original[i], dsts[i]) == 0);
    }

    /* Batch 7: the hand-built hostile corpus, through the public entry and
     * a real warp (issue #154). tests/snappy_block_device.cu parses the same
     * bytes with one thread on both compilers; this is the arm that runs
     * them under the copy engine, where 32 lanes fan out over an overlapping
     * gather and a stream that behaves serially can still misbehave.
     *
     * The oracle decides every verdict, in both directions, with the one
     * documented exception counted rather than exempted: cudec refuses the
     * four-byte literal length that wraps to zero in the reference's 32-bit
     * addition, and that stream is the whole of the stricter set. Pinning it
     * by name rather than by count is what keeps the exception from growing
     * quietly. */
    const auto hostile = MakeAdversarialSnappyBlocks();
    REQUIRE(!hostile.empty());
    std::vector<bool> hostile_accepts;
    std::vector<std::vector<unsigned char>> hostile_outputs;
    std::vector<Chunk> hostile_chunks;
    for (const auto& b : hostile) {
        std::vector<unsigned char> decoded;
        const bool ok = SnappyOracleDecodes(b.stream, &decoded);
        hostile_accepts.push_back(ok);
        hostile_outputs.push_back(ok ? decoded
                                     : std::vector<unsigned char>{});
    }
    for (const auto& b : hostile) {
        hostile_chunks.push_back(Chunk{b.name, &b.stream, b.dst_capacity});
    }
    REQUIRE(RunBatch(hostile_chunks, cudec_snappy_decompress_batch, &results,
                     &dsts) == 0);
    size_t hostile_accepted = 0;
    size_t hostile_rejected = 0;
    std::vector<std::string> hostile_stricter;
    for (size_t i = 0; i < hostile_chunks.size(); i++) {
        const char* ctx = hostile_chunks[i].context.c_str();
        if (results[i].status == CUDEC_OK) {
            REQUIRE_CTX(hostile_accepts[i],
                        "cudec accepts, snappy rejects: %s", ctx);
            REQUIRE(CheckDecodedOk(ctx, results[i], hostile_outputs[i],
                                   dsts[i]) == 0);
            hostile_accepted++;
        } else {
            /* A capacity refusal and a stream refusal are different
             * statuses, and both are documented: only the declared length
             * against the caller's capacity reports OUTPUT_TOO_SMALL. */
            REQUIRE_CTX(results[i].status == CUDEC_ERR_CORRUPT_INPUT ||
                            results[i].status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                        "%s: undefined reject status %d", ctx,
                        static_cast<int>(results[i].status));
            REQUIRE_CTX(results[i].bytes_written == 0, "reject bw: %s", ctx);
            REQUIRE_CTX(results[i].reserved == 0, "reject reserved: %s", ctx);
            if (hostile_accepts[i]) {
                hostile_stricter.push_back(hostile[i].name);
            } else {
                hostile_rejected++;
            }
        }
    }
    REQUIRE(hostile_accepted > 0);
    REQUIRE(hostile_rejected > 0);
    REQUIRE(hostile_stricter.size() == 1);
    REQUIRE(hostile_stricter[0] == "literal-length-wraps-to-zero");
    /* The capacity adversarials are the reason this corpus carries its own
     * capacities, so their outcome is pinned rather than merely counted among
     * the rejects. Two things are asserted and they are different: the status
     * is the capacity refusal, which is the ONE place a declared length maps
     * to OUTPUT_TOO_SMALL, and the destination is still whole - a 4 GiB claim
     * is refused at the preamble, before an element is parsed and before a
     * byte is written.
     *
     * That second half is not the general reject contract and is not claimed
     * as one. A stream refused mid-decode has already executed the elements
     * before the refusal, and the header says the destination is then
     * unspecified; what it never is, on either path, is presented as a valid
     * decode, which bytes_written == 0 above is. */
    size_t capacity_refusals = 0;
    for (size_t i = 0; i < hostile_chunks.size(); i++) {
        if (hostile[i].name.rfind("declared-", 0) != 0 ||
            hostile[i].name == "declared-exactly-capacity") {
            continue;
        }
        const char* name = hostile[i].name.c_str();
        REQUIRE_CTX(results[i].status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                    "%s: expected the capacity refusal, got %d", name,
                    static_cast<int>(results[i].status));
        for (size_t j = 0; j < dsts[i].size(); j++) {
            REQUIRE_CTX(dsts[i][j] == kDstPoison, "%s: wrote at %zu", name, j);
        }
        capacity_refusals++;
    }
    REQUIRE(capacity_refusals > 0);

    std::printf("PASS: %zu LZ4 pairs decoded byte-exact + %zu mutants in "
                "oracle parity (%zu oracle-rejected, %zu offset==0 stricter) + "
                "40000 grid-stride wraparound chunks + %zu Snappy pairs + %zu "
                "Snappy mutants in oracle parity (%zu oracle-rejected, %zu "
                "stricter) + %zu vendored testdata files (%zu bytes) decoded "
                "byte-exact on the GPU decoder; failure contract and poison "
                "beyond bytes_written intact; %zu hand-built hostile Snappy "
                "streams in oracle parity (%zu accepted, %zu rejected, one "
                "documented stricter)\n",
                pairs.size(), mutant_chunks.size(), rejected_count,
                stricter_count, snappy_pairs.size(),
                snappy_mutant_chunks.size(), snappy_rejected, snappy_stricter,
                kTestdataCount, corpus_bytes, hostile_chunks.size(),
                hostile_accepted, hostile_rejected);
    return 0;
}
