/* The reusable streaming-context decode (cudec_lz4_decompress_stream_ctx) held
 * to the substitute-for-compute-sanitizer gate. The named conformance property
 * this locks: the SAME input decoded on a REUSED context - after any number of
 * prior decodes, including one that GREW the staging - is BIT-IDENTICAL to a
 * fresh-context decode, in both memory spaces, and equal to the CPU oracle's
 * decode (correctness). Plus: the grow path decodes correctly (a larger batch
 * after a smaller one reallocs and round-trips); the fail-closed ladder (a
 * corrupt chunk mid-batch reports its error while neighbours decode OK and the
 * aggregate carries it; an undersized dst[k] fails only that chunk; zero chunks
 * / a NULL dst / a NULL ctx reject synchronously) leaves the context USABLE
 * (argument and per-chunk rejects do not poison); and the poison path - a
 * decode forced to a DEFINED CUDA fault poisons the context, the next decode
 * returns CUDEC_ERR_CUDA, and destroy still frees. #29. */
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

/* Runs one batch through the streaming context entry for a given memory space
 * on the PROVIDED context (fresh or reused). Returns the per-chunk results, the
 * host-visible destination bytes (downloaded for device space), and the
 * aggregate status. Device allocations for the caller's device destinations are
 * reclaimed at process exit (the gpu label is RUN_SERIAL and no state is kept
 * across tests) - the same convention as gpu_fixture. */
int RunStreamCtx(cudec_stream_ctx* ctx, const std::vector<SChunk>& chunks,
                 cudec_mem_space space,
                 std::vector<cudec_chunk_result>* results,
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
    *aggregate = cudec_lz4_decompress_stream_ctx(
        ctx, h_src.data(), h_ssz.data(), h_dst.data(), h_cap.data(), n, space,
        results->data());

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

/* Decodes `chunks` on a FRESH context (create, one decode, destroy). */
int RunFreshCtx(const std::vector<SChunk>& chunks, cudec_mem_space space,
                std::vector<cudec_chunk_result>* results,
                std::vector<std::vector<unsigned char>>* dst_bytes,
                cudec_status* aggregate) {
    cudec_stream_ctx* ctx = nullptr;
    REQUIRE(cudec_stream_ctx_create(&ctx) == CUDEC_OK);
    REQUIRE(ctx != nullptr);
    const int rc =
        RunStreamCtx(ctx, chunks, space, results, dst_bytes, aggregate);
    cudec_stream_ctx_destroy(ctx);
    return rc;
}

}  // namespace

int main() {
    const auto fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());

    /* A batch spanning several 64-chunk waves. */
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

    /* A single-chunk batch and a LARGER batch (more waves), for the grow-then-
     * reuse history: decoding the small one, then the large one (which grows
     * the staging), then `batch` again (which reuses the grown staging). */
    std::vector<SChunk> tiny = {
        SChunk{&fixtures[0].compressed, fixtures[0].original.size() + 8}};
    std::vector<SChunk> big;
    std::vector<const std::vector<unsigned char>*> big_expect;
    for (int rep = 0; rep < 60; rep++) {
        for (const auto& f : fixtures) {
            big.push_back(SChunk{&f.compressed, f.original.size() + 8});
            big_expect.push_back(&f.original);
        }
    }
    REQUIRE(big.size() > n); /* the grow is a real increase over `batch` */

    /* Reference: a fresh context, device space. Every other decode of `batch`
     * (fresh or reused, either space) must match its decoded bytes exactly. */
    std::vector<cudec_chunk_result> ref_res;
    std::vector<std::vector<unsigned char>> ref_dst;
    cudec_status ref_agg;
    REQUIRE(RunFreshCtx(batch, CUDEC_MEM_DEVICE, &ref_res, &ref_dst,
                        &ref_agg) == 0);
    REQUIRE(ref_agg == CUDEC_OK);
    for (size_t i = 0; i < n; i++) {
        REQUIRE_CTX(ref_res[i].status == CUDEC_OK, "ref chunk %zu", i);
        REQUIRE_CTX(ref_res[i].bytes_written == expect[i]->size(),
                    "ref bw %zu", i);
        REQUIRE_CTX(equal_bytes(ref_dst[i].data(), expect[i]->data(),
                                expect[i]->size()),
                    "ref bytes %zu", i);
        /* The kernel writes exactly bytes_written and leaves the poison beyond
         * it untouched. */
        for (size_t j = expect[i]->size(); j < ref_dst[i].size(); j++) {
            REQUIRE_CTX(ref_dst[i][j] == kPoison, "ref poison %zu@%zu", i, j);
        }
    }

    /* The named conformance property: a REUSED context - after a prior decode
     * AND a grow - decodes `batch` bit-identically to the fresh reference, in
     * BOTH spaces. Comparing the WHOLE caller buffer (decoded prefix + the
     * untouched poison tail) also locks the no-clobber property. The reused
     * context first decodes `tiny`, then `big` (which grows the staging past
     * what `batch` needs), then `batch`. */
    const cudec_mem_space spaces[] = {CUDEC_MEM_DEVICE, CUDEC_MEM_HOST};
    for (cudec_mem_space space : spaces) {
        /* Fresh-context decode of `batch` in this space - must already match
         * the device reference (same input, same untouched tail). */
        std::vector<cudec_chunk_result> fresh_res;
        std::vector<std::vector<unsigned char>> fresh_dst;
        cudec_status fresh_agg;
        REQUIRE(RunFreshCtx(batch, space, &fresh_res, &fresh_dst, &fresh_agg) ==
                0);
        REQUIRE_CTX(fresh_agg == CUDEC_OK, "fresh agg space=%d",
                    static_cast<int>(space));

        /* Reused context: tiny -> big (grow) -> batch. */
        cudec_stream_ctx* ctx = nullptr;
        REQUIRE(cudec_stream_ctx_create(&ctx) == CUDEC_OK);
        REQUIRE(ctx != nullptr);

        std::vector<cudec_chunk_result> res;
        std::vector<std::vector<unsigned char>> dst;
        cudec_status agg;

        REQUIRE(RunStreamCtx(ctx, tiny, space, &res, &dst, &agg) == 0);
        REQUIRE_CTX(agg == CUDEC_OK, "tiny agg space=%d",
                    static_cast<int>(space));

        /* The grow: a larger batch reallocs the staging and must still decode
         * every chunk correctly (oracle parity on the grown decode). */
        REQUIRE(RunStreamCtx(ctx, big, space, &res, &dst, &agg) == 0);
        REQUIRE_CTX(agg == CUDEC_OK, "big agg space=%d",
                    static_cast<int>(space));
        for (size_t i = 0; i < big.size(); i++) {
            REQUIRE_CTX(res[i].status == CUDEC_OK, "big ok space=%d c=%zu",
                        static_cast<int>(space), i);
            REQUIRE_CTX(res[i].bytes_written == big_expect[i]->size(),
                        "big bw space=%d c=%zu", static_cast<int>(space), i);
            REQUIRE_CTX(equal_bytes(dst[i].data(), big_expect[i]->data(),
                                    big_expect[i]->size()),
                        "big bytes space=%d c=%zu", static_cast<int>(space), i);
        }

        /* Same input on the reused, grown context: bit-identical to fresh. */
        REQUIRE(RunStreamCtx(ctx, batch, space, &res, &dst, &agg) == 0);
        REQUIRE_CTX(agg == CUDEC_OK, "reused agg space=%d",
                    static_cast<int>(space));
        cudec_stream_ctx_destroy(ctx);

        for (size_t i = 0; i < n; i++) {
            REQUIRE_CTX(res[i].status == CUDEC_OK, "reuse ok space=%d c=%zu",
                        static_cast<int>(space), i);
            REQUIRE_CTX(res[i].bytes_written == ref_res[i].bytes_written,
                        "reuse bw space=%d c=%zu", static_cast<int>(space), i);
            REQUIRE_CTX(dst[i].size() == ref_dst[i].size(),
                        "reuse cap space=%d c=%zu", static_cast<int>(space), i);
            /* Reused == device reference (determinism across reuse + grow), and
             * fresh == reference too (so fresh == reused transitively). */
            REQUIRE_CTX(equal_bytes(dst[i].data(), ref_dst[i].data(),
                                    dst[i].size()),
                        "reuse determinism space=%d c=%zu",
                        static_cast<int>(space), i);
            REQUIRE_CTX(equal_bytes(fresh_dst[i].data(), ref_dst[i].data(),
                                    fresh_dst[i].size()),
                        "fresh determinism space=%d c=%zu",
                        static_cast<int>(space), i);
        }
    }

    /* Fail-closed ladder, on ONE reused context to also lock that neither an
     * argument reject nor a per-chunk reject poisons it (a clean decode after
     * each reject still succeeds). */

    /* The corrupt stream is the first oracle-rejected mutant (cudec rejects
     * everything the oracle rejects). */
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
        cudec_stream_ctx* ctx = nullptr;
        REQUIRE(cudec_stream_ctx_create(&ctx) == CUDEC_OK);
        REQUIRE(ctx != nullptr);

        std::vector<cudec_chunk_result> res;
        std::vector<std::vector<unsigned char>> dst;
        cudec_status agg;

        /* A corrupt chunk between two valid ones: it reports its defined error
         * with no output, the neighbours decode OK, and the aggregate carries
         * the corrupt chunk's status. */
        {
            std::vector<SChunk> b = {
                SChunk{&fixtures[0].compressed,
                       fixtures[0].original.size() + 8},
                SChunk{&corrupt, 1 << 16},
                SChunk{&fixtures[0].compressed,
                       fixtures[0].original.size() + 8}};
            REQUIRE(RunStreamCtx(ctx, b, space, &res, &dst, &agg) == 0);
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
        {
            std::vector<SChunk> b = {
                SChunk{&fixtures[0].compressed,
                       fixtures[0].original.size() + 8},
                SChunk{&fixtures[0].compressed,
                       fixtures[0].original.size() / 2}};
            REQUIRE(RunStreamCtx(ctx, b, space, &res, &dst, &agg) == 0);
            REQUIRE_CTX(res[0].status == CUDEC_OK, "space=%d",
                        static_cast<int>(space));
            REQUIRE_CTX(res[1].status == CUDEC_ERR_OUTPUT_TOO_SMALL, "space=%d",
                        static_cast<int>(space));
            REQUIRE_CTX(res[1].bytes_written == 0, "space=%d",
                        static_cast<int>(space));
            REQUIRE_CTX(agg == CUDEC_ERR_OUTPUT_TOO_SMALL, "space=%d",
                        static_cast<int>(space));
        }

        /* Neither reject poisoned the context: a clean batch still decodes. */
        {
            std::vector<SChunk> b = {SChunk{&fixtures[0].compressed,
                                            fixtures[0].original.size() + 8}};
            REQUIRE(RunStreamCtx(ctx, b, space, &res, &dst, &agg) == 0);
            REQUIRE_CTX(agg == CUDEC_OK, "post-reject reuse space=%d",
                        static_cast<int>(space));
            REQUIRE_CTX(res[0].status == CUDEC_OK, "post-reject space=%d",
                        static_cast<int>(space));
        }
        cudec_stream_ctx_destroy(ctx);
    }

    /* Synchronous argument rejects: NULL ctx, zero chunks, and a NULL dst with
     * a claimed capacity. A zero-chunk reject on a live context does not poison
     * it (a real decode after still succeeds). */
    {
        const void* s0 = fixtures[0].compressed.data();
        size_t sz0 = fixtures[0].compressed.size();
        void* d0 = nullptr;
        size_t cap0 = 100;
        cudec_chunk_result r0;
        REQUIRE(cudec_lz4_decompress_stream_ctx(nullptr, &s0, &sz0, &d0, &cap0,
                                                1, CUDEC_MEM_HOST, &r0) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* NULL ctx */

        cudec_stream_ctx* ctx = nullptr;
        REQUIRE(cudec_stream_ctx_create(&ctx) == CUDEC_OK);
        REQUIRE(ctx != nullptr);
        REQUIRE(cudec_lz4_decompress_stream_ctx(ctx, &s0, &sz0, &d0, &cap0, 0,
                                                CUDEC_MEM_HOST, &r0) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* zero chunks */
        REQUIRE(cudec_lz4_decompress_stream_ctx(ctx, &s0, &sz0, &d0, &cap0, 1,
                                                CUDEC_MEM_HOST, &r0) ==
                CUDEC_ERR_INVALID_ARGUMENT); /* NULL dst, claimed capacity */

        /* The rejects did not poison it. */
        std::vector<unsigned char> out(fixtures[0].original.size() + 8, 0);
        void* d1 = out.data();
        size_t cap1 = out.size();
        cudec_chunk_result r1;
        REQUIRE(cudec_lz4_decompress_stream_ctx(ctx, &s0, &sz0, &d1, &cap1, 1,
                                                CUDEC_MEM_HOST, &r1) ==
                CUDEC_OK);
        cudec_stream_ctx_destroy(ctx);
    }

    /* The one-shot wrapper (create->decode->destroy internally, no `streams`
     * parameter) decodes `batch` identically to the reference. */
    {
        const size_t m = batch.size();
        std::vector<const void*> h_src(m);
        std::vector<size_t> h_ssz(m);
        std::vector<void*> h_dst(m);
        std::vector<size_t> h_cap(m);
        std::vector<std::vector<unsigned char>> host_dst(m);
        for (size_t i = 0; i < m; i++) {
            h_src[i] = batch[i].src->data();
            h_ssz[i] = batch[i].src->size();
            h_cap[i] = batch[i].cap;
            host_dst[i].assign(batch[i].cap, kPoison);
            h_dst[i] = host_dst[i].data();
        }
        std::vector<cudec_chunk_result> res(m);
        REQUIRE(cudec_lz4_decompress_stream(h_src.data(), h_ssz.data(),
                                            h_dst.data(), h_cap.data(), m,
                                            CUDEC_MEM_HOST,
                                            res.data()) == CUDEC_OK);
        for (size_t i = 0; i < m; i++) {
            REQUIRE_CTX(res[i].status == CUDEC_OK, "oneshot ok %zu", i);
            REQUIRE_CTX(equal_bytes(host_dst[i].data(), ref_dst[i].data(),
                                    host_dst[i].size()),
                        "oneshot determinism %zu", i);
        }
    }

    /* Poison path: a decode forced to a DEFINED CUDA fault poisons the context.
     * An impossibly large device staging (an oversized dst capacity for the
     * host-output path) fails its cudaMalloc with a defined error BEFORE any
     * staging copy runs - the grow precedes the wave loop, so the huge capacity
     * is never dereferenced (no undefined behavior). The next decode returns
     * CUDEC_ERR_CUDA; destroy still frees. */
    {
        cudec_stream_ctx* ctx = nullptr;
        REQUIRE(cudec_stream_ctx_create(&ctx) == CUDEC_OK);
        REQUIRE(ctx != nullptr);

        const void* src = fixtures[0].compressed.data();
        size_t ssz = fixtures[0].compressed.size();
        unsigned char sink = 0; /* a real, tiny host dst; never written */
        void* dst = &sink;
        size_t huge_cap = static_cast<size_t>(1) << 60; /* impossible alloc */
        /* Pre-fill with an OK-looking record: the grow fault happens BEFORE the
         * wave loop, so nothing reads this chunk back. The contract still
         * requires the per-chunk channel to be left DEFINED non-OK, so this
         * OK-looking value must be overwritten - if it survives, the fail-closed
         * stamp is missing. */
        cudec_chunk_result r;
        r.status = CUDEC_OK;
        r.reserved = 0;
        r.bytes_written = 123456;
        REQUIRE(cudec_lz4_decompress_stream_ctx(ctx, &src, &ssz, &dst,
                                                &huge_cap, 1, CUDEC_MEM_HOST,
                                                &r) == CUDEC_ERR_CUDA);
        /* Fail-closed per-chunk on the pre-loop grow-fault path: a defined non-OK
         * status and no claimed output, never the stale OK-looking pre-fill. */
        REQUIRE_CTX(r.status == CUDEC_ERR_CUDA, "poison chunk status %d",
                    r.status);
        REQUIRE_CTX(r.bytes_written == 0, "poison chunk bytes_written %llu",
                    static_cast<unsigned long long>(r.bytes_written));

        /* Poisoned: a perfectly valid batch now returns CUDEC_ERR_CUDA. The
         * poisoned re-entry is a post-validation non-OK return, so it too must
         * leave the per-chunk channel a DEFINED non-OK status - never the stale
         * OK-looking pre-fill. */
        std::vector<unsigned char> out(fixtures[0].original.size() + 8, 0);
        void* d2 = out.data();
        size_t cap2 = out.size();
        const void* s2 = fixtures[0].compressed.data();
        size_t sz2 = fixtures[0].compressed.size();
        cudec_chunk_result r2;
        r2.status = CUDEC_OK;
        r2.reserved = 0;
        r2.bytes_written = 123456;
        REQUIRE(cudec_lz4_decompress_stream_ctx(ctx, &s2, &sz2, &d2, &cap2, 1,
                                                CUDEC_MEM_HOST, &r2) ==
                CUDEC_ERR_CUDA);
        REQUIRE_CTX(r2.status == CUDEC_ERR_CUDA, "poisoned re-entry status %d",
                    r2.status);
        REQUIRE_CTX(r2.bytes_written == 0, "poisoned re-entry bytes_written %llu",
                    static_cast<unsigned long long>(r2.bytes_written));
        cudec_stream_ctx_destroy(ctx); /* still frees, no crash */
    }

    std::printf("PASS: reused-context decode bit-identical to fresh across "
                "{host,device} and a grow, over %zu chunks; grow, corrupt "
                "isolation, undersized, zero-chunk, null-dst, null-ctx "
                "fail-closed without poisoning; one-shot wrapper matches; "
                "defined CUDA fault poisons and destroy still frees\n",
                n);
    return 0;
}
