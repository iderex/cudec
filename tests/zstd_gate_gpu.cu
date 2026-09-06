/* The standing device gate set for the Zstd path (issue #399). The kernel that
 * landed under #203 is held here to the three properties rung 3 of
 * docs/MASTERPLAN.md section 14.5 names, on the device rather than on the
 * twin:
 *
 *   DETERMINISM. One corpus, one reference decode, then the same batch across
 *   several launch geometries and a three-stream split, several runs each,
 *   with the whole destination arena re-poisoned before every run and compared
 *   in full. Which block decodes which frame, and in what order, changes in
 *   every direction the kernel admits, and not one output byte may notice.
 *
 *   TWO-DIRECTIONAL MUTANT REJECT PARITY. The #187 mutation layer over real
 *   frames, with the pinned libzstd as the authority. Where the kernel
 *   accepts, the reference accepted and the bytes agree; where the reference
 *   rejects, the kernel rejects. Over-strictness - the kernel calling a frame
 *   malformed that the reference decodes - is counted and required to be zero
 *   rather than exempted.
 *
 *   CAPACITY AND WINDOW. Every output bound comes from the caller's
 *   dst_capacity and from the frame's own declaration, and from nothing else,
 *   driven at and either side of both.
 *
 * THE GEOMETRY AXIS IS NARROWER HERE THAN FOR THE CHUNK DECODER, AND THAT IS
 * THE KERNEL RATHER THAN A GAP IN THIS FILE. A frame is decoded by a BLOCK
 * whose width the shared table set fixes at 128 threads (section 14.2), and
 * the kernel refuses any other width outright rather than decoding at it. So
 * what is free to move is the grid and the split across streams, and a block
 * width in the list below would be testing the refusal instead of the
 * mapping. The refusal itself is covered, once, in its own section.
 *
 * A DECLINE IS NOT A REFUSAL, AND THE PARITY SECTION TURNS ON THAT. The
 * accepted envelope of section 12.2 is narrower than libzstd's - no
 * dictionary, a content size required, a window ceiling - so a mutant that
 * moves a frame out of the envelope is answered CUDEC_ERR_UNSUPPORTED while
 * the reference decodes it happily. That is the subset working, not
 * over-strictness, and only CUDEC_ERR_CORRUPT_INPUT against an accepting
 * reference is the thing this file requires to be zero.
 *
 * TERMINATION IS NOT A SECTION, IT IS HOW EVERY SECTION RUNS. A block that
 * never leaves its block loop does not report a bad status, it holds the
 * launch and the stream behind it, so a plain stream synchronise would inherit
 * the hang and stall the suite. Every decode below is fenced with an event and
 * polled against a wall-clock deadline; an expiry FAILS with a message.
 *
 * WHAT THIS IS NOT. It is not the oracle diff over the whole corpus, which is
 * #203's and is in tests/zstd_device.cu, and it is not that file's
 * reject-parity rung, which holds the kernel's own walk to the host twin's
 * status classes. Nothing here is timed, and the compute-sanitizer half of the
 * standing gate is parked and unproducible on this host (#127, #258), so what
 * stands in its place is the recorded substitute: this file plus the
 * whole-corpus diff. */
#include "chunk_decode.cuh"
#include "cudec.h"
#include "require.h"
#include "zstd_corpus.h"
#include "zstd_decode.cuh"
#include "zstd_host_frame.h"

#include "vendor_rt_test.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

constexpr unsigned char kDstPoison = 0xA5;
constexpr int kRunsPerGeometry = 3;
/* Generous against a loaded GPU, tiny against "never": every frame here is a
 * bounded-block frame of at most a few hundred kilobytes. */
constexpr double kDeadlineSeconds = 30.0;
/* Slack past the declared size, so a write past bytes_written has poison to
 * disturb. */
constexpr size_t kCapacitySlack = 64;

/* ---- the batch, staged as two device arenas ---- */

struct Chunk {
    std::string name;
    Bytes src;
    size_t dst_capacity;
};

/* One source blob and one destination arena, so a whole launch's byte image is
 * a single download and the determinism compare covers every byte rather than
 * a sample of them. */
struct Batch {
    size_t n = 0;
    std::vector<std::string> names;
    std::vector<size_t> dst_offsets;
    std::vector<size_t> caps;
    unsigned char* d_src_blob = nullptr;
    unsigned char* d_dst_arena = nullptr;
    size_t dst_arena_size = 0;
    const void** d_srcs = nullptr;
    void** d_dsts = nullptr;
    size_t* d_sizes = nullptr;
    size_t* d_caps = nullptr;
    cudec_chunk_result* d_results = nullptr;
};

int BuildBatch(const std::vector<Chunk>& chunks, Batch* b) {
    const size_t n = chunks.size();
    b->n = n;
    b->names.clear();
    b->dst_offsets.clear();
    b->caps.clear();

    size_t src_total = 0;
    size_t dst_total = 0;
    for (size_t i = 0; i < n; i++) {
        src_total += chunks[i].src.size();
        /* Every destination gets at least one byte of arena so a capacity of
         * zero still has a poison region a write past it would disturb. */
        dst_total += chunks[i].dst_capacity ? chunks[i].dst_capacity : 1u;
    }
    REQUIRE_RT(
        cudec_rt::device_malloc(&b->d_src_blob, src_total ? src_total : 1));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_dst_arena, dst_total));
    b->dst_arena_size = dst_total;

    std::vector<const void*> h_srcs(n);
    std::vector<void*> h_dsts(n);
    std::vector<size_t> h_sizes(n);
    std::vector<size_t> h_caps(n);
    size_t src_off = 0;
    size_t dst_off = 0;
    for (size_t i = 0; i < n; i++) {
        if (!chunks[i].src.empty()) {
            REQUIRE_RT(cudec_rt::memcpy(b->d_src_blob + src_off,
                                        chunks[i].src.data(),
                                        chunks[i].src.size(),
                                        cudec_rt::memcpy_h2d));
        }
        h_srcs[i] = b->d_src_blob + src_off;
        h_sizes[i] = chunks[i].src.size();
        h_dsts[i] = b->d_dst_arena + dst_off;
        h_caps[i] = chunks[i].dst_capacity;
        b->names.push_back(chunks[i].name);
        b->dst_offsets.push_back(dst_off);
        b->caps.push_back(chunks[i].dst_capacity);
        src_off += chunks[i].src.size();
        dst_off += chunks[i].dst_capacity ? chunks[i].dst_capacity : 1u;
    }
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_srcs, n * sizeof(*b->d_srcs)));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_dsts, n * sizeof(*b->d_dsts)));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_sizes, n * sizeof(*b->d_sizes)));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_caps, n * sizeof(*b->d_caps)));
    REQUIRE_RT(
        cudec_rt::device_malloc(&b->d_results, n * sizeof(*b->d_results)));
    REQUIRE_RT(cudec_rt::memcpy(b->d_srcs, h_srcs.data(),
                                n * sizeof(*b->d_srcs), cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b->d_dsts, h_dsts.data(),
                                n * sizeof(*b->d_dsts), cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b->d_sizes, h_sizes.data(),
                                n * sizeof(*b->d_sizes), cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b->d_caps, h_caps.data(),
                                n * sizeof(*b->d_caps), cudec_rt::memcpy_h2d));
    return 0;
}

void FreeBatch(Batch* b) {
    (void)cudec_rt::device_free(b->d_src_blob);
    (void)cudec_rt::device_free(b->d_dst_arena);
    (void)cudec_rt::device_free(b->d_srcs);
    (void)cudec_rt::device_free(b->d_dsts);
    (void)cudec_rt::device_free(b->d_sizes);
    (void)cudec_rt::device_free(b->d_caps);
    (void)cudec_rt::device_free(b->d_results);
}

int ResetDeviceState(const Batch& b) {
    REQUIRE_RT(
        cudec_rt::device_memset(b.d_dst_arena, kDstPoison, b.dst_arena_size));
    /* 0xFF is outside the status enumeration, so a frame the kernel never
     * reached cannot read as a decoded one. */
    REQUIRE_RT(cudec_rt::device_memset(b.d_results, 0xFF,
                                       b.n * sizeof(*b.d_results)));
    return 0;
}

int Download(const Batch& b, Bytes* dst,
             std::vector<cudec_chunk_result>* results) {
    dst->assign(b.dst_arena_size, 0);
    results->assign(b.n, cudec_chunk_result{});
    REQUIRE_RT(cudec_rt::memcpy(dst->data(), b.d_dst_arena, b.dst_arena_size,
                                cudec_rt::memcpy_d2h));
    REQUIRE_RT(cudec_rt::memcpy(results->data(), b.d_results,
                                b.n * sizeof(*b.d_results),
                                cudec_rt::memcpy_d2h));
    return 0;
}

/* ---- the watchdog ---- */

/* Waits on an EVENT and not on the stream: a launch that does not finish must
 * be reported, never waited on. On expiry the launch is still resident, and
 * returning (and exiting) leaves the teardown to the driver - synchronising
 * or destroying the stream here would block on the very hang being reported. */
int WaitWithWatchdog(cudec_rt::stream_t stream, const char* what) {
    cudec_rt::event_t finished;
    REQUIRE_RT(cudec_rt::event_create_untimed(&finished));
    REQUIRE_RT(cudec_rt::event_record(finished, stream));
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        const cudec_rt::error_t query = cudec_rt::event_query(finished);
        if (query == cudec_rt::success) {
            break;
        }
        REQUIRE_CTX(query == cudec_rt::error_not_ready,
                    "%s: event query failed: %s", what,
                    cudec_rt::error_string(query));
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count();
        REQUIRE_CTX(elapsed < kDeadlineSeconds,
                    "%s did not complete within %.0f s - every block in it is "
                    "a fuel-bounded walk, so this is a non-terminating decode",
                    what, kDeadlineSeconds);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE_RT(cudec_rt::event_destroy(finished));
    return 0;
}

/* The shipped entry over the whole batch, fenced. */
int RunShippedEntry(const Batch& b, const char* what) {
    REQUIRE(ResetDeviceState(b) == 0);
    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    REQUIRE(cudec_zstd_decompress_batch(b.d_srcs, b.d_sizes, b.d_dsts, b.d_caps,
                                        b.n, b.d_results,
                                        cudec_rt::abi_stream(stream)) ==
            CUDEC_OK);
    REQUIRE(WaitWithWatchdog(stream, what) == 0);
    REQUIRE_RT(cudec_rt::stream_destroy(stream));
    return 0;
}

/* A grid the public ABI exposes no knob for is reached by launching the
 * internal kernel directly. The width is the kernel's own constant in every
 * entry, for the reason the file header gives. */
int RunGrid(const Batch& b, unsigned blocks, const char* what) {
    REQUIRE(ResetDeviceState(b) == 0);
    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    cudec_detail::zstd_decode_batch<cudec_detail::kCudaWaveSize>
        <<<blocks, cudec_detail::kZstdBlockThreads, 0, stream>>>(
            b.d_srcs, b.d_sizes, b.d_dsts, b.d_caps, b.n, b.d_results);
    REQUIRE_RT(cudec_rt::get_last_error());
    REQUIRE(WaitWithWatchdog(stream, what) == 0);
    REQUIRE_RT(cudec_rt::stream_destroy(stream));
    return 0;
}

/* The same batch as several sub-batches on concurrent streams: frame k is
 * decoded by a different block of a different launch than in any other
 * geometry, and the sub-batches complete in an order this test does not
 * control. Per-frame independence means the output must not notice. */
int RunSplitStreams(const Batch& b, unsigned stream_count, const char* what) {
    REQUIRE(ResetDeviceState(b) == 0);
    std::vector<cudec_rt::stream_t> streams(stream_count);
    for (unsigned s = 0; s < stream_count; s++) {
        REQUIRE_RT(cudec_rt::stream_create(&streams[s]));
    }
    const size_t per = (b.n + stream_count - 1) / stream_count;
    for (unsigned s = 0; s < stream_count; s++) {
        const size_t begin = s * per;
        if (begin >= b.n) {
            break;
        }
        const size_t count = (begin + per <= b.n) ? per : b.n - begin;
        /* cudec_chunk_result is 16 bytes, so an element offset keeps the
         * 16-byte alignment the ABI requires of d_results. */
        REQUIRE(cudec_zstd_decompress_batch(
                    b.d_srcs + begin, b.d_sizes + begin, b.d_dsts + begin,
                    b.d_caps + begin, count, b.d_results + begin,
                    cudec_rt::abi_stream(streams[s])) == CUDEC_OK);
    }
    for (unsigned s = 0; s < stream_count; s++) {
        REQUIRE(WaitWithWatchdog(streams[s], what) == 0);
        REQUIRE_RT(cudec_rt::stream_destroy(streams[s]));
    }
    return 0;
}

int CompareAgainstReference(const Batch& b, const std::string& context,
                            const Bytes& ref_dst,
                            const std::vector<cudec_chunk_result>& ref_results,
                            const Bytes& dst,
                            const std::vector<cudec_chunk_result>& results) {
    for (size_t i = 0; i < b.n; i++) {
        REQUIRE_CTX(results[i].status == ref_results[i].status,
                    "%s: frame %zu (%s) status %d, reference %d",
                    context.c_str(), i, b.names[i].c_str(),
                    static_cast<int>(results[i].status),
                    static_cast<int>(ref_results[i].status));
        REQUIRE_CTX(results[i].bytes_written == ref_results[i].bytes_written,
                    "%s: frame %zu (%s) wrote %llu bytes, reference %llu",
                    context.c_str(), i, b.names[i].c_str(),
                    static_cast<unsigned long long>(results[i].bytes_written),
                    static_cast<unsigned long long>(
                        ref_results[i].bytes_written));
        REQUIRE_CTX(results[i].reserved == ref_results[i].reserved,
                    "%s: frame %zu reserved word drifted", context.c_str(), i);
    }
    /* The whole arena, including the bytes a refused frame happened to write
     * before refusing. The contract does not promise those, so this asserts
     * more than the contract - deliberately, because a geometry-dependent
     * write on the reject path is exactly what the promised region would
     * hide. */
    REQUIRE_CTX(dst.size() == ref_dst.size(), "%s: arena size drifted",
                context.c_str());
    REQUIRE_CTX(equal_bytes(dst.data(), ref_dst.data(), dst.size()),
                "%s: destination arena differs from the reference decode",
                context.c_str());
    return 0;
}

/* ---- the corpus this file runs on ---- */

/* Real frames, plus the batch geometry the entry consumes. Both halves are the
 * #185 layer's, so the shapes this gate runs over are the shapes the corpus
 * self-proof already accounts for. */
std::vector<Chunk> BuildCorpusChunks(unsigned* out_decodable) {
    std::vector<Chunk> chunks;
    unsigned decodable = 0;
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    for (size_t i = 0; i < fixtures.size(); i++) {
        Bytes reference;
        const Bytes frame(fixtures[i].compressed.begin(),
                          fixtures[i].compressed.end());
        if (!ZstdOracleDecodes(frame, &reference)) {
            continue;
        }
        Chunk chunk;
        chunk.name = fixtures[i].name;
        chunk.src = frame;
        chunk.dst_capacity = reference.size() + kCapacitySlack;
        chunks.push_back(chunk);
        decodable++;
    }
    Bytes source;
    unsigned state = 0x2468ACEu;
    for (size_t i = 0; i < 256u * 1024u; i++) {
        state = state * 1103515245u + 12345u;
        const unsigned r = (state >> 16) & 0xFFFFu;
        source.push_back(static_cast<unsigned char>(
            (r % 6u == 0u) ? (r & 0xFFu) : ('a' + (r % 9u))));
    }
    const std::vector<Bytes> frames =
        MakeZstdBatchFrames(source, 48u * 1024u, 5);
    for (size_t i = 0; i < frames.size(); i++) {
        Bytes reference;
        if (!ZstdOracleDecodes(frames[i], &reference)) {
            continue;
        }
        Chunk chunk;
        chunk.name = "batch-" + std::to_string(i);
        chunk.src = frames[i];
        chunk.dst_capacity = reference.size() + kCapacitySlack;
        chunks.push_back(chunk);
        decodable++;
    }
    if (out_decodable != 0) {
        *out_decodable = decodable;
    }
    return chunks;
}

/* ---- section 1: determinism ---- */

int RunDeterminism() {
    unsigned decodable = 0;
    const std::vector<Chunk> chunks = BuildCorpusChunks(&decodable);
    REQUIRE(chunks.size() > 1);
    Batch batch;
    REQUIRE(BuildBatch(chunks, &batch) == 0);

    Bytes ref_dst;
    std::vector<cudec_chunk_result> ref_results;
    REQUIRE(RunShippedEntry(batch, "determinism reference") == 0);
    REQUIRE(Download(batch, &ref_dst, &ref_results) == 0);
    unsigned decoded = 0;
    unsigned declined = 0;
    for (size_t i = 0; i < batch.n; i++) {
        if (ref_results[i].status == CUDEC_ERR_UNSUPPORTED) {
            /* A fixture the reference decodes and the subset of section 12.2
             * declines. It stays in the batch rather than being filtered out:
             * a declined frame's answer has to be as stable across geometries
             * as a decoded one's, and it is a neighbour a decoded frame must
             * survive sharing a launch with. */
            declined++;
            continue;
        }
        REQUIRE_CTX(ref_results[i].status == CUDEC_OK,
                    "%s: the reference decode REFUSED a frame the oracle "
                    "decodes, status %d",
                    batch.names[i].c_str(),
                    static_cast<int>(ref_results[i].status));
        decoded++;
    }
    REQUIRE(decoded > 0);
    REQUIRE(decoded + declined == chunks.size());

    /* One block walking the whole batch through the grid-stride loop, a grid
     * that is neither a divisor nor a multiple of the frame count, a grid one
     * block per frame, a grid far larger than the batch, and the shipped
     * sizing. */
    unsigned grids[5];
    grids[0] = 1;
    grids[1] = 3;
    grids[2] = static_cast<unsigned>(batch.n);
    grids[3] = static_cast<unsigned>(batch.n) * 4u + 7u;
    grids[4] = cudec_detail::zstd_grid_blocks(batch.n);
    unsigned runs = 0;
    for (unsigned g = 0; g < 5; g++) {
        for (int run = 0; run < kRunsPerGeometry; run++) {
            char what[96];
            std::snprintf(what, sizeof(what), "grid %u run %d", grids[g], run);
            REQUIRE(RunGrid(batch, grids[g], what) == 0);
            Bytes dst;
            std::vector<cudec_chunk_result> results;
            REQUIRE(Download(batch, &dst, &results) == 0);
            REQUIRE(CompareAgainstReference(batch, what, ref_dst, ref_results,
                                            dst, results) == 0);
            runs++;
        }
    }
    for (int run = 0; run < kRunsPerGeometry; run++) {
        char what[96];
        std::snprintf(what, sizeof(what), "three streams run %d", run);
        REQUIRE(RunSplitStreams(batch, 3, what) == 0);
        Bytes dst;
        std::vector<cudec_chunk_result> results;
        REQUIRE(Download(batch, &dst, &results) == 0);
        REQUIRE(CompareAgainstReference(batch, what, ref_dst, ref_results, dst,
                                        results) == 0);
        runs++;
    }
    /* The shipped entry again at the end, so the reference itself is held to
     * repeating rather than being trusted as the one run nobody re-took. */
    REQUIRE(RunShippedEntry(batch, "determinism repeat") == 0);
    {
        Bytes dst;
        std::vector<cudec_chunk_result> results;
        REQUIRE(Download(batch, &dst, &results) == 0);
        REQUIRE(CompareAgainstReference(batch, "same batch twice", ref_dst,
                                        ref_results, dst, results) == 0);
        runs++;
    }
    FreeBatch(&batch);
    std::printf(
        "determinism: %zu frames, %u runs over five grids, a three-stream "
        "split and the shipped entry twice, every arena byte identical\n",
        chunks.size(), runs);
    return 0;
}

/* ---- section 2: the block width the kernel refuses ---- */

/* The one geometry the determinism section cannot vary, covered here instead
 * of being left unsaid: a block that is not the width the shared table set is
 * sized for is refused rather than decoded at, so the results the launch was
 * given keep the sentinel they were primed with. */
int RunWidthRefusal() {
    unsigned decodable = 0;
    std::vector<Chunk> chunks = BuildCorpusChunks(&decodable);
    chunks.resize(2);
    Batch batch;
    REQUIRE(BuildBatch(chunks, &batch) == 0);
    REQUIRE(ResetDeviceState(batch) == 0);
    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    cudec_detail::zstd_decode_batch<cudec_detail::kCudaWaveSize>
        <<<4, cudec_detail::kZstdBlockThreads / 2, 0, stream>>>(
            batch.d_srcs, batch.d_sizes, batch.d_dsts, batch.d_caps, batch.n,
            batch.d_results);
    REQUIRE_RT(cudec_rt::get_last_error());
    REQUIRE(WaitWithWatchdog(stream, "half-width launch") == 0);
    REQUIRE_RT(cudec_rt::stream_destroy(stream));
    Bytes dst;
    std::vector<cudec_chunk_result> results;
    REQUIRE(Download(batch, &dst, &results) == 0);
    for (size_t i = 0; i < batch.n; i++) {
        /* The sentinel ResetDeviceState primed the record with, which is
         * outside the status enumeration: a launch that wrote nothing leaves
         * it, and a launch that decoded would have replaced it. */
        REQUIRE_CTX(results[i].status == -1,
                    "%s: a half-width launch wrote a result, status %d",
                    batch.names[i].c_str(),
                    static_cast<int>(results[i].status));
    }
    for (size_t i = 0; i < dst.size(); i++) {
        REQUIRE_CTX(dst[i] == kDstPoison,
                    "a half-width launch wrote byte %zu of the arena", i);
    }
    FreeBatch(&batch);
    std::printf(
        "width: a launch at half the block width decodes nothing and writes "
        "nothing\n");
    return 0;
}

/* ---- the declared departures ---- */

/* A stream the pinned reference decodes and this decoder refuses.
 *
 * WHY THE LIST EXISTS AND WHY IT IS DANGEROUS. Over-strictness is a divergence
 * in exactly the way over-permissiveness is, and a gate where a failing case
 * can be waved through by adding a line would be worth nothing. So an entry
 * here is not an exemption: the parity section REQUIRES the departure to still
 * be real - the reference decodes the frame, and the host twin, which runs the
 * same format units the kernel compiles, refuses it at the exact stage and
 * rung named below. A departure written to silence a failure the twin does not
 * have fails the first check, one whose refusal has moved fails the second,
 * and one that has been repaired fails the third, which requires every
 * declared entry to be SEEN.
 *
 * An entry also carries the issue that holds its repair. A departure nobody is
 * holding is a permanent exemption wearing a temporary name. */
/* WHAT SEPARATES A REFERENCE THAT WAS RIGHT FROM ONE THAT WAS ONLY WILLING
 * (issue #461).
 *
 * The rule this file applies is mechanical - a refusal against an accepting
 * reference - and mechanical is the point, because a rule that asked which
 * answer was better would be arguing with its own evidence. But an accepting
 * reference is not by itself a legal frame: a mutation can leave a stream the
 * reference reads to the end and gets wrong, and on a fixture with no content
 * checksum nothing downstream has anything to check that against.
 *
 * So the classifier asks one more question before it calls a refusal
 * over-strict: did the reference produce the bytes the UNMUTATED fixture
 * produces? Where it did, the mutation was inert and a refusal is a departure
 * to explain. Where it did not, the reference decoded a corrupt frame into
 * wrong bytes without noticing, and this decoder refusing is the fail-closed
 * contract rather than strictness - that row is counted as
 * reference-permissive and named on the run's own output.
 *
 * THE DECLARED-DEPARTURE REGISTER IS GONE WITH IT AND THAT IS THE REPAIR
 * RATHER THAN A LOOSENING. Its one entry was this class, declared per name
 * because there was nothing that could tell the two apart; the comparison
 * above tells them apart from the bytes, so a name-list would now be a second
 * way of saying the same thing that could disagree with it. What the register
 * used to lock - that the twin refuses too - is asserted on every
 * reference-permissive row below rather than on the one that was written down,
 * and the instance itself is held at the unit by the jump-table case in
 * tests/zstd_literals_twin.cpp. An undeclared over-strict row still reds. */

/* ---- section 3: two-directional mutant reject parity ---- */

struct ParityCounts {
    unsigned mutants;
    unsigned both_accept;
    unsigned both_reject;
    unsigned declined;
    unsigned overstrict;
    unsigned reference_permissive;
};

int RunMutantParity(ParityCounts* counts) {
    std::memset(counts, 0, sizeof(*counts));
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());

    std::vector<Chunk> chunks;
    std::vector<Bytes> oracle_out;
    std::vector<bool> oracle_ok;
    std::vector<bool> oracle_matches_base;
    for (size_t i = 0; i < fixtures.size(); i++) {
        const Bytes frame(fixtures[i].compressed.begin(),
                          fixtures[i].compressed.end());
        Bytes plain;
        if (!ZstdOracleDecodes(frame, &plain)) {
            continue;
        }
        const std::vector<ZstdMutant> mutants =
            MutateZstdFrame(frame, 0x5EEDu + i);
        for (size_t m = 0; m < mutants.size(); m++) {
            Bytes decoded;
            const bool ok = ZstdOracleDecodes(mutants[m].frame, &decoded);
            /* Whether the reference's answer for this mutant is the frame's
             * own bytes. Taken against `plain`, the unmutated fixture's
             * decode, which this loop already has. */
            oracle_matches_base.push_back(ok && decoded == plain);
            Chunk chunk;
            chunk.name = fixtures[i].name + " / " + mutants[m].description;
            chunk.src = mutants[m].frame;
            /* The capacity is the ORIGINAL frame's output plus slack, not the
             * mutant's: a mutant that declares a larger content size must be
             * refused for what it says rather than accommodated. */
            chunk.dst_capacity = plain.size() + kCapacitySlack;
            chunks.push_back(chunk);
            oracle_out.push_back(decoded);
            oracle_ok.push_back(ok);
        }
    }
    REQUIRE(chunks.size() > 100);

    /* In slices, so one launch never carries every mutant's destination at
     * once and a failure names a bounded batch. */
    const size_t kSlice = 256;
    for (size_t begin = 0; begin < chunks.size(); begin += kSlice) {
        const size_t end =
            (begin + kSlice < chunks.size()) ? begin + kSlice : chunks.size();
        const std::vector<Chunk> slice(chunks.begin() + begin,
                                       chunks.begin() + end);
        Batch batch;
        REQUIRE(BuildBatch(slice, &batch) == 0);
        REQUIRE(RunShippedEntry(batch, "mutant slice") == 0);
        Bytes dst;
        std::vector<cudec_chunk_result> results;
        REQUIRE(Download(batch, &dst, &results) == 0);
        for (size_t i = 0; i < batch.n; i++) {
            const size_t at = begin + i;
            counts->mutants++;
            const cudec_status status =
                static_cast<cudec_status>(results[i].status);
            if (status == CUDEC_OK) {
                REQUIRE_CTX(oracle_ok[at],
                            "%s: the kernel ACCEPTED a frame the reference "
                            "rejects",
                            batch.names[i].c_str());
                counts->both_accept++;
                REQUIRE_CTX(
                    results[i].bytes_written == oracle_out[at].size(),
                    "%s: accepted and wrote %llu bytes, the reference "
                    "produced %llu",
                    batch.names[i].c_str(),
                    static_cast<unsigned long long>(results[i].bytes_written),
                    static_cast<unsigned long long>(oracle_out[at].size()));
                REQUIRE_CTX(
                    equal_bytes(dst.data() + batch.dst_offsets[i],
                                oracle_out[at].data(), oracle_out[at].size()),
                    "%s: accepted and the bytes differ from the reference",
                    batch.names[i].c_str());
                continue;
            }
            REQUIRE_CTX(results[i].bytes_written == 0,
                        "%s: refused and still reported %llu bytes",
                        batch.names[i].c_str(),
                        static_cast<unsigned long long>(
                            results[i].bytes_written));
            if (status == CUDEC_ERR_UNSUPPORTED) {
                /* Outside the accepted envelope of section 12.2. The
                 * reference may decode it; declining is the subset working
                 * and is not over-strictness. */
                counts->declined++;
                continue;
            }
            if (!oracle_ok[at]) {
                counts->both_reject++;
                continue;
            }
            /* The host twin on the same bytes, so the verdict says WHERE the
             * strictness is: a twin that refuses too means the divergence is
             * in the format units both residencies share, and a twin that
             * accepts means it is this kernel's own. */
            cudec_test::HostBytes host_out;
            int stage = -1;
            int rung = -1;
            const cudec_status twin = cudec_test::HostDecodeFrame(
                slice[i].src, slice[i].dst_capacity, &host_out, &stage, &rung);
            if (!oracle_matches_base[at]) {
                /* The reference decoded a corrupt frame into bytes that are
                 * not the frame's own and did not notice. Refusing is the
                 * contract; the row is not over-strictness. The twin is still
                 * held to the same answer, because a refusal only this kernel
                 * makes is a different fault and must not hide here. */
                REQUIRE_CTX(twin != CUDEC_OK,
                            "%s: the reference decoded it to bytes the "
                            "unmutated fixture does not produce, this kernel "
                            "refused with status %d, and the host twin "
                            "DECODED it - so the refusal is this kernel's own "
                            "rather than the format units'",
                            batch.names[i].c_str(), static_cast<int>(status));
                counts->reference_permissive++;
                continue;
            }
            counts->overstrict++;
            REQUIRE_CTX(false,
                        "%s: the kernel REFUSED with status %d a frame the "
                        "reference decodes to the unmutated fixture's own "
                        "bytes, so the mutation was inert and this is a "
                        "departure to explain (host twin %d, stage %d, rung "
                        "%d)",
                        batch.names[i].c_str(), static_cast<int>(status),
                        static_cast<int>(twin), stage, rung);
        }
        FreeBatch(&batch);
    }
    REQUIRE(counts->overstrict == 0);
    REQUIRE(counts->both_reject > 0);
    REQUIRE(counts->both_accept > 0);
    /* The reference-permissive class is not required to be non-empty here.
     * The corpus deciding to stop producing such a mutant is not a failure of
     * this gate, and the instance #461 is about is held at the unit by
     * tests/zstd_literals_twin.cpp, which DOES red if it stops being reached.
     * The count is printed so a change in it is visible. */
    std::printf(
        "mutant parity: %u mutants - %u accepted by both with identical "
        "bytes, %u rejected by both, %u declined as outside the subset, %u "
        "over-strict against a reference that decoded the frame's own bytes, "
        "%u refused where the reference decoded a corrupt frame into bytes "
        "the unmutated fixture does not produce\n",
        counts->mutants, counts->both_accept, counts->both_reject,
        counts->declined, counts->overstrict, counts->reference_permissive);
    return 0;
}

/* ---- section 4: capacity and the window ---- */

int RunCapacityAndWindow() {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    Bytes frame;
    Bytes plain;
    for (size_t i = 0; i < fixtures.size(); i++) {
        Bytes decoded;
        const Bytes candidate(fixtures[i].compressed.begin(),
                              fixtures[i].compressed.end());
        if (!ZstdOracleDecodes(candidate, &decoded) || decoded.size() < 64) {
            continue;
        }
        frame = candidate;
        plain = decoded;
        break;
    }
    REQUIRE(!frame.empty());

    struct CapacityCase {
        const char* name;
        size_t capacity;
        bool decodes;
    };
    std::vector<CapacityCase> cases;
    cases.push_back({"capacity zero", 0, false});
    cases.push_back({"one byte short", plain.size() - 1, false});
    cases.push_back({"exactly the declaration", plain.size(), true});
    cases.push_back({"one byte over", plain.size() + 1, true});
    cases.push_back({"generous", plain.size() + 4096, true});

    std::vector<Chunk> chunks;
    for (size_t i = 0; i < cases.size(); i++) {
        Chunk chunk;
        chunk.name = cases[i].name;
        chunk.src = frame;
        chunk.dst_capacity = cases[i].capacity;
        chunks.push_back(chunk);
    }
    /* A window past the ceiling section 12.2 authorises, which is a decline
     * and not a refusal: a Window_Descriptor of 0xFF is 3.75 x 2^41 bytes and
     * no destination makes it decodable here. */
    {
        Chunk chunk;
        chunk.name = "window past the ceiling";
        chunk.src.push_back(0x28);
        chunk.src.push_back(0xB5);
        chunk.src.push_back(0x2F);
        chunk.src.push_back(0xFD);
        /* Descriptor: Frame_Content_Size present at one byte, not single
         * segment, so a Window_Descriptor follows. */
        chunk.src.push_back(0x00);
        chunk.src.push_back(0xFF);
        chunk.src.push_back(0x20);
        chunk.src.push_back(0x1D);
        chunk.src.push_back(0x00);
        chunk.src.push_back(0x00);
        for (unsigned i = 0; i < 3; i++) {
            chunk.src.push_back('w');
        }
        chunk.dst_capacity = 4096;
        chunks.push_back(chunk);
    }

    Batch batch;
    REQUIRE(BuildBatch(chunks, &batch) == 0);
    REQUIRE(RunShippedEntry(batch, "capacity and window") == 0);
    Bytes dst;
    std::vector<cudec_chunk_result> results;
    REQUIRE(Download(batch, &dst, &results) == 0);

    for (size_t i = 0; i < cases.size(); i++) {
        if (cases[i].decodes) {
            REQUIRE_CTX(results[i].status == CUDEC_OK, "%s: status %d",
                        cases[i].name, static_cast<int>(results[i].status));
            REQUIRE_CTX(results[i].bytes_written == plain.size(),
                        "%s: wrote %llu bytes, the reference produced %llu",
                        cases[i].name,
                        static_cast<unsigned long long>(
                            results[i].bytes_written),
                        static_cast<unsigned long long>(plain.size()));
            REQUIRE_CTX(equal_bytes(dst.data() + batch.dst_offsets[i],
                                    plain.data(), plain.size()),
                        "%s: the bytes differ from the reference",
                        cases[i].name);
            /* Whatever the capacity left over the declaration is, the kernel
             * had no business touching it. */
            for (size_t at = plain.size(); at < batch.caps[i]; at++) {
                REQUIRE_CTX(dst[batch.dst_offsets[i] + at] == kDstPoison,
                            "%s: byte %zu past bytes_written is not the poison",
                            cases[i].name, at);
            }
            continue;
        }
        REQUIRE_CTX(results[i].status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                    "%s: status %d, want OUTPUT_TOO_SMALL", cases[i].name,
                    static_cast<int>(results[i].status));
        REQUIRE_CTX(results[i].bytes_written == 0,
                    "%s: refused and still reported bytes", cases[i].name);
        /* A capacity refusal is decided from the frame header before any
         * block runs, so this one really does leave the destination alone -
         * which is more than the contract promises and is asserted because
         * a write here would be a write past a capacity of zero. */
        const size_t span = batch.caps[i] ? batch.caps[i] : 1u;
        for (size_t at = 0; at < span; at++) {
            REQUIRE_CTX(dst[batch.dst_offsets[i] + at] == kDstPoison,
                        "%s: byte %zu of a capacity refusal's destination was "
                        "written",
                        cases[i].name, at);
        }
    }
    const size_t window_index = cases.size();
    REQUIRE_CTX(results[window_index].status == CUDEC_ERR_UNSUPPORTED,
                "window past the ceiling: status %d, want UNSUPPORTED",
                static_cast<int>(results[window_index].status));
    REQUIRE(results[window_index].bytes_written == 0);

    FreeBatch(&batch);
    std::printf(
        "capacity and window: five capacities either side of the declaration "
        "and a window past the ceiling, each answered from the declaration "
        "and the caller's bound alone\n");
    return 0;
}

}  // namespace

int main() {
    if (RunDeterminism() != 0) {
        return 1;
    }
    if (RunWidthRefusal() != 0) {
        return 1;
    }
    ParityCounts counts;
    if (RunMutantParity(&counts) != 0) {
        return 1;
    }
    if (RunCapacityAndWindow() != 0) {
        return 1;
    }
    std::printf(
        "PASS: the Zstd device gate set - determinism across five grids, a "
        "three-stream split and a repeat of the shipped entry; two-directional "
        "reject parity over %u mutants with %u over-strict and %u refused "
        "where the reference decoded a corrupt frame; and the capacity "
        "and window bounds\n",
        counts.mutants, counts.overstrict, counts.reference_permissive);
    return 0;
}
