/* Determinism across launch geometry (issue #72). docs/DETERMINISM.md names
 * the decode path's invariance level `gpu_to_gpu` in CCCL's vocabulary; the
 * existing same-batch-twice compare only covers the run-to-run axis on ONE
 * launch configuration, which is the axis a decoder is least likely to break.
 * The interesting axis is geometry: how many warps exist, how the grid-stride
 * loop maps chunks onto them, and whether the batch is split across concurrent
 * streams. All of that changes which warp decodes which chunk and in what
 * order - and none of it may change a single output byte.
 *
 * Per format: one corpus, one reference decode through the shipped entry
 * point, then five runs each across five grid/block geometries plus a
 * three-stream split, with the whole destination arena re-poisoned before
 * every run and compared in full - decoded bytes AND the untouched poison
 * beyond bytes_written - along with every per-chunk result record. The
 * geometries are launched directly on the shipped kernel because the public
 * ABI deliberately exposes no launch knobs.
 *
 * Both formats run it (issue #154). The geometry axis belongs to the chunk
 * decoder rather than to a parser, and the two share one, so a format left
 * out here would leave the shared mapping unproven for exactly the
 * instantiation nobody looked at. */
#include "cudec.h"
#include "fixtures.h"
#include "chunk_decode.cuh"
#include "lz4_block.h"
#include "snappy_block.h"
#include "require.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr unsigned char kDstPoison = 0xA5;
constexpr int kRunsPerGeometry = 5;

/* The batch, staged as two device arenas (one source blob, one destination
 * arena) plus the five pointer/size tables the ABI takes. One arena makes the
 * whole-output compare a single download, so a geometry's full byte image is
 * checked, not a sample of it. */
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

struct Chunk {
    std::string name;
    std::vector<unsigned char> src;
    size_t dst_capacity;
};

using Mutator = std::vector<Mutant> (*)(const std::vector<unsigned char>&,
                                        uint64_t);

/* Pristine fixtures and their whole mutant corpora: accepted and rejected
 * chunks side by side, so the compare covers the reject path's result records
 * and untouched destinations too, not just successful output. */
std::vector<Chunk> MakeChunks(const std::vector<Fixture>& fixtures,
                              Mutator mutate) {
    std::vector<Chunk> chunks;
    for (const Fixture& f : fixtures) {
        chunks.push_back({f.name, f.compressed, f.original.size()});
        for (const Mutant& m : mutate(f.compressed, f.seed)) {
            chunks.push_back(
                {f.name + "/" + m.description, m.stream, f.original.size()});
        }
    }
    return chunks;
}

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
    for (size_t i = 0; i < n; i++) {
        if (!chunks[i].src.empty()) {
            std::copy(chunks[i].src.begin(), chunks[i].src.end(),
                      src_blob.begin() + static_cast<long>(src_off[i]));
        }
    }

    batch->n = n;
    batch->dst_arena_size = dst_total;
    REQUIRE_CUDA(cudaMalloc(&batch->d_src_blob, src_total));
    REQUIRE_CUDA(cudaMalloc(&batch->d_dst_arena, dst_total));
    REQUIRE_CUDA(cudaMemcpy(batch->d_src_blob, src_blob.data(), src_total,
                            cudaMemcpyHostToDevice));

    std::vector<const void*> h_srcs(n);
    std::vector<void*> h_dsts(n);
    for (size_t i = 0; i < n; i++) {
        batch->names.push_back(chunks[i].name);
        h_srcs[i] = batch->d_src_blob + src_off[i];
        h_dsts[i] = batch->d_dst_arena + dst_off[i];
    }
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
    return 0;
}

/* Every run starts from the identical device state: a fully re-poisoned
 * destination arena and result records primed with the 0xFF non-OK sentinel.
 * Without this a later run could "match" by inheriting the previous run's
 * bytes. */
int ResetDeviceState(const Batch& b) {
    REQUIRE_CUDA(cudaMemset(b.d_dst_arena, kDstPoison, b.dst_arena_size));
    REQUIRE_CUDA(
        cudaMemset(b.d_results, 0xFF, b.n * sizeof(*b.d_results)));
    return 0;
}

int Download(const Batch& b, std::vector<unsigned char>* dst,
             std::vector<cudec_chunk_result>* results) {
    dst->assign(b.dst_arena_size, 0);
    results->assign(b.n, cudec_chunk_result{});
    REQUIRE_CUDA(cudaMemcpy(dst->data(), b.d_dst_arena, b.dst_arena_size,
                            cudaMemcpyDeviceToHost));
    REQUIRE_CUDA(cudaMemcpy(results->data(), b.d_results,
                            b.n * sizeof(*b.d_results),
                            cudaMemcpyDeviceToHost));
    return 0;
}

struct Geometry {
    const char* name;
    unsigned blocks;
    unsigned threads;
};

/* Grid/block shapes that change the chunk-to-warp mapping in every direction
 * the kernel admits: a single warp walking the whole batch through the
 * grid-stride loop, block sizes below the launch bound, a grid far larger than
 * the batch (most warps idle), and the shipped sizing. Block sizes stay
 * multiples of the warp size and within __launch_bounds__(128). */
std::vector<Geometry> Geometries(size_t chunk_count) {
    return {
        {"1x32", 1, 32},
        {"3x64", 3, 64},
        {"17x96", 17, 96},
        {"chunks x128", static_cast<unsigned>(chunk_count), 128},
        {"shipped",
         cudec_detail::decode_grid_blocks(chunk_count),
         cudec_detail::kBlockThreads},
    };
}

/* One format's two ways in: its public entry, and a direct launch of its
 * instantiation of the one chunk decoder, which is how a geometry the public
 * ABI exposes no knob for is reached. Both are function pointers so the whole
 * sequence below runs unchanged per format - a second copy of it would be a
 * second place for one format's axis to quietly stop being covered. */
using BatchEntry = cudec_status (*)(const void* const*, const size_t*,
                                    void* const*, const size_t*, size_t,
                                    cudec_chunk_result*, cudec_stream_t);
using DirectLaunch = void (*)(const Batch&, const Geometry&, cudaStream_t);

template <class Parser>
void LaunchDirect(const Batch& b, const Geometry& g, cudaStream_t stream) {
    cudec_detail::chunk_decode_batch<Parser, false,
                                    cudec_detail::kCudaWaveSize>
        <<<g.blocks, g.threads, 0, stream>>>(b.d_srcs, b.d_sizes, b.d_dsts,
                                             b.d_caps, b.n, b.d_results);
}

struct Format {
    const char* name;
    BatchEntry entry;
    DirectLaunch launch;
};

int RunDirect(const Format& f, const Batch& b, const Geometry& g) {
    REQUIRE(ResetDeviceState(b) == 0);
    cudaStream_t stream;
    REQUIRE_CUDA(cudaStreamCreate(&stream));
    f.launch(b, g, stream);
    REQUIRE_CUDA(cudaGetLastError());
    REQUIRE_CUDA(cudaStreamSynchronize(stream));
    REQUIRE_CUDA(cudaStreamDestroy(stream));
    return 0;
}

/* The shipped entry point, once over the whole batch. */
int RunShippedEntry(const Format& f, const Batch& b) {
    REQUIRE(ResetDeviceState(b) == 0);
    cudaStream_t stream;
    REQUIRE_CUDA(cudaStreamCreate(&stream));
    REQUIRE(f.entry(b.d_srcs, b.d_sizes, b.d_dsts, b.d_caps, b.n, b.d_results,
                    stream) == CUDEC_OK);
    REQUIRE_CUDA(cudaStreamSynchronize(stream));
    REQUIRE_CUDA(cudaStreamDestroy(stream));
    return 0;
}

/* The same batch as three sub-batches submitted to three concurrent streams:
 * chunk k is decoded by a different warp of a different launch than in any
 * other geometry, and the sub-batches complete in an order the test does not
 * control. Per-chunk independence means the output must not notice. */
int RunSplitStreams(const Format& f, const Batch& b, unsigned stream_count) {
    REQUIRE(ResetDeviceState(b) == 0);
    std::vector<cudaStream_t> streams(stream_count);
    for (unsigned s = 0; s < stream_count; s++) {
        REQUIRE_CUDA(cudaStreamCreate(&streams[s]));
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
        REQUIRE(f.entry(b.d_srcs + begin, b.d_sizes + begin,
                        b.d_dsts + begin, b.d_caps + begin, count,
                        b.d_results + begin, streams[s]) == CUDEC_OK);
    }
    for (unsigned s = 0; s < stream_count; s++) {
        REQUIRE_CUDA(cudaStreamSynchronize(streams[s]));
        REQUIRE_CUDA(cudaStreamDestroy(streams[s]));
    }
    return 0;
}

int CompareAgainstReference(
    const Batch& b, const std::string& context,
    const std::vector<unsigned char>& ref_dst,
    const std::vector<cudec_chunk_result>& ref_results,
    const std::vector<unsigned char>& dst,
    const std::vector<cudec_chunk_result>& results) {
    for (size_t i = 0; i < b.n; i++) {
        REQUIRE_CTX(results[i].status == ref_results[i].status,
                    "%s: chunk %zu (%s) status %d, reference %d",
                    context.c_str(), i, b.names[i].c_str(),
                    static_cast<int>(results[i].status),
                    static_cast<int>(ref_results[i].status));
        REQUIRE_CTX(results[i].bytes_written == ref_results[i].bytes_written,
                    "%s: chunk %zu (%s) wrote %llu bytes, reference %llu",
                    context.c_str(), i, b.names[i].c_str(),
                    static_cast<unsigned long long>(results[i].bytes_written),
                    static_cast<unsigned long long>(
                        ref_results[i].bytes_written));
        REQUIRE_CTX(results[i].reserved == ref_results[i].reserved,
                    "%s: chunk %zu reserved word drifted", context.c_str(), i);
    }
    /* The whole arena, including the bytes a rejected chunk happened to write
     * before rejecting. docs/DETERMINISM.md deliberately does NOT promise
     * those - so this asserts more than the contract. It holds today because
     * a chunk's writes are a pure function of its own bytes, and asserting it
     * catches a geometry-dependent write that the promised region would hide;
     * a future intentional change to reject-path writes is a contract-neutral
     * update to this test, not a determinism break. */
    REQUIRE_CTX(dst.size() == ref_dst.size(), "%s: arena size drifted",
                context.c_str());
    REQUIRE_CTX(equal_bytes(dst.data(), ref_dst.data(), dst.size()),
                "%s: destination arena differs from the reference decode",
                context.c_str());
    return 0;
}

/* The whole sequence for one format: build the batch, take a reference decode
 * through the shipped entry, then re-run it across every geometry and the
 * stream split and require the byte image and every result record to be
 * identical. */
int RunFormat(const Format& f, const std::vector<Fixture>& fixtures,
              Mutator mutate, size_t* chunks_out, size_t* decoded_out) {
    REQUIRE(!fixtures.empty());
    const std::vector<Chunk> chunks = MakeChunks(fixtures, mutate);
    REQUIRE(!chunks.empty());

    Batch batch;
    REQUIRE(BuildBatch(chunks, &batch) == 0);

    /* The reference: the shipped entry point on the shipped geometry. */
    std::vector<unsigned char> ref_dst;
    std::vector<cudec_chunk_result> ref_results;
    REQUIRE(RunShippedEntry(f, batch) == 0);
    REQUIRE(Download(batch, &ref_dst, &ref_results) == 0);
    size_t decoded_chunks = 0;
    for (const cudec_chunk_result& r : ref_results) {
        if (r.status == CUDEC_OK) {
            decoded_chunks++;
        }
    }
    /* A corpus that decoded nothing would compare poison against poison and
     * pass vacuously. */
    REQUIRE(decoded_chunks >= fixtures.size());

    std::vector<unsigned char> dst;
    std::vector<cudec_chunk_result> results;
    for (const Geometry& g : Geometries(batch.n)) {
        for (int run = 0; run < kRunsPerGeometry; run++) {
            REQUIRE(RunDirect(f, batch, g) == 0);
            REQUIRE(Download(batch, &dst, &results) == 0);
            REQUIRE(CompareAgainstReference(batch,
                                            std::string(f.name) + "/" +
                                                g.name + "/run-" +
                                                std::to_string(run),
                                            ref_dst, ref_results, dst,
                                            results) == 0);
        }
    }
    for (int run = 0; run < kRunsPerGeometry; run++) {
        REQUIRE(RunSplitStreams(f, batch, 3) == 0);
        REQUIRE(Download(batch, &dst, &results) == 0);
        REQUIRE(CompareAgainstReference(batch,
                                        std::string(f.name) +
                                            "/3-streams/run-" +
                                            std::to_string(run),
                                        ref_dst, ref_results, dst,
                                        results) == 0);
    }
    *chunks_out = batch.n;
    *decoded_out = decoded_chunks;
    return 0;
}

}  // namespace

int main() {
    /* Both formats, one sequence. The geometry axis is a property of the
     * chunk decoder rather than of a parser, so a format that skipped it
     * would leave the shared kernel's mapping unproven for exactly the
     * instantiation nobody looked at. */
    const Format formats[] = {
        {"lz4", cudec_lz4_decompress_batch,
         LaunchDirect<cudec_detail::Lz4Parser>},
        {"snappy", cudec_snappy_decompress_batch,
         LaunchDirect<cudec_detail::SnappyParser>},
    };
    const std::vector<Fixture> corpora[] = {MakeLz4BlockFixtures(),
                                            MakeSnappyFixtures()};
    const Mutator mutators[] = {MutateStream, MutateSnappyStream};

    for (size_t i = 0; i < 2; i++) {
        size_t chunk_count = 0;
        size_t decoded_chunks = 0;
        REQUIRE(RunFormat(formats[i], corpora[i], mutators[i], &chunk_count,
                          &decoded_chunks) == 0);
        std::printf("determinism_gpu: %s %zu chunks (%zu decoded) "
                    "bit-identical across 6 launch configurations x %d runs\n",
                    formats[i].name, chunk_count, decoded_chunks,
                    kRunsPerGeometry);
    }
    return 0;
}
