/* Whole-TileStream decode (issue #177): the host-side envelope walker in
 * src/tilestream.h feeding the GDeflate page batch decoder through the
 * reusable streaming context. Host orchestration on top of the device engine,
 * the sibling of src/frame.cpp - masterplan section 11.4 (M4).
 *
 * THE KERNEL STAYS ENVELOPE-IGNORANT, AND THAT IS THE POINT OF THIS FILE. A
 * TileStream is DirectStorage's container; a GDeflate page is the format. The
 * device decodes pages and knows nothing about tables of contents, so
 * everything the container says - where each page starts, how long it is, how
 * many bytes it must produce - is decided here, on the host, before a byte
 * reaches a launch. The envelope_out_of_device ctest verifies the other half
 * of that sentence against the shipped binary rather than against this
 * comment.
 *
 * THIS FILE ADDS NO STAGING. Every buffer it uses is the streaming context's,
 * reached through cudec_stream_detail::DecodeOnStreamCtx, so a caller decoding
 * many streams pays the allocation once and a fault poisons the context the
 * way the LZ4 streaming entry documents. What is new here is the translation
 * from one validated tile table into the batch arrays that core already
 * consumes.
 *
 * COMPILED AS PLAIN C++, NEVER AS DEVICE CODE. src/tilestream.h is header-only
 * and host-only so the GPU-less runner and the fuzz targets reach it; this
 * translation unit joins the library beside src/frame.cpp and src/stream.cpp
 * and pulls in no CUDA header of its own. */
#include "cudec.h"
#include "stream_decode.h"
#include "tilestream.h"

#include <cstddef>
#include <vector>

namespace {

using cudec_detail::TileStreamDeclaredTileCount;
using cudec_detail::TileStreamInfo;
using cudec_detail::TileStreamParse;
using cudec_detail::TileStreamTile;

/* Parses the envelope into `tiles`/`info`, sizing `tiles` from the declared
 * count. Returns what the parser returns; on any non-OK status `tiles` and
 * `info` say nothing. A zero declared count is handed to the parser anyway
 * rather than short-circuited here, so the refusal comes from the one place
 * that decides refusals - and the vector is one entry longer than the count so
 * even a zero-tile call has a non-null data() to hand over, which keeps the
 * parser's NULL-argument branch about a caller's mistake rather than about
 * this sizing. */
cudec_status ParseEnvelope(const unsigned char* stream, size_t stream_size,
                           std::vector<TileStreamTile>* tiles,
                           TileStreamInfo* info) {
    const size_t declared = static_cast<size_t>(TileStreamDeclaredTileCount(
        stream, static_cast<uint64_t>(stream_size)));
    /* EXACTLY the declared count, never a spare entry. The parser writes one
     * entry per tile the stream declares, so an array with slack puts an
     * off-by-one in that loop into allocator padding where the sanitizer
     * cannot see it - which is the discipline fuzz/fuzz_gdeflate_tilestream.cpp
     * states for its own array, and the library's own call has no business
     * being the softer of the two. The zero case still needs a non-null
     * data() to hand over, because a null one is the parser's
     * caller-mistake branch rather than its refusal of these bytes. */
    tiles->assign(declared != 0 ? declared : 1, TileStreamTile());
    return TileStreamParse(stream, static_cast<uint64_t>(stream_size),
                           tiles->data(), static_cast<uint64_t>(declared),
                           info);
}

/* The batch arrays an accepted call decodes with, and the total the envelope
 * declared them to produce. */
struct TileStreamPlan {
    size_t declared_total;
    std::vector<const void*> src_ptrs;
    std::vector<size_t> src_sizes;
    std::vector<void*> dst_ptrs;
    std::vector<size_t> dst_caps;
};

/* The whole synchronous reject surface of both decode entries, plus the plan
 * an accepted call runs. Everything here is host work on host bytes: argument
 * checks, the envelope walk, the two capacity refusals the envelope decides,
 * and the translation of the tile table into the batch arrays.
 *
 * IT IS ONE FUNCTION BECAUSE THE ONE-SHOT MUST ANSWER EVERY REJECT WITHOUT A
 * CONTEXT. Creating one first would put cudec_stream_ctx_create ahead of the
 * refusals, and on a GPU-less host that call fails for want of a device - so a
 * truncated envelope would come back CUDEC_ERR_CUDA instead of
 * CUDEC_ERR_CORRUPT_INPUT, and the conformance runner could reach none of it.
 * That is the reasoning the LZ4 one-shot records for its argument validation,
 * carried one step further because this entry's refusals depend on content and
 * not only on pointers.
 *
 * `ctx` is not examined here: a NULL one is the _ctx entry's own reject and the
 * one-shot never has one to check. */
cudec_status BuildTileStreamPlan(const void* stream, size_t stream_size,
                                 void* dst, size_t dst_capacity,
                                 cudec_mem_space dst_space,
                                 const cudec_chunk_result* h_tile_results,
                                 size_t tile_results_capacity,
                                 const size_t* bytes_written,
                                 TileStreamPlan* plan) {
    if (stream == nullptr || bytes_written == nullptr ||
        h_tile_results == nullptr ||
        (dst == nullptr && dst_capacity != 0) ||
        (dst_space != CUDEC_MEM_HOST && dst_space != CUDEC_MEM_DEVICE)) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }

    const unsigned char* bytes = static_cast<const unsigned char*>(stream);
    std::vector<TileStreamTile> tiles;
    TileStreamInfo info;
    const cudec_status parsed =
        ParseEnvelope(bytes, stream_size, &tiles, &info);
    if (parsed != CUDEC_OK) {
        return parsed;
    }
    const size_t n = static_cast<size_t>(info.tile_count);
    /* Both capacity refusals are decided from the PARSED envelope, so both are
     * content-dependent and both happen before a byte of dst is touched.
     * Refusing rather than decoding the prefix that would fit is the frame
     * precedent: a partially written destination is not an answer this ABI has
     * a way to describe. */
    if (tile_results_capacity < n) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    /* The narrowing is safe only because the container's ceiling is
     * 65535 * 65536 = 4294901760, which is inside a 32-bit size_t by 65536
     * bytes - the equality src/tilestream.h pins with a static_assert. It is
     * re-checked rather than trusted, because the failure mode if that ceiling
     * ever moves is a wrapped total comparing below dst_capacity and every
     * bound after it being computed from the wrong number. */
    if (info.total_uncompressed > static_cast<uint64_t>(SIZE_MAX)) {
        return CUDEC_ERR_OUTPUT_TOO_SMALL;
    }
    const size_t total = static_cast<size_t>(info.total_uncompressed);
    if (dst_capacity < total) {
        return CUDEC_ERR_OUTPUT_TOO_SMALL;
    }

    plan->declared_total = total;
    plan->src_ptrs.resize(n);
    plan->src_sizes.resize(n);
    plan->dst_ptrs.resize(n);
    plan->dst_caps.resize(n);
    unsigned char* out = static_cast<unsigned char*>(dst);
    size_t out_off = 0;
    for (size_t i = 0; i < n; i++) {
        plan->src_ptrs[i] = bytes + static_cast<size_t>(tiles[i].src_off);
        plan->src_sizes[i] = static_cast<size_t>(tiles[i].src_size);
        plan->dst_ptrs[i] = out + out_off;
        /* The capacity handed to the page decoder is the size the ENVELOPE
         * declares for this tile, never the space left in dst. That makes the
         * container's declaration a bound the kernel enforces: a page whose
         * bitstream decodes to more than its tile was declared to hold is
         * refused with CUDEC_ERR_OUTPUT_TOO_SMALL for that tile instead of
         * spilling into the next tile's bytes. */
        plan->dst_caps[i] = static_cast<size_t>(tiles[i].dst_size);
        out_off += plan->dst_caps[i];
    }
    return CUDEC_OK;
}

/* A defined not-produced record in every tile slot the envelope declared.
 * Used only once the call has been ACCEPTED, so the count is the parsed one
 * and the caller's array is already known to be at least that long. */
void StampTilesNotDecoded(cudec_chunk_result* results, size_t n) {
    for (size_t i = 0; i < n; i++) {
        results[i].status = static_cast<int32_t>(CUDEC_ERR_CUDA);
        results[i].reserved = 0;
        results[i].bytes_written = 0;
    }
}

/* Runs an accepted plan on `ctx` and decides the whole-stream answer. */
cudec_status RunTileStreamPlan(cudec_stream_ctx* ctx,
                               const TileStreamPlan& plan,
                               cudec_mem_space dst_space,
                               cudec_chunk_result* h_tile_results,
                               size_t* bytes_written) {
    const size_t n = plan.src_ptrs.size();
    const cudec_status st = cudec_stream_detail::DecodeOnStreamCtx(
        ctx, cudec_gdeflate_decompress_batch, plan.src_ptrs.data(),
        plan.src_sizes.data(), plan.dst_ptrs.data(), plan.dst_caps.data(), n,
        dst_space, h_tile_results);
    if (st != CUDEC_OK) {
        /* *bytes_written stays 0. Per-tile outcomes are already in
         * h_tile_results, each one defined, so a caller that wants to know
         * WHICH tile failed reads them; the aggregate says only that the
         * stream did not decode. */
        return st;
    }
    /* THE ENVELOPE'S DECLARED TILE SIZE IS A TWO-SIDED BOUND, AND THIS IS THE
     * SECOND SIDE. The capacity handed to the page decoder refuses a tile that
     * produces MORE than it was declared to. Nothing in the page format
     * refuses one that produces LESS: a GDeflate page ends at its final block
     * and reports what it produced, with no comparison against the capacity it
     * was given, so a valid page compressed from ten bytes decodes CUDEC_OK
     * into a tile the container declared as 65536.
     *
     * Accepting that would be the worst answer this entry could give. The
     * tiles are laid out at DECLARED offsets, so a short tile leaves the gap
     * between what it wrote and where its neighbour starts holding whatever
     * was in the caller's destination - prior contents on the host path,
     * uninitialised device memory on the CUDEC_MEM_DEVICE path - and the
     * summed bytes_written would hand the caller a range spanning it. That is
     * output the decoder never produced, presented as a successful decode,
     * and it is not even deterministic: the same stream would read back
     * differently depending on what the destination held.
     *
     * So a tile that decoded must have decoded exactly what the container said
     * it would. A disagreement between the two is the container being wrong
     * about its own contents, which is CUDEC_ERR_CORRUPT_INPUT, and it is
     * stamped onto the tile so the per-tile channel names which one. */
    size_t total = 0;
    cudec_status short_status = CUDEC_OK;
    for (size_t i = 0; i < n; i++) {
        if (h_tile_results[i].bytes_written !=
            static_cast<uint64_t>(plan.dst_caps[i])) {
            h_tile_results[i].status =
                static_cast<int32_t>(CUDEC_ERR_CORRUPT_INPUT);
            h_tile_results[i].reserved = 0;
            h_tile_results[i].bytes_written = 0;
            if (short_status == CUDEC_OK) {
                short_status = CUDEC_ERR_CORRUPT_INPUT;
            }
            continue;
        }
        total += plan.dst_caps[i];
    }
    if (short_status != CUDEC_OK) {
        return short_status;
    }
    /* Derived from the envelope rather than summed from the device, now that
     * the two are known to agree. A length a caller will read `dst` with is
     * not a number to take from a device-written record when a validated host
     * one says the same thing. */
    *bytes_written = plan.declared_total;
    (void)total;
    return CUDEC_OK;
}

}  // namespace

cudec_status cudec_gdeflate_tilestream_info(const void* stream,
                                            size_t stream_size,
                                            size_t* tile_count,
                                            size_t* uncompressed_size) {
    if (tile_count != nullptr) {
        *tile_count = 0;
    }
    if (uncompressed_size != nullptr) {
        *uncompressed_size = 0;
    }
    if (stream == nullptr || tile_count == nullptr ||
        uncompressed_size == nullptr) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    const unsigned char* bytes = static_cast<const unsigned char*>(stream);
    std::vector<TileStreamTile> tiles;
    TileStreamInfo info;
    cudec_status st;
    try {
        st = ParseEnvelope(bytes, stream_size, &tiles, &info);
    } catch (...) {
        /* The only allocation here is the tile table, bounded by the declared
         * count's own 16-bit width. A host OOM never crosses the C ABI as a
         * throw. */
        return CUDEC_ERR_CUDA;
    }
    if (st != CUDEC_OK) {
        return st;
    }
    *tile_count = static_cast<size_t>(info.tile_count);
    *uncompressed_size = static_cast<size_t>(info.total_uncompressed);
    return CUDEC_OK;
}

cudec_status cudec_gdeflate_tilestream_decompress_ctx(
    cudec_stream_ctx* ctx, const void* stream, size_t stream_size, void* dst,
    size_t dst_capacity, cudec_mem_space dst_space,
    cudec_chunk_result* h_tile_results, size_t tile_results_capacity,
    size_t* bytes_written) {
    if (bytes_written != nullptr) {
        *bytes_written = 0;
    }
    if (ctx == nullptr) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    TileStreamPlan plan;
    cudec_status planned;
    try {
        planned = BuildTileStreamPlan(stream, stream_size, dst, dst_capacity,
                                      dst_space, h_tile_results,
                                      tile_results_capacity, bytes_written,
                                      &plan);
    } catch (...) {
        return CUDEC_ERR_CUDA; /* host OOM never crosses the C ABI as a throw */
    }
    if (planned != CUDEC_OK) {
        return planned;
    }
    return RunTileStreamPlan(ctx, plan, dst_space, h_tile_results,
                             bytes_written);
}

cudec_status cudec_gdeflate_tilestream_decompress(
    const void* stream, size_t stream_size, void* dst, size_t dst_capacity,
    cudec_mem_space dst_space, cudec_chunk_result* h_tile_results,
    size_t tile_results_capacity, size_t* bytes_written) {
    if (bytes_written != nullptr) {
        *bytes_written = 0;
    }
    TileStreamPlan plan;
    cudec_status planned;
    try {
        planned = BuildTileStreamPlan(stream, stream_size, dst, dst_capacity,
                                      dst_space, h_tile_results,
                                      tile_results_capacity, bytes_written,
                                      &plan);
    } catch (...) {
        return CUDEC_ERR_CUDA;
    }
    if (planned != CUDEC_OK) {
        return planned;
    }
    cudec_stream_ctx* ctx = nullptr;
    const cudec_status created = cudec_stream_ctx_create(&ctx);
    if (created != CUDEC_OK) {
        /* The call was ACCEPTED and then a resource failed, which is the one
         * class where this entry owes the per-tile channel a defined value:
         * the caller has been told the stream is well-formed, and CUDEC_OK is
         * zero, so a results array it zero-initialised would otherwise read
         * as every tile having decoded nothing successfully. On a GPU-less
         * host this is the ordinary path, not the exotic one. */
        StampTilesNotDecoded(h_tile_results, plan.src_ptrs.size());
        return created;
    }
    const cudec_status st =
        RunTileStreamPlan(ctx, plan, dst_space, h_tile_results, bytes_written);
    cudec_stream_ctx_destroy(ctx);
    return st;
}
