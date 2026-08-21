/* Fuzz target over the GDeflate TileStream envelope walker (issue #184), the
 * third target in fuzz/ and the first one with no oracle behind it.
 *
 * WHY THERE IS NO DIFFERENTIAL HALF, STATED FIRST BECAUSE IT CHANGES WHAT THE
 * VERDICTS MEAN. The other two targets compare a twin against a reference and
 * trap on a divergence. This container has no reference in this tree: the
 * DirectStorage wrapper that would answer for it is not vendored, and the
 * GDeflate oracle that is vendored decodes raw pages and never sees an
 * envelope. So the property asserted here is an INVARIANT rather than a
 * parity: whatever the walker accepts must describe ranges that lie inside the
 * stream it was handed, in order, none of them empty, with an output total
 * that did not wrap. Reject-parity against the real vectors is issues #169 and
 * #179 and is not approximated here.
 *
 * That is a weaker proof than parity in one direction and a stronger one in
 * another. Weaker, because a walker that rejected everything would satisfy it;
 * the seed corpus carries accepted envelopes so that arm is exercised, and the
 * selftest twin below is what says the assertions are live at all. Stronger,
 * because the invariant is checked on the ACCEPTED output rather than on a
 * comparison, so a fail-open is caught by what it produced and not by what
 * some other implementation happened to think of it.
 *
 * THE TILE ARRAY IS ALLOCATED AT EXACTLY THE DECLARED COUNT. The walker writes
 * one entry per tile the stream declares, so an off-by-one in that loop lands
 * in allocator slack if the array is generous, and lands in a redzone if it is
 * not. The count is read here the same way the walker reads it, from the same
 * bytes, so the two agree by construction rather than by the harness being
 * kind.
 *
 * WHAT THIS TARGET CANNOT REACH, so its clean runs are not read as covering
 * it. The walker's total-output arithmetic is at its most interesting near the
 * container's ceiling of 65535 tiles, and a table that large is 262148 bytes
 * of table alone. libFuzzer runs here under an input bound far below that (the
 * CI job's -max_len), so no input this target ever sees can carry a tile count
 * anywhere near the ceiling. That arithmetic is pinned by a fixture in
 * tests/tilestream_host_negative.cpp instead, where the stream can simply be
 * built. */
#include "cudec.h"
#include "tilestream.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

using cudec_detail::kTileStreamHeaderBytes;
using cudec_detail::kTileStreamMaxTiles;
using cudec_detail::kTileStreamTileSize;
using cudec_detail::kTileStreamTocEntryBytes;
using cudec_detail::TileStreamInfo;
using cudec_detail::TileStreamParse;
using cudec_detail::TileStreamTile;

/* Bounded so libFuzzer explores the table walk rather than the allocator. The
 * number is this harness's, not the container's; the container's own ceiling
 * is far above it and is the part named as unreachable in the header above. */
constexpr size_t kMaxStream = 1u << 13;

void Trap(const char* what, size_t stream_size) {
    std::fprintf(stderr, "INVARIANT DIVERGENCE: %s; stream=%zu\n", what,
                 stream_size);
    __builtin_trap();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    /* One byte decides the capacity the walker is offered, so the caller-error
     * arm - a tile array too small for the count the stream declares - is
     * reachable rather than being an arm the harness never takes. The rest is
     * the stream. */
    if (size < 1) {
        return 0;
    }
    const uint8_t selector = data[0];
    size_t stream_size = size - 1;
    if (stream_size > kMaxStream) {
        stream_size = kMaxStream;
    }

    /* libFuzzer hands out a slice of a buffer sized to -max_len rather than to
     * this input, so a read past the stream would land in that slack and stay
     * green. The walker runs over an exactly-sized copy instead. */
    auto stream = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(stream.get(), data + 1, stream_size);
    }

    /* The declared count, read from the bytes the walker reads it from. Below
     * four bytes there is no count in the stream at all and the walker will
     * refuse on the header length before it looks. */
    uint64_t declared = 0;
    if (stream_size >= 4) {
        declared = cudec_detail::TileStreamRead16LE(stream.get() + 2);
    }
    uint64_t capacity = declared;
    if ((selector & 1u) != 0u) {
        const uint64_t shortfall = selector >> 1;
        capacity = (declared > shortfall) ? (declared - shortfall) : 0;
    }

    /* Exactly the capacity, so an entry written past it hits a redzone rather
     * than slack. A zero-length array is a legal allocation and is never null,
     * which keeps the walker's null-argument arm the caller's mistake it is
     * meant to be rather than something this harness manufactures. */
    auto tiles = std::make_unique<TileStreamTile[]>(capacity);
    TileStreamInfo info;
    std::memset(&info, 0, sizeof(info));

    const cudec_status status =
        TileStreamParse(stream.get(), stream_size, tiles.get(), capacity, &info);

    if (status != CUDEC_OK) {
        /* A refusal is the expected answer for almost every input, and the
         * only thing owed about it is that it was one of the two defined ones.
         * A third status here would be a walker returning something no caller
         * has a rule for. */
        if (status != CUDEC_ERR_CORRUPT_INPUT &&
            status != CUDEC_ERR_INVALID_ARGUMENT) {
            Trap("a refusal with an undefined status", stream_size);
        }
        return 0;
    }

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to show the assertions below are live
     * without waiting for a real fail-open: a second binary built with this
     * defined pushes an accepted tile past the end of the stream it came from,
     * so a harness that had silently stopped checking passes where this one
     * traps. Never define it in a build whose findings are being believed. */
    if (info.tile_count != 0) {
        tiles[info.tile_count - 1].src_off = stream_size;
        tiles[info.tile_count - 1].src_size = 1;
    }
#endif

    /* Everything below is asserted on the ACCEPTED result. This is the
     * invariant, not a byproduct of it. */
    if (info.tile_count == 0 || info.tile_count > kTileStreamMaxTiles) {
        Trap("an accepted envelope declaring no tile, or more than the field "
             "can hold",
             stream_size);
    }
    if (info.tile_count > capacity) {
        Trap("more tiles reported than the array offered room for",
             stream_size);
    }

    const uint64_t toc_end =
        kTileStreamHeaderBytes + info.tile_count * kTileStreamTocEntryBytes;
    uint64_t expected_total = 0;
    uint64_t prev_end = 0;
    for (uint64_t i = 0; i < info.tile_count; ++i) {
        const TileStreamTile& t = tiles[i];
        if (t.src_off < toc_end) {
            Trap("a tile starting inside the table that describes it",
                 stream_size);
        }
        if (t.src_size == 0) {
            Trap("a tile of no bytes", stream_size);
        }
        /* The addition is the thing under suspicion, so it is not the thing
         * doing the checking: the bound is compared by subtraction on operands
         * already known to be inside the stream. */
        if (t.src_off > stream_size || t.src_size > stream_size - t.src_off) {
            Trap("a tile running past the end of the stream", stream_size);
        }
        if (i > 0 && t.src_off < prev_end) {
            Trap("tiles that overlap or run backwards", stream_size);
        }
        prev_end = t.src_off + t.src_size;

        const uint64_t expected_dst =
            (i + 1 == info.tile_count) ? t.dst_size : kTileStreamTileSize;
        if (i + 1 != info.tile_count && t.dst_size != kTileStreamTileSize) {
            Trap("a tile before the last one that does not produce a full tile",
                 stream_size);
        }
        if (expected_dst == 0 || expected_dst > kTileStreamTileSize) {
            Trap("a tile producing nothing, or more than a tile holds",
                 stream_size);
        }
        expected_total += expected_dst;
    }

    if (info.total_uncompressed != expected_total) {
        std::fprintf(stderr, "total=%llu summed=%llu\n",
                     static_cast<unsigned long long>(info.total_uncompressed),
                     static_cast<unsigned long long>(expected_total));
        Trap("a reported total that is not the sum of the tiles reported",
             stream_size);
    }
    /* The ceiling the container's own fields imply. Reached by no input this
     * target can be given, and asserted anyway: the day the bound moves, this
     * is where the assumption was written down. */
    if (info.total_uncompressed > kTileStreamMaxTiles * kTileStreamTileSize) {
        Trap("a total above what the tile count and the tile size allow",
             stream_size);
    }
    return 0;
}
