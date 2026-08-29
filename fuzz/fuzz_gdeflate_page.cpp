/* Differential fuzz target over the whole GDeflate page decode (issue #192):
 * the round/refill schedule, the code-length rounds, the table construction,
 * the block loop and the in-tile LZ77, run end to end against
 * `libdeflate_gdeflate_decompress` from the pinned NVIDIA/libdeflate fork.
 *
 * WHY BYTE PARITY IS LOAD-BEARING HERE IN A WAY IT IS NOT ELSEWHERE. This
 * format has no checksum anywhere (docs/MASTERPLAN.md section 11.4). For LZ4
 * and Zstd a mutated stream that survives the parser still has to survive a
 * checksum; here nothing is behind the parser at all, so parity against the
 * reference is the only thing between a mutated page and silent wrong output.
 *
 * THE TWO SIDES GET THE SAME BYTES AND NOT THE SAME BUFFER, AND THE REASON IS
 * A DEFECT IN THE REFERENCE. `ENSURE_BITS` in the fork's
 * gdeflate_decompress_template.h reads a 32-bit word with no bound check
 * against the end of the page - the format's own watermark discipline makes
 * that safe on a well-formed page and this target's whole business is pages
 * that are not. Handed the fuzzer's bytes in a tight allocation the reference
 * reads past the end of it, and what it reads there decides what it answers,
 * so its verdict would depend on whatever the allocator left behind and the
 * comparison would be against a coin. So the reference gets its copy in a
 * buffer with a zero tail after it while being told the same `nbytes`, which
 * makes that read deterministic without changing one byte of what either side
 * is asked about. The twin gets the bytes in an allocation of exactly their
 * own size, so refill-past-end is reachable rather than masked - which is what
 * this issue asks for - and an over-read on OUR side lands in an
 * AddressSanitizer redzone instead of in slack.
 *
 * HOW FAR PAST THE END THAT REFILL REACHES IS A FUNCTION OF THE OUTPUT
 * CAPACITY AND NOT OF THE PAGE, which is the part that has to be got right
 * rather than guessed. A tail of one word per lane looks like the obvious size
 * and is wrong by orders of magnitude: on a compressed block the reference has
 * no bound of its own at all, so it keeps taking a word per round until the
 * OUTPUT fills, and every round is a literal, a match reservation or a copy.
 * Rounds are therefore bounded by the capacity, not by the bytes the page had
 * left, and this target sizes the tail from the capacity for that reason. The
 * bound is measured as well as argued: sized by the lane count instead, the
 * target segfaults inside the reference within a few thousand execs.
 *
 * WHICH DIRECTION IS ASSERTED, AND WHICH IS NOT. The fail-open direction
 * always traps: the twin accepting a stream the reference refuses, or the two
 * accepting and disagreeing on the bytes or on the size. The reverse - the
 * twin refusing what the reference accepts - is NOT trapped, and that is a
 * deliberate weakening rather than an omission, because src/ is stricter than
 * the reference in three places that each say so at their own site:
 *
 *   - an empty code is refused on use, where the reference resolves it to a
 *     synthetic symbol the stream never encoded (src/gdeflate_tables.h);
 *   - a code-length repeat run reaching past HLIT + HDIST is refused, where
 *     the reference absorbs it into 137 slack entries it then never reads
 *     (src/gdeflate_tables.h);
 *   - a refill past the last word of the page is refused, which is the
 *     unchecked read above (src/gdeflate_schedule.h).
 *
 * All three are the fail-CLOSED direction, all three are reachable from
 * fuzzer bytes constantly, and trapping on them would make this target report
 * its own design. What watches that half instead is the byte parity on the
 * pages both sides accept, and the sanitizers, which have no opinion about
 * which decoder is stricter.
 *
 * THE PAGE COUNT AND THE CAPACITY COME FROM THE INPUT, and both sides receive
 * the same values. The reference hands each page whatever capacity is left
 * rather than a tile each, and returns what the page actually produced rather
 * than requiring it to fill; the twin is driven the same way, so a divergence
 * is a divergence about the page and never about the convention. */
#include "gdeflate_block.h"

#include <libdeflate.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using cudec_detail::GDeflateDecodePage;
using cudec_detail::GDeflatePageState;
using cudec_detail::kGDeflateNumStreams;

/* Bounded so libFuzzer explores the rounds rather than the allocator. Four
 * pages of a quarter tile each is far more round structure than an input this
 * size can fill, and it keeps the capacity below a megabyte. */
constexpr uint32_t kMaxPages = 4;
constexpr size_t kTileBytes = 64u * 1024u;
constexpr size_t kMaxPayload = 1u << 15;
/* The capacity the input names, capped. It is deliberately far below a tile:
 * the tail each page owes the reference is sized from this number, so a
 * generous capacity is paid for four times over in allocation on every single
 * exec. What a larger capacity would buy is longer outputs, and what it costs
 * is the rate at which the engine explores the rounds. */
constexpr size_t kMaxCapacity = 4096;

/* The shortest page the schedule admits: the priming round is 32 words, so a
 * chunk below this is refused before a bit is read. */
constexpr size_t kMinPageBytes = kGDeflateNumStreams * 4u;

/* Slack after the reference's copy of a page, for its unchecked refill, sized
 * from the capacity for the reason the file header gives. One word per round,
 * one round per output byte at worst, plus the rounds a block spends producing
 * nothing - its drain is one per lane and a dynamic header is at most 19
 * precode rounds and 320 length rounds. Six words per capacity byte plus a
 * page of slack covers all of it with a wide margin, and it is zero-filled
 * because the point is a deterministic answer rather than a plausible one. */
size_t OracleTailBytes(size_t capacity) {
    return 6u * capacity + 4096u;
}

void Trap(const char* what, size_t size) {
    std::fprintf(stderr, "PARITY DIVERGENCE: %s; input=%zu\n", what, size);
    __builtin_trap();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    /* Three bytes of envelope, then the pages. The page count and the capacity
     * are the two values a caller supplies and a page never states, so they
     * come from the input rather than from a constant here - a target that
     * fixed them would leave the capacity boundary and the multi-page walk
     * unexplored. */
    if (size < 4) {
        return 0;
    }
    const uint32_t npages = 1u + (data[0] % kMaxPages);
    size_t capacity = 1u + (static_cast<size_t>(data[1]) |
                            (static_cast<size_t>(data[2]) << 8));
    if (capacity > kMaxCapacity) {
        capacity = kMaxCapacity;
    }
    if (capacity > npages * kTileBytes) {
        capacity = npages * kTileBytes;
    }
    const size_t tail = OracleTailBytes(capacity);

    const uint8_t* payload = data + 3;
    size_t payload_size = size - 3;
    if (payload_size > kMaxPayload) {
        payload_size = kMaxPayload;
    }
    const size_t chunk = (payload_size / npages) & ~static_cast<size_t>(3);
    if (chunk < kMinPageBytes) {
        return 0;
    }

    /* The twin's copies: exactly their own size, so a read past the end of a
     * page is a redzone read rather than a read of the next page. */
    std::vector<std::unique_ptr<unsigned char[]> > tight(npages);
    /* The reference's copies: the same bytes with a zero tail behind them, for
     * the reason the file header gives. */
    std::vector<std::unique_ptr<unsigned char[]> > padded(npages);
    std::vector<libdeflate_gdeflate_in_page> in_pages(npages);
    for (uint32_t i = 0; i < npages; i++) {
        tight[i].reset(new unsigned char[chunk]);
        std::memcpy(tight[i].get(), payload + i * chunk, chunk);
        padded[i].reset(new unsigned char[chunk + tail]);
        std::memcpy(padded[i].get(), payload + i * chunk, chunk);
        std::memset(padded[i].get() + chunk, 0, tail);
        in_pages[i].data = padded[i].get();
        in_pages[i].nbytes = chunk;
    }

    std::unique_ptr<unsigned char[]> oracle_out(new unsigned char[capacity]);
    std::memset(oracle_out.get(), 0, capacity);
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        return 0;
    }
    /* The reference accumulates into this rather than assigning it, so a
     * caller that left it uninitialised would read its own stack as a decoded
     * size. */
    size_t oracle_size = 0;
    const libdeflate_result oracle_status = libdeflate_gdeflate_decompress(
        d, in_pages.data(), npages, oracle_out.get(), capacity, &oracle_size);
    libdeflate_free_gdeflate_decompressor(d);
    const bool oracle_ok = oracle_status == LIBDEFLATE_SUCCESS;

    std::unique_ptr<unsigned char[]> twin_out(new unsigned char[capacity]);
    std::memset(twin_out.get(), 0, capacity);
    size_t twin_size = 0;
    bool twin_ok = true;
    for (uint32_t i = 0; i < npages && twin_ok; i++) {
        GDeflatePageState st;
        uint64_t produced = 0;
        if (!GDeflateDecodePage(st, tight[i].get(), chunk,
                                twin_out.get() + twin_size,
                                capacity - twin_size, &produced)) {
            /* The contract src/gdeflate_schedule.h states: a refusal leaves
             * the sticky flag set, so a caller reading the flag and a caller
             * reading the return value never disagree. */
            if (!st.s.failed) {
                Trap("the page decode refused without failing the schedule",
                     size);
            }
            twin_ok = false;
            break;
        }
        if (st.s.failed) {
            Trap("the page decode accepted with the schedule failed", size);
        }
        twin_size += static_cast<size_t>(produced);
    }

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to show the comparison below is live
     * without waiting for a real divergence: a second binary built with this
     * defined perturbs an accepted decode, so a harness that had silently
     * stopped comparing passes where this one traps. Never define it in a
     * build whose findings are being believed. */
    if (twin_ok && oracle_ok) {
        if (twin_size != 0) {
            twin_out[twin_size - 1] = static_cast<unsigned char>(
                twin_out[twin_size - 1] ^ 0xFFu);
        } else {
            twin_size = capacity;
        }
    }
#endif

    /* The fail-open direction. A stream the reference calls corrupt that this
     * decoder accepts is the finding this target exists for, and it is the
     * only direction asserted - the file header says which reverse cases are
     * this decoder being deliberately stricter and why they are not trapped. */
    if (twin_ok && !oracle_ok) {
        Trap("the twin accepted a stream the reference refused", size);
    }
    if (!twin_ok || !oracle_ok) {
        return 0;
    }
    if (twin_size != oracle_size) {
        Trap("the two accepted and disagree on the decoded size", size);
    }
    if (twin_size != 0 &&
        std::memcmp(twin_out.get(), oracle_out.get(), twin_size) != 0) {
        Trap("the two accepted and disagree on the decoded bytes", size);
    }
    return 0;
}
