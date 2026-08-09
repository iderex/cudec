/* Dossier facts about the GDeflate format, executed against the reference
 * instead of read (issue #170). The dossier in docs/MASTERPLAN.md section 11
 * is assembled from an expired IETF draft, a reference implementation and a
 * Vulkan extension, with no canonical specification behind any of them, so
 * anything the M4 correctness ladder rests on and that is only asserted in
 * prose is an unverified assumption. The format carries no checksum anywhere
 * (section 11.4), so a wrong belief about it is not caught downstream by a
 * CRC: it becomes a bound enforced against the wrong number.
 *
 * Each probe below names the dossier item it pins and the draft section that
 * item cites. A probe with no citation would be an opinion.
 *
 * WHAT THIS FILE CAN AND CANNOT REACH, STATED SO THE GAP IS NOT MISTAKEN FOR
 * COVERAGE. Everything here is measured through the reference's public API -
 * compress, decompress, and the sizes they report. That reaches every fact
 * whose consequence is visible in what a compressor can and cannot encode. It
 * does not reach a fact about how the bits are laid out inside a page: the
 * block header rounds (D5, D6), the stored block's bit-packed body (D4) and
 * the lane-order refill schedule are all statements about the substream
 * machine, and reading them out of an emitted page means implementing that
 * machine, which is the CPU twin (#178, #176, #182). Those probes are owed
 * and are named on #170 rather than being approximated here.
 *
 * No cudec code is under test. cudec has no GDeflate decoder yet; that is the
 * point of pinning these before one is written. */
#include "require.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/* 64 KiB, the page, fixed by the format rather than by this test
 * (docs/MASTERPLAN.md section 11.2, draft section 4). */
constexpr size_t kPageBytes = 65536;

/* Draft section 5.3, quoted in dossier 11.2: the first round always loads 32
 * consecutive words, "as a result, any valid GDeflate bit stream cannot be
 * smaller than 4 * 32 = 128 bytes". */
constexpr size_t kMinStreamBytes = 128;

/* A deterministic byte source. Not std::rand: the probes below compare two
 * compressions of related inputs, and the comparison is only evidence if the
 * bytes are the same on every machine and every run. */
class Lcg {
  public:
    explicit Lcg(unsigned seed) : state_(seed) {}
    unsigned char Next() {
        state_ = state_ * 1103515245u + 12345u;
        return static_cast<unsigned char>(state_ >> 16);
    }

  private:
    unsigned state_;
};

std::vector<unsigned char> Incompressible(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = lcg.Next();
    }
    return v;
}

/* Compress with the page split the reference itself asks for, and report
 * both the total compressed size and the page count. Returns 0 on a
 * compression failure, which every caller treats as fatal. */
size_t Compress(int level, const std::vector<unsigned char>& in,
                size_t* npages_out) {
    libdeflate_gdeflate_compressor* c =
        libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        return 0;
    }
    size_t npages = 0;
    const size_t bound =
        libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    if (bound == 0 || npages == 0) {
        libdeflate_free_gdeflate_compressor(c);
        return 0;
    }
    std::vector<unsigned char> pool(bound, 0);
    std::vector<libdeflate_gdeflate_out_page> pages(npages);
    const size_t per = bound / npages;
    for (size_t i = 0; i < npages; i++) {
        pages[i].data = pool.data() + i * per;
        pages[i].nbytes = per;
    }
    const size_t total =
        libdeflate_gdeflate_compress(c, in.data(), in.size(), pages.data(),
                                     npages);
    libdeflate_free_gdeflate_compressor(c);
    if (npages_out != nullptr) {
        *npages_out = npages;
    }
    return total;
}

/* Every probe compresses; every compression is also decompressed, so a probe
 * can never draw a conclusion from a stream the reference would not accept
 * back. Returns false if the round trip is not byte-identical. */
bool RoundTrips(int level, const std::vector<unsigned char>& in) {
    libdeflate_gdeflate_compressor* c =
        libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        return false;
    }
    size_t npages = 0;
    const size_t bound =
        libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    std::vector<unsigned char> pool(bound, 0);
    std::vector<libdeflate_gdeflate_out_page> out(npages);
    const size_t per = bound / npages;
    for (size_t i = 0; i < npages; i++) {
        out[i].data = pool.data() + i * per;
        out[i].nbytes = per;
    }
    const bool compressed =
        libdeflate_gdeflate_compress(c, in.data(), in.size(), out.data(),
                                     npages) != 0;
    libdeflate_free_gdeflate_compressor(c);
    if (!compressed) {
        return false;
    }
    std::vector<libdeflate_gdeflate_in_page> pages(npages);
    for (size_t i = 0; i < npages; i++) {
        pages[i].data = out[i].data;
        pages[i].nbytes = out[i].nbytes;
    }
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        return false;
    }
    std::vector<unsigned char> back(in.size(), 0);
    size_t got = 0;
    const libdeflate_result r = libdeflate_gdeflate_decompress(
        d, pages.data(), pages.size(), back.data(), back.size(), &got);
    libdeflate_free_gdeflate_decompressor(d);
    return r == LIBDEFLATE_SUCCESS && got == in.size() &&
           std::memcmp(back.data(), in.data(), in.size()) == 0;
}

}  // namespace

int main() {
    /* ---- Probe 1. The minimum valid stream size is 128 bytes.
     *
     * Dossier 11.2, "the refill schedule is the interleaving", citing draft
     * section 5.3: the first round loads 32 consecutive words to initialise
     * all 32 lane states, so no valid stream is shorter than 4 * 32 bytes.
     *
     * Why the ladder needs it: it is the one length a page can be rejected on
     * before anything is decoded, and it is the cheapest guard in M4. Dossier
     * 11.5 also records that the same figure is NOT a recommended trailing
     * pad, which is the belief this probe exists to keep from creeping back:
     * what 128 bounds is the stream, not the slack after it. */
    {
        const std::vector<unsigned char> one(1, 'x');
        size_t npages = 0;
        const size_t total = Compress(6, one, &npages);
        REQUIRE(total != 0);
        REQUIRE_CTX(npages == 1, "%zu pages for one byte", npages);
        REQUIRE_CTX(total >= kMinStreamBytes,
                    "one byte compressed to %zu, below the %zu-byte floor the "
                    "32-word first round forces",
                    total, kMinStreamBytes);
        REQUIRE(RoundTrips(6, one));

        /* And the floor is a floor rather than a coincidence of that one
         * input: a 32-byte input, one byte per lane, is still above it. */
        const std::vector<unsigned char> thirty_two(32, 0);
        const size_t small = Compress(6, thirty_two, nullptr);
        REQUIRE(small != 0);
        REQUIRE_CTX(small >= kMinStreamBytes, "32 bytes compressed to %zu",
                    small);
        REQUIRE(RoundTrips(6, thirty_two));
    }

    /* ---- Probe 2. D1: length code 285 carries 16 extra bits and reaches
     * 65538, rather than RFC 1951's fixed 258.
     *
     * Dossier 11.3 D1, citing draft section 2.1. Measured from outside as
     * what a compressor can encode: a run of identical bytes is one copy
     * under D1 whatever its length, and ceil((n - 3) / 258) copies under the
     * RFC 1951 cap.
     *
     * The evidence is that the compressed size does not grow with the run.
     * Under the cap, going from a 261-byte run to a 65536-byte run adds
     * ceil(65533 / 258) - ceil(258 / 258) = 254 - 1 = 253 copies. A copy is a
     * length symbol plus a distance symbol, and no Huffman code is shorter
     * than one bit, so those 253 copies cost at least 506 bits = 63 bytes
     * however favourably they are coded. The slack allowed below is 32 bytes,
     * which is half of that: the assertion cannot be satisfied by a decoder
     * capped at 258 and is not brittle against a pin whose entropy coding
     * shifts by a few bytes.
     *
     * Why the ladder needs it: a twin built on RFC 1951's length table
     * decodes 285 as 258 and then walks the rest of the page from the wrong
     * bit position, in every lane, with no checksum to catch it. */
    {
        std::vector<unsigned char> run(kPageBytes, 'A');
        const std::vector<unsigned char> short_run(261, 'A');

        const size_t big = Compress(12, run, nullptr);
        const size_t small = Compress(12, short_run, nullptr);
        REQUIRE(big != 0);
        REQUIRE(small != 0);
        REQUIRE_CTX(big <= small + 32,
                    "a %zu-byte run of one byte compressed to %zu and a "
                    "261-byte run to %zu; the %zu-byte growth is what a "
                    "258-byte length cap would force, so code 285 is not "
                    "reaching past 258 here",
                    run.size(), big, small, big - small);
        REQUIRE(RoundTrips(12, run));

        /* The same measurement across the range, so the flatness is the
         * claim rather than one lucky pair. Each step past 258 would add at
         * least one more copy under the cap. */
        const size_t lengths[] = {261, 519, 1035, 4107, 16395};
        for (size_t n : lengths) {
            const std::vector<unsigned char> v(n, 'A');
            const size_t got = Compress(12, v, nullptr);
            REQUIRE(got != 0);
            REQUIRE_CTX(got <= small + 32, "run of %zu compressed to %zu", n,
                        got);
            REQUIRE(RoundTrips(12, v));
        }
    }

    /* ---- Probe 3. D2: distance codes 30 and 31 exist and reach back past
     * RFC 1951's 32768-byte ceiling, up to 65536.
     *
     * Dossier 11.3 D2, citing draft section 2.1. Measured as what the
     * compressor can reach: one page holds 20000 bytes of incompressible
     * data at offset 45000 that repeat the page's first 20000 bytes, so the
     * only match that saves anything sits at distance 45000. Under RFC 1951's
     * 32768 ceiling that match is unreachable and the region has to be coded
     * literally.
     *
     * The control is the same page with the repeat removed, which fixes
     * everything except the reachability of that one distance, so the
     * difference between the two is the match and nothing else.
     *
     * Why the ladder needs it: D2 is what makes a 64 KiB page self-contained,
     * and a twin with RFC 1951's distance table treats codes 30 and 31 as
     * invalid - it would reject streams the reference produces. */
    {
        std::vector<unsigned char> with_repeat = Incompressible(7, kPageBytes);
        const std::vector<unsigned char> control = Incompressible(7, kPageBytes);
        std::memcpy(with_repeat.data() + 45000, with_repeat.data(), 20000);

        size_t npages = 0;
        const size_t matched = Compress(12, with_repeat, &npages);
        const size_t plain = Compress(12, control, nullptr);
        REQUIRE(matched != 0);
        REQUIRE(plain != 0);
        REQUIRE_CTX(npages == 1, "%zu pages for one page of input", npages);
        REQUIRE_CTX(plain > matched + 15000,
                    "a 20000-byte repeat at distance 45000 saved only %zd "
                    "bytes (%zu against %zu); a distance past 32768 is not "
                    "being reached",
                    static_cast<ptrdiff_t>(plain) -
                        static_cast<ptrdiff_t>(matched),
                    matched, plain);
        REQUIRE(RoundTrips(12, with_repeat));
    }

    /* ---- Probe 4. Pages are completely independent: no match crosses a page
     * boundary.
     *
     * Dossier 11.2, citing draft section 4: pages are "completely
     * independent, enabling compression and decompression to operate on
     * multiple pages in parallel". Measured as the compressor declining the
     * best match there is: two pages whose contents are byte-identical
     * compress to exactly twice one of them, because the second page cannot
     * see the first.
     *
     * The input is incompressible inside a page, so the only compression
     * available anywhere is the cross-page repeat. If any part of it were
     * taken, the total would collapse towards half.
     *
     * Why the ladder needs it: page independence is the whole basis of the
     * decode geometry - one warp per page, no ordering between pages, and a
     * match bound that may be enforced against the page rather than against
     * the output buffer. If a match could reach into the previous page, that
     * bound would be wrong in the direction that reads as a correct decode. */
    {
        const std::vector<unsigned char> page = Incompressible(99, kPageBytes);
        std::vector<unsigned char> twice(2 * kPageBytes);
        std::memcpy(twice.data(), page.data(), kPageBytes);
        std::memcpy(twice.data() + kPageBytes, page.data(), kPageBytes);

        size_t npages = 0;
        const size_t both = Compress(12, twice, &npages);
        const size_t one = Compress(12, page, nullptr);
        REQUIRE(both != 0);
        REQUIRE(one != 0);
        REQUIRE_CTX(npages == 2, "%zu pages for two pages of input", npages);
        REQUIRE_CTX(both == 2 * one,
                    "two identical pages compressed to %zu, against %zu for "
                    "one page alone; a match crossed the page boundary",
                    both, one);
        REQUIRE(RoundTrips(12, twice));
    }

    std::printf("PASS: gdeflate probes - 128-byte minimum stream, length 285 "
                "past 258 (D1), distance past 32768 (D2), pages "
                "independent\n");
    return 0;
}
