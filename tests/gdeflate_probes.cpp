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
 * WHAT THIS FILE REACHES, IN TWO KINDS, AND WHERE THE SECOND KIND ARRIVED.
 * Probes 1 to 4 are measured through the reference's public API - compress,
 * decompress, and the sizes they report - which reaches every fact whose
 * consequence is visible in what a compressor can and cannot encode. That is
 * all this file could reach while it was written, and it said so: the block
 * header rounds (D5, D6), the stored block's bit-packed body (D3, D4) and the
 * lane-order refill schedule are statements about where bits sit INSIDE a
 * page, and reading one out of an emitted page means walking the 32 substreams
 * in the order the refill schedule produced them.
 *
 * That walk exists now. src/gdeflate_schedule.h owns the bit cursor and
 * nothing else, so probes 5 to 7 open a real page with it and read the
 * layout facts off the reference's own output. They decode no symbol and
 * build no table: everything below is reachable with the cursor alone, which
 * is why it does not wait for the code-length decode (#176) or the block loop
 * (#182).
 *
 * EACH OF THE THREE CARRIES ITS OWN FALSIFIER, and that is the part to keep if
 * these are ever rewritten. A layout probe that only reads the layout it
 * believes in passes for a page whose layout is something else entirely, so
 * every one of them also performs the WRONG reading - the one a decoder built
 * on RFC 1951's shape or on a rotate-before-refill schedule would perform -
 * and requires it to disagree. Without that half the probe pins nothing.
 *
 * WHAT IS STILL NOT REACHED, so the gap is not mistaken for coverage: the
 * permutation the precode lengths are stored in (16, 17, 18, 0, 8, 7, ...) is
 * NOT pinned here and cannot be by these means. A permutation reorders the
 * same multiset of lengths, so every completeness test in probe 6 answers
 * identically under the permuted and the unpermuted reading; separating them
 * needs a decode that assigns those lengths to symbols, which is #176. The
 * dynamic block's literal/length and distance lengths, the tables built from
 * them, and every block type beyond the header round are the same case.
 *
 * No cudec code is under test. cudec has no GDeflate decoder yet; that is the
 * point of pinning these before one is written. */
#include "gdeflate_schedule.h"
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

/* The single page the reference emitted for an input that fits in one. Probes
 * 1 to 4 never needed the bytes, only the sizes; probes 5 to 7 read the bytes,
 * so this returns them. An input above 64 KiB would be split and the caller
 * would be reading a page it did not choose, so more than one page is a
 * failure here rather than a silently-taken first element. */
bool CompressOnePage(int level, const std::vector<unsigned char>& in,
                     std::vector<unsigned char>& page) {
    libdeflate_gdeflate_compressor* c =
        libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        return false;
    }
    size_t npages = 0;
    const size_t bound =
        libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    if (bound == 0 || npages != 1) {
        libdeflate_free_gdeflate_compressor(c);
        return false;
    }
    std::vector<unsigned char> pool(bound, 0);
    libdeflate_gdeflate_out_page out;
    out.data = pool.data();
    out.nbytes = bound;
    const size_t total = libdeflate_gdeflate_compress(c, in.data(), in.size(),
                                                     &out, 1);
    libdeflate_free_gdeflate_compressor(c);
    if (total == 0) {
        return false;
    }
    const unsigned char* p = static_cast<const unsigned char*>(out.data);
    page.assign(p, p + out.nbytes);
    return true;
}

/* Input the reference answers with a DYNAMIC block. Incompressible() above is
 * the wrong tool for that: the compressor answers pure noise with a stored
 * block at every level, which is what probe 7 wants and the opposite of what
 * probes 5 and 6 need. Mixing a skewed alphabet with noise gives a symbol
 * distribution worth describing but not worth a static table. Which block type
 * a given input actually draws is the reference's decision, so every probe
 * below asserts the type it read rather than assuming the one it wanted. */
std::vector<unsigned char> MixedEntropy(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        const unsigned char r = lcg.Next();
        v[i] = r < 200 ? static_cast<unsigned char>('a' + (r % 26)) : r;
    }
    return v;
}

/* GDeflate block types, the RFC 1951 values the format keeps (dossier 11.3:
 * D3 changes what a stored block CONTAINS, not the two bits that name it). */
constexpr uint32_t kBlockStored = 0;
constexpr uint32_t kBlockStatic = 1;
constexpr uint32_t kBlockDynamic = 2;

/* The 19 code-length-code symbols and the order their 3-bit lengths are
 * stored in. The permutation is the reference's own array
 * (deflate_precode_lens_permutation in the pinned fork) and is NOT pinned by
 * anything below - see the file header for why these means cannot pin it. */
constexpr int kPrecodeSyms = 19;
const int kPrecodePermutation[kPrecodeSyms] = {16, 17, 18, 0,  8,  7, 9,
                                               6,  10, 5,  11, 4,  12, 3,
                                               13, 2,  14, 1,  15};

/* A precode length is 3 bits, so 7 is the longest codeword it can describe.
 * Completeness is the Kraft equality sum(2^-len) == 1 over the non-zero
 * lengths, scaled by 2^7 so it is integer arithmetic: a float sum of a dozen
 * negative powers of two is exact today and is one refactor away from not
 * being, and this is the assertion the whole probe rests on. */
constexpr unsigned kPrecodeMaxLen = 7;
constexpr unsigned kKraftUnity = 1u << kPrecodeMaxLen;

unsigned KraftScaled(const unsigned char (&lens)[kPrecodeSyms]) {
    unsigned sum = 0;
    for (int i = 0; i < kPrecodeSyms; i++) {
        if (lens[i] != 0) {
            sum += 1u << (kPrecodeMaxLen - lens[i]);
        }
    }
    return sum;
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

    /* ---- Probe 5. D5 and D6: a block header rides substream 0 alone, and
     * every field of it sits on that one lane with no round between them.
     *
     * Dossier 11.3 D5, citing draft sections 2.2 and 6: per D3 the bitstream
     * is contiguous across blocks and the header still rides substream 0, so
     * "the header is not findable by scanning; it is reached only by decoding
     * to it". D6, citing sections 8.1, 8.2 and 9, is the consequence this
     * probe reads: one table description per block, so HLIT, HDIST and HCLEN
     * are read once, by lane 0, and every lane then decodes against what they
     * describe.
     *
     * The reading: reset to lane 0, take BFINAL (1 bit) and BTYPE (2 bits),
     * and for a dynamic block HLIT (5), HDIST (5) and HCLEN (4) - seventeen
     * bits, one lane, no advance anywhere in them.
     *
     * The falsifier is the whole probe. Seventeen bits read off ANY lane
     * produce three numbers, and all three have wide legal ranges, so a
     * reading that lands in range proves little on its own. What proves it is
     * that the OTHER reading - one round between each field, which is what a
     * decoder that mistook the header for ordinary lane-serial data would do -
     * does not survive: it is required below to disagree, either on the block
     * type or on the completeness of the precode that follows.
     *
     * Why the ladder needs it: if the header did not ride one lane, a kernel
     * would have to synchronise 32 lanes to read it. The whole warp-per-tile
     * design (#94) rests on it being warp-uniform, read once, broadcast. */
    {
        const std::vector<unsigned char> in = MixedEntropy(11, 60000);
        std::vector<unsigned char> page;
        REQUIRE(CompressOnePage(6, in, page));
        REQUIRE(RoundTrips(6, in));

        cudec_detail::GDeflateSchedule s;
        REQUIRE(cudec_detail::GDeflateInit(s, page.data(), page.size()));
        cudec_detail::GDeflateReset(s);
        const uint32_t is_final = cudec_detail::GDeflatePop(s, 1);
        const uint32_t block_type = cudec_detail::GDeflatePop(s, 2);
        cudec_detail::GDeflateEnsure(s, page.data());
        REQUIRE(!s.failed);
        REQUIRE_CTX(is_final <= 1, "BFINAL read as %u", is_final);
        REQUIRE_CTX(block_type == kBlockDynamic,
                    "expected a dynamic block from mixed-entropy input at "
                    "level 6, read BTYPE %u",
                    block_type);

        const uint32_t hlit = cudec_detail::GDeflatePop(s, 5) + 257;
        const uint32_t hdist = cudec_detail::GDeflatePop(s, 5) + 1;
        const uint32_t hclen = cudec_detail::GDeflatePop(s, 4) + 4;
        cudec_detail::GDeflateEnsure(s, page.data());
        REQUIRE(!s.failed);
        REQUIRE_CTX(hlit >= 257 && hlit <= 288, "HLIT+257 = %u", hlit);
        REQUIRE_CTX(hdist >= 1 && hdist <= 32, "HDIST+1 = %u", hdist);
        REQUIRE_CTX(hclen >= 4 && hclen <= 19, "HCLEN+4 = %u", hclen);

        /* The wrong reading: one round between each header field. */
        cudec_detail::GDeflateSchedule w;
        REQUIRE(cudec_detail::GDeflateInit(w, page.data(), page.size()));
        cudec_detail::GDeflateReset(w);
        cudec_detail::GDeflatePop(w, 1);
        cudec_detail::GDeflateAdvance(w, page.data());
        const uint32_t wrong_type = cudec_detail::GDeflatePop(w, 2);
        cudec_detail::GDeflateAdvance(w, page.data());
        const uint32_t wrong_hlit = cudec_detail::GDeflatePop(w, 5) + 257;
        cudec_detail::GDeflateAdvance(w, page.data());
        const uint32_t wrong_hdist = cudec_detail::GDeflatePop(w, 5) + 1;
        cudec_detail::GDeflateAdvance(w, page.data());
        const uint32_t wrong_hclen = cudec_detail::GDeflatePop(w, 4) + 4;
        cudec_detail::GDeflateAdvance(w, page.data());
        unsigned char wrong_lens[kPrecodeSyms];
        std::memset(wrong_lens, 0, sizeof(wrong_lens));
        for (uint32_t i = 0; i < wrong_hclen && i < kPrecodeSyms; i++) {
            wrong_lens[kPrecodePermutation[i]] =
                static_cast<unsigned char>(cudec_detail::GDeflatePop(w, 3));
            cudec_detail::GDeflateAdvance(w, page.data());
        }
        REQUIRE_CTX(w.failed || wrong_type != block_type ||
                        wrong_hlit != hlit || wrong_hdist != hdist ||
                        wrong_hclen != hclen ||
                        KraftScaled(wrong_lens) != kKraftUnity,
                    "a header read with a round between each field agreed with "
                    "the lane-0 reading in every field (type %u, HLIT %u, "
                    "HDIST %u, HCLEN %u) and produced a complete precode; this "
                    "probe would pass on a page whose header is not lane-0",
                    wrong_type, wrong_hlit, wrong_hdist, wrong_hclen);
    }

    /* ---- Probe 6. The precode round is one 3-bit length per lane, in lane
     * order, with the refill happening before the rotation.
     *
     * Dossier 11.2, "the refill schedule IS the interleaving", citing draft
     * section 5.3, and 11.3 D5 for why the round starts at lane 0. This is
     * named on #170 as "the single most load-bearing assumption in the warp
     * decode design ... and the one with no independent second source", so it
     * is the one fact here that most needs an executed check rather than a
     * cited one.
     *
     * The evidence is completeness, and it is what makes this probe possible
     * without a decoder. A precode is a canonical Huffman code, so its
     * non-zero lengths satisfy the Kraft equality exactly: sum(2^-len) == 1.
     * Reading HCLEN+4 three-bit lengths off the wrong lanes yields a length
     * vector that is arithmetically unrelated to any code, and the chance of
     * such a vector landing exactly on the equality is small - which is
     * exactly why the falsifier below is required to miss it on every page
     * rather than argued to be unlikely.
     *
     * Why the ladder needs it: the refill order has no marker in the stream to
     * check it against. Get it wrong by one lane and every symbol after it
     * decodes from bits the encoder wrote for another lane, in silence, with
     * no checksum anywhere in the format to catch it (dossier 11.4). */
    {
        struct Case {
            unsigned seed;
            int level;
            size_t size;
        };
        const Case cases[] = {{11, 1, 60000},  {11, 6, 60000},
                              {11, 12, 60000}, {23, 6, 40000},
                              {41, 12, 50000}, {57, 1, 30000}};
        int dynamic_seen = 0;
        for (const Case& c : cases) {
            const std::vector<unsigned char> in = MixedEntropy(c.seed, c.size);
            std::vector<unsigned char> page;
            REQUIRE(CompressOnePage(c.level, in, page));
            REQUIRE(RoundTrips(c.level, in));

            cudec_detail::GDeflateSchedule s;
            REQUIRE(cudec_detail::GDeflateInit(s, page.data(), page.size()));
            cudec_detail::GDeflateReset(s);
            cudec_detail::GDeflatePop(s, 1);
            const uint32_t block_type = cudec_detail::GDeflatePop(s, 2);
            cudec_detail::GDeflateEnsure(s, page.data());
            REQUIRE(!s.failed);
            /* A stored or static block carries no precode. Skipped rather
             * than failed: which type the reference picks is its decision,
             * and the count below is what refuses a run that found none. */
            if (block_type != kBlockDynamic) {
                continue;
            }
            dynamic_seen++;

            cudec_detail::GDeflatePop(s, 5);
            cudec_detail::GDeflatePop(s, 5);
            const uint32_t hclen = cudec_detail::GDeflatePop(s, 4) + 4;
            cudec_detail::GDeflateEnsure(s, page.data());
            REQUIRE(!s.failed);

            unsigned char lens[kPrecodeSyms];
            std::memset(lens, 0, sizeof(lens));
            for (uint32_t i = 0; i < hclen; i++) {
                lens[kPrecodePermutation[i]] =
                    static_cast<unsigned char>(cudec_detail::GDeflatePop(s, 3));
                cudec_detail::GDeflateAdvance(s, page.data());
            }
            REQUIRE_CTX(!s.failed,
                        "seed %u level %d: the schedule refused inside the "
                        "precode round",
                        c.seed, c.level);
            REQUIRE_CTX(KraftScaled(lens) == kKraftUnity,
                        "seed %u level %d: precode lengths read one per lane "
                        "in lane order are not a complete code (Kraft * %u = "
                        "%u, want %u), so this reading is not the layout the "
                        "reference emitted",
                        c.seed, c.level, kKraftUnity, KraftScaled(lens),
                        kKraftUnity);

            /* The falsifier: the same HCLEN+4 lengths taken from lane 0 alone,
             * which is what a decoder that read the precode as one serial bit
             * stream would do. It must not produce a complete code - a refusal
             * (lane 0 runs out of bits) counts, since both answers are "this
             * is not a code". */
            cudec_detail::GDeflateSchedule w;
            REQUIRE(cudec_detail::GDeflateInit(w, page.data(), page.size()));
            cudec_detail::GDeflateReset(w);
            cudec_detail::GDeflatePop(w, 1);
            cudec_detail::GDeflatePop(w, 2);
            cudec_detail::GDeflateEnsure(w, page.data());
            cudec_detail::GDeflatePop(w, 5);
            cudec_detail::GDeflatePop(w, 5);
            cudec_detail::GDeflatePop(w, 4);
            cudec_detail::GDeflateEnsure(w, page.data());
            unsigned char flat[kPrecodeSyms];
            std::memset(flat, 0, sizeof(flat));
            for (uint32_t i = 0; i < hclen; i++) {
                flat[kPrecodePermutation[i]] =
                    static_cast<unsigned char>(cudec_detail::GDeflatePop(w, 3));
            }
            REQUIRE_CTX(w.failed || KraftScaled(flat) != kKraftUnity,
                        "seed %u level %d: reading the precode off lane 0 "
                        "alone also produced a complete code, so the "
                        "lane-order assertion above pins nothing on this page",
                        c.seed, c.level);
        }
        /* All six drew a dynamic block when this was written, and the floor
         * is set below that on purpose: which block type the reference picks
         * is its decision, and an oracle pin that shifts one case to static
         * should not red a probe about the precode. Four is still enough that
         * the run cannot pass having read none, which is the failure this
         * guards - every assertion above is inside the loop, so a run that
         * skipped every case would be green and empty. */
        REQUIRE_CTX(dynamic_seen >= 4,
                    "only %d of %zu inputs drew a dynamic block (six did when "
                    "this was written); the precode layout is unpinned on a "
                    "run that reads too few",
                    dynamic_seen, sizeof(cases) / sizeof(cases[0]));
    }

    /* ---- Probe 7. D3 and D4: the stored block keeps neither the length
     * complement nor the byte alignment, and its body is 8-bit atoms spread
     * one per lane per round.
     *
     * Dossier 11.3 D3, citing draft section 2.2: "the one's complement of
     * length is dropped from the header and the header is no longer required
     * to be byte-aligned. As a result, data across all blocks, compressed or
     * non-compressed, forms one contiguous bit stream". D4, citing section 7:
     * the body's bytes are "fixed-size 8 bit atoms", one per lane per round,
     * so the block takes floor(len / 32) full rounds and a final round with
     * len % 32 lanes active.
     *
     * Four readings, three of them falsifiers:
     *
     *   - LEN is the 16 bits immediately after BFINAL and BTYPE, and it equals
     *     the bytes the compressor was given. Unaligned by construction: those
     *     three bits are not padded out to a byte.
     *   - Aligning to the next byte boundary first yields a different number,
     *     which is what a decoder carrying RFC 1951's alignment would read.
     *   - The 16 bits following LEN are not its one's complement, so there is
     *     no NLEN there to check LEN against. This is the half of D3 with a
     *     security consequence and the dossier states it as such: with the
     *     complement gone, "the declared length reaches the decoder unchecked
     *     by the format".
     *   - The body bytes come back identical to the input when read as one
     *     8-bit pop per lane with a round between them, and the lane index
     *     afterwards is len % 32, which is D4's round arithmetic. The page is
     *     consumed exactly at that point, and the end-of-block round that
     *     follows takes no further word - both asserted, for the reasons
     *     given where they are asserted.
     *
     * AND ONE OF THE FOUR ANSWERS A QUESTION THE DOSSIER RECORDS AS OPEN.
     * Section 11.5 lists three things not to read into the source, and the
     * third is that "the draft states no bit width for a non-compressed
     * block's length field. It says what was removed, in D3, and no more."
     * Sixteen bits is the reference's answer rather than the draft's, and the
     * LEN assertion below is where that stops being an assumption: a stored
     * block whose declared length equals the input at six different sizes is
     * a width measured, not one carried over from RFC 1951 because it happens
     * to be the same number. The dossier item stays true as written - the
     * draft still says nothing - so there is nothing to report back to #145.
     *
     * Why the ladder needs it: a stored block is the one place a decoder is
     * handed a length it must bound itself. RFC 1951 gives a decoder a free
     * consistency check on that number and GDeflate does not, so the M4
     * validation ladder has to bound the declared length against what the page
     * can still supply - which is what the reference does, and the only thing
     * standing between a declared length and a write past the end. */
    {
        /* Sizes chosen for their remainders mod 32 - 0, 1, 15 and a size that
         * is a whole number of rounds - so the final partial round is
         * exercised rather than assumed away. Noise at level 0: the reference
         * answers it with a stored block, and the assertion below refuses the
         * run if it ever stops doing so. */
        const size_t sizes[] = {20000, 20001, 20015, 20032, 4096, 4097};
        for (size_t n : sizes) {
            const std::vector<unsigned char> in =
                Incompressible(static_cast<unsigned>(n), n);
            std::vector<unsigned char> page;
            REQUIRE_CTX(CompressOnePage(0, in, page), "%zu bytes", n);
            REQUIRE(RoundTrips(0, in));

            cudec_detail::GDeflateSchedule s;
            REQUIRE(cudec_detail::GDeflateInit(s, page.data(), page.size()));
            cudec_detail::GDeflateReset(s);
            cudec_detail::GDeflatePop(s, 1);
            const uint32_t block_type = cudec_detail::GDeflatePop(s, 2);
            cudec_detail::GDeflateEnsure(s, page.data());
            REQUIRE(!s.failed);
            REQUIRE_CTX(block_type == kBlockStored,
                        "%zu bytes of noise at level 0 drew BTYPE %u, not a "
                        "stored block",
                        n, block_type);

            const uint32_t len = cudec_detail::GDeflatePop(s, 16);
            REQUIRE(!s.failed);
            REQUIRE_CTX(len == n,
                        "the 16 bits after the 3-bit header read %u for a "
                        "%zu-byte block; LEN is not sitting there unaligned",
                        len, n);

            /* Falsifier 1: no NLEN. */
            {
                cudec_detail::GDeflateSchedule w = s;
                const uint32_t next16 = cudec_detail::GDeflatePop(w, 16);
                REQUIRE_CTX(next16 != ((~len) & 0xFFFFu),
                            "the 16 bits after LEN are 0x%04x, the one's "
                            "complement of LEN (%u); the header carries an "
                            "NLEN after all",
                            next16, len);
            }

            /* Falsifier 2: no byte alignment. Five more bits take lane 0 from
             * bit 3 to bit 8, which is where a decoder holding RFC 1951's
             * alignment rule would start reading LEN. */
            {
                cudec_detail::GDeflateSchedule w;
                REQUIRE(
                    cudec_detail::GDeflateInit(w, page.data(), page.size()));
                cudec_detail::GDeflateReset(w);
                cudec_detail::GDeflatePop(w, 1);
                cudec_detail::GDeflatePop(w, 2);
                cudec_detail::GDeflateEnsure(w, page.data());
                cudec_detail::GDeflatePop(w, 5);
                const uint32_t aligned = cudec_detail::GDeflatePop(w, 16);
                REQUIRE_CTX(w.failed || aligned != len,
                            "LEN read after padding to the next byte boundary "
                            "is also %u, so this probe would pass on a "
                            "byte-aligned stored block",
                            aligned);
            }

            /* D4: one 8-bit atom per lane per round. */
            for (uint32_t i = 0; i < len; i++) {
                const uint32_t byte = cudec_detail::GDeflatePop(s, 8);
                REQUIRE_CTX(!s.failed,
                            "the schedule refused at body byte %u of %u", i,
                            len);
                REQUIRE_CTX(byte == in[i],
                            "body byte %u of %u read %u, input holds %u", i,
                            len, byte, static_cast<unsigned>(in[i]));
                cudec_detail::GDeflateAdvance(s, page.data());
                REQUIRE(!s.failed);
            }
            const uint32_t lanes = cudec_detail::kGDeflateNumStreams;
            REQUIRE_CTX(s.idx == len % lanes,
                        "after %u body bytes the current lane is %u, not %u: "
                        "the body is not floor(len/%u) full rounds plus a "
                        "final round of len%%%u lanes",
                        len, s.idx, len % lanes, lanes, lanes);

            /* The reference closes a block by walking all 32 lanes once -
             * that is where its deferred copies run - and a stored block is
             * no exception even though it defers none. On this block type
             * that round takes NO word, and the assertion says so rather
             * than passing over it: the body left every lane holding at
             * least a full packet (measured floor of 32 bits over these six
             * sizes), so no lane is under the watermark for the closing
             * round to refill. A schedule that topped a lane up early would
             * move the cursor here and be caught, which a bare "the round
             * ran" would not be.
             *
             * DELETING THIS BLOCK LEAVES THE PAGE-EXHAUSTION CHECK BELOW
             * GREEN, and that is stated rather than left for a reader to
             * assume otherwise: the cursor is already at the last word when
             * the body ends. So the exhaustion check is an end-to-end
             * statement about the body walk - no word skipped, none taken
             * twice - and not evidence about the closing round. */
            const uint64_t cursor_at_body_end = s.cursor;
            for (uint32_t k = 0; k < cudec_detail::kGDeflateNumStreams; k++) {
                cudec_detail::GDeflateAdvance(s, page.data());
                REQUIRE(!s.failed);
            }
            REQUIRE_CTX(s.cursor == cursor_at_body_end,
                        "the end-of-block round consumed %llu word(s); a "
                        "stored block leaves every lane above the watermark, "
                        "so it should consume none",
                        static_cast<unsigned long long>(s.cursor -
                                                        cursor_at_body_end));
            REQUIRE_CTX(s.cursor == s.word_count,
                        "the walk ended at word %llu of %llu",
                        static_cast<unsigned long long>(s.cursor),
                        static_cast<unsigned long long>(s.word_count));
        }
    }

    std::printf("PASS: gdeflate probes - 128-byte minimum stream, length 285 "
                "past 258 (D1), distance past 32768 (D2), pages independent, "
                "header on substream 0 (D5, D6), precode round one length per "
                "lane, stored block unaligned and complement-free (D3) as "
                "8-bit atoms (D4)\n");
    return 0;
}

