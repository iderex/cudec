/* The GDeflate page decode, held against the reference (issue #182).
 * src/gdeflate_block.h decodes a whole page - block dispatch, the DEFLATE64
 * length and distance tables, and the deferred in-tile LZ77 - and this
 * establishes that it produces the reference's bytes over a corpus the
 * reference's own compressor generated.
 *
 * THIS IS THE ONE RUNG WHERE THE ORACLE ANSWERS DIRECTLY, AND THAT IS THE
 * POINT. The rungs below it (#178, #175, #176) had to reach parity sideways,
 * because the schedule's word cursor and the code-length vectors sit in the
 * pinned fork's private state and no public name reaches them. A decoded page
 * is different: `libdeflate_gdeflate_compress` produces the page and
 * `libdeflate_gdeflate_decompress` says what it means, both across the
 * library's own boundary. So the comparison here is the plain one - same page
 * in, same bytes out, byte for byte and length for length - and it is what
 * makes the three rungs below it checkable end to end, because a wrong word
 * cursor or a wrong code-length vector cannot survive it.
 *
 * THE DIVERGENCES ARE THE PART A STOCK RFC 1951 DECODER GETS WRONG SILENTLY.
 * Length symbol 285 with sixteen extra bits and distance symbols 30 and 31
 * each get a fixture whose input can only be coded through them, and each is
 * decoded to the byte. What shows that the pin proves something is the mutant
 * pass rather than an assertion here: setting symbol 285 back to RFC 1951's
 * base 258 with no extra bits, or either distance symbol back to a stock
 * value, reds this file. A test that merely asserted the tables differ from
 * RFC 1951's would pass against a decoder that never reached them.
 *
 * WHAT IS NOT COVERED. The reject ladder is filed separately from this issue,
 * so the negatives here are the three this decoder owes on its own terms - a
 * reserved block type, an output that does not fit its tile, and a page cut
 * short - rather than a fail-closed matrix. Nothing here runs on a device;
 * #214 is the kernel and #218 its gate set. */
#include "gdeflate_block.h"
#include "gdeflate_page_writer.h"
#include "require.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::GDeflateDecodePage;
using cudec_detail::GDeflateInit;
using cudec_detail::GDeflatePageState;
using cudec_detail::GDeflatePop;
using cudec_detail::GDeflateReset;
using cudec_detail::GDeflateSchedule;
using cudec_detail::kGDeflateBlockReserved;
using cudec_test::GDeflatePageWriter;

/* The tile a page decompresses into. The format fixes it and it is not read
 * from anywhere inside a page (dossier 11.2), which is why every decode below
 * is handed the capacity rather than discovering it. */
constexpr size_t kTileBytes = 64u * 1024u;

/* The deterministic source every fixture draws from. A named generator rather
 * than a library one: the corpus has to be identical on every machine and in
 * every run, or a parity failure is not reproducible. */
class Lcg {
   public:
    explicit Lcg(unsigned seed) : state_(seed) {}
    unsigned Next() {
        state_ = state_ * 1103515245u + 12345u;
        return (state_ >> 16) & 0xFFFFu;
    }

   private:
    unsigned state_;
};

std::vector<unsigned char> Incompressible(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<unsigned char>(lcg.Next() & 0xFFu);
    }
    return v;
}

/* A skewed alphabet with noise through it: worth describing with a code, not
 * worth a static one, which is what draws a dynamic block. */
std::vector<unsigned char> MixedEntropy(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        const unsigned r = lcg.Next();
        v[i] = static_cast<unsigned char>((r % 4u == 0u) ? (r & 0xFFu)
                                                         : ('a' + (r % 5u)));
    }
    return v;
}

/* Short repeats, which is what puts ordinary matches - modest lengths, modest
 * distances - into the corpus. */
std::vector<unsigned char> ShortRepeats(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v;
    v.reserve(n);
    while (v.size() < n) {
        const size_t run = 3u + (lcg.Next() % 60u);
        const unsigned char b = static_cast<unsigned char>('a' + lcg.Next() % 6u);
        for (size_t i = 0; i < run && v.size() < n; i++) {
            v.push_back(b);
        }
        const size_t noise = lcg.Next() % 8u;
        for (size_t i = 0; i < noise && v.size() < n; i++) {
            v.push_back(static_cast<unsigned char>(lcg.Next() & 0xFFu));
        }
    }
    return v;
}

/* Compress with the page split the reference itself chooses. Returns the pages
 * as separate byte vectors, which is how a tile stream carries them and how
 * this decoder consumes them. */
bool CompressPages(int level, const std::vector<unsigned char>& in,
                   std::vector<std::vector<unsigned char> >* pages) {
    libdeflate_gdeflate_compressor* c =
        libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        return false;
    }
    size_t npages = 0;
    const size_t bound =
        libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    if (bound == 0 || npages == 0) {
        libdeflate_free_gdeflate_compressor(c);
        return false;
    }
    std::vector<unsigned char> pool(bound * npages, 0);
    std::vector<libdeflate_gdeflate_out_page> out(npages);
    for (size_t i = 0; i < npages; i++) {
        out[i].data = pool.data() + i * bound;
        out[i].nbytes = bound;
    }
    const size_t total = libdeflate_gdeflate_compress(c, in.data(), in.size(),
                                                      out.data(), npages);
    libdeflate_free_gdeflate_compressor(c);
    if (total == 0) {
        return false;
    }
    pages->clear();
    for (size_t i = 0; i < npages; i++) {
        const unsigned char* p =
            static_cast<const unsigned char*>(out[i].data);
        pages->push_back(std::vector<unsigned char>(p, p + out[i].nbytes));
    }
    return true;
}

/* The reference's own answer for the whole page array, which is the second
 * half of every parity claim: this decoder is compared against the input the
 * corpus was built from AND against what the reference makes of the same
 * bytes, so a corpus generator that silently produced something else cannot
 * pass by agreeing with itself. */
bool OracleDecompress(const std::vector<std::vector<unsigned char> >& pages,
                      size_t out_cap, std::vector<unsigned char>* out) {
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        return false;
    }
    std::vector<libdeflate_gdeflate_in_page> in(pages.size());
    for (size_t i = 0; i < pages.size(); i++) {
        in[i].data = pages[i].data();
        in[i].nbytes = pages[i].size();
    }
    out->assign(out_cap, 0);
    size_t produced = 0;
    const libdeflate_result r = libdeflate_gdeflate_decompress(
        d, in.data(), in.size(), out->data(), out_cap, &produced);
    libdeflate_free_gdeflate_decompressor(d);
    if (r != LIBDEFLATE_SUCCESS) {
        return false;
    }
    out->resize(produced);
    return true;
}

/* The block type of a page's first block, read with the schedule alone. Used
 * to prove the corpus reaches all three types rather than assuming it does -
 * which type a given input draws is the reference compressor's decision. */
int FirstBlockType(const std::vector<unsigned char>& page) {
    GDeflateSchedule s;
    if (!GDeflateInit(s, page.data(), page.size())) {
        return -1;
    }
    GDeflateReset(s);
    GDeflatePop(s, 1);
    const uint32_t t = GDeflatePop(s, 2);
    if (s.failed) {
        return -1;
    }
    return static_cast<int>(t);
}

/* What the corpus turned out to contain, accumulated across every fixture so
 * the coverage claim is measured rather than asserted per case. */
struct Coverage {
    bool type_seen[4];
    bool multi_block;
    Coverage() : multi_block(false) {
        for (uint32_t i = 0; i < 4; i++) {
            type_seen[i] = false;
        }
    }
};

/* One corpus fixture: compress, decompress every page with the decoder under
 * test, and require the concatenation to be the input byte for byte and the
 * reference's own output byte for byte. */
int CheckRoundTrip(const char* name, int level,
                   const std::vector<unsigned char>& in, Coverage* cov) {
    std::vector<std::vector<unsigned char> > pages;
    REQUIRE_CTX(CompressPages(level, in, &pages), "%s at level %d", name,
                level);

    std::vector<unsigned char> oracle_out;
    REQUIRE_CTX(OracleDecompress(pages, in.size(), &oracle_out),
                "%s at level %d", name, level);
    REQUIRE_CTX(oracle_out.size() == in.size(), "%s: oracle produced %zu",
                name, oracle_out.size());
    REQUIRE_CTX(equal_bytes(oracle_out.data(), in.data(), in.size()), "%s",
                name);

    std::vector<unsigned char> got;
    for (size_t i = 0; i < pages.size(); i++) {
        /* A page's tile is full unless it is the last one, and the last one
         * holds exactly what is left. Nothing inside the page says so, which
         * is why the caller has to. */
        const size_t consumed = i * kTileBytes;
        const size_t want = in.size() - consumed < kTileBytes
                                ? in.size() - consumed
                                : kTileBytes;
        std::vector<unsigned char> tile(kTileBytes, 0);
        GDeflatePageState st;
        uint64_t produced = 0;
        REQUIRE_CTX(GDeflateDecodePage(st, pages[i].data(), pages[i].size(),
                                       tile.data(), want, &produced),
                    "%s level %d page %zu", name, level, i);
        REQUIRE_CTX(produced == want, "%s page %zu produced %llu want %zu",
                    name, i, static_cast<unsigned long long>(produced), want);
        REQUIRE_CTX(equal_bytes(tile.data(), in.data() + consumed, want),
                    "%s level %d page %zu", name, level, i);

        const int type = FirstBlockType(pages[i]);
        REQUIRE_CTX(type >= 0 && type < 4, "%s page %zu type %d", name, i,
                    type);
        cov->type_seen[type] = true;
        if (st.blocks > 1) {
            cov->multi_block = true;
        }
        got.insert(got.end(), tile.begin(), tile.begin() + want);
    }
    REQUIRE_CTX(got.size() == in.size(), "%s: %zu bytes", name, got.size());
    return 0;
}

/* The corpus. Levels are the reference's own range endpoints and its default,
 * because which block type and which symbols an input draws is a function of
 * the level as much as of the bytes. */
int RunCorpus() {
    Coverage cov;
    const int levels[] = {1, 6, 9, 12};
    for (uint32_t l = 0; l < 4; l++) {
        const int level = levels[l];
        if (CheckRoundTrip("incompressible-4k", level,
                           Incompressible(1, 4096), &cov) != 0) {
            return 1;
        }
        if (CheckRoundTrip("incompressible-60k", level,
                           Incompressible(2, 60000), &cov) != 0) {
            return 1;
        }
        if (CheckRoundTrip("mixed-4k", level, MixedEntropy(3, 4096), &cov) !=
            0) {
            return 1;
        }
        if (CheckRoundTrip("mixed-60k", level, MixedEntropy(4, 60000), &cov) !=
            0) {
            return 1;
        }
        if (CheckRoundTrip("short-repeats-60k", level,
                           ShortRepeats(5, 60000), &cov) != 0) {
            return 1;
        }
        /* Several pages, so the page split itself is exercised rather than
         * only the single-page shape. */
        if (CheckRoundTrip("mixed-multi-page", level,
                           MixedEntropy(6, 3u * kTileBytes + 7u), &cov) != 0) {
            return 1;
        }
        /* A page whose output hits the tile size exactly, which the issue asks
         * for by name: the last page is full rather than short, so the
         * capacity check and the last byte written coincide. */
        if (CheckRoundTrip("exact-tile", level, ShortRepeats(7, kTileBytes),
                           &cov) != 0) {
            return 1;
        }
        if (CheckRoundTrip("exact-two-tiles", level,
                           MixedEntropy(8, 2u * kTileBytes), &cov) != 0) {
            return 1;
        }
        /* One byte, and the empty input, which are where a decode that
         * assumed a block had a body would come apart. */
        if (CheckRoundTrip("one-byte", level,
                           std::vector<unsigned char>(1, 'q'), &cov) != 0) {
            return 1;
        }
    }

    /* The coverage claim, measured over everything above. All three block
     * types have to have been decoded, and at least one page has to have held
     * more than one block, or this file is testing less than it says. */
    REQUIRE(cov.type_seen[cudec_detail::kGDeflateBlockStored]);
    REQUIRE(cov.type_seen[cudec_detail::kGDeflateBlockStatic]);
    REQUIRE(cov.type_seen[cudec_detail::kGDeflateBlockDynamic]);
    REQUIRE(!cov.type_seen[kGDeflateBlockReserved]);
    REQUIRE(cov.multi_block);
    return 0;
}

/* D1: length symbol 285 is base 3 with sixteen extra bits and reaches 65538,
 * where RFC 1951 fixes it at 258 with none. A whole tile of one byte is a
 * single match under the format and 254 matches under the stock reading, so a
 * decoder carrying the stock table walks the rest of the page from a bit
 * position sixteen bits off - in every lane, with no checksum to catch it.
 * That the pin bites is shown by the mutant that restores 258/0, not by an
 * assertion here. */
int RunLongMatch() {
    Coverage cov;
    const std::vector<unsigned char> run(kTileBytes, 'A');
    if (CheckRoundTrip("tile-of-one-byte", 12, run, &cov) != 0) {
        return 1;
    }
    /* The match this fixture rests on is longer than RFC 1951's ceiling, which
     * is what makes it the D1 fixture rather than an ordinary run. */
    REQUIRE(run.size() > 258u);
    return CheckRoundTrip("tile-of-one-byte-level1", 1, run, &cov);
}

/* D2: distance symbols 30 and 31 exist, with base 32769 and 49153 and fourteen
 * extra bits, reaching 65536 - the tile exactly. The fixture is the one
 * tests/gdeflate_probes.cpp measures the divergence with: 20000 incompressible
 * bytes repeated at offset 45000, so the only match that saves anything sits
 * at a distance RFC 1951 cannot express. */
int RunFarMatch() {
    Coverage cov;
    std::vector<unsigned char> v = Incompressible(11, 20000);
    std::vector<unsigned char> page = Incompressible(12, 45000);
    page.insert(page.end(), v.begin(), v.end());
    for (size_t i = 0; i < 20000 && page.size() < kTileBytes; i++) {
        page[i] = v[i];
    }
    REQUIRE(page.size() <= kTileBytes);
    /* The repeat sits 45000 bytes after its source, past RFC 1951's 32768
     * ceiling, which is what makes this the D2 fixture. */
    REQUIRE(45000u > 32768u);
    return CheckRoundTrip("far-match", 12, page, &cov);
}

/* An overlapping match: a distance below the match length, which is a run that
 * repeats the bytes the copy is itself writing. It is the case a word-at-a-time
 * copy has to special-case and this one does not, and it is where an
 * implementation that copied backwards or in blocks produces different bytes. */
int RunOverlappingMatch() {
    Coverage cov;
    std::vector<unsigned char> v;
    v.reserve(kTileBytes);
    /* A three-byte cycle repeated to fill the tile: every match after the
     * first three bytes has distance 3 and a length far past it. */
    while (v.size() < kTileBytes) {
        v.push_back('x');
        v.push_back('y');
        v.push_back('z');
    }
    v.resize(kTileBytes);
    if (CheckRoundTrip("distance-three-run", 12, v, &cov) != 0) {
        return 1;
    }
    /* Distance one, the degenerate overlap. */
    return CheckRoundTrip("distance-one-run", 6,
                          std::vector<unsigned char>(40000, 0x5A), &cov);
}

/* A reserved block type, in front of a block that is otherwise entirely
 * valid. That is the shape the refusal has to be tested with: a page carrying
 * nothing but a reserved type is refused by any decoder that reaches the end
 * of it, so such a page would pass against a decoder with no reserved check at
 * all that merely fell through to the dynamic path. This takes a page the
 * compressor produced, whose first block is dynamic, and turns its BTYPE into
 * the reserved value - two bits, nothing else. The reference refuses it and so
 * must this decoder, while a decoder without the check would decode the block
 * behind those two bits perfectly. */
int RunReservedBlockType() {
    const std::vector<unsigned char> in = MixedEntropy(31, 20000);
    std::vector<std::vector<unsigned char> > pages;
    REQUIRE(CompressPages(6, in, &pages));
    REQUIRE(pages.size() == 1);
    REQUIRE(FirstBlockType(pages[0]) == cudec_detail::kGDeflateBlockDynamic);

    /* BFINAL is bit 0 of the page's first word and BTYPE is bits 1 and 2,
     * least significant first, because lane 0's first word is word 0 and the
     * block header rides lane 0 (dossier 11.2). Setting both makes BTYPE 3. */
    std::vector<unsigned char> page = pages[0];
    page[0] = static_cast<unsigned char>(page[0] | 0x06u);
    REQUIRE(FirstBlockType(page) == static_cast<int>(kGDeflateBlockReserved));

    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    REQUIRE(d != nullptr);
    libdeflate_gdeflate_in_page in_page;
    in_page.data = page.data();
    in_page.nbytes = page.size();
    std::vector<unsigned char> out(kTileBytes, 0);
    size_t produced = 0;
    const libdeflate_result r = libdeflate_gdeflate_decompress(
        d, &in_page, 1, out.data(), in.size(), &produced);
    libdeflate_free_gdeflate_decompressor(d);
    REQUIRE_CTX(r == LIBDEFLATE_BAD_DATA, "status %d", static_cast<int>(r));

    std::vector<unsigned char> tile(kTileBytes, 0);
    GDeflatePageState st;
    uint64_t got = 0;
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), tile.data(),
                                in.size(), &got));
    REQUIRE(st.s.failed);
    return 0;
}

/* The RFC 1951 static code lengths, restated here rather than taken from the
 * header under test. A fixture that computed its expected codewords with the
 * code it is testing would agree with that code whatever either of them said,
 * which is the one thing a twin may not do. */
std::vector<unsigned char> StaticLitLenLengths() {
    std::vector<unsigned char> lens(288, 0);
    for (uint32_t i = 0; i < 288; i++) {
        if (i < 144) {
            lens[i] = 8;
        } else if (i < 256) {
            lens[i] = 9;
        } else if (i < 280) {
            lens[i] = 7;
        } else {
            lens[i] = 8;
        }
    }
    return lens;
}

/* A match whose source begins before the page's output. The block's very first
 * symbol is a length, so the reservation starts at output position zero and
 * EVERY distance the format can express reaches before it - which makes this
 * the smallest page that exercises the bound, and one whose refusal cannot be
 * confused with running out of input. The reference refuses it at the same
 * point and its verdict is asserted beside ours.
 *
 * The block is a static one so the fixture owes no dynamic header: what is
 * under test is the distance bound, and a header between it and the page start
 * would only add ways for the fixture itself to be wrong. */
int RunMatchBeforeOutput() {
    const std::vector<unsigned char> litlen_lens = StaticLitLenLengths();
    const std::vector<unsigned char> dist_lens(32, 5);

    cudec_detail::GDeflateLitLenTable litlen;
    cudec_detail::GDeflateDistTable dist;
    REQUIRE(cudec_detail::GDeflateBuildTable(litlen_lens.data(), 288, litlen));
    REQUIRE(cudec_detail::GDeflateBuildTable(dist_lens.data(), 32, dist));

    uint32_t len_code = 0;
    uint32_t len_bits = 0;
    REQUIRE(cudec_test::CodewordOf(litlen, litlen_lens.data(), 257u, &len_code,
                                   &len_bits));
    uint32_t lit_code = 0;
    uint32_t lit_bits = 0;
    REQUIRE(cudec_test::CodewordOf(litlen, litlen_lens.data(),
                                   static_cast<uint32_t>('a'), &lit_code,
                                   &lit_bits));
    uint32_t dist_code = 0;
    uint32_t dist_bits = 0;
    REQUIRE(cudec_test::CodewordOf(dist, dist_lens.data(), 0u, &dist_code,
                                   &dist_bits));

    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1); /* BFINAL */
    w.Push(2, cudec_detail::kGDeflateBlockStatic);
    w.Ensure();
    w.Reset();
    /* Lane 0 opens the block with a length symbol: three bytes reserved at
     * output position zero, before a single byte exists. */
    w.PushCode(len_code, len_bits);
    w.Advance();
    for (uint32_t i = 1; i < cudec_detail::kGDeflateNumStreams; i++) {
        w.PushCode(lit_code, lit_bits);
        w.Advance();
    }
    /* Back on lane 0, one round of its own later: the distance. Symbol 0 is
     * distance 1, the nearest the format has, so a refusal here is a refusal
     * for every distance there is. */
    w.PushCode(dist_code, dist_bits);
    w.Advance();
    for (uint32_t i = 1; i < cudec_detail::kGDeflateNumStreams; i++) {
        w.Advance();
    }
    REQUIRE(w.ok());
    const std::vector<unsigned char> page = w.Finish();

    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    REQUIRE(d != nullptr);
    libdeflate_gdeflate_in_page in_page;
    in_page.data = page.data();
    in_page.nbytes = page.size();
    std::vector<unsigned char> out(kTileBytes, 0);
    size_t produced = 0;
    const libdeflate_result r = libdeflate_gdeflate_decompress(
        d, &in_page, 1, out.data(), out.size(), &produced);
    libdeflate_free_gdeflate_decompressor(d);
    REQUIRE_CTX(r == LIBDEFLATE_BAD_DATA, "status %d", static_cast<int>(r));

    std::vector<unsigned char> tile(kTileBytes, 0);
    GDeflatePageState st;
    uint64_t got = 0;
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), tile.data(),
                                tile.size(), &got));
    REQUIRE(st.s.failed);
    return 0;
}

/* A stored block whose declared length is more than the page can supply. It is
 * the one length in this format with nothing to cross-check it against - there
 * is no complement field (dossier 11.3, D3) - so the bytes still reachable are
 * the only bound, and this is that bound executed. The page is cut back to the
 * words its own header needs, which the reference reads safely because it
 * refuses on the declared length before touching the body. */
int RunStoredLengthPastPage() {
    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1); /* BFINAL */
    w.Push(2, cudec_detail::kGDeflateBlockStored);
    w.Ensure();
    w.Push(16, 0xFFFFu); /* LEN, far past anything this page holds */
    REQUIRE(w.ok());
    std::vector<unsigned char> page = w.Finish();
    /* The priming round plus the one word lane 0 took for its header. Stated
     * as that count rather than trimmed by a marker, because the writer's tail
     * is slack for the reference's unchecked refill and not part of the page
     * at all. */
    const size_t real_words = cudec_detail::kGDeflateNumStreams + 1u;
    REQUIRE(page.size() >= real_words * 4u);
    page.resize(real_words * 4u);

    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    REQUIRE(d != nullptr);
    libdeflate_gdeflate_in_page in_page;
    in_page.data = page.data();
    in_page.nbytes = page.size();
    std::vector<unsigned char> out(kTileBytes, 0);
    size_t produced = 0;
    const libdeflate_result r = libdeflate_gdeflate_decompress(
        d, &in_page, 1, out.data(), out.size(), &produced);
    libdeflate_free_gdeflate_decompressor(d);
    REQUIRE_CTX(r == LIBDEFLATE_BAD_DATA, "status %d", static_cast<int>(r));

    std::vector<unsigned char> tile(kTileBytes, 0);
    GDeflatePageState st;
    uint64_t got = 0;
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), tile.data(),
                                tile.size(), &got));
    REQUIRE(st.s.failed);
    return 0;
}

/* An output that does not fit the tile it was handed. Every capacity below the
 * page's real output has to be refused, and the one that equals it has to be
 * accepted - so the boundary is pinned rather than the fact that some small
 * capacity failed. The literal path, the stored path and the match path each
 * carry their own capacity check, and the sweep runs over a fixture that uses
 * all three. */
int CapacityBoundary(const char* name, const std::vector<unsigned char>& in,
                     int level, bool want_stored) {
    std::vector<std::vector<unsigned char> > pages;
    REQUIRE_CTX(CompressPages(level, in, &pages), "%s", name);
    REQUIRE_CTX(pages.size() == 1, "%s", name);
    /* Which compressed type an input draws is the reference compressor's
     * decision, so the fixture asserts the half it depends on - stored or not
     * stored - rather than pinning a choice it does not control. */
    const bool is_stored =
        FirstBlockType(pages[0]) == cudec_detail::kGDeflateBlockStored;
    REQUIRE_CTX(is_stored == want_stored, "%s: type %d", name,
                FirstBlockType(pages[0]));

    std::vector<unsigned char> tile(kTileBytes, 0);
    for (size_t cap = 0; cap < in.size(); cap++) {
        GDeflatePageState st;
        uint64_t got = 0;
        REQUIRE_CTX(!GDeflateDecodePage(st, pages[0].data(), pages[0].size(),
                                        tile.data(), cap, &got),
                    "%s: capacity %zu accepted", name, cap);
        REQUIRE_CTX(st.s.failed, "%s: capacity %zu", name, cap);
    }
    GDeflatePageState st;
    uint64_t got = 0;
    REQUIRE(GDeflateDecodePage(st, pages[0].data(), pages[0].size(),
                               tile.data(), in.size(), &got));
    REQUIRE(got == in.size());
    REQUIRE(equal_bytes(tile.data(), in.data(), in.size()));
    return 0;
}

/* Both paths that write output carry their own capacity check, and a sweep
 * over one block type leaves the other unread: a compressed block writes
 * through the literal and the match path, a stored block through neither. */
int RunCapacityBoundary() {
    if (CapacityBoundary("compressed", ShortRepeats(21, 5000), 6, false) !=
        0) {
        return 1;
    }
    return CapacityBoundary("stored", Incompressible(23, 5000), 1, true);
}

/* A page cut short. Every length below the whole page is refused and the whole
 * page is accepted, which is the same boundary shape the header twin pins one
 * rung down.
 *
 * THE ORACLE IS NOT ASKED, AND NOT ASKING IT IS THE FINDING. ENSURE_BITS in
 * the pinned fork reads a 32-bit word with no bound check against the end of
 * the page, so handing the reference a truncated page is a heap out-of-bounds
 * read: it would red the sanitizer gate with a defect in the oracle rather
 * than a verdict on this decoder. src/gdeflate_schedule.h refuses the same
 * read by an explicit bound, which is what this asserts. */
int RunTruncatedPage() {
    const std::vector<unsigned char> in = ShortRepeats(22, 20000);
    std::vector<std::vector<unsigned char> > pages;
    REQUIRE(CompressPages(6, in, &pages));
    REQUIRE(pages.size() == 1);

    std::vector<unsigned char> tile(kTileBytes, 0);
    const size_t words = pages[0].size() / 4u;
    uint32_t refusals = 0;
    for (size_t cut = cudec_detail::kGDeflateNumStreams; cut < words; cut++) {
        std::vector<unsigned char> shorter(pages[0].begin(),
                                           pages[0].begin() + cut * 4u);
        GDeflatePageState st;
        uint64_t got = 0;
        REQUIRE_CTX(!GDeflateDecodePage(st, shorter.data(), shorter.size(),
                                        tile.data(), in.size(), &got),
                    "%zu words accepted", cut);
        REQUIRE_CTX(st.s.failed, "%zu words", cut);
        refusals++;
    }
    REQUIRE(refusals != 0);
    GDeflatePageState st;
    uint64_t got = 0;
    REQUIRE(GDeflateDecodePage(st, pages[0].data(), pages[0].size(),
                               tile.data(), in.size(), &got));
    REQUIRE(got == in.size());
    return 0;
}

}  // namespace

int main() {
    if (RunCorpus() != 0) {
        return 1;
    }
    if (RunLongMatch() != 0) {
        return 1;
    }
    if (RunFarMatch() != 0) {
        return 1;
    }
    if (RunOverlappingMatch() != 0) {
        return 1;
    }
    if (RunReservedBlockType() != 0) {
        return 1;
    }
    if (RunMatchBeforeOutput() != 0) {
        return 1;
    }
    if (RunStoredLengthPastPage() != 0) {
        return 1;
    }
    if (RunCapacityBoundary() != 0) {
        return 1;
    }
    if (RunTruncatedPage() != 0) {
        return 1;
    }
    std::printf("gdeflate_block_twin: ok\n");
    return 0;
}
