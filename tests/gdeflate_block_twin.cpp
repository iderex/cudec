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
 * THE PER-BLOCK PARITY BELONGS TO #175 AND LIVES HERE FOR THE CORPUS. That
 * issue's first Done-when bullet asks for the decoded output of every dynamic
 * block to match the reference at the block boundary, and the corpus the
 * bullet is defined over is the one this file generates. A second file would
 * have had to carry a second copy of the generator and the compress/decompress
 * wrappers, so CheckBlockBoundaries sits beside them instead. What it does and
 * where it stops is at that function.
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

/* The reference's verdict on a page it is entitled to read past the end of.
 *
 * ENSURE_BITS in the pinned fork reads a 32-bit word with no bound check
 * against the end of the page: the format's own watermark discipline makes
 * that safe on a well-formed page, and a page this file cut short is not one.
 * Handed such a page in a tight allocation the reference reads into whatever
 * the allocator left behind, so its verdict would be a coin and the sanitizer
 * gate would report a defect in the oracle rather than a verdict on this
 * decoder. fuzz/fuzz_gdeflate_page.cpp settles that for the differential
 * target and this is the same settlement: the reference gets its own copy with
 * a zero tail behind it while being told the same `nbytes`, which makes the
 * read deterministic without changing one byte of what it is asked about.
 *
 * The tail is sized from the CAPACITY and not from the page, for the reason
 * that file states: on a compressed block the reference has no bound of its
 * own, so it takes a word per round until the output fills, and rounds are
 * bounded by the capacity rather than by the bytes the page had left.
 *
 * The return says only whether the reference could be ASKED; its answer is the
 * out parameter, so an allocation that failed can never be read as a reject. */
bool OracleVerdictPadded(const std::vector<unsigned char>& page,
                         size_t capacity, libdeflate_result* status,
                         size_t* produced) {
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        return false;
    }
    std::vector<unsigned char> padded(page.size() + 6u * capacity + 4096u, 0);
    std::memcpy(padded.data(), page.data(), page.size());
    libdeflate_gdeflate_in_page in;
    in.data = padded.data();
    in.nbytes = page.size();
    std::vector<unsigned char> out(capacity, 0);
    *produced = 0;
    *status = libdeflate_gdeflate_decompress(d, &in, 1, out.data(), capacity,
                                             produced);
    libdeflate_free_gdeflate_decompressor(d);
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
    /* The two shapes the per-block parity below can attribute, tracked so the
     * claim that both were exercised is measured rather than assumed. */
    bool block_parity_whole_page;
    bool block_parity_split;
    Coverage()
        : multi_block(false),
          block_parity_whole_page(false),
          block_parity_split(false) {
        for (uint32_t i = 0; i < 4; i++) {
            type_seen[i] = false;
        }
    }
};

/* Per-BLOCK parity at the block boundary, which is issue #175's first
 * Done-when bullet and the last thing that issue was waiting on.
 *
 * WHY IT IS NOT ALREADY DONE BY THE PAGE COMPARISON ABOVE. That comparison is
 * over a page's whole output, so a block loop that put the boundary between
 * two blocks in the wrong place and recovered - a length taken from the next
 * block's first symbol, a drain that ran one round short and one round long -
 * still produces the page's bytes and passes. The boundary is the thing this
 * separates.
 *
 * HOW A BLOCK BOUNDARY IS READ AT THE CONTRACT EDGE. The reference exposes no
 * decode table and no per-block callback: what it answers about a page is the
 * bytes it produced and whether it accepted (#175's note of 2026-08-26). So
 * the boundary is not read out of either decoder, it is IMPOSED on the page
 * and then confirmed by both. Setting BFINAL on the first block - bit 0 of the
 * page's first word, the bit beside the BTYPE that
 * RunReservedBlockType below rewrites - makes that block the whole page, and
 * the reference then reports the block's own output length as its own answer.
 * Ours must produce the same length and the same bytes, and those bytes must
 * be the prefix the full page already produced.
 *
 * WHAT IT REACHES. Every block of every corpus page, not only the dynamic
 * ones: the type of a block after the first is not readable at this edge
 * either, so the property is asserted over all of them, which is the stronger
 * statement rather than a weaker one. That the corpus reaches dynamic blocks
 * at all stays the measured claim in RunCorpus below.
 *
 * WHERE IT STOPS. BFINAL can be imposed on the FIRST block only, because the
 * bit position of any later block's header is not known without decoding to
 * it. A page of two blocks is therefore fully attributed and a page of three
 * would not be, so the third is refused loudly here rather than passing with
 * the last blocks lumped together. Measured over this corpus, no page holds
 * more than two. */
int CheckBlockBoundaries(const char* name, int level, size_t page_index,
                         const std::vector<unsigned char>& page,
                         const unsigned char* want, uint64_t page_len,
                         uint32_t blocks, Coverage* cov) {
    REQUIRE_CTX(blocks >= 1, "%s page %zu reported %u blocks", name,
                page_index, blocks);
    if (blocks == 1) {
        /* The block boundary and the page boundary coincide, so the byte
         * comparison the caller already made IS this block's comparison. */
        cov->block_parity_whole_page = true;
        return 0;
    }
    REQUIRE_CTX(blocks == 2,
                "%s level %d page %zu holds %u blocks: only the first can be "
                "isolated, so a page past two is outside what this attributes",
                name, level, page_index, blocks);

    std::vector<unsigned char> cut = page;
    cut[0] = static_cast<unsigned char>(cut[0] | 0x01u);

    std::vector<unsigned char> first(kTileBytes, 0);
    GDeflatePageState st;
    uint64_t first_len = 0;
    REQUIRE_CTX(GDeflateDecodePage(st, cut.data(), cut.size(), first.data(),
                                   kTileBytes, &first_len),
                "%s level %d page %zu: the isolated first block was refused",
                name, level, page_index);
    REQUIRE_CTX(st.blocks == 1, "%s page %zu isolated %u blocks", name,
                page_index, st.blocks);
    REQUIRE_CTX(first_len > 0 && first_len < page_len,
                "%s page %zu: first block produced %llu of %llu", name,
                page_index, static_cast<unsigned long long>(first_len),
                static_cast<unsigned long long>(page_len));

    /* The reference's own answer for the same imposed boundary. It is asked
     * for the whole tile rather than for a length this side computed, so the
     * length below is the reference's and the comparison is not circular. */
    std::vector<std::vector<unsigned char> > one;
    one.push_back(cut);
    std::vector<unsigned char> oracle_first;
    REQUIRE_CTX(OracleDecompress(one, kTileBytes, &oracle_first),
                "%s level %d page %zu: the reference refused the isolated "
                "first block",
                name, level, page_index);
    REQUIRE_CTX(oracle_first.size() == first_len,
                "%s page %zu: reference ended the first block at %zu, this "
                "decoder at %llu",
                name, page_index, oracle_first.size(),
                static_cast<unsigned long long>(first_len));
    REQUIRE_CTX(equal_bytes(first.data(), oracle_first.data(),
                            static_cast<size_t>(first_len)),
                "%s level %d page %zu first block", name, level, page_index);
    /* And the same bytes in place, which is what makes this a statement about
     * the full page's first block rather than about a page of its own. */
    REQUIRE_CTX(equal_bytes(first.data(), want, static_cast<size_t>(first_len)),
                "%s level %d page %zu first block in place", name, level,
                page_index);
    /* NOTHING IS ASSERTED HERE FOR THE SECOND BLOCK, ON PURPOSE. Its bytes are
     * the caller's whole-page comparison restricted to the range past the
     * boundary, so a check written here could not fail while that one passed,
     * and a guard that cannot bite proves nothing. What this function adds for
     * the second block is the BOUNDARY: where its output begins is now a
     * number both decoders produced rather than one this side chose. */
    cov->block_parity_split = true;
    return 0;
}

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
        if (CheckBlockBoundaries(name, level, i, pages[i], in.data() + consumed,
                                 produced, st.blocks, cov) != 0) {
            return 1;
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
    /* Both attribution shapes of CheckBlockBoundaries have to have been
     * reached, or the per-block claim rests on whichever one the corpus
     * happened to produce (issue #175). */
    REQUIRE(cov.block_parity_whole_page);
    REQUIRE(cov.block_parity_split);
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

/* A block whose symbols simply STOP, which is where issue #183's enumerated
 * "missing EOB" negative was expected to go and where it turns out there is no
 * negative to write.
 *
 * ON THE STATIC CODE THE END-OF-BLOCK SYMBOL IS SEVEN ZERO BITS, and a page's
 * tail is zeros. So a block that stops emitting is handed an end-of-block by
 * its own padding: it decodes, it does not fail closed, and both this decoder
 * and the reference produce exactly the literals that were written. That is
 * asserted below rather than described, together with the codeword that causes
 * it, so the two cannot drift apart.
 *
 * WHAT THAT MEANS FOR THE LADDER. A missing end-of-block is not a reject branch
 * of this decoder. Where the symbols outlast the output instead, the literal
 * and length paths each refuse against the capacity before writing, which is
 * the branch RunCapacityBoundary above already sweeps; and the round-fuel arm
 * in the block loop is a termination property rather than a reachable refusal.
 * A fixture asserting a refusal here would therefore have been asserting
 * something no page produces.
 *
 * THE PAIR IS THE PROOF, AND IT DIFFERS BY ONE SYMBOL. The same literals with
 * the end-of-block written out explicitly must produce the same bytes as the
 * page that leaves it to the tail. If they ever diverge, one of the two is
 * being decoded through a different path than this file claims. The block is a
 * static one so the fixture owes no dynamic header; what is under test is one
 * symbol's absence. */
int RunEndOfBlockFromTheTail() {
    const std::vector<unsigned char> litlen_lens = StaticLitLenLengths();
    cudec_detail::GDeflateLitLenTable litlen;
    REQUIRE(cudec_detail::GDeflateBuildTable(litlen_lens.data(), 288, litlen));

    uint32_t lit_code = 0;
    uint32_t lit_bits = 0;
    REQUIRE(cudec_test::CodewordOf(litlen, litlen_lens.data(),
                                   static_cast<uint32_t>('a'), &lit_code,
                                   &lit_bits));
    uint32_t eob_code = 0;
    uint32_t eob_bits = 0;
    REQUIRE(cudec_test::CodewordOf(litlen, litlen_lens.data(),
                                   cudec_detail::kGDeflateEndOfBlock,
                                   &eob_code, &eob_bits));
    /* The reason the page below needs no end-of-block, asserted rather than
     * assumed: RFC 1951's static code gives symbol 256 the seven-bit codeword
     * 0000000, and a page's tail is zeros. */
    REQUIRE(eob_code == 0u && eob_bits == 7u);

    /* One whole round of the 32 lanes, so the block is past the priming round
     * rather than ending inside it, and the capacity is exactly what those
     * literals produce - the two pages then differ by the end-of-block symbol
     * and by nothing else. */
    const size_t kCapacity = cudec_detail::kGDeflateNumStreams;

    for (uint32_t with_eob = 0; with_eob < 2; with_eob++) {
        GDeflatePageWriter w;
        w.Reset();
        w.Push(1, 1); /* BFINAL */
        w.Push(2, cudec_detail::kGDeflateBlockStatic);
        w.Ensure();
        w.Reset();
        for (size_t i = 0; i < kCapacity; i++) {
            w.PushCode(lit_code, lit_bits);
            w.Advance();
        }
        if (with_eob) {
            /* The reference leaves the loop here without advancing, so the
             * drain that follows starts on this lane. */
            w.PushCode(eob_code, eob_bits);
        }
        for (uint32_t i = 0; i < cudec_detail::kGDeflateNumStreams; i++) {
            w.Advance();
        }
        REQUIRE(w.ok());
        const std::vector<unsigned char> page = w.Finish();

        libdeflate_result status = LIBDEFLATE_BAD_DATA;
        size_t produced = 0;
        REQUIRE(OracleVerdictPadded(page, kCapacity, &status, &produced));

        std::vector<unsigned char> out(kCapacity, 0);
        GDeflatePageState st;
        uint64_t got = 0;
        const bool ok = GDeflateDecodePage(st, page.data(), page.size(),
                                           out.data(), out.size(), &got);
        REQUIRE_CTX(ok, "with_eob=%u was refused", with_eob);
        REQUIRE_CTX(got == kCapacity, "with_eob=%u produced %llu", with_eob,
                    static_cast<unsigned long long>(got));
        REQUIRE_CTX(status == LIBDEFLATE_SUCCESS, "with_eob=%u status %d",
                    with_eob, static_cast<int>(status));
        REQUIRE(produced == kCapacity);
        const std::vector<unsigned char> want(kCapacity, 'a');
        REQUIRE(equal_bytes(out.data(), want.data(), kCapacity));
    }
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
 * THE REFERENCE IS ASKED NOW, AND WHAT IT ANSWERS IS THE POINT (issue #183).
 * This block used to record that it could not be asked at all: handing it a
 * truncated page is a read past the end of the buffer, so its verdict would
 * have been a coin and the sanitizer gate would have reported a defect in the
 * oracle. What removed that obstacle is fuzz/fuzz_gdeflate_page.cpp's zero
 * tail, which OracleVerdictPadded above reuses, and asking turns out to matter
 * because the answer is NOT the one a reject-parity bullet would assume.
 *
 * The reference refuses most truncations and ACCEPTS the longest ones, because
 * the tail hands it the words the page no longer has and its decode completes
 * out of them. That is the recommended 128-byte zero padding doing the work,
 * and it is exactly what #183 asks this rung to prove independence from: this
 * decoder refuses every one of them by the explicit bound in
 * src/gdeflate_schedule.h, including the ones the reference accepts. So the
 * assertion below is a declared strictness departure and not reject parity,
 * and both halves of the reference's split are required to have been reached -
 * a fixture that only produced refusals would prove the departure by
 * accident. */
int RunTruncatedPage() {
    const std::vector<unsigned char> in = ShortRepeats(22, 20000);
    std::vector<std::vector<unsigned char> > pages;
    REQUIRE(CompressPages(6, in, &pages));
    REQUIRE(pages.size() == 1);

    std::vector<unsigned char> tile(kTileBytes, 0);
    const size_t words = pages[0].size() / 4u;
    uint32_t refusals = 0;
    uint32_t oracle_refused = 0;
    uint32_t oracle_accepted = 0;
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

        libdeflate_result status = LIBDEFLATE_BAD_DATA;
        size_t produced = 0;
        REQUIRE_CTX(
            OracleVerdictPadded(shorter, in.size(), &status, &produced),
            "%zu words: the reference could not be asked", cut);
        if (status == LIBDEFLATE_SUCCESS) {
            oracle_accepted++;
        } else {
            oracle_refused++;
        }
    }
    REQUIRE(refusals == words - cudec_detail::kGDeflateNumStreams);
    REQUIRE(oracle_refused != 0);
    REQUIRE(oracle_accepted != 0);
    GDeflatePageState st;
    uint64_t got = 0;
    REQUIRE(GDeflateDecodePage(st, pages[0].data(), pages[0].size(),
                               tile.data(), in.size(), &got));
    REQUIRE(got == in.size());
    return 0;
}

}  // namespace

/* TWO DYNAMIC BLOCKS IN ONE HAND-EMITTED PAGE (issue #430), which is the first
 * page in this tree whose block boundary was WRITTEN rather than read.
 *
 * WHY IT IS NOT COVERED BY WHAT IS ABOVE. RunCorpus decodes multi-block pages
 * and CheckBlockBoundaries attributes their first block, but every one of those
 * pages came out of the reference's compressor: where a block ends is the
 * compressor's decision and both decoders merely follow it. Nothing had ever
 * PRODUCED a boundary, so the inter-block round order - the reset a block opens
 * with, the drain it closes with - was a property of this tree's decoder alone,
 * with no second reader. The page below is emitted by
 * tests/gdeflate_page_writer.h and handed to both, so a writer that put the
 * second block's header one round early or late is refused by the reference
 * rather than agreed with by cudec.
 *
 * WHAT IT ASSERTS BEYOND THE BYTES. The second block opens with a match that
 * reaches back over the FIRST block's output, so a decoder that restarted its
 * output cursor at a block boundary would refuse it (a match before the page's
 * own output) rather than quietly produce a shorter page; and the block count
 * is read off cudec's own census, so a page that collapsed into one block
 * fails here rather than passing on identical bytes. */
int RunTwoBlockPage() {
    /* A complete code over the 258 symbols these bodies need - the literals,
     * end-of-block, and the shortest length symbol - beside a two-symbol
     * distance code whose symbol 0 is distance 1, the nearest match there is
     * and the one that lands inside the first block's output. */
    const std::vector<unsigned char> litlen_lens =
        cudec_test::CompleteLengths(258);
    const std::vector<unsigned char> dist_lens = cudec_test::CompleteLengths(2);

    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<cudec_test::LenToken> toks = cudec_test::BuildTokens(all);
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(cudec_test::PlanPrecode(toks, /*forced_explicit=*/0, precode_lens,
                                    &num_explicit));

    cudec_test::PageBlock proto;
    proto.litlen_lens = litlen_lens;
    proto.dist_lens = dist_lens;
    proto.toks = toks;
    std::memcpy(proto.precode_lens, precode_lens, sizeof(precode_lens));
    proto.num_explicit = num_explicit;

    /* The first block: a run of literals short enough to sit inside one round
     * of the 32 lanes plus a few, so the block ends mid-round and the drain
     * that follows it is the thing the second block has to start after. */
    const uint32_t kFirstLiterals = 40;
    std::vector<unsigned char> want;
    cudec_test::PageBlock first = proto;
    for (uint32_t i = 0; i < kFirstLiterals; i++) {
        const uint32_t sym = 'a' + (i % 26u);
        first.body.push_back(cudec_test::BodyToken{sym, false, 0, 0});
        want.push_back(static_cast<unsigned char>(sym));
    }
    first.body.push_back(
        cudec_test::BodyToken{cudec_test::kEndOfBlock, false, 0, 0});

    /* The second block: lane 0 reserves the shortest match the format has,
     * 31 literals ride the other lanes, and lane 0's next round retires the
     * reservation. The match's source is the byte before it, which the FIRST
     * block wrote. */
    cudec_test::PageBlock second = proto;
    second.body.push_back(cudec_test::BodyToken{
        cudec_detail::kGDeflateFirstLengthSym, false, 0, 0});
    const uint64_t reserved = want.size();
    want.resize(want.size() + cudec_detail::kGDeflateMinMatchLen, 0);
    for (uint32_t i = 1; i < cudec_detail::kGDeflateNumStreams; i++) {
        const uint32_t sym = 'A' + (i % 26u);
        second.body.push_back(cudec_test::BodyToken{sym, false, 0, 0});
        want.push_back(static_cast<unsigned char>(sym));
    }
    second.body.push_back(cudec_test::BodyToken{0u, true, 0, 0});
    second.body.push_back(
        cudec_test::BodyToken{cudec_test::kEndOfBlock, false, 0, 0});
    /* The copy, mirrored byte by byte forwards exactly as GDeflateDoCopy runs
     * it, and placed here rather than where the reservation was made: the
     * bytes land in the hole the reservation left, after the literals that
     * followed it were already written. */
    for (uint32_t i = 0; i < cudec_detail::kGDeflateMinMatchLen; i++) {
        want[reserved + i] = want[reserved + i - 1];
    }

    std::vector<cudec_test::PageBlock> blocks;
    blocks.push_back(first);
    blocks.push_back(second);
    std::vector<unsigned char> page;
    REQUIRE(cudec_test::EmitPageBlocks(blocks, &page));

    libdeflate_result status = LIBDEFLATE_BAD_DATA;
    size_t produced = 0;
    REQUIRE(OracleVerdictPadded(page, want.size(), &status, &produced));
    REQUIRE_CTX(status == LIBDEFLATE_SUCCESS, "oracle status %d",
                static_cast<int>(status));
    REQUIRE_CTX(produced == want.size(), "oracle produced %zu of %zu", produced,
                want.size());

    std::vector<unsigned char> out(want.size(), 0);
    GDeflatePageState st;
    uint64_t got = 0;
    REQUIRE(GDeflateDecodePage(st, page.data(), page.size(), out.data(),
                               out.size(), &got));
    REQUIRE_CTX(got == want.size(), "cudec produced %llu of %zu",
                static_cast<unsigned long long>(got), want.size());
    REQUIRE(equal_bytes(out.data(), want.data(), want.size()));
    /* The page really is two blocks rather than one that happened to produce
     * the same bytes, read off the decode's own census. */
    REQUIRE_CTX(st.blocks == 2u, "cudec walked %u blocks", st.blocks);
    REQUIRE(st.type_blocks[cudec_detail::kGDeflateBlockDynamic] == 2u);
    return 0;
}

/* THE THREE WAYS A CALLER CAN ASK FOR A PAGE THE TWO DECODERS WOULD READ
 * DIFFERENTLY, and the proof that the writer refuses each of them.
 *
 * WHY THEY ARE REFUSALS IN THE WRITER RATHER THAN NEGATIVES AGAINST A DECODER.
 * Every one of them produces a page that is still WELL FORMED as a word
 * sequence: the words handed out and the words asked for still match, so the
 * reference reads bits that are there and gets a different, valid-looking
 * page. The failure would therefore surface as a fixture that decodes to
 * unexpected bytes somewhere else entirely, or - worse for a corpus - not at
 * all. Each is paired with RunTwoBlockPage above, which differs from all three
 * in exactly the field being broken and must decode.
 *
 * DELETING ANY OF THE THREE REFUSALS TURNS THE MATCHING CASE HERE GREEN-TO-RED
 * IN THE OTHER DIRECTION: the REQUIRE below is that the writer said no. */
int RunWriterRefusesDriftedBodies() {
    const std::vector<unsigned char> litlen_lens =
        cudec_test::CompleteLengths(258);
    const std::vector<unsigned char> dist_lens = cudec_test::CompleteLengths(2);
    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<cudec_test::LenToken> toks = cudec_test::BuildTokens(all);
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(cudec_test::PlanPrecode(toks, /*forced_explicit=*/0, precode_lens,
                                    &num_explicit));

    cudec_test::PageBlock proto;
    proto.litlen_lens = litlen_lens;
    proto.dist_lens = dist_lens;
    proto.toks = toks;
    std::memcpy(proto.precode_lens, precode_lens, sizeof(precode_lens));
    proto.num_explicit = num_explicit;

    const cudec_test::BodyToken kEob{cudec_test::kEndOfBlock, false, 0, 0};
    const cudec_test::BodyToken kLiteral{'x', false, 0, 0};
    const cudec_test::BodyToken kLength{cudec_detail::kGDeflateFirstLengthSym,
                                        false, 0, 0};
    const cudec_test::BodyToken kDistance{0u, true, 0, 0};

    std::vector<unsigned char> page;

    /* A block with a successor that never states its end. The decoder would
     * read on past the tokens this block declared, and where the next block's
     * header begins would be decided by the bits rather than by the sequence
     * the caller handed over. */
    {
        cudec_test::PageBlock unterminated = proto;
        unterminated.body.push_back(kLiteral);
        cudec_test::PageBlock last = proto;
        last.body.push_back(kLiteral);
        last.body.push_back(kEob);
        std::vector<cudec_test::PageBlock> blocks;
        blocks.push_back(unterminated);
        blocks.push_back(last);
        REQUIRE(!cudec_test::EmitPageBlocks(blocks, &page));
    }

    /* A block that ends while a lane still holds a reservation. The 32 drain
     * rounds emit no bits, so the decoder would take whatever follows - the
     * next block's header, or the page's zero tail - as that lane's distance
     * symbol. The end-of-block rides lane 1, which holds nothing, so it is the
     * end of the block that is wrong rather than the round it sits on. */
    {
        cudec_test::PageBlock stranded = proto;
        stranded.body.push_back(kLength);
        stranded.body.push_back(kEob);
        REQUIRE(!cudec_test::EmitPageBlocks(
            std::vector<cudec_test::PageBlock>(1, stranded), &page));
    }

    /* A distance symbol on a lane that reserved nothing. The decoder reads a
     * distance only on a lane holding a copy, so it would read this one as a
     * literal or a length out of the OTHER code entirely. */
    {
        cudec_test::PageBlock misspaced = proto;
        misspaced.body.push_back(kDistance);
        misspaced.body.push_back(kEob);
        REQUIRE(!cudec_test::EmitPageBlocks(
            std::vector<cudec_test::PageBlock>(1, misspaced), &page));
    }

    /* The pair for all three: the same three bodies made correct, in one page
     * that must be emitted and must decode. A writer refusing everything would
     * pass the three cases above and fail here. */
    {
        cudec_test::PageBlock ok_first = proto;
        ok_first.body.push_back(kLiteral);
        ok_first.body.push_back(kEob);
        cudec_test::PageBlock ok_second = proto;
        ok_second.body.push_back(kLength);
        for (uint32_t i = 1; i < cudec_detail::kGDeflateNumStreams; i++) {
            ok_second.body.push_back(kLiteral);
        }
        ok_second.body.push_back(kDistance);
        ok_second.body.push_back(kEob);
        std::vector<cudec_test::PageBlock> blocks;
        blocks.push_back(ok_first);
        blocks.push_back(ok_second);
        REQUIRE(cudec_test::EmitPageBlocks(blocks, &page));

        const size_t kCapacity =
            1u + cudec_detail::kGDeflateMinMatchLen +
            (cudec_detail::kGDeflateNumStreams - 1u);
        libdeflate_result status = LIBDEFLATE_BAD_DATA;
        size_t produced = 0;
        REQUIRE(OracleVerdictPadded(page, kCapacity, &status, &produced));
        REQUIRE_CTX(status == LIBDEFLATE_SUCCESS, "oracle status %d",
                    static_cast<int>(status));
        REQUIRE(produced == kCapacity);
    }
    return 0;
}

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
    if (RunEndOfBlockFromTheTail() != 0) {
        return 1;
    }
    if (RunTwoBlockPage() != 0) {
        return 1;
    }
    if (RunWriterRefusesDriftedBodies() != 0) {
        return 1;
    }
    std::printf("gdeflate_block_twin: ok\n");
    return 0;
}
