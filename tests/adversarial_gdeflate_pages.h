/* Hostile GDeflate pages, header-only so the host twin and the device gate
 * drive the SAME BYTES (issue #218). A negative that exists in only one of the
 * two proves the arm it ran on and says nothing about the other, and for a
 * checksum-less format the interesting failure is exactly the one where the
 * two arms disagree: a page the twin terminates on and the kernel does not, or
 * a rung one of them takes and the other walks past.
 *
 * EACH PAGE CARRIES THE RUNG IT MUST LAND ON, not merely "is refused". A page
 * refused for the wrong reason is a page whose guard is not the one under
 * test, and a corpus of them reads green while the branch it was written for
 * has stopped being reachable. The rungs here were read off the host twin
 * rather than predicted, and the device is then held to the twin's answer.
 *
 * WHAT IS NOT HERE. Nothing in this header is a mutant. Mutation parity
 * against the reference is a different question with a different authority -
 * the oracle decides there, and the corpus is generated - and it lives in the
 * gate that includes this file. These are CRAFTED pages reaching branches a
 * mutation corpus cannot structurally build, which is the same division
 * tests/adversarial_blocks.h draws for LZ4.
 *
 * THE CONSTRUCTIONS ARE THE LADDER LOCK'S. tests/gdeflate_ladder_lock.cpp
 * established every rung below on the host under #183; what this header adds
 * is that the same bytes are reachable from a second translation unit, so the
 * device can be held to them. It is deliberately not a second set of pages:
 * two corpora that are meant to agree and are written twice disagree the first
 * time one of them is edited. */
#ifndef CUDEC_TESTS_ADVERSARIAL_GDEFLATE_PAGES_H
#define CUDEC_TESTS_ADVERSARIAL_GDEFLATE_PAGES_H

#include "gdeflate_block.h"
#include "gdeflate_page_writer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cudec_test {

/* 64 KiB, the format's tile, fixed by the format rather than by this header
 * (docs/MASTERPLAN.md section 11.2). */
constexpr size_t kGDeflateTileBytes = 65536;

struct AdversarialPage {
    std::string name;
    std::vector<unsigned char> page;
    /* The destination capacity the page is driven at. It is part of the
     * fixture rather than a caller's choice: three of the rungs below are
     * refusals AGAINST the capacity, so the same bytes at a different capacity
     * land somewhere else. */
    size_t dst_capacity;
    /* The rung the host twin takes. The device must take the same one. */
    cudec_detail::GDeflateReject rung;
};

/* Words a schedule can be primed from, each carrying its own index so a lane
 * that read the wrong word is visible in a dump. */
inline std::vector<unsigned char> GDeflateIndexedWords(uint32_t words) {
    std::vector<unsigned char> page(static_cast<size_t>(words) * 4u, 0);
    for (uint32_t n = 0; n < words; n++) {
        page[n * 4u + 0] = static_cast<unsigned char>(n & 0xFFu);
        page[n * 4u + 1] = static_cast<unsigned char>((n >> 8) & 0xFFu);
        page[n * 4u + 2] = 0x5Au;
        page[n * 4u + 3] = 0xA5u;
    }
    return page;
}

/* A page that opens with the reserved block type. Nothing after the three
 * header bits is ever read. */
inline std::vector<unsigned char> GDeflateReservedTypePage() {
    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, 3);
    return w.Finish();
}

/* One final stored block declaring `len` bytes and supplying none of them.
 * `trim` removes the writer's zero tail, which is what makes the page run out
 * of words rather than out of capacity. */
inline std::vector<unsigned char> GDeflateStoredPage(uint32_t len, bool trim) {
    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, 0);
    w.Ensure();
    w.Push(16, len);
    std::vector<unsigned char> page = w.Finish();
    if (trim) {
        page.resize(page.size() -
                    static_cast<size_t>(kGDeflateWriterTailWords) * 4u);
    }
    return page;
}

/* Non-final stored blocks declaring zero bytes each, which is the cheapest
 * block the format has, with the tail removed. A page of them never states a
 * final block and never produces a byte, so it exercises the round loop's own
 * exhaustion rather than any output bound. */
inline std::vector<unsigned char> GDeflateNoFinalBlockPage() {
    GDeflatePageWriter w;
    for (uint32_t block = 0; block < 4096u; block++) {
        w.Reset();
        w.Push(1, 0);
        w.Push(2, 0);
        w.Ensure();
        w.Push(16, 0);
    }
    std::vector<unsigned char> page = w.Finish();
    page.resize(page.size() -
                static_cast<size_t>(kGDeflateWriterTailWords) * 4u);
    return page;
}

/* A dynamic block whose body is `symbols`, over a literal/length code that can
 * spell every symbol they use. */
inline bool GDeflatePageWithSymbols(
    const std::vector<unsigned char>& litlen_lens,
    const std::vector<unsigned char>& dist_lens,
    const std::vector<uint32_t>& symbols, std::vector<unsigned char>* page) {
    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = BuildTokens(all);
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    if (!PlanPrecode(toks, 0, precode_lens, &num_explicit)) {
        return false;
    }
    return EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                    symbols, page);
}

/* Literals only, `count` of them, so a capacity below `count` is refused on
 * the literal that does not fit and before anything is written. */
inline bool GDeflateLiteralPage(uint32_t count,
                                std::vector<unsigned char>* page) {
    std::vector<unsigned char> litlen_lens(257, 0);
    litlen_lens[65] = 1;
    litlen_lens[kEndOfBlock] = 1;
    const std::vector<unsigned char> dist_lens(1, 1);
    std::vector<uint32_t> symbols;
    for (uint32_t i = 0; i < count; i++) {
        symbols.push_back(65);
    }
    symbols.push_back(kEndOfBlock);
    return GDeflatePageWithSymbols(litlen_lens, dist_lens, symbols, page);
}

/* A dynamic block whose literal/length vector is all zeros: the empty code,
 * which the construction admits and the decode refuses on use. The body is
 * empty because there is no codeword to spell one with - the decoder reaches
 * the round loop, asks the empty table for a symbol, and refuses there. */
inline bool GDeflateEmptyTablePage(std::vector<unsigned char>* page) {
    PageBlock block;
    block.litlen_lens.assign(257, 0);
    block.dist_lens.assign(1, 0);
    std::vector<unsigned char> all(block.litlen_lens);
    all.insert(all.end(), block.dist_lens.begin(), block.dist_lens.end());
    block.toks = BuildTokens(all);
    if (!PlanPrecode(block.toks, 0, block.precode_lens, &block.num_explicit)) {
        return false;
    }
    return EmitPageBlocks(std::vector<PageBlock>(1, block), page);
}

/* A page whose first round reserves a match of the minimum length and whose
 * second round ends the block, so the deferred copy is retired in the drain,
 * which is where the distance symbol is decoded. The distance is 1, and no
 * byte has been produced when the copy runs - so with room it reaches before
 * the tile start, and without room it is refused against the capacity first.
 * Written straight against the writer because EmitPage stops at literals and
 * end-of-block. */
inline bool GDeflateMatchPage(std::vector<unsigned char>* page) {
    std::vector<unsigned char> litlen_lens(258, 0);
    litlen_lens[65] = 2;
    litlen_lens[kEndOfBlock] = 2;
    litlen_lens[257] = 1; /* length 3, no extra bits behind it */
    const std::vector<unsigned char> dist_lens(1, 1);

    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = BuildTokens(all);
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    if (!PlanPrecode(toks, 0, precode_lens, &num_explicit)) {
        return false;
    }
    cudec_detail::GDeflatePrecodeTable precode;
    if (!cudec_detail::GDeflateBuildTable(
            precode_lens, cudec_detail::kGDeflateNumPrecodeSyms, precode)) {
        return false;
    }
    cudec_detail::GDeflateLitLenTable litlen;
    if (!cudec_detail::GDeflateBuildTable(
            litlen_lens.data(), static_cast<uint32_t>(litlen_lens.size()),
            litlen)) {
        return false;
    }
    cudec_detail::GDeflateDistTable dist;
    if (!cudec_detail::GDeflateBuildTable(
            dist_lens.data(), static_cast<uint32_t>(dist_lens.size()), dist)) {
        return false;
    }

    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, kBlockTypeDynamic);
    w.Ensure();
    w.Push(5, static_cast<uint32_t>(litlen_lens.size()) - 257u);
    w.Push(5, static_cast<uint32_t>(dist_lens.size()) - 1u);
    w.Push(4, num_explicit - 4u);
    w.Ensure();
    for (uint32_t i = 0; i < num_explicit; i++) {
        w.Push(3, precode_lens[cudec_detail::GDeflatePrecodeOrder(i)]);
        w.Advance();
    }
    w.Reset();
    for (size_t i = 0; i < toks.size(); i++) {
        uint32_t code = 0;
        uint32_t len = 0;
        if (!CodewordOf(precode, precode_lens, toks[i].presym, &code, &len)) {
            return false;
        }
        w.PushCode(code, len);
        if (toks[i].extra_bits != 0) {
            w.Push(toks[i].extra_bits, toks[i].extra_value);
        }
        w.Advance();
    }

    w.Reset();
    uint32_t code = 0;
    uint32_t len = 0;
    /* Lane 0: the length symbol, then - in the same stream of bits, because a
     * lane is one sequence - the distance symbol the drain asks this lane for
     * thirty-two rounds later. */
    if (!CodewordOf(litlen, litlen_lens.data(), 257, &code, &len)) {
        return false;
    }
    w.PushCode(code, len);
    uint32_t dcode = 0;
    uint32_t dlen = 0;
    if (!CodewordOf(dist, dist_lens.data(), 0, &dcode, &dlen)) {
        return false;
    }
    w.PushCode(dcode, dlen);
    w.Advance();
    /* Lane 1: end of block, which the decoder leaves without advancing. */
    if (!CodewordOf(litlen, litlen_lens.data(), kEndOfBlock, &code, &len)) {
        return false;
    }
    w.PushCode(code, len);
    for (uint32_t i = 0; i < cudec_detail::kGDeflateNumStreams; i++) {
        w.Advance();
    }
    if (!w.ok()) {
        return false;
    }
    *page = w.Finish();
    return true;
}

/* The corpus. Returns false if any construction failed, which is a defect in
 * this header rather than a result - a caller must refuse to run on a corpus
 * it could not build, because a shorter corpus passes quietly. */
inline bool MakeAdversarialGDeflatePages(std::vector<AdversarialPage>* out) {
    out->clear();

    AdversarialPage p;

    p.name = "reserved block type";
    p.page = GDeflateReservedTypePage();
    p.dst_capacity = 64;
    p.rung = cudec_detail::kGDeflateRejectBlockTypeReserved;
    out->push_back(p);

    p.name = "stored length past the capacity";
    p.page = GDeflateStoredPage(8, false);
    p.dst_capacity = 4;
    p.rung = cudec_detail::kGDeflateRejectStoredPastCap;
    out->push_back(p);

    p.name = "stored length past what the page can supply";
    p.page = GDeflateStoredPage(4000, true);
    p.dst_capacity = kGDeflateTileBytes;
    p.rung = cudec_detail::kGDeflateRejectStoredPastPage;
    out->push_back(p);

    p.name = "page below the priming round";
    p.page = GDeflateIndexedWords(31);
    p.dst_capacity = kGDeflateTileBytes;
    p.rung = cudec_detail::kGDeflateRejectPageBelowPrimingRound;
    out->push_back(p);

    p.name = "page with a trailing partial word";
    p.page = GDeflateIndexedWords(64);
    p.page.pop_back();
    p.dst_capacity = kGDeflateTileBytes;
    p.rung = cudec_detail::kGDeflateRejectPagePartialWord;
    out->push_back(p);

    p.name = "refill past the end of the page";
    p.page = GDeflateNoFinalBlockPage();
    p.dst_capacity = kGDeflateTileBytes;
    p.rung = cudec_detail::kGDeflateRejectRefillPastEnd;
    out->push_back(p);

    p.name = "empty literal/length code used";
    if (!GDeflateEmptyTablePage(&p.page)) {
        return false;
    }
    p.dst_capacity = kGDeflateTileBytes;
    p.rung = cudec_detail::kGDeflateRejectEmptyTableUsed;
    out->push_back(p);

    std::vector<unsigned char> literals;
    if (!GDeflateLiteralPage(3, &literals)) {
        return false;
    }
    p.name = "literal past the capacity";
    p.page = literals;
    p.dst_capacity = 2;
    p.rung = cudec_detail::kGDeflateRejectLiteralPastCap;
    out->push_back(p);

    p.name = "literal past a capacity of zero";
    p.page = literals;
    p.dst_capacity = 0;
    p.rung = cudec_detail::kGDeflateRejectLiteralPastCap;
    out->push_back(p);

    std::vector<unsigned char> match;
    if (!GDeflateMatchPage(&match)) {
        return false;
    }
    p.name = "match reservation past the capacity";
    p.page = match;
    p.dst_capacity = 2;
    p.rung = cudec_detail::kGDeflateRejectMatchPastCap;
    out->push_back(p);

    p.name = "match reaching before the tile start";
    p.page = match;
    p.dst_capacity = 64;
    p.rung = cudec_detail::kGDeflateRejectMatchBeforeOutput;
    out->push_back(p);

    /* The same match page at the tile bound and one byte past it: the rung is
     * about the distance rather than about the capacity here, so it must not
     * move when the capacity crosses the format's own tile size. A refusal
     * that changed at that boundary would be a bound taken from the format's
     * convention rather than from the caller's argument. */
    p.name = "match reaching before the tile start, at the tile bound";
    p.page = match;
    p.dst_capacity = kGDeflateTileBytes;
    p.rung = cudec_detail::kGDeflateRejectMatchBeforeOutput;
    out->push_back(p);

    p.name = "match reaching before the tile start, past the tile bound";
    p.page = match;
    p.dst_capacity = kGDeflateTileBytes + 1u;
    p.rung = cudec_detail::kGDeflateRejectMatchBeforeOutput;
    out->push_back(p);

    return true;
}

}  // namespace cudec_test

#endif /* CUDEC_TESTS_ADVERSARIAL_GDEFLATE_PAGES_H */
