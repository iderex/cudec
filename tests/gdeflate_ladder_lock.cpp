/* The run-time half of the GDeflate reject-ladder lock (issue #183).
 *
 * The configure-time half in tests/CMakeLists.txt reads the three headers and
 * requires every declared branch of `enum GDeflateReject` to be named by
 * exactly one refusal site. That says nothing about whether any input reaches
 * the site. This file is the other half: one negative per declared branch,
 * each asserting the branch it lands on, and a final sweep requiring every
 * branch in the enumeration either to have been reached here or to be declared
 * unreachable with its argument written beside it.
 *
 * WHAT THE PAIR BUYS, AND IT IS THE STANDING GATE THE ISSUE ASKS FOR. Adding a
 * rung costs an enumerator, a refusal site and a negative: the enumerator with
 * no site reds the configure, the site with no enumerator reds it too, and the
 * pair of them with no negative reds this test. A rung that borrows a
 * neighbouring enumerator to avoid all three is what the exactly-one count
 * refuses next door.
 *
 * WHY IT IS ONE BINARY RATHER THAN A SECTION IN EACH TWIN. The ladder is one
 * enumeration across three headers because one sticky field carries it, and
 * the negatives that reach it are spread over four twins today - the
 * code-length rungs live in gdeflate_header_twin.cpp, the construction rungs
 * in gdeflate_tables_twin.cpp, the block rungs in gdeflate_block_twin.cpp. A
 * coverage claim assembled from four processes is a claim about whichever of
 * them ran, so the sweep needs one process that can reach the whole
 * enumeration. That is this file, and the cost is that its fixtures are its
 * own rather than the ones the twins carry - it proves the ladder is
 * reachable, and the twins remain where a rung is held against the reference.
 *
 * NO ORACLE HERE, DELIBERATELY. Every assertion below is about which rung the
 * decoder in this tree takes, which is a question about this tree and not
 * about the reference. Reject parity with the reference is what the twins
 * establish and it is not restated here. */
#include "gdeflate_block.h"
#include "gdeflate_page_writer.h"
#include "require.h"

#include <cstdio>
#include <vector>

namespace {

using cudec_detail::GDeflateAdvance;
using cudec_detail::GDeflateBuildTable;
using cudec_detail::GDeflateDecodePage;
using cudec_detail::GDeflateDecodeSymbol;
using cudec_detail::GDeflateInit;
using cudec_detail::GDeflateLitLenTable;
using cudec_detail::GDeflatePageState;
using cudec_detail::GDeflatePop;
using cudec_detail::GDeflateReject;
using cudec_detail::GDeflateRemove;
using cudec_detail::GDeflateSchedule;
using cudec_detail::kGDeflateNoSymbol;
using cudec_detail::kGDeflateNumStreams;
using cudec_detail::kGDeflateRejectCount;
using cudec_detail::kGDeflateRejectNone;
using cudec_test::CompleteLengths;
using cudec_test::EmitPage;
using cudec_test::GDeflatePageWriter;
using cudec_test::LenToken;
using cudec_test::PlanPrecode;

bool g_covered[kGDeflateRejectCount] = {false};

/* Mark and assert in one act. A negative that lands on a NEIGHBOURING rung
 * would otherwise fill the sweep below while testing something else, which is
 * the failure mode a coverage array has: it counts arrivals, and only the
 * expected branch makes an arrival evidence of anything. */
bool Landed(const GDeflateSchedule& s, GDeflateReject expected) {
    if (!s.failed || s.reject != expected) {
        std::fprintf(stderr,
                     "FAIL ladder: failed=%d reject=%d, wanted reject=%d\n",
                     static_cast<int>(s.failed), static_cast<int>(s.reject),
                     static_cast<int>(expected));
        return false;
    }
    g_covered[expected] = true;
    return true;
}

/* A construction rung has no schedule to record into, so it names itself into
 * the slot its caller supplied. */
bool NamedInSlot(bool built, GDeflateReject slot, GDeflateReject expected) {
    if (built || slot != expected) {
        std::fprintf(stderr, "FAIL ladder: built=%d slot=%d, wanted slot=%d\n",
                     static_cast<int>(built), static_cast<int>(slot),
                     static_cast<int>(expected));
        return false;
    }
    g_covered[expected] = true;
    return true;
}

std::vector<unsigned char> IndexedWords(uint32_t count) {
    std::vector<unsigned char> page(static_cast<size_t>(count) * 4u);
    for (uint32_t n = 0; n < count; n++) {
        page[n * 4u + 0] = static_cast<unsigned char>(n);
        page[n * 4u + 1] = 0;
        page[n * 4u + 2] = 0;
        page[n * 4u + 3] = 0xA5;
    }
    return page;
}

/* ---- src/gdeflate_schedule.h ---- */

int ScheduleRungs() {
    const std::vector<unsigned char> page = IndexedWords(64);

    /* A trailing partial word is not a word any lane could be handed. */
    GDeflateSchedule partial;
    REQUIRE(!GDeflateInit(partial, page.data(), page.size() - 1));
    REQUIRE(Landed(partial, cudec_detail::kGDeflateRejectPagePartialWord));

    /* Draft section 5.3: the priming round needs 32 words. */
    const std::vector<unsigned char> short_page = IndexedWords(31);
    GDeflateSchedule below;
    REQUIRE(!GDeflateInit(below, short_page.data(), short_page.size()));
    REQUIRE(Landed(below, cudec_detail::kGDeflateRejectPageBelowPrimingRound));

    /* The page ends under a lane that still wants a word. Drained lane by lane
     * rather than jumped to, so the rung is reached the way a page reaches
     * it. */
    const uint32_t words = 40;
    const std::vector<unsigned char> ending = IndexedWords(words);
    GDeflateSchedule out_of_words;
    REQUIRE(GDeflateInit(out_of_words, ending.data(), ending.size()));
    for (uint64_t guard = 0; !out_of_words.failed && guard < 100000u; guard++) {
        (void)GDeflatePop(out_of_words, 8);
        GDeflateAdvance(out_of_words, ending.data());
    }
    REQUIRE(Landed(out_of_words, cudec_detail::kGDeflateRejectRefillPastEnd));

    /* A remove wider than the lane holds, reached directly: a codeword width
     * comes out of a decode table, so this guard stands between a table entry
     * and an occupancy that would wrap. */
    GDeflateSchedule removes;
    REQUIRE(GDeflateInit(removes, page.data(), page.size()));
    GDeflateRemove(removes, removes.bitsleft[0] + 1u);
    REQUIRE(Landed(removes, cudec_detail::kGDeflateRejectRemovePastLane));

    /* A read wider than the widest field the format has, on a lane that could
     * have served a legal one. */
    GDeflateSchedule too_wide;
    REQUIRE(GDeflateInit(too_wide, page.data(), page.size()));
    REQUIRE(GDeflatePop(too_wide, cudec_detail::kGDeflateMaxPopBits + 1u) == 0);
    REQUIRE(Landed(too_wide, cudec_detail::kGDeflateRejectPopWidthPastFormat));

    /* And a legal width the lane cannot serve, which is the other half of the
     * condition that used to be one. The lane is emptied first, or the width
     * bound above would answer instead and this rung would never be seen. */
    GDeflateSchedule empty_lane;
    REQUIRE(GDeflateInit(empty_lane, page.data(), page.size()));
    (void)GDeflatePop(empty_lane, cudec_detail::kGDeflateMaxPopBits);
    REQUIRE(!empty_lane.failed);
    REQUIRE(empty_lane.bitsleft[0] == 0);
    REQUIRE(GDeflatePop(empty_lane, 1) == 0);
    REQUIRE(Landed(empty_lane, cudec_detail::kGDeflateRejectPopPastLane));
    return 0;
}

/* ---- src/gdeflate_tables.h, the construction ---- */

int ConstructionRungs() {
    GDeflateLitLenTable t;
    std::vector<unsigned char> lens;

    /* An alphabet past the capacity of the table. */
    lens.assign(cudec_detail::kGDeflateNumLitLenSyms + 1u, 0);
    GDeflateReject slot = kGDeflateRejectNone;
    bool built = GDeflateBuildTable(
        lens.data(), cudec_detail::kGDeflateNumLitLenSyms + 1u, t, &slot);
    REQUIRE(NamedInSlot(built, slot,
                        cudec_detail::kGDeflateRejectTablePastCapacity));

    /* A length past the longest codeword the alphabet admits. */
    lens.assign(4, 0);
    lens[0] = static_cast<unsigned char>(cudec_detail::kGDeflateMaxCodeLen + 1u);
    slot = kGDeflateRejectNone;
    built = GDeflateBuildTable(lens.data(), 4, t, &slot);
    REQUIRE(NamedInSlot(built, slot,
                        cudec_detail::kGDeflateRejectTableLengthPastMax));

    /* Over-subscribed: three symbols at length 1 claim more codespace than
     * exists, so no assignment is possible. */
    lens.assign(4, 0);
    lens[0] = 1;
    lens[1] = 1;
    lens[2] = 1;
    slot = kGDeflateRejectNone;
    built = GDeflateBuildTable(lens.data(), 4, t, &slot);
    REQUIRE(NamedInSlot(built, slot,
                        cudec_detail::kGDeflateRejectTableOverSubscribed));

    /* Incomplete, and not one of the two shapes the reference admits: two
     * symbols at length 2 leave half the codespace unclaimed. */
    lens.assign(4, 0);
    lens[0] = 2;
    lens[1] = 2;
    slot = kGDeflateRejectNone;
    built = GDeflateBuildTable(lens.data(), 4, t, &slot);
    REQUIRE(NamedInSlot(built, slot,
                        cudec_detail::kGDeflateRejectTableIncomplete));
    return 0;
}

/* ---- src/gdeflate_tables.h, the decode ---- */

int EmptyTableRung() {
    /* An all-zero vector is the empty code, which the construction ADMITS and
     * this header refuses on use - stricter than the reference, which resolves
     * such a codeword to a symbol the stream never encoded. */
    GDeflateLitLenTable t;
    const std::vector<unsigned char> lens(cudec_detail::kGDeflateNumLitLenSyms,
                                          0);
    GDeflateReject slot = kGDeflateRejectNone;
    REQUIRE(GDeflateBuildTable(lens.data(),
                               cudec_detail::kGDeflateNumLitLenSyms, t, &slot));
    REQUIRE(t.kind == cudec_detail::kGDeflateTableEmpty);

    const std::vector<unsigned char> page = IndexedWords(64);
    GDeflateSchedule s;
    REQUIRE(GDeflateInit(s, page.data(), page.size()));
    REQUIRE(GDeflateDecodeSymbol(s, t) == kGDeflateNoSymbol);
    REQUIRE(Landed(s, cudec_detail::kGDeflateRejectEmptyTableUsed));
    return 0;
}

/* ---- src/gdeflate_tables.h, the code-length rounds ---- */

/* A page whose code-length stream is exactly `toks`. The literal/length and
 * distance vectors are the smallest the header fields can state, because what
 * is under test is the expansion of the token stream and not the code those
 * lengths would describe. */
bool PageWithTokens(const std::vector<LenToken>& toks,
                    std::vector<unsigned char>* page) {
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    if (!PlanPrecode(toks, 0, precode_lens, &num_explicit)) {
        return false;
    }
    const std::vector<unsigned char> litlen_lens = CompleteLengths(257);
    const std::vector<unsigned char> dist_lens(1, 1);
    const std::vector<uint32_t> no_symbols;
    return EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                    no_symbols, page);
}

/* Every fixture below is handed to the page decode rather than to the function
 * that owns the rung. The rungs are reached through the block header the
 * decoder reads first, which is what a page reaches them through - calling the
 * code-length round straight after the priming round would read the block
 * header bits as HLIT and test a misalignment this file invented. */
int DecodeRefuses(const std::vector<unsigned char>& page, uint64_t out_cap,
                  GDeflateReject expected) {
    GDeflatePageState st;
    uint64_t out_len = 0;
    std::vector<unsigned char> out(static_cast<size_t>(out_cap), 0);
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), out.data(),
                                out.size(), &out_len));
    REQUIRE(Landed(st.s, expected));
    return 0;
}

int CodeLengthRungs() {
    /* A repeat-previous code with nothing before it. */
    std::vector<LenToken> toks;
    LenToken repeat_prev = {16, 2, 0};
    toks.push_back(repeat_prev);
    std::vector<unsigned char> page;
    REQUIRE(PageWithTokens(toks, &page));
    REQUIRE(DecodeRefuses(
                page, 64,
                cudec_detail::kGDeflateRejectRepeatNothingBefore) == 0);

    /* A zero run reaching past the end of the two vectors the header declared.
     * The smallest legal HLIT and HDIST make 258 lengths, and one explicit
     * length followed by two maximal zero runs is 277. */
    toks.clear();
    LenToken explicit_one = {1, 0, 0};
    LenToken zero_run = {18, 7, 127};
    toks.push_back(explicit_one);
    toks.push_back(zero_run);
    toks.push_back(zero_run);
    std::vector<unsigned char> overrun;
    REQUIRE(PageWithTokens(toks, &overrun));
    REQUIRE(DecodeRefuses(
                overrun, 64,
                cudec_detail::kGDeflateRejectRepeatRunPastAlphabet) == 0);
    return 0;
}

/* ---- src/gdeflate_block.h ---- */

/* A page opening with BFINAL set and the given block type, and nothing after
 * the three header bits. Every rung that refuses inside the block header needs
 * no more of a page than this. */
std::vector<unsigned char> PageWithBlockType(uint32_t block_type) {
    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, block_type);
    return w.Finish();
}

int ReservedBlockTypeRung() {
    const std::vector<unsigned char> page = PageWithBlockType(3);
    GDeflatePageState st;
    uint64_t out_len = 0;
    std::vector<unsigned char> out(64, 0);
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), out.data(),
                                out.size(), &out_len));
    REQUIRE(Landed(st.s, cudec_detail::kGDeflateRejectBlockTypeReserved));
    return 0;
}

/* A stored block declaring `len` bytes and carrying none of them. `trim` drops
 * the zero tail the writer appends for the reference, which is what makes the
 * page small enough for a declared length to outrun the bytes it can still
 * supply. */
std::vector<unsigned char> StoredPage(uint32_t len, bool trim) {
    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, 0);
    w.Ensure();
    w.Push(16, len);
    std::vector<unsigned char> page = w.Finish();
    if (trim) {
        page.resize(page.size() -
                    static_cast<size_t>(cudec_test::kGDeflateWriterTailWords) *
                        4u);
    }
    return page;
}

int StoredRungs() {
    /* Past the capacity of the caller, which is the only output bound a stored
     * block has: the format states no uncompressed size inside a page. */
    const std::vector<unsigned char> page = StoredPage(8, false);
    GDeflatePageState st;
    uint64_t out_len = 0;
    std::vector<unsigned char> small(4, 0);
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), small.data(),
                                small.size(), &out_len));
    REQUIRE(Landed(st.s, cudec_detail::kGDeflateRejectStoredPastCap));

    /* Past what the page can still supply, with a capacity wide enough that
     * the rung above cannot answer first. */
    const std::vector<unsigned char> tight = StoredPage(4000, true);
    GDeflatePageState st2;
    std::vector<unsigned char> wide(65536, 0);
    REQUIRE(!GDeflateDecodePage(st2, tight.data(), tight.size(), wide.data(),
                                wide.size(), &out_len));
    REQUIRE(Landed(st2.s, cudec_detail::kGDeflateRejectStoredPastPage));
    return 0;
}

/* A page of non-final stored blocks, each declaring zero bytes, which is the
 * cheapest block the format has. It is the measurement behind the
 * kGDeflateRejectNoFinalBlock declaration at the bottom of this file: the
 * argument there is that a page runs out of words long before the block cap,
 * and an argument about a fuel cap is exactly the kind that is wrong quietly.
 * The page is asserted to land on the refill rung instead. */
int NoFinalBlockStopsAtTheRefill() {
    GDeflatePageWriter w;
    for (uint32_t block = 0; block < 4096u; block++) {
        w.Reset();
        w.Push(1, 0);
        w.Push(2, 0);
        w.Ensure();
        w.Push(16, 0);
    }
    REQUIRE(w.ok());
    std::vector<unsigned char> page = w.Finish();
    page.resize(page.size() -
                static_cast<size_t>(cudec_test::kGDeflateWriterTailWords) * 4u);

    GDeflatePageState st;
    uint64_t out_len = 0;
    std::vector<unsigned char> out(64, 0);
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), out.data(),
                                out.size(), &out_len));
    REQUIRE(Landed(st.s, cudec_detail::kGDeflateRejectRefillPastEnd));
    return 0;
}

/* A dynamic block whose body is `symbols`, over a literal/length code that can
 * spell every symbol they use. */
bool PageWithSymbols(const std::vector<unsigned char>& litlen_lens,
                     const std::vector<unsigned char>& dist_lens,
                     const std::vector<uint32_t>& symbols,
                     std::vector<unsigned char>* page) {
    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = cudec_test::BuildTokens(all);
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    if (!PlanPrecode(toks, 0, precode_lens, &num_explicit)) {
        return false;
    }
    return EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                    symbols, page);
}

int LiteralPastCapRung() {
    /* Three literals into a two-byte destination. The refusal is on the third,
     * before anything is written. */
    std::vector<unsigned char> litlen_lens(257, 0);
    litlen_lens[65] = 1;
    litlen_lens[cudec_test::kEndOfBlock] = 1;
    const std::vector<unsigned char> dist_lens(1, 1);
    std::vector<uint32_t> symbols;
    symbols.push_back(65);
    symbols.push_back(65);
    symbols.push_back(65);
    symbols.push_back(cudec_test::kEndOfBlock);
    std::vector<unsigned char> page;
    REQUIRE(PageWithSymbols(litlen_lens, dist_lens, symbols, &page));

    GDeflatePageState st;
    uint64_t out_len = 0;
    std::vector<unsigned char> out(2, 0);
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), out.data(),
                                out.size(), &out_len));
    REQUIRE(Landed(st.s, cudec_detail::kGDeflateRejectLiteralPastCap));
    return 0;
}

/* A page whose first round reserves a match of the minimum length and whose
 * second round ends the block, so the deferred copy is retired in the drain -
 * which is where the distance symbol is decoded. Written straight against the
 * writer because EmitPage stops at literals and end-of-block. */
bool MatchPage(std::vector<unsigned char>* page) {
    std::vector<unsigned char> litlen_lens(258, 0);
    litlen_lens[65] = 2;
    litlen_lens[cudec_test::kEndOfBlock] = 2;
    litlen_lens[257] = 1; /* length 3, and no extra bits behind it */
    const std::vector<unsigned char> dist_lens(1, 1);

    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = cudec_test::BuildTokens(all);
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    if (!PlanPrecode(toks, 0, precode_lens, &num_explicit)) {
        return false;
    }
    cudec_detail::GDeflatePrecodeTable precode;
    if (!GDeflateBuildTable(precode_lens, cudec_detail::kGDeflateNumPrecodeSyms,
                            precode)) {
        return false;
    }
    GDeflateLitLenTable litlen;
    if (!GDeflateBuildTable(litlen_lens.data(),
                            static_cast<uint32_t>(litlen_lens.size()),
                            litlen)) {
        return false;
    }
    cudec_detail::GDeflateDistTable dist;
    if (!GDeflateBuildTable(dist_lens.data(),
                            static_cast<uint32_t>(dist_lens.size()), dist)) {
        return false;
    }

    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, cudec_test::kBlockTypeDynamic);
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
        if (!cudec_test::CodewordOf(precode, precode_lens, toks[i].presym,
                                    &code, &len)) {
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
    if (!cudec_test::CodewordOf(litlen, litlen_lens.data(), 257, &code, &len)) {
        return false;
    }
    w.PushCode(code, len);
    uint32_t dcode = 0;
    uint32_t dlen = 0;
    if (!cudec_test::CodewordOf(dist, dist_lens.data(), 0, &dcode, &dlen)) {
        return false;
    }
    w.PushCode(dcode, dlen);
    w.Advance();
    /* Lane 1: end of block, which the decoder leaves without advancing. */
    if (!cudec_test::CodewordOf(litlen, litlen_lens.data(),
                                cudec_test::kEndOfBlock, &code, &len)) {
        return false;
    }
    w.PushCode(code, len);
    for (uint32_t i = 0; i < kGDeflateNumStreams; i++) {
        w.Advance();
    }
    if (!w.ok()) {
        return false;
    }
    *page = w.Finish();
    return true;
}

int MatchRungs() {
    std::vector<unsigned char> page;
    REQUIRE(MatchPage(&page));

    /* The reservation is refused against the capacity before the hole is
     * claimed: three bytes do not fit in two. */
    GDeflatePageState st;
    uint64_t out_len = 0;
    std::vector<unsigned char> tight(2, 0);
    REQUIRE(!GDeflateDecodePage(st, page.data(), page.size(), tight.data(),
                                tight.size(), &out_len));
    REQUIRE(Landed(st.s, cudec_detail::kGDeflateRejectMatchPastCap));

    /* With room for the match, the same page reaches the copy - and its
     * distance points before the first byte this page produced. */
    GDeflatePageState st2;
    std::vector<unsigned char> roomy(64, 0);
    REQUIRE(!GDeflateDecodePage(st2, page.data(), page.size(), roomy.data(),
                                roomy.size(), &out_len));
    REQUIRE(Landed(st2.s, cudec_detail::kGDeflateRejectMatchBeforeOutput));
    return 0;
}

/* ---- The sweep ---- */

/* The three rungs no input reaches, each with the argument that says why. They
 * are guards the decoder already carried rather than guards this ladder added,
 * and the assertion below is that they stay unreached: the day one of them
 * becomes reachable, this test fails and a negative is owed for it.
 *
 * kGDeflateRejectCodewordNotInCode - the walk past the longest codeword. A
 * table that decodes is complete, empty or the single-symbol degenerate, and
 * the last two are answered above the walk; on a COMPLETE canonical code every
 * bit sequence resolves at or before the longest length, so the walk cannot
 * fall out of its loop.
 *
 * kGDeflateRejectRoundFuelExhausted - the round cap with no end-of-block.
 * Every round writes a byte, reserves a match of at least the minimum length,
 * or retires a reservation, and the cap is two per output byte plus one per
 * lane, so the capacity refusals answer first on every page that would run
 * that far.
 *
 * kGDeflateRejectNoFinalBlock - the block cap with no final block. The
 * cheapest non-final block is a stored one declaring zero bytes, which costs
 * lane 0 nineteen bits - three of header and sixteen of length - and lane 0 is
 * refilled from one word each time those reads drop it under the watermark, so
 * a page yields fewer than two blocks per word it carries. The cap is
 * twenty-one per word, so a page of non-final blocks runs out of words first
 * and kGDeflateRejectRefillPastEnd is what it lands on. */
bool DeclaredUnreachable(int branch) {
    return branch == cudec_detail::kGDeflateRejectCodewordNotInCode ||
           branch == cudec_detail::kGDeflateRejectRoundFuelExhausted ||
           branch == cudec_detail::kGDeflateRejectNoFinalBlock;
}

/* A sentinel is not a rung: it carries an explicit value, so it shares a
 * number with the branch it bounds and is covered when that branch is. The
 * sweep walks NUMBERS, so it sees each number once whatever it is called. */
int Sweep() {
    int reached = 0;
    int unreachable = 0;
    for (int branch = kGDeflateRejectNone + 1; branch < kGDeflateRejectCount;
         branch++) {
        const bool covered = g_covered[branch];
        if (DeclaredUnreachable(branch)) {
            REQUIRE_CTX(!covered,
                        "ladder branch %d is declared unreachable and a "
                        "negative reached it: it owes a negative and the "
                        "declaration above owes a correction",
                        branch);
            unreachable++;
            continue;
        }
        REQUIRE_CTX(covered,
                    "ladder branch %d is declared by enum GDeflateReject and "
                    "no negative in this file reaches it (issue #183)",
                    branch);
        reached++;
    }
    std::printf(
        "gdeflate_ladder_lock: %d rungs reached by a negative, %d declared "
        "unreachable, %d declared in total\n",
        reached, unreachable, static_cast<int>(kGDeflateRejectCount) - 1);
    return 0;
}

}  // namespace

int main() {
    if (ScheduleRungs() != 0) {
        return 1;
    }
    if (ConstructionRungs() != 0) {
        return 1;
    }
    if (EmptyTableRung() != 0) {
        return 1;
    }
    if (CodeLengthRungs() != 0) {
        return 1;
    }
    if (ReservedBlockTypeRung() != 0) {
        return 1;
    }
    if (StoredRungs() != 0) {
        return 1;
    }
    if (NoFinalBlockStopsAtTheRefill() != 0) {
        return 1;
    }
    if (LiteralPastCapRung() != 0) {
        return 1;
    }
    if (MatchRungs() != 0) {
        return 1;
    }
    if (Sweep() != 0) {
        return 1;
    }
    std::printf("gdeflate_ladder_lock: ok\n");
    return 0;
}
