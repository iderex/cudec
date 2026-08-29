/* The GDeflate dynamic block header, held against the reference (issue #176).
 * src/gdeflate_tables.h turns a page into the two code-length vectors and the
 * two tables built from them; this file establishes that it recovers the
 * vectors the page encodes, that it walks exactly the rounds the reference
 * walks, and that it refuses the malformed round shapes.
 *
 * HOW PARITY IS ESTABLISHED, AND WHY IT IS NOT A VECTOR COMPARISON. The
 * reference's code-length vectors are unreachable: the pinned fork keeps
 * `precode_lens` and `lens` in the decompressor's private state inside a
 * translation unit compiled with HIDE_INTERFACE, and the whole public
 * GDeflate surface is thirteen names, none of which reaches them (measured on
 * #176 at the pinned commit). So the comparison this file makes is the one the
 * oracle's door opens, and it is three statements rather than one:
 *
 *   1. The page is the format. The reference DECODES every positive fixture
 *      here and returns the exact bytes the fixture encoded, so the header the
 *      emitter wrote is one the reference reads as a legal dynamic block.
 *   2. The vectors are recovered. The header decode under test reads that same
 *      page back and produces, symbol by symbol, the vectors the emitter put
 *      in. Emitter and decoder are independent code, and clause 1 is what ties
 *      the emitter to the format rather than to this file's opinion of it.
 *   3. The rounds are the same rounds. After the header, the block's body is
 *      decoded through the recovered literal/length table and required to
 *      match the reference's output byte for byte. A header that consumed one
 *      round too many or too few leaves the body starting on the wrong lane,
 *      and no bit layout recovers from that - which is the observable parity
 *      the rewritten first Done-when asks for, for the literal-bodied case.
 *
 * WHAT IS NOT COVERED, SAID PLAINLY SO THE GAP IS NOT READ AS COVERAGE. A
 * body carrying MATCHES is #182: a length symbol's extra bits, the distance
 * symbol in the round after it, and the deferred in-tile copy are the block
 * loop, and none of them is decoded here. Clause 3 therefore holds over
 * literal-and-end-of-block bodies only, so the distance table is exercised
 * through its construction and never through a decoded match. */
#include "gdeflate_page_writer.h"
#include "gdeflate_tables.h"
#include "require.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::GDeflateAdvance;
using cudec_detail::GDeflateBuildTable;
using cudec_detail::GDeflateCodeLengths;
using cudec_detail::GDeflateDecodeSymbol;
using cudec_detail::GDeflateDistTable;
using cudec_detail::GDeflateInit;
using cudec_detail::GDeflateLitLenTable;
using cudec_detail::GDeflatePop;
using cudec_detail::GDeflatePrecodeOrder;
using cudec_detail::GDeflatePrecodeTable;
using cudec_detail::GDeflateReadCodeLengths;
using cudec_detail::GDeflateReadDynamicTables;
using cudec_detail::GDeflateReset;
using cudec_detail::GDeflateSchedule;
using cudec_detail::kGDeflateMaxCodeLen;
using cudec_detail::kGDeflateNumDistSyms;
using cudec_detail::kGDeflateNumLitLenSyms;
using cudec_detail::kGDeflateNumPrecodeSyms;
using cudec_detail::kGDeflateTableEmpty;
using cudec_test::CodewordOf;
using cudec_test::CompleteLengths;
using cudec_test::GDeflatePageWriter;

/* DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN in the reference's deflate_constants.h. */
constexpr uint32_t kBlockTypeDynamic = 2;
/* The end-of-block symbol, the one literal/length symbol every fixture here
 * has to code for the reference to finish a block. */
constexpr uint32_t kEndOfBlock = 256;

/* One element of the code-length stream: a precode symbol and, for the three
 * repeat codes, the extra-bit field that follows it in the SAME round. */
struct LenToken {
    uint32_t presym;
    uint32_t extra_bits;
    uint32_t extra_value;
};

/* Run-length encode a concatenated length vector the way a compressor does,
 * using all three repeat codes wherever they apply. This is what puts codes
 * 16, 17 and 18 into the fixtures at all: an explicit-only encoding is a legal
 * header that never reaches the branches this issue is about. */
std::vector<LenToken> BuildTokens(const std::vector<unsigned char>& all) {
    std::vector<LenToken> out;
    size_t i = 0;
    for (uint32_t guard = 0; guard < 1024u && i < all.size(); guard++) {
        const unsigned char v = all[i];
        size_t run = 1;
        for (size_t j = i + 1; j < all.size() && all[j] == v; j++) {
            run++;
        }
        if (v == 0) {
            for (uint32_t g = 0; g < 1024u && run >= 11; g++) {
                const size_t take = run < 138 ? run : 138;
                out.push_back({18, 7, static_cast<uint32_t>(take - 11)});
                i += take;
                run -= take;
            }
            for (uint32_t g = 0; g < 1024u && run >= 3; g++) {
                const size_t take = run < 10 ? run : 10;
                out.push_back({17, 3, static_cast<uint32_t>(take - 3)});
                i += take;
                run -= take;
            }
        } else {
            out.push_back({v, 0, 0});
            i++;
            run--;
            for (uint32_t g = 0; g < 1024u && run >= 3; g++) {
                const size_t take = run < 6 ? run : 6;
                out.push_back({16, 2, static_cast<uint32_t>(take - 3)});
                i += take;
                run -= take;
            }
        }
        for (size_t k = 0; k < run; k++) {
            out.push_back({v, 0, 0});
            i++;
        }
    }
    return out;
}

/* Precode lengths covering exactly the symbols the token stream uses, in the
 * shape CompleteLengths gives, plus the smallest HCLEN that reaches them all.
 * `forced_explicit` overrides that count where a fixture is about the field
 * rather than about the vector; 0 means derive it. */
bool PlanPrecode(const std::vector<LenToken>& toks, uint32_t forced_explicit,
                 unsigned char precode_lens[kGDeflateNumPrecodeSyms],
                 uint32_t* num_explicit) {
    unsigned char used[kGDeflateNumPrecodeSyms];
    std::memset(used, 0, sizeof(used));
    for (size_t i = 0; i < toks.size(); i++) {
        if (toks[i].presym >= kGDeflateNumPrecodeSyms) {
            return false;
        }
        used[toks[i].presym] = 1;
    }
    uint32_t n_used = 0;
    for (uint32_t v = 0; v < kGDeflateNumPrecodeSyms; v++) {
        n_used += used[v];
    }
    const std::vector<unsigned char> shape = CompleteLengths(n_used);
    std::memset(precode_lens, 0, kGDeflateNumPrecodeSyms);
    uint32_t k = 0;
    for (uint32_t v = 0; v < kGDeflateNumPrecodeSyms; v++) {
        if (used[v]) {
            precode_lens[v] = shape[k++];
        }
    }
    /* The permutation is what decides how many lengths the header must state:
     * a symbol at position p is only reachable when HCLEN + 4 exceeds p. */
    uint32_t needed = 4;
    for (uint32_t i = 0; i < kGDeflateNumPrecodeSyms; i++) {
        if (used[GDeflatePrecodeOrder(i)] && i + 1u > needed) {
            needed = i + 1u;
        }
    }
    if (forced_explicit != 0) {
        if (forced_explicit < needed) {
            return false;
        }
        needed = forced_explicit;
    }
    *num_explicit = needed;
    return true;
}

/* Emit one whole page: a single final dynamic block whose header carries
 * `toks` and whose body encodes `symbols`. Every round here mirrors
 * gdeflate_decompress_template.h in the pinned fork - the header fields ride
 * lane 0 with no round between them, each precode length is its own round,
 * and a repeat code's extra bits are pushed before the round ends. */
bool EmitPage(const std::vector<unsigned char>& litlen_lens,
              const std::vector<unsigned char>& dist_lens,
              const std::vector<LenToken>& toks,
              const unsigned char precode_lens[kGDeflateNumPrecodeSyms],
              uint32_t num_explicit, const std::vector<uint32_t>& symbols,
              std::vector<unsigned char>* page) {
    GDeflatePrecodeTable precode;
    if (!GDeflateBuildTable(precode_lens, kGDeflateNumPrecodeSyms, precode)) {
        return false;
    }
    GDeflateLitLenTable litlen;
    const bool litlen_ok =
        GDeflateBuildTable(litlen_lens.data(),
                           static_cast<uint32_t>(litlen_lens.size()), litlen);

    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);                 /* BFINAL: one block per page here. */
    w.Push(2, kBlockTypeDynamic); /* BTYPE */
    w.Ensure();
    w.Push(5, static_cast<uint32_t>(litlen_lens.size()) - 257u); /* HLIT */
    w.Push(5, static_cast<uint32_t>(dist_lens.size()) - 1u);     /* HDIST */
    w.Push(4, num_explicit - 4u);                                /* HCLEN */
    w.Ensure();
    for (uint32_t i = 0; i < num_explicit; i++) {
        w.Push(3, precode_lens[GDeflatePrecodeOrder(i)]);
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

    if (!symbols.empty()) {
        if (!litlen_ok) {
            return false;
        }
        w.Reset();
        for (size_t i = 0; i < symbols.size(); i++) {
            uint32_t code = 0;
            uint32_t len = 0;
            if (!CodewordOf(litlen, litlen_lens.data(), symbols[i], &code,
                            &len)) {
                return false;
            }
            w.PushCode(code, len);
            if (symbols[i] == kEndOfBlock) {
                /* The reference leaves the decode loop on end-of-block without
                 * advancing, then runs 32 rounds to drain deferred copies.
                 * Those rounds refill, so the writer owes their words. */
                break;
            }
            w.Advance();
        }
        for (uint32_t i = 0; i < cudec_detail::kGDeflateNumStreams; i++) {
            w.Advance();
        }
    }

    if (!w.ok()) {
        return false;
    }
    *page = w.Finish();
    return true;
}

/* The reference's verdict on one page, plus what it produced. */
struct OracleAnswer {
    int status;
    std::vector<unsigned char> out;
};

OracleAnswer AskOracle(const std::vector<unsigned char>& page, size_t out_cap) {
    OracleAnswer a;
    a.status = LIBDEFLATE_BAD_DATA;
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        return a;
    }
    libdeflate_gdeflate_in_page in;
    in.data = page.data();
    in.nbytes = page.size();
    a.out.assign(out_cap, 0);
    size_t produced = 0;
    a.status =
        libdeflate_gdeflate_decompress(d, &in, 1, a.out.data(), out_cap,
                                       &produced);
    libdeflate_free_gdeflate_decompressor(d);
    a.out.resize(a.status == LIBDEFLATE_SUCCESS ? produced : 0);
    return a;
}

/* Position a schedule immediately after BTYPE, which is where the header
 * decode under test is entered. */
bool OpenBlock(GDeflateSchedule& s, const std::vector<unsigned char>& page) {
    if (!GDeflateInit(s, page.data(), page.size())) {
        return false;
    }
    GDeflateReset(s);
    GDeflatePop(s, 1); /* BFINAL */
    return GDeflatePop(s, 2) == kBlockTypeDynamic && !s.failed;
}

/* The reference's main loop restricted to its literal path: one symbol per
 * round, end-of-block leaving the loop without advancing. A length symbol is
 * refused rather than skipped, because a fixture that produced one would be
 * silently testing less than it says. */
bool DecodeLiteralBody(GDeflateSchedule& s, const unsigned char* page,
                       const GDeflateLitLenTable& litlen,
                       std::vector<unsigned char>* out) {
    GDeflateReset(s);
    out->clear();
    for (uint32_t fuel = 0; fuel < 65536u; fuel++) {
        const uint32_t sym = GDeflateDecodeSymbol(s, litlen);
        if (sym == cudec_detail::kGDeflateNoSymbol) {
            return false;
        }
        if (sym == kEndOfBlock) {
            return true;
        }
        if (sym > kEndOfBlock) {
            return false;
        }
        out->push_back(static_cast<unsigned char>(sym));
        GDeflateAdvance(s, page);
        if (s.failed) {
            return false;
        }
    }
    return false;
}

/* The three statements the file comment sets out, run over one fixture. */
int CheckFixture(const char* name,
                 const std::vector<unsigned char>& litlen_lens,
                 const std::vector<unsigned char>& dist_lens,
                 const std::vector<LenToken>& toks, uint32_t forced_explicit,
                 const std::vector<uint32_t>& symbols) {
    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE_CTX(PlanPrecode(toks, forced_explicit, precode_lens, &num_explicit),
                "%s", name);

    std::vector<unsigned char> page;
    REQUIRE_CTX(EmitPage(litlen_lens, dist_lens, toks, precode_lens,
                         num_explicit, symbols, &page),
                "%s", name);

    /* The bytes the body names, which are both what the reference must return
     * and what the recovered table must decode to. */
    std::vector<unsigned char> want;
    for (size_t i = 0; i < symbols.size(); i++) {
        if (symbols[i] != kEndOfBlock) {
            want.push_back(static_cast<unsigned char>(symbols[i]));
        }
    }

    /* Statement 1: the reference reads this page as a legal dynamic block. */
    const OracleAnswer a = AskOracle(page, want.size());
    REQUIRE_CTX(a.status == LIBDEFLATE_SUCCESS, "%s: status %d", name,
                a.status);
    REQUIRE_CTX(a.out.size() == want.size(), "%s: %zu bytes", name,
                a.out.size());
    REQUIRE_CTX(equal_bytes(a.out.data(), want.data(), want.size()), "%s",
                name);

    /* Statement 2: the header decode recovers both vectors exactly. */
    GDeflateSchedule s;
    REQUIRE_CTX(OpenBlock(s, page), "%s", name);
    GDeflateCodeLengths lens;
    GDeflateLitLenTable litlen;
    GDeflateDistTable dist;
    REQUIRE_CTX(
        GDeflateReadDynamicTables(s, page.data(), lens, litlen, dist), "%s",
        name);
    REQUIRE_CTX(lens.num_litlen == litlen_lens.size(), "%s: HLIT %u", name,
                lens.num_litlen);
    REQUIRE_CTX(lens.num_dist == dist_lens.size(), "%s: HDIST %u", name,
                lens.num_dist);
    REQUIRE_CTX(equal_bytes(lens.lens, litlen_lens.data(), litlen_lens.size()),
                "%s: litlen vector", name);
    REQUIRE_CTX(equal_bytes(lens.lens + lens.num_litlen, dist_lens.data(),
                            dist_lens.size()),
                "%s: distance vector", name);

    /* Statement 3: the body starts where the reference left it. */
    std::vector<unsigned char> got;
    REQUIRE_CTX(DecodeLiteralBody(s, page.data(), litlen, &got), "%s", name);
    REQUIRE_CTX(got.size() == want.size(), "%s: %zu body bytes", name,
                got.size());
    REQUIRE_CTX(equal_bytes(got.data(), want.data(), want.size()),
                "%s: body", name);
    return 0;
}

/* A literal/length vector coding every symbol 0..256, the alphabet fixture.
 * Its long run of equal lengths is what puts code 16 into the token stream. */
std::vector<unsigned char> AlphabetLens() {
    return CompleteLengths(257);
}

int RunAlphabet() {
    const std::vector<unsigned char> litlen_lens = AlphabetLens();
    /* A block with no matches declares a distance code with nothing in it,
     * which is the reference's empty-code admission. */
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());

    std::vector<uint32_t> symbols;
    for (uint32_t b = 0; b < 256; b++) {
        symbols.push_back(b);
    }
    symbols.push_back(kEndOfBlock);
    return CheckFixture("alphabet", litlen_lens, dist_lens, BuildTokens(all), 0,
                        symbols);
}

/* The literal/length vector the extremes fixture is built on: 32 symbols at
 * length 5 spend the whole codespace (32 * 2^10 = 2^15), and WHERE those 32
 * sit is free, because completeness is a property of the multiset alone. That
 * freedom is what lets one legal vector carry every repeat-code extreme at
 * once. The runs, in order: 4 and 7 and 5 and 5 and 8 and 3 coded symbols,
 * separated by 3, 10, 11, 138 and 65 zeroes. */
std::vector<unsigned char> ExtremesLens() {
    std::vector<unsigned char> lens(kGDeflateNumLitLenSyms, 0);
    const uint32_t kRun[6][2] = {{0, 4},   {7, 7},   {24, 5},
                                 {40, 5},  {183, 8}, {256, 3}};
    for (uint32_t r = 0; r < 6; r++) {
        for (uint32_t k = 0; k < kRun[r][1]; k++) {
            lens[kRun[r][0] + k] = 5;
        }
    }
    return lens;
}

int RunExtremes() {
    const std::vector<unsigned char> litlen_lens = ExtremesLens();
    /* HDIST at its maximum, all of it uncoded: the distance code is empty, as
     * it is in every literal-only block. */
    const std::vector<unsigned char> dist_lens(kGDeflateNumDistSyms, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = BuildTokens(all);

    /* Each repeat code at both ends of its extra-bit field, asserted present
     * rather than assumed: a fixture that stopped producing one of these would
     * otherwise keep passing while covering less. */
    const uint32_t kWanted[6][2] = {{16, 0}, {16, 3},  {17, 0},
                                    {17, 7}, {18, 0},  {18, 127}};
    for (uint32_t n = 0; n < 6; n++) {
        bool seen = false;
        for (size_t i = 0; i < toks.size(); i++) {
            if (toks[i].presym == kWanted[n][0] &&
                toks[i].extra_value == kWanted[n][1]) {
                seen = true;
            }
        }
        REQUIRE_CTX(seen, "repeat code %u with extra %u is not in the token "
                          "stream",
                    kWanted[n][0], kWanted[n][1]);
    }

    std::vector<uint32_t> symbols;
    for (uint32_t i = 0; i < kGDeflateNumLitLenSyms; i++) {
        if (litlen_lens[i] != 0 && i < kEndOfBlock) {
            symbols.push_back(i);
        }
    }
    symbols.push_back(kEndOfBlock);
    /* HLIT and HDIST are already at their maxima; HCLEN is forced to its own
     * so one fixture pins all three field ceilings. */
    return CheckFixture("extremes", litlen_lens, dist_lens, toks,
                        kGDeflateNumPrecodeSyms, symbols);
}

/* HLIT, HDIST and HCLEN at their floors, in the smallest shape that still
 * decodes. HCLEN + 4 = 5 reaches permutation positions 16, 17, 18, 0 and 8, so
 * length 8 is the only codeword length available - and 256 symbols at length 8
 * spend the codespace exactly. */
int RunFieldFloors() {
    std::vector<unsigned char> litlen_lens(257, 8);
    litlen_lens[255] = 0;
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());

    std::vector<uint32_t> symbols;
    for (uint32_t b = 0; b < 255; b++) {
        symbols.push_back(b);
    }
    symbols.push_back(kEndOfBlock);
    return CheckFixture("field-floors", litlen_lens, dist_lens,
                        BuildTokens(all), 5, symbols);
}

/* HCLEN at its absolute floor, which no decodable block can reach. HCLEN + 4 =
 * 4 reaches permutation positions 16, 17, 18 and 0 only, so every length the
 * header can state is zero and the literal/length code is empty.
 *
 * THE ORACLE IS NOT ASKED HERE AND THE REASON IS THE DIVERGENCE ITSELF. An
 * empty literal/length code is where src/gdeflate_tables.h is deliberately
 * stricter than the reference, which fills its table with a synthetic
 * "symbol 0, length 1" entry and decodes bytes the stream never encoded; that
 * divergence is executed against the reference in
 * tests/gdeflate_tables_twin.cpp and is not re-measured here. What this case
 * pins is the field floor: the header read SUCCEEDS, the vectors come back
 * all-zero, and the emptiness is carried in the table's kind rather than in a
 * refusal at read time. */
int RunHclenFloor() {
    const std::vector<unsigned char> litlen_lens(257, 0);
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = BuildTokens(all);

    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(PlanPrecode(toks, 4, precode_lens, &num_explicit));
    REQUIRE(num_explicit == 4);

    std::vector<unsigned char> page;
    REQUIRE(EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                     std::vector<uint32_t>(), &page));

    GDeflateSchedule s;
    REQUIRE(OpenBlock(s, page));
    GDeflateCodeLengths lens;
    GDeflateLitLenTable litlen;
    GDeflateDistTable dist;
    REQUIRE(GDeflateReadDynamicTables(s, page.data(), lens, litlen, dist));
    REQUIRE(lens.num_litlen == 257);
    REQUIRE(lens.num_dist == 1);
    for (uint32_t i = 0; i < lens.num_litlen + lens.num_dist; i++) {
        REQUIRE_CTX(lens.lens[i] == 0, "lens[%u]", i);
    }
    REQUIRE(litlen.kind == kGDeflateTableEmpty);
    REQUIRE(dist.kind == kGDeflateTableEmpty);
    return 0;
}

/* The permutation, which is the one thing about this header that no
 * completeness test can see: it reorders a multiset, so a vector read in
 * permutation order and one read in symbol order are equally complete and
 * equally plausible (measured as a surviving mutant on #170). What separates
 * them is lengths ASSIGNED TO SYMBOLS, and this asserts the alphabet fixture
 * is one where the two readings actually differ - a fixture whose precode
 * lengths happened to be invariant under the permutation would pass whichever
 * order the decode used. */
int RunPermutationIsLoadBearing() {
    std::vector<unsigned char> all = AlphabetLens();
    all.push_back(0);
    const std::vector<LenToken> toks = BuildTokens(all);
    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(PlanPrecode(toks, kGDeflateNumPrecodeSyms, precode_lens,
                        &num_explicit));
    bool differs = false;
    for (uint32_t i = 0; i < kGDeflateNumPrecodeSyms; i++) {
        if (precode_lens[GDeflatePrecodeOrder(i)] != precode_lens[i]) {
            differs = true;
        }
    }
    REQUIRE(differs);
    return 0;
}

/* Repeat code 16 as the first code of the vector: there is no previous length
 * to repeat. The one repeat-code refusal both decoders make, so the
 * reference's verdict is asserted beside ours. */
int RunRepeatWithNothingToRepeat() {
    const std::vector<unsigned char> litlen_lens = AlphabetLens();
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<LenToken> toks;
    toks.push_back({16, 2, 0});
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> rest = BuildTokens(all);
    for (size_t i = 0; i < rest.size(); i++) {
        toks.push_back(rest[i]);
    }

    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(PlanPrecode(toks, 0, precode_lens, &num_explicit));
    std::vector<unsigned char> page;
    REQUIRE(EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                     std::vector<uint32_t>(), &page));

    const OracleAnswer a = AskOracle(page, 4096);
    REQUIRE_CTX(a.status == LIBDEFLATE_BAD_DATA, "status %d", a.status);

    GDeflateSchedule s;
    REQUIRE(OpenBlock(s, page));
    GDeflateCodeLengths lens;
    REQUIRE(!GDeflateReadCodeLengths(s, page.data(), lens));
    REQUIRE(s.failed);
    return 0;
}

/* A precode vector that is not a code, in both directions. Over-subscribed:
 * three symbols at length 1 claim more codespace than exists. Incomplete
 * without being one of the two shapes the reference admits: two symbols at
 * length 2 leave half the codespace unreachable. Both are refused by the
 * reference and by the table construction this header feeds. */
int RunPrecodeIsNotACode() {
    const std::vector<unsigned char> litlen_lens = AlphabetLens();
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = BuildTokens(all);

    for (uint32_t shape = 0; shape < 2; shape++) {
        unsigned char precode_lens[kGDeflateNumPrecodeSyms];
        std::memset(precode_lens, 0, sizeof(precode_lens));
        const unsigned char len = shape == 0 ? 1 : 2;
        const uint32_t count = shape == 0 ? 3 : 2;
        for (uint32_t i = 0; i < count; i++) {
            precode_lens[GDeflatePrecodeOrder(i)] = len;
        }
        /* Emitted by hand: EmitPage would refuse to build a table from these,
         * which is the point, so the page is written without one. */
        GDeflatePageWriter w;
        w.Reset();
        w.Push(1, 1);
        w.Push(2, kBlockTypeDynamic);
        w.Ensure();
        w.Push(5, static_cast<uint32_t>(litlen_lens.size()) - 257u);
        w.Push(5, static_cast<uint32_t>(dist_lens.size()) - 1u);
        w.Push(4, kGDeflateNumPrecodeSyms - 4u);
        w.Ensure();
        for (uint32_t i = 0; i < kGDeflateNumPrecodeSyms; i++) {
            w.Push(3, precode_lens[GDeflatePrecodeOrder(i)]);
            w.Advance();
        }
        /* Enough rounds after the header for the reference to have somewhere
         * to read from had it accepted the precode. */
        w.Reset();
        for (size_t i = 0; i < toks.size(); i++) {
            w.Advance();
        }
        REQUIRE(w.ok());
        const std::vector<unsigned char> page = w.Finish();

        const OracleAnswer a = AskOracle(page, 4096);
        REQUIRE_CTX(a.status == LIBDEFLATE_BAD_DATA, "shape %u: status %d",
                    shape, a.status);

        GDeflateSchedule s;
        REQUIRE(OpenBlock(s, page));
        GDeflateCodeLengths lens;
        REQUIRE_CTX(!GDeflateReadCodeLengths(s, page.data(), lens), "shape %u",
                    shape);
        /* The flag as well as the return value. This case refused with the
         * schedule left clean until fuzz/fuzz_gdeflate_tables.cpp caught it,
         * and the test that was here asserted only the return - so the two
         * ways a caller can read a verdict disagreed and nothing said so. */
        REQUIRE_CTX(s.failed, "shape %u refused with a clean schedule", shape);
    }
    return 0;
}

/* Truncation part-way through the code-length rounds. The page is cut to the
 * words the header itself needs, so the expansion runs out of stream.
 *
 * THE ORACLE IS NOT ASKED, AND NOT ASKING IT IS THE FINDING. ENSURE_BITS in
 * the pinned fork reads a 32-bit word with no bound check against the end of
 * the page, so handing the reference a truncated page is a heap
 * out-of-bounds read - it would red the sanitizer gate with a defect in the
 * oracle rather than a verdict on this decoder. src/gdeflate_schedule.h
 * refuses the same read by an explicit bound, which is what this asserts. */
int TruncationBoundary(const char* name,
                       const std::vector<unsigned char>& litlen_lens,
                       const std::vector<unsigned char>& dist_lens) {
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = BuildTokens(all);
    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(PlanPrecode(toks, 0, precode_lens, &num_explicit));
    std::vector<unsigned char> page;
    REQUIRE(EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                     std::vector<uint32_t>(), &page));

    /* Every page length from the shortest one that is a page at all up to the
     * whole thing. What is asserted is the shape rather than a number: there
     * is a smallest length the header survives, every length below it is
     * refused, and none above it is - so the sweep pins the boundary instead
     * of pinning that a refusal happened somewhere. Word by word rather than
     * in strides, because the cut that removes exactly the word the last round
     * would have refilled with is the only one that reaches the failure check
     * after the expansion loop, and a coarser sweep steps over it. */
    const uint64_t words = page.size() / 4u;
    const uint64_t shortest = cudec_detail::kGDeflateNumStreams + 1u;
    uint64_t smallest_ok = 0;
    uint64_t refusals = 0;
    for (uint64_t cut = shortest; cut <= words; cut++) {
        std::vector<unsigned char> shorter(
            page.begin(), page.begin() + static_cast<size_t>(cut * 4u));
        GDeflateSchedule s;
        REQUIRE_CTX(OpenBlock(s, shorter), "%s: %llu words", name,
                    static_cast<unsigned long long>(cut));
        GDeflateCodeLengths lens;
        if (GDeflateReadCodeLengths(s, shorter.data(), lens)) {
            /* An accepted read may not leave the schedule failed. The cut that
             * removes exactly the word the last round would have refilled with
             * is where the two can come apart: the expansion finishes, its
             * closing advance runs out of page, and only the check after the
             * loop turns that into a refusal. */
            REQUIRE_CTX(!s.failed,
                        "%s: accepted at %llu words with the schedule failed",
                        name, static_cast<unsigned long long>(cut));
            if (smallest_ok == 0) {
                smallest_ok = cut;
            }
            continue;
        }
        REQUIRE_CTX(s.failed, "%s: %llu words", name,
                    static_cast<unsigned long long>(cut));
        /* A refusal above the boundary would mean the sweep found two
         * regimes rather than one, which is a different defect than the one
         * this test is about and may not pass silently. */
        REQUIRE_CTX(smallest_ok == 0,
                    "%s: refused at %llu words after accepting at %llu", name,
                    static_cast<unsigned long long>(cut),
                    static_cast<unsigned long long>(smallest_ok));
        refusals++;
    }
    REQUIRE_CTX(smallest_ok != 0, "%s", name);
    REQUIRE_CTX(refusals == smallest_ok - shortest,
                "%s: %llu refusals below %llu", name,
                static_cast<unsigned long long>(refusals),
                static_cast<unsigned long long>(smallest_ok));
    REQUIRE_CTX(refusals != 0, "%s", name);
    return 0;
}

/* The sweep over several fixtures rather than one, because where the header's
 * LAST refill falls is a property of the fixture: whether the closing advance
 * of the expansion is the operation that takes the final word decides which
 * refusal a one-word-short page reaches. One fixture pins one of those paths
 * and reads as if it pinned both. */
int RunTruncatedMidRound() {
    const std::vector<unsigned char> one_empty_dist(1, 0);
    if (TruncationBoundary("alphabet", AlphabetLens(), one_empty_dist) != 0) {
        return 1;
    }
    if (TruncationBoundary("extremes", ExtremesLens(),
                           std::vector<unsigned char>(kGDeflateNumDistSyms,
                                                      0)) != 0) {
        return 1;
    }
    std::vector<unsigned char> floors(257, 8);
    floors[255] = 0;
    if (TruncationBoundary("field-floors", floors, one_empty_dist) != 0) {
        return 1;
    }
    std::vector<unsigned char> deep(257, 0);
    for (uint32_t i = 0; i < kGDeflateMaxCodeLen; i++) {
        deep[i] = static_cast<unsigned char>(i + 1);
    }
    deep[kEndOfBlock] = static_cast<unsigned char>(kGDeflateMaxCodeLen);
    return TruncationBoundary("deepest-code", deep, one_empty_dist);
}

/* A repeat run reaching past HLIT + HDIST, which is where this decoder is
 * deliberately stricter than the reference and the divergence is executed
 * rather than argued.
 *
 * The vector's tail is zeroes, so the final code-length token can be widened
 * from the run it needs to a full 138-zero run. The reference absorbs the
 * overrun into the 137 slack entries it keeps after the alphabet, never reads
 * them, and decodes the block to exactly the same bytes as the honest
 * encoding - so it returns SUCCESS and the expected output, which is what
 * makes this a divergence rather than a guess about one. */
int RunOverrunningRunIsRefused() {
    std::vector<unsigned char> litlen_lens(257, 8);
    litlen_lens[255] = 0;
    const std::vector<unsigned char> dist_lens(kGDeflateNumDistSyms, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    std::vector<LenToken> toks = BuildTokens(all);
    REQUIRE(!toks.empty());
    REQUIRE(toks.back().presym == 18);
    const uint32_t honest = toks.back().extra_value;
    toks.back().extra_value = 127; /* 11 + 127 = 138 zeroes */
    REQUIRE(honest < 127);

    std::vector<uint32_t> symbols;
    for (uint32_t b = 0; b < 255; b++) {
        symbols.push_back(b);
    }
    symbols.push_back(kEndOfBlock);

    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(PlanPrecode(toks, 0, precode_lens, &num_explicit));
    std::vector<unsigned char> page;
    REQUIRE(EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                     symbols, &page));

    std::vector<unsigned char> want;
    for (uint32_t b = 0; b < 255; b++) {
        want.push_back(static_cast<unsigned char>(b));
    }
    const OracleAnswer a = AskOracle(page, want.size());
    REQUIRE_CTX(a.status == LIBDEFLATE_SUCCESS, "status %d", a.status);
    REQUIRE(equal_bytes(a.out.data(), want.data(), want.size()));

    GDeflateSchedule s;
    REQUIRE(OpenBlock(s, page));
    GDeflateCodeLengths lens;
    REQUIRE(!GDeflateReadCodeLengths(s, page.data(), lens));
    REQUIRE(s.failed);
    return 0;
}

/* A literal/length vector the code-length rounds recover perfectly and that is
 * not a code: three symbols at length 1 claim more codespace than exists. The
 * read succeeds and the table construction refuses, which is the seam this
 * pins - the refusal has to reach the schedule's failure flag, because a
 * caller that checks only the flag would otherwise carry on with a page whose
 * tables were never built. */
int RunLitLenIsNotACode() {
    std::vector<unsigned char> litlen_lens(257, 0);
    for (uint32_t i = 0; i < 3; i++) {
        litlen_lens[i] = 1;
    }
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<LenToken> toks = BuildTokens(all);

    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    REQUIRE(PlanPrecode(toks, 0, precode_lens, &num_explicit));
    std::vector<unsigned char> page;
    REQUIRE(EmitPage(litlen_lens, dist_lens, toks, precode_lens, num_explicit,
                     std::vector<uint32_t>(), &page));

    const OracleAnswer a = AskOracle(page, 4096);
    REQUIRE_CTX(a.status == LIBDEFLATE_BAD_DATA, "status %d", a.status);

    GDeflateSchedule s;
    REQUIRE(OpenBlock(s, page));
    GDeflateCodeLengths lens;
    /* The vectors themselves come back intact - this is a table failure and
     * not a round failure, and the two are separated on purpose. */
    REQUIRE(GDeflateReadCodeLengths(s, page.data(), lens));
    REQUIRE(!s.failed);
    REQUIRE(equal_bytes(lens.lens, litlen_lens.data(), litlen_lens.size()));

    GDeflateSchedule s2;
    REQUIRE(OpenBlock(s2, page));
    GDeflateCodeLengths lens2;
    GDeflateLitLenTable litlen;
    GDeflateDistTable dist;
    REQUIRE(!GDeflateReadDynamicTables(s2, page.data(), lens2, litlen, dist));
    REQUIRE(s2.failed);
    return 0;
}

/* The alphabet fixture with every repeat code taken out of the encoding: the
 * same vectors, stated one length at a time. It is here because the repeat
 * codes and the explicit lengths are different branches of the same round, and
 * a suite that only ever exercised the compressed form would leave the plain
 * one unread. */
int RunExplicitOnly() {
    const std::vector<unsigned char> litlen_lens = AlphabetLens();
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    std::vector<LenToken> toks;
    for (size_t i = 0; i < all.size(); i++) {
        toks.push_back({all[i], 0, 0});
    }
    std::vector<uint32_t> symbols;
    for (uint32_t b = 0; b < 256; b++) {
        symbols.push_back(b);
    }
    symbols.push_back(kEndOfBlock);
    return CheckFixture("explicit-only", litlen_lens, dist_lens, toks, 0,
                        symbols);
}

/* The longest codeword the format admits, carried through the header rather
 * than handed to the table builder directly: lengths 1..15 spend the codespace
 * exactly, and 15 is both the maximum and the far side of the root
 * accelerator. A header decode that clipped a length would build a different
 * code and the body would not come back. */
int RunDeepestCode() {
    std::vector<unsigned char> litlen_lens(257, 0);
    for (uint32_t i = 0; i < kGDeflateMaxCodeLen; i++) {
        litlen_lens[i] = static_cast<unsigned char>(i + 1);
    }
    litlen_lens[kEndOfBlock] = static_cast<unsigned char>(kGDeflateMaxCodeLen);
    const std::vector<unsigned char> dist_lens(1, 0);
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());

    std::vector<uint32_t> symbols;
    for (uint32_t b = 0; b < kGDeflateMaxCodeLen; b++) {
        symbols.push_back(b);
    }
    symbols.push_back(kEndOfBlock);
    return CheckFixture("deepest-code", litlen_lens, dist_lens,
                        BuildTokens(all), 0, symbols);
}

}  // namespace

int main() {
    if (RunAlphabet() != 0) {
        return 1;
    }
    if (RunExplicitOnly() != 0) {
        return 1;
    }
    if (RunExtremes() != 0) {
        return 1;
    }
    if (RunDeepestCode() != 0) {
        return 1;
    }
    if (RunFieldFloors() != 0) {
        return 1;
    }
    if (RunHclenFloor() != 0) {
        return 1;
    }
    if (RunPermutationIsLoadBearing() != 0) {
        return 1;
    }
    if (RunRepeatWithNothingToRepeat() != 0) {
        return 1;
    }
    if (RunPrecodeIsNotACode() != 0) {
        return 1;
    }
    if (RunTruncatedMidRound() != 0) {
        return 1;
    }
    if (RunOverrunningRunIsRefused() != 0) {
        return 1;
    }
    if (RunLitLenIsNotACode() != 0) {
        return 1;
    }
    std::printf("gdeflate_header_twin: ok\n");
    return 0;
}
