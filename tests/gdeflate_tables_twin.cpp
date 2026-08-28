/* The GDeflate canonical table construction, held against the reference
 * (issue #175). src/gdeflate_tables.h turns a code-length vector into a decode
 * table or refuses it; this file establishes that it assigns the same codeword
 * to every symbol the reference does, and that it refuses every vector the
 * reference refuses.
 *
 * HOW PARITY IS ESTABLISHED HERE, AND WHY IT IS NOT A TABLE COMPARISON. The
 * reference's table is not reachable. The pinned fork compiles
 * `gdeflate_decompress.c` with `HIDE_INTERFACE`, so `build_decode_table` is
 * static in that translation unit and the plain DEFLATE entry points are not
 * compiled at all; the only door is `libdeflate_gdeflate_decompress` and a
 * whole page. So parity is established in the direction that door opens: the
 * test builds a table with the header under test, ENCODES a chosen symbol
 * sequence with the codewords that table assigns, and requires the reference
 * to hand back exactly the bytes those symbols name. A single wrong codeword
 * anywhere in the alphabet produces a different byte or a refusal, and cannot
 * produce the expected output. That is a stronger statement than comparing two
 * tables field by field, because it runs through the reference's real decode
 * path rather than through a copy of it.
 *
 * WHAT IS NOT COVERED, AND IT IS THE ISSUE'S FIRST BULLET. The corpus half -
 * "for every dynamic block in the oracle-generated corpus" - needs the code
 * length vectors that sit inside blocks the reference's own compressor
 * produced, and reading one out of a page IS the dynamic block header decode,
 * which is #176 and does not exist. Nothing in this file reads a
 * compressor-produced page. The alphabet is covered by construction instead:
 * the fixtures below encode every literal and the end-of-block symbol, and
 * codeword lengths from 1 to 15, which is both sides of the root accelerator.
 *
 * WHAT THE LENGTH AND DISTANCE SYMBOLS ARE NOT. Emitting a length symbol means
 * emitting its extra bits and the distance symbol that follows it in the next
 * round, and then the in-tile LZ77 copy that pair names - the block loop,
 * which is #182. The distance code is therefore exercised here through its
 * TABLE (built, accepted, refused) and never through a decoded match. Said
 * plainly so the gap is not read as coverage. */
#include "gdeflate_page_writer.h"
#include "gdeflate_tables.h"
#include "require.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::GDeflateBuildTable;
using cudec_detail::GDeflateDecodeSymbol;
using cudec_detail::GDeflateDistTable;
using cudec_detail::GDeflateInit;
using cudec_detail::GDeflateLitLenTable;
using cudec_detail::GDeflateSchedule;
using cudec_detail::kGDeflateNoSymbol;
using cudec_detail::kGDeflateTableEmpty;
using cudec_detail::kGDeflateTableSingle;
using cudec_test::CompleteLengths;
using cudec_test::GDeflatePageWriter;

/* The precode alphabet and the order its lengths are stored in, both from the
 * reference (`deflate_precode_lens_permutation` in
 * gdeflate_decompress_template.h). The permutation is data the writer has to
 * agree with, so it is here rather than derived. */
constexpr uint32_t kNumPrecodeSyms = 19;
constexpr uint32_t kMaxPrecodeLen = 7;
const unsigned char kPrecodePermutation[kNumPrecodeSyms] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

using PrecodeTable =
    cudec_detail::GDeflateHuffTable<kNumPrecodeSyms, kMaxPrecodeLen,
                                    kMaxPrecodeLen>;

/* The codeword a built table assigns to `sym`, recovered from the same three
 * arrays the decoder walks. Returns false for a symbol the vector left out. */
template <typename Table>
bool CodewordOf(const Table& t, const unsigned char* lens, uint32_t sym,
                uint32_t* code, uint32_t* len) {
    const uint32_t l = lens[sym];
    if (l == 0) {
        return false;
    }
    if (t.kind == kGDeflateTableSingle) {
        /* The reference gives the one symbol both codewords, 0 and 1, so
         * either bit resolves it; 0 is the one zlib's decompressor assumes and
         * the one the reference's comment names. */
        *code = 0;
        *len = 1;
        return true;
    }
    for (uint32_t i = 0; i < t.count[l]; i++) {
        if (t.sorted[t.first_index[l] + i] == sym) {
            *code = t.first_code[l] + i;
            *len = l;
            return true;
        }
    }
    return false;
}

/* One dynamic block, emitted whole. `litlen_lens` and `dist_lens` are the
 * vectors under test; `symbols` is what the block's body encodes, which the
 * caller ends with 256 when it wants a block the reference can finish. An
 * empty `symbols` emits the header alone, which is all a vector the reference
 * refuses at table-build time ever gets read. */
bool EmitDynamicBlock(const std::vector<unsigned char>& litlen_lens,
                      const std::vector<unsigned char>& dist_lens,
                      const std::vector<uint32_t>& symbols,
                      std::vector<unsigned char>* page) {
    if (litlen_lens.size() < 257 || litlen_lens.size() > 288 ||
        dist_lens.empty() || dist_lens.size() > 32) {
        return false;
    }

    /* The precode encodes the concatenated vector with explicit lengths only:
     * no repeat codes. Run-length encoding of the vector is the header's
     * business (#176), and a fixture that leaned on it would be pinning that
     * decode rather than this construction. */
    std::vector<unsigned char> all = litlen_lens;
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());

    unsigned char used[16];
    std::memset(used, 0, sizeof(used));
    for (size_t i = 0; i < all.size(); i++) {
        used[all[i]] = 1;
    }
    uint32_t n_used = 0;
    for (uint32_t v = 0; v < 16; v++) {
        n_used += used[v];
    }
    const std::vector<unsigned char> shape = CompleteLengths(n_used);
    unsigned char precode_lens[kNumPrecodeSyms];
    std::memset(precode_lens, 0, sizeof(precode_lens));
    uint32_t k = 0;
    for (uint32_t v = 0; v < 16; v++) {
        if (used[v]) {
            precode_lens[v] = shape[k++];
        }
    }
    PrecodeTable precode;
    if (!GDeflateBuildTable(precode_lens, kNumPrecodeSyms, precode)) {
        return false;
    }

    GDeflateLitLenTable litlen;
    GDeflateDistTable dist;
    const bool litlen_ok =
        GDeflateBuildTable(litlen_lens.data(),
                           static_cast<uint32_t>(litlen_lens.size()), litlen);
    const bool dist_ok = GDeflateBuildTable(
        dist_lens.data(), static_cast<uint32_t>(dist_lens.size()), dist);

    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1); /* BFINAL: one block per page here. */
    w.Push(2, 2); /* BTYPE = DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN. */
    w.Ensure();
    w.Push(5, static_cast<uint32_t>(litlen_lens.size()) - 257u); /* HLIT */
    w.Push(5, static_cast<uint32_t>(dist_lens.size()) - 1u);     /* HDIST */
    w.Push(4, kNumPrecodeSyms - 4u);                             /* HCLEN */
    w.Ensure();
    /* All 19 precode lengths, in the permuted order the format stores them,
     * one per round. */
    for (uint32_t i = 0; i < kNumPrecodeSyms; i++) {
        w.Push(3, precode_lens[kPrecodePermutation[i]]);
        w.Advance();
    }
    w.Reset();
    for (size_t i = 0; i < all.size(); i++) {
        uint32_t code = 0;
        uint32_t len = 0;
        if (!CodewordOf(precode, precode_lens, all[i], &code, &len)) {
            return false;
        }
        w.PushCode(code, len);
        w.Advance();
    }

    if (!symbols.empty()) {
        if (!litlen_ok || !dist_ok) {
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
            if (symbols[i] == 256) {
                /* The reference leaves the decode loop on the end-of-block
                 * symbol without advancing, then runs 32 rounds to drain the
                 * deferred copies. Those rounds refill, so the writer owes
                 * their words. */
                break;
            }
            w.Advance();
        }
        for (uint32_t i = 0; i < 32; i++) {
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

OracleAnswer AskOracle(const std::vector<unsigned char>& page,
                       size_t out_cap) {
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
    a.status = libdeflate_gdeflate_decompress(d, &in, 1, a.out.data(), out_cap,
                                              &produced);
    libdeflate_free_gdeflate_decompressor(d);
    a.out.resize(a.status == LIBDEFLATE_SUCCESS ? produced : 0);
    return a;
}

/* A literal/length vector over 257 symbols, every one of 0..256 coded. This is
 * the alphabet fixture: every literal the format has, plus end-of-block. */
std::vector<unsigned char> AllLiteralsLens() {
    return CompleteLengths(257);
}

/* A literal/length vector whose codeword lengths run 1, 2, ... 15, 15 over
 * symbols 0..14 and 256. The lengths spend the codespace exactly, and they are
 * the fixture that reaches both sides of the root accelerator: 1 through 10
 * resolve in one lookup, 11 through 15 fall through to the length-by-length
 * walk. A vector of uniform lengths would exercise one of the two and read
 * exactly like a vector that exercised both. */
std::vector<unsigned char> DeepCodeLens() {
    std::vector<unsigned char> lens(257, 0);
    for (uint32_t i = 0; i < 15; i++) {
        lens[i] = static_cast<unsigned char>(i + 1);
    }
    lens[256] = 15;
    return lens;
}

int RunAlphabetParity() {
    /* Every literal, in order, then end-of-block. */
    std::vector<uint32_t> symbols;
    for (uint32_t b = 0; b < 256; b++) {
        symbols.push_back(b);
    }
    symbols.push_back(256);

    std::vector<unsigned char> page;
    /* HDIST is at least one symbol; a block with no matches declares a
     * distance code with nothing in it, which is the reference's empty-code
     * admission and is pinned again below on its own. */
    const std::vector<unsigned char> no_dist(1, 0);
    REQUIRE(EmitDynamicBlock(AllLiteralsLens(), no_dist, symbols, &page));

    const OracleAnswer a = AskOracle(page, 4096);
    REQUIRE_CTX(a.status == LIBDEFLATE_SUCCESS, "status %d", a.status);
    REQUIRE(a.out.size() == 256);
    std::vector<unsigned char> want(256);
    for (uint32_t b = 0; b < 256; b++) {
        want[b] = static_cast<unsigned char>(b);
    }
    REQUIRE(equal_bytes(a.out.data(), want.data(), want.size()));
    return 0;
}

int RunDeepCodeParity() {
    std::vector<uint32_t> symbols;
    for (uint32_t b = 0; b < 15; b++) {
        symbols.push_back(b);
    }
    symbols.push_back(256);

    std::vector<unsigned char> page;
    const std::vector<unsigned char> no_dist(1, 0);
    REQUIRE(EmitDynamicBlock(DeepCodeLens(), no_dist, symbols, &page));

    const OracleAnswer a = AskOracle(page, 4096);
    REQUIRE_CTX(a.status == LIBDEFLATE_SUCCESS, "status %d", a.status);
    REQUIRE(a.out.size() == 15);
    std::vector<unsigned char> want(15);
    for (uint32_t b = 0; b < 15; b++) {
        want[b] = static_cast<unsigned char>(b);
    }
    REQUIRE(equal_bytes(a.out.data(), want.data(), want.size()));

    /* And the same lengths through the header under test, so the claim that
     * the walk was reached is a fact about this table rather than about the
     * fixture's intent: symbol 14 carries a 15-bit codeword, which is five
     * bits past the 10-bit root. */
    GDeflateLitLenTable t;
    const std::vector<unsigned char> lens = DeepCodeLens();
    REQUIRE(GDeflateBuildTable(lens.data(), 257u, t));
    uint32_t code = 0;
    uint32_t len = 0;
    REQUIRE(CodewordOf(t, lens.data(), 14u, &code, &len));
    REQUIRE(len == 15);
    REQUIRE(CodewordOf(t, lens.data(), 0u, &code, &len));
    REQUIRE(len == 1);
    return 0;
}

/* The length-by-length walk, written independently of the header under test so
 * that the root accelerator has something to be compared against. `bits` is a
 * probe as the lane hands it over: least significant bit first, which is the
 * most significant bit of the codeword. */
template <typename Table>
uint32_t WalkSymbol(const Table& t, uint32_t bits, uint32_t nbits,
                    uint32_t* used_len) {
    uint32_t code = 0;
    for (uint32_t len = 1; len <= nbits; len++) {
        code = (code << 1) | ((bits >> (len - 1)) & 1u);
        if (t.count[len] != 0 && code >= t.first_code[len] &&
            (code - t.first_code[len]) < t.count[len]) {
            *used_len = len;
            return t.sorted[t.first_index[len] + (code - t.first_code[len])];
        }
    }
    return kGDeflateNoSymbol;
}

/* The root accelerator is only an accelerator: a slot it fails to fill falls
 * through to the walk and still answers correctly, so nothing that decodes can
 * see an under-filled root. That is exactly why it needs its own check. Both
 * halves are asserted - every filled slot agrees with the walk, and the number
 * of filled slots is the closed form the layout claims, which is what an
 * under-filled root fails. */
template <typename Table>
int CheckRoot(const char* what, const Table& t, uint32_t root_bits) {
    uint32_t filled = 0;
    for (uint32_t i = 0; i < (1u << root_bits); i++) {
        const uint32_t entry = t.root[i];
        const uint32_t hit_len = entry & cudec_detail::kGDeflateRootLenMask;
        if (hit_len == 0) {
            continue;
        }
        filled++;
        uint32_t walk_len = 0;
        const uint32_t walk = WalkSymbol(t, i, root_bits, &walk_len);
        REQUIRE_CTX(walk == (entry >> cudec_detail::kGDeflateRootLenBits),
                    "%s: slot %u names %u, the walk says %u", what, i,
                    entry >> cudec_detail::kGDeflateRootLenBits, walk);
        REQUIRE_CTX(walk_len == hit_len, "%s: slot %u length %u vs %u", what, i,
                    hit_len, walk_len);
    }
    uint32_t expect = 0;
    for (uint32_t len = 1; len <= root_bits; len++) {
        expect += static_cast<uint32_t>(t.count[len]) << (root_bits - len);
    }
    REQUIRE_CTX(filled == expect, "%s: %u slots filled, %u owed", what, filled,
                expect);
    return 0;
}

/* Canonical ORDER, asserted directly rather than inferred from the codewords
 * that were decoded through it. A table can hand back the right symbol for
 * every codeword the fixtures happen to emit and still order `sorted` by
 * something else, because the first-index arithmetic and the fill walk the
 * same array - so an ordering defect hides behind exactly the evidence the
 * parity fixtures produce. The property is the reference's own: primarily by
 * increasing codeword length, secondarily by increasing symbol value, with the
 * uncoded symbols left out entirely. */
template <typename Table>
int CheckCanonicalOrder(const char* what, const Table& t,
                        const std::vector<unsigned char>& lens,
                        uint32_t max_len) {
    uint32_t n = 0;
    for (size_t sym = 0; sym < lens.size(); sym++) {
        n += (lens[sym] != 0) ? 1u : 0u;
    }
    uint32_t placed = 0;
    for (uint32_t len = 1; len <= max_len; len++) {
        REQUIRE_CTX(t.first_index[len] == placed, "%s: length %u starts at %u, "
                    "%u symbols placed before it", what, len,
                    static_cast<uint32_t>(t.first_index[len]), placed);
        uint32_t previous = 0;
        for (uint32_t i = 0; i < t.count[len]; i++) {
            const uint32_t sym = t.sorted[placed + i];
            REQUIRE_CTX(sym < lens.size(), "%s: symbol %u is outside the "
                        "alphabet", what, sym);
            REQUIRE_CTX(lens[sym] == len, "%s: symbol %u sits at length %u and "
                        "its length is %u", what, sym, len,
                        static_cast<uint32_t>(lens[sym]));
            REQUIRE_CTX(i == 0 || sym > previous, "%s: symbol %u follows %u at "
                        "length %u", what, sym, previous, len);
            previous = sym;
        }
        placed += t.count[len];
    }
    REQUIRE_CTX(placed == n, "%s: %u symbols placed, %u coded", what, placed, n);
    return 0;
}

int RunCanonicalOrder() {
    const std::vector<unsigned char> flat_lens = AllLiteralsLens();
    GDeflateLitLenTable flat;
    REQUIRE(GDeflateBuildTable(flat_lens.data(), 257u, flat));
    if (CheckCanonicalOrder("all-literals", flat, flat_lens,
                            cudec_detail::kGDeflateMaxCodeLen) != 0) {
        return 1;
    }

    const std::vector<unsigned char> deep_lens = DeepCodeLens();
    GDeflateLitLenTable deep;
    REQUIRE(GDeflateBuildTable(deep_lens.data(), 257u, deep));
    if (CheckCanonicalOrder("deep-code", deep, deep_lens,
                            cudec_detail::kGDeflateMaxCodeLen) != 0) {
        return 1;
    }

    /* A vector whose coded symbols are scattered rather than leading, so that
     * an implementation ordering by symbol index alone, or filling in input
     * order, produces a different array. Symbols 200, 5, 91 and 3 all carry
     * length 2, which spends the codespace exactly. */
    std::vector<unsigned char> scattered(257, 0);
    scattered[200] = 2;
    scattered[5] = 2;
    scattered[91] = 2;
    scattered[3] = 2;
    GDeflateLitLenTable mixed;
    REQUIRE(GDeflateBuildTable(scattered.data(), 257u, mixed));
    REQUIRE(mixed.sorted[0] == 3);
    REQUIRE(mixed.sorted[1] == 5);
    REQUIRE(mixed.sorted[2] == 91);
    REQUIRE(mixed.sorted[3] == 200);
    if (CheckCanonicalOrder("scattered", mixed, scattered,
                            cudec_detail::kGDeflateMaxCodeLen) != 0) {
        return 1;
    }
    return 0;
}

int RunRootAccelerator() {
    GDeflateLitLenTable flat;
    const std::vector<unsigned char> flat_lens = AllLiteralsLens();
    REQUIRE(GDeflateBuildTable(flat_lens.data(), 257u, flat));
    /* Every codeword here is 8 or 9 bits, so the 10-bit root resolves the whole
     * alphabet and no slot may be left over. */
    if (CheckRoot("all-literals", flat, cudec_detail::kGDeflateLitLenRootBits) !=
        0) {
        return 1;
    }

    GDeflateLitLenTable deep;
    const std::vector<unsigned char> deep_lens = DeepCodeLens();
    REQUIRE(GDeflateBuildTable(deep_lens.data(), 257u, deep));
    /* Here five of the sixteen codewords are longer than the root, so the root
     * is deliberately incomplete and the closed form is what says by how much.
     */
    if (CheckRoot("deep-code", deep, cudec_detail::kGDeflateLitLenRootBits) !=
        0) {
        return 1;
    }
    return 0;
}

/* A vector the reference refuses, and the one-field neighbour it accepts. The
 * pair is the point: a writer that emitted nothing usable would fail the
 * accepting half, so a refusal here cannot be read as agreement by accident. */
int RequireRejectedPair(const char* what,
                        const std::vector<unsigned char>& bad_litlen,
                        const std::vector<unsigned char>& bad_dist,
                        const std::vector<unsigned char>& good_litlen,
                        const std::vector<unsigned char>& good_dist) {
    /* The header under test refuses at least one of the two vectors. */
    GDeflateLitLenTable lt;
    GDeflateDistTable dt;
    const bool twin_litlen = GDeflateBuildTable(
        bad_litlen.data(), static_cast<uint32_t>(bad_litlen.size()), lt);
    const bool twin_dist = GDeflateBuildTable(
        bad_dist.data(), static_cast<uint32_t>(bad_dist.size()), dt);
    REQUIRE_CTX(!twin_litlen || !twin_dist, "%s: the twin accepted it", what);

    /* The reference refuses the page carrying them. The body is left out: a
     * vector rejected at table-build time is never read past the header. */
    std::vector<unsigned char> bad_page;
    REQUIRE_CTX(EmitDynamicBlock(bad_litlen, bad_dist, std::vector<uint32_t>(),
                                 &bad_page),
                "%s: could not emit", what);
    const OracleAnswer bad = AskOracle(bad_page, 4096);
    REQUIRE_CTX(bad.status != LIBDEFLATE_SUCCESS, "%s: the oracle accepted it",
                what);

    /* The neighbour decodes, which is what makes the refusal above evidence.
     * It is a full block, so it exercises the same writer end to end. */
    std::vector<uint32_t> symbols;
    symbols.push_back(65);
    symbols.push_back(256);
    std::vector<unsigned char> good_page;
    REQUIRE_CTX(EmitDynamicBlock(good_litlen, good_dist, symbols, &good_page),
                "%s: could not emit the neighbour", what);
    const OracleAnswer good = AskOracle(good_page, 4096);
    REQUIRE_CTX(good.status == LIBDEFLATE_SUCCESS, "%s: neighbour status %d",
                what, good.status);
    REQUIRE(good.out.size() == 1 && good.out[0] == 'A');
    return 0;
}

int RunRejects() {
    const std::vector<unsigned char> good_litlen = AllLiteralsLens();
    const std::vector<unsigned char> no_dist(1, 0);

    /* Over-subscribed: one codeword shortened by a bit, so the lengths claim
     * more codespace than exists. */
    std::vector<unsigned char> over = good_litlen;
    over[0] = static_cast<unsigned char>(over[0] - 1);
    if (RequireRejectedPair("over-subscribed lit/len", over, no_dist,
                            good_litlen, no_dist) != 0) {
        return 1;
    }

    /* Incomplete: one codeword lengthened by a bit, so part of the codespace
     * is unreachable and the vector is neither empty nor the single-symbol
     * shape the reference admits. */
    std::vector<unsigned char> under = good_litlen;
    under[0] = static_cast<unsigned char>(under[0] + 1);
    if (RequireRejectedPair("incomplete lit/len", under, no_dist, good_litlen,
                            no_dist) != 0) {
        return 1;
    }

    /* Incomplete distance code that is not one of the two admitted shapes:
     * two symbols at length 2 spend half the codespace, and count[1] is 0. */
    std::vector<unsigned char> half_dist(2, 2);
    if (RequireRejectedPair("incomplete distance", good_litlen, half_dist,
                            good_litlen, no_dist) != 0) {
        return 1;
    }
    return 0;
}

/* The two incomplete vectors the reference admits, pinned as admissions rather
 * than assumed, and the one place this header is deliberately stricter. */
int RunAdmittedIncomplete() {
    const std::vector<unsigned char> good_litlen = AllLiteralsLens();
    std::vector<uint32_t> symbols;
    symbols.push_back(65);
    symbols.push_back(256);

    /* The empty distance code: one symbol, length 0. Both sides accept the
     * vector, and a block that declares no distance and uses none decodes. */
    const std::vector<unsigned char> empty_dist(1, 0);
    GDeflateDistTable dt;
    REQUIRE(GDeflateBuildTable(empty_dist.data(), 1u, dt));
    REQUIRE(dt.kind == kGDeflateTableEmpty);
    std::vector<unsigned char> page;
    REQUIRE(EmitDynamicBlock(good_litlen, empty_dist, symbols, &page));
    OracleAnswer a = AskOracle(page, 4096);
    REQUIRE_CTX(a.status == LIBDEFLATE_SUCCESS, "empty distance: status %d",
                a.status);

    /* And the strictness that is this header's own: the reference resolves
     * every codeword of an empty table to a synthetic symbol 0 so that a
     * malformed stream still lands on an initialised entry. Decoding from one
     * here refuses instead, which is only reachable by a page that uses a
     * distance it never declared. */
    GDeflateSchedule s;
    const std::vector<unsigned char> probe(256, 0xFF);
    REQUIRE(GDeflateInit(s, probe.data(), probe.size()));
    const uint32_t before = s.bitsleft[0];
    REQUIRE(GDeflateDecodeSymbol(s, dt) == kGDeflateNoSymbol);
    REQUIRE(s.failed);
    /* And it refuses without moving the lane. Every other refusal in this
     * family reaches the same verdict by exhausting the length walk, which
     * costs the lane fifteen bits first; a table with nothing in it is known
     * to be unusable before a single bit is read, and the refusal that reads
     * nothing is the one that keeps "once failed, nothing moves" (the sticky
     * flag rule in src/gdeflate_schedule.h) literally true here. */
    REQUIRE(s.bitsleft[0] == before);

    /* The single-symbol distance code: one symbol at length 1, which spends
     * half the codespace. The reference admits it by name and assigns the
     * symbol both codewords; this pins the admission. Reading the symbol back
     * out needs a decoded match, which is #182. */
    const std::vector<unsigned char> single_dist(1, 1);
    GDeflateDistTable st;
    REQUIRE(GDeflateBuildTable(single_dist.data(), 1u, st));
    REQUIRE(st.kind == kGDeflateTableSingle);
    REQUIRE(EmitDynamicBlock(good_litlen, single_dist, symbols, &page));
    a = AskOracle(page, 4096);
    REQUIRE_CTX(a.status == LIBDEFLATE_SUCCESS, "single distance: status %d",
                a.status);
    REQUIRE(a.out.size() == 1 && a.out[0] == 'A');

    /* One bit resolves it whichever way the bit falls, which is what "both
     * codewords" means where a decoder is concerned. */
    GDeflateSchedule z;
    const std::vector<unsigned char> zeros(256, 0x00);
    REQUIRE(GDeflateInit(z, zeros.data(), zeros.size()));
    REQUIRE(GDeflateDecodeSymbol(z, st) == 0u);
    GDeflateSchedule o;
    const std::vector<unsigned char> ones(256, 0xFF);
    REQUIRE(GDeflateInit(o, ones.data(), ones.size()));
    REQUIRE(GDeflateDecodeSymbol(o, st) == 0u);
    return 0;
}

/* An all-zero literal/length vector is the empty code again, in the one place
 * it can never be right: the reference admits the TABLE and then resolves
 * every codeword to a synthetic literal, so the page runs until the output
 * buffer is full and is refused there. Both sides refuse the page, by
 * different routes, and the route matters enough to say which is which. */
int RunAllZeroLitLen() {
    const std::vector<unsigned char> zero_litlen(257, 0);
    const std::vector<unsigned char> no_dist(1, 0);

    GDeflateLitLenTable lt;
    REQUIRE(GDeflateBuildTable(zero_litlen.data(), 257u, lt));
    REQUIRE(lt.kind == kGDeflateTableEmpty);
    GDeflateSchedule s;
    const std::vector<unsigned char> probe(256, 0x5A);
    REQUIRE(GDeflateInit(s, probe.data(), probe.size()));
    REQUIRE(GDeflateDecodeSymbol(s, lt) == kGDeflateNoSymbol);
    REQUIRE(s.failed);

    std::vector<unsigned char> page;
    REQUIRE(EmitDynamicBlock(zero_litlen, no_dist, std::vector<uint32_t>(),
                             &page));
    const OracleAnswer a = AskOracle(page, 64);
    REQUIRE_CTX(a.status != LIBDEFLATE_SUCCESS, "all-zero lit/len: status %d",
                a.status);
    return 0;
}

/* Two refusals the format cannot deliver, so the reference cannot be asked
 * about them and is not. Explicit code lengths arrive as precode symbols below
 * 16 and an alphabet size is five bits plus a constant, so neither shape can
 * be written into a page at all; both are caller errors on the header's own
 * surface. They are refused rather than clamped because a clamped length
 * describes a different code, silently. */
int RunUnreachableByFormat() {
    /* The fixture is a vector that is a COMPLETE code over the first 257
     * symbols with one extra symbol at length 16, and it is that shape for a
     * reason worth keeping. A vector whose over-long length replaced a coded
     * one is refused either way - the code it leaves behind is incomplete - so
     * it cannot tell the length check from the completeness check. This one
     * can: without the length check the over-long symbol is counted in a slot
     * one past the end of the per-length array, the fifteen lengths that
     * remain still spend the whole codespace, and the vector is accepted. */
    std::vector<unsigned char> too_long = CompleteLengths(257);
    too_long.push_back(16);
    GDeflateLitLenTable lt;
    REQUIRE(!GDeflateBuildTable(too_long.data(), 258u, lt));

    const std::vector<unsigned char> lens = CompleteLengths(257);
    GDeflateDistTable dt;
    REQUIRE(!GDeflateBuildTable(lens.data(), 257u, dt));
    return 0;
}

}  // namespace

int main() {
    if (RunCanonicalOrder() != 0) {
        return 1;
    }
    if (RunAlphabetParity() != 0) {
        return 1;
    }
    if (RunDeepCodeParity() != 0) {
        return 1;
    }
    if (RunRootAccelerator() != 0) {
        return 1;
    }
    if (RunRejects() != 0) {
        return 1;
    }
    if (RunAdmittedIncomplete() != 0) {
        return 1;
    }
    if (RunAllZeroLitLen() != 0) {
        return 1;
    }
    if (RunUnreachableByFormat() != 0) {
        return 1;
    }
    std::printf("gdeflate_tables_twin: ok\n");
    return 0;
}
