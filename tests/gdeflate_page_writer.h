/* A GDeflate page writer, for the tests and the bench (issues #175, #226,
 * #430). It exists because the reference's answer to "is this a code?" is
 * not callable: the pinned fork compiles `gdeflate_decompress.c` with
 * `HIDE_INTERFACE` defined, so
 * `build_decode_table` is static inside that translation unit and the plain
 * DEFLATE entry points are not compiled at all. The only door into the
 * reference's table construction is a whole GDeflate page, so a test that
 * wants the reference's verdict on a chosen code-length vector has to emit
 * one.
 *
 * WHO READS IT. tests/ drives it for the fixtures below; fuzz/ drives it
 * through fuzz/gdeflate_structure.h; and bench/bench_gdeflate.cpp drives it for
 * the adversarial corpora (#226, #430), which is why the header is not
 * test-only any more. The bench consumer is also the one that broke the size
 * assumption stated at LaneBits below.
 *
 * IT EMITS SEVERAL BLOCKS INTO ONE PAGE (#430) AND IT IS THE ONLY THING THAT
 * DOES. Every multi-block page this project had seen before came out of the
 * reference's compressor, so where a block ended was the compressor's decision
 * and both decoders merely followed it. Writing one means holding the
 * INTER-BLOCK round order, which is at EmitBlock below and is read off
 * src/gdeflate_block.h rather than chosen.
 *
 * WHAT MAKES THIS SAFE TO TRUST. A writer that is wrong produces pages the
 * reference rejects, and a reject is exactly what a negative test is looking
 * for - so a broken writer reads as a passing suite. Two things stop that.
 * Every negative below is paired with a positive that differs in one field and
 * must DECODE, so a writer that emitted nothing usable fails the pair rather
 * than passing it. And the positive pages carry a symbol sequence chosen by
 * the test and are required to come back byte-identical, which no accidental
 * bit layout produces.
 *
 * IT IS THE MIRROR OF src/gdeflate_schedule.h AND NOT A SECOND SCHEDULE. The
 * decoder's rules are: one shared word cursor; a lane is refilled when it
 * falls below the 32-bit watermark and only ever by the operation that is
 * looking at it; ADVANCE refills the current lane and then rotates; RESET
 * returns to lane 0 consuming nothing. A page is therefore 32 independent
 * least-significant-bit-first streams plus the order in which their words were
 * handed out, and this writer builds exactly those two things: bits go into
 * the current lane's stream, and every refill the decoder will perform appends
 * that lane to the word order. Assembly at the end is mechanical.
 *
 * THE TAIL IS FOR THE REFERENCE, NOT FOR THE FORMAT. `ENSURE_BITS` in the
 * fork's `gdeflate_decompress_template.h` reads a 32-bit word with no bound
 * check against the end of the page, so a page whose emission this writer got
 * wrong by one refill would have the reference read off the end of a heap
 * buffer, and the sanitized CI job would report a writer bug as an
 * out-of-bounds read in the oracle. The zero tail below turns that into a read
 * of zeros. cudec's own schedule refuses the same read by an explicit bound
 * (`GDeflateEnsure`), which is the distinction docs/MASTERPLAN.md draws
 * between a padding convention and a check. */
#ifndef CUDEC_TESTS_GDEFLATE_PAGE_WRITER_H
#define CUDEC_TESTS_GDEFLATE_PAGE_WRITER_H

#include "gdeflate_tables.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace cudec_test {

/* One byte per bit while the page is being built: the shape that is easiest to
 * reason about wins over the shape that is compact.
 *
 * THE SENTENCE THAT USED TO BOUND IT IS GONE. It said a page here is a few
 * hundred symbols, and the adversarial corpus (#226) drives whole 64 KiB pages
 * through this writer instead - tens of thousands of tokens, and one byte of
 * lane storage for every bit the page spends, so of the order of a megabyte
 * for one page. The shape still stands, because a page is built and released
 * one at a time. No figure is quoted here: a cost that moves with the caller
 * is not a property of this header, and the caller that pays it prints its own
 * numbers with its own methodology. */
using LaneBits = std::vector<unsigned char>;

/* Slack for the reference's unchecked refill, not a property of the format.
 * Sized for the fixture that deliberately drives the reference past the
 * emitted words - an all-zero literal/length vector is an empty code, which
 * the reference resolves to a synthetic literal that consumes one bit and
 * produces one byte until the output buffer is full, so the words it asks for
 * are bounded by the output cap and not by anything the page said. 16 KiB of
 * zeros covers that cap with room, and a page here is still a quarter of the
 * 64 KiB the format allows.
 *
 * DECLARED HERE RATHER THAN INSIDE THE WRITER because a caller that wants the
 * emitted words WITHOUT this tail has to subtract it, and the alternative is
 * scanning back over trailing zeros - which silently eats an emitted word that
 * happened to be zero. fuzz/gdeflate_structure.h is that caller (issue #193).
 */
constexpr uint32_t kGDeflateWriterTailWords = 4096;

class GDeflatePageWriter {
   public:
    GDeflatePageWriter() {
        for (uint32_t n = 0; n < kStreams; n++) {
            bitsleft_[n] = 0;
        }
        idx_ = 0;
        ok_ = true;
        /* The priming round the format opens with: one word into each lane, in
         * lane order (src/gdeflate_schedule.h, draft section 5.3). It is the
         * decoder's own opening loop of 32 ADVANCEs, mirrored. */
        for (uint32_t n = 0; n < kStreams; n++) {
            Advance();
        }
    }

    /* RESET(): back to lane 0, nothing consumed. */
    void Reset() { idx_ = 0; }

    /* ENSURE_BITS(LOW_WATERMARK_BITS): the current lane takes the next word if
     * it has fallen below the watermark. */
    void Ensure() {
        if (bitsleft_[idx_] >= kWatermark) {
            return;
        }
        order_.push_back(idx_);
        bitsleft_[idx_] += kPacket;
    }

    /* ADVANCE(): ensure, then rotate. */
    void Advance() {
        Ensure();
        idx_ = (idx_ + 1u) % kStreams;
    }

    /* POP_BITS(n) seen from the other side: n bits of `value`, least
     * significant first, into the current lane. A lane that does not hold n
     * bits at this point means the emission sequence has drifted from the
     * decoder's, which is a bug in the test rather than in anything under
     * test - so it is recorded and asserted on, never silently corrected. */
    void Push(uint32_t n, uint32_t value) {
        if (n > bitsleft_[idx_]) {
            ok_ = false;
            return;
        }
        for (uint32_t i = 0; i < n; i++) {
            lane_[idx_].push_back(static_cast<unsigned char>((value >> i) & 1u));
        }
        bitsleft_[idx_] -= n;
    }

    /* A Huffman codeword: the canonical code most significant bit first, which
     * is the order the decoder rebuilds it in, so the value handed to Push is
     * the codeword reversed. */
    void PushCode(uint32_t code, uint32_t len) {
        Push(len, cudec_detail::GDeflateReverseBits(code, len));
    }

    bool ok() const { return ok_; }

    /* Which lane the next Push rides. Exposed because a body carrying matches
     * has a per-lane invariant - a lane holding a reservation reads a distance
     * on its next round and nothing else - and the emitter below cannot check
     * that invariant without naming the lane. Nothing here decides anything on
     * it; it is read, never written. */
    uint32_t lane() const { return idx_; }

    /* Lane streams padded to whole words, then interleaved in the order the
     * decoder took them. */
    std::vector<unsigned char> Finish() const {
        std::vector<std::vector<uint32_t>> words(kStreams);
        for (uint32_t n = 0; n < kStreams; n++) {
            const LaneBits& b = lane_[n];
            for (size_t i = 0; i < b.size(); i += kPacket) {
                uint32_t w = 0;
                for (uint32_t k = 0; k < kPacket && i + k < b.size(); k++) {
                    w |= static_cast<uint32_t>(b[i + k]) << k;
                }
                words[n].push_back(w);
            }
        }
        std::vector<size_t> taken(kStreams, 0);
        std::vector<unsigned char> page;
        page.reserve((order_.size() + kTailWords) * 4u);
        for (size_t i = 0; i < order_.size(); i++) {
            const uint32_t lane = order_[i];
            const uint32_t w =
                taken[lane] < words[lane].size() ? words[lane][taken[lane]] : 0u;
            taken[lane]++;
            page.push_back(static_cast<unsigned char>(w & 0xFFu));
            page.push_back(static_cast<unsigned char>((w >> 8) & 0xFFu));
            page.push_back(static_cast<unsigned char>((w >> 16) & 0xFFu));
            page.push_back(static_cast<unsigned char>((w >> 24) & 0xFFu));
        }
        page.resize(page.size() + kTailWords * 4u, 0);
        return page;
    }

   private:
    static constexpr uint32_t kStreams = 32;
    static constexpr uint32_t kPacket = 32;
    static constexpr uint32_t kWatermark = 32;
    static constexpr uint32_t kTailWords = kGDeflateWriterTailWords;

    LaneBits lane_[kStreams];
    uint32_t bitsleft_[kStreams];
    uint32_t idx_;
    std::vector<uint32_t> order_;
    bool ok_;
};

/* Code lengths for a complete canonical code over `n` symbols, the shape every
 * fixture here needs and none of them should invent twice. Two lengths suffice:
 * with b = ceil(log2(n)), giving 2^b - n symbols length b-1 and the rest length
 * b spends exactly the whole codespace, since 2(2^b - n) + (2n - 2^b) = 2^b.
 * n = 1 is the degenerate single-symbol code the reference admits at length 1,
 * and it is returned rather than refused because a fixture that needs it is
 * pinning that admission. */
inline std::vector<unsigned char> CompleteLengths(uint32_t n) {
    std::vector<unsigned char> lens;
    if (n == 0) {
        return lens;
    }
    if (n == 1) {
        lens.push_back(1);
        return lens;
    }
    uint32_t b = 0;
    while ((1u << b) < n) {
        b++;
    }
    const uint32_t short_count = (1u << b) - n;
    for (uint32_t i = 0; i < n; i++) {
        lens.push_back(static_cast<unsigned char>(i < short_count ? b - 1 : b));
    }
    return lens;
}

/* The codeword a built table assigns to `sym`, recovered from the same three
 * arrays the decoder walks. Returns false for a symbol the vector left out.
 * It lives beside the writer because every fixture that emits a body needs it
 * and, like CompleteLengths above, none of them should invent it twice. */
template <typename Table>
inline bool CodewordOf(const Table& t, const unsigned char* lens, uint32_t sym,
                       uint32_t* code, uint32_t* len) {
    const uint32_t l = lens[sym];
    if (l == 0) {
        return false;
    }
    if (t.kind == cudec_detail::kGDeflateTableSingle) {
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

/* THE DYNAMIC-BLOCK EMITTER BELOW WAS LIFTED OUT OF ONE FIXTURE FILE (issue
 * #193). It emitted the pages tests/gdeflate_header_twin.cpp needed and lived
 * in that file's anonymous namespace, so the fuzz layer that now needs the
 * same pages could only have had a second copy of the decoder's round order -
 * the one thing this header exists to hold exactly once. What moved is the
 * whole emitter and not a slice of it: a token stream, the precode plan the
 * permutation forces, and the page those two produce. Nothing about it
 * changed in the move, which is why the fixtures that drive it are unchanged
 * too.
 */

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
inline std::vector<LenToken> BuildTokens(const std::vector<unsigned char>& all) {
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
inline bool PlanPrecode(const std::vector<LenToken>& toks, uint32_t forced_explicit,
                 unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms],
                 uint32_t* num_explicit) {
    unsigned char used[cudec_detail::kGDeflateNumPrecodeSyms];
    std::memset(used, 0, sizeof(used));
    for (size_t i = 0; i < toks.size(); i++) {
        if (toks[i].presym >= cudec_detail::kGDeflateNumPrecodeSyms) {
            return false;
        }
        used[toks[i].presym] = 1;
    }
    uint32_t n_used = 0;
    for (uint32_t v = 0; v < cudec_detail::kGDeflateNumPrecodeSyms; v++) {
        n_used += used[v];
    }
    const std::vector<unsigned char> shape = CompleteLengths(n_used);
    std::memset(precode_lens, 0, cudec_detail::kGDeflateNumPrecodeSyms);
    uint32_t k = 0;
    for (uint32_t v = 0; v < cudec_detail::kGDeflateNumPrecodeSyms; v++) {
        if (used[v]) {
            precode_lens[v] = shape[k++];
        }
    }
    /* The permutation is what decides how many lengths the header must state:
     * a symbol at position p is only reachable when HCLEN + 4 exceeds p. */
    uint32_t needed = 4;
    for (uint32_t i = 0; i < cudec_detail::kGDeflateNumPrecodeSyms; i++) {
        if (used[cudec_detail::GDeflatePrecodeOrder(i)] && i + 1u > needed) {
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

/* One element of the BODY stream. A literal or a length symbol reads out of
 * the literal/length code; the distance symbol a reserved match needs reads
 * out of the distance code, on the same lane one round of its own later
 * (src/gdeflate_block.h: a round whose lane holds a pending copy reads the
 * distance and nothing else). Both kinds can carry the extra-bit field that
 * follows them in the SAME round, which is what lets a body carry matches at
 * all rather than literals only. */
struct BodyToken {
    uint32_t sym;
    bool from_dist;
    uint32_t extra_bits;
    uint32_t extra_value;
};

/* One block of a page: the header fields the writer must state and the body it
 * must emit. BFINAL is deliberately NOT among them - which block is final is
 * its position in the sequence handed to EmitPageBlocks, so a caller cannot
 * declare one thing in the field and emit another after it. */
struct PageBlock {
    std::vector<unsigned char> litlen_lens;
    std::vector<unsigned char> dist_lens;
    std::vector<LenToken> toks;
    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit;
    std::vector<BodyToken> body;
};

/* Emit one dynamic block into a page already in progress. Every round here
 * mirrors gdeflate_decompress_template.h in the pinned fork and the block loop
 * in src/gdeflate_block.h - the header fields ride lane 0 with no round
 * between them, each precode length is its own round, and a repeat code's
 * extra bits are pushed before the round ends.
 *
 * THE INTER-BLOCK ORDER IS THE RESET THIS FUNCTION OPENS WITH AND THE DRAIN IT
 * CLOSES WITH, and both are read off the decoder rather than chosen here.
 * src/gdeflate_block.h enters every block with GDeflateReset - back to lane 0,
 * consuming nothing - and leaves each compressed block through 32 rounds that
 * retire whatever is still outstanding. So a block's last emitted round and
 * the next block's first are separated by exactly the drain, and a second
 * block starts on lane 0 exactly as the first one did.
 *
 * THE BODY IS A TOKEN STREAM AND NOT A SYMBOL LIST, because a match is two
 * rounds on one lane rather than one symbol: the token at position p rides
 * lane p mod 32, so a length token at p has its distance token at p+32 by
 * construction. A caller that gets that spacing wrong emits a page the
 * reference reads differently from this writer, which is why the spacing is
 * REFUSED here rather than left to the caller: the decoder reads a distance
 * and nothing else on a lane that holds a reservation, so a distance token
 * arriving on a lane that holds none - or a symbol arriving on a lane that
 * does - is the two sides having drifted apart. */
inline bool EmitBlock(GDeflatePageWriter& w, const PageBlock& b,
                      bool final_block) {
    cudec_detail::GDeflatePrecodeTable precode;
    if (!cudec_detail::GDeflateBuildTable(b.precode_lens,
                                          cudec_detail::kGDeflateNumPrecodeSyms,
                                          precode)) {
        return false;
    }
    cudec_detail::GDeflateLitLenTable litlen;
    const bool litlen_ok = cudec_detail::GDeflateBuildTable(
        b.litlen_lens.data(), static_cast<uint32_t>(b.litlen_lens.size()),
        litlen);
    cudec_detail::GDeflateDistTable dist;
    const bool dist_ok = cudec_detail::GDeflateBuildTable(
        b.dist_lens.data(), static_cast<uint32_t>(b.dist_lens.size()), dist);

    w.Reset();
    w.Push(1, final_block ? 1u : 0u); /* BFINAL */
    w.Push(2, kBlockTypeDynamic);     /* BTYPE */
    w.Ensure();
    w.Push(5, static_cast<uint32_t>(b.litlen_lens.size()) - 257u); /* HLIT */
    w.Push(5, static_cast<uint32_t>(b.dist_lens.size()) - 1u);     /* HDIST */
    w.Push(4, b.num_explicit - 4u);                                /* HCLEN */
    w.Ensure();
    for (uint32_t i = 0; i < b.num_explicit; i++) {
        w.Push(3, b.precode_lens[cudec_detail::GDeflatePrecodeOrder(i)]);
        w.Advance();
    }
    w.Reset();
    for (size_t i = 0; i < b.toks.size(); i++) {
        uint32_t code = 0;
        uint32_t len = 0;
        if (!CodewordOf(precode, b.precode_lens, b.toks[i].presym, &code,
                        &len)) {
            return false;
        }
        w.PushCode(code, len);
        if (b.toks[i].extra_bits != 0) {
            w.Push(b.toks[i].extra_bits, b.toks[i].extra_value);
        }
        w.Advance();
    }

    if (b.body.empty()) {
        /* A header and nothing after it, which is what the header negatives
         * need. It can only be the LAST block of a page: a block that emits no
         * end-of-block leaves no boundary for a successor to start after, so a
         * page claiming one would be emitting a block the decoder never
         * reaches the front of. */
        return final_block;
    }
    if (!litlen_ok) {
        return false;
    }
    /* The body starts at a reset exactly as the header did, so the first
     * symbol rides lane 0 (src/gdeflate_block.h: GDeflateReset between the
     * tables and the round loop). */
    w.Reset();
    /* Which lanes hold a reservation, mirroring GDeflatePageState::copies. */
    bool pending[cudec_detail::kGDeflateNumStreams];
    for (uint32_t n = 0; n < cudec_detail::kGDeflateNumStreams; n++) {
        pending[n] = false;
    }
    bool ended = false;
    for (size_t i = 0; i < b.body.size(); i++) {
        const BodyToken& t = b.body[i];
        const uint32_t lane = w.lane();
        if (t.from_dist != pending[lane]) {
            return false;
        }
        if (t.from_dist && !dist_ok) {
            return false;
        }
        uint32_t code = 0;
        uint32_t len = 0;
        const bool found =
            t.from_dist
                ? CodewordOf(dist, b.dist_lens.data(), t.sym, &code, &len)
                : CodewordOf(litlen, b.litlen_lens.data(), t.sym, &code, &len);
        if (!found) {
            return false;
        }
        w.PushCode(code, len);
        if (t.extra_bits != 0) {
            w.Push(t.extra_bits, t.extra_value);
        }
        if (t.from_dist) {
            pending[lane] = false;
        } else if (t.sym == kEndOfBlock) {
            /* The reference leaves the decode loop on end-of-block without
             * advancing, then runs 32 rounds to drain deferred copies.
             * Those rounds refill, so the writer owes their words. */
            ended = true;
            break;
        } else if (t.sym > kEndOfBlock) {
            pending[lane] = true;
        }
        w.Advance();
    }
    for (uint32_t n = 0; n < cudec_detail::kGDeflateNumStreams; n++) {
        if (pending[n]) {
            /* A lane still holding a reservation when the block ended reads a
             * distance symbol in the drain, and the drain below emits no bits
             * for it - so the decoder would take the next block's header, or
             * the tail, as that distance. The words this writer hands out and
             * the words the decoder asks for still MATCH in that case, which
             * is why nothing downstream catches it and it is caught here. */
            return false;
        }
    }
    if (!final_block && !ended) {
        /* A block with a successor must state its own end. Without the
         * end-of-block symbol the decoder reads on past the tokens this block
         * declared, and where the next block's header begins is then decided
         * by the bits rather than by this sequence. */
        return false;
    }
    for (uint32_t i = 0; i < cudec_detail::kGDeflateNumStreams; i++) {
        w.Advance();
    }
    return true;
}

/* Emit one whole page out of `blocks`, in order, with BFINAL set on the last
 * of them and clear on every other. One block is the shape every fixture
 * written before issue #430 needs, and is what EmitPageTokens below asks
 * for. */
inline bool EmitPageBlocks(const std::vector<PageBlock>& blocks,
                           std::vector<unsigned char>* page) {
    if (blocks.empty()) {
        return false;
    }
    GDeflatePageWriter w;
    for (size_t i = 0; i < blocks.size(); i++) {
        if (!EmitBlock(w, blocks[i], i + 1u == blocks.size())) {
            return false;
        }
    }
    if (!w.ok()) {
        return false;
    }
    *page = w.Finish();
    return true;
}

/* One final dynamic block, which is the whole page. The round order is
 * EmitBlock's above rather than a second copy of it. */
inline bool EmitPageTokens(
    const std::vector<unsigned char>& litlen_lens,
    const std::vector<unsigned char>& dist_lens,
    const std::vector<LenToken>& toks,
    const unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms],
    uint32_t num_explicit, const std::vector<BodyToken>& body,
    std::vector<unsigned char>* page) {
    PageBlock block;
    block.litlen_lens = litlen_lens;
    block.dist_lens = dist_lens;
    block.toks = toks;
    std::memcpy(block.precode_lens, precode_lens, sizeof(block.precode_lens));
    block.num_explicit = num_explicit;
    block.body = body;
    return EmitPageBlocks(std::vector<PageBlock>(1, block), page);
}

/* The literals-only body, which is what every fixture written before matches
 * were expressible needs. One token per symbol, no extra bits, no distance
 * code read - so this is the same page it always emitted, and the round order
 * is still the one above rather than a second copy of it. */
inline bool EmitPage(const std::vector<unsigned char>& litlen_lens,
              const std::vector<unsigned char>& dist_lens,
              const std::vector<LenToken>& toks,
              const unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms],
              uint32_t num_explicit, const std::vector<uint32_t>& symbols,
              std::vector<unsigned char>* page) {
    std::vector<BodyToken> body;
    body.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); i++) {
        body.push_back(BodyToken{symbols[i], false, 0, 0});
    }
    return EmitPageTokens(litlen_lens, dist_lens, toks, precode_lens,
                          num_explicit, body, page);
}

}  // namespace cudec_test

#endif /* CUDEC_TESTS_GDEFLATE_PAGE_WRITER_H */
