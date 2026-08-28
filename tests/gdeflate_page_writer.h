/* A GDeflate page writer, for tests only (issue #175). It exists because the
 * reference's answer to "is this a code?" is not callable: the pinned fork
 * compiles `gdeflate_decompress.c` with `HIDE_INTERFACE` defined, so
 * `build_decode_table` is static inside that translation unit and the plain
 * DEFLATE entry points are not compiled at all. The only door into the
 * reference's table construction is a whole GDeflate page, so a test that
 * wants the reference's verdict on a chosen code-length vector has to emit
 * one.
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
#include <vector>

namespace cudec_test {

/* One byte per bit while the page is being built. A page here is a few hundred
 * symbols, so the shape that is easiest to reason about wins over the shape
 * that is compact. */
using LaneBits = std::vector<unsigned char>;

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
    /* See the file comment: slack for the reference's unchecked refill, not a
     * property of the format. Sized for the fixture that deliberately drives
     * the reference past the emitted words - an all-zero literal/length vector
     * is an empty code, which the reference resolves to a synthetic literal
     * that consumes one bit and produces one byte until the output buffer is
     * full, so the words it asks for are bounded by the output cap and not by
     * anything the page said. 16 KiB of zeros covers that cap with room, and a
     * page here is still a quarter of the 64 KiB the format allows. */
    static constexpr uint32_t kTailWords = 4096;

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

}  // namespace cudec_test

#endif /* CUDEC_TESTS_GDEFLATE_PAGE_WRITER_H */
