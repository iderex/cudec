/* The structure-aware mutation layer for the two GDeflate fuzz targets (issue
 * #193): one shared generator of structurally valid pages with a hostile
 * interior, plus the switch that turns it off.
 *
 * WHY THE PLAIN BYTE FUZZER NEEDS HELP HERE AND DID NOT FOR LZ4 OR SNAPPY. A
 * GDeflate page is not a byte stream with a header on the front. It is 32
 * least-significant-bit-first streams interleaved word by word in the order a
 * decoder's refills happen, so a block header is spread across 32 lanes and
 * every field in it sits at a bit offset that depends on every field before
 * it. Random bytes reach the code-length rounds and die there: the precode
 * vector they describe is over-subscribed almost always, and the decode-table
 * construction, the block loop and the in-tile LZ77 behind it are never
 * entered. That is the plateau the parent issue names, and it is a property of
 * the format rather than of the harnesses.
 *
 * WHAT IS GENERATED, AND WHAT IS DELIBERATELY NOT. The generator emits one
 * final dynamic-Huffman block whose envelope is correct by construction -
 * BFINAL, BTYPE, HLIT, HDIST, HCLEN, a precode that IS a code, and
 * literal/length and distance vectors that are codes - and whose interior is
 * whatever the engine's bytes say. The interior is where the findings are, so
 * nothing about it is repaired: a body may name a length symbol whose extra
 * bits were never written, may run past the output capacity, may end without
 * an end-of-block, and the emitted words may be perturbed afterwards.
 *
 * IT EMITS PAGES, NOT VALID DECODES. A page this produces is one the decoder
 * gets past the header rounds on, which is the whole claim. Whether it then
 * decodes, is refused, or diverges from the reference is what the target
 * asserts, and this file has no opinion about it.
 *
 * THE ROUND ORDER IS NOT RESTATED HERE. Emission is cudec_test::EmitPage in
 * tests/gdeflate_page_writer.h, the same mirror of src/gdeflate_schedule.h the
 * header fixtures drive, reached over the include path fuzz/CMakeLists.txt
 * already adds. A second copy of that order in this directory would drift
 * against the decoder silently, and in the direction where the fuzzer stops
 * reaching anything.
 *
 * THE LAYER IS SWITCHABLE OFF AT RUN TIME, which the parent issue requires:
 * the unstructured corpus keeps running, because a structural generator can
 * only ever reach what its own grammar can express. CUDEC_FUZZ_STRUCTURED=0 in
 * the environment leaves every mutation to libFuzzer's own, and the default is
 * on. An environment switch rather than a build one on purpose - the two arms
 * of the coverage comparison this layer is kept or dropped on have to be the
 * same binary, or the comparison is between two builds. */
#ifndef CUDEC_FUZZ_GDEFLATE_STRUCTURE_H
#define CUDEC_FUZZ_GDEFLATE_STRUCTURE_H

#include "gdeflate_page_writer.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

/* libFuzzer's own mutator, callable only from inside a custom one. */
extern "C" size_t LLVMFuzzerMutate(uint8_t* data, size_t size,
                                   size_t max_size);

namespace cudec_fuzz {

/* splitmix64. The generator needs a reproducible stream from libFuzzer's seed
 * and nothing else; a global rand() would make one target's choices depend on
 * how often the other had been called. */
class Rng {
   public:
    explicit Rng(uint64_t seed) : s_(seed + 0x9E3779B97F4A7C15ull) {}

    uint64_t Next() {
        uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /* Bounded, biased, and that is fine: nothing here is a statistic. */
    uint32_t Below(uint32_t n) {
        return n == 0 ? 0u : static_cast<uint32_t>(Next() % n);
    }

   private:
    uint64_t s_;
};

/* Read once. getenv on every mutation would be a lookup per exec on a path
 * that runs hundreds of thousands of times a minute. */
inline bool StructuredEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("CUDEC_FUZZ_STRUCTURED");
        return v == 0 || std::strcmp(v, "0") != 0;
    }();
    return on;
}

/* A complete canonical code over `alphabet` symbols in which exactly the
 * chosen positions are coded. The shape comes from CompleteLengths, so the
 * codespace is spent exactly and the vector is a code by construction - which
 * is the property the header rounds are being got past, and not a property of
 * the body. */
inline std::vector<unsigned char> LiveCode(uint32_t alphabet,
                                           const std::vector<uint32_t>& live) {
    std::vector<unsigned char> lens(alphabet, 0);
    const std::vector<unsigned char> shape =
        cudec_test::CompleteLengths(static_cast<uint32_t>(live.size()));
    for (size_t i = 0; i < live.size() && i < shape.size(); i++) {
        lens[live[i]] = shape[i];
    }
    return lens;
}

/* Add `sym` if it is not already live. Linear, over a list of at most a few
 * dozen: a set here would cost more than the scan. */
inline void AddLive(std::vector<uint32_t>* live, uint32_t sym) {
    for (size_t k = 0; k < live->size(); k++) {
        if ((*live)[k] == sym) {
            return;
        }
    }
    live->push_back(sym);
}

/* One structurally valid page. `tail_words` is how many zero words follow the
 * emitted ones: the writer's own tail is sized for the reference's unchecked
 * refill in a fixture that decodes, and a fuzz input carrying 16 KiB of it
 * would spend the whole -max_len on padding. A short tail is also the more
 * interesting input, because refill past the last word is a branch this
 * decoder refuses explicitly. Empty on a generator that could not emit, which
 * the caller reads as "no structured mutation this time" rather than as a
 * page. */
inline std::vector<unsigned char> SynthPage(Rng& rng, uint32_t tail_words) {
    /* HLIT is a five-bit field over 257 and HDIST a five-bit field over 1, so
     * the two sizes below are exactly what a header can state. */
    const uint32_t num_lit = 257u + rng.Below(32u);
    const uint32_t num_dist = 1u + rng.Below(30u);

    /* The live literal/length symbols. 256 is always among them: without an
     * end-of-block the reference cannot finish a block, and a generator that
     * never coded one would have a hole in its grammar rather than a hostile
     * interior. */
    std::vector<uint32_t> live_lit;
    live_lit.push_back(256u);
    const uint32_t n_lit = 1u + rng.Below(24u);
    for (uint32_t i = 0; i < n_lit; i++) {
        AddLive(&live_lit, rng.Below(num_lit));
    }
    std::vector<uint32_t> live_dist;
    const uint32_t n_dist = 1u + rng.Below(4u);
    for (uint32_t i = 0; i < n_dist; i++) {
        AddLive(&live_dist, rng.Below(num_dist));
    }

    const std::vector<unsigned char> litlen_lens = LiveCode(num_lit, live_lit);
    const std::vector<unsigned char> dist_lens = LiveCode(num_dist, live_dist);

    std::vector<unsigned char> all(litlen_lens);
    all.insert(all.end(), dist_lens.begin(), dist_lens.end());
    const std::vector<cudec_test::LenToken> toks = cudec_test::BuildTokens(all);

    unsigned char precode_lens[cudec_detail::kGDeflateNumPrecodeSyms];
    uint32_t num_explicit = 0;
    if (!cudec_test::PlanPrecode(toks, 0, precode_lens, &num_explicit)) {
        return std::vector<unsigned char>();
    }

    /* The body. Symbols are drawn from the coded set with no regard for what
     * they mean: a length symbol's extra bits and its distance code are NOT
     * emitted, so a body that draws one walks into the block loop reading the
     * following codewords as extra bits. That is the interior this layer is
     * for. The end-of-block is drawn like any other symbol, so a body that
     * never reaches one runs to the emitted words and stops there. */
    std::vector<uint32_t> body;
    const uint32_t n_body = rng.Below(400u);
    for (uint32_t i = 0; i < n_body; i++) {
        body.push_back(
            live_lit[rng.Below(static_cast<uint32_t>(live_lit.size()))]);
    }

    std::vector<unsigned char> page;
    if (!cudec_test::EmitPage(litlen_lens, dist_lens, toks, precode_lens,
                              num_explicit, body, &page)) {
        return std::vector<unsigned char>();
    }

    /* The writer appends a fixed tail sized for a decoding fixture. What was
     * emitted is everything before it, derived from the writer's own declared
     * constant rather than found by scanning back over zeros - a page whose
     * last emitted word happens to be zero would lose it to a scan. */
    const size_t writer_tail = cudec_test::kGDeflateWriterTailWords * 4u;
    if (page.size() < writer_tail) {
        return std::vector<unsigned char>();
    }
    page.resize(page.size() - writer_tail + tail_words * 4u, 0);
    return page;
}

/* Perturb the bytes after `keep`, in place. This is the second half of "valid
 * envelope, hostile interior": the words a page got through its header rounds
 * on are left alone and the ones behind them are damaged, which reaches the
 * block loop with a table that was built rather than refused. */
inline void PerturbTail(Rng& rng, uint8_t* data, size_t size, size_t keep) {
    if (size <= keep) {
        return;
    }
    const uint32_t flips = 1u + rng.Below(8u);
    for (uint32_t i = 0; i < flips; i++) {
        const size_t at = keep + rng.Below(static_cast<uint32_t>(size - keep));
        data[at] = static_cast<uint8_t>(data[at] ^ (1u << rng.Below(8u)));
    }
}

}  // namespace cudec_fuzz

#endif /* CUDEC_FUZZ_GDEFLATE_STRUCTURE_H */
