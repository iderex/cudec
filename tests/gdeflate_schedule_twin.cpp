/* The GDeflate round/refill schedule under test (issue #178). Three things
 * are asked of src/gdeflate_schedule.h and each has its own section below:
 * that the priming round loads one word into each of the 32 lanes in lane
 * order, that the three off-by-one traps around a refill are answered the way
 * the format answers them, and that a real page emitted by the reference is
 * walked word for word with nothing skipped and nothing taken twice.
 *
 * WHY A STORED-BLOCK CORPUS IS THE TRACE, AND WHAT THAT DOES AND DOES NOT
 * REACH. A GDeflate stored block is the one block type whose whole body is
 * readable through the schedule alone - a 16-bit length and then one 8-bit
 * literal per round, with no Huffman table anywhere (dossier 11.3). So a page
 * of stored blocks can be decoded here end to end, byte-compared against the
 * reference's own output, and checked to land exactly on the page's last word.
 * That is a full trace of the cursor over a real corpus, and it is a corpus of
 * one block type. Dynamic and static pages are walked only as far as their
 * header round, because reading further needs the code-length decode (#176)
 * and the symbol tables (#182); the byte parity that covers them is #182's
 * acceptance test, and it fails exactly when this schedule is wrong.
 *
 * The reference is the pinned NVIDIA/libdeflate fork, reached through its
 * public API only - nothing here reads its internals, and no claim below rests
 * on its source. */
#include "gdeflate_schedule.h"
#include "require.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::GDeflateAdvance;
using cudec_detail::GDeflateBufferedBits;
using cudec_detail::GDeflateEnsure;
using cudec_detail::GDeflateInit;
using cudec_detail::GDeflatePop;
using cudec_detail::GDeflateReset;
using cudec_detail::GDeflateSchedule;
using cudec_detail::kGDeflateBitsPerPacket;
using cudec_detail::kGDeflateLowWatermarkBits;
using cudec_detail::kGDeflateNumStreams;

constexpr uint32_t kMinStreamBytes = kGDeflateNumStreams * 4;

/* A deterministic byte source. Not std::rand: every claim below is a
 * comparison against the reference, and a comparison is only evidence if the
 * bytes are the same on every machine and every run. */
class Lcg {
  public:
    explicit Lcg(uint32_t seed) : state_(seed) {}
    uint32_t Next() {
        state_ = state_ * 1103515245u + 12345u;
        return state_ >> 8;
    }
    unsigned char NextByte() { return static_cast<unsigned char>(Next()); }

  private:
    uint32_t state_;
};

/* A word buffer whose contents identify their own index, so a lane holding
 * the wrong word says which one it took. */
std::vector<unsigned char> IndexedWords(uint32_t count) {
    std::vector<unsigned char> bytes(static_cast<size_t>(count) * 4);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t w = 0xA5000000u | i;
        bytes[i * 4 + 0] = static_cast<unsigned char>(w);
        bytes[i * 4 + 1] = static_cast<unsigned char>(w >> 8);
        bytes[i * 4 + 2] = static_cast<unsigned char>(w >> 16);
        bytes[i * 4 + 3] = static_cast<unsigned char>(w >> 24);
    }
    return bytes;
}

/* ---- 1. The priming round ---- */

int PrimingRound() {
    const std::vector<unsigned char> page = IndexedWords(64);
    GDeflateSchedule s;
    REQUIRE(GDeflateInit(s, page.data(), page.size()));

    /* One word each, in lane order, and the cursor left immediately after
     * them. This is the whole of "lane order" as an executed statement: lane i
     * holds word i and nothing else. */
    REQUIRE(s.cursor == kGDeflateNumStreams);
    REQUIRE(s.idx == 0);
    for (uint32_t n = 0; n < kGDeflateNumStreams; ++n) {
        REQUIRE_CTX(s.bitsleft[n] == kGDeflateBitsPerPacket, "lane %u", n);
        REQUIRE_CTX(s.bitbuf[n] == (0xA5000000u | n), "lane %u holds %llx", n,
                    static_cast<unsigned long long>(s.bitbuf[n]));
    }
    REQUIRE(GDeflateBufferedBits(s) ==
            static_cast<uint64_t>(kGDeflateNumStreams) * kGDeflateBitsPerPacket);

    /* Draft section 5.3: 128 bytes is the floor because the priming round
     * needs 32 words. One word short is refused, and refused before any lane
     * is primed rather than part-way through. */
    const std::vector<unsigned char> short_page = IndexedWords(31);
    GDeflateSchedule t;
    REQUIRE(!GDeflateInit(t, short_page.data(), short_page.size()));
    REQUIRE(t.failed);
    REQUIRE(t.cursor == 0);

    const std::vector<unsigned char> exact = IndexedWords(32);
    REQUIRE(exact.size() == kMinStreamBytes);
    GDeflateSchedule u;
    REQUIRE(GDeflateInit(u, exact.data(), exact.size()));
    REQUIRE(u.cursor == 32);

    /* A trailing partial word is not a word any lane could be handed. */
    GDeflateSchedule v;
    REQUIRE(!GDeflateInit(v, page.data(), page.size() - 1));
    REQUIRE(v.failed);
    return 0;
}

/* ---- 2. The three off-by-one traps ---- */

int TrapFirstRoundAfterTheHeader() {
    /* A block header rides lane 0 and consumes bits without advancing, so the
     * first refill of the page after the priming round lands on a lane that
     * already holds a residue. The trap is where the incoming word is placed:
     * it goes ABOVE the bits still there, not at bit 0. */
    const std::vector<unsigned char> page = IndexedWords(64);
    GDeflateSchedule s;
    REQUIRE(GDeflateInit(s, page.data(), page.size()));
    GDeflateReset(s);

    const uint32_t bfinal_btype = GDeflatePop(s, 3);
    REQUIRE(!s.failed);
    REQUIRE(bfinal_btype == (0xA5000000u & 0x7u));
    REQUIRE(s.bitsleft[0] == kGDeflateBitsPerPacket - 3);

    GDeflateEnsure(s, page.data());
    REQUIRE(!s.failed);
    /* Word 32 is the next unconsumed one, and it is lane 0 that takes it. */
    REQUIRE(s.cursor == 33);
    REQUIRE(s.bitsleft[0] == kGDeflateBitsPerPacket - 3 + kGDeflateBitsPerPacket);
    const uint64_t residue = (0xA5000000u | 0u) >> 3;
    const uint64_t refilled = static_cast<uint64_t>(0xA5000000u | 32u) << 29;
    REQUIRE_CTX(s.bitbuf[0] == (residue | refilled), "have %llx want %llx",
                static_cast<unsigned long long>(s.bitbuf[0]),
                static_cast<unsigned long long>(residue | refilled));
    return 0;
}

int TrapRefillOnAWordBoundary() {
    /* The watermark is a floor, not a strict one: a lane sitting at exactly 32
     * bits is satisfied and must NOT take a word. One bit lower and it must.
     * A `>` where the format has `>=` moves every subsequent word by one and
     * is invisible in any single round. */
    const std::vector<unsigned char> page = IndexedWords(64);

    GDeflateSchedule at;
    REQUIRE(GDeflateInit(at, page.data(), page.size()));
    REQUIRE(at.bitsleft[0] == kGDeflateLowWatermarkBits);
    const uint64_t before = at.cursor;
    GDeflateEnsure(at, page.data());
    REQUIRE(!at.failed);
    REQUIRE_CTX(at.cursor == before, "a lane at the watermark took a word");

    GDeflateSchedule below;
    REQUIRE(GDeflateInit(below, page.data(), page.size()));
    (void)GDeflatePop(below, 1);
    REQUIRE(below.bitsleft[0] == kGDeflateLowWatermarkBits - 1);
    GDeflateEnsure(below, page.data());
    REQUIRE(!below.failed);
    REQUIRE_CTX(below.cursor == before + 1,
                "a lane one bit below the watermark took no word");
    return 0;
}

int TrapTheLastWordOfTheStream() {
    /* The page ends. The refusal is on the refill that has no word left, the
     * cursor does not move past the end, and the failure is sticky - a caller
     * that keeps going gets no further consumption and no further bits. */
    const uint32_t words = 40;
    const std::vector<unsigned char> page = IndexedWords(words);
    GDeflateSchedule s;
    REQUIRE(GDeflateInit(s, page.data(), page.size()));

    /* Drain lane by lane until the page is out of words. Each lane needs four
     * 8-bit reads before it drops under the watermark, so this walks rounds
     * rather than jumping to the end. */
    uint64_t guard = 0;
    while (!s.failed && guard < 100000) {
        (void)GDeflatePop(s, 8);
        GDeflateAdvance(s, page.data());
        ++guard;
    }
    REQUIRE(s.failed);
    REQUIRE_CTX(s.cursor == words, "stopped at word %llu of %u",
                static_cast<unsigned long long>(s.cursor), words);

    const uint64_t frozen = s.cursor;
    GDeflateAdvance(s, page.data());
    GDeflateEnsure(s, page.data());
    REQUIRE(GDeflatePop(s, 8) == 0);
    REQUIRE(s.cursor == frozen);
    return 0;
}

int PeekAndRemoveRefuseSeparately() {
    /* A Huffman codeword is read as peek-then-remove rather than as one read
     * of a known width, because the width is what the peek decides (#176,
     * #182). So the remove is reached directly, with a count that came out of
     * a decode table, and its own guard is the one standing between a table
     * entry and an occupancy that wraps to four billion bits. */
    const std::vector<unsigned char> page = IndexedWords(64);
    GDeflateSchedule s;
    REQUIRE(GDeflateInit(s, page.data(), page.size()));

    REQUIRE(cudec_detail::GDeflatePeek(s, 0) == 0);
    REQUIRE(cudec_detail::GDeflatePeek(s, 8) == (0xA5000000u & 0xFFu));
    cudec_detail::GDeflateRemove(s, 8);
    REQUIRE(!s.failed);
    REQUIRE(s.bitsleft[0] == kGDeflateBitsPerPacket - 8);

    const uint32_t held = s.bitsleft[0];
    cudec_detail::GDeflateRemove(s, held + 1);
    REQUIRE_CTX(s.failed, "a remove wider than the lane was accepted");
    REQUIRE_CTX(s.bitsleft[0] == held, "the occupancy moved on a refusal");

    /* And a read wider than the watermark is refused as a caller error rather
     * than served from a lane that happens to be full. The peek has no flag to
     * set, so it answers with nothing; the pop refuses outright. */
    GDeflateSchedule t;
    REQUIRE(GDeflateInit(t, page.data(), page.size()));
    REQUIRE(cudec_detail::GDeflatePeek(t, cudec_detail::kGDeflateMaxPopBits) !=
            0);
    REQUIRE(cudec_detail::GDeflatePeek(
                t, cudec_detail::kGDeflateMaxPopBits + 1) == 0);
    REQUIRE(cudec_detail::GDeflatePeek(t, 64) == 0);
    REQUIRE(!t.failed);

    /* The width bound has to be tested on a lane that COULD have served the
     * read, or the occupancy check answers first and the bound is never
     * reached. A lane past a refill holds up to 63 bits, and a 33-bit read out
     * of it would come back through a 32-bit return value with its top bit
     * silently gone - which is why the refusal is on the width and not only on
     * the occupancy. */
    (void)GDeflatePop(t, 3);
    GDeflateEnsure(t, page.data());
    REQUIRE(!t.failed);
    REQUIRE(t.bitsleft[0] > cudec_detail::kGDeflateMaxPopBits);
    REQUIRE(GDeflatePop(t, cudec_detail::kGDeflateMaxPopBits + 1) == 0);
    REQUIRE_CTX(t.failed, "a read wider than the field bound was served");

    /* Once failed, nothing moves - the lane index included, which is the one
     * piece of state a caller could otherwise steer after a refusal. The lane
     * is moved off zero first, or the assertion would hold whether or not the
     * reset was guarded. */
    GDeflateSchedule f;
    REQUIRE(GDeflateInit(f, page.data(), page.size()));
    GDeflateAdvance(f, page.data());
    REQUIRE(f.idx == 1);
    (void)GDeflatePop(f, kGDeflateLowWatermarkBits + 1);
    REQUIRE(f.failed);
    GDeflateReset(f);
    REQUIRE_CTX(f.idx == 1, "the lane index moved after a refusal");
    return 0;
}

int InitRefusesBeforeItReads() {
    /* An empty candidate range is refused on its declared size, before a byte
     * of it is touched: nothing here is ever sized from the stream, so a page
     * that cannot carry the priming round is answered from the size alone. */
    GDeflateSchedule s;
    REQUIRE(!GDeflateInit(s, nullptr, 0));
    REQUIRE(s.failed);
    REQUIRE(s.cursor == 0);
    return 0;
}

/* ---- 3. The property test ---- */

/* Walk a randomised stream with randomised read widths and check the
 * invariants after every single operation rather than at the end: the lane
 * index cycles 0..31 in order, the cursor moves by at most one word per
 * advance and never backwards, a lane never holds more than the buffer's
 * bound, and no bit sits at or above the occupancy the lane reports. */
int PropertyOverRandomStreams() {
    for (uint32_t seed = 1; seed <= 64; ++seed) {
        Lcg rng(seed);
        const uint32_t words = kGDeflateNumStreams + (rng.Next() % 400);
        std::vector<unsigned char> page(static_cast<size_t>(words) * 4);
        for (size_t i = 0; i < page.size(); ++i) {
            page[i] = rng.NextByte();
        }

        GDeflateSchedule s;
        REQUIRE_CTX(GDeflateInit(s, page.data(), page.size()), "seed %u", seed);

        /* One mark per word, so "consumed twice" and "skipped" are checked
         * against the page rather than inferred from a counter. */
        std::vector<unsigned char> taken(words, 0);
        for (uint64_t w = 0; w < s.cursor; ++w) {
            taken[static_cast<size_t>(w)] = 1;
        }

        uint32_t expected_idx = 0;
        uint64_t steps = 0;
        while (!s.failed && steps < 200000) {
            const uint32_t width = 1 + (rng.Next() % kGDeflateLowWatermarkBits);
            if (width <= s.bitsleft[s.idx]) {
                (void)GDeflatePop(s, width);
                REQUIRE_CTX(!s.failed, "seed %u: a satisfiable read refused",
                            seed);
            }
            const uint64_t before = s.cursor;
            const uint32_t lane = s.idx;
            GDeflateAdvance(s, page.data());
            if (s.failed) {
                break;
            }

            REQUIRE_CTX(s.cursor >= before && s.cursor <= before + 1,
                        "seed %u: the cursor moved %llu words in one round",
                        seed,
                        static_cast<unsigned long long>(s.cursor - before));
            if (s.cursor == before + 1) {
                const size_t w = static_cast<size_t>(before);
                REQUIRE_CTX(!taken[w], "seed %u: word %zu taken twice", seed, w);
                taken[w] = 1;
                REQUIRE_CTX(s.bitsleft[lane] >= kGDeflateLowWatermarkBits,
                            "seed %u: lane %u refilled and is still short",
                            seed, lane);
            } else {
                REQUIRE_CTX(s.bitsleft[lane] >= kGDeflateLowWatermarkBits,
                            "seed %u: lane %u took no word while short", seed,
                            lane);
            }
            expected_idx = (expected_idx + 1) % kGDeflateNumStreams;
            REQUIRE_CTX(s.idx == expected_idx, "seed %u: lane order broke",
                        seed);

            for (uint32_t n = 0; n < kGDeflateNumStreams; ++n) {
                REQUIRE_CTX(s.bitsleft[n] <= cudec_detail::kGDeflateMaxLaneBits,
                            "seed %u: lane %u holds %u bits", seed, n,
                            s.bitsleft[n]);
                const uint32_t held = s.bitsleft[n];
                const uint64_t above =
                    held >= 64 ? 0 : (s.bitbuf[n] >> held);
                REQUIRE_CTX(above == 0,
                            "seed %u: lane %u carries bits above its occupancy",
                            seed, n);
            }
            ++steps;
        }

        /* The run ends when the page runs out, and the page runs out exactly
         * once: every word is marked and the cursor is at the end. */
        REQUIRE_CTX(s.failed, "seed %u: the walk never reached the end", seed);
        REQUIRE_CTX(s.cursor == words, "seed %u: stopped at %llu of %u", seed,
                    static_cast<unsigned long long>(s.cursor), words);
        for (uint32_t w = 0; w < words; ++w) {
            REQUIRE_CTX(taken[w], "seed %u: word %u was never consumed", seed,
                        w);
        }
    }
    return 0;
}

/* ---- 4. The oracle-anchored trace ---- */

/* Perturbations of the schedule state, injected between operations rather than
 * by reimplementing them. Each is a one-word or one-lane error of the kind the
 * traps above are about, and the trace below has to catch every one of them -
 * otherwise the trace is decoration. */
enum class Defect { kNone, kSkipOneWord, kOneLaneAhead, kRefillEarly };

/* Decode a stored-block page through the schedule alone and report the bytes.
 * Returns false if the schedule refused or the page was not what it claimed. */
bool WalkStoredPage(const unsigned char* page, size_t page_bytes,
                    Defect defect, std::vector<unsigned char>& out,
                    uint64_t& words_consumed) {
    GDeflateSchedule s;
    if (!GDeflateInit(s, page, page_bytes)) {
        return false;
    }
    if (defect == Defect::kSkipOneWord) {
        s.cursor += 1;
    } else if (defect == Defect::kOneLaneAhead) {
        s.idx = 1;
    }

    for (;;) {
        GDeflateReset(s);
        if (defect == Defect::kOneLaneAhead) {
            s.idx = 1;
        }
        const uint32_t is_final = GDeflatePop(s, 1);
        const uint32_t block_type = GDeflatePop(s, 2);
        GDeflateEnsure(s, page);
        if (s.failed || block_type != 0) {
            return false;
        }
        const uint32_t len = GDeflatePop(s, 16);
        if (s.failed) {
            return false;
        }
        if (defect == Defect::kRefillEarly) {
            s.bitsleft[s.idx] = kGDeflateLowWatermarkBits - 1;
            GDeflateEnsure(s, page);
        }
        for (uint32_t i = 0; i < len; ++i) {
            out.push_back(static_cast<unsigned char>(GDeflatePop(s, 8)));
            GDeflateAdvance(s, page);
            if (s.failed) {
                return false;
            }
        }
        /* The reference closes a block by walking all 32 lanes once, which is
         * where the deferred copies are run. There are none in a stored block,
         * but the round still happens and the words it takes are part of the
         * page. */
        for (uint32_t n = 0; n < kGDeflateNumStreams; ++n) {
            GDeflateAdvance(s, page);
            if (s.failed) {
                return false;
            }
        }
        if (is_final) {
            break;
        }
    }
    words_consumed = s.cursor;
    return true;
}

struct Compressed {
    std::vector<unsigned char> bytes;
    std::vector<size_t> page_sizes;
};

bool CompressPages(int level, const std::vector<unsigned char>& in,
                   Compressed& out) {
    libdeflate_gdeflate_compressor* c = libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        return false;
    }
    size_t npages = 0;
    const size_t bound = libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    if (npages == 0) {
        libdeflate_free_gdeflate_compressor(c);
        return false;
    }
    std::vector<unsigned char> buf(bound);
    std::vector<libdeflate_gdeflate_out_page> pages(npages);
    const size_t per = bound / npages;
    for (size_t i = 0; i < npages; ++i) {
        pages[i].data = buf.data() + i * per;
        pages[i].nbytes = per;
    }
    const size_t total = libdeflate_gdeflate_compress(c, in.data(), in.size(),
                                                      pages.data(), npages);
    libdeflate_free_gdeflate_compressor(c);
    if (total == 0) {
        return false;
    }
    out.bytes.clear();
    out.page_sizes.clear();
    for (size_t i = 0; i < npages; ++i) {
        const unsigned char* p = static_cast<const unsigned char*>(pages[i].data);
        out.bytes.insert(out.bytes.end(), p, p + pages[i].nbytes);
        out.page_sizes.push_back(pages[i].nbytes);
    }
    return true;
}

int TraceStoredPagesAgainstTheOracle() {
    /* Level 0 forces stored blocks out of the reference compressor, which is
     * the one block type the schedule can decode without a table. */
    const size_t sizes[] = {1, 37, 1024, 4096, 65536, 70000};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        Lcg rng(static_cast<uint32_t>(9000 + sizes[si]));
        std::vector<unsigned char> in(sizes[si]);
        for (size_t i = 0; i < in.size(); ++i) {
            in[i] = rng.NextByte();
        }

        Compressed c;
        REQUIRE_CTX(CompressPages(0, in, c), "size %zu did not compress",
                    sizes[si]);

        std::vector<unsigned char> decoded;
        size_t off = 0;
        for (size_t p = 0; p < c.page_sizes.size(); ++p) {
            const unsigned char* page = c.bytes.data() + off;
            const size_t bytes = c.page_sizes[p];
            off += bytes;

            uint64_t words = 0;
            std::vector<unsigned char> page_out;
            REQUIRE_CTX(WalkStoredPage(page, bytes, Defect::kNone, page_out,
                                       words),
                        "size %zu page %zu: the walk refused", sizes[si], p);
            /* Nothing skipped and nothing left: the walk ends on the page's
             * last word, which is the cursor claim this issue is about. */
            REQUIRE_CTX(words == bytes / 4,
                        "size %zu page %zu: consumed %llu of %zu words",
                        sizes[si], p, static_cast<unsigned long long>(words),
                        bytes / 4);
            decoded.insert(decoded.end(), page_out.begin(), page_out.end());
        }

        REQUIRE_CTX(decoded.size() == in.size(), "size %zu: produced %zu bytes",
                    sizes[si], decoded.size());
        REQUIRE_CTX(equal_bytes(decoded.data(), in.data(), in.size()),
                    "size %zu: byte divergence", sizes[si]);

        /* And the reference agrees the stream is what it looks like, so the
         * comparison above is against a page the oracle also accepts. */
        libdeflate_gdeflate_decompressor* d =
            libdeflate_alloc_gdeflate_decompressor();
        REQUIRE(d != nullptr);
        std::vector<libdeflate_gdeflate_in_page> in_pages(c.page_sizes.size());
        off = 0;
        for (size_t p = 0; p < c.page_sizes.size(); ++p) {
            in_pages[p].data = c.bytes.data() + off;
            in_pages[p].nbytes = c.page_sizes[p];
            off += c.page_sizes[p];
        }
        std::vector<unsigned char> ref(in.size());
        size_t got = 0;
        const libdeflate_result r = libdeflate_gdeflate_decompress(
            d, in_pages.data(), in_pages.size(), ref.data(), ref.size(), &got);
        libdeflate_free_gdeflate_decompressor(d);
        REQUIRE_CTX(r == LIBDEFLATE_SUCCESS, "size %zu: oracle refused",
                    sizes[si]);
        REQUIRE(got == in.size());
        REQUIRE_CTX(equal_bytes(decoded.data(), ref.data(), ref.size()),
                    "size %zu: divergence from the oracle", sizes[si]);
    }
    return 0;
}

int TheTraceBitesOnEveryPerturbation() {
    Lcg rng(4242);
    std::vector<unsigned char> in(8192);
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = rng.NextByte();
    }
    Compressed c;
    REQUIRE(CompressPages(0, in, c));
    REQUIRE(c.page_sizes.size() == 1);

    const Defect defects[] = {Defect::kSkipOneWord, Defect::kOneLaneAhead,
                              Defect::kRefillEarly};
    for (size_t i = 0; i < sizeof(defects) / sizeof(defects[0]); ++i) {
        uint64_t words = 0;
        std::vector<unsigned char> out;
        const bool walked = WalkStoredPage(c.bytes.data(), c.page_sizes[0],
                                           defects[i], out, words);
        const bool matches = walked && words == c.page_sizes[0] / 4 &&
                             out.size() == in.size() &&
                             std::memcmp(out.data(), in.data(), in.size()) == 0;
        REQUIRE_CTX(!matches, "perturbation %zu produced the right answer",
                    i);
    }
    return 0;
}

int TheHeaderRoundOnCompressedPages() {
    /* Compressed pages cannot be walked past their header here, but the header
     * itself rides lane 0 and is readable. BTYPE 3 is the reserved value the
     * format never emits, so a page whose header round reads 3 is a page this
     * schedule opened in the wrong place. */
    for (int level = 1; level <= 12; ++level) {
        Lcg rng(static_cast<uint32_t>(500 + level));
        std::vector<unsigned char> in(60000);
        for (size_t i = 0; i < in.size(); ++i) {
            /* Compressible, so the compressor reaches for a Huffman block. */
            in[i] = static_cast<unsigned char>(rng.NextByte() & 0x0Fu);
        }
        Compressed c;
        REQUIRE_CTX(CompressPages(level, in, c), "level %d", level);

        size_t off = 0;
        for (size_t p = 0; p < c.page_sizes.size(); ++p) {
            GDeflateSchedule s;
            REQUIRE_CTX(
                GDeflateInit(s, c.bytes.data() + off, c.page_sizes[p]),
                "level %d page %zu", level, p);
            off += c.page_sizes[p];
            GDeflateReset(s);
            (void)GDeflatePop(s, 1);
            const uint32_t block_type = GDeflatePop(s, 2);
            REQUIRE(!s.failed);
            REQUIRE_CTX(block_type != 3,
                        "level %d page %zu: header round read the reserved "
                        "block type",
                        level, p);
        }
    }
    return 0;
}

}  // namespace

int main() {
    if (PrimingRound() != 0) {
        return 1;
    }
    if (TrapFirstRoundAfterTheHeader() != 0) {
        return 1;
    }
    if (TrapRefillOnAWordBoundary() != 0) {
        return 1;
    }
    if (TrapTheLastWordOfTheStream() != 0) {
        return 1;
    }
    if (PeekAndRemoveRefuseSeparately() != 0) {
        return 1;
    }
    if (InitRefusesBeforeItReads() != 0) {
        return 1;
    }
    if (PropertyOverRandomStreams() != 0) {
        return 1;
    }
    if (TraceStoredPagesAgainstTheOracle() != 0) {
        return 1;
    }
    if (TheTraceBitesOnEveryPerturbation() != 0) {
        return 1;
    }
    if (TheHeaderRoundOnCompressedPages() != 0) {
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
