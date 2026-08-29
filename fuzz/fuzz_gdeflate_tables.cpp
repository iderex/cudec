/* Fuzz target over the GDeflate table pipeline (issue #186): the canonical
 * construction in src/gdeflate_tables.h and the dynamic block's code-length
 * rounds that feed it.
 *
 * WHY THERE IS NO DIFFERENTIAL HALF, STATED FIRST BECAUSE IT CHANGES WHAT THE
 * VERDICTS MEAN. The reference's decode-table builder is `static` inside a
 * translation unit the pinned fork compiles with HIDE_INTERFACE, and its
 * code-length vectors never cross a public name, so there is nothing here to
 * diff against - which is exactly why this surface gets an entry point of its
 * own rather than being folded into the page-decode target's byte parity. What
 * is asserted instead is an INVARIANT computed beside the builder from the
 * same lengths: a code-length vector describes a code or it does not, that is
 * decidable by counting codespace, and the builder's verdict must equal that
 * count on every input.
 *
 * That is weaker than parity in one direction and stronger in another. Weaker,
 * because a builder that refused everything would satisfy the reject half; the
 * seed corpus carries vectors that must be ACCEPTED and the kind each one must
 * come back as, so that arm is pinned rather than merely exercised. Stronger,
 * because the invariant is checked against the accepted RESULT - the kind, the
 * symbols the table can produce, the vector the rounds recovered - rather than
 * against another implementation's opinion.
 *
 * THIS IS THE CLASSIC DEFLATE MEMORY-SAFETY SURFACE. The lengths are
 * attacker-chosen and they become an index, so the tables are heap-allocated
 * at exactly their own size: an over-write lands in an AddressSanitizer
 * redzone rather than in allocator slack, which is the tests/parser_twin.cpp
 * trick carried over. GDeflate has no checksum anywhere
 * (docs/MASTERPLAN.md section 11.4), so an over-permissive table is a live
 * decode of symbols the stream never encoded rather than a caught error.
 *
 * WHAT THIS TARGET CANNOT REACH, so a clean run is not read as covering it.
 * Arm B enters the code-length rounds on bytes the fuzzer chose, so the pages
 * it builds are almost never pages a compressor would emit; whether the rounds
 * agree with the reference on a REAL page is byte parity and belongs to the
 * page-decode target (#192) and to tests/gdeflate_header_twin.cpp. And nothing
 * here decodes a block body, so the length and distance tables are exercised
 * as tables and never through a decoded match. */
#include "gdeflate_tables.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

using cudec_detail::GDeflateBuildTable;
using cudec_detail::GDeflateCodeLengths;
using cudec_detail::GDeflateDecodeSymbol;
using cudec_detail::GDeflateDistTable;
using cudec_detail::GDeflateInit;
using cudec_detail::GDeflateLitLenTable;
using cudec_detail::GDeflatePop;
using cudec_detail::GDeflateReadCodeLengths;
using cudec_detail::GDeflateReset;
using cudec_detail::GDeflateSchedule;
using cudec_detail::kGDeflateMaxCodeLen;
using cudec_detail::kGDeflateNoSymbol;
using cudec_detail::kGDeflateNumDistSyms;
using cudec_detail::kGDeflateNumLitLenSyms;
using cudec_detail::kGDeflateNumStreams;
using cudec_detail::kGDeflateTableComplete;
using cudec_detail::kGDeflateTableEmpty;
using cudec_detail::kGDeflateTableSingle;

/* Bounded so libFuzzer explores the rounds rather than the allocator. A page
 * this size still carries far more rounds than a code-length vector needs. */
constexpr size_t kMaxPageBytes = 1u << 13;

/* The shortest page the schedule admits: the priming round is 32 words
 * (src/gdeflate_schedule.h), so anything below this is refused before a bit is
 * read and arm B would test nothing. */
constexpr size_t kMinPageBytes = kGDeflateNumStreams * 4u;

/* How many symbols an accepted table is asked for. Enough that a table whose
 * root accelerator and length walk disagreed would be caught, small enough
 * that the engine is not spending its budget here. */
constexpr uint32_t kSymbolProbes = 256;

void Trap(const char* what, size_t size) {
    std::fprintf(stderr, "INVARIANT DIVERGENCE: %s; input=%zu\n", what, size);
    __builtin_trap();
}

/* What the lengths describe, counted rather than asked of the thing under
 * test. The accumulator is the reference's own: the running total shifted left
 * once per length, which reaches num_syms << (kMaxLen - 1) at worst and cannot
 * wrap in 32 bits for any alphabet this file sizes for. */
struct Verdict {
    bool ok;
    uint32_t kind;
};

Verdict Expected(const unsigned char* lens, uint32_t num_syms, uint32_t cap) {
    Verdict v;
    v.ok = false;
    v.kind = kGDeflateTableComplete;
    if (num_syms > cap) {
        return v;
    }
    uint32_t count[kGDeflateMaxCodeLen + 1];
    for (uint32_t i = 0; i <= kGDeflateMaxCodeLen; i++) {
        count[i] = 0;
    }
    for (uint32_t sym = 0; sym < num_syms; sym++) {
        if (lens[sym] > kGDeflateMaxCodeLen) {
            return v;
        }
        count[lens[sym]]++;
    }
    uint32_t codespace = 0;
    for (uint32_t len = 1; len <= kGDeflateMaxCodeLen; len++) {
        codespace = (codespace << 1) + count[len];
    }
    const uint32_t full = 1u << kGDeflateMaxCodeLen;
    if (codespace > full) {
        return v;
    }
    if (codespace == full) {
        v.ok = true;
        v.kind = kGDeflateTableComplete;
        return v;
    }
    if (codespace == 0) {
        v.ok = true;
        v.kind = kGDeflateTableEmpty;
        return v;
    }
    /* The reference admits exactly one incomplete shape besides the empty
     * code: a single symbol holding the whole codespace at length 1. */
    if (codespace == (full >> 1) && count[1] == 1) {
        v.ok = true;
        v.kind = kGDeflateTableSingle;
        return v;
    }
    return v;
}

/* Ask an accepted table for symbols until the schedule runs out. Every answer
 * has to be the sentinel or a symbol the vector actually coded - that is the
 * property a table built from attacker-chosen lengths owes, and it is what
 * separates "the lengths describe a code" from "the lengths describe
 * something, so let us index with it". */
template <typename Table>
void ProbeSymbols(const Table& table, const unsigned char* lens,
                  uint32_t num_syms, const unsigned char* page,
                  size_t page_bytes, size_t input_size, const char* what) {
    GDeflateSchedule s;
    if (!GDeflateInit(s, page, page_bytes)) {
        return;
    }
    for (uint32_t n = 0; n < kSymbolProbes; n++) {
        const uint32_t sym = GDeflateDecodeSymbol(s, table);
        if (sym == kGDeflateNoSymbol) {
            if (!s.failed) {
                Trap(what, input_size);
            }
            return;
        }
        if (sym >= num_syms || lens[sym] == 0) {
            Trap(what, input_size);
        }
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    /* One selector byte, then the payload. The selector decides how wide the
     * alphabet is and whether the lengths arrive masked into the format's
     * range or raw - raw is where a length past the maximum is reachable, and
     * masked is where the interesting codespace arithmetic lives, because an
     * unmasked random vector is refused on its first byte almost always. */
    if (size < 2) {
        return 0;
    }
    const uint8_t selector = data[0];
    const uint8_t* payload = data + 1;
    const size_t payload_size = size - 1;

    /* ---- Arm A: the canonical construction, entered directly on a length
     * vector the fuzzer chose. The table is heap-allocated at exactly its own
     * size so an over-write lands in a redzone. */
    {
        const bool masked = (selector & 1u) != 0u;
        uint32_t num_syms = 1u + static_cast<uint32_t>(
                                     payload_size % kGDeflateNumLitLenSyms);
        if (num_syms > payload_size) {
            num_syms = static_cast<uint32_t>(payload_size);
        }
        if (num_syms != 0) {
            std::unique_ptr<unsigned char[]> lens(
                new unsigned char[num_syms]);
            for (uint32_t i = 0; i < num_syms; i++) {
                lens[i] = masked ? static_cast<unsigned char>(payload[i] & 0x0Fu)
                                 : payload[i];
            }
            std::unique_ptr<GDeflateLitLenTable> table(
                new GDeflateLitLenTable());
            Verdict want = Expected(lens.get(), num_syms,
                                    kGDeflateNumLitLenSyms);
            const bool got = GDeflateBuildTable(lens.get(), num_syms, *table);
#ifdef CUDEC_FUZZ_SELFTEST_BREAK
            /* Off by default, and the only way to show the comparison below is
             * live without waiting for a real divergence: a second binary
             * built with this defined inverts the counted verdict on every
             * vector the builder ACCEPTED, so a harness that had silently
             * stopped comparing passes where this one traps. Never define it
             * in a build whose findings are being believed. */
            if (got) {
                want.ok = false;
            }
#endif
            if (got != want.ok) {
                Trap("the builder and the codespace count disagree", size);
            }
            if (got) {
                if (table->kind != want.kind) {
                    Trap("the builder and the codespace count disagree on the "
                         "kind",
                         size);
                }
                /* An accepted table may only ever answer with a symbol the
                 * vector coded. The page it is driven over is the payload
                 * itself, which is bits nobody chose to be codewords. */
                if (payload_size >= kMinPageBytes) {
                    size_t page_bytes = payload_size > kMaxPageBytes
                                            ? kMaxPageBytes
                                            : payload_size;
                    page_bytes -= page_bytes % 4u;
                    ProbeSymbols(*table, lens.get(), num_syms, payload,
                                 page_bytes, size,
                                 "an accepted table produced a symbol its "
                                 "vector never coded");
                }
            }
        }
    }

    /* ---- Arm B: the code-length rounds, entered on the payload read as a
     * page. BFINAL and BTYPE are consumed without being judged - which block
     * type these bits claim to be is the page decode's business, and the
     * rounds under test begin where the reference's do. */
    if (payload_size < kMinPageBytes) {
        return 0;
    }
    size_t page_bytes =
        payload_size > kMaxPageBytes ? kMaxPageBytes : payload_size;
    page_bytes -= page_bytes % 4u;

    GDeflateSchedule s;
    if (!GDeflateInit(s, payload, page_bytes)) {
        return 0;
    }
    GDeflateReset(s);
    GDeflatePop(s, 1);
    GDeflatePop(s, 2);
    if (s.failed) {
        return 0;
    }

    std::unique_ptr<GDeflateCodeLengths> lens(new GDeflateCodeLengths());
    if (!GDeflateReadCodeLengths(s, payload, *lens)) {
        if (!s.failed) {
            Trap("the code-length rounds refused without failing the "
                 "schedule",
                 size);
        }
        return 0;
    }
    if (s.failed) {
        Trap("the code-length rounds accepted with the schedule failed", size);
    }
    /* HLIT and HDIST are five-bit fields, so an accepted read can only ever
     * name a count inside these ranges. A count outside them is the shape that
     * would index a table past its own alphabet. */
    if (lens->num_litlen < 257u || lens->num_litlen > kGDeflateNumLitLenSyms) {
        Trap("an accepted read named a literal/length count outside the "
             "field's range",
             size);
    }
    if (lens->num_dist < 1u || lens->num_dist > kGDeflateNumDistSyms) {
        Trap("an accepted read named a distance count outside the field's "
             "range",
             size);
    }
    const uint32_t total = lens->num_litlen + lens->num_dist;
    for (uint32_t i = 0; i < total; i++) {
        if (lens->lens[i] > kGDeflateMaxCodeLen) {
            Trap("an accepted read produced a length past the longest "
                 "codeword",
                 size);
        }
    }

    /* The two tables the rounds exist to build, each judged against the same
     * count as arm A and each asked for symbols afterwards. */
    std::unique_ptr<GDeflateLitLenTable> litlen(new GDeflateLitLenTable());
    const Verdict want_litlen =
        Expected(lens->lens, lens->num_litlen, kGDeflateNumLitLenSyms);
    if (GDeflateBuildTable(lens->lens, lens->num_litlen, *litlen) !=
        want_litlen.ok) {
        Trap("the literal/length table and the codespace count disagree",
             size);
    }
    if (want_litlen.ok) {
        ProbeSymbols(*litlen, lens->lens, lens->num_litlen, payload,
                     page_bytes, size,
                     "an accepted literal/length table produced a symbol its "
                     "vector never coded");
    }

    std::unique_ptr<GDeflateDistTable> dist(new GDeflateDistTable());
    const unsigned char* dist_lens = lens->lens + lens->num_litlen;
    const Verdict want_dist =
        Expected(dist_lens, lens->num_dist, kGDeflateNumDistSyms);
    if (GDeflateBuildTable(dist_lens, lens->num_dist, *dist) != want_dist.ok) {
        Trap("the distance table and the codespace count disagree", size);
    }
    if (want_dist.ok) {
        ProbeSymbols(*dist, dist_lens, lens->num_dist, payload, page_bytes,
                     size,
                     "an accepted distance table produced a symbol its vector "
                     "never coded");
    }
    return 0;
}
