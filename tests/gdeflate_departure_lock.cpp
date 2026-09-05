/* The lock over the declared strictness departures (issue #183).
 *
 * `GDeflateRejectIsDeclaredDeparture` in src/gdeflate_schedule.h names the
 * rungs on which this decoder refuses a page libdeflate's GDeflate
 * decompressor decodes. The differential fuzz target reads that predicate as
 * an exemption: the reverse direction - the twin refusing what the reference
 * accepts - traps unless the rung is declared there. A predicate consulted
 * that way is an allowlist, and an allowlist nothing checks is a place where a
 * real over-strictness can be made to disappear by adding one line.
 *
 * THIS FILE IS WHAT THAT LINE COSTS. For every rung the predicate declares,
 * there is a page here that the reference decodes and this decoder refuses on
 * exactly that rung, so the declaration is executed rather than asserted. A
 * rung added to the predicate with no such page fails the sweep at the bottom;
 * a rung that stops being a departure - because the two decoders agreed again,
 * or because the refusal moved - fails its own case rather than sitting in the
 * exemption forever.
 *
 * THREE OF THE FOUR FIXTURES ARE THE FUZZER'S AND NOT MINE, which is the point
 * rather than a convenience. Each page in gdeflate_departure_pages.h is the
 * shortest input a run of fuzz_gdeflate_page reached its rung with, carried
 * over as bytes; the same inputs are committed as seeds under
 * fuzz/corpus/fuzz_gdeflate_page, so the CI replay leg drives the exemption
 * path on every run. A hand-written page would prove the departure I imagined;
 * these prove the one the reference and this decoder actually disagree about.
 *
 * THE FOURTH COULD NOT COME FROM THERE, AND THE REASON IS THE FINDING RATHER
 * THAN AN EXCUSE (issue #453). kGDeflateRejectPagePartialWord refuses a page
 * whose length is not a whole number of 32-bit words, and fuzz_gdeflate_page
 * masks every length it hands either decoder down to a multiple of four, so
 * that rung is unreachable in the target whose exemption this file exists to
 * price. Its population is the GPU gate's single-byte truncations of real
 * pages, and the fixture below is built the same way here: the reference's own
 * compressor makes a page, one byte comes off the end, and both decoders are
 * asked about what is left. It is generated rather than pinned because a
 * pinned copy would fix the compressor's output of one day, and what the
 * departure is about is any page of that shape rather than one page.
 *
 * IT ASSERTS THE ROUND TRIP AND NOT ONLY THE ACCEPTANCE, which the three
 * pinned pages cannot: the reference reporting SUCCESS means its bits ran out
 * cleanly and not that its answer is right (tests/oracle_gdeflate.cpp). For a
 * truncation of a real page the original bytes are known, so this one holds
 * the reference to reproducing them exactly - which is the same rule the GPU
 * gate applies before it calls a refusal over-strict, and it is what separates
 * a departure from a page neither decoder reproduced.
 *
 * THE REFERENCE GETS A ZERO TAIL AFTER ITS COPY AND THE TWIN DOES NOT, for the
 * reason fuzz/fuzz_gdeflate_page.cpp argues at its head: `ENSURE_BITS` in the
 * fork reads a 32-bit word with no bound against the end of the page, so in a
 * tight allocation its verdict would depend on whatever the allocator left
 * behind. Both sides are told the same `nbytes` and are asked about the same
 * bytes; only what lies past the end differs, and on the reference's side it
 * is zeros rather than luck. That convention is load-bearing here in a way it
 * is not in the twins next door: one of the three departures IS the read past
 * the end, so a comparison made without it would be a comparison against an
 * uninitialised buffer. */
#include "gdeflate_block.h"
#include "gdeflate_departure_pages.h"
#include "require.h"

#include <libdeflate.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using cudec_detail::GDeflateDecodePage;
using cudec_detail::GDeflatePageState;
using cudec_detail::GDeflateReject;
using cudec_detail::GDeflateRejectIsDeclaredDeparture;
using cudec_detail::kGDeflateRejectCount;
using cudec_detail::kGDeflateRejectEmptyTableUsed;
using cudec_detail::kGDeflateRejectNone;
using cudec_detail::kGDeflateRejectPagePartialWord;
using cudec_detail::kGDeflateRejectRefillPastEnd;
using cudec_detail::kGDeflateRejectRepeatRunPastAlphabet;

bool g_shown[kGDeflateRejectCount] = {false};

/* The tail the reference's copy carries, sized from the capacity by the
 * argument the fuzz target makes: on a compressed block the reference has no
 * bound of its own, so it keeps taking a word per round until the OUTPUT
 * fills, and the rounds are therefore bounded by the capacity rather than by
 * what the page had left. Six words per capacity byte plus a page of slack is
 * the same margin that target uses. */
size_t OracleTailBytes(size_t capacity) {
    return 6u * capacity + 4096u;
}

/* One declared departure, executed. Both halves are asserted rather than one:
 * that the reference decodes the page, and that this decoder refuses it on the
 * rung the predicate names. Asserting only the refusal would pass for a page
 * both sides reject, which is not a departure at all.
 *
 * `expect` is the third half where it can be had. A page found by the fuzzer
 * has no known-good output - it is a mutation of nothing in particular - so
 * those three pass a null pointer and the reference's SUCCESS is all there is.
 * A page built by truncating a real one does have one, and there the
 * reference is held to reproducing it rather than merely to finishing. */
int Departure(const char* name, const unsigned char* page, size_t page_size,
              size_t capacity, GDeflateReject expected,
              const std::vector<unsigned char>* expect) {
    REQUIRE_CTX(GDeflateRejectIsDeclaredDeparture(expected),
                "%s: rung %d is not declared a departure, so this fixture "
                "belongs in the ladder lock rather than here",
                name, static_cast<int>(expected));

    const size_t tail = OracleTailBytes(capacity);
    std::vector<unsigned char> padded(page_size + tail, 0);
    std::memcpy(padded.data(), page, page_size);

    libdeflate_gdeflate_in_page in_page;
    in_page.data = padded.data();
    in_page.nbytes = page_size;

    std::vector<unsigned char> oracle_out(capacity, 0);
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    REQUIRE(d != nullptr);
    size_t oracle_size = 0;
    const libdeflate_result oracle_status =
        libdeflate_gdeflate_decompress(d, &in_page, 1, oracle_out.data(),
                                       capacity, &oracle_size);
    libdeflate_free_gdeflate_decompressor(d);
    REQUIRE_CTX(oracle_status == LIBDEFLATE_SUCCESS,
                "%s: the reference refused this page (status=%d), so it is not "
                "a page the two decoders disagree about",
                name, static_cast<int>(oracle_status));
    if (expect != nullptr) {
        REQUIRE_CTX(oracle_size == expect->size(),
                    "%s: the reference produced %zu bytes and the unmutated "
                    "page gives %zu, so it did not reproduce the original",
                    name, oracle_size, expect->size());
        REQUIRE_CTX(std::memcmp(oracle_out.data(), expect->data(),
                                expect->size()) == 0,
                    "%s: the reference produced other bytes than the "
                    "unmutated page gives, so this is not a departure",
                    name);
    }

    /* The twin's copy is exactly its own size, so a read past the end lands in
     * an AddressSanitizer redzone rather than in slack - which is what makes
     * the refill departure below a measurement rather than a guess. */
    std::vector<unsigned char> tight(page, page + page_size);
    std::vector<unsigned char> twin_out(capacity, 0);
    GDeflatePageState st;
    uint64_t produced = 0;
    const bool twin_ok =
        GDeflateDecodePage(st, tight.data(), page_size, twin_out.data(),
                           capacity, &produced);
    REQUIRE_CTX(!twin_ok,
                "%s: this decoder accepted the page too, so rung %d is no "
                "longer a departure and belongs out of the predicate",
                name, static_cast<int>(expected));
    REQUIRE_CTX(st.s.failed, "%s: refused without failing the schedule", name);
    REQUIRE_CTX(st.s.reject == expected,
                "%s: refused on rung %d, wanted rung %d", name,
                static_cast<int>(st.s.reject), static_cast<int>(expected));

    std::printf("  %-28s reference produced %zu byte(s), rung %d\n", name,
                oracle_size, static_cast<int>(expected));
    g_shown[expected] = true;
    return 0;
}

/* The reference's own compressor, one page. Real input rather than a literal,
 * because the departure below is about what happens to a page a compressor
 * actually emitted. */
bool CompressOnePage(int level, const std::vector<unsigned char>& in,
                     std::vector<unsigned char>* page) {
    libdeflate_gdeflate_compressor* c =
        libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        return false;
    }
    size_t npages = 0;
    const size_t bound =
        libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    if (bound == 0 || npages != 1) {
        libdeflate_free_gdeflate_compressor(c);
        return false;
    }
    std::vector<unsigned char> pool(bound, 0);
    libdeflate_gdeflate_out_page out;
    out.data = pool.data();
    out.nbytes = bound;
    const size_t total =
        libdeflate_gdeflate_compress(c, in.data(), in.size(), &out, 1);
    libdeflate_free_gdeflate_compressor(c);
    if (total == 0) {
        return false;
    }
    const unsigned char* p = static_cast<const unsigned char*>(out.data);
    page->assign(p, p + out.nbytes);
    return true;
}

/* Compressible without being trivial: a long stretch of one byte compresses to
 * a page too short to have a byte worth losing, and incompressible input turns
 * into stored blocks whose length fields the truncation would corrupt into a
 * refusal on the reference's side too. */
std::vector<unsigned char> DepartureSource(size_t n) {
    std::vector<unsigned char> v;
    v.reserve(n);
    uint32_t x = 0x1234567u;
    while (v.size() < n) {
        x = x * 1664525u + 1013904223u;
        const size_t run = 3u + (x >> 24) % 40u;
        const unsigned char b =
            static_cast<unsigned char>('a' + ((x >> 8) % 7u));
        for (size_t i = 0; i < run && v.size() < n; i++) {
            v.push_back(b);
        }
        x = x * 1664525u + 1013904223u;
        const size_t noise = (x >> 16) % 6u;
        for (size_t i = 0; i < noise && v.size() < n; i++) {
            v.push_back(static_cast<unsigned char>((x >> i) & 0xFFu));
        }
    }
    return v;
}

/* The partial-word departure, at every level the GPU gate's parity uses, so
 * this does not rest on one compressor setting happening to leave a droppable
 * byte. A level whose page truncates into something the reference will not
 * reproduce is not a failure of the departure and is skipped with a line
 * saying so; the sweep at the bottom is what requires at least one to have
 * held. */
int PartialWordDepartures() {
    const int levels[] = {0, 1, 6, 12};
    int shown = 0;
    for (size_t l = 0; l < sizeof levels / sizeof levels[0]; l++) {
        const std::vector<unsigned char> source = DepartureSource(20000);
        std::vector<unsigned char> page;
        REQUIRE_CTX(CompressOnePage(levels[l], source, &page),
                    "level %d did not compress to a single page",
                    levels[l]);
        REQUIRE_CTX(page.size() % 4u == 0u,
                    "level %d produced a %zu-byte page, which is not a whole "
                    "number of words before anything was removed",
                    levels[l], page.size());
        REQUIRE(page.size() > 128u);

        /* One byte off the end. The reference reads words it needs and stops
         * at a full output, so where the last word was never reached this is
         * still the same page to it and is no longer a page at all here. */
        page.resize(page.size() - 1u);

        const std::string name =
            "partial word, level " + std::to_string(levels[l]);
        const size_t capacity = source.size() + 64u;

        /* Whether the reference still reproduces the source is what decides
         * if this level's page is a departure, and it is asked here rather
         * than asserted, because it is a property of that page and not of the
         * rung. */
        const size_t tail = OracleTailBytes(capacity);
        std::vector<unsigned char> padded(page.size() + tail, 0);
        std::memcpy(padded.data(), page.data(), page.size());
        libdeflate_gdeflate_in_page probe;
        probe.data = padded.data();
        probe.nbytes = page.size();
        std::vector<unsigned char> probe_out(capacity, 0);
        libdeflate_gdeflate_decompressor* d =
            libdeflate_alloc_gdeflate_decompressor();
        REQUIRE(d != nullptr);
        size_t probe_size = 0;
        const libdeflate_result probe_status = libdeflate_gdeflate_decompress(
            d, &probe, 1, probe_out.data(), capacity, &probe_size);
        libdeflate_free_gdeflate_decompressor(d);
        if (probe_status != LIBDEFLATE_SUCCESS || probe_size != source.size() ||
            std::memcmp(probe_out.data(), source.data(), source.size()) != 0) {
            std::printf("  %-28s the reference does not reproduce this "
                        "truncation, so this level shows nothing\n",
                        name.c_str());
            continue;
        }

        if (Departure(name.c_str(), page.data(), page.size(), capacity,
                      kGDeflateRejectPagePartialWord,
                      &source) != 0) {
            return 1;
        }
        shown++;
    }
    REQUIRE_CTX(shown > 0,
                "no compression level produced a page whose last byte the "
                "reference did not need, so this run showed the partial-word "
                "departure nowhere (issue #453)");
    return 0;
}

/* Every rung the predicate declares has been shown to be one, and nothing
 * else has. The second half is what stops this file from growing a case for a
 * rung the predicate does not exempt: such a case would be a page the
 * reference decodes and the fuzz target traps on, which is a finding rather
 * than a fixture. */
int Sweep() {
    int declared = 0;
    for (int i = 0; i < static_cast<int>(kGDeflateRejectCount); i++) {
        const GDeflateReject rung = static_cast<GDeflateReject>(i);
        if (!GDeflateRejectIsDeclaredDeparture(rung)) {
            REQUIRE_CTX(!g_shown[i],
                        "rung %d was shown to be a departure and the predicate "
                        "does not declare it",
                        i);
            continue;
        }
        declared++;
        REQUIRE_CTX(g_shown[i],
                    "rung %d is declared a strictness departure and no page "
                    "here shows the reference decoding it (issue #183)",
                    i);
    }
    REQUIRE(!GDeflateRejectIsDeclaredDeparture(kGDeflateRejectNone));
    std::printf("gdeflate_departure_lock: %d declared departure(s), each "
                "shown against the reference\n",
                declared);
    return 0;
}

}  // namespace

int main() {
    if (Departure("refill past the end", kPageRefillPastEnd,
                  sizeof kPageRefillPastEnd, kCapacityRefillPastEnd,
                  kGDeflateRejectRefillPastEnd, nullptr) != 0) {
        return 1;
    }
    if (Departure("empty code used", kPageEmptyTableUsed,
                  sizeof kPageEmptyTableUsed, kCapacityEmptyTableUsed,
                  kGDeflateRejectEmptyTableUsed, nullptr) != 0) {
        return 1;
    }
    if (Departure("repeat run past the alphabet", kPageRepeatRunPastAlphabet,
                  sizeof kPageRepeatRunPastAlphabet,
                  kCapacityRepeatRunPastAlphabet,
                  kGDeflateRejectRepeatRunPastAlphabet, nullptr) != 0) {
        return 1;
    }
    if (PartialWordDepartures() != 0) {
        return 1;
    }
    if (Sweep() != 0) {
        return 1;
    }
    return 0;
}
