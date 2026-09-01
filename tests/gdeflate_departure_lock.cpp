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
 * THE FIXTURES ARE THE FUZZER'S AND NOT MINE, which is the point rather than a
 * convenience. Each page below is the shortest input a run of
 * fuzz_gdeflate_page reached its rung with, carried over as bytes; the same
 * inputs are committed as seeds under fuzz/corpus/fuzz_gdeflate_page, so the
 * CI replay leg drives the exemption path on every run. A hand-written page
 * would prove the departure I imagined; these prove the one the reference and
 * this decoder actually disagree about.
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

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::GDeflateDecodePage;
using cudec_detail::GDeflatePageState;
using cudec_detail::GDeflateReject;
using cudec_detail::GDeflateRejectIsDeclaredDeparture;
using cudec_detail::kGDeflateRejectCount;
using cudec_detail::kGDeflateRejectEmptyTableUsed;
using cudec_detail::kGDeflateRejectNone;
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
 * both sides reject, which is not a departure at all. */
int Departure(const char* name, const unsigned char* page, size_t page_size,
              size_t capacity, GDeflateReject expected) {
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
                  kGDeflateRejectRefillPastEnd) != 0) {
        return 1;
    }
    if (Departure("empty code used", kPageEmptyTableUsed,
                  sizeof kPageEmptyTableUsed, kCapacityEmptyTableUsed,
                  kGDeflateRejectEmptyTableUsed) != 0) {
        return 1;
    }
    if (Departure("repeat run past the alphabet", kPageRepeatRunPastAlphabet,
                  sizeof kPageRepeatRunPastAlphabet,
                  kCapacityRepeatRunPastAlphabet,
                  kGDeflateRejectRepeatRunPastAlphabet) != 0) {
        return 1;
    }
    if (Sweep() != 0) {
        return 1;
    }
    return 0;
}
