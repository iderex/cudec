/* The Zstd mutation corpus proving itself (issue #187), the counterpart of
 * the mutant halves of oracle_lz4.cpp and snappy_corpus.cpp.
 *
 * What this establishes, and none of it is about cudec - there is no Zstd
 * decoder in this tree yet, and this file would be wrong to wait for one:
 *
 *  - the mutants are generated deterministically, two passes byte for byte,
 *    with a pinned digest that catches a drift both passes share;
 *  - the mutations BITE: every fixture yields at least one mutant the pinned
 *    reference refuses, so a generator that quietly stopped changing
 *    anything reds here rather than reporting a clean negative net;
 *  - the aimed mutations are present by name. The blind truncations and bit
 *    flips would keep the count up on their own, so the named header fields
 *    are asserted individually - a walker that stopped reporting an offset
 *    would otherwise drop a whole branch class in silence;
 *  - the accepted mutants are kept and counted rather than filtered out.
 *    cudec being STRICTER than the reference is a divergence in exactly the
 *    way cudec being looser is, and the corpus that will catch it has to
 *    carry the streams the reference accepts.
 *
 * The verdict is always the reference's. ZstdOracleDecodes is the only thing
 * in this file that decides whether a stream is valid.
 *
 * WHY THE GENERATOR IS NOT IN tests/fixtures.cpp, which is where the issue
 * put it. That translation unit lives in the cudec_test_fixtures archive,
 * which links the decode-only zstd oracle; a Zstd mutation corpus needs the
 * compressor, and tests/CMakeLists.txt states that the decode-only and full
 * archives must never meet in one link because every decode symbol is in
 * both. So the layer sits beside MakeZstdFixtures in tests/zstd_corpus.cpp,
 * which already links the full archive and already owns the frame walker the
 * aimed mutations are positioned from. */
#include "require.h"
#include "zstd_corpus.h"

#include <cstdio>
#include <set>
#include <string>

namespace {

using Bytes = std::vector<unsigned char>;

/* Mutant-corpus drift tripwire, the instrument oracle_lz4.cpp,
 * snappy_corpus.cpp and zstd_corpus_selfproof.cpp all carry: a zstd bump or
 * a generator change moves these bytes, and it should have to update this
 * number on purpose. The verdicts are folded in too, so a mutant that starts
 * being accepted moves the digest even when its bytes do not. FNV-1a, not
 * crypto: a tripwire against drift, not a defence. */
constexpr uint64_t kExpectedZstdMutantDigest = 0xd92c1bc7ecf524e6ull;

uint64_t Fnv1a64(uint64_t hash, const void* data, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

/* The aimed mutations that must exist somewhere in the corpus, by the prefix
 * their description carries. Each names a field the reference decoder
 * branches on; a walker or generator change that stops producing one lands
 * here as a missing name rather than as a slightly smaller corpus. */
const char* const kRequiredAims[] = {
    "magic-byte-0",
    "magic-byte-3",
    "descriptor-reserved-bit",
    "descriptor-checksum-flag",
    "descriptor-content-size-width",
    "descriptor-dictionary-id-width",
    "descriptor-single-segment",
    "byte-after-descriptor",
    "block0-symbol-modes-reserved-bits",
    "block0-symbol-modes-litlen",
    "block0-symbol-modes-offset",
    "block0-first-table-accuracy-log",
    "block0-first-table-body",
    "block0-huffman-description-header",
    "block0-huffman-description-body",
    "block0-payload-last-byte"};

bool StartsWith(const std::string& text, const char* prefix) {
    const std::string want(prefix);
    return text.size() >= want.size() &&
           text.compare(0, want.size(), want) == 0;
}

}  // namespace

int main() {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());

    /* The corpus is only a negative net if the unmutated frames are on the
     * other side of the line. A generator whose fixtures the reference
     * already refused would make every "rejected" below meaningless. */
    for (const ZstdFixture& fixture : fixtures) {
        Bytes decoded;
        REQUIRE_CTX(ZstdOracleDecodes(fixture.compressed, &decoded),
                    "%s: the reference refused an unmutated fixture",
                    fixture.name.c_str());
    }

    uint64_t digest = 14695981039346656037ull;
    size_t mutants = 0;
    size_t rejected = 0;
    size_t accepted = 0;
    std::set<std::string> aims_seen;

    for (size_t index = 0; index < fixtures.size(); index++) {
        const ZstdFixture& fixture = fixtures[index];
        const uint64_t seed = 0x5a5a0000ull + index;
        const std::vector<ZstdMutant> first =
            MutateZstdFrame(fixture.compressed, seed);
        const std::vector<ZstdMutant> second =
            MutateZstdFrame(fixture.compressed, seed);
        REQUIRE_CTX(!first.empty(), "%s: no mutants at all",
                    fixture.name.c_str());
        REQUIRE_CTX(first.size() == second.size(),
                    "%s: two passes produced %zu and %zu mutants",
                    fixture.name.c_str(), first.size(), second.size());

        size_t rejected_here = 0;
        for (size_t m = 0; m < first.size(); m++) {
            REQUIRE_CTX(first[m].description == second[m].description,
                        "%s: mutant %zu is named %s in one pass and %s in the "
                        "other",
                        fixture.name.c_str(), m, first[m].description.c_str(),
                        second[m].description.c_str());
            REQUIRE_CTX(first[m].frame == second[m].frame,
                        "%s: mutant %s differs between passes",
                        fixture.name.c_str(), first[m].description.c_str());

            for (const char* aim : kRequiredAims) {
                if (StartsWith(first[m].description, aim)) {
                    aims_seen.insert(aim);
                }
            }

            Bytes decoded;
            const bool ok = ZstdOracleDecodes(first[m].frame, &decoded);
            if (ok) {
                accepted++;
            } else {
                rejected++;
                rejected_here++;
            }
            digest = Fnv1a64(digest, fixture.name.data(), fixture.name.size());
            digest = Fnv1a64(digest, first[m].description.data(),
                             first[m].description.size());
            if (!first[m].frame.empty()) {
                digest = Fnv1a64(digest, first[m].frame.data(),
                                 first[m].frame.size());
            }
            const unsigned char verdict = ok ? 1u : 0u;
            digest = Fnv1a64(digest, &verdict, 1);
            mutants++;
        }
        REQUIRE_CTX(rejected_here > 0,
                    "%s: not one of its %zu mutants was refused by the "
                    "reference - the mutations are not biting on this fixture",
                    fixture.name.c_str(), first.size());
    }

    /* Every aimed field reached, by name. */
    for (const char* aim : kRequiredAims) {
        REQUIRE_CTX(aims_seen.count(aim) == 1,
                    "no mutant aims at '%s' - the generator or the frame "
                    "walker stopped producing that class, and the count "
                    "alone would not have shown it",
                    aim);
    }

    /* The accepted side is not empty either. A mutation set that only ever
     * produced refusals would prove nothing about over-strictness, which is
     * half of what the corpus is for. */
    REQUIRE_CTX(accepted > 0,
                "every mutant was refused - the corpus carries nothing the "
                "reference accepts, so it cannot catch a decoder that is "
                "stricter than the reference");

    REQUIRE_CTX(digest == kExpectedZstdMutantDigest,
                "mutant corpus digest is 0x%016llx - an oracle or generator "
                "change moved it; update the pin deliberately",
                static_cast<unsigned long long>(digest));

    std::printf("PASS: %zu mutants over %zu zstd fixtures, %zu refused and "
                "%zu accepted by the pinned reference, %zu aimed field "
                "classes present by name, generation deterministic\n",
                mutants, fixtures.size(), rejected, accepted,
                aims_seen.size());
    return 0;
}
