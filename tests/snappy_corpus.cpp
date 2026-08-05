/* The Snappy corpus proving itself before any Snappy decoder exists, the
 * counterpart of oracle_lz4.cpp: the fixture pairs are real (the oracle
 * round-trips every one), the mutation machinery is deterministic (two
 * generation passes agree on the mutants and on their verdicts), and the
 * mutations bite (at least one rejected mutant per fixture - "all rejected"
 * would be false, since a non-minimal preamble is a mutation the reference
 * accepts by design).
 *
 * The oracle is the sole authority on validity here: docs/MASTERPLAN.md, "the
 * oracles decide". */
#include "fixtures.h"
#include "require.h"

#include <cstdio>
#include <vector>

namespace {

/* Corpus drift tripwire, the same instrument oracle_lz4.cpp carries and for
 * the same reason: a snappy bump that moves RawCompress's output moves every
 * mutant and every verdict with it, and the bump PR should have to update this
 * number on purpose instead of watching a changed corpus pass. FNV-1a, not
 * crypto: a tripwire against drift, not a defense. */
constexpr uint64_t kExpectedSnappyCorpusDigest = 0x79af9459a11bfd4cull;

uint64_t Fnv1a64(uint64_t hash, const void* data, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace

int main() {
    const auto fixtures = MakeSnappyFixtures();
    REQUIRE(!fixtures.empty());

    uint64_t digest = 14695981039346656037ull;
    for (const auto& f : fixtures) {
        digest = Fnv1a64(digest, f.name.data(), f.name.size());
        digest = Fnv1a64(digest, f.compressed.data(), f.compressed.size());
    }
    REQUIRE_CTX(digest == kExpectedSnappyCorpusDigest,
                "corpus digest is 0x%016llx - an oracle or generator change "
                "moved the fixtures; verify deliberately, then update the pin",
                static_cast<unsigned long long>(digest));

    size_t mutant_total = 0;
    size_t rejected_total = 0;
    for (const auto& f : fixtures) {
        std::vector<unsigned char> decoded;
        REQUIRE_CTX(SnappyOracleDecodes(f.compressed, &decoded), "fixture %s",
                    f.name.c_str());
        REQUIRE_CTX(decoded.size() == f.original.size(), "fixture %s",
                    f.name.c_str());
        REQUIRE_CTX(
            equal_bytes(decoded.data(), f.original.data(), decoded.size()),
            "fixture %s", f.name.c_str());
        /* The two verdict routes must agree on a stream the compressor made:
         * a validator that quietly disagreed with the decoder would make every
         * reject-parity claim built on it ambiguous. */
        REQUIRE_CTX(SnappyOracleAccepts(f.compressed), "fixture %s",
                    f.name.c_str());

        const auto mutants = MutateSnappyStream(f.compressed, f.seed);
        const auto mutants_again = MutateSnappyStream(f.compressed, f.seed);
        REQUIRE_CTX(!mutants.empty(), "fixture %s", f.name.c_str());
        /* The branch-derived half has to be there, per fixture. Without this
         * the suite would pass on the generic truncation ladder alone if the
         * element walk ever stopped finding elements - the corpus would still
         * look large, deterministic and biting while having stopped targeting
         * the reference's branches at all. */
        size_t preamble_mutants = 0;
        size_t element_mutants = 0;
        for (const auto& m : mutants) {
            if (m.description.compare(0, 9, "preamble-") == 0) {
                preamble_mutants++;
            }
            if (m.description.compare(0, 8, "element-") == 0) {
                element_mutants++;
            }
        }
        REQUIRE_CTX(preamble_mutants >= 6,
                    "fixture %s carries %zu preamble mutants", f.name.c_str(),
                    preamble_mutants);
        /* Four is what one element yields at the floor: the three tag classes
         * it is not, plus one reachable step of its length field. The
         * one-byte fixture is exactly that case. */
        REQUIRE_CTX(element_mutants >= 4,
                    "fixture %s carries %zu element mutants - the element walk "
                    "found nothing to target",
                    f.name.c_str(), element_mutants);
        REQUIRE_CTX(mutants.size() == mutants_again.size(), "fixture %s",
                    f.name.c_str());
        size_t rejected = 0;
        for (size_t i = 0; i < mutants.size(); i++) {
            REQUIRE_CTX(mutants[i].stream == mutants_again[i].stream,
                        "generation determinism: fixture %s mutant %zu (%s)",
                        f.name.c_str(), i, mutants[i].description.c_str());
            REQUIRE_CTX(mutants[i].description == mutants_again[i].description,
                        "generation determinism: fixture %s mutant %zu",
                        f.name.c_str(), i);
            std::vector<unsigned char> first_out;
            std::vector<unsigned char> second_out;
            const bool first =
                SnappyOracleDecodes(mutants[i].stream, &first_out);
            const bool second =
                SnappyOracleDecodes(mutants[i].stream, &second_out);
            REQUIRE_CTX(first == second,
                        "verdict determinism: fixture %s mutant %zu (%s)",
                        f.name.c_str(), i, mutants[i].description.c_str());
            if (first) {
                REQUIRE_CTX(first_out == second_out,
                            "output determinism: fixture %s mutant %zu (%s)",
                            f.name.c_str(), i,
                            mutants[i].description.c_str());
            } else {
                REQUIRE_CTX(first_out.empty(),
                            "a rejected mutant produced output: fixture %s "
                            "mutant %zu (%s)",
                            f.name.c_str(), i,
                            mutants[i].description.c_str());
                rejected++;
            }
        }
        REQUIRE_CTX(rejected > 0,
                    "no mutant rejected for fixture %s - mutations do not bite",
                    f.name.c_str());
        mutant_total += mutants.size();
        rejected_total += rejected;
    }
    /* Both directions across the whole corpus, so the suite cannot pass with a
     * generator that only ever produces streams the reference refuses - which
     * would make every accept-side claim built on this corpus vacuous. */
    REQUIRE(rejected_total > 0);
    REQUIRE(rejected_total < mutant_total);

    std::printf("PASS: %zu snappy fixtures round-trip; %zu mutants, %zu "
                "rejected, machinery deterministic and biting\n",
                fixtures.size(), mutant_total, rejected_total);
    return 0;
}
