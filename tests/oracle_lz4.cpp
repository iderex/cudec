/* The oracle net proves itself before any kernel exists: the fixture
 * pairs are real (oracle self-round-trip), the mutation machinery is
 * deterministic (two generation passes agree, verdicts agree), and the
 * mutations actually bite (at least one rejected mutant per fixture -
 * "all rejected" would be false: some bit flips yield valid streams). */
#include "fixtures.h"
#include "require.h"

#include <cstdio>
#include <vector>

namespace {

/* Corpus drift tripwire: the determinism invariant ("bit-identical corpora
 * across platforms and runs") must be executable, not argued. The pinned
 * digest moves whenever an oracle bump shifts LZ4_compress_default output
 * (and with it the whole mutant corpus and its verdicts) - the bump PR
 * then updates the pin consciously instead of the corpus drifting green.
 * FNV-1a, not crypto: a tripwire against drift, not a defense. */
constexpr uint64_t kExpectedCorpusDigest = 0x04e2fb5078db9e2bull;

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
    const auto fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());

    uint64_t digest = 14695981039346656037ull;
    for (const auto& f : fixtures) {
        digest = Fnv1a64(digest, f.name.data(), f.name.size());
        digest = Fnv1a64(digest, f.compressed.data(), f.compressed.size());
    }
    REQUIRE_CTX(digest == kExpectedCorpusDigest,
                "corpus digest is 0x%016llx - an oracle or generator change "
                "moved the fixtures; verify deliberately, then update the pin",
                static_cast<unsigned long long>(digest));

    size_t mutant_total = 0;
    for (const auto& f : fixtures) {
        std::vector<unsigned char> decoded;
        REQUIRE_CTX(OracleDecodes(f.compressed, f.original.size(), &decoded),
                    "fixture %s", f.name.c_str());
        REQUIRE_CTX(decoded.size() == f.original.size(), "fixture %s",
                    f.name.c_str());
        REQUIRE_CTX(equal_bytes(decoded.data(), f.original.data(),
                                decoded.size()),
                    "fixture %s", f.name.c_str());

        const auto mutants = MutateStream(f.compressed, f.seed);
        const auto mutants_again = MutateStream(f.compressed, f.seed);
        REQUIRE_CTX(mutants.size() == mutants_again.size(), "fixture %s",
                    f.name.c_str());
        REQUIRE_CTX(!mutants.empty(), "fixture %s", f.name.c_str());
        size_t rejected = 0;
        for (size_t i = 0; i < mutants.size(); i++) {
            REQUIRE_CTX(mutants[i].stream == mutants_again[i].stream,
                        "generation determinism: fixture %s mutant %zu (%s)",
                        f.name.c_str(), i, mutants[i].description.c_str());
            std::vector<unsigned char> first_out;
            std::vector<unsigned char> second_out;
            const bool first = OracleDecodes(mutants[i].stream,
                                             f.original.size(), &first_out);
            const bool second = OracleDecodes(mutants[i].stream,
                                              f.original.size(), &second_out);
            REQUIRE_CTX(first == second,
                        "verdict determinism: fixture %s mutant %zu (%s)",
                        f.name.c_str(), i, mutants[i].description.c_str());
            if (!first) {
                rejected++;
            }
        }
        REQUIRE_CTX(rejected > 0,
                    "no mutant rejected for fixture %s - mutations do not "
                    "bite",
                    f.name.c_str());
        mutant_total += mutants.size();
    }
    std::printf("PASS: %zu fixtures round-trip; %zu mutants, machinery "
                "deterministic and biting\n",
                fixtures.size(), mutant_total);
    return 0;
}
