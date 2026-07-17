/* Deterministic test corpus: LZ4 block pairs compressed by the CPU oracle,
 * plus a mutation generator for the negative net. Everything is seeded and
 * reproducible - a failure's (fixture, mutant) coordinates are a complete
 * repro key. The oracle (liblz4) is the sole authority on stream validity:
 * docs/MASTERPLAN.md, "the oracles decide". */
#ifndef CUDEC_TESTS_FIXTURES_H
#define CUDEC_TESTS_FIXTURES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Fixture {
    std::string name;
    uint64_t seed;
    std::vector<unsigned char> original;
    std::vector<unsigned char> compressed; /* LZ4 block format */
};

/* Corpus chosen against the future kernel geometry: max-compressible,
 * incompressible, match-heavy text, and sizes crossing warp/block
 * boundaries. */
std::vector<Fixture> MakeLz4BlockFixtures();

struct Mutant {
    std::string description;
    std::vector<unsigned char> stream;
};

/* Deterministic truncations (fixed fractions; 1..8 bytes off either end)
 * and seeded single-bit flips of one compressed stream. */
std::vector<Mutant> MutateStream(const std::vector<unsigned char>& stream,
                                 uint64_t seed);

/* CPU-reference decode verdict (and output when accepted). */
bool OracleDecodes(const std::vector<unsigned char>& stream,
                   size_t original_size, std::vector<unsigned char>* decoded);

/* CPU-reference block compression. Fixture generation and the bench
 * harness share it so every compressed stream in the project has a single
 * provenance; aborts on compressor failure (infrastructure, not a test). */
std::vector<unsigned char> Lz4CompressBlock(
    const std::vector<unsigned char>& original);

#endif /* CUDEC_TESTS_FIXTURES_H */
