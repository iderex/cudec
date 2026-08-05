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

/* The Snappy oracle (google/snappy, pinned in tests/CMakeLists.txt), same
 * authority over stream validity that liblz4 holds for LZ4.
 *
 * The parity call: snappy's exact-consume, exact-produce decode. The declared
 * uncompressed length comes out of the stream itself, so a verdict needs no
 * size argument - unlike the LZ4 block format, which has none.
 *
 * kSnappyOracleMaxOutput is this wrapper's bound and not snappy's: a stream
 * may declare up to 4 GiB - 1 of output, and a mutated one will, so the
 * harness refuses to allocate on a declared length no corpus here produces.
 * A generator that ever needs more must raise the bound consciously, because
 * above it this wrapper rejects a stream snappy itself might accept. */
const size_t kSnappyOracleMaxOutput = 256u * 1024u * 1024u;
bool SnappyOracleDecodes(const std::vector<unsigned char>& stream,
                         std::vector<unsigned char>* decoded);

/* The no-write validation analog: snappy's own verdict on a stream, reached
 * without producing output. */
bool SnappyOracleAccepts(const std::vector<unsigned char>& stream);

/* CPU-reference Snappy compression, the single provenance for every Snappy
 * stream in the project; aborts on compressor failure for the same reason
 * Lz4CompressBlock does. */
std::vector<unsigned char> SnappyCompressBlock(
    const std::vector<unsigned char>& original);

#endif /* CUDEC_TESTS_FIXTURES_H */
