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
    /* The compressed form, in whichever format the generator that produced
     * this fixture names: LZ4 block for MakeLz4BlockFixtures, Snappy block
     * for MakeSnappyFixtures. */
    std::vector<unsigned char> compressed;
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

/* The Snappy corpus, compressed by the pinned oracle so every pair is real by
 * construction. Sizes are chosen against the format's own boundaries - the
 * literal-length class break at 60, the 64-byte copy cap, and the 65536-byte
 * block the compressor splits its input into - rather than against any
 * geometry of the decoder that will read them. Fixture::compressed carries
 * Snappy block streams here, the format the generator names. */
std::vector<Fixture> MakeSnappyFixtures();

/* Mutations placed on the branch set read out of snappy 1.2.2's decoder: the
 * varint preamble, the literal-length classes, all three copy tags, and the
 * offset and length bounds that AppendFromSelf refuses on. The generic
 * truncation and bit-flip ladder from MutateStream is included, so this is a
 * superset of the format-agnostic mutations.
 *
 * The placement walks the stream's elements the way snappy does, and that walk
 * is the harness's, not the decoder's: it decides only WHERE a mutation lands.
 * The verdict on every mutant still comes from the oracle, so a wrong step in
 * the walk can only make a mutation less targeted - never turn a rejected
 * stream into an accepted one or the other way round.
 *
 * As with MutateStream, a mutant is NOT necessarily invalid: a non-minimal
 * preamble is a mutation the reference accepts by design. Consumers ask the
 * oracle. */
std::vector<Mutant> MutateSnappyStream(const std::vector<unsigned char>& stream,
                                       uint64_t seed);

/* The Zstd oracle (facebook/zstd, pinned in tests/CMakeLists.txt), the same
 * authority over stream validity that liblz4 holds for LZ4 and snappy for
 * Snappy. Decode only: the wrapper links the decode-only archive, so nothing
 * reachable from here can compress.
 *
 * A zstd frame carries its own content size when the compressor wrote one, so
 * the parity call needs no size argument in the common case. Two values it can
 * return instead are NOT sizes and are refused here rather than passed on as
 * one: ZSTD_CONTENTSIZE_UNKNOWN, which a streaming-produced frame legitimately
 * carries, and ZSTD_CONTENTSIZE_ERROR, which is the header itself being
 * malformed. A caller that wants the first case decoded must say how large a
 * buffer it is willing to own, which is what the size argument below is for.
 *
 * kZstdOracleMaxOutput is this wrapper's bound and not zstd's, for the same
 * reason kSnappyOracleMaxOutput is: a mutated frame will declare a content
 * size no corpus here produces, and the harness refuses to allocate on it. */
const size_t kZstdOracleMaxOutput = 256u * 1024u * 1024u;
bool ZstdOracleDecodes(const std::vector<unsigned char>& stream,
                       std::vector<unsigned char>* decoded);

/* The declared-size surface on its own, for the tests that are about the
 * frame header rather than about the payload. Returns false when zstd calls
 * the header malformed; sets *unknown when the frame declares no content
 * size, which is a valid frame and not a rejection. */
bool ZstdOracleContentSize(const std::vector<unsigned char>& stream,
                           uint64_t* size, bool* unknown);

/* zstd's own error CLASS for a stream it refuses, so a fail-closed matrix row
 * can name the rejection reason instead of settling for a bare non-zero.
 * Returns 0 when zstd's decode succeeded. The values are ZSTD_ErrorCode from
 * zstd_errors.h, kept as an int here so no test translation unit needs the
 * oracle header - the include paths are PRIVATE to the fixtures archive by
 * design.
 *
 * MEASURED, and it bounds what a matrix row may claim: on zstd 1.5.7's simple
 * API the code alone does NOT separate the refusal shapes. An unrecognised
 * prefix, a pre-v0.8 legacy magic with the legacy decoders compiled out, and a
 * valid frame with its last four bytes removed all come back
 * ZSTD_error_srcSize_wrong (72). What separates them is the pair of this value
 * and the content-size verdict above: the truncated frame's header still
 * parses and yields its real size, the other two do not parse at all.
 * tests/zstd_oracle_smoke.cpp pins all three. */
int ZstdOracleErrorCode(const std::vector<unsigned char>& stream);

/* Returned when this wrapper refused before zstd was reached. Negative, so it
 * cannot collide with a ZSTD_ErrorCode, which is an unsigned enumeration. */
const int kZstdOracleNotAFrame = -1;

#endif /* CUDEC_TESTS_FIXTURES_H */
