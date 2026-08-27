/* The M5 Zstd corpus: fixtures that reach every decode surface the kernel
 * must handle, plus the frame walker that proves each one got the mode it
 * asked for (issue #185).
 *
 * Two halves, deliberately separate. MakeZstdFixtures() drives the pinned
 * compressor through its advanced API; ParseZstdFrameShape() reads the
 * emitted bytes back and reports which modes are actually in them. A forced
 * parameter the compressor declines is then a test failure rather than
 * coverage silently lost, which is the rule this corpus exists to carry.
 *
 * The walker reads headers only. Block sizes, literals-section sizes and the
 * sequence count are all in the clear, so every mode byte is reachable
 * without entropy decoding, and no second decoder enters the tree.
 *
 * Source bytes are constructed here, never fetched: the self-proof runs on
 * the GPU-less CI runner, which fetches no corpus. docs/ZSTD-CORPUS.md
 * records that decision and where the Silesia rung lives instead. */
#ifndef CUDEC_TESTS_ZSTD_CORPUS_H
#define CUDEC_TESTS_ZSTD_CORPUS_H

#include <cstdint>
#include <string>
#include <vector>

/* Literals_Block_Type, RFC 8878 section 3.1.1.3.1.1. */
enum ZstdLiteralsType {
    kZstdLiteralsRaw = 0,
    kZstdLiteralsRle = 1,
    kZstdLiteralsCompressed = 2,
    kZstdLiteralsTreeless = 3
};

/* Block_Type, RFC 8878 section 3.1.1.2. */
enum ZstdBlockType {
    kZstdBlockRaw = 0,
    kZstdBlockRle = 1,
    kZstdBlockCompressed = 2
};

/* Symbol_Compression_Mode, RFC 8878 section 3.1.1.3.2.1.1. */
enum ZstdTableMode {
    kZstdTableBasic = 0,
    kZstdTableRle = 1,
    kZstdTableCompressed = 2,
    kZstdTableRepeat = 3
};

/* What one block of an emitted frame turned out to be. Per-block rather than
 * per-frame because a single frame legitimately mixes modes - a Treeless
 * literals block only exists next to the Compressed one whose table it
 * reuses, and that pairing is the surface, not an accident. */
struct ZstdBlockShape {
    unsigned block_type = 0;
    bool last = false;
    /* Set only for a compressed block. */
    unsigned literals_type = 0;
    unsigned literals_streams = 0; /* 1 or 4 */
    unsigned sequence_count = 0;
    /* Set only where sequence_count > 0. */
    unsigned ll_mode = 0;
    unsigned of_mode = 0;
    unsigned ml_mode = 0;
    /* Where the three FSE table descriptions begin, as an offset into the
     * frame, and where the block ends. The walk already stands on the first
     * of those bytes when it has read the mode byte, and throwing the
     * position away made every consumer re-derive it - so it is reported
     * instead. Set only where sequence_count > 0; block_end is set for every
     * compressed block. */
    size_t tables_offset = 0;
    size_t block_end = 0;
    /* Where a Compressed or Treeless literals section's payload begins and
     * how long it is, as an offset into the frame. Reported for the same
     * reason as the two above: the walk reads the size in order to step over
     * it, and a consumer that wants the Huffman tree description would
     * otherwise re-read the header to find where the walk had just been. Set
     * only for those two literals types. */
    size_t literals_payload_offset = 0;
    size_t literals_payload_size = 0;
    /* What the literals section regenerates, for every literals type. The
     * walk reads it in order to step over a Raw section and carries it in the
     * same packed word as the compressed size for the other two, so it is
     * reported for the reason the three fields above are. It is the
     * denominator any statement about how much literal WORK sits in a given
     * section shape has to be taken over: the section count answers a
     * different question and the two diverge sharply, because the
     * single-stream spelling caps regenerated size at 1023 bytes while the
     * four-stream ones reach 262143. */
    size_t literals_regenerated_size = 0;
};

struct ZstdFrameShape {
    bool single_segment = false;
    bool content_size_present = false;
    bool checksum_present = false;
    std::vector<ZstdBlockShape> blocks;
};

/* An expectation, one field per decode surface. Negative means unconstrained.
 * Frame-level fields must hold for the frame; block-level fields must hold
 * for at least one block in it. */
struct ZstdDemand {
    int single_segment = -1;
    int content_size = -1;
    int checksum = -1;
    int block_type = -1;
    int literals_type = -1;
    int literals_streams = -1;
    int ll_mode = -1;
    int of_mode = -1;
    int ml_mode = -1;
    int min_blocks = -1;
};

struct ZstdFixture {
    std::string name;
    std::string family;
    std::vector<unsigned char> original;
    std::vector<unsigned char> compressed;
    ZstdDemand demand;
};

/* Walks a frame's headers. Returns false on anything it cannot account for,
 * so a shape is never reported from a frame the walker only partly
 * understood. */
bool ParseZstdFrameShape(const std::vector<unsigned char>& frame,
                         ZstdFrameShape* out, std::string* why);

/* True when every field the demand constrains is present in the shape. */
bool ZstdShapeSatisfies(const ZstdFrameShape& shape, const ZstdDemand& demand,
                        std::string* why);

/* Deterministic: two calls return byte-identical fixtures in the same order.
 * Empty on a compressor failure, which the self-proof treats as a failure. */
std::vector<ZstdFixture> MakeZstdFixtures();

/* The batch rung: one source cut at a fixed chunk size, every chunk its own
 * independent frame, which is the geometry cudec_zstd_decompress_batch
 * consumes. Source-agnostic on purpose - pointing it at fetched Silesia
 * bytes is a caller's choice and needs no change here. */
std::vector<std::vector<unsigned char>> MakeZstdBatchFrames(
    const std::vector<unsigned char>& source, size_t chunk_size, int level);

/* Round-trips one frame through the pinned decoder. */
bool ZstdOracleDecodes(const std::vector<unsigned char>& frame,
                       std::vector<unsigned char>* out);

/* One mutated frame and what was done to it. The description is the repro
 * key: a failure names the fixture and this string and nothing else is
 * needed to rebuild the bytes. */
struct ZstdMutant {
    std::string description;
    std::vector<unsigned char> frame;
};

/* The mutation layer over Zstd frames (issue #187).
 *
 * TWO KINDS OF MUTATION, and the second is the point. The blind kind -
 * truncations and seeded single-bit flips - is what the LZ4 and Snappy
 * corpora already do, and over a frame format with a header, a block table
 * and two checksums it mostly lands in an entropy payload and produces a
 * checksum failure. The aimed kind walks the frame's own headers and moves
 * one named field at a time: the magic, each descriptor flag, the window
 * byte, the content size, a block header's type and size, the literals
 * section's type and size format, the Huffman description's first byte, the
 * sequence count, the Symbol_Compression_Modes byte including its reserved
 * bits, and an FSE table description's accuracy log. That set is read out of
 * what the reference decoder branches on rather than out of anything cudec
 * does.
 *
 * WHERE THE AIM STOPS is where the walker stops: it reads headers and steps
 * over payloads by their declared sizes, so a mutation cannot be aimed at a
 * field inside an entropy bitstream. The blind flips are what reach those,
 * and they reach them without knowing what they hit.
 *
 * A MUTANT IS NOT ASSUMED INVALID. Some of these are accepted by the
 * reference, and that is kept rather than filtered: cudec being stricter than
 * the reference is a divergence in the same way cudec being looser is. The
 * verdict always comes from ZstdOracleDecodes, never from this function.
 *
 * Deterministic: same frame and seed, same mutants, byte for byte. */
std::vector<ZstdMutant> MutateZstdFrame(const std::vector<unsigned char>& frame,
                                        uint64_t seed);

#endif /* CUDEC_TESTS_ZSTD_CORPUS_H */
