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

#endif /* CUDEC_TESTS_ZSTD_CORPUS_H */
