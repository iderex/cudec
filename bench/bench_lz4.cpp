/* The benchmark harness (M0 skeleton): times batch LZ4 block decode
 * through the CPU oracle - the only decoder that exists until M1, which
 * plugs in as a second timed path in this same harness. A report cannot
 * be produced without its methodology block; that is the point
 * (docs/MASTERPLAN.md section 5, honest numbers). */
#include "bench_stats.h"
#include "cudec.h"
#include "fixtures.h"
#include "gpu_bench.h"

#include <lz4.h>
#include <lz4frame.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

/* The LZ4 block-format ceiling used across the project. */
constexpr size_t kChunkBytes = 65536;

constexpr size_t kMaxRuns = 1000000;

/* The worst-case sweep replicates one adversarial block to this many chunks:
 * enough 64 KB warps to saturate the RTX 3080 (68 SMs, >= 32 warps/SM) and
 * the same ~200 MB scale as the Silesia GPU row, so the two throughput
 * numbers are directly comparable. */
constexpr size_t kWorst4bChunks = 3200;

/* The CI rot check (--worst4b --selfcheck) exercises the identical
 * construction on a handful of chunks so it stays fast on the GPU-less
 * runner; the block itself is the same regardless of the replica count. */
constexpr size_t kWorst4bSelfcheckChunks = 4;

/* The long-non-overlap-match corpus (issue #36). Same 64 KB chunk and
 * ~200 MB scale as the Silesia/worst-4Bmatch GPU rows so the three
 * throughput numbers are directly comparable. Its purpose is to expose the
 * ONE regime neither recorded corpus measures: long matches whose source
 * range does not overlap the destination (offset >= match_len), where the
 * match copy has enough bytes to become issue/ALU-bound and the per-byte
 * 64-bit modulo in the closed-form gather - unnecessary when offset >=
 * match_len - would dominate if it were on the critical path. */
constexpr size_t kLongmatchChunks = 3200;
constexpr size_t kLongmatchSelfcheckChunks = 4;

/* The block's shape: an initial literal seed the length of the match offset
 * (so the first match's source is in-bounds), then back-to-back 255-byte
 * matches at that fixed offset, then a literal tail. The offset is
 * held >= the match length so EVERY match is non-overlapping - the exact,
 * and only, case the fast path targets; the static_assert locks that. */
constexpr size_t kLongmatchOffset = 512;
constexpr size_t kLongmatchMatchLen = 255;
static_assert(kLongmatchOffset >= kLongmatchMatchLen,
              "longmatch must stay non-overlapping (offset >= match_len): "
              "that disjoint-range regime is exactly what issue #36's fast "
              "path targets, and an overlapping block would measure the "
              "wrong thing");

/* The game-asset-like corpus (issue #139). Same 64 KB chunk and ~200 MB
 * scale as the three rows above so every throughput number in the record is
 * directly comparable. It models the payload of a shipped asset package -
 * block-compressed texture, interleaved geometry, streamed audio - and it is
 * the one regime none of the recorded corpora reaches: most of the block is
 * incompressible, so the decode is dominated by literal transfer rather than
 * by sequence parsing (worst-4Bmatch) or by match copying (longmatch).
 *
 * It is a MODEL of the workload, not the workload. A synthetic mixture is
 * not a measurement on real game data, and no number taken here may be
 * quoted as one; docs/BENCHMARK-METHODOLOGY.md carries the same sentence
 * beside the corpus entry. The maintainer's decision on issue #139 chose the
 * generator over a hash-pinned asset pack deliberately: no network, no mirror
 * that can rot, no licence review on a third-party package, and CI stays
 * offline. A vetted pack may later join the set beside this generator; it
 * does not replace it. */
constexpr size_t kAssetlikeChunks = 3200;
constexpr size_t kAssetlikeSelfcheckChunks = 4;

/* The three regions of the block, in bytes, tiling one chunk exactly. The
 * proportions follow a shipped package: block-compressed texture dominates,
 * geometry follows, streamed audio fills the rest. */
constexpr size_t kAssetTextureBytes = 32768;
constexpr size_t kAssetVertexBytes = 14336;
constexpr size_t kAssetIndexBytes = 2048;
constexpr size_t kAssetAudioBytes = 16384;
static_assert(kAssetTextureBytes + kAssetVertexBytes + kAssetIndexBytes +
                      kAssetAudioBytes ==
                  kChunkBytes,
              "the asset-like regions must tile the 64 KB chunk exactly: a "
              "short block would change the chunk-size distribution the "
              "report attests");

/* BC1/DXT1 layout: two 16-bit endpoints plus 32 bits of 2-bit indices. */
constexpr size_t kAssetTexBlockBytes = 8;
/* A 64-block-wide image, in 8x8-block tiles. */
constexpr size_t kAssetTexBlocksPerRow = 64;
constexpr size_t kAssetTexTileBlocks = 8;
/* Interleaved position/normal/uv/colour/tangent record. */
constexpr size_t kAssetVertexStride = 32;
/* The lattice the geometry sits on, and the row step the index buffer uses. */
constexpr size_t kAssetLatticeWidth = 28;

struct Corpus {
    std::string name;
    std::vector<std::vector<unsigned char>> originals;
    std::vector<std::vector<unsigned char>> compressed;
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    /* How the compressed streams were produced - printed verbatim in the
     * methodology block, so it must stay true for whichever corpus ran. */
    std::string provenance = "compressed in-harness via LZ4_compress_default";
};

bool AppendFileChunked(const std::string& path, Corpus* corpus) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open corpus file: %s\n", path.c_str());
        return false;
    }
    const size_t chunks_before = corpus->originals.size();
    while (true) {
        std::vector<unsigned char> chunk(kChunkBytes);
        in.read(reinterpret_cast<char*>(chunk.data()),
                static_cast<std::streamsize>(kChunkBytes));
        const std::streamsize got = in.gcount();
        if (got <= 0) {
            break;
        }
        chunk.resize(static_cast<size_t>(got));
        corpus->originals.push_back(std::move(chunk));
    }
    /* Fail closed on I/O trouble and on zero contribution: a file that
     * adds no chunks (empty, unreadable, a directory) must never end up
     * attested in the methodology block. Per-FILE contribution, not the
     * accumulated corpus - the accumulated check goes vacuous from the
     * second argument on. */
    if (in.bad()) {
        std::fprintf(stderr, "read error in corpus file: %s\n", path.c_str());
        return false;
    }
    if (corpus->originals.size() == chunks_before) {
        std::fprintf(stderr, "corpus file contributed no data: %s\n",
                     path.c_str());
        return false;
    }
    return true;
}

void CompressAll(Corpus* corpus) {
    for (const auto& original : corpus->originals) {
        corpus->compressed.push_back(Lz4CompressBlock(original));
        corpus->original_bytes += original.size();
        corpus->compressed_bytes += corpus->compressed.back().size();
    }
}

/* Builds one adversarial-but-valid LZ4 block: back-to-back minimum matches
 * (match length 4, offset 1). This is the maximum sequence density a valid
 * block can carry - one parsed sequence per 4 decoded bytes - and it drives
 * the kernel's per-byte closed-form modular gather on every match byte, so
 * the redundant lockstep parse floors here. LZ4_compress_default never emits
 * it (it extends any offset-1 run into a single long match, the best case),
 * so the stream is constructed directly and proven valid by the oracle
 * before timing.
 *
 * The block decodes to `out_bytes` copies of one seed byte. Wire layout:
 *   token 0x10, 1 seed literal, offset 0x0001      (1 literal + 4 match)
 *   token 0x00, offset 0x0001               x M    (4 match bytes each)
 *   token (literal-length tail), >= 12 trailing literals
 * The trailing literal run keeps the last match clear of the block end,
 * satisfying LZ4's parsing restrictions (LASTLITERALS = 5, last match >= 12
 * bytes before the end); the oracle is the sole authority and confirms the
 * verdict in BuildWorst4bCorpus before any timing. */
bool BuildWorst4bBlock(size_t out_bytes, std::vector<unsigned char>* original,
                       std::vector<unsigned char>* compressed) {
    constexpr unsigned char kSeed = 0xA5;
    constexpr size_t kMinTail = 12;   /* >= LZ4's last-match distance rule */
    constexpr size_t kMinBytes = 256; /* fail loud below this, never wrap the
                                       * out_bytes - kMinTail subtraction */

    /* Only ever called with kChunkBytes today, but the signature invites
     * reuse: a small out_bytes would wrap the size_t subtraction below into
     * a huge loop bound and OOM. Reject it loudly instead. */
    if (out_bytes < kMinBytes) {
        std::fprintf(stderr, "worst-4Bmatch block needs at least %zu output "
                             "bytes, got %zu\n",
                     kMinBytes, out_bytes);
        return false;
    }

    original->assign(out_bytes, kSeed);

    std::vector<unsigned char>& c = *compressed;
    c.clear();
    /* Seed sequence: one real literal, then a length-4 offset-1 match that
     * copies it forward (offset 1 = run-length; the match reads the byte
     * just written). */
    c.push_back(0x10);  /* literal length 1, match length 4 */
    c.push_back(kSeed); /* the one literal byte */
    c.push_back(0x01);  /* offset low byte (offset = 1) */
    c.push_back(0x00);  /* offset high byte */
    size_t produced = 5; /* 1 literal + 4 match bytes */

    /* Minimum matches until only the tail (>= kMinTail literals) remains. */
    while (produced + 4 <= out_bytes - kMinTail) {
        c.push_back(0x00); /* literal length 0, match length 4 */
        c.push_back(0x01);
        c.push_back(0x00);
        produced += 4;
    }

    /* Literals-only tail sequence (no offset/match follows the literals; the
     * decoder detects end-of-block when the input is exhausted). */
    const size_t tail = out_bytes - produced;
    if (tail < 15) {
        c.push_back(static_cast<unsigned char>(tail << 4));
    } else {
        c.push_back(0xF0); /* literal length >= 15: read extension bytes */
        size_t rem = tail - 15;
        while (rem >= 255) {
            c.push_back(255);
            rem -= 255;
        }
        c.push_back(static_cast<unsigned char>(rem));
    }
    c.insert(c.end(), tail, kSeed);
    return true;
}

/* Replicates the worst-case block to `chunks` identical chunks. Rejects an
 * invalid construction before any timing: the oracle (liblz4) is the sole
 * authority on validity, and honest numbers require a stream that actually
 * decodes (docs/MASTERPLAN.md, "the oracles decide"). */
bool BuildWorst4bCorpus(Corpus* corpus, size_t chunks) {
    std::vector<unsigned char> original;
    std::vector<unsigned char> compressed;
    if (!BuildWorst4bBlock(kChunkBytes, &original, &compressed)) {
        return false;
    }

    std::vector<unsigned char> decoded;
    if (!OracleDecodes(compressed, original.size(), &decoded) ||
        decoded.size() != original.size() ||
        std::memcmp(decoded.data(), original.data(), decoded.size()) != 0) {
        std::fprintf(stderr, "worst-4Bmatch construction rejected by the "
                             "oracle - refusing to time an invalid stream\n");
        return false;
    }

    /* Lock the WORST-CASE property, not just validity. A future edit that
     * yields a valid-but-non-adversarial block (say, one long match) would
     * still round-trip and leave CI green while the report claims "worst
     * case". The intended block is ~0.75 (minimum matches barely compress);
     * a single-long-match best case is ~0.0001. Requiring the compression
     * ratio to stay >= 0.70 reds the selfcheck if the generator ever stops
     * being adversarial. original.size() is non-zero (kChunkBytes). */
    const double ratio = static_cast<double>(compressed.size()) /
                         static_cast<double>(original.size());
    if (ratio < 0.70) {
        std::fprintf(stderr, "worst-4Bmatch block is not adversarial: "
                             "compressed/original %.4f below the 0.70 "
                             "sequence-density floor\n",
                     ratio);
        return false;
    }

    corpus->name = "worst-4Bmatch";
    corpus->originals.assign(chunks, original);
    corpus->compressed.assign(chunks, compressed);
    corpus->original_bytes = original.size() * chunks;
    corpus->compressed_bytes = compressed.size() * chunks;
    corpus->provenance = "hand-constructed offset-1 minmatch worst case "
                         "(oracle-validated; LZ4_compress_default never emits "
                         "it)";
    return true;
}

/* Appends an LZ4 length-field extension for a field whose 4-bit token nibble
 * is saturated at 15: the reference format encodes (value - 15) as a run of
 * 255 bytes followed by the remainder. `value` is the full field (>= 15;
 * literal length, or match length minus MINMATCH). */
void EmitLengthExtension(std::vector<unsigned char>* c, size_t value) {
    size_t rem = value - 15;
    while (rem >= 255) {
        c->push_back(255);
        rem -= 255;
    }
    c->push_back(static_cast<unsigned char>(rem));
}

unsigned char LengthNibble(size_t value) {
    return static_cast<unsigned char>(value < 15 ? value : 15);
}

/* Builds one valid LZ4 block of long, NON-overlapping matches (issue #36):
 * a `kLongmatchOffset`-byte literal seed, then back-to-back matches of
 * `kLongmatchMatchLen` bytes at that fixed offset, then a literal tail. The
 * output is periodic with period `kLongmatchOffset` (each match copies the
 * window one offset back), so the whole block is `seed[i % offset]` and the
 * hand-built wire reproduces it exactly - proven by the oracle before timing.
 *
 * Every match satisfies offset >= match_len (512 >= 255), so its source and
 * destination ranges are disjoint: this is precisely the case where the
 * closed-form gather's per-byte 64-bit modulo is provably unnecessary
 * (i % offset == i for i < match_len <= offset). The standard compressor
 * never emits this shape (it would extend the fixed-offset run into a single
 * long match), so the stream is constructed directly. */
bool BuildLongmatchBlock(size_t out_bytes,
                         std::vector<unsigned char>* original,
                         std::vector<unsigned char>* compressed) {
    constexpr size_t kMinTail = 12; /* >= LZ4's last-match distance rule */
    /* Enough output for the seed, at least one match, and the tail; below
     * this the size_t match-count arithmetic below would wrap. */
    constexpr size_t kMinBytes = kLongmatchOffset + kLongmatchMatchLen +
                                 kMinTail + 16;
    if (out_bytes < kMinBytes) {
        std::fprintf(stderr, "longmatch block needs at least %zu output "
                             "bytes, got %zu\n",
                     kMinBytes, out_bytes);
        return false;
    }

    /* The seed pattern. Any deterministic non-degenerate fill works: the
     * matches reproduce it and the oracle is the sole validity authority. */
    original->resize(out_bytes);
    for (size_t i = 0; i < out_bytes; i++) {
        (*original)[i] =
            static_cast<unsigned char>((i % kLongmatchOffset) * 191 + 13);
    }

    /* As many full matches as fit while leaving >= kMinTail literal bytes. */
    const size_t matches =
        (out_bytes - kLongmatchOffset - kMinTail) / kLongmatchMatchLen;
    const size_t body = kLongmatchOffset + matches * kLongmatchMatchLen;
    const size_t tail = out_bytes - body;

    const size_t match_field = kLongmatchMatchLen - 4; /* minus MINMATCH */
    std::vector<unsigned char>& c = *compressed;
    c.clear();

    /* Seed sequence: `kLongmatchOffset` literals, then the first match. */
    c.push_back(static_cast<unsigned char>(
        (LengthNibble(kLongmatchOffset) << 4) | LengthNibble(match_field)));
    if (kLongmatchOffset >= 15) {
        EmitLengthExtension(&c, kLongmatchOffset);
    }
    c.insert(c.end(), original->begin(),
             original->begin() + static_cast<long>(kLongmatchOffset));
    c.push_back(static_cast<unsigned char>(kLongmatchOffset & 0xFF));
    c.push_back(static_cast<unsigned char>((kLongmatchOffset >> 8) & 0xFF));
    if (match_field >= 15) {
        EmitLengthExtension(&c, match_field);
    }

    /* Remaining matches: zero literals, same fixed offset and length. */
    for (size_t m = 1; m < matches; m++) {
        c.push_back(static_cast<unsigned char>(LengthNibble(match_field)));
        c.push_back(static_cast<unsigned char>(kLongmatchOffset & 0xFF));
        c.push_back(static_cast<unsigned char>((kLongmatchOffset >> 8) & 0xFF));
        if (match_field >= 15) {
            EmitLengthExtension(&c, match_field);
        }
    }

    /* Literals-only tail (no offset/match follows; end-of-block is detected
     * at exact input consumption), keeping the last match clear of the block
     * end per LZ4's parsing restrictions. */
    c.push_back(static_cast<unsigned char>(LengthNibble(tail) << 4));
    if (tail >= 15) {
        EmitLengthExtension(&c, tail);
    }
    c.insert(c.end(), original->begin() + static_cast<long>(body),
             original->end());
    return true;
}

/* Replicates the long-non-overlap-match block to `chunks` identical chunks.
 * Like BuildWorst4bCorpus, the oracle (liblz4) is the sole validity authority
 * and must accept and round-trip the hand-built stream before any timing, and
 * a shape lock guards against generator rot leaving a valid-but-wrong block. */
bool BuildLongmatchCorpus(Corpus* corpus, size_t chunks) {
    std::vector<unsigned char> original;
    std::vector<unsigned char> compressed;
    if (!BuildLongmatchBlock(kChunkBytes, &original, &compressed)) {
        return false;
    }

    std::vector<unsigned char> decoded;
    if (!OracleDecodes(compressed, original.size(), &decoded) ||
        decoded.size() != original.size() ||
        std::memcmp(decoded.data(), original.data(), decoded.size()) != 0) {
        std::fprintf(stderr, "longmatch construction rejected by the oracle "
                             "- refusing to time an invalid stream\n");
        return false;
    }

    /* Lock the SHAPE, not just validity. The intended block is many long
     * matches over a small literal seed + tail, which compresses hard (ratio
     * ~0.027). A generator that regressed to short matches or mostly literals
     * would climb toward ~1.0; one that collapsed into a single giant match
     * (a decompression-bomb shape, the opposite of this many-long-matches
     * throughput probe) would fall toward ~1e-4. Requiring the ratio to sit
     * in a band reds the selfcheck on either drift. original.size() is
     * non-zero (kChunkBytes). */
    const double ratio = static_cast<double>(compressed.size()) /
                         static_cast<double>(original.size());
    if (ratio < 0.005 || ratio > 0.10) {
        std::fprintf(stderr, "longmatch block is not the intended shape: "
                             "compressed/original %.5f outside the "
                             "[0.005, 0.10] long-non-overlap-match band\n",
                     ratio);
        return false;
    }

    corpus->name = "longmatch";
    corpus->originals.assign(chunks, original);
    corpus->compressed.assign(chunks, compressed);
    corpus->original_bytes = original.size() * chunks;
    corpus->compressed_bytes = compressed.size() * chunks;
    corpus->provenance = "hand-constructed long non-overlapping matches "
                         "(offset 512 >= match length 255; oracle-validated; "
                         "LZ4_compress_default never emits it)";
    return true;
}

/* Deterministic byte source for the asset-like block. SplitMix64, chosen
 * because it is four lines of integer arithmetic with no state beyond a
 * uint64: the corpus must be byte-identical on the CI runner and on the GPU
 * box, and a library generator's stream is not fixed across standard
 * libraries. */
uint64_t SplitMix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Parabolic sine approximation over a 1024-step period, in [-16384, 16384].
 * Integer-only on purpose: a libm sine is free to differ in the last bits
 * between platforms, which would make the corpus - and therefore every
 * recorded number taken on it - not reproducible off this machine. */
int32_t PseudoSine(uint32_t phase) {
    const uint32_t p = phase & 1023u;
    const uint32_t half = p & 511u;
    const int32_t y = static_cast<int32_t>((half * (512u - half)) >> 2);
    return p < 512u ? y : -y;
}

void PushLe16(std::vector<unsigned char>* out, uint16_t v) {
    out->push_back(static_cast<unsigned char>(v & 0xFF));
    out->push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
}

/* A 32-bit float's byte pattern without touching a float: the high half
 * carries sign, exponent and the top mantissa bits, which barely move across
 * neighbouring vertices, while the low half is noise. That split is what
 * makes vertex data compress the way it does - partial matches at the record
 * stride - and building it from integers keeps the block reproducible and
 * the harness free of the floating-point types the project bans in sources. */
void PushPseudoFloat(std::vector<unsigned char>* out, uint16_t hi,
                     uint16_t lo) {
    PushLe16(out, lo);
    PushLe16(out, hi);
}

/* Block-compressed texture: 8-byte BC1 blocks in raster order over a
 * 64-block-wide image. A third of the 8x8-block tiles are flat - every block
 * in them identical, which is what a BC1 flat region really is - and the
 * rest carry smoothly varying endpoints with high-entropy index bits. The
 * flat tiles are the only source of long matches in the whole corpus; the
 * detailed ones are where its incompressible share comes from. */
void BuildTextureRegion(std::vector<unsigned char>* out, uint64_t* rng) {
    const size_t blocks = kAssetTextureBytes / kAssetTexBlockBytes;
    for (size_t b = 0; b < blocks; b++) {
        const size_t bx = b % kAssetTexBlocksPerRow;
        const size_t by = b / kAssetTexBlocksPerRow;
        const size_t tx = bx / kAssetTexTileBlocks;
        const size_t ty = by / kAssetTexTileBlocks;
        const bool flat = ((tx + ty) % 3u) == 0u;
        /* Inside a flat tile the endpoints depend on the TILE, not on the
         * block, so all 64 of its blocks are byte-identical. */
        const uint16_t r =
            static_cast<uint16_t>((flat ? (tx * 5u) : (bx >> 1)) & 0x1F);
        const uint16_t g = static_cast<uint16_t>((flat ? (ty * 9u) : by) &
                                                 0x3F);
        const uint16_t bl = static_cast<uint16_t>(
            (flat ? (tx + ty) : ((bx + by) >> 2)) & 0x1F);
        const uint16_t c0 = static_cast<uint16_t>((r << 11) | (g << 5) | bl);
        const uint16_t c1 = static_cast<uint16_t>(c0 - 0x0821u);
        PushLe16(out, c0);
        PushLe16(out, c1);
        if (flat) {
            for (size_t i = 0; i < 4; i++) {
                out->push_back(0x00);
            }
        } else {
            const uint64_t bits = SplitMix64(rng);
            for (size_t i = 0; i < 4; i++) {
                out->push_back(static_cast<unsigned char>(bits >> (i * 8)));
            }
        }
    }
}

/* Interleaved vertex records at a fixed 32-byte stride: position, packed
 * normal, uv, colour, packed tangent. Everything that varies smoothly across
 * the lattice sits in the high halves; everything noisy sits in the low
 * halves. */
void BuildVertexRegion(std::vector<unsigned char>* out, uint64_t* rng) {
    const size_t records = kAssetVertexBytes / kAssetVertexStride;
    static const unsigned char kPacked[8][4] = {
        {0x7F, 0x00, 0x00, 0x00}, {0x00, 0x7F, 0x00, 0x00},
        {0x00, 0x00, 0x7F, 0x00}, {0x5A, 0x5A, 0x00, 0x00},
        {0x5A, 0x00, 0x5A, 0x00}, {0x00, 0x5A, 0x5A, 0x00},
        {0x49, 0x49, 0x49, 0x00}, {0x81, 0x00, 0x00, 0x00}};
    for (size_t v = 0; v < records; v++) {
        const uint16_t gx = static_cast<uint16_t>(v % kAssetLatticeWidth);
        const uint16_t gy = static_cast<uint16_t>(v / kAssetLatticeWidth);
        const uint64_t noise = SplitMix64(rng);
        PushPseudoFloat(out, static_cast<uint16_t>(0x3F80u + gx),
                        static_cast<uint16_t>(noise & 0xFFFFu));
        PushPseudoFloat(out, static_cast<uint16_t>(0x3F80u + gy),
                        static_cast<uint16_t>((noise >> 16) & 0xFFFFu));
        PushPseudoFloat(out, static_cast<uint16_t>(0x3E00u + ((gx + gy) & 7u)),
                        static_cast<uint16_t>((noise >> 32) & 0xFFFFu));
        const unsigned char* n = kPacked[(gx + gy) & 7u];
        for (size_t i = 0; i < 4; i++) {
            out->push_back(n[i]);
        }
        PushPseudoFloat(out, static_cast<uint16_t>(0x3F00u + (gx & 3u)),
                        static_cast<uint16_t>((noise >> 48) & 0xFFFFu));
        PushPseudoFloat(out, static_cast<uint16_t>(0x3F00u + (gy & 3u)),
                        static_cast<uint16_t>(SplitMix64(rng) & 0xFFFFu));
        out->push_back(static_cast<unsigned char>(0xC0u + (gx & 0x1Fu)));
        out->push_back(static_cast<unsigned char>(0xC0u + (gy & 0x1Fu)));
        out->push_back(0xB4);
        out->push_back(0xFF);
        const unsigned char* t = kPacked[(gx * 3u + gy) & 7u];
        for (size_t i = 0; i < 4; i++) {
            out->push_back(t[i]);
        }
    }
}

/* A triangle-list index buffer over the same lattice: three 16-bit indices
 * per triangle, the base walking forward one vertex at a time. Slowly
 * increasing 16-bit values are the shape that defeats a 4-byte minimum
 * match, which is why the region is carried rather than assumed away. */
void BuildIndexRegion(std::vector<unsigned char>* out) {
    const size_t indices = kAssetIndexBytes / 2;
    const size_t records = kAssetVertexBytes / kAssetVertexStride;
    /* Every emitted index stays inside the vertex region. */
    const size_t bases = records - (kAssetLatticeWidth + 2);
    static const size_t kCorner[3] = {0, 1, kAssetLatticeWidth};
    for (size_t i = 0; i < indices; i++) {
        const size_t base = (i / 3) % bases;
        PushLe16(out, static_cast<uint16_t>(base + kCorner[i % 3]));
    }
}

/* Streamed audio: 16-bit stereo PCM, three detuned oscillators plus a small
 * dither. The high byte of each sample moves smoothly and the low byte does
 * not, so the region is effectively incompressible - which is what audio in
 * an asset package is, and a large part of why this corpus sits where it
 * does on the ratio scale. */
void BuildAudioRegion(std::vector<unsigned char>* out, uint64_t* rng) {
    const size_t frames = kAssetAudioBytes / 4;
    for (size_t f = 0; f < frames; f++) {
        const uint64_t noise = SplitMix64(rng);
        for (size_t ch = 0; ch < 2; ch++) {
            const uint32_t phase = static_cast<uint32_t>(f * 7u + ch * 61u);
            int32_t s = PseudoSine(phase) + (PseudoSine(phase * 3u) >> 1) +
                        (PseudoSine(phase * 11u) >> 2);
            s += static_cast<int32_t>((noise >> (ch * 8)) & 0x7F) - 64;
            PushLe16(out, static_cast<uint16_t>(s & 0xFFFF));
        }
    }
}

void BuildAssetlikeSource(std::vector<unsigned char>* original) {
    uint64_t rng = 0x5C0DEC0A55E7ull;
    original->clear();
    original->reserve(kChunkBytes);
    BuildTextureRegion(original, &rng);
    BuildVertexRegion(original, &rng);
    BuildIndexRegion(original);
    BuildAudioRegion(original, &rng);
}

/* What one pass over a valid LZ4 block says about its shape. */
struct BlockShape {
    size_t sequences;
    size_t literal_bytes;
    size_t match_bytes;
};

/* Walks an LZ4 block and totals its sequence count, literal bytes and match
 * bytes. Used only on a stream the oracle has already accepted, so a
 * malformed walk is a harness bug rather than an input verdict - it returns
 * false and the caller refuses to time anything. Every loop advances `pos`
 * on each iteration and `pos` is bounded by the block size, so the walk
 * terminates on any input. */
bool WalkBlock(const std::vector<unsigned char>& c, size_t out_bytes,
               BlockShape* shape) {
    *shape = BlockShape{0, 0, 0};
    size_t pos = 0;
    while (pos < c.size()) {
        const unsigned char token = c[pos++];
        size_t lit = token >> 4;
        if (lit == 15) {
            while (pos < c.size() && c[pos] == 255) {
                lit += 255;
                pos++;
            }
            if (pos >= c.size()) {
                return false;
            }
            lit += c[pos++];
        }
        if (c.size() - pos < lit) {
            return false;
        }
        shape->literal_bytes += lit;
        pos += lit;
        /* A block ends on a literals-only sequence: no offset follows. */
        if (pos == c.size()) {
            break;
        }
        if (c.size() - pos < 2) {
            return false;
        }
        pos += 2; /* the two little-endian offset bytes */
        size_t match = token & 0x0F;
        if (match == 15) {
            while (pos < c.size() && c[pos] == 255) {
                match += 255;
                pos++;
            }
            if (pos >= c.size()) {
                return false;
            }
            match += c[pos++];
        }
        match += 4; /* MINMATCH */
        shape->sequences++;
        shape->match_bytes += match;
    }
    return shape->literal_bytes + shape->match_bytes == out_bytes;
}

/* Replicates the asset-like block to `chunks` identical chunks. Unlike the
 * two adversarial corpora above, the compressed stream here is ordinary
 * `LZ4_compress_default` output: the shape being modelled is the SOURCE
 * data, and letting the standard compressor decide the sequence structure is
 * the whole point - a hand-built wire would model the compressor too.
 *
 * The oracle still round-trips it before any timing, and the shape lock
 * below is over the four quantities that define the regime rather than over
 * the ratio alone. */
bool BuildAssetlikeCorpus(Corpus* corpus, size_t chunks) {
    std::vector<unsigned char> original;
    BuildAssetlikeSource(&original);
    if (original.size() != kChunkBytes) {
        std::fprintf(stderr, "asset-like block is %zu bytes, expected %zu\n",
                     original.size(), kChunkBytes);
        return false;
    }
    std::vector<unsigned char> compressed = Lz4CompressBlock(original);

    std::vector<unsigned char> decoded;
    if (!OracleDecodes(compressed, original.size(), &decoded) ||
        decoded.size() != original.size() ||
        std::memcmp(decoded.data(), original.data(), decoded.size()) != 0) {
        std::fprintf(stderr, "asset-like construction rejected by the oracle "
                             "- refusing to time an invalid stream\n");
        return false;
    }

    /* Lock the REGIME, not just validity. The intended block is mostly
     * incompressible with a modest number of short-to-medium matches, which
     * is what separates it from every recorded corpus: worst-4Bmatch is
     * parse-bound (one sequence per 4 output bytes), longmatch is copy-bound
     * (few sequences, very long matches), Silesia and enwik8 compress far
     * harder. A generator that drifted toward any of those - or that lost
     * its noise sources and became repetitive, or lost its structure and
     * became pure noise - leaves at least one of these four bands, and the
     * selfcheck reds. The numbers below were measured on the block this
     * generator produces against the pinned liblz4; the bands carry margin
     * for a compressor version bump, not for a changed generator. */
    BlockShape shape;
    if (!WalkBlock(compressed, original.size(), &shape)) {
        std::fprintf(stderr, "asset-like block failed the shape walk - the "
                             "harness cannot attest a regime it cannot "
                             "parse\n");
        return false;
    }
    const double ratio = static_cast<double>(compressed.size()) /
                         static_cast<double>(original.size());
    const double literal_share = static_cast<double>(shape.literal_bytes) /
                                 static_cast<double>(original.size());
    const double mean_match =
        shape.sequences == 0 ? 0.0
                             : static_cast<double>(shape.match_bytes) /
                                   static_cast<double>(shape.sequences);
    if (ratio < 0.70 || ratio > 0.90) {
        std::fprintf(stderr, "asset-like block is not the intended shape: "
                             "compressed/original %.4f outside the "
                             "[0.70, 0.90] band\n",
                     ratio);
        return false;
    }
    if (literal_share < 0.55) {
        std::fprintf(stderr, "asset-like block is not literal-dominated: "
                             "literal share %.4f below the 0.55 floor\n",
                     literal_share);
        return false;
    }
    if (shape.sequences < 1200 || shape.sequences > 4000) {
        std::fprintf(stderr, "asset-like block has %zu sequences, outside the "
                             "[1200, 4000] density band\n",
                     shape.sequences);
        return false;
    }
    if (mean_match < 6.0 || mean_match > 20.0) {
        std::fprintf(stderr, "asset-like block's mean match length %.2f is "
                             "outside the [6, 20] band\n",
                     mean_match);
        return false;
    }

    corpus->name = "asset-like";
    corpus->originals.assign(chunks, original);
    corpus->compressed.assign(chunks, compressed);
    corpus->original_bytes = original.size() * chunks;
    corpus->compressed_bytes = compressed.size() * chunks;
    corpus->provenance = "generated in-harness, a MODEL of a game asset "
                         "package (BC1 texture, interleaved geometry, 16-bit "
                         "PCM audio) and not a measurement on real game "
                         "data; compressed by LZ4_compress_default and "
                         "oracle-validated";
    return true;
}

/* One measured repetition: decode the whole batch, wall clock. The timed
 * region contains ONLY LZ4_decompress_safe calls into a pre-sized scratch
 * buffer - no buffer clears, no allocation - so the label on the number
 * is the number (OracleDecodes zero-fills its output and is therefore
 * used on the untimed verify pass only). Exits on any decode failure:
 * numbers for a broken decoder are not numbers. */
double DecodeAllSeconds(const Corpus& corpus, unsigned char* scratch) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        const int written = LZ4_decompress_safe(
            reinterpret_cast<const char*>(corpus.compressed[i].data()),
            reinterpret_cast<char*>(scratch),
            static_cast<int>(corpus.compressed[i].size()),
            static_cast<int>(corpus.originals[i].size()));
        if (written < 0 ||
            static_cast<size_t>(written) != corpus.originals[i].size()) {
            std::fprintf(stderr, "chunk %zu failed to decode - refusing to "
                                 "time a broken decoder\n",
                         i);
            std::exit(1);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

/* The device line moved to gpu_bench (issue #167): the Snappy harness needs
 * the identical string, and a methodology block is comparable across reports
 * only while every report names the machine the same way. */
std::string CudaDeviceLine() {
    char line[256];
    (void)cudec_bench_gpu_device_line(line, sizeof(line));
    return std::string(line);
}

/* Strict count parsing: whole-string decimal in [min, max] - rejects
 * strtoul's silent garbage-to-0 and negative wraparound alike. */
bool ParseCount(const char* text, size_t min, size_t max, size_t* out) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value < min || value > max) {
        return false;
    }
    *out = value;
    return true;
}

/* The nearest-rank definition this report shares with the GPU rows, from
 * bench/bench_stats.h. */
using cudec_bench::Percentile;

/* ---- the frame mode (issue #132) ----------------------------------------
 *
 * Everything above times the BLOCK path with the transfers excluded. The
 * frame entry point is a different measurement and needs a different one:
 * cudec_lz4f_decompress takes a host frame and returns host bytes, so H2D,
 * the decode, the D2H and the assembly are all inside one synchronous call
 * and the wall around that call is the whole cost. Excluding the transfers
 * here would hide the thing the frame path is suspected of.
 *
 * The independent variable is the block count, which the frame's block-max
 * setting fixes for a given corpus. LZ4F offers four sizes and the three
 * below span 16x on the same bytes; the fourth (4 MB) is not swept because
 * the issue's rungs are these three. Each rung is reported with the block
 * count actually decoded rather than the setting alone, because a corpus
 * shorter than one block would silently collapse the sweep. */
struct FrameRung {
    const char* name;
    LZ4F_blockSizeID_t block_size_id;
    /* The block-max the id names, so the reported block count can be
     * derived by a reader instead of trusted. */
    size_t block_max_bytes;
};

constexpr FrameRung kFrameRungs[] = {
    {"64 KB", LZ4F_max64KB, 1u << 16},
    {"256 KB", LZ4F_max256KB, 1u << 18},
    {"1 MB", LZ4F_max1MB, 1u << 20},
};

/* Compresses `in` into one block-independent frame with a content checksum.
 * Block-independent is not a preference: it is the subset src/frame.cpp
 * accepts, and liblz4's compressor defaults to the linked mode cudec
 * refuses. The content checksum is left on so the timed call pays the
 * verification a real .lz4 file carries. */
bool CompressFrameAt(const std::vector<unsigned char>& in,
                     const FrameRung& rung,
                     std::vector<unsigned char>* frame) {
    LZ4F_preferences_t prefs;
    std::memset(&prefs, 0, sizeof(prefs));
    prefs.frameInfo.blockMode = LZ4F_blockIndependent;
    prefs.frameInfo.blockSizeID = rung.block_size_id;
    prefs.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
    const size_t bound = LZ4F_compressFrameBound(in.size(), &prefs);
    frame->assign(bound, 0);
    const size_t r = LZ4F_compressFrame(frame->data(), bound, in.data(),
                                        in.size(), &prefs);
    if (LZ4F_isError(r)) {
        std::fprintf(stderr, "LZ4F_compressFrame failed at the %s rung: %s\n",
                     rung.name, LZ4F_getErrorName(r));
        return false;
    }
    frame->resize(r);
    return true;
}

/* The block count liblz4 actually emitted, read off the frame rather than
 * computed from the corpus size: a compressor free to end a block early
 * would make the derived number a fiction. Walks the block headers, which
 * is the same walk src/frame.cpp does, and stops at the end mark.
 * Fail-closed: a header that does not fit, or a size that runs past the
 * frame, returns false rather than a partial count. */
bool CountFrameBlocks(const std::vector<unsigned char>& frame,
                      size_t* blocks) {
    /* Magic (4) + FLG + BD, then the optional content size and the header
     * checksum. Only the frames this harness just built are parsed here, so
     * the descriptor's shape is known: no content size, no dictionary id. */
    if (frame.size() < 7) {
        return false;
    }
    const unsigned flg = frame[4];
    const bool content_size = (flg & 0x08) != 0;
    const bool dict_id = (flg & 0x01) != 0;
    const bool block_checksum = (flg & 0x10) != 0;
    size_t pos = 6 + (content_size ? 8u : 0u) + (dict_id ? 4u : 0u) + 1;
    size_t count = 0;
    while (true) {
        if (pos + 4 > frame.size()) {
            return false;
        }
        uint32_t header = 0;
        std::memcpy(&header, frame.data() + pos, 4);
        pos += 4;
        if (header == 0) {
            break; /* the end mark */
        }
        const size_t stored = header & 0x7FFFFFFFu;
        if (stored > frame.size() - pos) {
            return false;
        }
        pos += stored + (block_checksum ? 4u : 0u);
        count++;
    }
    *blocks = count;
    return true;
}

/* One timed decode of the whole frame through the public entry point.
 * Returns the wall seconds, or a negative value if the decode did not
 * produce exactly `expected` bytes - a bench must never time a decoder that
 * is not decoding. */
double FrameDecodeSeconds(const std::vector<unsigned char>& frame,
                          unsigned char* out, size_t out_capacity,
                          size_t expected) {
    size_t written = 0;
    const auto start = std::chrono::steady_clock::now();
    const cudec_status status = cudec_lz4f_decompress(
        frame.data(), frame.size(), out, out_capacity, &written);
    const auto end = std::chrono::steady_clock::now();
    if (status != CUDEC_OK || written != expected) {
        std::fprintf(stderr,
                     "frame decode returned status %d and %zu bytes, wanted "
                     "0 and %zu - refusing to time it\n",
                     static_cast<int>(status), written, expected);
        return -1.0;
    }
    return std::chrono::duration<double>(end - start).count();
}

/* Times one rung and prints its report block. */
bool RunFrameRung(const std::vector<unsigned char>& source,
                  const std::string& corpus_name, const FrameRung& rung,
                  size_t warmup, size_t runs) {
    std::vector<unsigned char> frame;
    if (!CompressFrameAt(source, rung, &frame)) {
        return false;
    }
    size_t blocks = 0;
    if (!CountFrameBlocks(frame, &blocks) || blocks == 0) {
        std::fprintf(stderr, "could not read a block count out of the %s "
                             "frame\n",
                     rung.name);
        return false;
    }

    /* Verified once against the original bytes before anything is timed. A
     * bench that skips this can time a silently wrong decode and report a
     * number for it. */
    std::vector<unsigned char> out(source.size());
    if (FrameDecodeSeconds(frame, out.data(), out.size(), source.size()) < 0 ||
        std::memcmp(out.data(), source.data(), source.size()) != 0) {
        std::fprintf(stderr, "the %s rung does not round-trip; nothing is "
                             "timed\n",
                     rung.name);
        return false;
    }

    for (size_t i = 0; i < warmup; i++) {
        if (FrameDecodeSeconds(frame, out.data(), out.size(),
                               source.size()) < 0) {
            return false;
        }
    }
    std::vector<double> times;
    for (size_t i = 0; i < runs; i++) {
        const double seconds =
            FrameDecodeSeconds(frame, out.data(), out.size(), source.size());
        if (seconds < 0) {
            return false;
        }
        times.push_back(seconds);
    }
    std::sort(times.begin(), times.end());
    const double to_gbps = static_cast<double>(source.size()) / 1e9;

    std::printf("## bench_lz4 frame report\n");
    std::printf("- decoder: cudec_lz4f_decompress (host frame in, host bytes "
                "out; H2D, decode, D2H, assembly and checksums are all "
                "inside the timed call)\n");
    std::printf("- host CPU: %s\n", cudec_bench::HostCpuName().c_str());
    std::printf("- CUDA device: %s\n", CudaDeviceLine().c_str());
    std::printf("- cudec: %d\n", cudec_version());
    std::printf("- corpus: %s, %.2f MB original, %.2f MB frame (ratio "
                "%.3f), one block-independent frame with a content checksum, "
                "compressed in-harness via LZ4F_compressFrame (liblz4 %s)\n",
                corpus_name.c_str(), static_cast<double>(source.size()) / 1e6,
                static_cast<double>(frame.size()) / 1e6,
                static_cast<double>(frame.size()) /
                    static_cast<double>(source.size()),
                LZ4_versionString());
    std::printf("- rung: block max %s (%zu bytes), %zu blocks decoded\n",
                rung.name, rung.block_max_bytes, blocks);
    std::printf("- method: %zu warmup + %zu measured runs, wall clock around "
                "the whole synchronous cudec_lz4f_decompress call; output "
                "byte-verified against the original once before timing; "
                "percentiles are nearest-rank\n",
                warmup, runs);
    std::printf("- wall per run: p50 %.3f ms / p90 %.3f ms / p99 %.3f ms\n",
                Percentile(times, 50) * 1e3, Percentile(times, 90) * 1e3,
                Percentile(times, 99) * 1e3);
    std::printf("- end-to-end throughput: p50 %.3f GB/s / p90 %.3f GB/s / "
                "p99 %.3f GB/s\n",
                to_gbps / Percentile(times, 50),
                to_gbps / Percentile(times, 90),
                to_gbps / Percentile(times, 99));
    return true;
}

/* The frame path reads a corpus file whole rather than in 64 KB pieces:
 * the frame's own block-max is what cuts it, and pre-cutting it here would
 * make the block count a property of this harness instead of of the rung. */
bool AppendFileBytes(const std::string& path,
                     std::vector<unsigned char>* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open corpus file: %s\n", path.c_str());
        return false;
    }
    const size_t before = out->size();
    char buffer[1 << 16];
    while (in.read(buffer, sizeof(buffer)) || in.gcount() > 0) {
        out->insert(out->end(), buffer, buffer + in.gcount());
    }
    if (in.bad()) {
        std::fprintf(stderr, "read error in corpus file: %s\n", path.c_str());
        return false;
    }
    if (out->size() == before) {
        std::fprintf(stderr, "corpus file contributed no data: %s\n",
                     path.c_str());
        return false;
    }
    return true;
}

/* The selfcheck source. Several blocks at every rung and compressible
 * enough that the frame is not just stored blocks, from a fixed PRNG so a
 * failure reproduces. */
constexpr size_t kFrameSelfcheckBytes = 3u << 20;

std::vector<unsigned char> MakeFrameSelfcheckSource(size_t bytes) {
    std::vector<unsigned char> out(bytes);
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < bytes; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        /* Runs of a repeating alphabet with occasional noise: matches for
         * the decoder to execute, without collapsing to one long match. */
        out[i] = (i % 61 == 0) ? static_cast<unsigned char>(state >> 56)
                               : static_cast<unsigned char>('a' + (i / 7) % 26);
    }
    return out;
}

/* The whole sweep: the same bytes at every rung, so the only variable
 * between the report blocks is the block count. */
bool RunFrameSweep(const std::vector<unsigned char>& source,
                   const std::string& corpus_name, size_t warmup,
                   size_t runs) {
    for (const FrameRung& rung : kFrameRungs) {
        if (!RunFrameRung(source, corpus_name, rung, warmup, runs)) {
            return false;
        }
        std::printf("\n");
    }
    return true;
}

void PrintReport(const Corpus& corpus, const std::vector<double>& sorted,
                 size_t warmup, size_t runs) {
    std::vector<size_t> sizes;
    for (const auto& original : corpus.originals) {
        sizes.push_back(original.size());
    }
    std::sort(sizes.begin(), sizes.end());
    const double to_gbps = static_cast<double>(corpus.original_bytes) / 1e9;

    std::printf("## bench_lz4 report\n");
    std::printf("- decoder: CPU oracle, LZ4_decompress_safe (liblz4 %s), "
                "single thread\n",
                LZ4_versionString());
    std::printf("- host CPU: %s\n", cudec_bench::HostCpuName().c_str());
    std::printf("- CUDA device: %s\n", CudaDeviceLine().c_str());
    std::printf("- cudec: %d (the CPU rows time the liblz4 oracle baseline; "
                "the GPU rows below, when --gpu is set, time cudec's "
                "decoder)\n",
                cudec_version());
    std::printf("- corpus: %s, %zu chunks, %.2f MB original, %.2f MB "
                "compressed (ratio %.3f), %s\n",
                corpus.name.c_str(), corpus.originals.size(),
                static_cast<double>(corpus.original_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) / 1e6,
                static_cast<double>(corpus.compressed_bytes) /
                    static_cast<double>(corpus.original_bytes),
                corpus.provenance.c_str());
    std::printf("- chunk sizes: min %zu / median %zu / max %zu bytes\n",
                sizes.front(), sizes[sizes.size() / 2], sizes.back());
    std::printf("- method: %zu warmup + %zu measured runs, wall clock per "
                "whole-batch decode; the timed region is "
                "LZ4_decompress_safe only (no clears, no allocation); "
                "output byte-verified once before timing; percentiles are "
                "nearest-rank\n",
                warmup, runs);
    std::printf("- wall per run: p50 %.3f ms / p90 %.3f ms / p99 %.3f ms\n",
                Percentile(sorted, 50) * 1e3, Percentile(sorted, 90) * 1e3,
                Percentile(sorted, 99) * 1e3);
    std::printf("- decode throughput: p50 %.3f GB/s / p90 %.3f GB/s / p99 "
                "%.3f GB/s\n",
                to_gbps / Percentile(sorted, 50),
                to_gbps / Percentile(sorted, 90),
                to_gbps / Percentile(sorted, 99));
}

}  // namespace

int main(int argc, char** argv) {
    size_t runs = 30;
    size_t warmup = 3;
    bool selfcheck = false;
    bool gpu = false;
    bool gpu_stream_ctx = false;
    bool worst4b = false;
    bool longmatch = false;
    bool assetlike = false;
    bool frame = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--selfcheck") {
            selfcheck = true;
        } else if (arg == "--gpu") {
            gpu = true;
        } else if (arg == "--gpu-stream-ctx") {
            gpu_stream_ctx = true;
        } else if (arg == "--worst4b") {
            worst4b = true;
        } else if (arg == "--longmatch") {
            longmatch = true;
        } else if (arg == "--assetlike") {
            assetlike = true;
        } else if (arg == "--frame") {
            frame = true;
        } else if (arg == "--runs" && i + 1 < argc) {
            if (!ParseCount(argv[++i], 1, kMaxRuns, &runs)) {
                std::fprintf(stderr, "--runs must be in [1, %zu]\n",
                             kMaxRuns);
                return 2;
            }
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!ParseCount(argv[++i], 0, kMaxRuns, &warmup)) {
                std::fprintf(stderr, "--warmup must be in [0, %zu]\n",
                             kMaxRuns);
                return 2;
            }
        } else if (arg == "--runs" || arg == "--warmup") {
            std::fprintf(stderr, "%s needs a value\n", arg.c_str());
            return 2;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr,
                         "usage: bench_lz4 [--runs N] [--warmup N] [--gpu] "
                         "[--gpu-stream-ctx] [--worst4b] [--longmatch] "
                         "[--assetlike] [--frame] [--selfcheck] "
                         "[corpus files...]\n");
            return 2;
        } else {
            files.push_back(arg);
        }
    }
    if (selfcheck) {
        warmup = 1;
        runs = 3;
    }

    const int generated = (worst4b ? 1 : 0) + (longmatch ? 1 : 0) +
                          (assetlike ? 1 : 0);
    if (generated > 1) {
        std::fprintf(stderr, "--worst4b, --longmatch and --assetlike each "
                             "build their own corpus; pass at most one\n");
        return 2;
    }

    /* The frame sweep is a different measurement of a different entry point
     * and shares none of the block harness below, so it runs on its own and
     * returns rather than falling through. */
    if (frame) {
        if (generated != 0 || gpu || gpu_stream_ctx) {
            std::fprintf(stderr, "--frame times the frame entry point over "
                                 "its own sweep; it takes neither the "
                                 "generated block corpora nor the block "
                                 "GPU modes\n");
            return 2;
        }
        std::vector<unsigned char> source;
        std::string name;
        if (files.empty()) {
            if (!selfcheck) {
                std::fprintf(stderr, "--frame needs corpus files (the "
                                     "recorded rungs use bench/corpora/"
                                     "silesia/*); --selfcheck runs it on a "
                                     "generated source instead\n");
                return 2;
            }
            /* Big enough that every rung holds several blocks, so a rung
             * that collapsed to one block is a failure here rather than in
             * a recorded run. */
            source = MakeFrameSelfcheckSource(kFrameSelfcheckBytes);
            name = "generated (selfcheck)";
        } else {
            for (const auto& path : files) {
                if (!AppendFileBytes(path, &source)) {
                    return 1;
                }
                const size_t slash = path.find_last_of("/\\");
                name += (name.empty() ? "" : "+") +
                        path.substr(slash == std::string::npos ? 0
                                                               : slash + 1);
            }
        }
        if (source.empty()) {
            std::fprintf(stderr, "corpus is empty - nothing to benchmark\n");
            return 1;
        }
        if (!RunFrameSweep(source, name, warmup, runs)) {
            return 1;
        }
        if (selfcheck) {
            std::printf("PASS: selfcheck complete\n");
        }
        return 0;
    }

    Corpus corpus;
    if (worst4b) {
        /* The worst-case corpus is generated, not read: it carries its own
         * hand-built compressed streams, so it must not also take files. */
        if (!files.empty()) {
            std::fprintf(stderr, "--worst4b builds its own corpus; do not "
                                 "also pass corpus files\n");
            return 2;
        }
        if (!BuildWorst4bCorpus(&corpus, selfcheck ? kWorst4bSelfcheckChunks
                                                   : kWorst4bChunks)) {
            return 1;
        }
    } else if (longmatch) {
        /* Generated and hand-built like --worst4b, for the same reason. */
        if (!files.empty()) {
            std::fprintf(stderr, "--longmatch builds its own corpus; do not "
                                 "also pass corpus files\n");
            return 2;
        }
        if (!BuildLongmatchCorpus(&corpus, selfcheck ? kLongmatchSelfcheckChunks
                                                     : kLongmatchChunks)) {
            return 1;
        }
    } else if (assetlike) {
        /* Generated like the two above, though its wire is ordinary
         * compressor output rather than hand-built; it carries its own
         * compressed streams either way, so it must not also take files. */
        if (!files.empty()) {
            std::fprintf(stderr, "--assetlike builds its own corpus; do not "
                                 "also pass corpus files\n");
            return 2;
        }
        if (!BuildAssetlikeCorpus(&corpus, selfcheck
                                               ? kAssetlikeSelfcheckChunks
                                               : kAssetlikeChunks)) {
            return 1;
        }
    } else if (files.empty()) {
        corpus.name = "builtin";
        for (auto& fixture : MakeLz4BlockFixtures()) {
            corpus.originals.push_back(std::move(fixture.original));
        }
    } else {
        for (const auto& path : files) {
            if (!AppendFileChunked(path, &corpus)) {
                return 1;
            }
            const size_t slash = path.find_last_of("/\\");
            corpus.name += (corpus.name.empty() ? "" : "+") +
                           path.substr(slash == std::string::npos
                                           ? 0
                                           : slash + 1);
        }
    }
    /* An empty corpus has nothing to attest: refuse instead of emitting a
     * report over zero bytes (and indexing empty vectors). */
    if (corpus.originals.empty()) {
        std::fprintf(stderr, "corpus is empty - nothing to benchmark\n");
        return 1;
    }
    /* The timed scratch buffer is sized to kChunkBytes; enforce the
     * chunking invariant here instead of trusting fixture growth in
     * another directory to keep it (a larger chunk would otherwise hand
     * LZ4_decompress_safe an overstated capacity). */
    for (const auto& original : corpus.originals) {
        if (original.size() > kChunkBytes) {
            std::fprintf(stderr, "chunk of %zu bytes exceeds the %zu-byte "
                                 "scratch invariant\n",
                         original.size(), kChunkBytes);
            return 1;
        }
    }
    /* The worst-case and longmatch corpora already carry their hand-built
     * streams; the standard compressor would replace them (a single long
     * match, or a fixed-offset run collapsed into one match), defeating the
     * point. The asset-like corpus is compressor output, but it was
     * compressed inside its builder so its shape could be locked there.
     * Every other corpus is compressed by the oracle here. */
    if (generated == 0) {
        CompressAll(&corpus);
    }

    /* Byte-verify every chunk once, outside the timed region. */
    std::vector<unsigned char> scratch;
    for (size_t i = 0; i < corpus.compressed.size(); i++) {
        if (!OracleDecodes(corpus.compressed[i], corpus.originals[i].size(),
                           &scratch) ||
            scratch.size() != corpus.originals[i].size() ||
            std::memcmp(scratch.data(), corpus.originals[i].data(),
                        scratch.size()) != 0) {
            std::fprintf(stderr, "verification failed at chunk %zu\n", i);
            return 1;
        }
    }

    /* One pre-sized buffer for the timed loops; every chunk fits by the
     * kChunkBytes chunking invariant. */
    std::vector<unsigned char> timed_scratch(kChunkBytes);
    for (size_t i = 0; i < warmup; i++) {
        (void)DecodeAllSeconds(corpus, timed_scratch.data());
    }
    std::vector<double> times;
    for (size_t i = 0; i < runs; i++) {
        times.push_back(DecodeAllSeconds(corpus, timed_scratch.data()));
    }
    std::sort(times.begin(), times.end());

    PrintReport(corpus, times, warmup, runs);

    if (gpu) {
        std::vector<const unsigned char*> comp_ptrs(corpus.compressed.size());
        std::vector<size_t> comp_sizes(corpus.compressed.size());
        std::vector<size_t> orig_sizes(corpus.originals.size());
        for (size_t i = 0; i < corpus.compressed.size(); i++) {
            comp_ptrs[i] = corpus.compressed[i].data();
            comp_sizes[i] = corpus.compressed[i].size();
            orig_sizes[i] = corpus.originals[i].size();
        }
        cudec_gpu_result g;
        if (!cudec_bench_gpu(comp_ptrs.data(), comp_sizes.data(),
                             orig_sizes.data(), corpus.originals.size(),
                             static_cast<int>(warmup), static_cast<int>(runs),
                             &g)) {
            std::fprintf(stderr, "GPU bench failed\n");
            return 1;
        }
        std::printf("- GPU decode (device-resident, CUDA-event timed, "
                    "%d warmup + %d runs): p50 %.3f ms, %.1f GB/s\n",
                    static_cast<int>(warmup), static_cast<int>(runs),
                    g.full_ms_p50, g.full_gbps_p50);
        std::printf("- GPU parse-only ceiling (copies elided): p50 %.3f ms, "
                    "%.1f GB/s - ceilings this design AND any two-phase "
                    "phase-1 (shared parse)\n",
                    g.parse_only_ms_p50, g.parse_only_gbps_p50);
    }

    if (gpu_stream_ctx) {
        std::vector<const unsigned char*> comp_ptrs(corpus.compressed.size());
        std::vector<size_t> comp_sizes(corpus.compressed.size());
        std::vector<size_t> orig_sizes(corpus.originals.size());
        for (size_t i = 0; i < corpus.compressed.size(); i++) {
            comp_ptrs[i] = corpus.compressed[i].data();
            comp_sizes[i] = corpus.compressed[i].size();
            orig_sizes[i] = corpus.originals[i].size();
        }
        cudec_stream_ctx_result s;
        if (!cudec_bench_gpu_stream_ctx(
                comp_ptrs.data(), comp_sizes.data(), orig_sizes.data(),
                corpus.originals.size(), static_cast<int>(warmup),
                static_cast<int>(runs), &s)) {
            std::fprintf(stderr, "GPU streaming-context bench failed\n");
            return 1;
        }
        /* End-to-end throughput = decoded output bytes / wall time (H2D and,
         * for host output, D2H included). STEADY-STATE is the setup-free datum:
         * repeated decodes on one reused context whose staging is already
         * grown, so the per-call allocation the reusable context amortizes away
         * is out of the wall. COLD is the first decode on a fresh context,
         * which pays that staging grow; (cold - steady) is the amortized setup.
         * For LZ4 the steady-state wall is the compressed-H2D + decode floor -
         * see docs/BENCHMARKS.md for why input-H2D overlap does not pay for
         * this format and where the output-D2H lever would. */
        std::printf("- GPU streaming, reusable context, end-to-end (host "
                    "compressed in -> decoded out; wall clock around the whole "
                    "synchronous decode call; %d warmup + %d runs):\n",
                    static_cast<int>(warmup), static_cast<int>(runs));
        std::printf("    device out: steady-state (reused ctx) p50 %.1f ms, "
                    "%.2f GB/s ; cold (fresh ctx, first call) p50 %.1f ms, "
                    "%.2f GB/s\n",
                    s.device_steady_ms, s.device_steady_gbps, s.device_cold_ms,
                    s.device_cold_gbps);
        std::printf("    host out (readback synchronous): steady-state p50 %.1f "
                    "ms, %.2f GB/s ; cold p50 %.1f ms, %.2f GB/s\n",
                    s.host_steady_ms, s.host_steady_gbps, s.host_cold_ms,
                    s.host_cold_gbps);
        std::printf("    amortized setup removed by the reusable context "
                    "(cold - steady): device %.1f ms, host %.1f ms; %.2f MB "
                    "compressed in, %.2f MB decoded out\n",
                    s.device_cold_ms - s.device_steady_ms,
                    s.host_cold_ms - s.host_steady_ms,
                    static_cast<double>(s.compressed_bytes) / 1e6,
                    static_cast<double>(s.output_bytes) / 1e6);
    }

    if (selfcheck) {
        std::printf("PASS: selfcheck complete\n");
    }
    return 0;
}
