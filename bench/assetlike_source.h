/* The game-asset-like source block (issue #139), as one generator two
 * benchmark harnesses read.
 *
 * It models the payload of a shipped asset package - block-compressed
 * texture, interleaved geometry, streamed audio - and it is the one regime
 * none of the fetched corpora reaches: most of the block is incompressible,
 * so a decode over it is dominated by literal transfer rather than by
 * sequence parsing or match copying.
 *
 * It is a MODEL of the workload, not the workload. A synthetic mixture is not
 * a measurement on real game data, and no number taken here may be quoted as
 * one; docs/BENCHMARK-METHODOLOGY.md carries the same sentence beside the
 * corpus entry. The decision on issue #139 chose the generator over a
 * hash-pinned asset pack deliberately: no network, no mirror that can rot, no
 * licence review on a third-party package, and CI stays offline. A vetted
 * pack may later join the set beside this generator; it does not replace it.
 *
 * WHY IT IS A HEADER RATHER THAN A COPY. The block's bytes are what a
 * recorded number is attested against, so two harnesses generating "the
 * asset-like block" from two sources would be two corpora under one name, and
 * a drift between them would move a number with nothing to notice it. The
 * shape LOCKS stay in the harnesses: what a block has to look like on the
 * wire is a per-format question, and only the source bytes are shared.
 *
 * Bench-only. Nothing here is compiled into the library. */
#ifndef CUDEC_BENCH_ASSETLIKE_SOURCE_H
#define CUDEC_BENCH_ASSETLIKE_SOURCE_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudec_bench {

/* The block is one 64 KiB unit, which is the LZ4/Snappy chunk and the
 * GDeflate page alike. Each consumer static_asserts its own unit against this
 * one, so a harness whose granularity moved fails to compile rather than
 * silently benchmarking a differently-tiled block. */
constexpr size_t kAssetlikeBlockBytes = 65536;

/* The three regions of the block, in bytes, tiling one chunk exactly. The
 * proportions follow a shipped package: block-compressed texture dominates,
 * geometry follows, streamed audio fills the rest. */
constexpr size_t kAssetTextureBytes = 32768;
constexpr size_t kAssetVertexBytes = 14336;
constexpr size_t kAssetIndexBytes = 2048;
constexpr size_t kAssetAudioBytes = 16384;
static_assert(kAssetTextureBytes + kAssetVertexBytes + kAssetIndexBytes +
                      kAssetAudioBytes ==
                  kAssetlikeBlockBytes,
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
    original->reserve(kAssetlikeBlockBytes);
    BuildTextureRegion(original, &rng);
    BuildVertexRegion(original, &rng);
    BuildIndexRegion(original);
    BuildAudioRegion(original, &rng);
}

}  // namespace cudec_bench

#endif /* CUDEC_BENCH_ASSETLIKE_SOURCE_H */
