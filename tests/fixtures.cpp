#include "fixtures.h"

#include <lz4.h>

#include <cstdlib>
#include <cstring>

namespace {

/* Own PRNG: rand() and std:: distributions are implementation-defined and
 * would break bit-identical corpora across platforms. */
uint64_t SplitMix64(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

std::vector<unsigned char> RandomBytes(size_t size, uint64_t seed) {
    std::vector<unsigned char> out(size);
    uint64_t s = seed;
    for (size_t i = 0; i < size; i++) {
        out[i] = static_cast<unsigned char>(SplitMix64(s));
    }
    return out;
}

/* Token text with splices copied from exponentially growing distances, so
 * the compressed streams carry matches across the whole LZ4 offset range
 * (1..65535), not just the short distances a plain dictionary produces. */
std::vector<unsigned char> TextLike(size_t size, uint64_t seed) {
    static const char* const kWords[] = {
        "the",  "quick",  "brown", "fox",    "jumps", "over",
        "lazy", "dog",    "gpu",   "decode", "batch", "chunk",
        "warp", "stream", "block", "lane"};
    std::vector<unsigned char> out;
    out.reserve(size + 32);
    uint64_t s = seed;
    size_t distance = 1;
    while (out.size() < size) {
        const char* word = kWords[SplitMix64(s) % 16];
        out.insert(out.end(), word, word + std::strlen(word));
        out.push_back(' ');
        if (out.size() > distance && SplitMix64(s) % 4 == 0) {
            const size_t start = out.size() - distance;
            const size_t len = distance < 16 ? distance : 16;
            for (size_t i = 0; i < len; i++) {
                out.push_back(out[start + i]);
            }
            distance =
                distance >= 65535
                    ? 1
                    : (distance * 2 < 65535 ? distance * 2 : size_t{65535});
        }
    }
    out.resize(size);
    return out;
}

}  // namespace

std::vector<unsigned char> Lz4CompressBlock(
    const std::vector<unsigned char>& original) {
    const int bound = LZ4_compressBound(static_cast<int>(original.size()));
    std::vector<unsigned char> compressed(static_cast<size_t>(bound));
    const int written = LZ4_compress_default(
        reinterpret_cast<const char*>(original.data()),
        reinterpret_cast<char*>(compressed.data()),
        static_cast<int>(original.size()), bound);
    /* Corpus generation is infrastructure, not a test: a compressor
     * failure here would silently hollow out every downstream consumer. */
    if (written <= 0) {
        std::abort();
    }
    compressed.resize(static_cast<size_t>(written));
    return compressed;
}

std::vector<Fixture> MakeLz4BlockFixtures() {
    std::vector<Fixture> out;
    const auto add = [&out](std::string name, uint64_t seed,
                            std::vector<unsigned char> original) {
        Fixture f;
        f.name = std::move(name);
        f.seed = seed;
        f.original = std::move(original);
        f.compressed = Lz4CompressBlock(f.original);
        out.push_back(std::move(f));
    };
    add("zeros-65536", 0x01, std::vector<unsigned char>(65536, 0));
    add("random-65536", 0x11, RandomBytes(65536, 0x11));
    static const size_t kEdgeSizes[] = {1,   31,  32,   33,   255,
                                        256, 257, 4095, 4096, 65536};
    for (const size_t n : kEdgeSizes) {
        add("text-" + std::to_string(n), 0x22 + n, TextLike(n, 0x22 + n));
    }
    return out;
}

std::vector<Mutant> MutateStream(const std::vector<unsigned char>& stream,
                                 uint64_t seed) {
    std::vector<Mutant> out;
    const size_t n = stream.size();
    if (n == 0) {
        /* Nothing to mutate; the flip loop's % n below must never run.
         * Callers assert non-empty mutant lists, so this stays loud. */
        return out;
    }
    const auto add = [&out](std::string description,
                            std::vector<unsigned char> s) {
        out.push_back(Mutant{std::move(description), std::move(s)});
    };
    /* Truncations at fixed fractions: a stream ending mid-sequence. */
    for (const size_t quarters : {size_t{1}, size_t{2}, size_t{3}}) {
        const size_t keep = n * quarters / 4;
        if (keep < n) {
            add("truncate-to-" + std::to_string(keep),
                {stream.begin(), stream.begin() + static_cast<long>(keep)});
        }
    }
    /* Shave 1..8 bytes off either end. */
    for (size_t k = 1; k <= 8 && k < n; k++) {
        add("drop-" + std::to_string(k) + "-tail",
            {stream.begin(), stream.end() - static_cast<long>(k)});
        add("drop-" + std::to_string(k) + "-head",
            {stream.begin() + static_cast<long>(k), stream.end()});
    }
    /* Seeded single-bit flips. A flipped stream is NOT necessarily invalid
     * (the oracle may accept it) - consumers must ask the oracle, never
     * assume mutated means rejected. */
    uint64_t s = seed;
    for (int i = 0; i < 8; i++) {
        std::vector<unsigned char> flipped = stream;
        const size_t offset = SplitMix64(s) % n;
        flipped[offset] ^=
            static_cast<unsigned char>(1u << (SplitMix64(s) % 8));
        add("flip-bit-at-" + std::to_string(offset), std::move(flipped));
    }
    return out;
}

bool OracleDecodes(const std::vector<unsigned char>& stream,
                   size_t original_size, std::vector<unsigned char>* decoded) {
    if (stream.empty()) {
        return false; /* never hand liblz4 a null source pointer */
    }
    decoded->assign(original_size, 0);
    const int written = LZ4_decompress_safe(
        reinterpret_cast<const char*>(stream.data()),
        reinterpret_cast<char*>(decoded->data()),
        static_cast<int>(stream.size()), static_cast<int>(original_size));
    if (written < 0) {
        return false;
    }
    decoded->resize(static_cast<size_t>(written));
    return true;
}
