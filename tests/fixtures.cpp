#include "fixtures.h"

#include <lz4.h>
#include <snappy.h>

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

bool SnappyOracleDecodes(const std::vector<unsigned char>& stream,
                         std::vector<unsigned char>* decoded) {
    decoded->clear();
    if (stream.empty()) {
        return false; /* never hand snappy a null source pointer */
    }
    const char* src = reinterpret_cast<const char*>(stream.data());
    size_t declared = 0;
    if (!snappy::GetUncompressedLength(src, stream.size(), &declared)) {
        return false;
    }
    if (declared > kSnappyOracleMaxOutput) {
        return false; /* the wrapper's bound, documented in fixtures.h */
    }
    decoded->assign(declared, 0);
    if (!snappy::RawUncompress(src, stream.size(),
                               reinterpret_cast<char*>(decoded->data()))) {
        /* A rejected stream produces no output: a caller that reads the
         * buffer after a false verdict must find nothing to mistake for one. */
        decoded->clear();
        return false;
    }
    return true;
}

bool SnappyOracleAccepts(const std::vector<unsigned char>& stream) {
    if (stream.empty()) {
        return false;
    }
    return snappy::IsValidCompressedBuffer(
        reinterpret_cast<const char*>(stream.data()), stream.size());
}

namespace {

/* Where one Snappy element begins and how long it is. The walk exists to place
 * mutations on the reference's branches; it is not a decoder and never decides
 * whether a stream is valid. */
struct Element {
    size_t offset; /* of the tag byte */
    size_t size;   /* tag plus its trailing bytes and any literal payload */
    unsigned char tag;
};

/* The varint preamble's length in bytes, or 0 if it does not terminate inside
 * the stream. Five groups at most, the bound Varint::Parse32WithLimit stops at
 * (snappy-stubs-internal.h:455-475). */
size_t SnappyPreambleSize(const std::vector<unsigned char>& stream) {
    for (size_t i = 0; i < 5 && i < stream.size(); i++) {
        if ((stream[i] & 0x80u) == 0) {
            return i + 1;
        }
    }
    return 0;
}

/* Element boundaries, in stream order, stopping at the first byte sequence the
 * walk cannot account for. Tag encoding: snappy.cc:1613-1631 for literals,
 * snappy.cc:1650-1663 for the three copy forms. */
std::vector<Element> SnappyElements(const std::vector<unsigned char>& stream) {
    std::vector<Element> out;
    size_t at = SnappyPreambleSize(stream);
    if (at == 0) {
        return out;
    }
    while (at < stream.size()) {
        const unsigned char tag = stream[at];
        size_t size = 0;
        if ((tag & 3u) == 0u) {
            const unsigned length_class = tag >> 2;
            if (length_class < 60u) {
                size = 1 + (length_class + 1u);
            } else {
                const size_t extra = length_class - 59u;
                if (at + 1 + extra > stream.size()) {
                    break;
                }
                size_t length = 0;
                for (size_t k = 0; k < extra; k++) {
                    length |= static_cast<size_t>(stream[at + 1 + k]) << (8 * k);
                }
                size = 1 + extra + (length + 1);
            }
        } else if ((tag & 3u) == 1u) {
            size = 2;
        } else if ((tag & 3u) == 2u) {
            size = 3;
        } else {
            size = 5;
        }
        if (size == 0 || at + size > stream.size()) {
            break;
        }
        out.push_back(Element{at, size, tag});
        at += size;
    }
    return out;
}

/* The declared uncompressed length as a 5-byte varint, which is the longest
 * form the parser accepts and therefore the one a length mutation should use:
 * it changes the number without changing where the elements start. */
std::vector<unsigned char> SnappyWithDeclaredLength(
    const std::vector<unsigned char>& stream, uint32_t declared) {
    const size_t preamble = SnappyPreambleSize(stream);
    std::vector<unsigned char> out;
    uint32_t v = declared;
    for (int i = 0; i < 4; i++) {
        out.push_back(static_cast<unsigned char>((v & 0x7fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<unsigned char>(v & 0x0fu));
    out.insert(out.end(), stream.begin() + static_cast<long>(preamble),
               stream.end());
    return out;
}

}  // namespace

std::vector<Fixture> MakeSnappyFixtures() {
    std::vector<Fixture> out;
    const auto add = [&out](std::string name, uint64_t seed,
                            std::vector<unsigned char> original) {
        Fixture f;
        f.name = std::move(name);
        f.seed = seed;
        f.original = std::move(original);
        f.compressed = SnappyCompressBlock(f.original);
        out.push_back(std::move(f));
    };
    /* One long run: every copy element the compressor emits sits at the
     * 64-byte cap, which is the shape the length field's upper end takes. */
    add("zeros-65536", 0x41, std::vector<unsigned char>(65536, 0));
    /* Incompressible, so the stream is literals only, including the long
     * length classes that carry their own length bytes. */
    add("random-65536", 0x51, RandomBytes(65536, 0x51));
    /* Two blocks: the compressor splits at 65536 and starts a fresh element
     * stream, so a stream carrying that seam is in the corpus. */
    add("random-70000", 0x52, RandomBytes(70000, 0x52));
    add("zeros-70000", 0x42, std::vector<unsigned char>(70000, 0));
    /* Sizes across the literal-length class break at 60 and the small-size
     * edges, where the preamble is one byte and an element may be the whole
     * stream. */
    static const size_t kEdgeSizes[] = {1,   2,   59,   60,   61,  62,
                                        63,  64,  65,   127,  128, 255,
                                        256, 257, 4095, 4096, 65535};
    for (const size_t n : kEdgeSizes) {
        add("text-" + std::to_string(n), 0x62 + n, TextLike(n, 0x62 + n));
    }
    return out;
}

std::vector<Mutant> MutateSnappyStream(const std::vector<unsigned char>& stream,
                                       uint64_t seed) {
    /* The format-agnostic ladder first: truncations at fixed fractions, 1..8
     * bytes off either end, and seeded bit flips. */
    std::vector<Mutant> out = MutateStream(stream, seed);
    const size_t n = stream.size();
    if (n == 0) {
        return out;
    }
    const auto add = [&out](std::string description,
                            std::vector<unsigned char> s) {
        out.push_back(Mutant{std::move(description), std::move(s)});
    };
    const size_t preamble = SnappyPreambleSize(stream);

    /* The preamble, which is its own branch: a length that terminates, one
     * that does not, and one the parser must refuse for being too long. */
    if (preamble > 0) {
        std::vector<unsigned char> continued = stream;
        continued[preamble - 1] |= 0x80u;
        add("preamble-continuation-bit-set", std::move(continued));

        size_t declared = 0;
        for (size_t k = 0; k < preamble; k++) {
            declared |= static_cast<size_t>(stream[k] & 0x7fu) << (7 * k);
        }
        add("preamble-declares-0", SnappyWithDeclaredLength(stream, 0));
        add("preamble-declares-1", SnappyWithDeclaredLength(stream, 1));
        add("preamble-declares-one-less",
            SnappyWithDeclaredLength(
                stream, static_cast<uint32_t>(declared > 0 ? declared - 1 : 0)));
        add("preamble-declares-one-more",
            SnappyWithDeclaredLength(stream,
                                     static_cast<uint32_t>(declared + 1)));
        /* The upper end of what a 32-bit length can say. This one is refused
         * by the decode wrapper's own bound before snappy sees it, so what it
         * exercises is that bound; SnappyOracleAccepts still routes it to the
         * reference, which has no such limit. */
        add("preamble-declares-max", SnappyWithDeclaredLength(stream,
                                                              0xffffffffu));
    }

    /* One mutation per element branch, on the first elements and the last: the
     * first are where a wrong bound shows immediately, the last is where a
     * length that overruns the declared output has nothing after it to
     * absorb the error. */
    const std::vector<Element> elements = SnappyElements(stream);
    std::vector<size_t> targets;
    for (size_t i = 0; i < elements.size() && i < 3; i++) {
        targets.push_back(i);
    }
    if (elements.size() > 3) {
        targets.push_back(elements.size() - 1);
    }
    for (const size_t index : targets) {
        const Element& e = elements[index];
        const std::string at = "element-" + std::to_string(index);
        /* Each of the four tag classes in turn, the length bits left alone:
         * a literal read as a copy takes its offset from bytes that were
         * payload, and a copy read as a literal consumes payload that was a
         * neighbouring element. */
        for (unsigned kind = 0; kind < 4; kind++) {
            if ((e.tag & 3u) == kind) {
                continue;
            }
            std::vector<unsigned char> retagged = stream;
            retagged[e.offset] =
                static_cast<unsigned char>((e.tag & ~3u) | kind);
            add(at + "-tag-class-" + std::to_string(kind),
                std::move(retagged));
        }
        /* The length field, one step in each direction: for a copy this is
         * the length AppendFromSelf bounds against the output; for a short
         * literal it is the payload the element claims. */
        for (const int delta : {-1, 1}) {
            const unsigned high = e.tag >> 2;
            if (delta < 0 && high == 0u) {
                continue;
            }
            if (delta > 0 && high == 63u) {
                continue;
            }
            const unsigned moved = delta < 0 ? high - 1u : high + 1u;
            std::vector<unsigned char> relength = stream;
            relength[e.offset] =
                static_cast<unsigned char>((moved << 2) | (e.tag & 3u));
            add(at + "-length-bits-" + (delta < 0 ? "minus-1" : "plus-1"),
                std::move(relength));
        }
        /* The offset a copy element carries, which snappy refuses at 0 and
         * beyond what it has already produced (snappy.cc:2200-2212). Both
         * bounds, written into whichever width this tag has. */
        const unsigned kind = e.tag & 3u;
        if (kind != 0u) {
            const size_t offset_bytes = kind == 1u ? 1 : (kind == 2u ? 2 : 4);
            for (const unsigned long long value : {0ull, 0xffffffffull}) {
                std::vector<unsigned char> reoffset = stream;
                for (size_t k = 0; k < offset_bytes; k++) {
                    reoffset[e.offset + 1 + k] =
                        static_cast<unsigned char>((value >> (8 * k)) & 0xffu);
                }
                if (kind == 1u) {
                    /* The 1-byte form keeps three offset bits in the tag; a
                     * zero offset needs those cleared too. */
                    reoffset[e.offset] = static_cast<unsigned char>(
                        (e.tag & 0x1fu) |
                        (value == 0ull ? 0u : 0xe0u));
                }
                add(at + "-copy-offset-" + (value == 0ull ? "0" : "max"),
                    std::move(reoffset));
            }
        }
    }
    /* A trailing byte, the mutation the reference refuses on its eof() half
     * rather than on any output count (snappy.cc:1789). */
    std::vector<unsigned char> trailing = stream;
    trailing.push_back(0x00);
    add("trailing-byte", std::move(trailing));
    return out;
}

std::vector<unsigned char> SnappyCompressBlock(
    const std::vector<unsigned char>& original) {
    std::vector<unsigned char> compressed(
        snappy::MaxCompressedLength(original.size()));
    size_t written = 0;
    snappy::RawCompress(reinterpret_cast<const char*>(original.data()),
                        original.size(),
                        reinterpret_cast<char*>(compressed.data()), &written);
    /* Corpus generation is infrastructure, not a test, exactly as in
     * Lz4CompressBlock: a compressor that wrote more than the bound it
     * promised has already overrun this buffer, so there is nothing left to
     * report from. */
    if (written == 0 || written > compressed.size()) {
        std::abort();
    }
    compressed.resize(written);
    return compressed;
}
