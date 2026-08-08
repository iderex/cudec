/* Implementation of the M5 Zstd corpus and its frame walker. See
 * zstd_corpus.h for what this is for and why the source bytes are built
 * rather than fetched. */
#include "zstd_corpus.h"

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include <cstddef>
#include <cstring>

namespace {

using Bytes = std::vector<unsigned char>;

/* ---- Source constructions ---------------------------------------------
 *
 * Every source is a pure function of its length, so two generation passes
 * agree byte for byte without a seeded RNG anywhere. The shapes are chosen
 * for what the compressor does with them, not for realism: a decode surface
 * the compressor never emits cannot be covered by a corpus of realistic
 * data, which is the whole reason the forced-mode families exist. */

/* Splitmix64, inlined so the corpus does not depend on a library RNG whose
 * output could move under us. Used only where incompressible bytes are
 * wanted. */
uint64_t Mix(uint64_t* state) {
    *state += 0x9e3779b97f4a7c15ull;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* High entropy: no match longer than chance, so the compressor cannot beat
 * the input and emits a Raw block. */
Bytes IncompressibleSource(size_t size) {
    Bytes out(size);
    uint64_t state = 0x5eed1234ull;
    for (size_t i = 0; i < size; i++) {
        out[i] = static_cast<unsigned char>(Mix(&state) & 0xff);
    }
    return out;
}

/* Text-like: a small word list joined by a walk that revisits words, so the
 * match distribution is wide enough for FSE-compressed tables and the
 * literal distribution is skewed enough for a Huffman tree. */
Bytes TextSource(size_t size) {
    static const char* const kWords[] = {
        "the ",     "decoder ", "rejects ", "a ",        "malformed ",
        "stream ",  "and ",     "never ",   "guesses ",  "at ",
        "the ",     "output ",  "bytes ",   "it ",       "cannot ",
        "produce ", "from ",    "hostile ", "input ",    "safely "};
    const size_t word_count = sizeof(kWords) / sizeof(kWords[0]);
    Bytes out;
    out.reserve(size + 16);
    uint64_t state = 0xc0ffeeull;
    while (out.size() < size) {
        const char* word = kWords[Mix(&state) % word_count];
        out.insert(out.end(), word, word + std::strlen(word));
    }
    out.resize(size);
    return out;
}

/* A fixed-period pattern: every match the compressor finds has the same
 * offset and the same length, which is the one-distinct-symbol case
 * Symbol_Compression_Mode RLE exists for. */
Bytes PeriodicSource(size_t size, size_t period) {
    Bytes out(size);
    for (size_t i = 0; i < size; i++) {
        out[i] = static_cast<unsigned char>('a' + (i % period));
    }
    return out;
}

/* Runs of 64 identical bytes, each run a different byte. Every sequence is
 * then literals_length 1, match_length 63 and offset 1 - and offset 1 is the
 * initial repcode, so even the first sequence encodes as rep1 and the offset
 * alphabet holds exactly one symbol. One distinct symbol per field is what
 * Symbol_Compression_Mode RLE is selected on. */
Bytes RleSequenceSource(size_t size) {
    Bytes out;
    out.reserve(size + 64);
    /* 7 is odd, so i*7 walks all 256 values before repeating: every run holds
     * a byte no other run holds. A repeated run byte would give the match
     * finder a second, longer candidate at a different offset, and the field
     * that stops being one symbol is the one this fixture is for. */
    for (unsigned i = 0; out.size() < size && i < 256; i++) {
        out.insert(out.end(), 64, static_cast<unsigned char>((i * 7u + 1u) & 0xffu));
    }
    out.resize(size);
    return out;
}

/* Compressible bytes first, then a pure run. With the block size cut down by
 * a small window the run lands in blocks of its own, and a block that is not
 * the frame's first can be emitted as RLE - zstd refuses to make the first
 * one RLE whatever it holds. */
Bytes TextThenRunSource(size_t size) {
    const size_t head = size / 4;
    Bytes out = TextSource(head);
    out.insert(out.end(), size - head, 'q');
    return out;
}

/* ---- The frame walker -------------------------------------------------- */

struct Reader {
    const unsigned char* data;
    size_t size;
    size_t pos = 0;

    bool Take(size_t count, const unsigned char** at) {
        if (count > size - pos) {
            return false;
        }
        *at = data + pos;
        pos += count;
        return true;
    }
    bool Skip(size_t count) {
        if (count > size - pos) {
            return false;
        }
        pos += count;
        return true;
    }
};

/* Number_Of_Sequences, RFC 8878 section 3.1.1.3.2.1: one byte below 128, two
 * bytes up to 254, three bytes at 255. */
bool ReadSequenceCount(Reader* r, unsigned* out) {
    const unsigned char* first = nullptr;
    if (!r->Take(1, &first)) {
        return false;
    }
    if (*first < 128) {
        *out = *first;
        return true;
    }
    const unsigned char* more = nullptr;
    if (*first < 255) {
        if (!r->Take(1, &more)) {
            return false;
        }
        *out = ((static_cast<unsigned>(*first) - 128u) << 8) + *more;
        return true;
    }
    if (!r->Take(2, &more)) {
        return false;
    }
    *out = more[0] + (static_cast<unsigned>(more[1]) << 8) + 0x7f00u;
    return true;
}

/* Literals_Section_Header, RFC 8878 section 3.1.1.3.1.1. Reports the type,
 * the stream count and how many bytes of the block the section occupies in
 * total, which is what lets the walk reach the sequences section without
 * decoding anything. */
bool ReadLiteralsSection(Reader* r, ZstdBlockShape* shape) {
    const unsigned char* head = nullptr;
    if (!r->Take(1, &head)) {
        return false;
    }
    const unsigned type = *head & 0x3u;
    const unsigned size_format = (*head >> 2) & 0x3u;
    shape->literals_type = type;
    shape->literals_streams = 0;

    if (type == kZstdLiteralsRaw || type == kZstdLiteralsRle) {
        size_t regenerated = 0;
        if (size_format == 0 || size_format == 2) {
            regenerated = *head >> 3;
        } else if (size_format == 1) {
            const unsigned char* rest = nullptr;
            if (!r->Take(1, &rest)) {
                return false;
            }
            regenerated = (*head >> 4) | (static_cast<size_t>(rest[0]) << 4);
        } else {
            const unsigned char* rest = nullptr;
            if (!r->Take(2, &rest)) {
                return false;
            }
            regenerated = (*head >> 4) | (static_cast<size_t>(rest[0]) << 4) |
                          (static_cast<size_t>(rest[1]) << 12);
        }
        /* An RLE literals section stores exactly one byte whatever it
         * regenerates; a Raw one stores every byte it regenerates. */
        return r->Skip(type == kZstdLiteralsRle ? 1u : regenerated);
    }

    /* Compressed and Treeless carry a compressed size as well, and the
     * stream count is the Size_Format: 00 is the only single-stream form. */
    size_t compressed = 0;
    if (size_format == 0 || size_format == 1) {
        const unsigned char* rest = nullptr;
        if (!r->Take(2, &rest)) {
            return false;
        }
        const uint32_t packed = (static_cast<uint32_t>(*head) >> 4) |
                                (static_cast<uint32_t>(rest[0]) << 4) |
                                (static_cast<uint32_t>(rest[1]) << 12);
        compressed = (packed >> 10) & 0x3ffu;
    } else if (size_format == 2) {
        const unsigned char* rest = nullptr;
        if (!r->Take(3, &rest)) {
            return false;
        }
        const uint32_t packed = (static_cast<uint32_t>(*head) >> 4) |
                                (static_cast<uint32_t>(rest[0]) << 4) |
                                (static_cast<uint32_t>(rest[1]) << 12) |
                                (static_cast<uint32_t>(rest[2]) << 20);
        compressed = (packed >> 14) & 0x3fffu;
    } else {
        const unsigned char* rest = nullptr;
        if (!r->Take(4, &rest)) {
            return false;
        }
        const uint64_t packed = (static_cast<uint64_t>(*head) >> 4) |
                                (static_cast<uint64_t>(rest[0]) << 4) |
                                (static_cast<uint64_t>(rest[1]) << 12) |
                                (static_cast<uint64_t>(rest[2]) << 20) |
                                (static_cast<uint64_t>(rest[3]) << 28);
        compressed = static_cast<size_t>((packed >> 18) & 0x3ffffu);
    }
    shape->literals_streams = size_format == 0 ? 1u : 4u;
    return r->Skip(compressed);
}

}  // namespace

bool ParseZstdFrameShape(const Bytes& frame, ZstdFrameShape* out,
                         std::string* why) {
    *out = ZstdFrameShape();
    Reader r{frame.data(), frame.size(), 0};
    const unsigned char* magic = nullptr;
    if (!r.Take(4, &magic) || magic[0] != 0x28 || magic[1] != 0xb5 ||
        magic[2] != 0x2f || magic[3] != 0xfd) {
        *why = "not a zstd frame magic";
        return false;
    }
    const unsigned char* descriptor = nullptr;
    if (!r.Take(1, &descriptor)) {
        *why = "frame header descriptor truncated";
        return false;
    }
    const unsigned fcs_flag = (*descriptor >> 6) & 0x3u;
    out->single_segment = ((*descriptor >> 5) & 1u) != 0;
    out->checksum_present = ((*descriptor >> 2) & 1u) != 0;
    const unsigned did_flag = *descriptor & 0x3u;

    if (!out->single_segment && !r.Skip(1)) {
        *why = "window descriptor truncated";
        return false;
    }
    static const size_t kDidBytes[4] = {0, 1, 2, 4};
    if (!r.Skip(kDidBytes[did_flag])) {
        *why = "dictionary id truncated";
        return false;
    }
    size_t fcs_bytes = 0;
    if (fcs_flag == 0) {
        fcs_bytes = out->single_segment ? 1u : 0u;
    } else if (fcs_flag == 1) {
        fcs_bytes = 2;
    } else if (fcs_flag == 2) {
        fcs_bytes = 4;
    } else {
        fcs_bytes = 8;
    }
    out->content_size_present = fcs_bytes != 0;
    if (!r.Skip(fcs_bytes)) {
        *why = "frame content size truncated";
        return false;
    }

    for (;;) {
        const unsigned char* header = nullptr;
        if (!r.Take(3, &header)) {
            *why = "block header truncated";
            return false;
        }
        const uint32_t value = static_cast<uint32_t>(header[0]) |
                               (static_cast<uint32_t>(header[1]) << 8) |
                               (static_cast<uint32_t>(header[2]) << 16);
        ZstdBlockShape block;
        block.last = (value & 1u) != 0;
        block.block_type = (value >> 1) & 0x3u;
        const size_t block_size = (value >> 3);
        if (block.block_type == 3) {
            *why = "reserved block type";
            return false;
        }
        if (block.block_type == kZstdBlockCompressed) {
            const size_t block_start = r.pos;
            if (!ReadLiteralsSection(&r, &block)) {
                *why = "literals section truncated";
                return false;
            }
            if (!ReadSequenceCount(&r, &block.sequence_count)) {
                *why = "sequence count truncated";
                return false;
            }
            if (block.sequence_count > 0) {
                const unsigned char* modes = nullptr;
                if (!r.Take(1, &modes)) {
                    *why = "symbol compression modes truncated";
                    return false;
                }
                block.ll_mode = (*modes >> 6) & 0x3u;
                block.of_mode = (*modes >> 4) & 0x3u;
                block.ml_mode = (*modes >> 2) & 0x3u;
                if ((*modes & 0x3u) != 0) {
                    *why = "reserved bits set in symbol compression modes";
                    return false;
                }
                block.tables_offset = r.pos;
            }
            /* The walk stops at the mode byte, so it resumes from the block
             * size the header declared rather than from where it stopped. */
            if (block_start + block_size > frame.size()) {
                *why = "compressed block runs past the frame";
                return false;
            }
            block.block_end = block_start + block_size;
            r.pos = block_start + block_size;
        } else if (!r.Skip(block.block_type == kZstdBlockRle ? 1u
                                                             : block_size)) {
            *why = "block payload truncated";
            return false;
        }
        out->blocks.push_back(block);
        if (block.last) {
            break;
        }
        if (out->blocks.size() > 4096) {
            *why = "more blocks than this corpus ever generates";
            return false;
        }
    }
    if (out->checksum_present && !r.Skip(4)) {
        *why = "content checksum truncated";
        return false;
    }
    if (r.pos != frame.size()) {
        *why = "trailing bytes after the frame";
        return false;
    }
    return true;
}

bool ZstdShapeSatisfies(const ZstdFrameShape& shape, const ZstdDemand& demand,
                        std::string* why) {
    if (demand.single_segment >= 0 &&
        shape.single_segment != (demand.single_segment != 0)) {
        *why = "single-segment flag";
        return false;
    }
    if (demand.content_size >= 0 &&
        shape.content_size_present != (demand.content_size != 0)) {
        *why = "content size presence";
        return false;
    }
    if (demand.checksum >= 0 &&
        shape.checksum_present != (demand.checksum != 0)) {
        *why = "checksum presence";
        return false;
    }
    if (demand.min_blocks >= 0 &&
        shape.blocks.size() < static_cast<size_t>(demand.min_blocks)) {
        *why = "block count";
        return false;
    }
    const bool wants_block =
        demand.block_type >= 0 || demand.literals_type >= 0 ||
        demand.literals_streams >= 0 || demand.ll_mode >= 0 ||
        demand.of_mode >= 0 || demand.ml_mode >= 0;
    if (!wants_block) {
        return true;
    }
    for (const auto& block : shape.blocks) {
        if (demand.block_type >= 0 &&
            block.block_type != static_cast<unsigned>(demand.block_type)) {
            continue;
        }
        if (demand.literals_type >= 0 &&
            (block.block_type != kZstdBlockCompressed ||
             block.literals_type !=
                 static_cast<unsigned>(demand.literals_type))) {
            continue;
        }
        if (demand.literals_streams >= 0 &&
            block.literals_streams !=
                static_cast<unsigned>(demand.literals_streams)) {
            continue;
        }
        if (demand.ll_mode >= 0 &&
            (block.sequence_count == 0 ||
             block.ll_mode != static_cast<unsigned>(demand.ll_mode))) {
            continue;
        }
        if (demand.of_mode >= 0 &&
            (block.sequence_count == 0 ||
             block.of_mode != static_cast<unsigned>(demand.of_mode))) {
            continue;
        }
        if (demand.ml_mode >= 0 &&
            (block.sequence_count == 0 ||
             block.ml_mode != static_cast<unsigned>(demand.ml_mode))) {
            continue;
        }
        return true;
    }
    *why = "no block carries the demanded mode combination";
    return false;
}

namespace {

/* One knob setting. Kept as data so a family reads as what it asked the
 * compressor for. */
struct Param {
    ZSTD_cParameter id;
    int value;
};

struct FamilySpec {
    const char* name;
    const char* family;
    Bytes (*source)(size_t);
    size_t source_size;
    int level;
    std::vector<Param> params;
    ZstdDemand demand;
};

Bytes SourceText(size_t n) { return TextSource(n); }
Bytes SourceIncompressible(size_t n) { return IncompressibleSource(n); }
Bytes SourcePeriodic(size_t n) { return PeriodicSource(n, 8); }
Bytes SourceRleSequences(size_t n) { return RleSequenceSource(n); }
Bytes SourceTextThenRun(size_t n) { return TextThenRunSource(n); }

bool Compress(const Bytes& source, int level, const std::vector<Param>& params,
              Bytes* out) {
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (cctx == nullptr) {
        return false;
    }
    bool ok = !ZSTD_isError(
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level));
    for (const auto& p : params) {
        if (!ok) {
            break;
        }
        ok = !ZSTD_isError(ZSTD_CCtx_setParameter(cctx, p.id, p.value));
    }
    if (ok) {
        out->assign(ZSTD_compressBound(source.size()), 0);
        const size_t written = ZSTD_compress2(cctx, out->data(), out->size(),
                                              source.data(), source.size());
        ok = !ZSTD_isError(written);
        if (ok) {
            out->resize(written);
        }
    }
    ZSTD_freeCCtx(cctx);
    return ok;
}

}  // namespace

bool ZstdOracleDecodes(const Bytes& frame, Bytes* out) {
    const unsigned long long declared =
        ZSTD_getFrameContentSize(frame.data(), frame.size());
    /* A frame without a content size still has to be decodable here, so the
     * bound comes from the corpus rather than from the frame: nothing this
     * generator emits exceeds it, and a hostile frame is not this function's
     * job. */
    const size_t bound =
        (declared == ZSTD_CONTENTSIZE_UNKNOWN ||
         declared == ZSTD_CONTENTSIZE_ERROR)
            ? size_t{1} << 22
            : static_cast<size_t>(declared);
    out->assign(bound, 0);
    const size_t written =
        ZSTD_decompress(out->data(), out->size(), frame.data(), frame.size());
    if (ZSTD_isError(written)) {
        out->clear();
        return false;
    }
    out->resize(written);
    return true;
}

std::vector<std::vector<unsigned char>> MakeZstdBatchFrames(
    const Bytes& source, size_t chunk_size, int level) {
    std::vector<Bytes> frames;
    if (chunk_size == 0) {
        return frames;
    }
    for (size_t at = 0; at < source.size(); at += chunk_size) {
        const size_t take =
            at + chunk_size <= source.size() ? chunk_size : source.size() - at;
        /* ptrdiff_t rather than long: long is 32-bit on LLP64 hosts, and this
         * function is meant to take a fetched corpus of any size. */
        const Bytes chunk(
            source.begin() + static_cast<std::ptrdiff_t>(at),
            source.begin() + static_cast<std::ptrdiff_t>(at + take));
        Bytes frame;
        if (!Compress(chunk, level, {}, &frame)) {
            return std::vector<Bytes>();
        }
        frames.push_back(frame);
    }
    return frames;
}

std::vector<ZstdFixture> MakeZstdFixtures() {
    /* Every row states what it asks the compressor for and what must come
     * back. docs/ZSTD-CORPUS.md is the same table in prose. */
    static const FamilySpec kSpecs[] = {
        /* Frame envelope. */
        {"envelope-content-size-and-checksum", "envelope", SourceText, 8192, 3,
         {{ZSTD_c_contentSizeFlag, 1}, {ZSTD_c_checksumFlag, 1}},
         [] { ZstdDemand d; d.content_size = 1; d.checksum = 1; return d; }()},
        {"envelope-no-content-size-no-checksum", "envelope", SourceText, 8192,
         3, {{ZSTD_c_contentSizeFlag, 0}, {ZSTD_c_checksumFlag, 0}},
         [] { ZstdDemand d; d.content_size = 0; d.checksum = 0; return d; }()},
        {"envelope-single-segment", "envelope", SourceText, 1024, 3,
         {{ZSTD_c_contentSizeFlag, 1}},
         [] { ZstdDemand d; d.single_segment = 1; return d; }()},
        {"envelope-windowed", "envelope", SourceText, 262144, 3,
         {{ZSTD_c_windowLog, 10}, {ZSTD_c_contentSizeFlag, 1}},
         [] { ZstdDemand d; d.single_segment = 0; d.min_blocks = 2; return d; }()},

        /* Block types. */
        {"block-raw", "block", SourceIncompressible, 4096, 3, {},
         [] { ZstdDemand d; d.block_type = kZstdBlockRaw; return d; }()},
        {"block-rle", "block", SourceTextThenRun, 8192, 3,
         {{ZSTD_c_windowLog, 10}},
         [] { ZstdDemand d; d.block_type = kZstdBlockRle; return d; }()},
        {"block-compressed", "block", SourceText, 8192, 3, {},
         [] { ZstdDemand d; d.block_type = kZstdBlockCompressed; return d; }()},

        /* Literals section. */
        {"literals-raw", "literals", SourcePeriodic, 8192, 3,
         {{ZSTD_c_literalCompressionMode, ZSTD_ps_disable}},
         [] { ZstdDemand d; d.literals_type = kZstdLiteralsRaw; return d; }()},
        {"literals-compressed-four-stream", "literals", SourceText, 65536, 3,
         {},
         [] { ZstdDemand d; d.literals_type = kZstdLiteralsCompressed;
              d.literals_streams = 4; return d; }()},
        {"literals-compressed-one-stream", "literals", SourceText, 2048, 3,
         {{ZSTD_c_targetCBlockSize, 340}},
         [] { ZstdDemand d; d.literals_type = kZstdLiteralsCompressed;
              d.literals_streams = 1; return d; }()},
        {"literals-treeless", "literals", SourceText, 262144, 3,
         {{ZSTD_c_targetCBlockSize, 1300}},
         [] { ZstdDemand d; d.literals_type = kZstdLiteralsTreeless; return d; }()},

        /* Sequence table modes, one family per mode per field position. */
        {"tables-basic", "tables", SourcePeriodic, 32768, 3, {},
         [] { ZstdDemand d; d.ll_mode = kZstdTableBasic;
              d.of_mode = kZstdTableBasic; d.ml_mode = kZstdTableBasic;
              return d; }()},
        /* Offsets and match lengths only. On this source the compressor
         * emitted a compressed literal-lengths table beside the two RLE
         * ones; that is the measurement, and no cause is claimed for it.
         * The literal-lengths RLE cell is covered by the hand-built frames
         * in tests/zstd_probes.cpp, which set Symbol_Compression_Mode 1 for
         * all three fields. docs/ZSTD-CORPUS.md carries the pointer. */
        {"tables-rle", "tables", SourceRleSequences, 16384, 3, {},
         [] { ZstdDemand d; d.of_mode = kZstdTableRle;
              d.ml_mode = kZstdTableRle; return d; }()},
        {"tables-compressed", "tables", SourceText, 262144, 9, {},
         [] { ZstdDemand d; d.ll_mode = kZstdTableCompressed;
              d.of_mode = kZstdTableCompressed;
              d.ml_mode = kZstdTableCompressed; return d; }()},
        {"tables-repeat", "tables", SourceText, 262144, 9,
         {{ZSTD_c_targetCBlockSize, 2600}},
         [] { ZstdDemand d; d.ll_mode = kZstdTableRepeat;
              d.of_mode = kZstdTableRepeat; d.ml_mode = kZstdTableRepeat;
              return d; }()},

        /* The high-search family: the class where interop bugs have shipped
         * upstream, so it is carried whatever the block modes turn out to
         * be. */
        {"level-18", "level", SourceText, 131072, 18, {},
         [] { ZstdDemand d; d.block_type = kZstdBlockCompressed; return d; }()},
        {"level-19", "level", SourceText, 131072, 19, {},
         [] { ZstdDemand d; d.block_type = kZstdBlockCompressed; return d; }()},
        {"level-22-long-window", "level", SourceText, 131072, 22,
         {{ZSTD_c_windowLog, 27}},
         [] { ZstdDemand d; d.block_type = kZstdBlockCompressed; return d; }()},
    };

    std::vector<ZstdFixture> fixtures;
    const size_t count = sizeof(kSpecs) / sizeof(kSpecs[0]);
    for (size_t i = 0; i < count; i++) {
        const FamilySpec& spec = kSpecs[i];
        ZstdFixture fixture;
        fixture.name = spec.name;
        fixture.family = spec.family;
        fixture.original = spec.source(spec.source_size);
        fixture.demand = spec.demand;
        if (!Compress(fixture.original, spec.level, spec.params,
                      &fixture.compressed)) {
            return std::vector<ZstdFixture>();
        }
        fixtures.push_back(fixture);
    }

    /* One surface the pinned compressor will not emit, so it is written by
     * hand instead - the shape tests/zstd_probes.cpp already uses for
     * behaviours no compressor produces.
     *
     * An RLE literals section needs a block whose literal alphabet is one
     * symbol AND enough literals for a Huffman attempt to be made at all
     * (zstd_compress_literals.c:158). Both together are unreachable from the
     * advanced API: the sources that make every literal identical also make
     * the run matchable, and the match finder then absorbs the literals into
     * an offset-1 match, leaving too few to reach the floor. Every variant
     * tried came back Raw. So this fixture carries a hand-built block with
     * RLE literals and no sequences at all, and the oracle round-trip below
     * is what says it is a legal frame rather than a plausible one.
     *
     * Frame: magic, descriptor 0x00 (no content size, not single segment, no
     * checksum), Window_Descriptor 0x00 (windowLog 10). One last compressed
     * block of three bytes: the Literals_Section_Header for RLE with
     * Size_Format 00 and Regenerated_Size 20, the repeated byte, then
     * Number_Of_Sequences 0. */
    {
        ZstdFixture fixture;
        fixture.name = "literals-rle-handbuilt";
        fixture.family = "literals";
        fixture.original = Bytes(20, 'z');
        fixture.compressed =
            Bytes{0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00, 0x1d, 0x00, 0x00,
                  static_cast<unsigned char>(kZstdLiteralsRle | (20u << 3)),
                  'z', 0x00};
        fixture.demand.literals_type = kZstdLiteralsRle;
        fixture.demand.block_type = kZstdBlockCompressed;
        fixtures.push_back(fixture);
    }
    return fixtures;
}
