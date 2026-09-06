/* The Zstd block-per-frame kernel against the oracle (issue #203). The
 * pinned compressor produces the frames, the pinned decompressor says what
 * they mean, and every frame goes through cudec_zstd_decompress_batch on the
 * device into a poisoned destination: the bytes must equal what libzstd
 * decodes, bytes_written must equal that length, and the poison past
 * bytes_written must survive.
 *
 * WHAT IS AND IS NOT HERE. The #185 corpus, which is built to reach every
 * decode surface the subset admits, plus the batch geometry the entry
 * actually consumes - one source cut into many independent frames, all of
 * them in one launch. The census below is REQUIRED rather than reported: a
 * corpus that decodes byte-identically proves nothing about a surface it
 * never reached, so a generator that stops emitting Treeless literals or a
 * Repeat table mode reds here instead of costing that arm its coverage in
 * silence.
 *
 * The standing device gate set - same-batch-twice determinism, the
 * two-directional mutant reject parity against libzstd, and the capacity and
 * window adversarials - is #399 and is deliberately not restated here.
 * Nothing here is timed. */
#include "cudec.h"
#include "require.h"
#include "zstd_corpus.h"

#include "vendor_rt_test.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

constexpr unsigned char kDstPoison = 0xA5;
/* Slack past the declared content size, so the poison beyond bytes_written
 * has somewhere to survive. */
constexpr size_t kCapacitySlack = 96;

struct Chunk {
    const Bytes* src;
    size_t dst_capacity;
};

/* One batch through the real plumbing: device pointer tables, per-frame sizes
 * and capacities, poisoned destinations, and the results primed with a non-OK
 * sentinel so an entry the kernel never wrote reads as a failure. */
int RunBatch(const std::vector<Chunk>& chunks,
             std::vector<cudec_chunk_result>* results,
             std::vector<Bytes>* dst_bytes) {
    const size_t n = chunks.size();
    std::vector<const void*> h_srcs(n);
    std::vector<void*> h_dsts(n);
    std::vector<size_t> h_sizes(n);
    std::vector<size_t> h_caps(n);
    for (size_t i = 0; i < n; i++) {
        void* d_src = nullptr;
        void* d_dst = nullptr;
        const size_t src_size = chunks[i].src->size();
        REQUIRE_RT(cudec_rt::device_malloc(&d_src, src_size ? src_size : 1));
        if (src_size) {
            REQUIRE_RT(cudec_rt::memcpy(d_src, chunks[i].src->data(), src_size,
                                        cudec_rt::memcpy_h2d));
        }
        const size_t cap = chunks[i].dst_capacity;
        REQUIRE_RT(cudec_rt::device_malloc(&d_dst, cap ? cap : 1));
        if (cap) {
            REQUIRE_RT(cudec_rt::device_memset(d_dst, kDstPoison, cap));
        }
        h_srcs[i] = d_src;
        h_dsts[i] = d_dst;
        h_sizes[i] = src_size;
        h_caps[i] = cap;
    }
    const void** d_srcs;
    void** d_dsts;
    size_t* d_sizes;
    size_t* d_caps;
    cudec_chunk_result* d_results;
    REQUIRE_RT(cudec_rt::device_malloc(&d_srcs, n * sizeof(*d_srcs)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_dsts, n * sizeof(*d_dsts)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_sizes, n * sizeof(*d_sizes)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_caps, n * sizeof(*d_caps)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_results, n * sizeof(*d_results)));
    REQUIRE_RT(cudec_rt::memcpy(d_srcs, h_srcs.data(), n * sizeof(*d_srcs),
                                cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_dsts, h_dsts.data(), n * sizeof(*d_dsts),
                                cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_sizes, h_sizes.data(), n * sizeof(*d_sizes),
                                cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_caps, h_caps.data(), n * sizeof(*d_caps),
                                cudec_rt::memcpy_h2d));
    REQUIRE_RT(
        cudec_rt::device_memset(d_results, 0xFF, n * sizeof(*d_results)));

    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    REQUIRE(cudec_zstd_decompress_batch(d_srcs, d_sizes, d_dsts, d_caps, n,
                                        d_results,
                                        cudec_rt::abi_stream(stream)) ==
            CUDEC_OK);
    REQUIRE_RT(cudec_rt::stream_synchronize(stream));
    REQUIRE_RT(cudec_rt::stream_destroy(stream));

    results->assign(n, cudec_chunk_result{});
    REQUIRE_RT(cudec_rt::memcpy(results->data(), d_results,
                                n * sizeof(*d_results), cudec_rt::memcpy_d2h));
    dst_bytes->assign(n, Bytes());
    for (size_t i = 0; i < n; i++) {
        (*dst_bytes)[i].assign(h_caps[i], 0);
        if (h_caps[i]) {
            REQUIRE_RT(cudec_rt::memcpy((*dst_bytes)[i].data(), h_dsts[i],
                                        h_caps[i], cudec_rt::memcpy_d2h));
        }
        REQUIRE_RT(cudec_rt::device_free(h_dsts[i]));
        REQUIRE_RT(cudec_rt::device_free(const_cast<void*>(h_srcs[i])));
    }
    REQUIRE_RT(cudec_rt::device_free(d_srcs));
    REQUIRE_RT(cudec_rt::device_free(d_dsts));
    REQUIRE_RT(cudec_rt::device_free(d_sizes));
    REQUIRE_RT(cudec_rt::device_free(d_caps));
    REQUIRE_RT(cudec_rt::device_free(d_results));
    return 0;
}

/* One frame's decode held to the oracle's: the length, the bytes, and the
 * poison the kernel had no business touching. */
int CheckDecode(const char* name, const cudec_chunk_result& result,
                const Bytes& produced, const Bytes& expected) {
    REQUIRE_CTX(result.status == CUDEC_OK, "%s: status %d", name,
                static_cast<int>(result.status));
    REQUIRE_CTX(result.bytes_written == expected.size(),
                "%s: wrote %llu bytes, the oracle produced %llu", name,
                static_cast<unsigned long long>(result.bytes_written),
                static_cast<unsigned long long>(expected.size()));
    REQUIRE_CTX(produced.size() >= expected.size() + kCapacitySlack,
                "%s: destination readback is short", name);
    for (size_t i = 0; i < expected.size(); i++) {
        REQUIRE_CTX(produced[i] == expected[i],
                    "%s: byte %llu is 0x%02X, the oracle says 0x%02X", name,
                    static_cast<unsigned long long>(i),
                    static_cast<unsigned>(produced[i]),
                    static_cast<unsigned>(expected[i]));
    }
    for (size_t i = expected.size(); i < produced.size(); i++) {
        REQUIRE_CTX(produced[i] == kDstPoison,
                    "%s: byte %llu past bytes_written is 0x%02X, not the "
                    "poison",
                    name, static_cast<unsigned long long>(i),
                    static_cast<unsigned>(produced[i]));
    }
    return 0;
}

/* What the corpus actually reached, counted off the frame walker rather than
 * off what the generator was asked for. */
struct Census {
    unsigned literals[4];
    unsigned modes[4];
    unsigned block_types[3];
    unsigned four_stream;
    unsigned single_stream;
    unsigned checksummed;
    unsigned multi_block;
};

void CountFrame(const ZstdFrameShape& shape, Census* census) {
    if (shape.checksum_present) {
        census->checksummed++;
    }
    if (shape.blocks.size() > 1) {
        census->multi_block++;
    }
    for (size_t i = 0; i < shape.blocks.size(); i++) {
        const ZstdBlockShape& block = shape.blocks[i];
        if (block.block_type < 3) {
            census->block_types[block.block_type]++;
        }
        if (block.block_type != kZstdBlockCompressed) {
            continue;
        }
        if (block.literals_type < 4) {
            census->literals[block.literals_type]++;
        }
        if (block.literals_streams == 4) {
            census->four_stream++;
        } else if (block.literals_streams == 1) {
            census->single_stream++;
        }
        if (block.sequence_count == 0) {
            continue;
        }
        const unsigned modes[3] = {block.ll_mode, block.of_mode,
                                   block.ml_mode};
        for (unsigned m = 0; m < 3; m++) {
            if (modes[m] < 4) {
                census->modes[modes[m]]++;
            }
        }
    }
}

/* ---- The corpus rung ---------------------------------------------------- */

int RunCorpus(Census* census) {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());

    std::vector<Chunk> chunks;
    std::vector<Bytes> frames;
    std::vector<Bytes> expected;
    std::vector<std::string> names;
    frames.reserve(fixtures.size());
    expected.reserve(fixtures.size());
    for (size_t i = 0; i < fixtures.size(); i++) {
        Bytes reference;
        const Bytes frame(fixtures[i].compressed.begin(),
                          fixtures[i].compressed.end());
        REQUIRE_CTX(ZstdOracleDecodes(frame, &reference),
                    "%s: the oracle refused a fixture it produced",
                    fixtures[i].name.c_str());
        frames.push_back(frame);
        expected.push_back(reference);
        names.push_back(fixtures[i].name);
    }
    for (size_t i = 0; i < frames.size(); i++) {
        Chunk chunk;
        chunk.src = &frames[i];
        chunk.dst_capacity = expected[i].size() + kCapacitySlack;
        chunks.push_back(chunk);
    }

    std::vector<cudec_chunk_result> results;
    std::vector<Bytes> produced;
    if (RunBatch(chunks, &results, &produced) != 0) {
        return 1;
    }

    unsigned decoded = 0;
    unsigned declined = 0;
    for (size_t i = 0; i < frames.size(); i++) {
        if (results[i].status == CUDEC_ERR_UNSUPPORTED) {
            /* A fixture outside the v1 subset is declined rather than
             * refused, and declining is not this file's subject. */
            REQUIRE_CTX(results[i].bytes_written == 0,
                        "%s: declined and still reported bytes",
                        names[i].c_str());
            std::printf("corpus: declined %s\n", names[i].c_str());
            declined++;
            continue;
        }
        if (CheckDecode(names[i].c_str(), results[i], produced[i],
                        expected[i]) != 0) {
            return 1;
        }
        decoded++;
        ZstdFrameShape shape;
        std::string why;
        REQUIRE_CTX(ParseZstdFrameShape(frames[i], &shape, &why),
                    "%s: the frame walker could not account for a frame the "
                    "kernel decoded: %s",
                    names[i].c_str(), why.c_str());
        CountFrame(shape, census);
    }
    std::printf("corpus: %u frames decoded byte-identically, %u declined\n",
                decoded, declined);
    REQUIRE(decoded > 0);
    return 0;
}

/* ---- The surfaces the pinned compressor will not emit ------------------- */

/* An RLE literals section inside the accepted envelope.
 *
 * WHY IT IS BUILT HERE AND NOT TAKEN FROM THE CORPUS. The corpus already
 * carries one, and it is declined rather than decoded: its frame header
 * declares no content size, which section 12.2 puts outside the subset, so
 * the fixture proves the section legal and leaves the kernel's RLE arm
 * unrun. tests/zstd_corpus.cpp records why no compressor emits this section
 * at all - the sources that make every literal identical also make the run
 * matchable - so a compressed corpus cannot supply one at any level.
 *
 * The frame: magic, a descriptor with Single_Segment set (which is what
 * makes Frame_Content_Size present and the window the content size), the
 * one-byte content size, then one last Compressed block of three bytes - the
 * Literals_Section_Header for RLE with Size_Format 00 and Regenerated_Size
 * 20, the repeated byte, and Number_Of_Sequences 0.
 *
 * The oracle round-trip below is what says this is a legal frame rather than
 * a plausible one; the kernel is never held to bytes only this file
 * believes. */
int RunHandBuilt(Census* census) {
    const unsigned char kRegenerated = 20;
    Bytes frame;
    frame.push_back(0x28);
    frame.push_back(0xB5);
    frame.push_back(0x2F);
    frame.push_back(0xFD);
    frame.push_back(0x20);
    frame.push_back(kRegenerated);
    /* Block_Header: Last_Block, Block_Type Compressed, Block_Size 3. */
    frame.push_back(0x1D);
    frame.push_back(0x00);
    frame.push_back(0x00);
    frame.push_back(
        static_cast<unsigned char>(kZstdLiteralsRle | (kRegenerated << 3)));
    frame.push_back('z');
    frame.push_back(0x00);

    Bytes expected;
    REQUIRE_CTX(ZstdOracleDecodes(frame, &expected),
                "literals-rle-in-subset: the reference refused the frame this "
                "test hand-built, so it is not a legal frame to hold the "
                "kernel to");
    REQUIRE(expected.size() == kRegenerated);

    std::vector<Chunk> chunks;
    Chunk chunk;
    chunk.src = &frame;
    chunk.dst_capacity = expected.size() + kCapacitySlack;
    chunks.push_back(chunk);

    std::vector<cudec_chunk_result> results;
    std::vector<Bytes> produced;
    if (RunBatch(chunks, &results, &produced) != 0) {
        return 1;
    }
    if (CheckDecode("literals-rle-in-subset", results[0], produced[0],
                    expected) != 0) {
        return 1;
    }
    ZstdFrameShape shape;
    std::string why;
    REQUIRE_CTX(ParseZstdFrameShape(frame, &shape, &why),
                "literals-rle-in-subset: the frame walker could not account "
                "for it: %s",
                why.c_str());
    CountFrame(shape, census);
    std::printf("hand-built: one RLE literals section in the subset\n");
    return 0;
}

/* ---- The batch rung ----------------------------------------------------- */

/* The geometry the entry actually consumes: one source cut at a fixed size,
 * every chunk its own independent frame, all of them in ONE launch. It is a
 * different question from the corpus above - that one asks whether a frame
 * decodes, this one asks whether many frames decode side by side without
 * reading each other's shared memory or each other's destination. */
int RunBatchGeometry(size_t chunk_size, int level) {
    Bytes source;
    /* A deterministic mixed-entropy source: literal runs the compressor can
     * match, punctuated by bytes it cannot, so the frames carry both
     * sequences and incompressible literals rather than one shape. */
    unsigned state = 0x1234567u;
    for (size_t i = 0; i < 512u * 1024u; i++) {
        state = state * 1103515245u + 12345u;
        const unsigned r = (state >> 16) & 0xFFFFu;
        source.push_back(static_cast<unsigned char>(
            (r % 5u == 0u) ? (r & 0xFFu) : ('a' + (r % 7u))));
    }
    const std::vector<Bytes> frames =
        MakeZstdBatchFrames(source, chunk_size, level);
    REQUIRE(!frames.empty());

    std::vector<Bytes> expected(frames.size());
    std::vector<Chunk> chunks;
    for (size_t i = 0; i < frames.size(); i++) {
        REQUIRE(ZstdOracleDecodes(frames[i], &expected[i]));
        Chunk chunk;
        chunk.src = &frames[i];
        chunk.dst_capacity = expected[i].size() + kCapacitySlack;
        chunks.push_back(chunk);
    }

    std::vector<cudec_chunk_result> results;
    std::vector<Bytes> produced;
    if (RunBatch(chunks, &results, &produced) != 0) {
        return 1;
    }
    Bytes rejoined;
    for (size_t i = 0; i < frames.size(); i++) {
        char name[64];
        std::snprintf(name, sizeof(name), "batch %llu chunk %llu",
                      static_cast<unsigned long long>(chunk_size),
                      static_cast<unsigned long long>(i));
        if (CheckDecode(name, results[i], produced[i], expected[i]) != 0) {
            return 1;
        }
        rejoined.insert(rejoined.end(), expected[i].begin(), expected[i].end());
    }
    /* The cut and the rejoin are the caller's contract, so the batch is held
     * to reproducing the source it was cut from rather than only to each
     * frame's own oracle answer. */
    REQUIRE(rejoined.size() == source.size());
    REQUIRE(std::memcmp(rejoined.data(), source.data(), source.size()) == 0);
    std::printf("batch: %llu frames at a chunk size of %llu, level %d\n",
                static_cast<unsigned long long>(frames.size()),
                static_cast<unsigned long long>(chunk_size), level);
    return 0;
}

}  // namespace

int main() {
    Census census;
    std::memset(&census, 0, sizeof(census));

    if (RunCorpus(&census) != 0) {
        return 1;
    }
    if (RunHandBuilt(&census) != 0) {
        return 1;
    }
    if (RunBatchGeometry(64u * 1024u, 3) != 0) {
        return 1;
    }
    if (RunBatchGeometry(200u * 1024u, 9) != 0) {
        return 1;
    }

    std::printf(
        "census: the Zstd kernel decoded a corpus carrying "
        "libzstd - blocks raw/rle/compressed %u/%u/%u, literals "
        "raw/rle/compressed/treeless %u/%u/%u/%u, table modes "
        "basic/rle/compressed/repeat %u/%u/%u/%u, %u four-stream and %u "
        "single-stream literals sections, %u checksummed frames, %u frames "
        "of more than one block\n",
        census.block_types[kZstdBlockRaw], census.block_types[kZstdBlockRle],
        census.block_types[kZstdBlockCompressed],
        census.literals[kZstdLiteralsRaw], census.literals[kZstdLiteralsRle],
        census.literals[kZstdLiteralsCompressed],
        census.literals[kZstdLiteralsTreeless],
        census.modes[kZstdTableBasic], census.modes[kZstdTableRle],
        census.modes[kZstdTableCompressed], census.modes[kZstdTableRepeat],
        census.four_stream, census.single_stream, census.checksummed,
        census.multi_block);

    /* The surfaces the decoded corpus actually carried. Required, not
     * reported: without this a generator that quietly stopped emitting one of
     * them would leave this file green and that arm of the kernel unrun. */
    REQUIRE(census.block_types[kZstdBlockRaw] > 0);
    REQUIRE(census.block_types[kZstdBlockRle] > 0);
    REQUIRE(census.block_types[kZstdBlockCompressed] > 0);
    REQUIRE(census.literals[kZstdLiteralsRaw] > 0);
    REQUIRE(census.literals[kZstdLiteralsRle] > 0);
    REQUIRE(census.literals[kZstdLiteralsCompressed] > 0);
    REQUIRE(census.literals[kZstdLiteralsTreeless] > 0);
    REQUIRE(census.modes[kZstdTableBasic] > 0);
    REQUIRE(census.modes[kZstdTableRle] > 0);
    REQUIRE(census.modes[kZstdTableCompressed] > 0);
    REQUIRE(census.modes[kZstdTableRepeat] > 0);
    REQUIRE(census.four_stream > 0);
    REQUIRE(census.single_stream > 0);
    REQUIRE(census.checksummed > 0);
    REQUIRE(census.multi_block > 0);

    std::printf(
        "PASS: every frame of the Zstd corpus and of the batch geometry "
        "decodes on the device byte-identically to the pinned reference, "
        "with the poison past bytes_written intact\n");
    return 0;
}
