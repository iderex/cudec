/* The GDeflate warp kernel against the oracle and against its own twin
 * (issue #214). The reference's compressor produces the pages, the reference's
 * decompressor and the source say what they mean, and every page goes through
 * cudec_gdeflate_decompress_batch on the device with a poisoned destination:
 * the bytes must equal the source, the poison beyond bytes_written must
 * survive, and the answer must equal what src/gdeflate_block.h produces on the
 * host for the same page - byte for byte on a decode, rung for rung on a
 * refusal. The twin is the thing the oracle diff and the reject ladder were
 * established on, so agreeing with it is what makes those results the
 * kernel's.
 *
 * WHAT IS AND IS NOT HERE. The corpus reaches all three block types - counted
 * off the twin's own census and REQUIRED rather than reported, so a generator
 * that stops reaching one reds instead of costing that arm its coverage in
 * silence - several blocks in a page and several pages in a stream, at five
 * compression levels including the level 0 that forces the stored arm, plus
 * every page the kernel refuses being refused for the twin's reason. The
 * standing device gate set - determinism across launches, the two-directional
 * mutant parity, the capacity adversarials driven at and beyond the tile - is
 * #218 and is not restated here. Nothing here is timed. */
#include "cudec.h"
#include "gdeflate_decode.cuh"
#include "gdeflate_page_writer.h"
#include "require.h"

#include "vendor_rt_test.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::GDeflateDecodePage;
using cudec_detail::GDeflatePageState;
using cudec_detail::GDeflateStatusFor;
using cudec_detail::kGDeflateBlockDynamic;
using cudec_detail::kGDeflateBlockStatic;
using cudec_detail::kGDeflateBlockStored;
using cudec_detail::kGDeflateBlockTypeCount;
using cudec_test::GDeflatePageWriter;

constexpr size_t kTileBytes = 64u * 1024u;
constexpr unsigned char kDstPoison = 0xA5;

/* The same deterministic generators the block twin draws its corpus from: a
 * named recurrence rather than a library one, so the corpus is identical on
 * every machine and a parity failure reproduces. */
class Lcg {
   public:
    explicit Lcg(unsigned seed) : state_(seed) {}
    unsigned Next() {
        state_ = state_ * 1103515245u + 12345u;
        return (state_ >> 16) & 0xFFFFu;
    }

   private:
    unsigned state_;
};

std::vector<unsigned char> Incompressible(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<unsigned char>(lcg.Next() & 0xFFu);
    }
    return v;
}

std::vector<unsigned char> MixedEntropy(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        const unsigned r = lcg.Next();
        v[i] = static_cast<unsigned char>((r % 4u == 0u) ? (r & 0xFFu)
                                                         : ('a' + (r % 5u)));
    }
    return v;
}

std::vector<unsigned char> ShortRepeats(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v;
    v.reserve(n);
    while (v.size() < n) {
        const size_t run = 3u + (lcg.Next() % 60u);
        const unsigned char b = static_cast<unsigned char>('a' + lcg.Next() % 6u);
        for (size_t i = 0; i < run && v.size() < n; i++) {
            v.push_back(b);
        }
        const size_t noise = lcg.Next() % 8u;
        for (size_t i = 0; i < noise && v.size() < n; i++) {
            v.push_back(static_cast<unsigned char>(lcg.Next() & 0xFFu));
        }
    }
    return v;
}

/* A seven-byte alphabet repeated with no noise at all, which is the input
 * character bench/bench_gdeflate.cpp's forced-static family uses: at a low
 * level a dynamic table description costs more than the fixed code it would
 * replace. It is a second route to the static arm rather than the only one -
 * measured, the corpus below reaches 20 static blocks without it and 24 with
 * it - and a second route is what keeps that arm's coverage from resting on
 * one generator. */
std::vector<unsigned char> LowEntropy(size_t n) {
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<unsigned char>('a' + (i % 7u));
    }
    return v;
}

/* Long runs of one byte, which is what draws the deepest overlapping copies:
 * a distance of one under a length reaching the DEFLATE64 maximum. */
std::vector<unsigned char> LongRuns(unsigned seed, size_t n) {
    Lcg lcg(seed);
    std::vector<unsigned char> v;
    v.reserve(n);
    while (v.size() < n) {
        const size_t run = 200u + (lcg.Next() % 70000u);
        const unsigned char b = static_cast<unsigned char>(lcg.Next() & 0xFFu);
        for (size_t i = 0; i < run && v.size() < n; i++) {
            v.push_back(b);
        }
    }
    return v;
}

bool CompressPages(int level, const std::vector<unsigned char>& in,
                   std::vector<std::vector<unsigned char> >* pages) {
    libdeflate_gdeflate_compressor* c =
        libdeflate_alloc_gdeflate_compressor(level);
    if (c == nullptr) {
        return false;
    }
    size_t npages = 0;
    const size_t bound =
        libdeflate_gdeflate_compress_bound(c, in.size(), &npages);
    if (bound == 0 || npages == 0) {
        libdeflate_free_gdeflate_compressor(c);
        return false;
    }
    /* The bound is over the WHOLE input and the per-page bound is its share
     * of it, which is what the reference's own header says and what a corpus
     * of a hundred megabytes finds out: a pool of bound times pages is the
     * square of the input. */
    const size_t per_page = (bound + npages - 1u) / npages;
    std::vector<unsigned char> pool(per_page * npages, 0);
    std::vector<libdeflate_gdeflate_out_page> out(npages);
    for (size_t i = 0; i < npages; i++) {
        out[i].data = pool.data() + i * per_page;
        out[i].nbytes = per_page;
    }
    const size_t total = libdeflate_gdeflate_compress(c, in.data(), in.size(),
                                                      out.data(), npages);
    libdeflate_free_gdeflate_compressor(c);
    if (total == 0) {
        return false;
    }
    pages->clear();
    for (size_t i = 0; i < npages; i++) {
        const unsigned char* p =
            static_cast<const unsigned char*>(out[i].data);
        pages->push_back(std::vector<unsigned char>(p, p + out[i].nbytes));
    }
    return true;
}

struct Chunk {
    const std::vector<unsigned char>* src;
    size_t dst_capacity;
};

/* One batch through the real plumbing: device pointer tables, per-page sizes
 * and capacities, poisoned destinations, the results primed with a non-OK
 * sentinel so an entry the kernel never wrote reads as a failure. */
int RunBatch(const std::vector<Chunk>& chunks,
             std::vector<cudec_chunk_result>* results,
             std::vector<std::vector<unsigned char> >* dst_bytes) {
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
    REQUIRE_RT(cudec_rt::device_memset(d_results, 0xFF, n * sizeof(*d_results)));

    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    REQUIRE(cudec_gdeflate_decompress_batch(d_srcs, d_sizes, d_dsts, d_caps, n,
                                            d_results,
                                            cudec_rt::abi_stream(stream)) ==
            CUDEC_OK);
    REQUIRE_RT(cudec_rt::stream_synchronize(stream));
    REQUIRE_RT(cudec_rt::stream_destroy(stream));

    results->assign(n, cudec_chunk_result{});
    REQUIRE_RT(cudec_rt::memcpy(results->data(), d_results,
                                n * sizeof(*d_results), cudec_rt::memcpy_d2h));
    dst_bytes->assign(n, std::vector<unsigned char>());
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

/* The host twin's verdict on one page: the bytes on a decode, the rung on a
 * refusal, and the block types the page turned out to be made of. The census
 * is the twin's because the twin is what says what a page CONTAINS; the
 * kernel's job is to agree with it, which is what every check below asks. */
struct TwinVerdict {
    bool ok;
    cudec_status status;
    std::vector<unsigned char> bytes;
    uint32_t type_blocks[kGDeflateBlockTypeCount];
};

TwinVerdict TwinDecode(const std::vector<unsigned char>& page, size_t cap) {
    TwinVerdict v;
    GDeflatePageState st;
    uint64_t out_len = 0;
    v.bytes.assign(cap, 0);
    v.ok = GDeflateDecodePage(st, page.data(), page.size(), v.bytes.data(),
                              cap, &out_len);
    for (uint32_t t = 0; t < kGDeflateBlockTypeCount; t++) {
        v.type_blocks[t] = v.ok ? st.type_blocks[t] : 0;
    }
    if (v.ok) {
        v.bytes.resize(static_cast<size_t>(out_len));
        v.status = CUDEC_OK;
    } else {
        v.bytes.clear();
        v.status = GDeflateStatusFor(st.s.reject);
    }
    return v;
}

/* One page's device result against its twin verdict and, on a decode, against
 * the expected bytes: status, bytes_written, the bytes, and the poison. */
int CheckPage(const char* ctx, const cudec_chunk_result& result,
              const std::vector<unsigned char>& dst, const TwinVerdict& twin,
              const std::vector<unsigned char>* expected) {
    REQUIRE_CTX(result.reserved == 0, "%s", ctx);
    REQUIRE_CTX(result.status == twin.status, "%s device status %d twin %d",
                ctx, static_cast<int>(result.status),
                static_cast<int>(twin.status));
    if (!twin.ok) {
        REQUIRE_CTX(result.bytes_written == 0, "%s", ctx);
        return 0;
    }
    REQUIRE_CTX(result.bytes_written == twin.bytes.size(),
                "%s bytes_written %llu twin %zu", ctx,
                static_cast<unsigned long long>(result.bytes_written),
                twin.bytes.size());
    REQUIRE_CTX(dst.size() >= twin.bytes.size(), "%s", ctx);
    REQUIRE_CTX(std::memcmp(dst.data(), twin.bytes.data(),
                            twin.bytes.size()) == 0,
                "%s device bytes differ from the twin", ctx);
    if (expected != nullptr) {
        REQUIRE_CTX(expected->size() == twin.bytes.size(), "%s", ctx);
        REQUIRE_CTX(std::memcmp(dst.data(), expected->data(),
                                expected->size()) == 0,
                    "%s device bytes differ from the source", ctx);
    }
    for (size_t j = twin.bytes.size(); j < dst.size(); j++) {
        REQUIRE_CTX(dst[j] == kDstPoison, "%s poison at %zu", ctx, j);
    }
    return 0;
}

/* What a corpus run turned out to have covered: pages decoded, and blocks of
 * each type inside them. The second half is the reason it exists - a corpus
 * whose pages all happen to be dynamic proves nothing about the stored and
 * static arms, and a generator that stops reaching one costs that coverage
 * silently unless somebody counts. */
struct Coverage {
    size_t pages;
    uint64_t type_blocks[kGDeflateBlockTypeCount];
};

/* Every page of a compressed stream in ONE batch, one page per chunk, each
 * into its own poisoned tile. */
int CheckStream(const char* name, int level,
                const std::vector<unsigned char>& in, Coverage* cov) {
    std::vector<std::vector<unsigned char> > pages;
    REQUIRE_CTX(CompressPages(level, in, &pages), "%s level %d", name, level);
    std::vector<Chunk> chunks(pages.size());
    for (size_t i = 0; i < pages.size(); i++) {
        chunks[i].src = &pages[i];
        chunks[i].dst_capacity = kTileBytes;
    }
    std::vector<cudec_chunk_result> results;
    std::vector<std::vector<unsigned char> > dst;
    REQUIRE(RunBatch(chunks, &results, &dst) == 0);
    for (size_t i = 0; i < pages.size(); i++) {
        char ctx[128];
        std::snprintf(ctx, sizeof(ctx), "%s level %d page %zu", name, level, i);
        const size_t consumed = i * kTileBytes;
        const size_t want =
            in.size() - consumed < kTileBytes ? in.size() - consumed : kTileBytes;
        const std::vector<unsigned char> tile(in.begin() + consumed,
                                              in.begin() + consumed + want);
        const TwinVerdict twin = TwinDecode(pages[i], kTileBytes);
        REQUIRE_CTX(twin.ok, "%s: the twin refused a page the oracle made",
                    ctx);
        REQUIRE(CheckPage(ctx, results[i], dst[i], twin, &tile) == 0);
        for (uint32_t t = 0; t < kGDeflateBlockTypeCount; t++) {
            cov->type_blocks[t] += twin.type_blocks[t];
        }
    }
    cov->pages += pages.size();
    return 0;
}

int PristineCorpus() {
    /* Level 0 is in the set because the reference's own header says it emits
     * uncompressed blocks by construction, which is the only way to force the
     * stored arm rather than hope an input reaches it. */
    const int levels[] = {0, 1, 6, 9, 12};
    Coverage cov;
    cov.pages = 0;
    for (uint32_t t = 0; t < kGDeflateBlockTypeCount; t++) {
        cov.type_blocks[t] = 0;
    }
    for (size_t l = 0; l < sizeof(levels) / sizeof(levels[0]); l++) {
        const int level = levels[l];
        REQUIRE(CheckStream("incompressible-60k", level,
                            Incompressible(2, 60000), &cov) == 0);
        REQUIRE(CheckStream("mixed-60k", level, MixedEntropy(4, 60000),
                            &cov) == 0);
        REQUIRE(CheckStream("short-repeats-60k", level, ShortRepeats(5, 60000),
                            &cov) == 0);
        REQUIRE(CheckStream("low-entropy-60k", level, LowEntropy(60000),
                            &cov) == 0);
        REQUIRE(CheckStream("long-runs-3tiles", level,
                            LongRuns(9, 3u * kTileBytes), &cov) == 0);
        REQUIRE(CheckStream("mixed-multi-page", level,
                            MixedEntropy(6, 3u * kTileBytes + 7u),
                            &cov) == 0);
        REQUIRE(CheckStream("exact-tile", level, ShortRepeats(7, kTileBytes),
                            &cov) == 0);
        REQUIRE(CheckStream("one-byte", level, MixedEntropy(11, 1),
                            &cov) == 0);
    }
    /* More pages than teams in the launch shape, so the grid-stride loop and
     * the reuse of one team's shared tables across pages are exercised. */
    REQUIRE(CheckStream("mixed-40-tiles", 6, MixedEntropy(12, 40u * kTileBytes),
                        &cov) == 0);
    /* The three block types are REQUIRED rather than reported: a change to a
     * generator, to the level set or to the reference that stopped reaching
     * one of them would otherwise cost that arm its whole-corpus coverage and
     * say nothing. */
    std::printf("gdeflate_device: block census over the corpus: %llu stored, "
                "%llu static, %llu dynamic\n",
                static_cast<unsigned long long>(
                    cov.type_blocks[kGDeflateBlockStored]),
                static_cast<unsigned long long>(
                    cov.type_blocks[kGDeflateBlockStatic]),
                static_cast<unsigned long long>(
                    cov.type_blocks[kGDeflateBlockDynamic]));
    REQUIRE(cov.type_blocks[kGDeflateBlockStored] > 0);
    REQUIRE(cov.type_blocks[kGDeflateBlockStatic] > 0);
    REQUIRE(cov.type_blocks[kGDeflateBlockDynamic] > 0);
    const size_t pages_seen = cov.pages;
    std::printf("gdeflate_device: %zu pages decoded on the device, each equal "
                "to the source and to the host twin\n",
                pages_seen);
    return 0;
}

/* Pages the kernel must refuse, each held to the twin's rung through the
 * status it maps to and to the failure contract: bytes_written zero, poison
 * intact. The negatives are the shapes the reject ladder reaches through a
 * page: a reserved block type, a stored length past the capacity, a literal
 * past the capacity, a page cut short, a page below the priming round, and a
 * TileStream envelope handed in as if it were a page. */
std::vector<unsigned char> PageWithBlockType(uint32_t block_type) {
    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, block_type);
    return w.Finish();
}

std::vector<unsigned char> StoredPage(uint32_t len) {
    GDeflatePageWriter w;
    w.Reset();
    w.Push(1, 1);
    w.Push(2, 0);
    w.Ensure();
    w.Push(16, len);
    return w.Finish();
}

int Negatives() {
    std::vector<std::vector<unsigned char> > pages;
    std::vector<size_t> caps;
    std::vector<const char*> names;

    pages.push_back(PageWithBlockType(3));
    caps.push_back(64);
    names.push_back("reserved block type");

    pages.push_back(StoredPage(8));
    caps.push_back(4);
    names.push_back("stored length past the capacity");

    std::vector<std::vector<unsigned char> > real;
    REQUIRE(CompressPages(6, MixedEntropy(4, 60000), &real));
    REQUIRE(real.size() == 1);
    pages.push_back(real[0]);
    caps.push_back(100);
    names.push_back("decode past a 100-byte capacity");

    std::vector<unsigned char> cut(real[0].begin(),
                                   real[0].begin() + (real[0].size() / 2u) / 4u * 4u);
    pages.push_back(cut);
    caps.push_back(kTileBytes);
    names.push_back("page cut in half");

    pages.push_back(std::vector<unsigned char>(64, 0));
    caps.push_back(kTileBytes);
    names.push_back("page below the priming round");

    std::vector<unsigned char> partial(real[0].begin(), real[0].begin() + 131);
    pages.push_back(partial);
    caps.push_back(kTileBytes);
    names.push_back("page with a trailing partial word");

    /* Zero capacity: a decode that must write nothing has nowhere to write
     * even one literal, and the twin says which rung answers. */
    pages.push_back(real[0]);
    caps.push_back(0);
    names.push_back("zero capacity");

    std::vector<Chunk> chunks(pages.size());
    for (size_t i = 0; i < pages.size(); i++) {
        chunks[i].src = &pages[i];
        chunks[i].dst_capacity = caps[i];
    }
    std::vector<cudec_chunk_result> results;
    std::vector<std::vector<unsigned char> > dst;
    REQUIRE(RunBatch(chunks, &results, &dst) == 0);
    for (size_t i = 0; i < pages.size(); i++) {
        const TwinVerdict twin = TwinDecode(pages[i], caps[i]);
        REQUIRE_CTX(!twin.ok, "%s: the twin accepted a negative", names[i]);
        /* The destination is unspecified on a refusal by contract, so no
         * poison is asserted here; what the contract does promise, and what
         * CheckPage holds, is the status and a bytes_written of zero. */
        REQUIRE(CheckPage(names[i], results[i], dst[i], twin, nullptr) == 0);
        std::printf("gdeflate_device: refused '%s' with status %d, as the "
                    "twin does\n",
                    names[i], static_cast<int>(results[i].status));
    }
    return 0;
}

/* Whole files handed on the command line, each compressed at the recorded M4
 * level set and every page decoded on the device: the whole-corpus diff the
 * kernel's Done-when asks for, over the corpora the bench records, run by hand
 * and pasted rather than registered, because ctest carries no corpus. */
std::vector<unsigned char> ReadFile(const char* path) {
    std::vector<unsigned char> bytes;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        return bytes;
    }
    unsigned char buf[65536];
    for (;;) {
        const size_t got = std::fread(buf, 1, sizeof(buf), f);
        if (got == 0) {
            break;
        }
        bytes.insert(bytes.end(), buf, buf + got);
    }
    std::fclose(f);
    return bytes;
}

int FileCorpus(int argc, char** argv) {
    const int levels[] = {1, 6, 12};
    Coverage cov;
    cov.pages = 0;
    for (uint32_t t = 0; t < kGDeflateBlockTypeCount; t++) {
        cov.type_blocks[t] = 0;
    }
    for (int a = 1; a < argc; a++) {
        const std::vector<unsigned char> in = ReadFile(argv[a]);
        REQUIRE_CTX(!in.empty(), "%s: unreadable or empty", argv[a]);
        for (size_t l = 0; l < sizeof(levels) / sizeof(levels[0]); l++) {
            REQUIRE(CheckStream(argv[a], levels[l], in, &cov) == 0);
        }
        std::printf("gdeflate_device: %s, %zu bytes, three levels, %zu pages "
                    "so far\n",
                    argv[a], in.size(), cov.pages);
    }
    std::printf("gdeflate_device: block census over the files: %llu stored, "
                "%llu static, %llu dynamic\n",
                static_cast<unsigned long long>(
                    cov.type_blocks[kGDeflateBlockStored]),
                static_cast<unsigned long long>(
                    cov.type_blocks[kGDeflateBlockStatic]),
                static_cast<unsigned long long>(
                    cov.type_blocks[kGDeflateBlockDynamic]));
    std::printf("gdeflate_device: %zu file pages decoded on the device, each "
                "equal to the source and to the host twin, 0 mismatches\n",
                cov.pages);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        return FileCorpus(argc, argv) != 0 ? 1 : 0;
    }
    if (PristineCorpus() != 0) {
        return 1;
    }
    if (Negatives() != 0) {
        return 1;
    }
    std::printf("PASS: gdeflate_device\n");
    return 0;
}
