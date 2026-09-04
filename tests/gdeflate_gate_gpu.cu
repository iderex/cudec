/* The standing device gate set for the GDeflate path (issue #218). The kernel
 * that landed under #214 is held here to the four properties this project
 * requires of every decode path before it is trusted, on the device rather
 * than on the twin:
 *
 *   DETERMINISM. One corpus, one reference decode, then the same batch across
 *   five launch geometries and a three-stream split, five runs each, with the
 *   whole destination arena re-poisoned before every run and compared in full.
 *   Which team decodes which page and in what order changes in every direction
 *   the kernel admits, and not one output byte may notice.
 *
 *   TWO-DIRECTIONAL MUTANT REJECT PARITY. A generated mutant corpus over real
 *   pages, with the vendored reference as the authority. Where the kernel
 *   accepts, the reference accepted and the bytes agree; where the reference
 *   rejects, the kernel rejects. Over-strictness is counted and required to be
 *   zero rather than exempted.
 *
 *   THE CRAFTED LADDER. tests/adversarial_gdeflate_pages.h, driven through the
 *   host twin and through the kernel in one process so the two are held to the
 *   same bytes and the same rung.
 *
 *   CAPACITY. Every output bound comes from the caller's dst_capacity and from
 *   nothing the page says, driven at and beyond the format's own 64 KiB tile.
 *
 * TERMINATION IS NOT A SECTION, IT IS HOW EVERY SECTION RUNS. A team that
 * never leaves its round loop does not report a bad status, it holds the
 * launch and the stream behind it, so a plain stream synchronise would inherit
 * the hang and stall the suite. Every decode below is fenced with an event and
 * polled against a wall-clock deadline; an expiry FAILS with a message. The
 * ctest TIMEOUT on this target is the second line of defence, and the process
 * exit is what releases a still-running launch.
 *
 * WHAT THIS IS NOT. It is not the oracle diff over the whole corpus, which is
 * #214's and is in tests/gdeflate_device.cu; the mutant work here starts from
 * real pages but its subject is the REJECT boundary rather than throughput of
 * agreement. Nothing here is timed, and the compute-sanitizer half of the
 * standing gate is parked and unproducible on this host (#127, #258), so what
 * stands in its place is the recorded substitute: this file plus the
 * whole-corpus diff. */
#include "adversarial_gdeflate_pages.h"
#include "cudec.h"
#include "gdeflate_decode.cuh"
#include "require.h"

#include "vendor_rt_test.h"

#include <libdeflate.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using cudec_detail::GDeflateDecodePage;
using cudec_detail::GDeflatePageState;
using cudec_detail::GDeflateReject;
using cudec_detail::GDeflateStatusFor;
using cudec_test::AdversarialPage;
using cudec_test::kGDeflateTileBytes;

constexpr unsigned char kDstPoison = 0xA5;
constexpr int kRunsPerGeometry = 5;
/* Generous against a loaded GPU, tiny against "never": every batch here is a
 * few megabytes of bounded-round pages. */
constexpr double kDeadlineSeconds = 30.0;

/* ---- the corpus generators, the same recurrence the device diff uses ---- */

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

/* The reference's compressor, one page per 64 KiB of input. */
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

/* ---- the reference's verdict on one page ---- */

/* What the reference says about a page, in the only form a parity test may
 * read it. `accepted` is NOT the status alone: tests/oracle_gdeflate.cpp
 * measured and recorded that libdeflate_gdeflate_decompress returns SUCCESS
 * when the bits ran out cleanly rather than when the answer is right, and its
 * own header says it "can be used only in cases where the actual uncompressed
 * size is known". So the status is kept as it came AND the bytes it produced
 * are kept beside it, and the rules below decide which of the two questions
 * each direction of the parity asks. */
struct OracleVerdict {
    bool status_ok;
    std::vector<unsigned char> bytes;
};

/* The zero tail the reference's copy gets, sized the way
 * fuzz/fuzz_gdeflate_page.cpp and tests/gdeflate_departure_lock.cpp size it:
 * on a compressed block the reference has no bound of its own, so it keeps
 * taking a word per round until the OUTPUT fills, and the rounds are bounded
 * by the capacity rather than by what the page had left. Six words per
 * capacity byte plus a page of slack. */
size_t OracleTailBytes(size_t capacity) { return 6u * capacity + 4096u; }

OracleVerdict OracleDecodePage(const std::vector<unsigned char>& page,
                               size_t capacity) {
    OracleVerdict v;
    v.status_ok = false;
    libdeflate_gdeflate_decompressor* d =
        libdeflate_alloc_gdeflate_decompressor();
    if (d == nullptr) {
        return v;
    }
    /* THE REFERENCE GETS A ZERO TAIL AFTER ITS COPY AND THE KERNEL DOES NOT,
     * and this is load-bearing rather than tidy. ENSURE_BITS in the pinned
     * fork reads a 32-bit word with no bound against the end of the page, so
     * asked about a tightly allocated buffer its verdict depends on whatever
     * the allocator left behind - which was MEASURED here as parity counts
     * that moved between two runs of the same binary on the same corpus.
     * Both sides are told the same nbytes and asked about the same bytes;
     * only what lies past the end differs, and on the reference's side it is
     * zeros rather than luck. cudec's own schedule refuses the same read by
     * an explicit bound, which is the distinction docs/MASTERPLAN.md draws
     * between a padding convention and a check. */
    std::vector<unsigned char> padded(page);
    padded.resize(page.size() + OracleTailBytes(capacity), 0);
    libdeflate_gdeflate_in_page in;
    in.data = padded.data();
    in.nbytes = page.size();
    std::vector<unsigned char> out(capacity ? capacity : 1u, 0);
    size_t got = 0;
    const libdeflate_result r = libdeflate_gdeflate_decompress(
        d, &in, 1, out.data(), capacity, &got);
    libdeflate_free_gdeflate_decompressor(d);
    if (r == LIBDEFLATE_SUCCESS && got <= capacity) {
        v.status_ok = true;
        out.resize(got);
        v.bytes = out;
    }
    return v;
}

/* The host twin's verdict on one page: whether it decoded, the rung it took,
 * and the bytes. */
struct TwinVerdict {
    bool ok;
    GDeflateReject rung;
    cudec_status status;
    std::vector<unsigned char> bytes;
};

TwinVerdict TwinDecode(const std::vector<unsigned char>& page, size_t cap) {
    TwinVerdict v;
    GDeflatePageState st;
    uint64_t out_len = 0;
    v.bytes.assign(cap ? cap : 1u, 0);
    v.ok = GDeflateDecodePage(st, page.data(), page.size(), v.bytes.data(), cap,
                              &out_len);
    v.rung = st.s.reject;
    if (v.ok) {
        v.bytes.resize(static_cast<size_t>(out_len));
        v.status = CUDEC_OK;
    } else {
        v.bytes.clear();
        v.status = GDeflateStatusFor(st.s.reject);
    }
    return v;
}

/* ---- the batch, staged as two device arenas ---- */

struct Chunk {
    std::string name;
    std::vector<unsigned char> src;
    size_t dst_capacity;
};

/* One source blob and one destination arena, so a whole launch's byte image is
 * a single download and the determinism compare covers every byte rather than
 * a sample of them. */
struct Batch {
    size_t n = 0;
    std::vector<std::string> names;
    std::vector<size_t> dst_offsets;
    std::vector<size_t> caps;
    unsigned char* d_src_blob = nullptr;
    unsigned char* d_dst_arena = nullptr;
    size_t dst_arena_size = 0;
    const void** d_srcs = nullptr;
    void** d_dsts = nullptr;
    size_t* d_sizes = nullptr;
    size_t* d_caps = nullptr;
    cudec_chunk_result* d_results = nullptr;
};

int BuildBatch(const std::vector<Chunk>& chunks, Batch* b) {
    const size_t n = chunks.size();
    b->n = n;
    b->names.clear();
    b->dst_offsets.clear();
    b->caps.clear();

    size_t src_total = 0;
    size_t dst_total = 0;
    for (size_t i = 0; i < n; i++) {
        src_total += chunks[i].src.size();
        /* Every destination gets at least one byte of arena so a capacity of
         * zero still has a poison region a write past it would disturb. */
        dst_total += chunks[i].dst_capacity ? chunks[i].dst_capacity : 1u;
    }
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_src_blob, src_total ? src_total : 1));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_dst_arena, dst_total));
    b->dst_arena_size = dst_total;

    std::vector<const void*> h_srcs(n);
    std::vector<void*> h_dsts(n);
    std::vector<size_t> h_sizes(n);
    std::vector<size_t> h_caps(n);
    size_t src_off = 0;
    size_t dst_off = 0;
    for (size_t i = 0; i < n; i++) {
        if (!chunks[i].src.empty()) {
            REQUIRE_RT(cudec_rt::memcpy(b->d_src_blob + src_off,
                                        chunks[i].src.data(),
                                        chunks[i].src.size(),
                                        cudec_rt::memcpy_h2d));
        }
        h_srcs[i] = b->d_src_blob + src_off;
        h_sizes[i] = chunks[i].src.size();
        h_dsts[i] = b->d_dst_arena + dst_off;
        h_caps[i] = chunks[i].dst_capacity;
        b->names.push_back(chunks[i].name);
        b->dst_offsets.push_back(dst_off);
        b->caps.push_back(chunks[i].dst_capacity);
        src_off += chunks[i].src.size();
        dst_off += chunks[i].dst_capacity ? chunks[i].dst_capacity : 1u;
    }
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_srcs, n * sizeof(*b->d_srcs)));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_dsts, n * sizeof(*b->d_dsts)));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_sizes, n * sizeof(*b->d_sizes)));
    REQUIRE_RT(cudec_rt::device_malloc(&b->d_caps, n * sizeof(*b->d_caps)));
    REQUIRE_RT(
        cudec_rt::device_malloc(&b->d_results, n * sizeof(*b->d_results)));
    REQUIRE_RT(cudec_rt::memcpy(b->d_srcs, h_srcs.data(),
                                n * sizeof(*b->d_srcs), cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b->d_dsts, h_dsts.data(),
                                n * sizeof(*b->d_dsts), cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b->d_sizes, h_sizes.data(),
                                n * sizeof(*b->d_sizes), cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(b->d_caps, h_caps.data(),
                                n * sizeof(*b->d_caps), cudec_rt::memcpy_h2d));
    return 0;
}

void FreeBatch(Batch* b) {
    (void)cudec_rt::device_free(b->d_src_blob);
    (void)cudec_rt::device_free(b->d_dst_arena);
    (void)cudec_rt::device_free(b->d_srcs);
    (void)cudec_rt::device_free(b->d_dsts);
    (void)cudec_rt::device_free(b->d_sizes);
    (void)cudec_rt::device_free(b->d_caps);
    (void)cudec_rt::device_free(b->d_results);
}

int ResetDeviceState(const Batch& b) {
    REQUIRE_RT(
        cudec_rt::device_memset(b.d_dst_arena, kDstPoison, b.dst_arena_size));
    /* 0xFF is outside the status enumeration, so a page the kernel never
     * reached cannot read as a decoded one. */
    REQUIRE_RT(cudec_rt::device_memset(b.d_results, 0xFF,
                                       b.n * sizeof(*b.d_results)));
    return 0;
}

int Download(const Batch& b, std::vector<unsigned char>* dst,
             std::vector<cudec_chunk_result>* results) {
    dst->assign(b.dst_arena_size, 0);
    results->assign(b.n, cudec_chunk_result{});
    REQUIRE_RT(cudec_rt::memcpy(dst->data(), b.d_dst_arena, b.dst_arena_size,
                                cudec_rt::memcpy_d2h));
    REQUIRE_RT(cudec_rt::memcpy(results->data(), b.d_results,
                                b.n * sizeof(*b.d_results),
                                cudec_rt::memcpy_d2h));
    return 0;
}

/* ---- the watchdog ---- */

/* Waits on an EVENT and not on the stream: a launch that does not finish must
 * be reported, never waited on. On expiry the launch is still resident, and
 * returning (and exiting) leaves the teardown to the driver - synchronising
 * or destroying the stream here would block on the very hang being reported. */
int WaitWithWatchdog(cudec_rt::stream_t stream, const char* what) {
    cudec_rt::event_t finished;
    REQUIRE_RT(cudec_rt::event_create_untimed(&finished));
    REQUIRE_RT(cudec_rt::event_record(finished, stream));
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        const cudec_rt::error_t query = cudec_rt::event_query(finished);
        if (query == cudec_rt::success) {
            break;
        }
        REQUIRE_CTX(query == cudec_rt::error_not_ready,
                    "%s: event query failed: %s", what,
                    cudec_rt::error_string(query));
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count();
        REQUIRE_CTX(elapsed < kDeadlineSeconds,
                    "%s did not complete within %.0f s - every page in it is a "
                    "bounded-round page, so this is a non-terminating decode",
                    what, kDeadlineSeconds);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE_RT(cudec_rt::event_destroy(finished));
    return 0;
}

/* The shipped entry over the whole batch, fenced. */
int RunShippedEntry(const Batch& b, const char* what) {
    REQUIRE(ResetDeviceState(b) == 0);
    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    REQUIRE(cudec_gdeflate_decompress_batch(b.d_srcs, b.d_sizes, b.d_dsts,
                                            b.d_caps, b.n, b.d_results,
                                            cudec_rt::abi_stream(stream)) ==
            CUDEC_OK);
    REQUIRE(WaitWithWatchdog(stream, what) == 0);
    REQUIRE_RT(cudec_rt::stream_destroy(stream));
    return 0;
}

struct Geometry {
    const char* name;
    unsigned blocks;
    unsigned threads;
};

/* Grid and block shapes that change the page-to-team mapping in every
 * direction the kernel admits: one team walking the whole batch through the
 * grid-stride loop, block sizes below the launch bound, a grid far larger than
 * the batch, and the shipped sizing. Block sizes stay multiples of the wave
 * width and within __launch_bounds__. */
std::vector<Geometry> Geometries(size_t chunk_count) {
    std::vector<Geometry> g;
    g.push_back({"1x32", 1, 32});
    g.push_back({"3x64", 3, 64});
    g.push_back({"17x96", 17, 96});
    g.push_back({"chunks x128", static_cast<unsigned>(chunk_count), 128});
    g.push_back({"shipped", cudec_detail::decode_grid_blocks(chunk_count),
                 cudec_detail::kBlockThreads});
    return g;
}

/* A geometry the public ABI exposes no knob for is reached by launching the
 * internal kernel directly. */
int RunGeometry(const Batch& b, const Geometry& g, const char* what) {
    REQUIRE(ResetDeviceState(b) == 0);
    cudec_rt::stream_t stream;
    REQUIRE_RT(cudec_rt::stream_create(&stream));
    cudec_detail::gdeflate_decode_batch<cudec_detail::kCudaWaveSize>
        <<<g.blocks, g.threads, 0, stream>>>(b.d_srcs, b.d_sizes, b.d_dsts,
                                             b.d_caps, b.n, b.d_results);
    REQUIRE_RT(cudec_rt::get_last_error());
    REQUIRE(WaitWithWatchdog(stream, what) == 0);
    REQUIRE_RT(cudec_rt::stream_destroy(stream));
    return 0;
}

/* The same batch as three sub-batches on three concurrent streams: page k is
 * decoded by a different team of a different launch than in any other
 * geometry, and the sub-batches complete in an order this test does not
 * control. Per-page independence means the output must not notice. */
int RunSplitStreams(const Batch& b, unsigned stream_count, const char* what) {
    REQUIRE(ResetDeviceState(b) == 0);
    std::vector<cudec_rt::stream_t> streams(stream_count);
    for (unsigned s = 0; s < stream_count; s++) {
        REQUIRE_RT(cudec_rt::stream_create(&streams[s]));
    }
    const size_t per = (b.n + stream_count - 1) / stream_count;
    for (unsigned s = 0; s < stream_count; s++) {
        const size_t begin = s * per;
        if (begin >= b.n) {
            break;
        }
        const size_t count = (begin + per <= b.n) ? per : b.n - begin;
        /* cudec_chunk_result is 16 bytes, so an element offset keeps the
         * 16-byte alignment the ABI requires of d_results. */
        REQUIRE(cudec_gdeflate_decompress_batch(
                    b.d_srcs + begin, b.d_sizes + begin, b.d_dsts + begin,
                    b.d_caps + begin, count, b.d_results + begin,
                    cudec_rt::abi_stream(streams[s])) == CUDEC_OK);
    }
    for (unsigned s = 0; s < stream_count; s++) {
        REQUIRE(WaitWithWatchdog(streams[s], what) == 0);
        REQUIRE_RT(cudec_rt::stream_destroy(streams[s]));
    }
    return 0;
}

int CompareAgainstReference(const Batch& b, const std::string& context,
                            const std::vector<unsigned char>& ref_dst,
                            const std::vector<cudec_chunk_result>& ref_results,
                            const std::vector<unsigned char>& dst,
                            const std::vector<cudec_chunk_result>& results) {
    for (size_t i = 0; i < b.n; i++) {
        REQUIRE_CTX(results[i].status == ref_results[i].status,
                    "%s: page %zu (%s) status %d, reference %d",
                    context.c_str(), i, b.names[i].c_str(),
                    static_cast<int>(results[i].status),
                    static_cast<int>(ref_results[i].status));
        REQUIRE_CTX(results[i].bytes_written == ref_results[i].bytes_written,
                    "%s: page %zu (%s) wrote %llu bytes, reference %llu",
                    context.c_str(), i, b.names[i].c_str(),
                    static_cast<unsigned long long>(results[i].bytes_written),
                    static_cast<unsigned long long>(
                        ref_results[i].bytes_written));
        REQUIRE_CTX(results[i].reserved == ref_results[i].reserved,
                    "%s: page %zu reserved word drifted", context.c_str(), i);
    }
    /* The whole arena, including the bytes a refused page happened to write
     * before refusing. The contract does not promise those, so this asserts
     * more than the contract - deliberately, because a geometry-dependent
     * write on the reject path is exactly what the promised region would
     * hide. */
    REQUIRE_CTX(dst.size() == ref_dst.size(), "%s: arena size drifted",
                context.c_str());
    REQUIRE_CTX(equal_bytes(dst.data(), ref_dst.data(), dst.size()),
                "%s: destination arena differs from the reference decode",
                context.c_str());
    return 0;
}

/* ---- section 1: determinism ---- */

int Determinism(const std::vector<Chunk>& chunks, size_t* decoded_out) {
    REQUIRE(!chunks.empty());
    Batch b;
    REQUIRE(BuildBatch(chunks, &b) == 0);

    std::vector<unsigned char> ref_dst;
    std::vector<cudec_chunk_result> ref_results;
    REQUIRE(RunShippedEntry(b, "the determinism reference decode") == 0);
    REQUIRE(Download(b, &ref_dst, &ref_results) == 0);

    size_t decoded = 0;
    size_t refused = 0;
    for (size_t i = 0; i < b.n; i++) {
        if (ref_results[i].status == CUDEC_OK) {
            decoded++;
        } else {
            refused++;
        }
    }
    /* A corpus that decoded nothing would compare poison against poison and
     * pass vacuously; one that refused nothing would leave the reject path
     * out of the geometry sweep entirely, and the reject path is where a
     * geometry-dependent write is most likely to live. */
    REQUIRE(decoded > 0);
    REQUIRE(refused > 0);

    std::vector<unsigned char> dst;
    std::vector<cudec_chunk_result> results;
    const std::vector<Geometry> geometries = Geometries(b.n);
    for (size_t gi = 0; gi < geometries.size(); gi++) {
        for (int run = 0; run < kRunsPerGeometry; run++) {
            const std::string what = std::string("gdeflate/") +
                                     geometries[gi].name + "/run-" +
                                     std::to_string(run);
            REQUIRE(RunGeometry(b, geometries[gi], what.c_str()) == 0);
            REQUIRE(Download(b, &dst, &results) == 0);
            REQUIRE(CompareAgainstReference(b, what, ref_dst, ref_results, dst,
                                            results) == 0);
        }
    }
    for (int run = 0; run < kRunsPerGeometry; run++) {
        const std::string what =
            std::string("gdeflate/3-streams/run-") + std::to_string(run);
        REQUIRE(RunSplitStreams(b, 3, what.c_str()) == 0);
        REQUIRE(Download(b, &dst, &results) == 0);
        REQUIRE(CompareAgainstReference(b, what, ref_dst, ref_results, dst,
                                        results) == 0);
    }

    FreeBatch(&b);
    *decoded_out = decoded;
    std::printf("gdeflate_gate_gpu: determinism: %zu pages (%zu decoded, %zu "
                "refused) bit-identical across %zu launch configurations x %d "
                "runs, 0 mismatches\n",
                b.n, decoded, refused, geometries.size() + 1u,
                kRunsPerGeometry);
    return 0;
}

/* ---- section 2: two-directional mutant reject parity ---- */

/* Truncations at fixed fractions and at every byte count from the end that a
 * word boundary admits, plus seeded single-bit flips and byte overwrites. The
 * placement is the harness's and decides only WHERE a mutation lands; the
 * verdict on every mutant comes from the reference, so a bad placement can
 * only make a mutation less targeted - never turn a rejected page into an
 * accepted one or the other way round. */
std::vector<std::vector<unsigned char> > MutatePage(
    const std::vector<unsigned char>& page, unsigned seed) {
    std::vector<std::vector<unsigned char> > out;
    if (page.size() < 8) {
        return out;
    }
    for (int frac = 1; frac < 8; frac++) {
        size_t keep = page.size() * static_cast<size_t>(frac) / 8u;
        out.push_back(std::vector<unsigned char>(page.begin(),
                                                 page.begin() + keep));
    }
    for (size_t off = 1; off <= 8; off++) {
        if (off < page.size()) {
            out.push_back(std::vector<unsigned char>(
                page.begin(), page.end() - static_cast<long>(off)));
        }
    }
    Lcg lcg(seed);
    for (int i = 0; i < 48; i++) {
        std::vector<unsigned char> m = page;
        const size_t at =
            (static_cast<size_t>(lcg.Next()) << 16 | lcg.Next()) % m.size();
        m[at] ^= static_cast<unsigned char>(1u << (lcg.Next() % 8u));
        out.push_back(m);
    }
    for (int i = 0; i < 24; i++) {
        std::vector<unsigned char> m = page;
        const size_t at =
            (static_cast<size_t>(lcg.Next()) << 16 | lcg.Next()) % m.size();
        m[at] = static_cast<unsigned char>(lcg.Next() & 0xFFu);
        out.push_back(m);
    }
    /* The first thirty-two words are the priming round, and the block header
     * rides the first of them: a mutation there is the one a truncation and a
     * uniform bit flip both under-sample. */
    for (size_t at = 0; at < 8 && at < page.size(); at++) {
        for (int bit = 0; bit < 8; bit++) {
            std::vector<unsigned char> m = page;
            m[at] ^= static_cast<unsigned char>(1u << bit);
            out.push_back(m);
        }
    }
    return out;
}

struct ParityCounts {
    size_t mutants;
    size_t oracle_rejects;
    size_t agreed_decodes;
    size_t oracle_accepted_other_bytes;
    size_t fail_opens;
    size_t over_strict;
    /* Refusals of a page the reference reproduces, on a rung already recorded
     * as a strictness departure. Counted and printed per rung rather than
     * ignored: an exemption whose population nobody prints is one nobody
     * notices growing. */
    size_t recorded_departures;
    size_t departure_hits[cudec_detail::kGDeflateRejectCount];
};

/* An over-strict reject is one this decoder makes on a page the reference
 * decodes to the bytes the unmutated page gives. Whether it is a DEFECT is
 * decided by a predicate in src/ and not by this file:
 * GDeflateRejectIsDeclaredDeparture names the rungs on which this decoder is
 * deliberately stricter than the reference, and
 * tests/gdeflate_departure_lock.cpp is what makes each of those declarations
 * cost a page that executes it. */
bool IsDeclaredDeparture(GDeflateReject rung) {
    return cudec_detail::GDeflateRejectIsDeclaredDeparture(rung);
}

/* ONE RUNG BELONGS IN NEITHER PLACE YET, AND THIS IS THE WHOLE OF THE
 * CARVE-OUT. Removing one byte from the end of a real page makes its size stop
 * being a whole number of 32-bit words; this decoder refuses that on
 * kGDeflateRejectPagePartialWord and the reference decodes it to the original
 * bytes, because the words it never needed are the ones that went missing. So
 * it is a strictness departure the two decoders genuinely have, and
 * GDeflateRejectIsDeclaredDeparture does not name it. Whether the predicate
 * should - which loosens the differential fuzz target's trap and is a change
 * to src/ rather than to a test - is issue #453, and it is not decided here.
 *
 * IT IS WRITTEN AS ONE NAMED RUNG RATHER THAN AS A SET, and the count it
 * absorbs is printed on every run, so it cannot grow into the allowlist the
 * departure lock exists against. Any other rung refusing a page the reference
 * reproduces still fails this gate. */
bool IsRecordedDeparture(GDeflateReject rung) {
    return IsDeclaredDeparture(rung) ||
           rung == cudec_detail::kGDeflateRejectPagePartialWord;
}

int MutantParity(ParityCounts* counts) {
    counts->mutants = 0;
    counts->oracle_rejects = 0;
    counts->agreed_decodes = 0;
    counts->oracle_accepted_other_bytes = 0;
    counts->fail_opens = 0;
    counts->over_strict = 0;
    counts->recorded_departures = 0;
    for (uint32_t r = 0; r < cudec_detail::kGDeflateRejectCount; r++) {
        counts->departure_hits[r] = 0;
    }

    const int levels[] = {0, 1, 6, 12};
    std::vector<std::vector<unsigned char> > bases;
    for (size_t l = 0; l < sizeof(levels) / sizeof(levels[0]); l++) {
        std::vector<std::vector<unsigned char> > pages;
        REQUIRE(CompressPages(levels[l], MixedEntropy(4, 40000), &pages));
        REQUIRE(pages.size() == 1);
        bases.push_back(pages[0]);
        REQUIRE(CompressPages(levels[l], ShortRepeats(5, 40000), &pages));
        REQUIRE(pages.size() == 1);
        bases.push_back(pages[0]);
    }

    std::vector<Chunk> chunks;
    /* Which base each entry was mutated from, carried beside the batch rather
     * than parsed back out of a name: the over-strictness rule below needs the
     * unmutated page's bytes, and reconstructing an index from a string is a
     * place for the two to drift apart. */
    std::vector<size_t> base_of;
    for (size_t bi = 0; bi < bases.size(); bi++) {
        /* The unmutated page rides along, so a run in which every mutant was
         * refused for a reason unrelated to its mutation is visible. */
        Chunk pristine;
        pristine.name = "base-" + std::to_string(bi);
        pristine.src = bases[bi];
        pristine.dst_capacity = kGDeflateTileBytes;
        chunks.push_back(pristine);
        base_of.push_back(bi);
        const std::vector<std::vector<unsigned char> > mutants =
            MutatePage(bases[bi], static_cast<unsigned>(17 + bi));
        for (size_t mi = 0; mi < mutants.size(); mi++) {
            Chunk c;
            c.name = "base-" + std::to_string(bi) + "/mutant-" +
                     std::to_string(mi);
            c.src = mutants[mi];
            c.dst_capacity = kGDeflateTileBytes;
            chunks.push_back(c);
            base_of.push_back(bi);
        }
    }

    /* Each base decoded once by the reference, so the round-trip test below
     * compares against the page's own truth rather than re-decoding it per
     * mutant. */
    std::vector<OracleVerdict> base_truth;
    for (size_t bi = 0; bi < bases.size(); bi++) {
        base_truth.push_back(OracleDecodePage(bases[bi], kGDeflateTileBytes));
        REQUIRE(base_truth[bi].status_ok);
    }

    Batch b;
    REQUIRE(BuildBatch(chunks, &b) == 0);
    REQUIRE(RunShippedEntry(b, "the mutant parity batch") == 0);
    std::vector<unsigned char> dst;
    std::vector<cudec_chunk_result> results;
    REQUIRE(Download(b, &dst, &results) == 0);

    for (size_t i = 0; i < b.n; i++) {
        const OracleVerdict oracle =
            OracleDecodePage(chunks[i].src, chunks[i].dst_capacity);
        const bool cudec_ok = results[i].status == CUDEC_OK;
        const unsigned char* out = dst.data() + b.dst_offsets[i];
        counts->mutants++;

        if (!oracle.status_ok) {
            /* THE SEVERE DIRECTION. Whenever the reference rejects, cudec
             * rejects: a stream the authority refuses and this decoder accepts
             * is a fail-open, and for a checksum-less format it is the failure
             * nothing downstream would notice. */
            counts->oracle_rejects++;
            if (cudec_ok) {
                counts->fail_opens++;
                REQUIRE_CTX(false,
                            "%s: FAIL-OPEN - the reference rejects and the "
                            "kernel accepted %llu bytes",
                            chunks[i].name.c_str(),
                            static_cast<unsigned long long>(
                                results[i].bytes_written));
            }
            REQUIRE_CTX(results[i].bytes_written == 0, "%s: refused with %llu "
                        "bytes_written",
                        chunks[i].name.c_str(),
                        static_cast<unsigned long long>(
                            results[i].bytes_written));
            continue;
        }

        if (cudec_ok) {
            /* Both accepted, so the bytes are held to each other. */
            REQUIRE_CTX(results[i].bytes_written == oracle.bytes.size(),
                        "%s: kernel wrote %llu bytes, reference %zu",
                        chunks[i].name.c_str(),
                        static_cast<unsigned long long>(
                            results[i].bytes_written),
                        oracle.bytes.size());
            REQUIRE_CTX(std::memcmp(out, oracle.bytes.data(),
                                    oracle.bytes.size()) == 0,
                        "%s: both accepted and the bytes differ",
                        chunks[i].name.c_str());
            for (size_t j = oracle.bytes.size(); j < b.caps[i]; j++) {
                REQUIRE_CTX(out[j] == kDstPoison, "%s: poison at %zu",
                            chunks[i].name.c_str(), j);
            }
            counts->agreed_decodes++;
            continue;
        }

        /* The reference accepted and the kernel did not. Whether that is
         * over-strictness depends on what the reference produced, and the
         * distinction is the recorded property of this oracle rather than a
         * convenience: SUCCESS here means the bits ran out cleanly, not that
         * the answer is right (tests/oracle_gdeflate.cpp). Where the mutated
         * page round-trips to the ORIGINAL bytes, the mutation was one the
         * format tolerates and refusing it is over-strict. Where it does not,
         * neither implementation reproduced anything, and requiring this
         * decoder to accept would be requiring it to reproduce the
         * reference's guess. */
        const OracleVerdict& truth = base_truth[base_of[i]];
        const bool round_trips =
            truth.bytes.size() == oracle.bytes.size() &&
            std::memcmp(truth.bytes.data(), oracle.bytes.data(),
                        oracle.bytes.size()) == 0;
        if (round_trips) {
            /* The rung comes from the twin, because the ABI status the kernel
             * reports collapses several rungs onto one value and the
             * departure predicate is written over rungs. The kernel is held
             * to the twin's status a line further down, so reading the rung
             * off the twin is not reading it off a different decoder. */
            const TwinVerdict twin =
                TwinDecode(chunks[i].src, chunks[i].dst_capacity);
            REQUIRE_CTX(!twin.ok,
                        "%s: the twin decoded a page the kernel refused",
                        chunks[i].name.c_str());
            REQUIRE_CTX(results[i].status == twin.status,
                        "%s: the kernel answered %d and the twin %d",
                        chunks[i].name.c_str(),
                        static_cast<int>(results[i].status),
                        static_cast<int>(twin.status));
            if (!IsRecordedDeparture(twin.rung)) {
                counts->over_strict++;
                REQUIRE_CTX(false,
                            "%s: OVER-STRICT - the reference decoded it to the "
                            "bytes the unmutated page gives, and this decoder "
                            "refused on rung %d, which is neither declared in "
                            "src/gdeflate_schedule.h nor recorded on #453",
                            chunks[i].name.c_str(),
                            static_cast<int>(twin.rung));
            }
            counts->recorded_departures++;
            counts->departure_hits[twin.rung]++;
            REQUIRE_CTX(results[i].bytes_written == 0,
                        "%s: refused with %llu bytes_written",
                        chunks[i].name.c_str(),
                        static_cast<unsigned long long>(
                            results[i].bytes_written));
            continue;
        }
        counts->oracle_accepted_other_bytes++;
        REQUIRE_CTX(results[i].bytes_written == 0,
                    "%s: refused with %llu bytes_written",
                    chunks[i].name.c_str(),
                    static_cast<unsigned long long>(results[i].bytes_written));
    }

    FreeBatch(&b);
    REQUIRE(counts->fail_opens == 0);
    REQUIRE(counts->over_strict == 0);
    /* A run in which the reference rejected nothing would prove nothing about
     * the severe direction, and one in which nothing agreed would mean the
     * corpus never decoded. */
    REQUIRE(counts->oracle_rejects > 0);
    REQUIRE(counts->agreed_decodes > 0);
    std::printf("gdeflate_gate_gpu: mutant parity: %zu pages, %zu the "
                "reference rejects and the kernel rejects too, %zu both accept "
                "with identical bytes, %zu the reference accepts as other "
                "bytes, %zu the kernel refuses on a rung already recorded as a "
                "strictness departure; 0 fail-opens, 0 undeclared over-strict "
                "rejects\n",
                counts->mutants, counts->oracle_rejects,
                counts->agreed_decodes, counts->oracle_accepted_other_bytes,
                counts->recorded_departures);
    /* Printed per rung rather than as one total, because the exemption's
     * population is the thing a reader has to be able to watch: a rung that
     * starts absorbing hundreds of mutants is an exemption that has quietly
     * become the rule. */
    for (uint32_t r = 0; r < cudec_detail::kGDeflateRejectCount; r++) {
        if (counts->departure_hits[r] != 0) {
            std::printf("gdeflate_gate_gpu:   rung %u absorbed %zu of them, "
                        "%s\n",
                        r, counts->departure_hits[r],
                        IsDeclaredDeparture(
                            static_cast<GDeflateReject>(r))
                            ? "declared in src/gdeflate_schedule.h and executed "
                              "by tests/gdeflate_departure_lock.cpp"
                            : "recorded on issue #453 and declared nowhere in "
                              "src/");
        }
    }
    return 0;
}

/* ---- section 3: the crafted ladder, twin and kernel on the same bytes ---- */

int CraftedLadder(size_t* pages_out) {
    std::vector<AdversarialPage> pages;
    REQUIRE(cudec_test::MakeAdversarialGDeflatePages(&pages));
    REQUIRE(!pages.empty());

    std::vector<Chunk> chunks;
    for (size_t i = 0; i < pages.size(); i++) {
        Chunk c;
        c.name = pages[i].name;
        c.src = pages[i].page;
        c.dst_capacity = pages[i].dst_capacity;
        chunks.push_back(c);
    }
    Batch b;
    REQUIRE(BuildBatch(chunks, &b) == 0);
    REQUIRE(RunShippedEntry(b, "the crafted ladder batch") == 0);
    std::vector<unsigned char> dst;
    std::vector<cudec_chunk_result> results;
    REQUIRE(Download(b, &dst, &results) == 0);

    for (size_t i = 0; i < pages.size(); i++) {
        const TwinVerdict twin =
            TwinDecode(pages[i].page, pages[i].dst_capacity);
        /* The rung the fixture declares, held against the twin first: a page
         * that stopped reaching its branch would otherwise take the kernel
         * with it and both would agree on the wrong thing. */
        REQUIRE_CTX(!twin.ok, "%s: the twin decoded a page that must refuse",
                    pages[i].name.c_str());
        REQUIRE_CTX(twin.rung == pages[i].rung,
                    "%s: the twin took rung %d, the fixture declares %d",
                    pages[i].name.c_str(), static_cast<int>(twin.rung),
                    static_cast<int>(pages[i].rung));
        REQUIRE_CTX(results[i].status == twin.status,
                    "%s: the kernel answered %d, the twin's rung maps to %d",
                    pages[i].name.c_str(),
                    static_cast<int>(results[i].status),
                    static_cast<int>(twin.status));
        REQUIRE_CTX(results[i].bytes_written == 0,
                    "%s: refused with %llu bytes_written",
                    pages[i].name.c_str(),
                    static_cast<unsigned long long>(results[i].bytes_written));
        REQUIRE_CTX(results[i].reserved == 0, "%s: reserved word not zero",
                    pages[i].name.c_str());
    }
    FreeBatch(&b);
    *pages_out = pages.size();
    std::printf("gdeflate_gate_gpu: crafted ladder: %zu hostile pages, each "
                "refused on the device with the status the host twin's rung "
                "maps to, bytes_written 0\n",
                pages.size());
    return 0;
}

/* ---- section 4: capacity, at and beyond the tile bound ---- */

/* The rule under test is the anti-pattern rule: the output is sized against
 * the caller's dst_capacity and never against anything the page declares or
 * the format conventionally allows. A real page whose output is known is
 * driven at capacities either side of its own length and either side of the
 * 64 KiB tile, and the answer must follow the ARGUMENT rather than the
 * convention. */
int CapacityAdversarials(size_t* cases_out) {
    std::vector<std::vector<unsigned char> > pages;
    const std::vector<unsigned char> source = ShortRepeats(7, kGDeflateTileBytes);
    REQUIRE(CompressPages(6, source, &pages));
    REQUIRE(pages.size() == 1);
    const size_t produced = source.size();

    struct Case {
        const char* name;
        size_t capacity;
    };
    const Case cases[] = {
        {"capacity 0", 0},
        {"capacity 1", 1},
        {"capacity one byte short", produced - 1u},
        {"capacity exactly the output", produced},
        {"capacity one byte over", produced + 1u},
        {"capacity the tile bound", kGDeflateTileBytes},
        {"capacity past the tile bound", kGDeflateTileBytes + 4096u},
    };
    const size_t n = sizeof(cases) / sizeof(cases[0]);

    std::vector<Chunk> chunks;
    for (size_t i = 0; i < n; i++) {
        Chunk c;
        c.name = cases[i].name;
        c.src = pages[0];
        c.dst_capacity = cases[i].capacity;
        chunks.push_back(c);
    }
    Batch b;
    REQUIRE(BuildBatch(chunks, &b) == 0);
    REQUIRE(RunShippedEntry(b, "the capacity batch") == 0);
    std::vector<unsigned char> dst;
    std::vector<cudec_chunk_result> results;
    REQUIRE(Download(b, &dst, &results) == 0);

    size_t refused = 0;
    size_t decoded = 0;
    for (size_t i = 0; i < n; i++) {
        const TwinVerdict twin = TwinDecode(pages[0], cases[i].capacity);
        REQUIRE_CTX(results[i].status == twin.status,
                    "%s: the kernel answered %d, the twin %d", cases[i].name,
                    static_cast<int>(results[i].status),
                    static_cast<int>(twin.status));
        const unsigned char* out = dst.data() + b.dst_offsets[i];
        if (cases[i].capacity < produced) {
            REQUIRE_CTX(results[i].status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                        "%s: a capacity below the output must be answered "
                        "with the output-too-small status, got %d",
                        cases[i].name,
                        static_cast<int>(results[i].status));
            REQUIRE_CTX(results[i].bytes_written == 0, "%s", cases[i].name);
            refused++;
            continue;
        }
        REQUIRE_CTX(results[i].status == CUDEC_OK,
                    "%s: a capacity at or above the output must decode, got %d",
                    cases[i].name, static_cast<int>(results[i].status));
        REQUIRE_CTX(results[i].bytes_written == produced, "%s: wrote %llu of "
                    "%zu",
                    cases[i].name,
                    static_cast<unsigned long long>(results[i].bytes_written),
                    produced);
        REQUIRE_CTX(std::memcmp(out, source.data(), produced) == 0, "%s",
                    cases[i].name);
        /* The bytes the caller lent beyond the output are the caller's, and a
         * decoder that wrote into them because the format says a tile is
         * 64 KiB would be sizing against a convention. */
        for (size_t j = produced; j < b.caps[i]; j++) {
            REQUIRE_CTX(out[j] == kDstPoison, "%s: poison at %zu",
                        cases[i].name, j);
        }
        decoded++;
    }
    FreeBatch(&b);
    REQUIRE(refused > 0);
    REQUIRE(decoded > 0);
    *cases_out = n;
    std::printf("gdeflate_gate_gpu: capacity: %zu cases either side of the "
                "output and of the 64 KiB tile, %zu refused as too small and "
                "%zu decoded, every refusal at bytes_written 0 and every "
                "decode leaving the lent bytes poisoned\n",
                n, refused, decoded);
    return 0;
}

/* The determinism corpus: real pages at several levels, plus the crafted
 * hostile pages, so the geometry sweep covers the reject path as well as the
 * decode path. */
int MakeDeterminismChunks(std::vector<Chunk>* chunks) {
    const int levels[] = {0, 1, 6, 12};
    for (size_t l = 0; l < sizeof(levels) / sizeof(levels[0]); l++) {
        std::vector<std::vector<unsigned char> > pages;
        REQUIRE(CompressPages(levels[l], MixedEntropy(4, 3u * kGDeflateTileBytes),
                              &pages));
        for (size_t i = 0; i < pages.size(); i++) {
            Chunk c;
            c.name = "mixed/level-" + std::to_string(levels[l]) + "/page-" +
                     std::to_string(i);
            c.src = pages[i];
            c.dst_capacity = kGDeflateTileBytes;
            chunks->push_back(c);
        }
        REQUIRE(CompressPages(levels[l], ShortRepeats(5, 2u * kGDeflateTileBytes),
                              &pages));
        for (size_t i = 0; i < pages.size(); i++) {
            Chunk c;
            c.name = "repeats/level-" + std::to_string(levels[l]) + "/page-" +
                     std::to_string(i);
            c.src = pages[i];
            c.dst_capacity = kGDeflateTileBytes;
            chunks->push_back(c);
        }
    }
    std::vector<AdversarialPage> hostile;
    REQUIRE(cudec_test::MakeAdversarialGDeflatePages(&hostile));
    for (size_t i = 0; i < hostile.size(); i++) {
        Chunk c;
        c.name = "hostile/" + hostile[i].name;
        c.src = hostile[i].page;
        c.dst_capacity = hostile[i].dst_capacity;
        chunks->push_back(c);
    }
    return 0;
}

}  // namespace

int main() {
    std::vector<Chunk> determinism_chunks;
    if (MakeDeterminismChunks(&determinism_chunks) != 0) {
        return 1;
    }
    size_t decoded = 0;
    if (Determinism(determinism_chunks, &decoded) != 0) {
        return 1;
    }
    ParityCounts counts;
    if (MutantParity(&counts) != 0) {
        return 1;
    }
    size_t hostile = 0;
    if (CraftedLadder(&hostile) != 0) {
        return 1;
    }
    size_t capacity_cases = 0;
    if (CapacityAdversarials(&capacity_cases) != 0) {
        return 1;
    }
    std::printf("gdeflate_gate_gpu: every batch above completed inside the "
                "%.0f s watchdog, so no hostile page held a launch\n",
                kDeadlineSeconds);
    std::printf("PASS: gdeflate_gate_gpu\n");
    return 0;
}
