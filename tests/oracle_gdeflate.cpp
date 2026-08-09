/* The GDeflate oracle proving itself before any GDeflate decoder exists
 * (issue #168): the pinned fork builds, both directions link and are
 * callable, a stream it produced comes back byte-identical, and the two
 * verdicts a later parity test will read are shown to be read from the
 * reference rather than assumed.
 *
 * No cudec code is under test here. This is the reference the whole M4 ladder
 * will be held to, checked for being present and honest at all.
 *
 * It also pins three facts about HOW the reference answers, because each one
 * decides how a parity test has to be written and each one is easy to get
 * wrong in the direction that reads green:
 *
 *   1. A wrong stream can return SUCCESS. `libdeflate_gdeflate_decompress`
 *      hands back an actual output size, and its own header says it "can be
 *      used only in cases where the actual uncompressed size is known" - so
 *      SUCCESS means "the bits ran out cleanly", not "you got what you put
 *      in". A parity test that reads the status and not the size has a hole
 *      exactly where a corrupt stream lands.
 *   2. Truncating a page does not reliably refuse it. GDeflate carries no
 *      checksum anywhere (docs/MASTERPLAN.md section 11.4) and the decoder
 *      stops when the output is full, so trailing compressed bytes it never
 *      needed can be cut away with the output still byte-identical. So
 *      truncation is not the lever that proves a verdict was read, the way
 *      it is for the Snappy oracle.
 *   3. Level 0 is the one level whose block type the source guarantees. The
 *      fork's own libdeflate.h defines it as "create a valid stream, but only
 *      emit uncompressed blocks", which is why section 11.8 puts it in the
 *      sweep, and it is measurable from the outside as output no smaller than
 *      input.
 *
 * The reject ladder, the dossier probes and the corpus are #183, #170 and
 * their siblings. This is the rung that says the oracle is there. */
#include "require.h"

#include <libdeflate.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/* 64 KiB, the GDeflate page, fixed by the format rather than by this test
 * (docs/MASTERPLAN.md section 11.2). Page counts below are derived from it so
 * that a pin whose geometry moved would red here rather than pass quietly. */
constexpr size_t kPageBytes = 65536;

/* Compressible but not trivially so: a run-length pattern would be carried by
 * a single block type and would prove nothing about the rest of the path. */
std::vector<unsigned char> MakeInput(size_t n) {
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<unsigned char>((i / 7) ^ (i >> 3));
    }
    return v;
}

struct Compressed {
    std::vector<unsigned char> bytes;
    std::vector<libdeflate_gdeflate_out_page> pages;
    size_t total;
};

/* Compress at `level`, with the page split the reference itself asks for:
 * compress_bound reports both the worst-case size and the page count, and the
 * page count is the thing the caller may not invent. */
bool Compress(int level, const std::vector<unsigned char>& in, Compressed* out) {
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
    out->bytes.assign(bound, 0);
    out->pages.assign(npages, libdeflate_gdeflate_out_page{});
    const size_t per = bound / npages;
    for (size_t i = 0; i < npages; i++) {
        out->pages[i].data = out->bytes.data() + i * per;
        out->pages[i].nbytes = per;
    }
    out->total = libdeflate_gdeflate_compress(c, in.data(), in.size(),
                                              out->pages.data(), npages);
    libdeflate_free_gdeflate_compressor(c);
    return out->total != 0;
}

/* The in-pages the decompressor reads, built from what the compressor
 * actually wrote into each out-page rather than from the sizes it was
 * offered. */
std::vector<libdeflate_gdeflate_in_page> InPages(const Compressed& c) {
    std::vector<libdeflate_gdeflate_in_page> in(c.pages.size());
    for (size_t i = 0; i < c.pages.size(); i++) {
        in[i].data = c.pages[i].data;
        in[i].nbytes = c.pages[i].nbytes;
    }
    return in;
}

}  // namespace

int main() {
    /* ---- Round trip at the four levels section 11.8 names, each with the
     * page count derived from the input rather than assumed. */
    const int levels[] = {0, 1, 6, 12};
    for (int level : levels) {
        const std::vector<unsigned char> original = MakeInput(300000);
        Compressed c;
        REQUIRE_CTX(Compress(level, original, &c), "level %d", level);

        const size_t want_pages =
            (original.size() + kPageBytes - 1) / kPageBytes;
        REQUIRE_CTX(c.pages.size() == want_pages,
                    "level %d: %zu pages for %zu bytes, want %zu", level,
                    c.pages.size(), original.size(), want_pages);

        libdeflate_gdeflate_decompressor* d =
            libdeflate_alloc_gdeflate_decompressor();
        REQUIRE(d != nullptr);
        std::vector<libdeflate_gdeflate_in_page> in = InPages(c);
        std::vector<unsigned char> back(original.size(), 0);
        size_t got = 0;
        const libdeflate_result r = libdeflate_gdeflate_decompress(
            d, in.data(), in.size(), back.data(), back.size(), &got);
        REQUIRE_CTX(r == LIBDEFLATE_SUCCESS, "level %d: result %d", level,
                    static_cast<int>(r));
        REQUIRE_CTX(got == original.size(), "level %d: %zu of %zu bytes",
                    level, got, original.size());
        REQUIRE_CTX(equal_bytes(back.data(), original.data(), original.size()),
                    "level %d", level);
        libdeflate_free_gdeflate_decompressor(d);

        /* Level 0 emits uncompressed blocks by construction, which is the
         * only block-type guarantee the source gives and the reason the sweep
         * carries the level at all. Measured from the outside: a stream made
         * only of stored blocks cannot be smaller than what went in. */
        if (level == 0) {
            REQUIRE_CTX(c.total >= original.size(),
                        "level 0 produced %zu bytes from %zu - it is not "
                        "emitting stored blocks",
                        c.total, original.size());
        } else {
            REQUIRE_CTX(c.total < original.size(), "level %d: %zu from %zu",
                        level, c.total, original.size());
        }
    }

    /* ---- One page exactly: the input that fills a tile to the byte, which
     * is the geometry every M4 rung is written against. */
    {
        const std::vector<unsigned char> original = MakeInput(kPageBytes);
        Compressed c;
        REQUIRE(Compress(6, original, &c));
        REQUIRE_CTX(c.pages.size() == 1, "%zu pages for one tile",
                    c.pages.size());

        libdeflate_gdeflate_decompressor* d =
            libdeflate_alloc_gdeflate_decompressor();
        REQUIRE(d != nullptr);
        std::vector<libdeflate_gdeflate_in_page> in = InPages(c);
        std::vector<unsigned char> back(original.size(), 0);
        size_t got = 0;
        REQUIRE(libdeflate_gdeflate_decompress(d, in.data(), in.size(),
                                               back.data(), back.size(),
                                               &got) == LIBDEFLATE_SUCCESS);
        REQUIRE(got == original.size());
        REQUIRE(equal_bytes(back.data(), original.data(), original.size()));

        /* The negative that proves the verdict is being read rather than
         * assumed. Byte 1 of the page is inside the header the whole page
         * hangs off, and corrupting it is refused outright. */
        std::vector<unsigned char> broken = c.bytes;
        broken[1] = static_cast<unsigned char>(broken[1] ^ 0x01u);
        std::vector<libdeflate_gdeflate_in_page> bad_in = in;
        bad_in[0].data = broken.data();
        size_t bad_got = 0;
        const libdeflate_result bad = libdeflate_gdeflate_decompress(
            d, bad_in.data(), bad_in.size(), back.data(), back.size(),
            &bad_got);
        REQUIRE_CTX(bad == LIBDEFLATE_BAD_DATA, "corrupt page gave result %d",
                    static_cast<int>(bad));
        REQUIRE(bad_got == 0);

        /* And the fact that decides how every parity test after this one has
         * to be written: a different corruption comes back SUCCESS with a
         * SHORT output. Reading the status alone would call this a decode. */
        std::vector<unsigned char> short_stream = c.bytes;
        short_stream[0] = static_cast<unsigned char>(short_stream[0] ^ 0x01u);
        std::vector<libdeflate_gdeflate_in_page> short_in = in;
        short_in[0].data = short_stream.data();
        std::vector<unsigned char> short_out(original.size(), 0);
        size_t short_got = 0;
        const libdeflate_result shortr = libdeflate_gdeflate_decompress(
            d, short_in.data(), short_in.size(), short_out.data(),
            short_out.size(), &short_got);
        REQUIRE_CTX(shortr == LIBDEFLATE_SUCCESS,
                    "expected the short-output shape, got result %d",
                    static_cast<int>(shortr));
        REQUIRE_CTX(short_got < original.size(),
                    "expected a short decode, got %zu of %zu bytes", short_got,
                    original.size());

        /* Truncation, recorded as the lever it is NOT. The trailing
         * compressed bytes of this page are never read, so cutting them away
         * leaves the output byte-identical, and a later test that reached for
         * truncation to prove a refusal would be proving nothing. */
        std::vector<libdeflate_gdeflate_in_page> cut_in = in;
        REQUIRE(cut_in[0].nbytes > 64);
        cut_in[0].nbytes -= 64;
        std::vector<unsigned char> cut_out(original.size(), 0);
        size_t cut_got = 0;
        const libdeflate_result cutr = libdeflate_gdeflate_decompress(
            d, cut_in.data(), cut_in.size(), cut_out.data(), cut_out.size(),
            &cut_got);
        REQUIRE_CTX(cutr == LIBDEFLATE_SUCCESS,
                    "a 64-byte truncation changed the verdict to %d - the "
                    "comment above this line is now wrong and the M4 reject "
                    "ladder should be told",
                    static_cast<int>(cutr));
        REQUIRE(cut_got == original.size());
        REQUIRE(equal_bytes(cut_out.data(), original.data(), original.size()));

        libdeflate_free_gdeflate_decompressor(d);
    }

    std::printf("PASS: gdeflate oracle round-trips at levels 0/1/6/12, "
                "refuses a corrupt page header, and reports a short decode "
                "as SUCCESS\n");
    return 0;
}
