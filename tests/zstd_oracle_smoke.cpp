/* The pinned zstd oracle, proving itself (issue #181). A pin that downloads
 * and links is not yet an oracle: what makes it one is that its decode entry
 * points agree with its own compressor on real bytes, and that the two flavours
 * the tree builds from the pinned tree are the flavours it thinks they are.
 *
 * This test links zstd_full, the compressor-carrying archive, and nothing
 * else - not cudec_test_fixtures. The two zstd archives must never meet in one
 * link, because every decode symbol is in both; the fixtures archive holds the
 * decode-only one, so a test needing a compressor has to stand alone. Stated
 * in tests/CMakeLists.txt beside the targets and repeated here because this is
 * the file where the rule is felt. */
#include "require.h"

#include <zstd.h>
#include <zstd_errors.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* Compressible without being trivially so: a run of zeros round-trips through
 * any decoder that only reads its own header correctly. */
Bytes TextLike(size_t size) {
    static const char* const kWords[] = {"the",   "quick", "brown", "fox",
                                         "warp",  "lane",  "chunk", "frame",
                                         "block", "zstd"};
    Bytes out;
    out.reserve(size + 16);
    uint64_t s = 0x9e3779b97f4a7c15ull;
    while (out.size() < size) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const char* word = kWords[(s >> 33) % 10];
        out.insert(out.end(), word, word + std::strlen(word));
        out.push_back(' ');
    }
    out.resize(size);
    return out;
}

/* Aborts rather than REQUIREs, for the reason Lz4CompressBlock and
 * SnappyCompressBlock do: producing the input is infrastructure, not the
 * thing under test, and a compressor that failed here would leave every
 * assertion below testing nothing. */
Bytes Compress(const Bytes& original, int level) {
    Bytes frame(ZSTD_compressBound(original.size()));
    const size_t written = ZSTD_compress(frame.data(), frame.size(),
                                         original.data(), original.size(),
                                         level);
    if (ZSTD_isError(written) || written == 0) {
        std::abort();
    }
    frame.resize(written);
    return frame;
}

}  // namespace

int main() {
    /* The version the harness linked, not the version somebody meant to pin.
     * A FetchContent pin that silently resolved to a system copy would show
     * up here and nowhere else in this file. */
    REQUIRE(ZSTD_versionNumber() == 10507);
    REQUIRE(std::string(ZSTD_versionString()) == "1.5.7");

    size_t frames = 0;
    for (const size_t n : {size_t{1}, size_t{2}, size_t{255}, size_t{4096},
                           size_t{65536}, size_t{200000}}) {
        const Bytes original = TextLike(n);
        for (const int level : {1, 9, 19}) {
            const Bytes frame = Compress(original, level);

            /* The declared-size surface, which is the half the M5 ladder reads
             * before it sizes anything: ZSTD_compress writes the content size
             * into the frame header, so it must be there and it must be the
             * size the payload actually produces. */
            const unsigned long long declared =
                ZSTD_getFrameContentSize(frame.data(), frame.size());
            REQUIRE(declared != ZSTD_CONTENTSIZE_ERROR);
            REQUIRE(declared != ZSTD_CONTENTSIZE_UNKNOWN);
            REQUIRE_CTX(declared == n, "level %d, %zu bytes", level, n);

            Bytes decoded(static_cast<size_t>(declared), 0);
            const size_t produced = ZSTD_decompress(
                decoded.data(), decoded.size(), frame.data(), frame.size());
            REQUIRE(!ZSTD_isError(produced));
            REQUIRE(produced == n);
            REQUIRE(std::memcmp(decoded.data(), original.data(), n) == 0);
            frames++;
        }
    }

    /* And the refusing direction, because an oracle that accepts everything
     * would pass every check above.
     *
     * What the three cases below pin is not what was expected of them, and
     * the difference is the point. The error CLASS was going to be the thing
     * the M5 fail-closed matrix read per rejection shape - but on the simple
     * API all three come back ZSTD_error_srcSize_wrong, measured here rather
     * than assumed. So the discriminating verdict is the PAIR: the error code
     * says a frame was refused, and ZSTD_getFrameContentSize says whether the
     * header parsed at all. A matrix row that named a class per shape would
     * have been writing down something zstd does not report on this surface.
     *
     * ZSTD_STRIP_ERROR_STRINGS removes the messages, so this enum and that
     * pair are the whole account of why a frame was refused. */
    const int kSrcSizeWrong = static_cast<int>(ZSTD_error_srcSize_wrong);
    {
        /* An unrecognised prefix: no frame header at all. */
        const Bytes garbage = {0xde, 0xad, 0xbe, 0xef, 0x00, 0x11, 0x22, 0x33};
        Bytes out(64, 0);
        const size_t produced = ZSTD_decompress(out.data(), out.size(),
                                                garbage.data(),
                                                garbage.size());
        REQUIRE(ZSTD_isError(produced));
        REQUIRE(static_cast<int>(ZSTD_getErrorCode(produced)) == kSrcSizeWrong);
        REQUIRE(ZSTD_getFrameContentSize(garbage.data(), garbage.size()) ==
                ZSTD_CONTENTSIZE_ERROR);
    }
    {
        /* The legacy decoders are compiled out (ZSTD_LEGACY_SUPPORT=0), so a
         * pre-v0.8 magic number is refused rather than decoded. Asserted
         * because an oracle that accepts more than the thing under test would
         * report a divergence that is not one. */
        const Bytes legacy = {0x25, 0xb5, 0x2a, 0xfd, 0x00, 0x00, 0x00, 0x00};
        Bytes out(64, 0);
        const size_t produced = ZSTD_decompress(out.data(), out.size(),
                                                legacy.data(), legacy.size());
        REQUIRE(ZSTD_isError(produced));
        REQUIRE(static_cast<int>(ZSTD_getErrorCode(produced)) == kSrcSizeWrong);
        REQUIRE(ZSTD_getFrameContentSize(legacy.data(), legacy.size()) ==
                ZSTD_CONTENTSIZE_ERROR);
    }
    {
        /* A real frame with its tail removed: the header is fine and the
         * blocks are not. Same error code as both cases above, and the
         * content-size call is what tells it apart from them - which is the
         * whole reason this case sits beside them rather than instead of
         * them. */
        const Bytes original = TextLike(4096);
        Bytes frame = Compress(original, 3);
        REQUIRE(frame.size() > 8);
        frame.resize(frame.size() - 4);
        Bytes out(4096, 0);
        const size_t produced =
            ZSTD_decompress(out.data(), out.size(), frame.data(), frame.size());
        REQUIRE(ZSTD_isError(produced));
        REQUIRE(static_cast<int>(ZSTD_getErrorCode(produced)) == kSrcSizeWrong);
        REQUIRE(ZSTD_getFrameContentSize(frame.data(), frame.size()) == 4096);
    }

    std::printf("PASS: zstd %s pinned and linked; %zu frames round-tripped "
                "with the declared content size agreeing on every one; "
                "unknown prefix, legacy magic and truncated tail all refused, "
                "all three as srcSize_wrong, told apart by the content-size "
                "verdict\n",
                ZSTD_versionString(), frames);
    return 0;
}
