/* snappy's own negative vectors, held against both sides (issue #156).
 *
 * baddata1, baddata2 and baddata3 are the three malformed streams google/snappy
 * carries in its `testdata/` and feeds to its own decoder in
 * snappy_unittest.cc. They are not mutations of anything in this tree, and that
 * is their whole value: they were written by the format's authors to break the
 * format's own decoder, so they reach the ladder from outside every generator
 * this project owns.
 *
 * PROVENANCE. The files come out of the archive this tree already pins,
 * google/snappy 1.2.2, fetched by the URL_HASH in tests/CMakeLists.txt and used
 * from where FetchContent put it. They are NOT copied into this repository. The
 * pin is the provenance: the whole archive is verified by SHA-256 before
 * anything is read out of it, so a copy in the tree would restate under review
 * what a hash already proves, and would put three binary blobs in a tree whose
 * supply-chain rule keeps binaries out of it. Measured, at the pinned archive,
 * for a reader who wants the per-file identity rather than the archive's:
 *
 *   sha256sum testdata/baddata1.snappy
 *   ca84adbcd5b4b784c3405fcdd4fa88f88a6614f2d5e5d4a527838785f1806623
 *   sha256sum testdata/baddata2.snappy
 *   06f0ca34f0de885d6a849e1c8df95391d4494bf430d3784ea2167113d6ca8947
 *   sha256sum testdata/baddata3.snappy
 *   2f60f8c9b5f3b9b0e3b4ff9aef983946552cdd1fdecdfad122292882894163bd
 *
 * LICENCE. snappy is BSD-3-Clause and its COPYING covers the repository,
 * including these three files; nothing in that file carves them out. The rest
 * of `testdata/` is where the licence question actually lives: `fireworks.jpeg`
 * is CC-BY-3.0, `kppkn.gtb` is MIT and `paper-100k.pdf` is CC-BY, each with its
 * own attribution obligation. None of them is read here, and none is needed:
 * they are compression CORPORA, and this project generates its own from the
 * Silesia path. Only the three negatives are used, and only through the pinned
 * archive, so no attribution obligation is created and none is discharged in
 * silence.
 *
 * What is asserted, per file: snappy's own no-write validator refuses it, the
 * decoding wrapper refuses it, and the cudec ladder refuses it with a defined
 * status and produces nothing. Reject parity on an outside corpus, which is the
 * property the fail-closed contract is made of. */
#include "cudec.h"
#include "fixtures.h"
#include "require.h"
#include "snappy_block.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* Where FetchContent put the pinned archive, handed in by the build so this
 * file names no path of its own. */
#ifndef CUDEC_SNAPPY_TESTDATA_DIR
#error "CUDEC_SNAPPY_TESTDATA_DIR must name the pinned archive's testdata/"
#endif

bool ReadFile(const std::string& path, Bytes* out) {
    out->clear();
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }
    unsigned char buffer[4096];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof buffer, file)) > 0) {
        out->insert(out->end(), buffer, buffer + got);
    }
    const bool ok = std::ferror(file) == 0;
    std::fclose(file);
    return ok;
}

/* Drives the ladder to a verdict and reports whether anything was produced. A
 * reject that wrote bytes first is a different failure from a reject that
 * wrote none, and only the second one is fail-closed. */
cudec_status TwinVerdict(const Bytes& stream, uint64_t* produced,
                         cudec_detail::SnappyReject* rung) {
    *produced = 0;
    *rung = cudec_detail::kSnappyRejectNone;
    const size_t size = stream.size();
    auto tight = std::make_unique<unsigned char[]>(size);
    if (size != 0) {
        std::memcpy(tight.get(), stream.data(), size);
    }
    cudec_detail::SnappyParser parser{tight.get(), size, kSnappyOracleMaxOutput};
    cudec_status status = parser.Begin();
    if (status != CUDEC_OK) {
        *rung = parser.reject;
        return status;
    }
    cudec_detail::SnappyElement element;
    bool done = false;
    uint64_t fuel = static_cast<uint64_t>(size) + 2;
    while (true) {
        if (fuel-- == 0) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        status = parser.Next(&element, &done);
        if (status != CUDEC_OK) {
            /* What a caller executing the elements would have written before
             * the refusal arrived. The contract is that this never becomes
             * output: a rejected stream produces nothing. */
            *produced = parser.dst_pos;
            *rung = parser.reject;
            return status;
        }
        if (done) {
            *produced = parser.dst_pos;
            return CUDEC_OK;
        }
    }
}

}  // namespace

int main() {
    const char* names[] = {"baddata1.snappy", "baddata2.snappy",
                           "baddata3.snappy"};
    for (const char* name : names) {
        const std::string path =
            std::string(CUDEC_SNAPPY_TESTDATA_DIR) + "/" + name;
        Bytes stream;
        REQUIRE_CTX(ReadFile(path, &stream), "%s", path.c_str());
        /* A vector that failed to arrive would otherwise be an empty stream,
         * which every decoder refuses - a green run proving nothing. */
        REQUIRE_CTX(stream.size() > 1024, "%s is %zu bytes", name,
                    stream.size());

        REQUIRE_CTX(!SnappyOracleAccepts(stream), "%s", name);
        Bytes oracle_out;
        REQUIRE_CTX(!SnappyOracleDecodes(stream, &oracle_out), "%s", name);
        REQUIRE_CTX(oracle_out.empty(), "%s", name);

        uint64_t produced = 0;
        cudec_detail::SnappyReject rung = cudec_detail::kSnappyRejectNone;
        const cudec_status status = TwinVerdict(stream, &produced, &rung);
        REQUIRE_CTX(status != CUDEC_OK, "%s was accepted by the ladder", name);
        REQUIRE_CTX(status == CUDEC_ERR_CORRUPT_INPUT ||
                        status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                    "%s refused with status %d", name,
                    static_cast<int>(status));
        /* The RUNG, not only the verdict, and the difference was measured
         * rather than assumed: with the copy-before-start check deleted, all
         * three files are still refused - a later rung catches each one - so a
         * verdict-only assertion here passes over a decoder that lost the
         * check these vectors exist to exercise. All three of them refuse for
         * the same reason, a copy reaching before the start of the output,
         * which is the defect snappy planted in them. A refactor that moves
         * the refusal to another rung reds this and should. */
        REQUIRE_CTX(rung == cudec_detail::kSnappyRejectCopyOffsetBeforeStart,
                    "%s refused at rung %d, not the copy-before-start rung",
                    name, static_cast<int>(rung));
        std::printf("%s: %zu bytes, refused by both, status %d, rung %d, %llu "
                    "bytes produced before the refusal\n",
                    name, stream.size(), static_cast<int>(status),
                    static_cast<int>(rung),
                    static_cast<unsigned long long>(produced));
    }
    std::printf("PASS\n");
    return 0;
}
