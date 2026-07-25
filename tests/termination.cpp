/* Termination as a tested invariant (issue #72). The decoder's liveness
 * argument - every Lz4Parser::Next call consumes at least the token byte,
 * so every decode loop terminates - existed only as prose above the
 * function, and a decoder that fails to terminate hangs the device: a
 * fail-closed violation exactly like an out-of-bounds read, and a
 * demonstrated bug class in this product category (nvCOMP 5.3's release
 * notes record fixing an indefinite Snappy-decompression hang on malformed
 * input).
 *
 * This test drives truncated, mutated, and adversarially crafted LZ4 blocks
 * through the shared parser and asserts, on every step, that a step
 * returning CUDEC_OK advanced the source cursor and that the parse reaches
 * a terminal state within the contract's step bound. The driving loop is
 * itself bounded one step past that budget, so a parser that stopped making
 * progress FAILS this test instead of hanging it.
 *
 * The frame cases drive src/frame.cpp's block-table walk, whose step count
 * is not observable from outside the library; there the assertion is that
 * the call returns at all with a defined status, and the ctest TIMEOUT on
 * this target is the backstop. Every frame here is malformed or wholly
 * uncompressed, so all of it rejects (or assembles) before the first CUDA
 * call - the test runs on the GPU-less runner and under the ASan/UBSan
 * gate, like tests/frame_host_negative.cpp. */
#include "adversarial_blocks.h"
#include "cudec.h"
#include "fixtures.h"
#include "lz4_block.h"
#include "require.h"
#include "xxhash32.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* The parser's whole status surface: anything else is an undefined outcome
 * on hostile input, which is the other half of fail-closed. */
bool IsDefinedParserStatus(cudec_status status) {
    return status == CUDEC_OK || status == CUDEC_ERR_CORRUPT_INPUT ||
           status == CUDEC_ERR_OUTPUT_TOO_SMALL;
}

/* Drives the parser the way every decode loop does and asserts both halves
 * of the liveness contract. The stream is parsed from an EXACTLY-sized heap
 * copy (as tests/parser_twin.cpp does) so an over-read past src_size lands
 * in an ASan redzone instead of a vector's rounded-up capacity slack. */
int RequireTerminates(const std::string& context, const Bytes& stream,
                      uint64_t dst_capacity) {
    const size_t stream_size = stream.size();
    auto tight = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(tight.get(), stream.data(), stream_size);
    }
    cudec_detail::Lz4Parser parser{tight.get(), stream_size, dst_capacity};
    cudec_detail::Lz4Sequence seq;

    const uint64_t budget = static_cast<uint64_t>(stream_size) + 1;
    uint64_t steps = 0;
    cudec_status status = CUDEC_OK;
    bool done = false;
    /* One step of slack past the budget, so exceeding it is reported here
     * rather than spun on. */
    while (steps <= budget) {
        const uint64_t src_pos_before = parser.src_pos;
        steps++;
        status = parser.Next(&seq, &done);
        REQUIRE_CTX(IsDefinedParserStatus(status),
                    "%s: step %llu returned undefined status %d",
                    context.c_str(), static_cast<unsigned long long>(steps),
                    static_cast<int>(status));
        if (status != CUDEC_OK) {
            break;
        }
        REQUIRE_CTX(parser.src_pos > src_pos_before,
                    "%s: step %llu returned CUDEC_OK without consuming a "
                    "source byte (src_pos stuck at %llu)",
                    context.c_str(), static_cast<unsigned long long>(steps),
                    static_cast<unsigned long long>(src_pos_before));
        if (done) {
            break;
        }
    }
    REQUIRE_CTX(steps <= budget,
                "%s: no terminal state within %llu steps for %llu source "
                "bytes",
                context.c_str(), static_cast<unsigned long long>(budget),
                static_cast<unsigned long long>(stream_size));
    /* Success is reported ONLY at exact consumption after a literals-only
     * tail: an OK that is not `done` would be a decode loop's exit without
     * a completed parse. */
    REQUIRE_CTX(status != CUDEC_OK || done,
                "%s: CUDEC_OK outside exact-consumption success",
                context.c_str());
    REQUIRE_CTX(parser.src_pos <= stream_size && parser.dst_pos <= dst_capacity,
                "%s: cursor left its bound (src %llu/%llu dst %llu)",
                context.c_str(),
                static_cast<unsigned long long>(parser.src_pos),
                static_cast<unsigned long long>(stream_size),
                static_cast<unsigned long long>(parser.dst_pos));
    return 0;
}

/* Capacities crossing the ladder's decision points: none at all, below the
 * end-of-block slack, a realistic buffer, and the SIZE_MAX the header
 * documents as a legal caller value (it disables the dst-slack terminal
 * rule, so the parse runs on until the source is exhausted - the longest
 * path through the loop). */
const uint64_t kCapacities[] = {0, 1, 12, 1u << 20, UINT64_MAX};

/* Truncation ladder: every prefix of a short stream, and a strided set plus
 * the last 32 prefixes of a long one - a stream cut mid-sequence, mid-length
 * extension, and mid-offset is where a length-driven loop would spin. */
std::vector<size_t> PrefixLengths(size_t n) {
    std::vector<size_t> lengths;
    if (n <= 256) {
        for (size_t k = 0; k <= n; k++) {
            lengths.push_back(k);
        }
        return lengths;
    }
    const size_t stride = n / 128;
    for (size_t k = 0; k < n; k += stride) {
        lengths.push_back(k);
    }
    for (size_t k = n - 32; k <= n; k++) {
        lengths.push_back(k);
    }
    return lengths;
}

int RunTruncationLadder(const std::vector<Fixture>& fixtures) {
    for (const Fixture& f : fixtures) {
        for (const size_t keep : PrefixLengths(f.compressed.size())) {
            const Bytes prefix(f.compressed.begin(),
                               f.compressed.begin() + static_cast<long>(keep));
            const std::string context =
                f.name + "/prefix-" + std::to_string(keep);
            /* The fixture's own size is the interesting capacity for a
             * prefix: it is the one a caller would have passed. */
            REQUIRE(RequireTerminates(context, prefix, f.original.size()) == 0);
        }
    }
    return 0;
}

int RunMutantCorpus(const std::vector<Fixture>& fixtures) {
    for (const Fixture& f : fixtures) {
        const std::vector<Mutant> mutants = MutateStream(f.compressed, f.seed);
        REQUIRE(!mutants.empty());
        for (const Mutant& m : mutants) {
            const std::string context = f.name + "/" + m.description;
            for (const uint64_t capacity : kCapacities) {
                REQUIRE(RequireTerminates(context, m.stream, capacity) == 0);
            }
        }
    }
    return 0;
}

int RunAdversarialBlocks() {
    const std::vector<AdversarialBlock> blocks = MakeAdversarialBlocks();
    REQUIRE(!blocks.empty());
    for (const AdversarialBlock& block : blocks) {
        REQUIRE(RequireTerminates(block.name + "/cap-suggested", block.stream,
                                  block.dst_capacity) == 0);
        for (const uint64_t capacity : kCapacities) {
            const std::string context =
                block.name + "/cap-" + std::to_string(capacity);
            REQUIRE(RequireTerminates(context, block.stream, capacity) == 0);
        }
        /* The mutation corpus of a hostile stream too: a hostile stream is
         * rarely the only hostile stream nearby. */
        if (block.stream.empty()) {
            continue;
        }
        for (const Mutant& m : MutateStream(block.stream, 0x72)) {
            REQUIRE(RequireTerminates(block.name + "/" + m.description,
                                      m.stream, block.dst_capacity) == 0);
        }
    }
    return 0;
}

/* ---- The frame block-table walk.
 *
 * A minimal builder, deliberately narrower than the one in
 * tests/frame_host_negative.cpp: these cases need only the block-record
 * walk, with a descriptor fixed at block-independent / 64 KiB block max and
 * no optional fields. The header checksum is computed with the library's
 * own xxhash32.h so the walk past the checksum gate is reachable at all. */
constexpr unsigned char kFlgIndependent = 0x60; /* version 01, block-indep */
constexpr unsigned char kBd64K = 0x40;

void Put32(Bytes* f, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        f->push_back(static_cast<unsigned char>((v >> (i * 8)) & 0xFF));
    }
}

Bytes FrameHeader() {
    Bytes f = {0x04, 0x22, 0x4D, 0x18, kFlgIndependent, kBd64K};
    f.push_back(static_cast<unsigned char>(
        (cudec_detail::xxhash32(f.data() + 4, 2) >> 8) & 0xFF));
    return f;
}

/* `blocks` uncompressed block payloads, optionally followed by the end
 * mark. Without it the walk must run out of frame and reject. */
Bytes BuildUncompressedFrame(const std::vector<Bytes>& blocks, bool end_mark) {
    Bytes f = FrameHeader();
    for (const Bytes& b : blocks) {
        Put32(&f, 0x80000000u | static_cast<uint32_t>(b.size()));
        f.insert(f.end(), b.begin(), b.end());
    }
    if (end_mark) {
        Put32(&f, 0);
    }
    return f;
}

/* Decodes from an exactly-sized copy so an over-read past frame_size reds
 * the sanitizer (same reasoning as tests/frame_host_negative.cpp). */
int RequireFrameReturns(const std::string& context, const Bytes& frame,
                        size_t dst_capacity) {
    const size_t n = frame.size();
    auto tight = std::make_unique<unsigned char[]>(n);
    if (n != 0) {
        std::memcpy(tight.get(), frame.data(), n);
    }
    std::vector<unsigned char> dst(dst_capacity ? dst_capacity : 1, 0);
    size_t written = 12345; /* must be overwritten, never left stale */
    const cudec_status status = cudec_lz4f_decompress(
        tight.get(), n, dst.data(), dst_capacity, &written);
    REQUIRE_CTX(status == CUDEC_OK || status == CUDEC_ERR_CORRUPT_INPUT ||
                    status == CUDEC_ERR_OUTPUT_TOO_SMALL ||
                    status == CUDEC_ERR_UNSUPPORTED ||
                    status == CUDEC_ERR_INVALID_ARGUMENT ||
                    status == CUDEC_ERR_CUDA,
                "%s: undefined status %d", context.c_str(),
                static_cast<int>(status));
    REQUIRE_CTX(status == CUDEC_OK || written == 0,
                "%s: rejected frame reported %llu bytes", context.c_str(),
                static_cast<unsigned long long>(written));
    return 0;
}

int RunFrameWalk() {
    /* Many minimal block records: the walk's step count is driven by the
     * frame's own contents, which is exactly the shape a fuel-free loop
     * would be at the mercy of. */
    std::vector<Bytes> many(4000, Bytes(1, 0x41));
    REQUIRE(RequireFrameReturns("frame/4000-blocks-end-mark",
                                BuildUncompressedFrame(many, true),
                                many.size()) == 0);
    REQUIRE(RequireFrameReturns("frame/4000-blocks-no-end-mark",
                                BuildUncompressedFrame(many, false),
                                many.size()) == 0);
    REQUIRE(RequireFrameReturns("frame/header-only", FrameHeader(), 64) == 0);

    /* A block-size field claiming the whole 31-bit range, and one claiming
     * zero length: both must reject without walking anywhere. */
    Bytes huge = FrameHeader();
    Put32(&huge, 0xFFFFFFFFu);
    REQUIRE(RequireFrameReturns("frame/block-size-max", huge, 64) == 0);
    Bytes zero_len = FrameHeader();
    Put32(&zero_len, 0x80000000u);
    REQUIRE(RequireFrameReturns("frame/block-size-zero", zero_len, 64) == 0);

    /* A header followed by pure garbage, and by a long run of 0x01 block
     * headers whose payloads run off the end. */
    Bytes garbage = FrameHeader();
    garbage.insert(garbage.end(), 4096, 0xFF);
    REQUIRE(RequireFrameReturns("frame/garbage-tail", garbage, 64) == 0);
    Bytes truncated_payloads = FrameHeader();
    for (int i = 0; i < 1024; i++) {
        Put32(&truncated_payloads, 0x80000001u);
        truncated_payloads.push_back(0x41);
    }
    REQUIRE(RequireFrameReturns("frame/truncated-payload-run",
                                truncated_payloads, 64) == 0);

    /* Every prefix of a well-formed many-block frame: each one cuts the
     * walk at a different point in a block record. */
    const Bytes complete = BuildUncompressedFrame(
        std::vector<Bytes>(64, Bytes(3, 0x41)), true);
    for (size_t keep = 0; keep <= complete.size(); keep++) {
        const Bytes prefix(complete.begin(),
                           complete.begin() + static_cast<long>(keep));
        REQUIRE(RequireFrameReturns("frame/prefix-" + std::to_string(keep),
                                    prefix, 256) == 0);
    }
    return 0;
}

}  // namespace

int main() {
    const std::vector<Fixture> fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());
    REQUIRE(RunTruncationLadder(fixtures) == 0);
    REQUIRE(RunMutantCorpus(fixtures) == 0);
    REQUIRE(RunAdversarialBlocks() == 0);
    REQUIRE(RunFrameWalk() == 0);
    std::printf("termination: parser liveness and frame-walk termination "
                "hold over the truncation ladder, the mutant corpus, the "
                "crafted streams, and the frame cases\n");
    return 0;
}
