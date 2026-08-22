/* The corpus-level differential run over the whole M5 entropy stage (issue
 * #223): every #185 fixture and every #187 mutant driven through the
 * single-sourced units in src/ and held to the pinned reference in BOTH
 * directions, on the GPU-less runner.
 *
 * WHAT THIS ADDS TO THE SIBLING TWINS, which is not more of the same. Each
 * entropy unit's twin proves that unit against the reference's matching entry
 * point and carries a crafted negative per reject rung. None of them decodes a
 * FRAME, so none of them can say whether the stage as a whole ever accepts
 * something the reference refuses - and that is the property a decompressor is
 * judged on. This file asks exactly that question, over the two corpora built
 * for it.
 *
 * TWO DIRECTIONS, AND THE SECOND ONE IS WHY THE DRIVER IS COMPLETE. Where the
 * reference accepts, the bytes must be identical; where it refuses, the driver
 * must refuse too. The second half is only worth something if the driver
 * enforces everything the reference does, so tests/zstd_twin_driver.h carries
 * the frame-level rules as well as the entropy ones: the declared content
 * size, the content checksum, the window bound on a match, and bytes after the
 * frame. A driver missing any of them would report the gap as a fail-open on
 * every mutant that reached it.
 *
 * A FRAME THIS DECODER DECLINES IS NOT A FRAME IT REFUSES. The v1 subset
 * (#102) leaves out frames with no declared content size, with a dictionary
 * id, and with a window past 8 MB, and a mutation lands on those descriptor
 * bits regularly. Those come back CUDEC_ERR_UNSUPPORTED and are counted under
 * their own name; treating them as divergences would report the subset as a
 * defect, and treating them as agreement would hide a real over-strictness
 * among them.
 *
 * THE COUNT LOCK IS A STATIC ASSERTION AND IT IS DELIBERATELY BLUNT. Every
 * reject rung across the Zstd ladder is counted at compile time and compared
 * against one number written down here. A rung added anywhere moves the sum
 * and reds the build until somebody looks - and what they have to do when they
 * look is add the negative that reaches it, which that unit's own twin already
 * refuses to pass without. The number is written down rather than derived
 * because a lock derived from the thing it locks is not a lock. */
#include "require.h"
#include "zstd_corpus.h"
#include "zstd_twin_driver.h"

/* The reference's Huffman entry points are internal to libzstd and its huf.h
 * carries no C++ linkage guard of its own, so the include is wrapped here
 * rather than its declarations restated. HUF_STATIC_LINKING_ONLY is defined by
 * the build, for the reason issue #180 landed. */
extern "C" {
#include <common/huf.h>
}

#include <zstd.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

using cudec_twin::Bytes;
using cudec_twin::DecodeFrame;
using cudec_twin::kStageNames;
using cudec_twin::Run;

/* ---- The count lock ----------------------------------------------------
 *
 * Every reject rung the Zstd surface can return, summed at compile time. The
 * seven ladders are the bitstream reader, the FSE table description, the
 * Huffman tree description, the literals section, the sequences section, the
 * repeat-offset history and the sequence execution - the whole surface rather
 * than the entropy subset, because a rung added to any of them is a refusal
 * that arrives with no negative behind it unless somebody wrote one.
 *
 * Each ladder's `...RejectCount` counts kNone as well, so the sum below
 * subtracts one per ladder and the number is the count of real rungs.
 *
 * WHEN THIS REDS, the repair is not to update the number. It is to write the
 * negative that reaches the new rung, in the twin that owns that unit - which
 * that twin's own coverage loop already refuses to pass without - and THEN to
 * update the number here, deliberately, as the record that somebody looked. */
constexpr int kZstdRejectRungs =
    (cudec_detail::kZstdBitRejectCount - 1) +
    (cudec_detail::kZstdFseRejectCount - 1) +
    (cudec_detail::kZstdHufRejectCount - 1) +
    (cudec_detail::kZstdLiteralsRejectCount - 1) +
    (cudec_detail::kZstdSeqRejectCount - 1) +
    (cudec_detail::kZstdRepcodeRejectCount - 1) +
    (cudec_detail::kZstdExecRejectCount - 1);

constexpr int kExpectedZstdRejectRungs = 67;

/* How many mutants land in the declared class below, measured on the pinned
 * zstd 1.5.7 and the pinned mutation corpus. Written down rather than derived
 * for the reason the count lock above is: a number that moves silently is not
 * a record that anybody looked. */
constexpr int kExpectedReferenceSelfDisagreements = 1;

static_assert(kZstdRejectRungs == kExpectedZstdRejectRungs,
              "a Zstd reject rung was added or removed: write the negative "
              "that reaches it in that unit's twin, then update "
              "kExpectedZstdRejectRungs here (issue #223)");

/* ---- The runs ---------------------------------------------------------- */

/* One capacity for every run, above anything either corpus regenerates, so a
 * refusal is never about room. */
constexpr uint64_t kCapacity = 8ull * 1024ull * 1024ull;

size_t g_fixtures = 0;
size_t g_fixture_bytes = 0;
size_t g_fixtures_declined = 0;
size_t g_mutants = 0;
size_t g_mutants_oracle_refused = 0;
size_t g_mutants_both_accepted = 0;
size_t g_mutants_declined = 0;
size_t g_stage_refusals[cudec_twin::kStageCount] = {0};
size_t g_reference_disagrees_with_itself = 0;

/* The first byte at which two strings differ, or the shorter length when one
 * is a prefix of the other. Reported on a divergence because "the bytes
 * differ" is not a repro key and an offset is. */
size_t FirstDifference(const Bytes& a, const Bytes& b) {
    const size_t shared = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < shared; i++) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return shared;
}

/* The accept direction over the forced-mode corpus. */
int Fixtures() {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());
    for (size_t f = 0; f < fixtures.size(); f++) {
        const ZstdFixture& fixture = fixtures[f];
        const char* name = fixture.name.c_str();
        Bytes reference;
        REQUIRE_CTX(ZstdOracleDecodes(fixture.compressed, &reference),
                    "%s: the reference refused a fixture", name);
        const Run run = DecodeFrame(fixture.compressed, kCapacity);
        if (!run.ok && run.status == CUDEC_ERR_UNSUPPORTED) {
            /* A legal frame outside the v1 subset. Counted and named on the
             * PASS line: a fixture that fell out of the sweep by accident
             * must not read like one that was swept. */
            g_fixtures_declined++;
            continue;
        }
        REQUIRE_CTX(run.ok, "%s: refused at the %s stage, rung %d: %s", name,
                    kStageNames[run.stage], run.rung, run.why.c_str());
        REQUIRE_CTX(run.output.size() == reference.size(),
                    "%s: %zu bytes, the reference produced %zu", name,
                    run.output.size(), reference.size());
        REQUIRE_CTX(equal_bytes(run.output.data(), reference.data(),
                                reference.size()),
                    "%s: bytes differ from the reference", name);
        /* And against the source the corpus compressed, which is the third
         * reader: the reference and the driver agreeing on wrong bytes would
         * pass the comparison above. */
        REQUIRE_CTX(run.output.size() == fixture.original.size() &&
                        equal_bytes(run.output.data(),
                                    fixture.original.data(),
                                    fixture.original.size()),
                    "%s: the round trip lost bytes", name);
        g_fixtures++;
        g_fixture_bytes += run.output.size();
    }
    REQUIRE(g_fixtures > 0);
    return 0;
}

/* Does the reference's OWN entropy entry point refuse this frame's literals?
 *
 * This exists for one measured case and it is not a shrug. libzstd's frame
 * path accepted a mutant whose second block declares a four-stream literals
 * section whose first stream is seventeen bytes and must regenerate three
 * hundred and sixty-seven symbols - which no code of one bit or more can
 * spell - and produced different bytes rather than refusing. Its own
 * HUF_decompress4X_usingDTable, handed the same payload and the same table,
 * refuses it. So the disagreement is inside the reference, and cudec siding
 * with the stricter of its two answers is not an over-strictness of cudec's.
 *
 * What this asks is exactly that and nothing wider: walk the frame, hand each
 * compressed block's literals payload to the reference's own Huffman decoder
 * with the table the previous block left, and report whether any of them came
 * back refused. A yes admits the mutant into the declared class below; a no
 * fails the test, because then the reference agrees with itself and cudec is
 * the odd one out.
 *
 * The walk uses cudec's own header parsers to find the payload, which the
 * fixture rows above have already held to the reference byte for byte. */
bool ReferenceEntropyRefusesALiteralsSection(const Bytes& frame) {
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject frame_rung =
        cudec_detail::kZstdFrameRejectNone;
    if (cudec_detail::ZstdParseFrameHeader(frame.data(), frame.size(), &header,
                                           &frame_rung) != CUDEC_OK) {
        return false;
    }
    HUF_CREATE_STATIC_DTABLEX1(reference_table, HUF_TABLELOG_MAX);
    bool table_present = false;
    std::vector<unsigned> workspace(HUF_DECOMPRESS_WORKSPACE_SIZE_U32, 0);
    uint64_t pos = header.header_size;
    for (;;) {
        cudec_detail::ZstdBlockHeader block;
        if (cudec_detail::ZstdParseBlockHeader(frame.data() + pos,
                                               frame.size() - pos,
                                               header.window_size, &block,
                                               &frame_rung) != CUDEC_OK) {
            return false;
        }
        const unsigned char* body = frame.data() + pos + 3;
        if (block.block_type == cudec_detail::kZstdBlockTypeCompressed) {
            cudec_detail::ZstdLiteralsHeader literals;
            cudec_detail::ZstdLiteralsReject literals_rung =
                cudec_detail::kZstdLiteralsRejectNone;
            if (cudec_detail::ZstdParseLiteralsHeader(body, block.body_size,
                                                      &literals,
                                                      &literals_rung) !=
                CUDEC_OK) {
                return false;
            }
            const bool compressed =
                literals.block_type ==
                cudec_detail::kZstdLiteralsTypeCompressed;
            const bool treeless =
                literals.block_type ==
                cudec_detail::kZstdLiteralsTypeTreeless;
            if ((compressed || treeless) &&
                literals.compressed_size <=
                    block.body_size - literals.header_size &&
                literals.regenerated_size != 0) {
                if (treeless && !table_present) {
                    return false;
                }
                const unsigned char* payload = body + literals.header_size;
                Bytes out(static_cast<size_t>(literals.regenerated_size), 0);
                size_t result;
                if (compressed) {
                    result =
                        literals.stream_count == 1
                            ? HUF_decompress1X1_DCtx_wksp(
                                  reference_table, out.data(), out.size(),
                                  payload,
                                  static_cast<size_t>(
                                      literals.compressed_size),
                                  workspace.data(),
                                  workspace.size() * sizeof(unsigned), 0)
                            : HUF_decompress4X_hufOnly_wksp(
                                  reference_table, out.data(), out.size(),
                                  payload,
                                  static_cast<size_t>(
                                      literals.compressed_size),
                                  workspace.data(),
                                  workspace.size() * sizeof(unsigned), 0);
                    if (!HUF_isError(result)) {
                        table_present = true;
                    }
                } else {
                    result = literals.stream_count == 1
                                 ? HUF_decompress1X_usingDTable(
                                       out.data(), out.size(), payload,
                                       static_cast<size_t>(
                                           literals.compressed_size),
                                       reference_table, 0)
                                 : HUF_decompress4X_usingDTable(
                                       out.data(), out.size(), payload,
                                       static_cast<size_t>(
                                           literals.compressed_size),
                                       reference_table, 0);
                }
                if (HUF_isError(result)) {
                    return true;
                }
            }
        }
        pos += 3 + block.body_size;
        if (block.last_block || pos >= frame.size()) {
            return false;
        }
    }
}

/* Both directions over the mutation corpus. */
int Mutants() {
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());
    size_t fixtures_with_a_driver_refusal = 0;
    for (size_t f = 0; f < fixtures.size(); f++) {
        const ZstdFixture& fixture = fixtures[f];
        const std::vector<ZstdMutant> mutants =
            MutateZstdFrame(fixture.compressed, f + 1);
        REQUIRE_CTX(!mutants.empty(), "%s: the mutation layer produced none",
                    fixture.name.c_str());
        bool driver_refused_one = false;
        for (size_t m = 0; m < mutants.size(); m++) {
            const ZstdMutant& mutant = mutants[m];
            g_mutants++;
            Bytes reference;
            const bool oracle_ok = ZstdOracleDecodes(mutant.frame, &reference);
            const Run run = DecodeFrame(mutant.frame, kCapacity);
            if (!run.ok) {
                driver_refused_one = true;
                g_stage_refusals[run.stage]++;
            }

            if (!oracle_ok) {
                g_mutants_oracle_refused++;
                /* THE FAIL-OPEN DIRECTION. A mutant the reference refuses and
                 * the driver accepts is the failure this whole file exists to
                 * find, and it is reported with everything needed to rebuild
                 * it: the fixture, the mutation's own description, and the
                 * index inside that fixture's mutant list. */
                REQUIRE_CTX(!run.ok,
                            "FAIL-OPEN: %s mutant %zu (%s) was refused by "
                            "libzstd and accepted by the driver, which "
                            "produced %zu bytes",
                            fixture.name.c_str(), m,
                            mutant.description.c_str(), run.output.size());
                continue;
            }

            if (!run.ok) {
                if (run.status == CUDEC_ERR_UNSUPPORTED) {
                    /* A legal frame this decoder declines rather than
                     * refuses. Counted under its own name. */
                    g_mutants_declined++;
                    continue;
                }
                /* The one admitted class, and it is checked rather than
                 * assumed: cudec refuses a literals section that the
                 * reference's own Huffman decoder refuses too, while the
                 * reference's frame path accepts it. */
                REQUIRE_CTX(
                    run.stage == cudec_twin::kStageLiterals &&
                        ReferenceEntropyRefusesALiteralsSection(mutant.frame),
                    "OVER-STRICT: %s mutant %zu (%s) was decoded by libzstd "
                    "to %zu bytes and refused by the driver at the %s stage, "
                    "rung %d: %s",
                    fixture.name.c_str(), m, mutant.description.c_str(),
                    reference.size(), kStageNames[run.stage], run.rung,
                    run.why.c_str());
                g_reference_disagrees_with_itself++;
                continue;
            }

            g_mutants_both_accepted++;
            REQUIRE_CTX(
                run.output.size() == reference.size() &&
                    FirstDifference(run.output, reference) == reference.size(),
                "%s mutant %zu (%s): both accepted and the bytes differ at "
                "offset %zu; driver produced %zu, libzstd %zu",
                fixture.name.c_str(), m, mutant.description.c_str(),
                FirstDifference(run.output, reference), run.output.size(),
                reference.size());
        }
        if (driver_refused_one) {
            fixtures_with_a_driver_refusal++;
        }
    }
    REQUIRE(g_mutants > 0);
    /* The mutations have to BITE on this side too. Every fixture yielding a
     * mutant the reference refuses is what tests/zstd_mutants.cpp already
     * establishes; that says nothing about whether the DRIVER ever refuses,
     * and a driver that accepted everything would pass every assertion above
     * except this one. */
    const size_t swept = g_fixtures + g_fixtures_declined;
    REQUIRE_CTX(fixtures_with_a_driver_refusal == swept,
                "%zu of %zu fixtures produced no mutant the driver refused",
                swept - fixtures_with_a_driver_refusal, swept);
    return 0;
}

}  // namespace

int main() {
    int rc = Fixtures();
    if (rc != 0) {
        return rc;
    }
    rc = Mutants();
    if (rc != 0) {
        return rc;
    }

    std::printf("PASS: %zu fixtures decoded byte-identical to libzstd and to "
                "their own sources (%zu bytes), %zu declined as outside the "
                "v1 subset; %zu mutants, %zu refused by libzstd with 0 "
                "fail-opens, %zu accepted by both with 0 byte divergences, "
                "%zu declined as outside the subset; %d reject rungs locked\n",
                g_fixtures, g_fixture_bytes, g_fixtures_declined, g_mutants,
                g_mutants_oracle_refused, g_mutants_both_accepted,
                g_mutants_declined, kZstdRejectRungs);
    std::printf("      driver refusals by stage:");
    for (unsigned stage = 1; stage < cudec_twin::kStageCount; stage++) {
        if (g_stage_refusals[stage] != 0) {
            std::printf(" %s %zu", kStageNames[stage],
                        g_stage_refusals[stage]);
        }
    }
    std::putchar('\n');

    /* The class above is real and it is not empty, so its count is asserted
     * both ways: a run in which it vanished would mean the reference changed
     * its mind, and a run in which it grew would mean a second such case
     * arrived unexamined. */
    REQUIRE_CTX(g_reference_disagrees_with_itself ==
                    kExpectedReferenceSelfDisagreements,
                "%zu mutants where libzstd's frame path accepts what its own "
                "Huffman decoder refuses; %d were recorded",
                g_reference_disagrees_with_itself,
                kExpectedReferenceSelfDisagreements);
    std::printf("      %zu mutants where libzstd's frame path accepts a "
                "literals section its own Huffman decoder refuses\n",
                g_reference_disagrees_with_itself);
    return 0;
}
