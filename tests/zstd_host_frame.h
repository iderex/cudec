/* One whole Zstd frame decoded on the HOST, through the same units the device
 * kernel compiles: the frame header, src/zstd_blocks.h's block loop, and the
 * content checksum. It is the twin a device answer is compared against - the
 * three steps are tests/zstd_twin_driver.h's, reduced to the status a caller
 * of the batch ABI would see, because that is what a device result carries.
 *
 * It lives in a header because two device tests need it and a second spelling
 * of a decode path is the one duplication this tree does not accept. */
#ifndef CUDEC_TESTS_ZSTD_HOST_FRAME_H
#define CUDEC_TESTS_ZSTD_HOST_FRAME_H

#include "cudec.h"
#include "xxhash64.h"
#include "zstd_blocks.h"

#include <cstdint>
#include <vector>

namespace cudec_test {

using HostBytes = std::vector<unsigned char>;

/* The format's own ceiling on a block's sequence count - every sequence emits
 * at least a three-byte match, so a 128 KiB block holds no more. Given to the
 * harness so the loop's storage rung is never the one that fires. */
constexpr uint32_t kHostSequenceCapacity = 43690;

struct HostHarness {
    std::vector<cudec_detail::ZstdHufCell> huf;
    std::vector<cudec_detail::ZstdFseCell> litlen;
    std::vector<cudec_detail::ZstdFseCell> matchlen;
    std::vector<cudec_detail::ZstdFseCell> offset;
    cudec_detail::ZstdLiteralsScratch literals_scratch;
    cudec_detail::ZstdSeqScratch seq_scratch;
    HostBytes literals;
    std::vector<cudec_detail::ZstdSequence> sequences;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> destinations;
    cudec_detail::ZstdFrameState state;

    HostHarness()
        : huf(1u << cudec_detail::kZstdLiteralsMaxTableLog),
          litlen(1u << cudec_detail::kZstdLitLenAccuracyLogMax),
          matchlen(1u << cudec_detail::kZstdMatchLenAccuracyLogMax),
          offset(1u << cudec_detail::kZstdOffsetAccuracyLogMax),
          literals(cudec_detail::kZstdBlockSizeCeiling),
          /* The format's own ceiling on a block's sequence count, so the
           * loop's storage rung is never the one that fires and the classes
           * being compared are the ones this test is about. */
          sequences(kHostSequenceCapacity),
          offsets(kHostSequenceCapacity),
          destinations(kHostSequenceCapacity + 1) {
        state.literals_table.cells = huf.data();
        state.literals_table.capacity = static_cast<uint32_t>(huf.size());
        state.litlen.cells = litlen.data();
        state.litlen.capacity = static_cast<uint32_t>(litlen.size());
        state.matchlen.cells = matchlen.data();
        state.matchlen.capacity = static_cast<uint32_t>(matchlen.size());
        state.offset.cells = offset.data();
        state.offset.capacity = static_cast<uint32_t>(offset.size());
        state.literals_scratch = &literals_scratch;
        state.seq_scratch = &seq_scratch;
        state.literals = literals.data();
        state.literals_capacity = literals.size();
        state.sequences = sequences.data();
        state.sequences_capacity = kHostSequenceCapacity;
        state.offsets = offsets.data();
        state.offsets_capacity = kHostSequenceCapacity;
        state.destinations = destinations.data();
        state.destinations_capacity = kHostSequenceCapacity + 1;
        cudec_detail::ZstdFrameStateInit(&state);
    }
};

/* The whole-frame host path, the same three steps tests/zstd_twin_driver.h
 * walks: the frame header, the block loop, and the content checksum. */
cudec_status HostDecodeFrame(const HostBytes& frame, size_t capacity,
                             HostBytes* out, int* out_stage = 0,
                             int* out_rung = 0) {
    if (out_stage != 0) {
        *out_stage = -1;
    }
    if (out_rung != 0) {
        *out_rung = -1;
    }
    HostHarness harness;
    cudec_detail::ZstdFrameHeader header;
    cudec_detail::ZstdFrameReject frame_rung =
        cudec_detail::kZstdFrameRejectNone;
    cudec_status status = cudec_detail::ZstdParseFrameHeader(
        frame.data(), frame.size(), &header, &frame_rung);
    if (status != CUDEC_OK) {
        return status;
    }
    if (header.frame_content_size > capacity) {
        return CUDEC_ERR_OUTPUT_TOO_SMALL;
    }
    out->assign(capacity, 0);
    uint64_t produced = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdBlocksReport report;
    cudec_detail::ZstdBlocksReject blocks_rung =
        cudec_detail::kZstdBlocksRejectNone;
    status = cudec_detail::ZstdDecodeBlocks(
        frame.data() + header.header_size, frame.size() - header.header_size,
        &header, &harness.state, out->data(), header.frame_content_size,
        &produced, &consumed, &report, &blocks_rung);
    if (out_stage != 0) {
        *out_stage = static_cast<int>(report.stage);
    }
    if (out_rung != 0) {
        *out_rung = report.stage == cudec_detail::kZstdBlocksStageContentSize
                        ? static_cast<int>(blocks_rung)
                        : report.rung;
    }
    if (status != CUDEC_OK) {
        return status;
    }
    uint64_t pos = header.header_size + consumed;
    if (header.content_checksum) {
        const uint64_t digest =
            cudec_detail::Xxh64(out->data(), static_cast<size_t>(produced));
        if (cudec_detail::ZstdVerifyContentChecksum(frame.data() + pos,
                                                    frame.size() - pos, digest,
                                                    &frame_rung) != CUDEC_OK) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        pos += 4;
    }
    if (pos != frame.size()) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    out->resize(static_cast<size_t>(produced));
    return CUDEC_OK;
}


}  // namespace cudec_test

#endif /* CUDEC_TESTS_ZSTD_HOST_FRAME_H */
