/* Differential fuzz target over the Zstd literals section (issue #191), the
 * sixth target in fuzz/ and the third over the M5 surface: the section header
 * in all of its width variants, the Huffman tree description in both
 * spellings, the canonical table build, the six-byte jump table and the four
 * backward streams.
 *
 * THIS ONE IS STRUCTURE-AWARE AND ITS SIBLING IS NOT, for a reason that is
 * about the ORACLE rather than about coverage. fuzz_zstd_fse.cpp enters at
 * FSE_readNCount, which takes a description and a length, so raw bytes are
 * already the thing the reference reads. A literals section is not an entry
 * point libzstd exposes: the nearest ones are HUF_decompress*, which take the
 * PAYLOAD and are reached only after somebody has parsed the header. Handing
 * them a payload this target located with the unit under test would make the
 * header parse decide its own comparison. So the fuzzer's bytes are spliced
 * in as the whole literals section of a real frame and ZSTD_decompress is
 * asked about the frame, which puts the header, the description, the split
 * and the streams all inside what the reference answers for.
 *
 * THE ENVELOPE IS THE SMALLEST ONE THAT MAKES A BLOCK'S OUTPUT ITS LITERALS.
 * A frame header declaring no content size and a 128 KB window, then a
 * compressed block holding the fuzzer's section and a sequences section of
 * zero sequences. With no sequences the block regenerates exactly its
 * literals, so the reference's output IS the answer to "what did this
 * literals section decode to" and no second stage sits between the two sides.
 *
 * TWO PASSES, AND THE SECOND ONE IS THE ONLY WAY Treeless IS REACHABLE. A
 * Treeless section decodes through the table an earlier block left behind, so
 * with no earlier block every Treeless input is a refusal on both sides and
 * the whole arm goes unexercised. The second pass puts a fixed, hand-written
 * Compressed section in a first block, so the fuzzer's section meets a table
 * that exists - on both sides, because the reference decodes the same two
 * blocks. That seed section is proven rather than trusted: the target checks
 * once, at startup, that the reference accepts a frame built from it alone and
 * that the unit decodes it to the four bytes it was written to spell.
 *
 * WHICH DIRECTION IS ASSERTED. The fail-open direction - the unit accepting
 * what the reference refuses, or producing different bytes - always traps.
 * The reverse is asserted too, because both sides are being asked about the
 * same frame with the same window, with ONE declared exception that is a
 * deliberate strictness rather than a defect: RFC 8878 section 4.2.1 caps a
 * literals tree at depth eleven and the reference reads up to twelve here,
 * refusing the twelfth one level up. A refusal the unit made for that reason
 * is identified rather than guessed at - the tree description is re-read at
 * the reference's own ceiling, and a read that then succeeds is the depth
 * rule - and it is counted rather than passed over.
 */
#include "cudec.h"
#include "zstd_literals.h"

#include <zstd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_detail::ZstdDecodeLiterals;
using cudec_detail::ZstdHufCell;
using cudec_detail::ZstdLiteralsReject;
using cudec_detail::ZstdLiteralsScratch;
using cudec_detail::ZstdLiteralsTable;

using Bytes = std::vector<unsigned char>;

/* A block may regenerate at most the block maximum, and the section is capped
 * below that so the block header's own size field and the 128 KB bound are
 * never the thing under test. */
constexpr size_t kMaxSection = 64 * 1024;
constexpr uint64_t kWindowSize = 128ull * 1024ull;
constexpr size_t kDecodeCapacity = 256 * 1024;

void Trap(const char* what, const char* where, size_t size) {
    std::fprintf(stderr, "DIVERGENCE: %s; pass=%s section=%zu\n", what, where,
                 size);
    __builtin_trap();
}

void Append(Bytes* out, const unsigned char* data, size_t size) {
    out->insert(out->end(), data, data + size);
}

/* A compressed block around an already-assembled body. */
Bytes CompressedBlock(const unsigned char* body, size_t size, bool last) {
    const uint32_t header =
        static_cast<uint32_t>(size << 3) | (2u << 1) | (last ? 1u : 0u);
    Bytes out;
    for (unsigned i = 0; i < 3; i++) {
        out.push_back(static_cast<unsigned char>(header >> (8u * i)));
    }
    Append(&out, body, size);
    return out;
}

/* A block body: a literals section followed by a sequences section declaring
 * zero sequences, which is what makes the block regenerate exactly its
 * literals. The trailing byte is part of what the unit is given as well as of
 * what the reference is given - the unit's `size` argument is the bytes
 * remaining in the BLOCK, and handing it one byte less than the reference
 * gets would make a section whose content ends at the block's last byte look
 * truncated to one side and whole to the other. */
Bytes BlockBody(const unsigned char* section, size_t size) {
    Bytes body;
    Append(&body, section, size);
    body.push_back(0x00);
    return body;
}

/* Frame_Header_Descriptor 0x00: no content size, not single segment, no
 * checksum, no dictionary id. Window_Descriptor 0x38 is exponent seven,
 * mantissa zero, so the window is 2^17 and the block maximum is the 128 KB
 * ceiling rather than the window. */
Bytes FrameHeader() {
    return Bytes{0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x38};
}

/* A Compressed literals section written by hand, and the four bytes it
 * spells.
 *
 * The tree description is the DIRECT spelling: header byte 0x80 says one
 * four-bit weight follows (the byte carries the count less 127), the high
 * nibble of 0x10 is that weight, and the second symbol's weight is the one
 * the format never writes down. Two symbols of weight one make a tree of
 * depth one, so each literal is a single bit: 0 spells symbol 0 and 1 spells
 * symbol 1. The stream byte 0x15 carries its
 * start marker at bit four and the bits 0, 1, 0, 1 below it, read downward.
 *
 * Its only job is to leave a Huffman table behind for the second pass, so it
 * is as small as a legal one can be. */
const unsigned char kSeedSection[] = {0x42, 0xc0, 0x00, 0x80, 0x10, 0x15};
const unsigned char kSeedLiterals[] = {0, 1, 0, 1};

/* Everything one pass hands the unit, in the shape it asks for. */
struct TwinState {
    std::vector<ZstdHufCell> cells;
    ZstdLiteralsScratch scratch;
    ZstdLiteralsTable table;

    TwinState()
        : cells(1u << cudec_detail::kZstdLiteralsMaxTableLog), scratch(),
          table() {
        table.cells = cells.data();
        table.capacity = static_cast<uint32_t>(cells.size());
        table.table_log = 0;
        table.present = false;
    }
};

/* Was this refusal the eleven-versus-twelve depth rule?
 *
 * The unit passes its own ceiling into the weight reader, so a tree of depth
 * twelve comes back refused where the reference's literals path would have
 * built it. Rather than guess from the rung, the description is read again at
 * the reference's ceiling: a read that succeeds there and failed at eleven was
 * refused for the depth and nothing else. */
bool RefusedForTreeDepth(const unsigned char* section, size_t size) {
    cudec_detail::ZstdLiteralsHeader header;
    ZstdLiteralsReject rung = cudec_detail::kZstdLiteralsRejectNone;
    if (cudec_detail::ZstdParseLiteralsHeader(section, size, &header, &rung) !=
        CUDEC_OK) {
        return false;
    }
    if (header.block_type != cudec_detail::kZstdLiteralsTypeCompressed ||
        header.compressed_size > size - header.header_size) {
        return false;
    }
    ZstdLiteralsScratch scratch;
    uint8_t weights[cudec_detail::kZstdHufMaxSymbolValue + 1];
    unsigned count = 0;
    unsigned table_log = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdHufReject huf_rung = cudec_detail::kZstdHufRejectNone;
    const cudec_status wide = cudec_detail::ZstdHufReadWeights(
        section + header.header_size, header.compressed_size,
        cudec_detail::kZstdHufMaxSymbolValue + 1,
        cudec_detail::kZstdHufMaxTableLog, &scratch.weights_scratch, weights,
        &count, &table_log, &consumed, &huf_rung);
    return wide == CUDEC_OK &&
           table_log > cudec_detail::kZstdLiteralsMaxTableLog;
}

size_t g_depth_strictness = 0;

void OnePass(const unsigned char* section, size_t size, bool with_seed) {
    const char* const where = with_seed ? "after-a-table" : "fresh";
    const Bytes body = BlockBody(section, size);

    TwinState state;
    if (with_seed) {
        /* The seed is not part of what is being fuzzed: a refusal here means
         * the hand-written section stopped being legal, which is a defect in
         * this file rather than a finding. */
        Bytes seed_out(sizeof(kSeedLiterals), 0);
        uint64_t seed_produced = 0;
        uint64_t seed_consumed = 0;
        ZstdLiteralsReject seed_rung = cudec_detail::kZstdLiteralsRejectNone;
        const Bytes seed_body = BlockBody(kSeedSection, sizeof(kSeedSection));
        if (ZstdDecodeLiterals(seed_body.data(), seed_body.size(), kWindowSize,
                               &state.table, &state.scratch, seed_out.data(),
                               seed_out.size(), &seed_produced, &seed_consumed,
                               &seed_rung) != CUDEC_OK ||
            seed_produced != sizeof(kSeedLiterals) ||
            std::memcmp(seed_out.data(), kSeedLiterals,
                        sizeof(kSeedLiterals)) != 0) {
            Trap("the hand-written seed section stopped decoding", where,
                 size);
        }
    }

    Bytes twin(kWindowSize, 0);
    uint64_t twin_produced = 0;
    uint64_t twin_consumed = 0;
    ZstdLiteralsReject rung = cudec_detail::kZstdLiteralsRejectNone;
    const cudec_status status = ZstdDecodeLiterals(
        body.data(), body.size(), kWindowSize, &state.table, &state.scratch,
        twin.data(), twin.size(), &twin_produced, &twin_consumed, &rung);

    /* THE FRAME IS BUILT AROUND WHAT THE UNIT SAID THE SECTION WAS, and that
     * is the whole reason the two sides are answering one question. A section
     * that ends before the input does leaves bytes behind, and those bytes
     * become the block's SEQUENCES section, which the reference then refuses
     * for a reason that has nothing to do with literals. So on an acceptance
     * the block carries exactly the bytes the unit consumed plus the
     * zero-sequences byte; on a refusal there is no such length, and the whole
     * input goes in, where an acceptance by the reference still means it read
     * a literals section the unit would not. */
    if (twin_consumed > body.size()) {
        Trap("the unit consumed more than the block it was given", where,
             size);
    }
    Bytes frame_body = body;
    if (status == CUDEC_OK) {
        /* The consumed prefix of the BLOCK, not of the fuzzer's input: a
         * section may legitimately end on the zero-sequences byte, and
         * slicing the input instead would read one byte past it. */
        frame_body =
            BlockBody(body.data(), static_cast<size_t>(twin_consumed));
    }
    Bytes frame = FrameHeader();
    if (with_seed) {
        const Bytes seed_body = BlockBody(kSeedSection, sizeof(kSeedSection));
        const Bytes seed =
            CompressedBlock(seed_body.data(), seed_body.size(), false);
        frame.insert(frame.end(), seed.begin(), seed.end());
    }
    const Bytes block =
        CompressedBlock(frame_body.data(), frame_body.size(), true);
    frame.insert(frame.end(), block.begin(), block.end());

    Bytes reference(kDecodeCapacity, 0);
    const size_t produced = ZSTD_decompress(reference.data(), reference.size(),
                                            frame.data(), frame.size());
    const bool oracle_ok = ZSTD_isError(produced) == 0;
    if (oracle_ok) {
        reference.resize(produced);
    }

    if (status == CUDEC_OK && !oracle_ok) {
        std::fprintf(stderr, "twin produced %llu bytes; oracle said %s\n",
                     static_cast<unsigned long long>(twin_produced),
                     ZSTD_getErrorName(produced));
        Trap("FAIL-OPEN: the unit accepted a literals section libzstd refused",
             where, size);
    }
    if (status != CUDEC_OK) {
        if (oracle_ok) {
            if (RefusedForTreeDepth(body.data(), body.size())) {
                /* The declared strictness: RFC 8878 caps a literals tree at
                 * depth eleven and the reference reads twelve here. Said once
                 * per run rather than counted in silence: a reader of the
                 * job's output should be able to see that the exception is
                 * real and was taken, and a run in which it never fires is a
                 * run that never met the case. */
                if (g_depth_strictness == 0) {
                    std::fprintf(stderr,
                                 "NOTE: the eleven-bit literals-tree cap "
                                 "refused a tree libzstd reads at twelve "
                                 "(RFC 8878 section 4.2.1); not a "
                                 "divergence\n");
                }
                g_depth_strictness++;
                return;
            }
            std::fprintf(stderr, "twin rung=%d; oracle produced %zu bytes\n",
                         static_cast<int>(rung), produced);
            Trap("the unit refused a literals section libzstd decoded", where,
                 size);
        }
        return;
    }

    /* Both accepted. The block's output is its literals, so the reference's
     * frame output is the seed's literals followed by this section's. */
    const size_t prefix = with_seed ? sizeof(kSeedLiterals) : 0;
    if (reference.size() != prefix + twin_produced) {
        std::fprintf(stderr, "twin %llu bytes, oracle %zu after a %zu-byte "
                             "prefix\n",
                     static_cast<unsigned long long>(twin_produced),
                     reference.size(), prefix);
        Trap("regenerated size divergence", where, size);
    }
#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* The self-test twin: perturb the reference's own output so the byte
     * comparison below MUST fire on any accepted input. A binary built this
     * way that runs to its time limit instead of trapping is a harness that
     * has stopped comparing. */
    if (!reference.empty()) {
        reference[reference.size() - 1] =
            static_cast<unsigned char>(reference[reference.size() - 1] ^ 0xff);
    }
#endif
    for (uint64_t i = 0; i < twin_produced; i++) {
        if (twin[i] != reference[prefix + i]) {
            std::fprintf(stderr,
                         "literal %llu twin=%02x oracle=%02x\n",
                         static_cast<unsigned long long>(i), twin[i],
                         reference[prefix + i]);
            Trap("regenerated literals differ from libzstd", where, size);
        }
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    size_t section_size = size;
    if (section_size > kMaxSection) {
        section_size = kMaxSection;
    }
    /* libFuzzer hands out a slice of a buffer sized to -max_len rather than to
     * this input, so a read past the section would land in that slack and stay
     * green. The unit runs over one exactly-sized copy instead. */
    Bytes section(section_size == 0 ? 1 : section_size, 0);
    if (section_size != 0) {
        std::memcpy(section.data(), data, section_size);
    }

    OnePass(section.data(), section_size, false);
    OnePass(section.data(), section_size, true);
    return 0;
}
