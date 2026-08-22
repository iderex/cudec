/* The CPU twin of the Zstd sequences section (src/zstd_seq.h): the section
 * header, the four table modes on each of the three fields, and the
 * three-state interleaved decode loop (issue #196). The sibling of
 * tests/zstd_fse_twin.cpp and tests/zstd_huf_twin.cpp - the single-source unit
 * executed on the host, on the GPU-less CI runner, and held to the pinned
 * reference's verdicts.
 *
 * FIVE PROOFS, AND THEY ARE DIFFERENT IN KIND.
 *
 * THE FRAME WRITER IS PROVEN BEFORE ANYTHING USES IT. Every negative below is
 * a one-byte mutation of one accepted frame, and that frame is first handed to
 * the reference and required to decode to bytes this file computed by hand.
 * Without that step a hand-built negative would only prove that the twin
 * agrees with a writer nobody checked; with it, each malformed stream is a
 * malformed version of bytes the reference itself accepts.
 *
 * THE HAND-BUILT TRACE isolates the three orders from every table builder.
 * Three tables written cell by cell in this file, two sequences, and a bit
 * string laid out bit by bit in the comment beside it: the states are
 * initialised literals-length, offset, match-length, the extra bits are read
 * offset, match-length, literals-length, and the states are updated
 * literals-length, match-length, offset. Each field's update reads a different
 * number of bits, so any other update order hands the wrong bits to the wrong
 * state and the second sequence's symbols move. That is the one proof here
 * that does not depend on the FSE table builder being right.
 *
 * THE VARINT AND MODE BRANCHES are read off hand-written headers, one per
 * encoding and one per mode on each field, with the expected count and the
 * expected three modes written down rather than derived.
 *
 * THE CORPUS SWEEP is the only place a compressor-written sequences section
 * appears. Every compressed block of every #185 fixture that carries sequences
 * is positioned from tests/zstd_corpus.h, its three tables are loaded through
 * the twin from the section's own bytes, and its bitstream is decoded. Two
 * things are asserted: the sequence count agrees with the frame walker's, and
 * the bitstream is consumed EXACTLY. The second is what covers the state
 * update order over real FSE tables - a wrong order advances the wrong state,
 * the next symbol's cell asks for a different number of bits, and the stream
 * stops landing on its last bit.
 *
 * THE NEGATIVES are one per reject rung, with the reference's verdict asserted
 * beside the twin's wherever the negative is a whole frame. Eight of them are
 * not: a caller-argument rung, a section fragment and a table handed in from
 * outside are not streams the reference has an opinion about, and each says so
 * where it is written. The reference's refusal codes do not discriminate -
 * unrelated malformed sections all come back corruption_detected - so a
 * negative asserts that the reference refused and that the twin refused
 * through the rung the negative was written for.
 *
 * WHAT IS NOT PROVEN HERE, because it is not this unit's claim: repcode
 * resolution (offset values 1, 2 and 3 are handed on raw), and executing the
 * tuples against the literals. Both are separate sub-issues. */
#include "require.h"
#include "zstd_corpus.h"
#include "zstd_frame.h"
#include "zstd_seq.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* Every declared rung must be reached by a declared negative; the check is at
 * the bottom of main. */
bool g_reject_covered[cudec_detail::kZstdSeqRejectCount] = {false};

void CoverRung(cudec_detail::ZstdSeqReject rung) {
    if (rung != cudec_detail::kZstdSeqRejectNone) {
        g_reject_covered[rung] = true;
    }
}

/* Cell storage for the three fields, sized by each one's accuracy-log
 * ceiling, plus the tables that point at it. One object per block, which is
 * also what makes a Repeat across two calls testable: the same object carries
 * the previous block's table, a fresh one carries none. */
constexpr unsigned kLitLenCells = 1u
                                  << cudec_detail::kZstdLitLenAccuracyLogMax;
constexpr unsigned kOffsetCells = 1u
                                  << cudec_detail::kZstdOffsetAccuracyLogMax;
constexpr unsigned kMatchLenCells =
    1u << cudec_detail::kZstdMatchLenAccuracyLogMax;

struct SeqTables {
    cudec_detail::ZstdFseCell litlen_cells[kLitLenCells];
    cudec_detail::ZstdFseCell offset_cells[kOffsetCells];
    cudec_detail::ZstdFseCell matchlen_cells[kMatchLenCells];
    cudec_detail::ZstdSeqTable litlen;
    cudec_detail::ZstdSeqTable offset;
    cudec_detail::ZstdSeqTable matchlen;
    cudec_detail::ZstdSeqScratch scratch;

    SeqTables() {
        litlen.cells = litlen_cells;
        litlen.capacity = kLitLenCells;
        litlen.table_size = 0;
        litlen.accuracy_log = 0;
        litlen.present = false;
        offset.cells = offset_cells;
        offset.capacity = kOffsetCells;
        offset.table_size = 0;
        offset.accuracy_log = 0;
        offset.present = false;
        matchlen.cells = matchlen_cells;
        matchlen.capacity = kMatchLenCells;
        matchlen.table_size = 0;
        matchlen.accuracy_log = 0;
        matchlen.present = false;
    }
};

/* What one whole sequences section decoded to, or the rung it was refused
 * through. The rung is what a negative asserts on: it names where the section
 * died, which a bare status cannot - every refusal here returns the same two
 * status codes. */
struct SectionResult {
    cudec_status status = CUDEC_OK;
    cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
    cudec_detail::ZstdSeqSectionHeader header;
    std::vector<cudec_detail::ZstdSequence> sequences;
};

/* Header, three table descriptions, bitstream - the whole section, in the
 * order the format writes them. This is the caller the block loop will be, and
 * it is written here rather than in the unit for the reason the sub-issue
 * gives: the block loop is a separate claim. */
SectionResult DecodeSection(const unsigned char* section, size_t size,
                            uint64_t block_size_max, SeqTables* tables) {
    SectionResult result;
    uint64_t consumed = 0;
    result.status = cudec_detail::ZstdParseSeqSectionHeader(
        section, size, block_size_max, &result.header, &consumed,
        &result.rung);
    if (result.status != CUDEC_OK) {
        return result;
    }
    size_t position = static_cast<size_t>(consumed);
    if (result.header.sequence_count == 0) {
        return result;
    }

    const unsigned fields[] = {cudec_detail::kZstdSeqFieldLitLen,
                               cudec_detail::kZstdSeqFieldOffset,
                               cudec_detail::kZstdSeqFieldMatchLen};
    const unsigned modes[] = {result.header.litlen_mode,
                              result.header.offset_mode,
                              result.header.matchlen_mode};
    cudec_detail::ZstdSeqTable* targets[] = {
        &tables->litlen, &tables->offset, &tables->matchlen};
    for (unsigned index = 0; index < 3; index++) {
        result.status = cudec_detail::ZstdSeqLoadTable(
            fields[index], modes[index], section + position, size - position,
            &tables->scratch, targets[index], &consumed, &result.rung);
        if (result.status != CUDEC_OK) {
            return result;
        }
        position += static_cast<size_t>(consumed);
        if (position > size) {
            /* Unreachable through the unit - every description is bounded by
             * the size handed to it - and asserted rather than assumed,
             * because the arithmetic below would wrap. */
            result.status = CUDEC_ERR_CORRUPT_INPUT;
            return result;
        }
    }

    result.sequences.resize(result.header.sequence_count);
    result.status = cudec_detail::ZstdDecodeSequences(
        section + position, size - position, result.header.sequence_count,
        &tables->litlen, &tables->offset, &tables->matchlen,
        result.sequences.data(),
        static_cast<uint32_t>(result.sequences.size()), &result.rung);
    return result;
}

/* ---- The frame writer. Small on purpose: one compressed block, raw
 * literals, and whatever sequences section the caller hands in. Every field
 * below is RFC 8878 section 3.1.1, and the sizes it can express are the ones
 * these tests need rather than the ones the format allows. */

constexpr unsigned kRawLiteralsMaxSize = 31;

Bytes RawLiteralsSection(const Bytes& literals) {
    /* Size_Format 00: a one-byte header carrying the type in the low two bits
     * and a five-bit regenerated size above them. */
    Bytes section;
    section.push_back(static_cast<unsigned char>(literals.size() << 3));
    section.insert(section.end(), literals.begin(), literals.end());
    return section;
}

Bytes BuildFrame(const Bytes& literals, const Bytes& sequences,
                 unsigned content_size) {
    Bytes frame;
    /* Magic, little-endian. */
    frame.push_back(0x28);
    frame.push_back(0xB5);
    frame.push_back(0x2F);
    frame.push_back(0xFD);
    /* Frame_Header_Descriptor: Single_Segment_flag alone, so the window is the
     * content size and the frame carries no window byte. With
     * Frame_Content_Size_flag zero and Single_Segment set, the content size is
     * one byte and is not biased. */
    frame.push_back(0x20);
    frame.push_back(static_cast<unsigned char>(content_size));

    Bytes block = RawLiteralsSection(literals);
    block.insert(block.end(), sequences.begin(), sequences.end());
    /* Block_Header: Last_Block, then a two-bit type, then the size. */
    const uint32_t header =
        (static_cast<uint32_t>(block.size()) << 3) | (2u << 1) | 1u;
    frame.push_back(static_cast<unsigned char>(header & 0xFFu));
    frame.push_back(static_cast<unsigned char>((header >> 8) & 0xFFu));
    frame.push_back(static_cast<unsigned char>((header >> 16) & 0xFFu));
    frame.insert(frame.end(), block.begin(), block.end());
    return frame;
}

/* The one accepted frame every negative is a mutation of.
 *
 * Eight raw literals and one sequence, all three fields in RLE mode:
 * literals-length code 8 (baseline 8, no extra bits), offset code 2 (baseline
 * 4, two extra bits, both zero, so Offset_Value 4 and a distance of 1) and
 * match-length code 5 (baseline 8, no extra bits). The block therefore
 * regenerates "abcdefgh" followed by eight copies of 'h'.
 *
 * THE SIZES ARE NOT ARBITRARY. A single-segment frame's window is its content
 * size, and the reference bounds a block's COMPRESSED size by that window - so
 * a frame declaring seven bytes of content cannot carry an eleven-byte block,
 * whatever the block decodes to. Sixteen bytes of content against a fifteen-
 * byte block is the smallest shape of this kind that clears it.
 *
 * The bitstream is one byte. An RLE table has one cell and an accuracy log of
 * zero, so the three state initialisations read nothing and the only bits in
 * the stream are the offset's two extra bits. 0x04 is the start marker at bit
 * 2 with two zero bits below it. */
constexpr unsigned kAcceptedModesByte = 0x54;
constexpr unsigned kAcceptedContentSize = 16;

Bytes AcceptedSequencesSection() {
    Bytes section;
    section.push_back(0x01); /* Number_of_Sequences */
    section.push_back(kAcceptedModesByte);
    section.push_back(0x08); /* Literals_Lengths_Mode RLE symbol */
    section.push_back(0x02); /* Offsets_Mode RLE symbol */
    section.push_back(0x05); /* Match_Lengths_Mode RLE symbol */
    section.push_back(0x04); /* the bitstream */
    return section;
}

Bytes AcceptedLiterals() {
    return Bytes{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
}

Bytes AcceptedFrame() {
    return BuildFrame(AcceptedLiterals(), AcceptedSequencesSection(),
                      kAcceptedContentSize);
}

/* Where the sequences section starts inside the accepted frame: four magic
 * bytes, the descriptor, the content size, three block-header bytes, and the
 * literals section's header plus its eight bytes. */
constexpr size_t kSectionOffsetInFrame = 4 + 1 + 1 + 3 + 1 + 8;

/* One mutated frame, its expected rung, and whether the reference has a
 * verdict on it at all. */
struct Negative {
    const char* name;
    Bytes section;
    Bytes frame;
    bool frame_is_real;
    cudec_detail::ZstdSeqReject rung;
};

Negative MutatedFrameNegative(const char* name, size_t index,
                              unsigned char value,
                              cudec_detail::ZstdSeqReject rung) {
    Negative negative;
    negative.name = name;
    negative.section = AcceptedSequencesSection();
    negative.section[index] = value;
    negative.frame = BuildFrame(AcceptedLiterals(), negative.section,
                                kAcceptedContentSize);
    negative.frame_is_real = true;
    negative.rung = rung;
    return negative;
}

Negative SectionOnlyNegative(const char* name, const Bytes& section,
                             cudec_detail::ZstdSeqReject rung) {
    Negative negative;
    negative.name = name;
    negative.section = section;
    negative.frame_is_real = false;
    negative.rung = rung;
    return negative;
}

void PrintHex(const char* label, const Bytes& bytes) {
    std::fprintf(stderr, "%s:", label);
    for (size_t i = 0; i < bytes.size(); i++) {
        std::fprintf(stderr, " %02x", bytes[i]);
    }
    std::fputc('\n', stderr);
}

}  // namespace

int main() {
    /* ---- Step 0: the writer's own proof. The accepted frame decodes through
     * the reference to bytes computed by hand here, and through the twin to
     * the one tuple that produces them. Everything below is a mutation of
     * these bytes, so this is the step that makes the mutations mean
     * something. */
    {
        const Bytes frame = AcceptedFrame();
        Bytes decoded;
        if (!ZstdOracleDecodes(frame, &decoded)) {
            PrintHex("frame", frame);
        }
        REQUIRE(ZstdOracleDecodes(frame, &decoded));
        const Bytes expected{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                             'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h'};
        REQUIRE(decoded.size() == expected.size());
        REQUIRE(equal_bytes(decoded.data(), expected.data(), expected.size()));

        const Bytes section = AcceptedSequencesSection();
        SeqTables tables;
        const SectionResult result = DecodeSection(
            section.data(), section.size(), kAcceptedContentSize, &tables);
        REQUIRE(result.status == CUDEC_OK);
        REQUIRE(result.header.sequence_count == 1);
        REQUIRE(result.header.litlen_mode == cudec_detail::kZstdSeqModeRle);
        REQUIRE(result.header.offset_mode == cudec_detail::kZstdSeqModeRle);
        REQUIRE(result.header.matchlen_mode == cudec_detail::kZstdSeqModeRle);
        REQUIRE(result.sequences.size() == 1);
        REQUIRE(result.sequences[0].literals_length == 8);
        REQUIRE(result.sequences[0].match_length == 8);
        REQUIRE(result.sequences[0].offset_value == 4);
        /* The section really does start where the mutations index it. */
        REQUIRE(frame.size() == kSectionOffsetInFrame + section.size());
        REQUIRE(equal_bytes(frame.data() + kSectionOffsetInFrame,
                            section.data(), section.size()));
    }

    /* ---- Step 1: the extra-bit widths and the baselines they sum to, pinned
     * at every boundary RFC 8878 section 3.1.1.3.2.1.1 names. The tail rule
     * and the irregular middle meet at codes 24/25 and 42/43, and the direct
     * runs end at 15 and 31, so those are where an off-by-one lands. */
    {
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(0) == 0);
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(15) == 0);
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(16) == 1);
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(19) == 1);
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(20) == 2);
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(24) == 4);
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(25) == 6);
        REQUIRE(cudec_detail::ZstdSeqLitLenExtraBits(35) == 16);
        REQUIRE(cudec_detail::ZstdSeqLitLenBaseline(0) == 0);
        REQUIRE(cudec_detail::ZstdSeqLitLenBaseline(15) == 15);
        REQUIRE(cudec_detail::ZstdSeqLitLenBaseline(16) == 16);
        REQUIRE(cudec_detail::ZstdSeqLitLenBaseline(20) == 24);
        REQUIRE(cudec_detail::ZstdSeqLitLenBaseline(24) == 48);
        REQUIRE(cudec_detail::ZstdSeqLitLenBaseline(25) == 64);
        REQUIRE(cudec_detail::ZstdSeqLitLenBaseline(35) == 65536);

        REQUIRE(cudec_detail::ZstdSeqMatchLenExtraBits(0) == 0);
        REQUIRE(cudec_detail::ZstdSeqMatchLenExtraBits(31) == 0);
        REQUIRE(cudec_detail::ZstdSeqMatchLenExtraBits(32) == 1);
        REQUIRE(cudec_detail::ZstdSeqMatchLenExtraBits(36) == 2);
        REQUIRE(cudec_detail::ZstdSeqMatchLenExtraBits(42) == 5);
        REQUIRE(cudec_detail::ZstdSeqMatchLenExtraBits(43) == 7);
        REQUIRE(cudec_detail::ZstdSeqMatchLenExtraBits(52) == 16);
        REQUIRE(cudec_detail::ZstdSeqMatchLenBaseline(0) == 3);
        REQUIRE(cudec_detail::ZstdSeqMatchLenBaseline(31) == 34);
        REQUIRE(cudec_detail::ZstdSeqMatchLenBaseline(32) == 35);
        REQUIRE(cudec_detail::ZstdSeqMatchLenBaseline(36) == 43);
        REQUIRE(cudec_detail::ZstdSeqMatchLenBaseline(42) == 99);
        REQUIRE(cudec_detail::ZstdSeqMatchLenBaseline(43) == 131);
        REQUIRE(cudec_detail::ZstdSeqMatchLenBaseline(52) == 65539);
    }

    /* ---- Step 2: the Number_of_Sequences varint, all three encodings plus
     * the zero that ends the section. The expected counts are the format's
     * arithmetic written out rather than recomputed from the code. */
    {
        struct VarintCase {
            const char* name;
            Bytes bytes;
            uint32_t count;
            uint64_t consumed;
        };
        const VarintCase cases[] = {
            /* One byte, below the escape. */
            {"one-byte", Bytes{0x7F, kAcceptedModesByte}, 127, 2},
            /* Two bytes: ((0x80 - 128) << 8) + 0x2A. */
            {"two-byte-low", Bytes{0x80, 0x2A, kAcceptedModesByte}, 42, 3},
            /* Two bytes at the top of the form: ((0xFE - 128) << 8) + 0xFF. */
            {"two-byte-high", Bytes{0xFE, 0xFF, kAcceptedModesByte}, 32511, 3},
            /* Three bytes: 0x0201 + 0x7F00. */
            {"three-byte", Bytes{0xFF, 0x01, 0x02, kAcceptedModesByte}, 33025,
             4},
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            cudec_detail::ZstdSeqSectionHeader header;
            uint64_t consumed = 0;
            cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
            const cudec_status status =
                cudec_detail::ZstdParseSeqSectionHeader(
                    cases[i].bytes.data(), cases[i].bytes.size(),
                    cudec_detail::kZstdSeqMinMatchLength * 100000, &header,
                    &consumed, &rung);
            REQUIRE_CTX(status == CUDEC_OK, "%s", cases[i].name);
            REQUIRE_CTX(header.sequence_count == cases[i].count,
                        "%s: got %u want %u", cases[i].name,
                        header.sequence_count, cases[i].count);
            REQUIRE_CTX(consumed == cases[i].consumed, "%s", cases[i].name);
        }

        /* Zero sequences: the section is that one byte and nothing else, and
         * the modes are absent rather than defaulted from a byte that is not
         * there. */
        const Bytes empty{0x00};
        cudec_detail::ZstdSeqSectionHeader header;
        uint64_t consumed = 0;
        cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
        REQUIRE(cudec_detail::ZstdParseSeqSectionHeader(
                    empty.data(), empty.size(), 1024, &header, &consumed,
                    &rung) == CUDEC_OK);
        REQUIRE(header.sequence_count == 0);
        REQUIRE(consumed == 1);
    }

    /* ---- Step 3: the Symbol_Compression_Modes byte, every mode on every
     * field. The byte carries literals-length in the top two bits, offsets
     * next, match-lengths next, and two reserved bits at the bottom. */
    {
        for (unsigned litlen = 0; litlen < 4; litlen++) {
            for (unsigned offset = 0; offset < 4; offset++) {
                for (unsigned matchlen = 0; matchlen < 4; matchlen++) {
                    const Bytes bytes{
                        0x01, static_cast<unsigned char>(
                                  (litlen << 6) | (offset << 4) |
                                  (matchlen << 2))};
                    cudec_detail::ZstdSeqSectionHeader header;
                    uint64_t consumed = 0;
                    cudec_detail::ZstdSeqReject rung =
                        cudec_detail::kZstdSeqRejectNone;
                    REQUIRE(cudec_detail::ZstdParseSeqSectionHeader(
                                bytes.data(), bytes.size(), 1024, &header,
                                &consumed, &rung) == CUDEC_OK);
                    REQUIRE_CTX(header.litlen_mode == litlen, "%u/%u/%u",
                                litlen, offset, matchlen);
                    REQUIRE_CTX(header.offset_mode == offset, "%u/%u/%u",
                                litlen, offset, matchlen);
                    REQUIRE_CTX(header.matchlen_mode == matchlen, "%u/%u/%u",
                                litlen, offset, matchlen);
                    REQUIRE(consumed == 2);
                }
            }
        }
    }

    /* ---- Step 4: the hand-built trace. Three tables written cell by cell
     * here, so the symbol sequence is what the bit string spells and nothing
     * depends on the FSE table builder.
     *
     * The bit string, in consumption order, is
     *
     *     00 01 10 | 11 | 1 011 10 | 101 1
     *
     * and it reads: three state initialisations of two bits each, which is
     * literals-length 0, offset 1, match-length 2; the first sequence's offset
     * extra bits; the first sequence's three state updates, one bit for
     * literals-length, three for match-length and two for offset; and the last
     * sequence's offset and match-length extra bits. Eighteen bits, packed
     * into three bytes with the start marker at bit 2 of the last one.
     *
     * THE UPDATE WIDTHS ARE ALL DIFFERENT ON PURPOSE. Update the states in any
     * other order and the 1, 3 and 2 bits go to the wrong states, so the
     * second sequence reads different cells and its tuple moves. The same
     * holds for the extra-bit order: offset takes two bits here and
     * match-length and literals-length take none, so reading them in any other
     * order changes the offset. */
    {
        SeqTables tables;
        for (unsigned cell = 0; cell < 4; cell++) {
            tables.litlen_cells[cell] = {0, 0, 0};
            tables.offset_cells[cell] = {0, 0, 0};
            tables.matchlen_cells[cell] = {0, 0, 0};
        }
        /* {new_state, symbol, nb_bits} */
        tables.litlen_cells[0] = {1, 0, 1};
        tables.litlen_cells[2] = {0, 4, 0};
        tables.offset_cells[1] = {0, 2, 2};
        tables.offset_cells[2] = {0, 3, 0};
        tables.matchlen_cells[2] = {0, 0, 3};
        tables.matchlen_cells[3] = {0, 32, 0};
        tables.litlen.table_size = 4;
        tables.litlen.accuracy_log = 2;
        tables.litlen.present = true;
        tables.offset.table_size = 4;
        tables.offset.accuracy_log = 2;
        tables.offset.present = true;
        tables.matchlen.table_size = 4;
        tables.matchlen.accuracy_log = 2;
        tables.matchlen.present = true;

        const Bytes stream{0xEB, 0x6E, 0x04};
        cudec_detail::ZstdSequence decoded[2];
        cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
        REQUIRE(cudec_detail::ZstdDecodeSequences(
                    stream.data(), stream.size(), 2, &tables.litlen,
                    &tables.offset, &tables.matchlen, decoded, 2,
                    &rung) == CUDEC_OK);
        /* Sequence one: literals-length code 0, match-length code 0, offset
         * code 2 with both extra bits set. */
        REQUIRE(decoded[0].literals_length == 0);
        REQUIRE(decoded[0].match_length == 3);
        REQUIRE(decoded[0].offset_value == 7);
        /* Sequence two, after the updates: literals-length code 4,
         * match-length code 32 with one extra bit set, offset code 3 with
         * extra bits 101. */
        REQUIRE(decoded[1].literals_length == 4);
        REQUIRE(decoded[1].match_length == 36);
        REQUIRE(decoded[1].offset_value == 13);
    }

    /* ---- Step 5: the corpus sweep. Every compressed block of every fixture
     * that carries sequences, positioned by the frame walker, its three tables
     * loaded from the section's own bytes and its bitstream decoded. */
    size_t corpus_blocks = 0;
    size_t corpus_sequences = 0;
    size_t corpus_headers = 0;
    unsigned mode_seen[4] = {0, 0, 0, 0};
    {
        const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
        REQUIRE(!fixtures.empty());
        for (size_t f = 0; f < fixtures.size(); f++) {
            const ZstdFixture& fixture = fixtures[f];
            ZstdFrameShape shape;
            std::string why;
            REQUIRE_CTX(ParseZstdFrameShape(fixture.compressed, &shape, &why),
                        "%s: %s", fixture.name.c_str(), why.c_str());
            /* One set of tables per FRAME, not per block: Repeat_Mode names
             * the table the previous block of this frame left behind, and a
             * fresh set per block would refuse every Repeat the corpus
             * carries. A new frame starts with none, which is what makes the
             * Repeat-without-a-table negative below reachable. */
            SeqTables tables;
            for (size_t b = 0; b < shape.blocks.size(); b++) {
                const ZstdBlockShape& block = shape.blocks[b];
                if (block.block_type != kZstdBlockCompressed ||
                    block.sequence_count == 0) {
                    continue;
                }
                corpus_blocks++;
                corpus_sequences += block.sequence_count;
                mode_seen[block.ll_mode]++;
                mode_seen[block.of_mode]++;
                mode_seen[block.ml_mode]++;
                char where[192];
                std::snprintf(where, sizeof(where), "%s block %zu",
                              fixture.name.c_str(), b);

                /* The walker stands on the first table description, so the
                 * section header sits just above it. Its length is one mode
                 * byte plus a varint of one, two or three bytes, and which of
                 * the three it is, is not derivable from the count alone - an
                 * encoder may spell a small count in a long form. So each
                 * start is tried and the one that reproduces the walker's
                 * count and its three modes, landing exactly on the first
                 * description, is the section header. */
                bool header_agreed = false;
                for (size_t length = 1; length <= 3 && !header_agreed;
                     length++) {
                    if (block.tables_offset < length + 1) {
                        continue;
                    }
                    const size_t start = block.tables_offset - length - 1;
                    cudec_detail::ZstdSeqSectionHeader header;
                    uint64_t consumed = 0;
                    cudec_detail::ZstdSeqReject rung =
                        cudec_detail::kZstdSeqRejectNone;
                    if (cudec_detail::ZstdParseSeqSectionHeader(
                            fixture.compressed.data() + start,
                            block.block_end - start,
                            cudec_detail::kZstdBlockSizeCeiling, &header,
                            &consumed, &rung) != CUDEC_OK) {
                        continue;
                    }
                    header_agreed =
                        consumed == length + 1 &&
                        header.sequence_count == block.sequence_count &&
                        header.litlen_mode == block.ll_mode &&
                        header.offset_mode == block.of_mode &&
                        header.matchlen_mode == block.ml_mode;
                }
                REQUIRE_CTX(header_agreed,
                            "%s: no section header start reproduces the "
                            "walker's count and modes",
                            where);
                corpus_headers++;

                /* The tables and the bitstream. The three modes come from the
                 * walker so this leg tests the loader and the loop rather than
                 * the header parse, which the leg above covered. */
                size_t position = block.tables_offset;
                const unsigned fields[] = {
                    cudec_detail::kZstdSeqFieldLitLen,
                    cudec_detail::kZstdSeqFieldOffset,
                    cudec_detail::kZstdSeqFieldMatchLen};
                const unsigned modes[] = {block.ll_mode, block.of_mode,
                                          block.ml_mode};
                cudec_detail::ZstdSeqTable* targets[] = {
                    &tables.litlen, &tables.offset, &tables.matchlen};
                for (unsigned index = 0; index < 3; index++) {
                    uint64_t consumed = 0;
                    cudec_detail::ZstdSeqReject rung =
                        cudec_detail::kZstdSeqRejectNone;
                    REQUIRE_CTX(
                        cudec_detail::ZstdSeqLoadTable(
                            fields[index], modes[index],
                            fixture.compressed.data() + position,
                            block.block_end - position, &tables.scratch,
                            targets[index], &consumed, &rung) == CUDEC_OK,
                        "%s: field %u mode %u refused through rung %d", where,
                        index, modes[index], static_cast<int>(rung));
                    position += static_cast<size_t>(consumed);
                    REQUIRE_CTX(position <= block.block_end, "%s", where);
                }

                std::vector<cudec_detail::ZstdSequence> sequences(
                    block.sequence_count);
                cudec_detail::ZstdSeqReject rung =
                    cudec_detail::kZstdSeqRejectNone;
                REQUIRE_CTX(
                    cudec_detail::ZstdDecodeSequences(
                        fixture.compressed.data() + position,
                        block.block_end - position, block.sequence_count,
                        &tables.litlen, &tables.offset, &tables.matchlen,
                        sequences.data(),
                        static_cast<uint32_t>(sequences.size()),
                        &rung) == CUDEC_OK,
                    "%s: %u sequences refused through rung %d", where,
                    block.sequence_count, static_cast<int>(rung));
                for (size_t s = 0; s < sequences.size(); s++) {
                    REQUIRE_CTX(sequences[s].match_length >=
                                    cudec_detail::kZstdSeqMinMatchLength,
                                "%s: sequence %zu", where, s);
                    REQUIRE_CTX(sequences[s].offset_value >= 1, "%s", where);
                }
            }
        }
        /* A sweep that found nothing passes every assertion above, which is
         * the shape this project has been bitten by before. */
        REQUIRE(corpus_blocks > 0);
        /* Predefined and Set_Compressed are what a compressor at these sizes
         * emits; RLE and Repeat are not required to appear, and the hand-built
         * sections above are where those two are covered. */
        REQUIRE(mode_seen[kZstdTableBasic] > 0);
        REQUIRE(mode_seen[kZstdTableCompressed] > 0);
    }

    /* ---- Step 6: the negatives, one per declared rung. */
    {
        std::vector<Negative> negatives;

        /* Mutations of the accepted frame: the reference has a verdict on
         * every one of these, and each differs from a frame it accepts by one
         * byte. */
        negatives.push_back(MutatedFrameNegative(
            "modes-reserved-bits", 1, kAcceptedModesByte | 0x01,
            cudec_detail::kZstdSeqRejectModesReserved));
        negatives.push_back(MutatedFrameNegative(
            "rle-symbol-past-max", 2,
            cudec_detail::kZstdSeqLitLenSymbolCount,
            cudec_detail::kZstdSeqRejectSymbolPastMax));
        negatives.push_back(MutatedFrameNegative(
            "repeat-without-table", 1,
            (cudec_detail::kZstdSeqModeRepeat << 6) | (1u << 4) | (1u << 2),
            cudec_detail::kZstdSeqRejectRepeatWithoutTable));
        negatives.push_back(MutatedFrameNegative(
            "bitstream-without-marker", 5, 0x00,
            cudec_detail::kZstdSeqRejectBitstreamMissing));
        negatives.push_back(MutatedFrameNegative(
            "one-sequence-too-many", 0, 0x02,
            cudec_detail::kZstdSeqRejectSequenceTruncated));
        negatives.push_back(MutatedFrameNegative(
            "bitstream-not-consumed", 5, 0x0C,
            cudec_detail::kZstdSeqRejectBitstreamNotConsumed));

        /* Section fragments. The reference decodes frames, not sections, and
         * wrapping a fragment in a frame would change which field the block
         * header declares - so these assert the twin's rung alone and say so
         * rather than manufacturing a verdict. */
        negatives.push_back(SectionOnlyNegative(
            "count-two-byte-truncated", Bytes{0x80},
            cudec_detail::kZstdSeqRejectCountTruncated));
        negatives.push_back(SectionOnlyNegative(
            "count-three-byte-truncated", Bytes{0xFF, 0x01},
            cudec_detail::kZstdSeqRejectCountTruncated));
        negatives.push_back(SectionOnlyNegative(
            "modes-missing", Bytes{0x01},
            cudec_detail::kZstdSeqRejectModesTruncated));
        negatives.push_back(SectionOnlyNegative(
            "rle-symbol-missing", Bytes{0x01, kAcceptedModesByte},
            cudec_detail::kZstdSeqRejectRleTruncated));
        /* A Set_Compressed description on every field with nothing behind the
         * mode byte to read it from. */
        negatives.push_back(SectionOnlyNegative(
            "table-description-truncated",
            Bytes{0x01, static_cast<unsigned char>(
                            (cudec_detail::kZstdSeqModeCompressed << 6) |
                            (cudec_detail::kZstdSeqModeCompressed << 4) |
                            (cudec_detail::kZstdSeqModeCompressed << 2))},
            cudec_detail::kZstdSeqRejectTableDescription));
        /* Three Predefined tables want six, five and six bits of state before
         * the first sequence; this stream carries three. */
        negatives.push_back(SectionOnlyNegative(
            "state-init-truncated", Bytes{0x01, 0x00, 0x08},
            cudec_detail::kZstdSeqRejectStateInitTruncated));

        for (size_t i = 0; i < negatives.size(); i++) {
            const Negative& negative = negatives[i];
            SeqTables tables;
            const SectionResult result =
                DecodeSection(negative.section.data(), negative.section.size(),
                              kAcceptedContentSize, &tables);
            REQUIRE_CTX(result.status != CUDEC_OK, "%s was accepted",
                        negative.name);
            REQUIRE_CTX(result.rung == negative.rung,
                        "%s: refused through rung %d, want %d", negative.name,
                        static_cast<int>(result.rung),
                        static_cast<int>(negative.rung));
            CoverRung(result.rung);
            if (!negative.frame_is_real) {
                continue;
            }
            Bytes decoded;
            if (ZstdOracleDecodes(negative.frame, &decoded)) {
                PrintHex(negative.name, negative.frame);
            }
            REQUIRE_CTX(!ZstdOracleDecodes(negative.frame, &decoded),
                        "%s: the reference accepted it", negative.name);
        }

        /* The count bound. Not a mutation of the accepted frame: what it
         * refuses is a count larger than the block's regenerated size can
         * hold, and the accepted frame's block regenerates sixteen bytes, so
         * the bound is five sequences. */
        {
            cudec_detail::ZstdSeqSectionHeader header;
            uint64_t consumed = 0;
            cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
            const Bytes bytes{0x06, kAcceptedModesByte};
            REQUIRE(cudec_detail::ZstdParseSeqSectionHeader(
                        bytes.data(), bytes.size(), kAcceptedContentSize,
                        &header, &consumed, &rung) != CUDEC_OK);
            REQUIRE(rung == cudec_detail::kZstdSeqRejectCountTooLarge);
            CoverRung(rung);
            /* One below the bound is accepted, so the refusal above is the
             * bound rather than a blanket refusal. */
            const Bytes allowed{0x05, kAcceptedModesByte};
            REQUIRE(cudec_detail::ZstdParseSeqSectionHeader(
                        allowed.data(), allowed.size(), kAcceptedContentSize,
                        &header, &consumed, &rung) == CUDEC_OK);
        }

        /* A section of no bytes at all. Written against a real pointer with
         * a zero length rather than an empty vector, whose data() may be null
         * and would reach the caller-argument rung instead. */
        {
            const unsigned char anchor = 0;
            cudec_detail::ZstdSeqSectionHeader header;
            uint64_t consumed = 0;
            cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
            REQUIRE(cudec_detail::ZstdParseSeqSectionHeader(
                        &anchor, 0, 1024, &header, &consumed, &rung) !=
                    CUDEC_OK);
            REQUIRE(rung == cudec_detail::kZstdSeqRejectCountTruncated);
            CoverRung(rung);
        }

        /* A caller argument rather than a stream: no buffer at all. */
        {
            cudec_detail::ZstdSeqSectionHeader header;
            uint64_t consumed = 0;
            cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
            REQUIRE(cudec_detail::ZstdParseSeqSectionHeader(
                        0, 4, 1024, &header, &consumed, &rung) !=
                    CUDEC_OK);
            REQUIRE(rung == cudec_detail::kZstdSeqRejectBadRequest);
            CoverRung(rung);
        }

        /* A table claiming four cells over an array of one. Nothing the unit
         * builds can look like this, and the cell reads in the loop are
         * bounded by the claim rather than by the array, so it is refused
         * before the first read rather than trusted. */
        {
            cudec_detail::ZstdFseCell one_cell = {0, 0, 0};
            cudec_detail::ZstdSeqTable narrow;
            narrow.cells = &one_cell;
            narrow.capacity = 1;
            narrow.table_size = 4;
            narrow.accuracy_log = 2;
            narrow.present = true;
            const Bytes stream{0x02};
            cudec_detail::ZstdSequence decoded[1];
            cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
            REQUIRE(cudec_detail::ZstdDecodeSequences(
                        stream.data(), stream.size(), 1, &narrow, &narrow,
                        &narrow, decoded, 1, &rung) != CUDEC_OK);
            REQUIRE(rung == cudec_detail::kZstdSeqRejectTableCapacity);
            CoverRung(rung);
        }

        /* A code out of a table rather than out of a description: a
         * hand-built literals-length cell naming a symbol the alphabet does
         * not have. No description this unit accepts produces one, which is
         * why the description-side rung's negative cannot stand in for it. */
        {
            SeqTables tables;
            tables.litlen_cells[0] = {0, 40, 0};
            tables.offset_cells[0] = {0, 0, 0};
            tables.matchlen_cells[0] = {0, 0, 0};
            tables.litlen.table_size = 1;
            tables.litlen.accuracy_log = 0;
            tables.litlen.present = true;
            tables.offset.table_size = 1;
            tables.offset.accuracy_log = 0;
            tables.offset.present = true;
            tables.matchlen.table_size = 1;
            tables.matchlen.accuracy_log = 0;
            tables.matchlen.present = true;
            const Bytes stream{0x02};
            cudec_detail::ZstdSequence decoded[1];
            cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
            REQUIRE(cudec_detail::ZstdDecodeSequences(
                        stream.data(), stream.size(), 1, &tables.litlen,
                        &tables.offset, &tables.matchlen, decoded, 1,
                        &rung) != CUDEC_OK);
            REQUIRE(rung == cudec_detail::kZstdSeqRejectDecodedSymbolPastMax);
            CoverRung(rung);
        }

        /* A cell array that came from somewhere else: a table declaring four
         * cells' worth of state bits over a table of one. No description this
         * unit accepts produces one, and the reference has no verdict on a
         * table rather than a stream. */
        {
            SeqTables tables;
            tables.litlen_cells[0] = {0, 0, 0};
            tables.offset_cells[0] = {0, 0, 0};
            tables.matchlen_cells[0] = {0, 0, 0};
            tables.litlen.table_size = 1;
            tables.litlen.accuracy_log = 2;
            tables.litlen.present = true;
            tables.offset.table_size = 1;
            tables.offset.accuracy_log = 0;
            tables.offset.present = true;
            tables.matchlen.table_size = 1;
            tables.matchlen.accuracy_log = 0;
            tables.matchlen.present = true;
            const Bytes stream{0x1F};
            cudec_detail::ZstdSequence decoded[1];
            cudec_detail::ZstdSeqReject rung = cudec_detail::kZstdSeqRejectNone;
            REQUIRE(cudec_detail::ZstdDecodeSequences(
                        stream.data(), stream.size(), 1, &tables.litlen,
                        &tables.offset, &tables.matchlen, decoded, 1,
                        &rung) != CUDEC_OK);
            REQUIRE(rung == cudec_detail::kZstdSeqRejectStateOutOfTable);
            CoverRung(rung);
        }
    }

    for (int declared = cudec_detail::kZstdSeqRejectNone + 1;
         declared < cudec_detail::kZstdSeqRejectCount; declared++) {
        REQUIRE_CTX(g_reject_covered[declared],
                    "reject rung %d has no declared negative that reaches it "
                    "- add one, or the rung is untested",
                    declared);
    }

    std::printf("PASS: %zu sequences decoded across %zu corpus blocks, %zu "
                "section headers reproduced against the frame walker, %d of "
                "%d reject rungs covered by a declared negative\n",
                corpus_sequences, corpus_blocks, corpus_headers,
                static_cast<int>(cudec_detail::kZstdSeqRejectCount) - 1,
                static_cast<int>(cudec_detail::kZstdSeqRejectCount) - 1);
    return 0;
}
