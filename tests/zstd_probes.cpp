/* The Zstd decode behaviours the M5 ladder took from a reading of RFC 8878
 * and the zstd source, executed against the pinned oracle instead of argued
 * from the text (issue #189). It is the Snappy probes' rule applied to M5:
 * the research established these by reading, so each one becomes a permanent
 * test that RUNS, and each assertion names the section or the reference line
 * it was read from.
 *
 * Line citations are into the pinned tree, facebook/zstd 1.5.7, as fetched by
 * the URL_HASH in tests/CMakeLists.txt. They move only when that pin moves.
 *
 * Most of the streams here are hand-built byte by byte rather than compressor
 * output, for the reason the Snappy probes give: a behaviour the compressor
 * never emits is exactly the one a hostile stream will carry. The repcode
 * shift and the rep1-1 corruption below cannot be reached from ZSTD_compress
 * at all - it never emits a repcode sequence whose resolution would be zero.
 *
 * The hand-built frames use RLE sequence tables (Symbol_Compression_Mode 1,
 * RFC 8878 section 3.1.1.3.2.1.1) for all three fields. An RLE table is one
 * cell, so its state costs zero bits to initialise and zero to update
 * (ZSTD_buildSeqTable_rle, zstd_decompress_block.c:657-663), which leaves a
 * sequence bitstream carrying nothing but the extra bits under test. That is
 * what makes a frame with an exactly-known bit layout writable by hand.
 *
 * No cudec code is under test here. This is the reference's behaviour, which
 * the M5 fail-closed matrix is then allowed to lock against. */
#include "require.h"

#include <zstd.h>
#include <zstd_errors.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

void Append(Bytes* out, const Bytes& more) {
    out->insert(out->end(), more.begin(), more.end());
}

/* ---- Frame construction ------------------------------------------------
 *
 * Frame_Header_Descriptor 0x00 (RFC 8878 section 3.1.1.1.1): no content
 * size, not single-segment, no checksum, no dictionary. The frame therefore
 * carries a Window_Descriptor, and 0x00 is the smallest legal window -
 * exponent 0, mantissa 0, so windowLog 10 and a 1 KiB window (section
 * 3.1.1.1.2). Every offset probed below stays inside it, and leaving the
 * content size out keeps the header at two bytes whatever the frame
 * produces. */
Bytes FrameHeader() {
    return Bytes{0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00};
}

/* Block_Header, 3 bytes little-endian: bit 0 Last_Block, bits 1-2
 * Block_Type, bits 3-23 Block_Size (RFC 8878 section 3.1.1.2). */
Bytes BlockHeader(bool last, unsigned type, size_t size) {
    const uint32_t value = (last ? 1u : 0u) | (type << 1) |
                           (static_cast<uint32_t>(size) << 3);
    return Bytes{static_cast<unsigned char>(value & 0xff),
                 static_cast<unsigned char>((value >> 8) & 0xff),
                 static_cast<unsigned char>((value >> 16) & 0xff)};
}

Bytes RawBlock(const Bytes& payload, bool last) {
    Bytes out = BlockHeader(last, 0, payload.size());
    Append(&out, payload);
    return out;
}

/* Bits in the order the DECODER reads them. The sequence bitstream is read
 * backward from its last byte, so the writer below is where that reversal
 * lives: callers think forward and never touch a byte order. */
struct BitWriter {
    std::string bits;

    void Add(uint32_t value, unsigned count) {
        for (unsigned i = count; i > 0; i--) {
            bits.push_back(((value >> (i - 1)) & 1u) ? '1' : '0');
        }
    }
};

/* Seal a bitstream: append the end marker and pad to a byte boundary.
 *
 * BIT_initDStream reads the LAST byte, takes its highest set bit as the end
 * marker and starts reading below it (bitstream.h:292-294). So the marker is
 * the first bit written after the payload, the padding above it is zero, and
 * the byte the decoder starts at is the byte this function emits last. */
Bytes SealBitstream(const std::string& read_order) {
    std::string all = "1" + read_order;
    while (all.size() % 8 != 0) {
        all.insert(all.begin(), '0');
    }
    const size_t byte_count = all.size() / 8;
    Bytes out(byte_count, 0);
    for (size_t chunk = 0; chunk < byte_count; chunk++) {
        unsigned char byte = 0;
        for (size_t bit = 0; bit < 8; bit++) {
            byte = static_cast<unsigned char>(
                (byte << 1) | (all[chunk * 8 + bit] == '1' ? 1 : 0));
        }
        out[byte_count - 1 - chunk] = byte;
    }
    return out;
}

/* A compressed block whose three sequence tables are all RLE.
 *
 * Layout, in order (RFC 8878 sections 3.1.1.3.1 and 3.1.1.3.2): the
 * Literals_Section_Header for a Raw_Literals_Block with Size_Format 00 (one
 * byte: type in bits 0-1, one size-format bit, then a 5-bit
 * Regenerated_Size), the literal bytes, Number_Of_Sequences,
 * Symbol_Compression_Modes, then one symbol byte per field in the order
 * Literals_Lengths, Offsets, Match_Lengths, then the bitstream. */
Bytes RleSequenceBlock(const Bytes& literals, unsigned nb_seq, unsigned ll_sym,
                       unsigned of_sym, unsigned ml_sym,
                       const std::string& read_order, bool last) {
    Bytes content;
    content.push_back(static_cast<unsigned char>(literals.size() << 3));
    Append(&content, literals);
    content.push_back(static_cast<unsigned char>(nb_seq));
    /* Literals_Lengths_Mode in bits 6-7, Offsets_Mode in 4-5,
     * Match_Lengths_Mode in 2-3, bits 0-1 reserved and zero. Mode 1 is RLE
     * for each of the three. */
    content.push_back(0x54);
    content.push_back(static_cast<unsigned char>(ll_sym));
    content.push_back(static_cast<unsigned char>(of_sym));
    content.push_back(static_cast<unsigned char>(ml_sym));
    Append(&content, SealBitstream(read_order));

    Bytes out = BlockHeader(last, 2, content.size());
    Append(&out, content);
    return out;
}

/* History for the frames that need earlier output to match into: a raw block
 * of distinct-enough bytes, so a wrong offset produces wrong bytes rather
 * than accidentally-right ones. */
Bytes HistoryBytes(size_t size) {
    Bytes out;
    out.reserve(size);
    for (size_t i = 0; i < size; i++) {
        out.push_back(static_cast<unsigned char>('A' + (i % 26)));
    }
    return out;
}

/* One capacity for every call, larger than anything any frame here
 * regenerates. It is a named constant rather than a per-call number because
 * the refusal helper below has to be able to say that a refusal was NOT
 * dstSize_tooSmall - and it could not say that if the capacity were ever the
 * binding constraint. An earlier draft sized this buffer per frame, and the
 * one probe whose frame outgrew it saw every mutant "rejected" for running
 * out of room; the mutation run that caught it is in the pull request. */
const size_t kDecodeCapacity = 256 * 1024;

bool OracleDecodes(const Bytes& frame, Bytes* out) {
    Bytes decoded(kDecodeCapacity, 0);
    const size_t produced = ZSTD_decompress(decoded.data(), decoded.size(),
                                            frame.data(), frame.size());
    if (ZSTD_isError(produced)) {
        out->clear();
        return false;
    }
    decoded.resize(produced);
    *out = decoded;
    return true;
}

/* A refusal, with the reason it was refused for. `expected` is the error code
 * the pinned oracle actually returns, measured rather than predicted; the
 * assertion is what keeps a probe from passing on a rejection that has
 * nothing to do with the rule under test. dstSize_tooSmall is the one this
 * suite has already been bitten by, and it can no longer occur silently
 * because the capacity above is fixed and generous. */
bool OracleRefuses(const Bytes& frame, ZSTD_ErrorCode expected) {
    Bytes decoded(kDecodeCapacity, 0);
    const size_t produced = ZSTD_decompress(decoded.data(), decoded.size(),
                                            frame.data(), frame.size());
    if (!ZSTD_isError(produced)) {
        std::fprintf(stderr, "a frame expected to be refused decoded to "
                             "%zu bytes\n",
                     produced);
        return false;
    }
    const ZSTD_ErrorCode code = ZSTD_getErrorCode(produced);
    if (code != expected) {
        std::fprintf(stderr, "refused with error code %d, expected %d\n",
                     static_cast<int>(code), static_cast<int>(expected));
        return false;
    }
    return true;
}

/* The decoded frame's bytes, compared against history plus an expected match
 * resolved at `offset` for `length` bytes. Writing the expectation this way
 * rather than as a literal is what makes the assertion about the OFFSET: the
 * bytes come out of the history the frame itself carried. */
bool MatchesHistoryAt(const Bytes& decoded, const Bytes& history,
                      size_t offset, size_t length) {
    /* The expectation is built by reaching backwards, so an offset larger
     * than what has been produced would index before the buffer. It cannot
     * happen from the callers below, and it is refused rather than trusted
     * because this file is built under the host sanitizer gate (issue #42)
     * and a test that reads out of bounds reds it for its own defect. */
    if (offset == 0 || offset > history.size()) {
        std::fprintf(stderr, "offset %zu is outside %zu bytes of history\n",
                     offset, history.size());
        return false;
    }
    if (decoded.size() != history.size() + length) {
        std::fprintf(stderr, "decoded %zu bytes, expected %zu\n",
                     decoded.size(), history.size() + length);
        return false;
    }
    Bytes expected = history;
    for (size_t i = 0; i < length; i++) {
        expected.push_back(expected[expected.size() - offset]);
    }
    return equal_bytes(decoded.data(), expected.data(), decoded.size());
}

}  // namespace

int main() {
    REQUIRE(std::string(ZSTD_versionString()) == "1.5.7");

    /* Every refusal below comes back under this one code, which is a
     * measurement and not a prediction: the frames here are refused for four
     * different reasons - a repeat offset that resolved to zero, a jump table
     * that leaves no fourth stream, an empty stream, and a missing end marker
     * - and zstd reports the same corruption_detected for all of them. So the
     * error code is not a surface the M5 matrix can read a REASON out of, only
     * a refusal, which is worth knowing before a row is written that assumes
     * otherwise. zstd_oracle_smoke.cpp records the same shape one level up,
     * where three different malformed frames all come back srcSize_wrong. */
    const ZSTD_ErrorCode kCorrupt = ZSTD_error_corruption_detected;

    /* ---- Probe 1: the repeat-offset list is indexed differently when
     * Literals_Length is zero.
     *
     * RFC 8878 section 3.1.1.5 "Repeat Offsets": with a non-zero
     * literals length, Offset_Value 1 means the most recent offset; with a
     * literals length of zero the meaning shifts by one, so Offset_Value 1
     * means the SECOND most recent. In the reference this is the `ll0` term
     * at zstd_decompress_block.c:1300-1303, where the index into
     * prevOffset[] is the flag itself.
     *
     * The two frames below differ in one field - the Literals_Lengths RLE
     * symbol, 0 against 1 - and in the one literal byte that a length of 1
     * needs. Same Offset_Value, from the same repeat-offset history
     * {1, 4, 8} at frame start (section 3.1.1.5). If the shift were not
     * there, both would resolve to offset 1. */
    {
        const Bytes history = HistoryBytes(8);

        /* Offsets_Mode RLE symbol 0: Offset_Code 0 carries no extra bits, so
         * Offset_Value is 1 and the sequence bitstream is empty - one byte
         * holding nothing but the end marker. Match_Lengths symbol 0 is
         * Match_Length 3 (section 3.1.1.3.2.1.1, Match_Length baselines). */
        Bytes ll_zero = FrameHeader();
        Append(&ll_zero, RawBlock(history, false));
        Append(&ll_zero, RleSequenceBlock(Bytes(), 1, 0, 0, 0, "", true));

        Bytes decoded;
        REQUIRE(OracleDecodes(ll_zero, &decoded));
        /* Offset_Value 1 with Literals_Length 0 resolves to the second entry
         * of the history, 4 - not the first, 1. */
        REQUIRE(MatchesHistoryAt(decoded, history, 4, 3));

        const Bytes one_literal = {'x'};
        Bytes ll_one = FrameHeader();
        Append(&ll_one, RawBlock(history, false));
        Append(&ll_one, RleSequenceBlock(one_literal, 1, 1, 0, 0, "", true));

        REQUIRE(OracleDecodes(ll_one, &decoded));
        /* Same Offset_Value, non-zero literals length: the first entry, 1.
         * The literal is emitted before the match, so the history the match
         * reads from is the frame's output including that byte. */
        Bytes with_literal = history;
        with_literal.push_back('x');
        REQUIRE(MatchesHistoryAt(decoded, with_literal, 1, 3));
    }

    /* ---- Probe 2: Offset_Value 3 with Literals_Length 0 resolves to
     * rep1 - 1, and a resulting zero is a corrupt frame.
     *
     * RFC 8878 section 3.1.1.5 again: in the zero-literals case
     * Offset_Value 3 means "the most recent offset, minus one", and the
     * result may not be zero. The reference forces the failure rather than
     * testing for it - `temp -= !temp` at zstd_decompress_block.c:1309 turns
     * a zero into an offset no output can satisfy, and the refusal lands in
     * execSequence.
     *
     * Offset_Code 1 is the code that can express Offset_Value 3: it carries
     * one extra bit, and the decoder adds the ll0 flag to the code's
     * baseline (zstd_decompress_block.c:1306), so with a zero literals
     * length the extra bit 1 selects the rep1-1 rule. */
    {
        const Bytes history = HistoryBytes(16);

        /* At frame start rep1 is 1, so rep1 - 1 is 0: refused. */
        BitWriter one_bit;
        one_bit.Add(1, 1);
        Bytes zero_offset = FrameHeader();
        Append(&zero_offset, RawBlock(history, false));
        Append(&zero_offset,
               RleSequenceBlock(Bytes(), 1, 0, 1, 0, one_bit.bits, true));
        REQUIRE(OracleRefuses(zero_offset, kCorrupt));

        /* The same frame with a NON-zero literals length: the shift is gone,
         * Offset_Value 3 means the third repeat offset, 8, and the frame
         * decodes. One byte of the block differs between this frame and the
         * refused one - the Literals_Lengths RLE symbol - so the refusal
         * above is attributable to the rule and not to the construction. */
        const Bytes one_literal = {'x'};
        Bytes third_rep = FrameHeader();
        Append(&third_rep, RawBlock(history, false));
        Append(&third_rep,
               RleSequenceBlock(one_literal, 1, 1, 1, 0, one_bit.bits, true));
        Bytes decoded;
        REQUIRE(OracleDecodes(third_rep, &decoded));
        Bytes with_literal = history;
        with_literal.push_back('x');
        REQUIRE(MatchesHistoryAt(decoded, with_literal, 8, 3));

        /* And rep1 - 1 where it is legal: a first block whose sequence uses
         * an explicit offset moves rep1 to 5 (Offset_Code 3 carries a
         * baseline of 5 - Offset_Value 8 less the 3 the repeat slots
         * occupy), so the second block's rep1-1 sequence resolves to 4. The
         * repeat-offset history survives the block boundary, which is the
         * other half of what this probe pins. */
        BitWriter three_zero_bits;
        three_zero_bits.Add(0, 3);
        Bytes rep_minus_one = FrameHeader();
        Append(&rep_minus_one, RawBlock(history, false));
        Append(&rep_minus_one, RleSequenceBlock(Bytes(), 1, 0, 3, 0,
                                                three_zero_bits.bits, false));
        Append(&rep_minus_one,
               RleSequenceBlock(Bytes(), 1, 0, 1, 0, one_bit.bits, true));

        REQUIRE(OracleDecodes(rep_minus_one, &decoded));
        Bytes expected = history;
        for (int i = 0; i < 3; i++) {
            expected.push_back(expected[expected.size() - 5]);
        }
        for (int i = 0; i < 3; i++) {
            expected.push_back(expected[expected.size() - 4]);
        }
        REQUIRE(decoded.size() == expected.size());
        REQUIRE(equal_bytes(decoded.data(), expected.data(), decoded.size()));
    }

    /* ---- Probe 3: the jump table of a 4-stream literals section is bounded
     * by the section, and the fourth stream's size is what the bound is
     * about.
     *
     * RFC 8878 section 3.1.1.3.1.2: the jump table is three 16-bit sizes and
     * the fourth stream's size is DERIVED - total, less the three, less the
     * six bytes of the table itself. A derived length is a subtraction on
     * attacker-controlled numbers, which is why it gets a probe of its own:
     * huf_decompress.c:626 computes it, and huf_decompress.c:642 is the
     * check that catches the underflow.
     *
     * Compressor output here rather than a hand-built section: this probe is
     * about the arithmetic over a real Huffman-coded section, and the
     * mutations below are what the hand-building would have been for. */
    {
        Bytes original;
        original.reserve(64 * 1024);
        uint64_t state = 0x9e3779b97f4a7c15ull;
        while (original.size() < 64 * 1024) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            /* A byte distribution wide enough to be Huffman-coded and narrow
             * enough not to be sent raw. */
            original.push_back(static_cast<unsigned char>((state >> 40) % 12));
        }
        Bytes frame(ZSTD_compressBound(original.size()));
        const size_t written = ZSTD_compress(frame.data(), frame.size(),
                                             original.data(), original.size(),
                                             3);
        REQUIRE(!ZSTD_isError(written));
        frame.resize(written);

        const size_t header = ZSTD_frameHeaderSize(frame.data(), frame.size());
        REQUIRE(!ZSTD_isError(header));
        const size_t block_header = header;
        REQUIRE(frame.size() > block_header + 3);
        const uint32_t bh = static_cast<uint32_t>(frame[block_header]) |
                            (static_cast<uint32_t>(frame[block_header + 1])
                             << 8) |
                            (static_cast<uint32_t>(frame[block_header + 2])
                             << 16);
        /* Block_Type 2 is Compressed_Block; anything else and the literals
         * section this probe needs is not there. */
        REQUIRE(((bh >> 1) & 3u) == 2u);

        const size_t lit = block_header + 3;
        const unsigned lit_type = frame[lit] & 3u;
        const unsigned size_format = (frame[lit] >> 2) & 3u;
        /* Literals_Block_Type 2 is Compressed_Literals_Block. Size_Format 1,
         * 2 and 3 are the 4-stream forms; 0 is the single-stream one, which
         * has no jump table at all and would make this probe vacuous. */
        REQUIRE(lit_type == 2u);
        REQUIRE(size_format != 0u);

        size_t lh_size = 0;
        size_t compressed_size = 0;
        /* Five bytes are read below - the four the widest header packs its
         * fields into, plus the fifth byte the Size_Format 3 form carries.
         * Bounded here rather than after the fact. */
        REQUIRE(frame.size() > lit + 5);
        const uint32_t lh = static_cast<uint32_t>(frame[lit]) |
                            (static_cast<uint32_t>(frame[lit + 1]) << 8) |
                            (static_cast<uint32_t>(frame[lit + 2]) << 16) |
                            (static_cast<uint32_t>(frame[lit + 3]) << 24);
        if (size_format == 1u) {
            lh_size = 3;
            compressed_size = (lh >> 14) & 0x3ffu;
        } else if (size_format == 2u) {
            lh_size = 4;
            compressed_size = (lh >> 18) & 0x3fffu;
        } else {
            lh_size = 5;
            compressed_size = ((lh >> 22) & 0x3ffu) |
                              (static_cast<size_t>(frame[lit + 4]) << 10);
        }

        /* The Huffman tree description sits between the literals header and
         * the jump table, and its own first byte says how long it is (RFC
         * 8878 section 3.1.1.3.1.2, "Huffman Tree Description"): below 128 it
         * is an FSE-compressed description of that many bytes, at or above
         * 128 it is 4-bit weights, two per byte, for headerByte - 127
         * symbols. */
        const size_t tree = lit + lh_size;
        REQUIRE(frame.size() > tree);
        const unsigned tree_header = frame[tree];
        const size_t tree_size = tree_header >= 128
                                     ? 1 + (tree_header - 127 + 1) / 2
                                     : 1 + tree_header;
        const size_t jump = tree + tree_size;
        REQUIRE(compressed_size > tree_size + 6);
        const size_t streams = compressed_size - tree_size;
        REQUIRE(frame.size() > jump + 6);

        const size_t l1 = static_cast<size_t>(frame[jump]) |
                          (static_cast<size_t>(frame[jump + 1]) << 8);
        const size_t l2 = static_cast<size_t>(frame[jump + 2]) |
                          (static_cast<size_t>(frame[jump + 3]) << 8);
        const size_t l3 = static_cast<size_t>(frame[jump + 4]) |
                          (static_cast<size_t>(frame[jump + 5]) << 8);
        /* The location is confirmed before anything is mutated: the three
         * sizes plus the table itself must leave a fourth stream inside the
         * section. A probe that mutated the wrong six bytes would still see
         * rejections, and they would mean nothing. */
        REQUIRE(l1 + l2 + l3 + 6 < streams);

        Bytes decoded(original.size(), 0);
        const size_t produced = ZSTD_decompress(decoded.data(), decoded.size(),
                                                frame.data(), frame.size());
        REQUIRE(!ZSTD_isError(produced));
        REQUIRE(produced == original.size());
        REQUIRE(std::memcmp(decoded.data(), original.data(), produced) == 0);

        struct JumpMutant {
            const char* name;
            size_t first_stream;
        };
        /* Three ways the first size can put the fourth stream outside the
         * section, each one a different arithmetic edge: the sum exactly
         * consuming the section so the fourth stream is empty, the sum one
         * past it so the subtraction underflows, and a size larger than the
         * whole section. */
        const JumpMutant mutants[] = {
            {"fourth stream empty", streams - 6 - l2 - l3},
            {"sum one past the section", streams - 5 - l2 - l3},
            {"first size past the section", streams + 1},
        };
        size_t refused = 0;
        for (const JumpMutant& mutant : mutants) {
            Bytes mutated = frame;
            REQUIRE(mutant.first_stream <= 0xffffu);
            mutated[jump] = static_cast<unsigned char>(mutant.first_stream &
                                                       0xffu);
            mutated[jump + 1] =
                static_cast<unsigned char>((mutant.first_stream >> 8) & 0xffu);
            REQUIRE_CTX(OracleRefuses(mutated, kCorrupt), "%s",
                        mutant.name);
            refused++;
        }
        REQUIRE(refused == 3);

        /* A zero-length stream is refused for a different reason than the
         * three above: BIT_initDStream refuses an empty bitstream outright
         * (bitstream.h:255), before any bound on the section is consulted. */
        Bytes empty_stream = frame;
        empty_stream[jump] = 0;
        empty_stream[jump + 1] = 0;
        REQUIRE(OracleRefuses(empty_stream, kCorrupt));
    }

    /* ---- Probe 4: the end marker of a backward bitstream, at every
     * position a byte has, and the zero byte that has none.
     *
     * RFC 8878 section 3.1.1.3.2.2: a bitstream's last byte carries a set
     * bit marking where the payload ends, everything above it is padding,
     * and a final byte of zero is corrupt. The reference reads the marker as
     * the highest set bit and refuses the zero byte in the same two lines
     * (bitstream.h:292-294).
     *
     * The eight frames below place the marker at every bit position a byte
     * has. Each carries one sequence whose Offset_Code is the loop variable,
     * and an Offset_Code of N reads exactly N extra bits (section
     * 3.1.1.3.2.1.1) - so the payload is N bits, the marker lands at N mod 8,
     * and N = 8 is the case where the marker byte holds nothing else. */
    {
        const Bytes history = HistoryBytes(256);
        /* Offset_Code N with all-zero extra bits resolves to the code's
         * baseline. Codes 2 and up are explicit offsets, (1 << N) - 3, the
         * three subtracted being the repeat slots. Code 1 is not an explicit
         * offset at all: with a zero literals length its extra bit 0 selects
         * the third repeat offset, 8, which is the value at frame start. */
        const size_t resolved[9] = {0, 8, 1, 5, 13, 29, 61, 125, 253};
        size_t widths = 0;
        for (unsigned code = 1; code <= 8; code++) {
            BitWriter zeros;
            zeros.Add(0, code);
            const Bytes sealed = SealBitstream(zeros.bits);
            /* The layout the probe is about, asserted rather than assumed:
             * the marker sits at bit code % 8 of the last byte, and code 8
             * is the width that needs a second byte to hold it. */
            REQUIRE_CTX(sealed.size() == (code == 8 ? 2u : 1u), "code %u",
                        code);
            REQUIRE_CTX(sealed.back() == (1u << (code % 8)), "code %u", code);

            Bytes frame = FrameHeader();
            Append(&frame, RawBlock(history, false));
            Append(&frame, RleSequenceBlock(Bytes(), 1, 0, code, 0,
                                            zeros.bits, true));
            Bytes decoded;
            REQUIRE_CTX(OracleDecodes(frame, &decoded), "code %u", code);
            REQUIRE_CTX(MatchesHistoryAt(decoded, history, resolved[code], 3),
                        "code %u", code);
            widths++;

            /* The same frame with the marker byte zeroed. Nothing else
             * changes - same block size, same field bytes - so the refusal
             * is the missing end marker and not a length that stopped
             * agreeing. */
            Bytes no_marker = frame;
            no_marker.back() = 0;
            REQUIRE_CTX(OracleRefuses(no_marker, kCorrupt), "code %u",
                        code);
        }
        REQUIRE(widths == 8);
    }

    std::printf("PASS: 4 zstd %s decode behaviours confirmed by execution - "
                "the literals-length-zero repcode shift, rep1-1 and its zero "
                "refusal, the 4-stream jump table's derived fourth size, and "
                "the end marker at all 8 bit positions\n",
                ZSTD_versionString());
    return 0;
}
