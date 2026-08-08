/* The CPU twin of the Zstd FSE table description decode and the decoding
 * table built from it (src/zstd_fse.h), issue #217. The sibling of
 * tests/parser_twin.cpp, tests/snappy_parser_twin.cpp and
 * tests/zstd_bitstream_twin.cpp: the single-source unit executed on the host,
 * on the GPU-less CI runner, and held to the pinned reference's verdicts.
 *
 * Three proofs, and they are different in kind.
 *
 * THE DESCRIPTIONS ARE BUILT BY AN ENCODER THIS FILE OWNS, and the first
 * thing the test does is prove that encoder byte for byte against
 * FSE_writeNCount over every vector it is used with. Without that step a
 * hand-built negative would only prove that the twin agrees with a writer
 * nobody checked; with it, the malformed streams below are malformed
 * versions of bytes the reference itself would have written.
 *
 * THE PARITY SWEEP runs over two sources. The vectors above, which reach
 * shapes a compressor rarely emits - a single symbol taking the whole table,
 * eight "less than one" probabilities, a zero run long enough to need the
 * repeat continuation - and the #185 corpus, which is what the issue's done
 * condition names: every Set_Compressed field of every block of every
 * fixture, positioned from the frame walker in tests/zstd_corpus.h. Both
 * compare the accuracy log, the counts, the highest symbol AND the consumed
 * byte count, then diff the built table cell by cell against
 * FSE_buildDTable_wksp. A table that differs in one cell is a failure even
 * where the header parse agreed.
 *
 * THE NEGATIVES are hand-built, one per reject rung, with the oracle's
 * verdict asserted beside the twin's wherever the oracle has one at this
 * layer. Three of them it does not have one, and each says so where it is
 * written: the caller-argument rung and the two build-side rungs are not
 * streams at all. The per-field accuracy-log bound is a fourth case and a
 * different one - the oracle ACCEPTS the description and refuses it one level
 * up, in ZSTD_buildSeqTable, so that negative asserts the acceptance.
 *
 * The reference's refusal codes do not discriminate - unrelated malformed
 * descriptions all come back corruption_detected, measured on #189 - so a
 * negative here asserts that the oracle refused and that the twin refused
 * through the rung the negative was written for. Reading a reason out of the
 * oracle's code would be reading something it does not report. */
#include "require.h"
#include "zstd_corpus.h"
#include "zstd_fse.h"

/* The reference's FSE entry points are internal to libzstd: `fse.h` carries
 * no C++ linkage guard of its own, unlike the public `zstd.h` the other M5
 * tests reach for, so the include is wrapped here rather than the header
 * being copied or its declarations restated. FSE_STATIC_LINKING_ONLY is what
 * exposes FSE_buildDTable_wksp, FSE_decode_t and the two sizing macros; the
 * non-wksp FSE_buildDTable is not exported by this release. */
extern "C" {
#include <common/fse.h>
}

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;
using Counts = std::vector<int16_t>;

/* Which reject rungs a declared negative reached. Same discipline as the
 * bitstream twin: the enumeration lives once, in the header, and main()
 * requires every rung to have been named by a negative written to reach it.
 * A rung added to the unit with no negative behind it reds this test. */
bool g_reject_covered[cudec_detail::kZstdFseRejectCount] = {false};

void CoverRung(cudec_detail::ZstdFseReject rung) {
    if (rung != cudec_detail::kZstdFseRejectNone) {
        g_reject_covered[rung] = true;
    }
}

/* The forward, low-bit-first writer a table description is spelled in. The
 * mirror of the reader in src/zstd_fse.h, and proven against
 * FSE_writeNCount before anything else uses it. */
struct BitWriter {
    Bytes bytes;
    unsigned bit = 0;

    void Put(uint32_t value, unsigned count) {
        for (unsigned i = 0; i < count; i++) {
            if (bit == 0) {
                bytes.push_back(0);
            }
            if ((value >> i) & 1u) {
                bytes.back() =
                    static_cast<unsigned char>(bytes.back() | (1u << bit));
            }
            bit++;
            if (bit == 8) {
                bit = 0;
            }
        }
    }
};

/* Writes the description of a normalized count vector, following
 * FSE_writeNCount_generic in lib/compress/fse_compress.c. The two-byte flush
 * the reference performs is a buffering detail; what it produces is the same
 * bit string this writer appends, and the byte-for-byte assertion in step 1
 * of main() is what says so rather than this comment. */
Bytes EncodeNCount(const Counts& counts, unsigned accuracy_log) {
    BitWriter writer;
    const int32_t table_size = static_cast<int32_t>(1u << accuracy_log);
    /* A vector whose magnitudes do not sum to the table size is not a
     * description this writer can spell: the threshold walk at the bottom of
     * the loop divides until the remaining budget fits, and a budget that
     * went past zero never does. Refused here, so a mistyped vector reds the
     * test instead of hanging it - which is how the one in this file's first
     * draft announced itself. */
    int32_t magnitude_sum = 0;
    for (size_t symbol = 0; symbol < counts.size(); symbol++) {
        magnitude_sum += counts[symbol] < 0 ? -counts[symbol] : counts[symbol];
    }
    if (magnitude_sum != table_size) {
        return Bytes();
    }
    writer.Put(accuracy_log - cudec_detail::kZstdFseMinAccuracyLog, 4);
    int32_t remaining = table_size + 1;
    int32_t threshold = table_size;
    unsigned field_bits = accuracy_log + 1;
    size_t symbol = 0;
    bool previous_zero = false;
    while (symbol < counts.size() && remaining > 1) {
        if (previous_zero) {
            size_t start = symbol;
            while (symbol < counts.size() && counts[symbol] == 0) {
                symbol++;
            }
            if (symbol == counts.size()) {
                break;
            }
            while (symbol >= start + 3) {
                start += 3;
                writer.Put(3, 2);
            }
            writer.Put(static_cast<uint32_t>(symbol - start), 2);
        }
        int32_t count = counts[symbol];
        symbol++;
        const int32_t small_limit = (2 * threshold - 1) - remaining;
        remaining -= count < 0 ? -count : count;
        count++;
        if (count >= threshold) {
            count += small_limit;
        }
        writer.Put(static_cast<uint32_t>(count),
                   field_bits - (count < small_limit ? 1u : 0u));
        previous_zero = count == 1;
        while (remaining < threshold) {
            field_bits--;
            threshold >>= 1;
        }
    }
    return writer.bytes;
}

/* One normalized distribution and the log it is normalized to. */
struct Vector {
    const char* name;
    unsigned accuracy_log;
    Counts counts;
};

std::vector<Vector> MakeVectors() {
    std::vector<Vector> vectors;
    /* The whole table on one symbol: the widest single probability the format
     * can express at this log, and the encoding that needs the full field
     * width on its first and only field. */
    vectors.push_back({"single-symbol", 5, {32}});
    /* An ordinary spread with no absent symbols. */
    vectors.push_back({"dense", 5, {20, 6, 4, 1, 1}});
    /* One "less than one" probability - the -1 code, which is what puts a
     * symbol at the top of the table instead of into the spread walk. */
    vectors.push_back({"one-lowprob", 5, {20, 6, 4, 1, -1}});
    /* Eight of them, so the reserved top of the table is a quarter of it and
     * the spread walk has to skip over the reserved area repeatedly. */
    vectors.push_back(
        {"many-lowprob", 5, {8, 8, 8, -1, -1, -1, -1, -1, -1, -1, -1}});
    /* Absent symbols between present ones: single zeros, which the format
     * spells as a two-bit run of length zero after a zero probability. */
    vectors.push_back({"sparse", 6, {30, 0, 0, 0, 20, 0, 10, -1, -1, 0, 2}});
    /* A run of twenty-nine absent symbols, which needs the 0b11 repeat
     * continuation nine times over - the branch a dense vector never
     * reaches. */
    {
        Counts counts(31, 0);
        counts[0] = 500;
        counts[30] = 12;
        vectors.push_back({"long-zero-run", 9, counts});
    }
    /* A literal-length shaped vector at the field's own maximum log and
     * highest symbol, so the sweep covers the widest sequence table the
     * format admits. */
    {
        Counts counts(36, 0);
        counts[0] = 200;
        counts[1] = 100;
        counts[2] = 100;
        counts[3] = 50;
        counts[35] = 50;
        for (size_t symbol = 4; symbol < 14; symbol++) {
            counts[symbol] = 1;
        }
        counts[16] = -1;
        counts[17] = -1;
        vectors.push_back({"litlen-wide", 9, counts});
    }
    return vectors;
}

/* The reference's decode of a description, as a verdict plus what it read. */
struct OracleRead {
    bool ok = false;
    unsigned accuracy_log = 0;
    unsigned max_symbol = 0;
    size_t consumed = 0;
    Counts counts;
};

OracleRead ReadWithOracle(const unsigned char* src, size_t size,
                          unsigned max_symbol_value) {
    OracleRead result;
    std::vector<short> counts(max_symbol_value + 1, 0);
    unsigned max_symbol = max_symbol_value;
    unsigned accuracy_log = 0;
    const size_t read = FSE_readNCount(counts.data(), &max_symbol,
                                       &accuracy_log, src, size);
    if (FSE_isError(read)) {
        return result;
    }
    result.ok = true;
    result.accuracy_log = accuracy_log;
    result.max_symbol = max_symbol;
    result.consumed = read;
    result.counts.assign(counts.begin(), counts.end());
    return result;
}

/* The reference's decoding table, flattened to the three fields per cell that
 * the twin also produces. */
bool BuildWithOracle(const Counts& counts, unsigned max_symbol,
                     unsigned accuracy_log,
                     std::vector<cudec_detail::ZstdFseCell>* out) {
    std::vector<short> narrow(counts.begin(), counts.end());
    std::vector<FSE_DTable> table(FSE_DTABLE_SIZE_U32(accuracy_log), 0);
    std::vector<unsigned> work(
        FSE_BUILD_DTABLE_WKSP_SIZE_U32(cudec_detail::kZstdFseMaxAccuracyLog,
                                       cudec_detail::kZstdFseMaxSymbolValue),
        0);
    const size_t built = FSE_buildDTable_wksp(
        table.data(), narrow.data(), max_symbol, accuracy_log, work.data(),
        work.size() * sizeof(unsigned));
    if (FSE_isError(built)) {
        return false;
    }
    const FSE_decode_t* cells =
        reinterpret_cast<const FSE_decode_t*>(table.data() + 1);
    const size_t table_size = static_cast<size_t>(1u) << accuracy_log;
    out->resize(table_size);
    for (size_t cell = 0; cell < table_size; cell++) {
        (*out)[cell].new_state = cells[cell].newState;
        (*out)[cell].symbol = cells[cell].symbol;
        (*out)[cell].nb_bits = cells[cell].nbBits;
    }
    return true;
}

/* Storage the twin writes into. Sized for the widest table the format admits
 * so one instance serves every case; the unit itself allocates nothing. */
struct TwinStorage {
    Counts counts =
        Counts(cudec_detail::kZstdFseMaxSymbolValue + 1, 0);
    std::vector<cudec_detail::ZstdFseCell> cells = std::vector<
        cudec_detail::ZstdFseCell>(
        static_cast<size_t>(1u) << cudec_detail::kZstdFseMaxAccuracyLog);
    std::vector<uint16_t> symbol_next = std::vector<uint16_t>(
        cudec_detail::kZstdFseMaxSymbolValue + 1, 0);
};

/* Both sides over one description, every quantity compared. Returns false on
 * the first divergence and says which one it was. */
bool ParityHolds(const unsigned char* src, size_t size,
                 unsigned max_symbol_value, unsigned max_accuracy_log,
                 const char* where, size_t* cells_compared) {
    TwinStorage storage;
    unsigned twin_max_symbol = 0;
    unsigned twin_accuracy_log = 0;
    uint64_t twin_consumed = 0;
    cudec_detail::ZstdFseReject rung = cudec_detail::kZstdFseRejectNone;
    const cudec_status status = cudec_detail::ZstdFseReadNCount(
        src, size, max_symbol_value, max_accuracy_log, storage.counts.data(),
        &twin_max_symbol, &twin_accuracy_log, &twin_consumed, &rung);
    const OracleRead oracle = ReadWithOracle(src, size, max_symbol_value);

    if (!oracle.ok) {
        std::fprintf(stderr,
                     "%s: the oracle refused a description the sweep expects "
                     "it to accept\n",
                     where);
        return false;
    }
    if (status != CUDEC_OK) {
        std::fprintf(stderr, "%s: twin refused (rung %d) where the oracle "
                             "accepted\n",
                     where, static_cast<int>(rung));
        return false;
    }
    if (twin_accuracy_log != oracle.accuracy_log) {
        std::fprintf(stderr, "%s: accuracy log %u vs %u\n", where,
                     twin_accuracy_log, oracle.accuracy_log);
        return false;
    }
    if (twin_max_symbol != oracle.max_symbol) {
        std::fprintf(stderr, "%s: highest symbol %u vs %u\n", where,
                     twin_max_symbol, oracle.max_symbol);
        return false;
    }
    if (twin_consumed != oracle.consumed) {
        std::fprintf(stderr, "%s: consumed %llu vs %zu bytes\n", where,
                     static_cast<unsigned long long>(twin_consumed),
                     oracle.consumed);
        return false;
    }
    for (unsigned symbol = 0; symbol <= max_symbol_value; symbol++) {
        if (storage.counts[symbol] != oracle.counts[symbol]) {
            std::fprintf(stderr, "%s: count[%u] %d vs %d\n", where, symbol,
                         static_cast<int>(storage.counts[symbol]),
                         static_cast<int>(oracle.counts[symbol]));
            return false;
        }
    }

    /* The table the counts describe, cell by cell. */
    std::vector<cudec_detail::ZstdFseCell> reference;
    if (!BuildWithOracle(storage.counts, twin_max_symbol, twin_accuracy_log,
                         &reference)) {
        std::fprintf(stderr, "%s: the oracle refused to build the table\n",
                     where);
        return false;
    }
    const cudec_status build = cudec_detail::ZstdFseBuildDTable(
        storage.counts.data(), twin_max_symbol, twin_accuracy_log,
        storage.cells.data(), static_cast<uint32_t>(storage.cells.size()),
        storage.symbol_next.data(), &rung);
    if (build != CUDEC_OK) {
        std::fprintf(stderr, "%s: twin refused to build (rung %d)\n", where,
                     static_cast<int>(rung));
        return false;
    }
    for (size_t cell = 0; cell < reference.size(); cell++) {
        const cudec_detail::ZstdFseCell& have = storage.cells[cell];
        const cudec_detail::ZstdFseCell& want = reference[cell];
        if (have.symbol != want.symbol || have.nb_bits != want.nb_bits ||
            have.new_state != want.new_state) {
            std::fprintf(stderr,
                         "%s: cell %zu is (symbol %u, nbBits %u, newState %u) "
                         "and the oracle has (symbol %u, nbBits %u, newState "
                         "%u)\n",
                         where, cell, have.symbol, have.nb_bits,
                         have.new_state, want.symbol, want.nb_bits,
                         want.new_state);
            return false;
        }
    }
    *cells_compared += reference.size();
    return true;
}

/* The per-field bounds, in the order the three descriptions appear inside a
 * compressed block: literal lengths, offsets, match lengths. */
struct FieldBound {
    const char* name;
    unsigned symbol_max;
    unsigned accuracy_log_max;
};

const FieldBound kSequenceFields[3] = {
    {"literal-lengths", cudec_detail::kZstdLitLenSymbolMax,
     cudec_detail::kZstdLitLenAccuracyLogMax},
    {"offsets", cudec_detail::kZstdOffsetSymbolMax,
     cudec_detail::kZstdOffsetAccuracyLogMax},
    {"match-lengths", cudec_detail::kZstdMatchLenSymbolMax,
     cudec_detail::kZstdMatchLenAccuracyLogMax}};

}  // namespace

int main() {
    const std::vector<Vector> vectors = MakeVectors();

    /* 1. The writer this file uses for everything below, proven against the
     * reference's own. Byte for byte: a description that differs from what
     * FSE_writeNCount emits is not the object the negatives claim to be
     * mutations of. */
    for (const Vector& vector : vectors) {
        const Bytes mine = EncodeNCount(vector.counts, vector.accuracy_log);
        REQUIRE_CTX(!mine.empty(),
                    "%s: the vector does not sum to its own table size",
                    vector.name);
        std::vector<short> narrow(vector.counts.begin(), vector.counts.end());
        Bytes theirs(mine.size() + 16, 0xCC);
        const size_t written = FSE_writeNCount(
            theirs.data(), theirs.size(), narrow.data(),
            static_cast<unsigned>(vector.counts.size() - 1),
            vector.accuracy_log);
        REQUIRE_CTX(!FSE_isError(written),
                    "%s: the reference refused to write this vector",
                    vector.name);
        REQUIRE_CTX(written == mine.size(),
                    "%s: reference wrote %zu bytes, this file's writer wrote "
                    "%zu",
                    vector.name, written, mine.size());
        REQUIRE_CTX(std::memcmp(mine.data(), theirs.data(), written) == 0,
                    "%s: the two descriptions differ", vector.name);
    }

    /* 2. Parity over those vectors. The description is followed by slack, as
     * it is inside a real block: the caller passes the bytes remaining rather
     * than the description's own length, which is not known until it has been
     * decoded. */
    size_t cells_compared = 0;
    size_t descriptions_compared = 0;
    for (const Vector& vector : vectors) {
        Bytes description = EncodeNCount(vector.counts, vector.accuracy_log);
        const size_t exact = description.size();
        description.resize(exact + 8, 0x00);
        const unsigned symbol_max =
            static_cast<unsigned>(vector.counts.size() - 1);
        REQUIRE_CTX(ParityHolds(description.data(), description.size(),
                                symbol_max,
                                cudec_detail::kZstdFseMaxAccuracyLog,
                                vector.name, &cells_compared),
                    "%s: with trailing slack", vector.name);
        /* And again at the exact length, which is the boundary the stricter
         * reading of the last field runs into. */
        REQUIRE_CTX(ParityHolds(description.data(), exact, symbol_max,
                                cudec_detail::kZstdFseMaxAccuracyLog,
                                vector.name, &cells_compared),
                    "%s: at the exact description length", vector.name);
        descriptions_compared += 2;
    }

    /* 3. The corpus half of the issue's done condition: every Set_Compressed
     * field of every block of every #185 fixture, positioned from the frame
     * walker. The walk over the three fields is the sequences section's own
     * order, and a field in any other mode is stepped over by the width the
     * format gives it - one byte for RLE, none for Predefined or Repeat. */
    const std::vector<ZstdFixture> fixtures = MakeZstdFixtures();
    REQUIRE(!fixtures.empty());
    size_t corpus_fields = 0;
    size_t corpus_blocks = 0;
    for (const ZstdFixture& fixture : fixtures) {
        ZstdFrameShape shape;
        std::string why;
        REQUIRE_CTX(ParseZstdFrameShape(fixture.compressed, &shape, &why),
                    "%s: %s", fixture.name.c_str(), why.c_str());
        for (const ZstdBlockShape& block : shape.blocks) {
            if (block.sequence_count == 0) {
                continue;
            }
            corpus_blocks++;
            const unsigned modes[3] = {block.ll_mode, block.of_mode,
                                       block.ml_mode};
            size_t offset = block.tables_offset;
            for (unsigned field = 0; field < 3; field++) {
                REQUIRE_CTX(offset <= block.block_end, "%s: field %s starts "
                                                       "past the block",
                            fixture.name.c_str(), kSequenceFields[field].name);
                if (modes[field] == kZstdTableRle) {
                    offset += 1;
                    continue;
                }
                if (modes[field] != kZstdTableCompressed) {
                    continue;
                }
                const std::string where =
                    fixture.name + " / " + kSequenceFields[field].name;
                const unsigned char* src = fixture.compressed.data() + offset;
                const size_t size = block.block_end - offset;
                REQUIRE_CTX(
                    ParityHolds(src, size, kSequenceFields[field].symbol_max,
                                kSequenceFields[field].accuracy_log_max,
                                where.c_str(), &cells_compared),
                    "%s", where.c_str());
                /* Advancing by the twin's own consumed count is what makes
                 * the next field's position depend on it: an off-by-one there
                 * lands the following description on the wrong byte and the
                 * sweep reds, which is the downstream failure the issue names
                 * as silent corruption. */
                TwinStorage storage;
                unsigned max_symbol = 0;
                unsigned accuracy_log = 0;
                uint64_t consumed = 0;
                REQUIRE(cudec_detail::ZstdFseReadNCount(
                            src, size, kSequenceFields[field].symbol_max,
                            kSequenceFields[field].accuracy_log_max,
                            storage.counts.data(), &max_symbol, &accuracy_log,
                            &consumed, nullptr) == CUDEC_OK);
                offset += static_cast<size_t>(consumed);
                corpus_fields++;
                descriptions_compared++;
            }
        }
    }
    REQUIRE_CTX(corpus_fields > 0,
                "the corpus produced no Set_Compressed table description - "
                "the tables-compressed family has stopped carrying one, and "
                "this half of the proof would pass vacuously");

    /* 4. The negatives, one per rung.
     *
     * The base description every mutation starts from, so a mutation's effect
     * is the only difference between it and a stream the reference writes. */
    const Vector& base = vectors[1];
    const Bytes valid = EncodeNCount(base.counts, base.accuracy_log);
    const unsigned base_symbol_max =
        static_cast<unsigned>(base.counts.size() - 1);
    TwinStorage storage;
    unsigned max_symbol = 0;
    unsigned accuracy_log = 0;
    uint64_t consumed = 0;
    cudec_detail::ZstdFseReject rung = cudec_detail::kZstdFseRejectNone;

    /* An alphabet wider than a count vector can be indexed by. Not a stream:
     * the oracle has no verdict to give on a caller's argument, and this rung
     * exists so that a caller bug is refused rather than clamped into a read
     * past the end of its own array. */
    REQUIRE(cudec_detail::ZstdFseReadNCount(
                valid.data(), valid.size(),
                cudec_detail::kZstdFseMaxSymbolValue + 1,
                cudec_detail::kZstdFseMaxAccuracyLog, storage.counts.data(),
                &max_symbol, &accuracy_log, &consumed, &rung) !=
            CUDEC_OK);
    REQUIRE(rung == cudec_detail::kZstdFseRejectBadRequest);
    CoverRung(rung);

    /* No bytes at all. */
    REQUIRE(cudec_detail::ZstdFseReadNCount(
                valid.data(), 0, base_symbol_max,
                cudec_detail::kZstdFseMaxAccuracyLog, storage.counts.data(),
                &max_symbol, &accuracy_log, &consumed, &rung) != CUDEC_OK);
    REQUIRE(rung == cudec_detail::kZstdFseRejectEmptyDescription);
    CoverRung(rung);

    /* An accuracy log past the absolute maximum the format defines. The
     * four-bit field is biased by five, so a field of eleven asks for
     * sixteen, and the reference refuses it in FSE_readNCount itself. */
    {
        Bytes stream = valid;
        stream[0] = static_cast<unsigned char>((stream[0] & 0xF0u) | 0x0Bu);
        REQUIRE(cudec_detail::ZstdFseReadNCount(
                    stream.data(), stream.size(), base_symbol_max,
                    cudec_detail::kZstdFseMaxAccuracyLog,
                    storage.counts.data(), &max_symbol, &accuracy_log,
                    &consumed, &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdFseRejectAccuracyLogTooLarge);
        CoverRung(rung);
        REQUIRE(!ReadWithOracle(stream.data(), stream.size(), base_symbol_max)
                     .ok);
    }

    /* The per-field bound, which is a different refusal from the one above
     * and is the one this unit is asked to enforce. A description at accuracy
     * log ten is well-formed and FSE_readNCount accepts it; what refuses it
     * is the offsets field's own maximum of eight, which the reference
     * applies one level up in ZSTD_buildSeqTable
     * (lib/decompress/zstd_decompress_block.c). So the oracle's verdict here
     * is an ACCEPT at ten, asserted, and the twin's refusal is the bound
     * being enforced where this tree enforces it. */
    {
        Counts wide(5, 0);
        wide[0] = 900;
        wide[1] = 100;
        wide[2] = 20;
        wide[3] = 3;
        wide[4] = 1;
        const Bytes stream = EncodeNCount(wide, 10);
        const OracleRead oracle =
            ReadWithOracle(stream.data(), stream.size(), 4);
        REQUIRE(oracle.ok);
        REQUIRE(oracle.accuracy_log == 10);
        REQUIRE(cudec_detail::ZstdFseReadNCount(
                    stream.data(), stream.size(), 4,
                    cudec_detail::kZstdOffsetAccuracyLogMax,
                    storage.counts.data(), &max_symbol, &accuracy_log,
                    &consumed, &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdFseRejectAccuracyLogTooLarge);
        CoverRung(rung);
    }

    /* A description cut short mid-probability. Every prefix shorter than the
     * whole is offered, so the cut lands in the accuracy-log field, in a
     * probability and in a repeat run in turn rather than at one chosen
     * place. */
    {
        size_t truncations = 0;
        for (size_t length = 0; length < valid.size(); length++) {
            const Bytes stream(valid.begin(), valid.begin() + length);
            const cudec_status status = cudec_detail::ZstdFseReadNCount(
                stream.empty() ? valid.data() : stream.data(), length,
                base_symbol_max, cudec_detail::kZstdFseMaxAccuracyLog,
                storage.counts.data(), &max_symbol, &accuracy_log, &consumed,
                &rung);
            REQUIRE_CTX(status != CUDEC_OK,
                        "a %zu-byte prefix of a %zu-byte description was "
                        "accepted",
                        length, valid.size());
            if (length > 0) {
                REQUIRE(rung ==
                        cudec_detail::kZstdFseRejectDescriptionTruncated);
                CoverRung(rung);
                REQUIRE_CTX(
                    !ReadWithOracle(stream.data(), length, base_symbol_max).ok,
                    "the oracle accepted a %zu-byte prefix", length);
                truncations++;
            }
        }
        REQUIRE(truncations > 0);
    }

    /* THE OVERSHOOT THE ISSUE ASKS FOR IS NOT A NEGATIVE, AND THIS IS WHERE
     * IT WOULD HAVE BEEN. A description cannot claim more slots than the
     * table has: the field's own limit is derived from the budget left, so
     * the widest value the encoding can carry decodes to exactly that budget
     * and no more. src/zstd_fse.h says so at the rung that is therefore
     * absent, and this is that statement executed rather than argued - the
     * widest field the format admits at accuracy log five, which is the
     * whole table on one symbol, lands on exactly one slot left over.
     *
     * A stream that CLAIMED more would be a stream the writer above cannot
     * spell, and every bit pattern it could carry instead is one of the ones
     * the sweep already walked. */
    {
        BitWriter writer;
        writer.Put(0, 4); /* accuracy log five, a table of thirty-two */
        /* remaining 33, threshold 32, field six bits, so the limit below
         * which the narrow form is used is thirty. All six bits set is the
         * widest value the field can hold. */
        writer.Put(63, 6);
        const Bytes stream = writer.bytes;
        REQUIRE(cudec_detail::ZstdFseReadNCount(
                    stream.data(), stream.size(), base_symbol_max,
                    cudec_detail::kZstdFseMaxAccuracyLog,
                    storage.counts.data(), &max_symbol, &accuracy_log,
                    &consumed, &rung) == CUDEC_OK);
        REQUIRE(max_symbol == 0);
        REQUIRE(storage.counts[0] == 32);
        const OracleRead oracle =
            ReadWithOracle(stream.data(), stream.size(), base_symbol_max);
        REQUIRE(oracle.ok);
        REQUIRE(oracle.max_symbol == 0);
        REQUIRE(oracle.counts[0] == 32);
    }

    /* An alphabet that runs out with probability still unallocated: a valid
     * six-symbol description read against a field whose highest symbol is
     * three. The reference reports maxSymbolValue_tooSmall for this, which is
     * a refusal like any other at this layer. */
    {
        const Bytes stream = EncodeNCount(base.counts, base.accuracy_log);
        REQUIRE(cudec_detail::ZstdFseReadNCount(
                    stream.data(), stream.size(), 3,
                    cudec_detail::kZstdFseMaxAccuracyLog,
                    storage.counts.data(), &max_symbol, &accuracy_log,
                    &consumed, &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdFseRejectSymbolPastMax);
        CoverRung(rung);
        REQUIRE(!ReadWithOracle(stream.data(), stream.size(), 3).ok);
    }

    /* The same rung reached through a repeat run instead: a zero probability
     * followed by continuation codes that walk the symbol index past the
     * field's highest symbol. */
    {
        BitWriter writer;
        writer.Put(0, 4); /* accuracy log five, table of thirty-two */
        /* remaining 33, threshold 32: a stored value of one is below the
         * narrow form's limit of thirty, so it is five bits wide, and a
         * stored one is a probability of zero, which opens a run. */
        writer.Put(1, 5);
        for (unsigned code = 0; code < 4; code++) {
            writer.Put(3, 2);
        }
        const Bytes stream = writer.bytes;
        REQUIRE(cudec_detail::ZstdFseReadNCount(
                    stream.data(), stream.size(), 5,
                    cudec_detail::kZstdFseMaxAccuracyLog,
                    storage.counts.data(), &max_symbol, &accuracy_log,
                    &consumed, &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdFseRejectSymbolPastMax);
        CoverRung(rung);
        REQUIRE(!ReadWithOracle(stream.data(), stream.size(), 5).ok);
    }

    /* The build's own two rungs. Neither is reachable from a description the
     * decode above accepted, which is why they are reached by calling the
     * build directly - the caller this refuses is a future one that assembles
     * a count vector itself. */
    {
        Counts counts(2, 0);
        counts[0] = 20;
        counts[1] = 12;
        REQUIRE(cudec_detail::ZstdFseBuildDTable(
                    counts.data(), 1, 5, storage.cells.data(), 31,
                    storage.symbol_next.data(), &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdFseRejectBuildCapacity);
        CoverRung(rung);

        counts[1] = 11; /* thirty-one slots claimed of thirty-two */
        REQUIRE(cudec_detail::ZstdFseBuildDTable(
                    counts.data(), 1, 5, storage.cells.data(),
                    static_cast<uint32_t>(storage.cells.size()),
                    storage.symbol_next.data(), &rung) != CUDEC_OK);
        REQUIRE(rung ==
                cudec_detail::kZstdFseRejectBuildCountsNotNormalized);
        CoverRung(rung);

        counts[1] = 13; /* one slot too many */
        REQUIRE(cudec_detail::ZstdFseBuildDTable(
                    counts.data(), 1, 5, storage.cells.data(),
                    static_cast<uint32_t>(storage.cells.size()),
                    storage.symbol_next.data(), &rung) != CUDEC_OK);
        REQUIRE(rung ==
                cudec_detail::kZstdFseRejectBuildCountsNotNormalized);

        counts[1] = -2; /* below the "less than one" code */
        REQUIRE(cudec_detail::ZstdFseBuildDTable(
                    counts.data(), 1, 5, storage.cells.data(),
                    static_cast<uint32_t>(storage.cells.size()),
                    storage.symbol_next.data(), &rung) != CUDEC_OK);
        REQUIRE(rung ==
                cudec_detail::kZstdFseRejectBuildCountsNotNormalized);

        /* And the build's caller-argument rung, for the same reason the
         * decode's exists. */
        REQUIRE(cudec_detail::ZstdFseBuildDTable(
                    counts.data(), 1, cudec_detail::kZstdFseMaxAccuracyLog + 1,
                    storage.cells.data(),
                    static_cast<uint32_t>(storage.cells.size()),
                    storage.symbol_next.data(), &rung) != CUDEC_OK);
        REQUIRE(rung == cudec_detail::kZstdFseRejectBadRequest);
    }

    /* 5. Every rung named by a negative written to reach it, walked rather
     * than restated: a rung added to the unit with none behind it lands here
     * as a hole with its number on it. */
    for (int declared = cudec_detail::kZstdFseRejectNone + 1;
         declared < cudec_detail::kZstdFseRejectCount; declared++) {
        REQUIRE_CTX(g_reject_covered[declared],
                    "reject rung %d has no declared negative that reaches it "
                    "- add one, or the rung is untested",
                    declared);
    }

    std::printf("PASS: %zu descriptions compared against FSE_readNCount "
                "(%zu of them Set_Compressed fields inside %zu corpus "
                "blocks), %zu table cells diffed against FSE_buildDTable, "
                "%d of %d reject rungs covered by a declared negative\n",
                descriptions_compared, corpus_fields, corpus_blocks,
                cells_compared,
                static_cast<int>(cudec_detail::kZstdFseRejectCount) - 1,
                static_cast<int>(cudec_detail::kZstdFseRejectCount) - 1);
    return 0;
}
