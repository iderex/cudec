/* Differential fuzz target over the Zstd FSE table description decode and the
 * decoding table built from it (issue #190), the fifth target in fuzz/ and the
 * second over the M5 surface.
 *
 * NO STRUCTURE-AWARE LAYER IS BUILT HERE AND THE ISSUE ASKED FOR ONE. Its
 * reason for asking was that a fuzzer aimed at the entropy stages "spends its
 * whole budget dying at the magic number", which is true of a target that
 * enters through a frame. This one does not: ZstdFseReadNCount takes the
 * description's bytes and a length, exactly as FSE_readNCount does, so raw
 * fuzzer bytes ARE a table description and the first byte already carries the
 * accuracy log. Splicing the same bytes below a synthesised frame header would
 * add an envelope both sides have to agree about before either reaches the
 * unit under test, which buys no coverage of the unit and adds a second thing
 * that can be wrong. Recorded rather than done quietly, because it is a
 * departure from the issue's written scope.
 *
 * WHAT THE ISSUE'S SURFACE LIST REACHES AND WHAT IT DOES NOT. The accuracy-log
 * bounds, the "less than one" probability, probability-budget violations and
 * truncated descriptions are all properties of a description and are reached
 * from any byte string. The Predefined / RLE / Repeat table modes are NOT:
 * they are the Symbol_Compression_Modes byte of the sequences section, one
 * layer above this unit, and no code in this tree reads that byte yet. They
 * are named as not covered rather than approximated by something that would
 * read like coverage.
 *
 * FOUR BOUND SETS PER INPUT, BECAUSE THE BOUND IS THE CALLER'S. The unit is
 * given the alphabet and the accuracy-log ceiling by whoever calls it, and the
 * three sequence fields carry different ones (RFC 8878 section 3.1.1.3.2.2).
 * One input is therefore read four times: once at the format's absolute bounds
 * and once at each field's. That reaches the per-field accuracy-log refusal,
 * which the absolute pass by construction cannot.
 *
 * WHICH DIRECTION IS ASSERTED, AND WHERE THE SECOND ONE IS LEGITIMATE. The
 * twin accepting a description the reference refuses is a fail-open on every
 * pass and always traps. The reverse - the reference accepting where the twin
 * refuses - is asserted only on the absolute-bound pass, and the reason is not
 * caution: on that pass the two sides are given the SAME alphabet ceiling and
 * the same accuracy-log ceiling that FSE_readNCount enforces internally
 * (FSE_TABLELOG_ABSOLUTE_MAX, fifteen), so there is no declared subset between
 * them and a disagreement is a defect. On the three field passes the twin is
 * given a ceiling of nine, eight or nine, which the reference does not apply at
 * this layer at all - it refuses those one level up, in ZSTD_buildSeqTable -
 * so trapping there would report agreement as a divergence.
 *
 * THE REFERENCE IS RUN OVER A PADDED COPY, AND THAT IS NOT A CONVENIENCE. Near
 * the end of its buffer FSE_readNCount stops advancing and re-reads the last
 * four bytes at a wrapped bit offset - `bitCount &= 31` with `ip = iend-4`, in
 * lib/common/entropy_common.c of the pinned 1.5.7 tarball - so a description
 * whose bits run out inside those last bytes is read from bits the reader has
 * already used, and it can come back accepted with a consumed count smaller
 * than the position it actually reached. Measured rather than inferred: one
 * fourteen-byte string is accepted at every prefix length from eight bytes up,
 * reporting six different highest symbols and a consumed count that falls from
 * thirteen to eleven as the buffer grows by one byte. Comparing against that
 * would report the reference's own tail handling as a cudec divergence, in both
 * directions. Eight zero bytes after the description remove it: a read that
 * finishes at least seven bytes before the end never takes the clamped branch
 * at all, so an acceptance whose consumed count fits inside the REAL input is
 * an exact read of that input and is what the twin is held to.
 *
 * A PADDED ACCEPTANCE THAT DOES NOT FIT IN THE REAL INPUT IS A TRUNCATION. It
 * means the description needed bytes the caller did not supply, so the twin
 * must refuse it and only the fail-open direction is asserted there.
 *
 * THE TABLE IS DIFFED CELL BY CELL AND NOT ONLY THE VERDICT. A description
 * both sides read identically can still build a table that differs in one
 * cell, and a wrong cell is a wrong symbol rather than a refusal: silent
 * corruption is the failure mode this unit has.
 */
#include "cudec.h"
#include "zstd_fse.h"

/* The reference's FSE entry points are internal to libzstd and its fse.h
 * carries no C++ linkage guard of its own, so the include is wrapped here
 * rather than its declarations restated. FSE_STATIC_LINKING_ONLY is defined by
 * the build, which is how tests/zstd_fse_twin.cpp reaches the same symbols;
 * defining it here as well is a redefinition the strict-warning build refuses,
 * which is the lesson issue #180 landed. */
extern "C" {
#include <common/fse.h>
}

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

using cudec_detail::kZstdFseMaxAccuracyLog;
using cudec_detail::kZstdFseMaxSymbolValue;
using cudec_detail::ZstdFseBuildDTable;
using cudec_detail::ZstdFseCell;
using cudec_detail::ZstdFseReadNCount;
using cudec_detail::ZstdFseReject;

/* A table description is at most a few dozen bytes: the alphabet is capped at
 * 256 symbols and each one costs at most sixteen bits. The cap is well above
 * that, so a long input exercises the trailing-slack path rather than the
 * allocator, and it bounds nothing the decode depends on. */
constexpr size_t kMaxDescription = 512;

void Trap(const char* what, const char* where, size_t size) {
    std::fprintf(stderr, "DIVERGENCE: %s; bounds=%s stream=%zu\n", what, where,
                 size);
    __builtin_trap();
}

/* The caller-supplied bounds, one set per pass. The three field sets are the
 * ones src/zstd_fse.h carries as format constants; the first is the widest the
 * encoding can express and is the only pass where the reference is held to the
 * same ceiling. */
struct Bounds {
    const char* name;
    unsigned max_symbol_value;
    unsigned max_accuracy_log;
    bool reference_shares_the_ceiling;
};

const Bounds kBounds[] = {
    {"absolute", kZstdFseMaxSymbolValue, kZstdFseMaxAccuracyLog, true},
    {"litlen", cudec_detail::kZstdLitLenSymbolMax,
     cudec_detail::kZstdLitLenAccuracyLogMax, false},
    {"offset", cudec_detail::kZstdOffsetSymbolMax,
     cudec_detail::kZstdOffsetAccuracyLogMax, false},
    {"matchlen", cudec_detail::kZstdMatchLenSymbolMax,
     cudec_detail::kZstdMatchLenAccuracyLogMax, false},
};

/* The reference's read of one description, as a verdict plus what it read. */
struct OracleRead {
    bool ok = false;
    unsigned accuracy_log = 0;
    unsigned max_symbol = 0;
    size_t consumed = 0;
    std::vector<short> counts;
};

/* Eight, because the clamped branch is only reachable once the cursor is
 * within seven bytes of the end, and one byte more keeps a read that ends
 * exactly at the description's last byte out of it too. */
constexpr size_t kOraclePadding = 8;

OracleRead ReadWithOracle(const std::vector<unsigned char>& padded,
                          unsigned max_symbol_value) {
    OracleRead result;
    result.counts.assign(max_symbol_value + 1, 0);
    unsigned max_symbol = max_symbol_value;
    unsigned accuracy_log = 0;
    const size_t read = FSE_readNCount(result.counts.data(), &max_symbol,
                                       &accuracy_log, padded.data(),
                                       padded.size());
    if (FSE_isError(read)) {
        /* A refused read leaves the count vector in whatever state it reached,
         * and nothing below may compare against that. */
        result.counts.assign(max_symbol_value + 1, 0);
        return result;
    }
    result.ok = true;
    result.accuracy_log = accuracy_log;
    result.max_symbol = max_symbol;
    result.consumed = read;
    return result;
}

/* The reference's decoding table, flattened to the three fields per cell the
 * twin also produces. False when the reference refuses to build. */
bool BuildWithOracle(const std::vector<short>& counts, unsigned max_symbol,
                     unsigned accuracy_log, std::vector<ZstdFseCell>* out) {
    std::vector<short> narrow(counts);
    std::vector<FSE_DTable> table(FSE_DTABLE_SIZE_U32(accuracy_log), 0);
    std::vector<unsigned> work(
        FSE_BUILD_DTABLE_WKSP_SIZE_U32(kZstdFseMaxAccuracyLog,
                                       kZstdFseMaxSymbolValue),
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

void OnePass(const unsigned char* src, size_t size,
             const std::vector<unsigned char>& padded, const Bounds& bounds) {
    std::vector<int16_t> counts(bounds.max_symbol_value + 1, 0);

    unsigned twin_max_symbol = 0;
    unsigned twin_accuracy_log = 0;
    uint64_t twin_consumed = 0;
    ZstdFseReject rung = cudec_detail::kZstdFseRejectNone;
    const cudec_status twin = ZstdFseReadNCount(
        src, size, bounds.max_symbol_value, bounds.max_accuracy_log,
        counts.data(), &twin_max_symbol, &twin_accuracy_log, &twin_consumed,
        &rung);

    /* Every argument handed in is inside what the unit documents as legal, so
     * INVALID_ARGUMENT here would mean a well-formed request was answered as a
     * caller bug - the one reject class a caller cannot act on. */
    if (twin != CUDEC_OK && twin != CUDEC_ERR_CORRUPT_INPUT) {
        Trap("a description verdict outside the documented set", bounds.name,
             size);
    }

    OracleRead oracle = ReadWithOracle(padded, bounds.max_symbol_value);

#ifdef CUDEC_FUZZ_SELFTEST_BREAK
    /* Off by default, and the only way to prove the comparisons below are live
     * without waiting for a real divergence: a second binary built with this
     * defined perturbs the accepted reference read, so a harness that had
     * silently stopped comparing passes where this one traps. Never define it
     * in a build whose findings are being believed. */
    if (oracle.ok) {
        oracle.accuracy_log ^= 1u;
    }
#endif

    /* An acceptance that reached past what the caller supplied is a truncation
     * as far as the twin is concerned: the description needed bytes that are
     * not there, and only the padding made them appear. */
    const bool oracle_fits = oracle.ok && oracle.consumed <= size;

    if (twin != CUDEC_OK) {
        /* The reverse direction. On the field passes the twin carries a
         * ceiling the reference does not apply at this layer, so its refusal
         * over an acceptance is the declared bound rather than a defect; the
         * absolute pass shares both ceilings and has no such excuse. */
        if (bounds.reference_shares_the_ceiling && oracle_fits) {
            std::fprintf(stderr, "twin rung=%d oracle log=%u maxsym=%u "
                                 "consumed=%zu\n",
                         static_cast<int>(rung), oracle.accuracy_log,
                         oracle.max_symbol, oracle.consumed);
            Trap("the twin refused a description libzstd read whole at the "
                 "same bounds",
                 bounds.name, size);
        }
        return;
    }

    if (!oracle.ok) {
        Trap("FAIL-OPEN: the twin read a description libzstd refused",
             bounds.name, size);
    }
    if (!oracle_fits) {
        std::fprintf(stderr, "oracle consumed=%zu of %zu supplied\n",
                     oracle.consumed, size);
        Trap("FAIL-OPEN: the twin read a description that needs bytes the "
             "caller did not supply",
             bounds.name, size);
    }
    if (twin_accuracy_log != oracle.accuracy_log) {
        Trap("accuracy log divergence on an accepted description", bounds.name,
             size);
    }
    if (twin_max_symbol != oracle.max_symbol) {
        Trap("highest-symbol divergence on an accepted description",
             bounds.name, size);
    }
    /* The byte count is what the sequences section and the Huffman weight path
     * position themselves from, so a divergence here is a wrong cursor rather
     * than a wrong answer, and nothing downstream would report it. */
    if (twin_consumed != oracle.consumed) {
        std::fprintf(stderr, "twin consumed=%llu oracle consumed=%zu\n",
                     static_cast<unsigned long long>(twin_consumed),
                     oracle.consumed);
        Trap("consumed-byte divergence on an accepted description",
             bounds.name, size);
    }
    for (unsigned symbol = 0; symbol <= bounds.max_symbol_value; symbol++) {
        if (counts[symbol] != oracle.counts[symbol]) {
            std::fprintf(stderr, "count[%u] twin=%d oracle=%d\n", symbol,
                         static_cast<int>(counts[symbol]),
                         static_cast<int>(oracle.counts[symbol]));
            Trap("normalized-count divergence on an accepted description",
                 bounds.name, size);
        }
    }

    /* THE REFERENCE'S READER AND ITS TABLE BUILDER DO NOT SHARE A CEILING, AND
     * THE DIFFERENCE IS THE REFERENCE'S OWN. FSE_readNCount accepts an accuracy
     * log up to FSE_TABLELOG_ABSOLUTE_MAX, fifteen; FSE_buildDTable_wksp
     * refuses anything above FSE_MAX_TABLELOG, twelve, which is derived from
     * that build's FSE_MAX_MEMORY_USAGE and is a memory budget rather than a
     * statement about the count vector (lib/common/fse_decompress.c line 72 and
     * lib/common/fse.h lines 612 and 618 of the pinned 1.5.7 tarball). So a
     * description at log thirteen through fifteen is one the reference reads
     * and declines to build, and comparing the two builders there would report
     * that budget as a cudec fail-open. Measured rather than reasoned about: a
     * four-byte log-fifteen description trapped this target on its first local
     * run, before any of it reached CI.
     *
     * Above the ceiling the description parity above has already been checked
     * and is the whole claim for those inputs. */
    if (twin_accuracy_log > FSE_MAX_TABLELOG) {
        return;
    }

    /* The table the counts describe. A description both sides read identically
     * must build on both sides, so a refusal from either is a divergence.
     *
     * The cell array is sized to the accuracy log the description carries
     * rather than to the widest the format admits: an input the reader refuses
     * never reaches this line, and allocating the maximum on every one of them
     * costs the fuzzer a quarter of a megabyte per pass for nothing. The
     * capacity handed in is the real one, so the unit's own capacity refusal is
     * unaffected. */
    std::vector<ZstdFseCell> cells(static_cast<size_t>(1u)
                                   << twin_accuracy_log);
    std::vector<uint16_t> symbol_next(bounds.max_symbol_value + 1, 0);
    std::vector<ZstdFseCell> reference;
    const bool oracle_built = BuildWithOracle(oracle.counts, twin_max_symbol,
                                              twin_accuracy_log, &reference);
    const cudec_status build = ZstdFseBuildDTable(
        counts.data(), twin_max_symbol, twin_accuracy_log, cells.data(),
        static_cast<uint32_t>(cells.size()), symbol_next.data(), &rung);
    if (build == CUDEC_OK && !oracle_built) {
        Trap("FAIL-OPEN: the twin built a table libzstd refused to build",
             bounds.name, size);
    }
    if (build != CUDEC_OK) {
        if (oracle_built) {
            std::fprintf(stderr, "twin build rung=%d\n",
                         static_cast<int>(rung));
            Trap("the twin refused to build a table libzstd built", bounds.name,
                 size);
        }
        return;
    }
    for (size_t cell = 0; cell < reference.size(); cell++) {
        const ZstdFseCell& have = cells[cell];
        const ZstdFseCell& want = reference[cell];
        if (have.symbol != want.symbol || have.nb_bits != want.nb_bits ||
            have.new_state != want.new_state) {
            std::fprintf(stderr, "cell %zu twin=(%u,%u,%u) oracle=(%u,%u,%u)\n",
                         cell, have.symbol, have.nb_bits, have.new_state,
                         want.symbol, want.nb_bits, want.new_state);
            Trap("decoding-table cell divergence", bounds.name, size);
        }
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    size_t description_size = size;
    if (description_size > kMaxDescription) {
        description_size = kMaxDescription;
    }

    /* libFuzzer hands out a slice of a buffer sized to -max_len rather than to
     * this input, so a read past the description would land in that slack and
     * stay green. Both sides run over one exactly-sized copy instead. The
     * allocation is never zero-length: a null src is a caller-argument refusal
     * in the unit, which is a different rung from the empty description this
     * target wants to reach. */
    auto description = std::make_unique<unsigned char[]>(
        description_size == 0 ? 1 : description_size);
    if (description_size != 0) {
        std::memcpy(description.get(), data, description_size);
    }

    /* The reference's copy, with the trailing zeroes that keep its clamped
     * tail branch out of the comparison. Built once and read by every pass. */
    std::vector<unsigned char> padded(description_size + kOraclePadding, 0);
    if (description_size != 0) {
        std::memcpy(padded.data(), data, description_size);
    }

    for (const Bounds& bounds : kBounds) {
        OnePass(description.get(), description_size, padded, bounds);
    }
    return 0;
}
