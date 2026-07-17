/* The CPU twin of the decoder core (masterplan section 9, ladder step 2):
 * the single-source parser + validation ladder from src/lz4_block.h,
 * executed sequentially on the host, held to oracle parity on the
 * GPU-less CI runner - the fail-closed heart of M1 lands under CI before
 * any kernel exists. Parity is the authority: whenever liblz4 rejects,
 * the twin must reject; where both accept, the bytes must match the
 * oracle's own output and size. */
#include "cudec.h"
#include "fixtures.h"
#include "lz4_block.h"
#include "require.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

/* Sequential reference execution of the parsed sequences. The match copy
 * is the chasing byte copy - reads may land on just-written bytes, which
 * is exactly LZ4's replication semantics for overlapping matches (and
 * equivalent to the kernel's closed-form modular gather). */
cudec_status TwinDecode(const std::vector<unsigned char>& stream,
                        size_t capacity, std::vector<unsigned char>* out) {
    out->assign(capacity, 0);
    /* Parse from an EXACTLY-sized copy of the stream: the mutation corpus
     * truncates by resize()-DOWN, which leaves the vector's rounded-up
     * capacity readable, so a src/lz4_block.h over-read past src_size would
     * land in that slack and leave ASan green. A tight allocation puts an ASan
     * redzone right after the last stream byte so the over-read reds instead.
     * Literals are still executed from `stream` below, but only over the
     * ranges the parser has already bound to <= src_size. */
    const size_t stream_size = stream.size();
    auto tight = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(tight.get(), stream.data(), stream_size);
    }
    cudec_detail::Lz4Parser parser{tight.get(), stream_size, capacity};
    cudec_detail::Lz4Sequence seq;
    bool done = false;
    while (true) {
        const cudec_status status = parser.Next(&seq, &done);
        if (status != CUDEC_OK) {
            return status;
        }
        for (uint64_t i = 0; i < seq.literals_len; i++) {
            (*out)[seq.literals_dst + i] = stream[seq.literals_src + i];
        }
        for (uint64_t i = 0; i < seq.match_len; i++) {
            (*out)[seq.match_dst + i] = (*out)[seq.match_src + i];
        }
        if (done) {
            break;
        }
    }
    out->resize(parser.dst_pos);
    return CUDEC_OK;
}

struct CraftedNegative {
    const char* name;
    std::vector<unsigned char> stream;
    size_t capacity;
    cudec_status expected;
};

/* One crafted stream per validation-ladder branch that liblz4 ALSO
 * rejects (offset==0 is the deliberate divergence and is tested
 * separately). Each case pins the twin's specific status and asserts the
 * oracle rejects too, so the crafted stream is demonstrably malformed and
 * the branch is demonstrably in parity. Byte layouts verified by trace. */
std::vector<CraftedNegative> MakeCraftedNegatives() {
    return {
        /* token needed, none present */
        {"empty-src", {}, 16, CUDEC_ERR_CORRUPT_INPUT},
        /* literal-length extension byte missing */
        {"literal-extension-truncated", {0xF0}, 1024,
         CUDEC_ERR_CORRUPT_INPUT},
        /* literal length accumulates past the destination capacity - the
         * stream is long enough that the extension read stays within
         * liblz4's iend-RUN_MASK bound, so the capacity check fires, not
         * the read-bound check */
        {"literal-length-past-capacity",
         {0xF0, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
         40, CUDEC_ERR_OUTPUT_TOO_SMALL},
        /* terminal run (close to the end) that does not consume the input
         * exactly - a match cannot follow this near the end */
        {"terminal-inexact-consumption", {0x10, 0x41, 0x01, 0x00}, 16,
         CUDEC_ERR_CORRUPT_INPUT},
        /* non-terminal match whose offset points before the output start */
        {"offset-beyond-written",
         {0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 16,
         CUDEC_ERR_CORRUPT_INPUT},
        /* non-terminal match whose length accumulates past capacity */
        {"match-length-past-capacity",
         {0x1F, 0x41, 0x01, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00}, 100,
         CUDEC_ERR_OUTPUT_TOO_SMALL},
    };
}

/* A match may not end within the last LASTLITERALS(5) output bytes (LZ4
 * parsing restriction). The mutation corpus never reaches this class (it
 * only truncates/drops/flips - it never grows a length), so it is swept
 * explicitly here, deriving the family from the REFERENCE restriction
 * rather than from cudec's own branches. Each stream: L literals, offset
 * 1, a length-extended match, then a T-literal terminal tail; the match
 * ends at cap - T, so T < 5 must reject on both sides and T >= 5 accept.
 * Verifies the fix that closed the fail-open divergence found in the #13
 * review. */
int SweepLastLiterals(size_t* accepts, size_t* rejects) {
    for (int L : {4, 8, 12}) {
        for (int ext : {0, 10, 100}) {
            for (int T : {4, 5, 6, 10}) {
                std::vector<unsigned char> s;
                s.push_back(static_cast<unsigned char>((L << 4) | 15));
                for (int i = 0; i < L; i++) {
                    s.push_back(static_cast<unsigned char>('A' + i % 26));
                }
                s.push_back(1); /* offset low  */
                s.push_back(0); /* offset high => offset 1 */
                s.push_back(static_cast<unsigned char>(ext));
                s.push_back(static_cast<unsigned char>(T << 4));
                for (int i = 0; i < T; i++) {
                    s.push_back(static_cast<unsigned char>('a' + i % 26));
                }
                const size_t match_len = 19 + static_cast<size_t>(ext);
                const size_t cap = static_cast<size_t>(L) + match_len +
                                   static_cast<size_t>(T);
                std::vector<unsigned char> twin_out;
                std::vector<unsigned char> oracle_out;
                const bool twin_ok =
                    TwinDecode(s, cap, &twin_out) == CUDEC_OK;
                const bool oracle_ok = OracleDecodes(s, cap, &oracle_out);
                REQUIRE_CTX(twin_ok == oracle_ok,
                            "last-5 parity: L=%d ext=%d T=%d", L, ext, T);
                REQUIRE_CTX(twin_ok == (T >= 5),
                            "last-5 expectation: L=%d ext=%d T=%d", L, ext, T);
                if (twin_ok) {
                    REQUIRE_CTX(twin_out.size() == oracle_out.size() &&
                                    equal_bytes(twin_out.data(),
                                                oracle_out.data(),
                                                twin_out.size()),
                                "last-5 bytes: L=%d ext=%d T=%d", L, ext, T);
                    (*accepts)++;
                } else {
                    (*rejects)++;
                }
            }
        }
    }
    return 0;
}

/* liblz4 caps a length-extension's byte reads at a margin before the input
 * end (iend - RUN_MASK for literals, iend - (LASTLITERALS-1) for matches);
 * a decoder reading up to the input end instead ACCEPTS streams liblz4
 * rejects. Neither the mutation corpus nor the last-5 sweep grows an
 * extension near end-of-source, so this class is swept here in both
 * directions - the second fail-open family found in the #13 review. */
int SweepExtensionReadBound(size_t* accepts, size_t* rejects) {
    /* Match-extension family: 1 literal, offset 1, a match extension of
     * `ff255s` 0xFF bytes plus a terminator, then a T-literal tail. As T
     * shrinks, the extension end crosses liblz4's iend-4 bound. */
    for (int ff255s : {0, 1, 2}) {
        for (int last : {0, 30, 200}) {
            for (int T : {0, 1, 2, 3, 4, 6}) {
                std::vector<unsigned char> s = {0x1F, 0x41, 0x01, 0x00};
                for (int i = 0; i < ff255s; i++) {
                    s.push_back(0xFF);
                }
                s.push_back(static_cast<unsigned char>(last));
                s.push_back(static_cast<unsigned char>(T << 4));
                for (int i = 0; i < T; i++) {
                    s.push_back(static_cast<unsigned char>('a' + i));
                }
                const size_t match_len = 4 + 15 +
                                         static_cast<size_t>(ff255s) * 255 +
                                         static_cast<size_t>(last);
                const size_t cap = 1 + match_len + static_cast<size_t>(T);
                std::vector<unsigned char> t_out;
                std::vector<unsigned char> o_out;
                const bool t_ok = TwinDecode(s, cap, &t_out) == CUDEC_OK;
                const bool o_ok = OracleDecodes(s, cap, &o_out);
                REQUIRE_CTX(t_ok == o_ok,
                            "match-ext parity: ff=%d last=%d T=%d", ff255s,
                            last, T);
                if (t_ok) {
                    REQUIRE_CTX(t_out.size() == o_out.size() &&
                                    equal_bytes(t_out.data(), o_out.data(),
                                                t_out.size()),
                                "match-ext bytes: ff=%d last=%d T=%d", ff255s,
                                last, T);
                    (*accepts)++;
                } else {
                    (*rejects)++;
                }
            }
        }
    }
    /* Literal-extension family: an extension of `ff255s` 0xFF plus a
     * terminator, its literals, and a capacity sweep across iend-15. */
    for (int ff255s : {0, 1}) {
        for (int last : {0, 30, 200}) {
            std::vector<unsigned char> base = {0xF0};
            for (int i = 0; i < ff255s; i++) {
                base.push_back(0xFF);
            }
            base.push_back(static_cast<unsigned char>(last));
            const size_t lit_len = 15 + static_cast<size_t>(ff255s) * 255 +
                                   static_cast<size_t>(last);
            for (size_t i = 0; i < lit_len; i++) {
                base.push_back(static_cast<unsigned char>('A' + i % 26));
            }
            for (size_t cap = lit_len; cap <= lit_len + 18; cap++) {
                std::vector<unsigned char> t_out;
                std::vector<unsigned char> o_out;
                const bool t_ok = TwinDecode(base, cap, &t_out) == CUDEC_OK;
                const bool o_ok = OracleDecodes(base, cap, &o_out);
                REQUIRE_CTX(t_ok == o_ok,
                            "literal-ext parity: ff=%d last=%d cap=%zu",
                            ff255s, last, cap);
                if (t_ok) {
                    (*accepts)++;
                } else {
                    (*rejects)++;
                }
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    const auto fixtures = MakeLz4BlockFixtures();
    REQUIRE(!fixtures.empty());

    /* Every fixture decodes to its original through the twin. */
    for (const auto& f : fixtures) {
        std::vector<unsigned char> out;
        REQUIRE_CTX(TwinDecode(f.compressed, f.original.size(), &out) ==
                        CUDEC_OK,
                    "fixture %s", f.name.c_str());
        REQUIRE_CTX(out.size() == f.original.size(), "fixture %s",
                    f.name.c_str());
        REQUIRE_CTX(equal_bytes(out.data(), f.original.data(), out.size()),
                    "fixture %s", f.name.c_str());
    }

    /* Full mutant parity, as the two security-critical directions of a
     * fail-closed decoder:
     *   1. where the twin ACCEPTS, liblz4 must accept and the bytes and
     *      size must match - the twin never invents an acceptance;
     *   2. where liblz4 REJECTS, the twin must reject - never more lenient.
     * Together these make the twin's accept set a subset of liblz4's, with
     * bit-exact output on it. The one allowed gap (twin rejects a stream
     * liblz4 accepts) is the deliberate offset==0 strictness; every such
     * case must still be a DEFINED reject status, and their count is
     * reported so the divergence stays visible, never silent. */
    size_t mutant_total = 0;
    size_t rejected_total = 0;
    size_t stricter_than_oracle = 0;
    for (const auto& f : fixtures) {
        for (const auto& m : MutateStream(f.compressed, f.seed)) {
            std::vector<unsigned char> oracle_out;
            const bool oracle_accepts =
                OracleDecodes(m.stream, f.original.size(), &oracle_out);
            std::vector<unsigned char> twin_out;
            const cudec_status twin_status =
                TwinDecode(m.stream, f.original.size(), &twin_out);
            mutant_total++;
            if (twin_status == CUDEC_OK) {
                /* Direction 1: an acceptance is never invented. */
                REQUIRE_CTX(oracle_accepts,
                            "twin accepts what liblz4 rejects: %s/%s",
                            f.name.c_str(), m.description.c_str());
                REQUIRE_CTX(twin_out.size() == oracle_out.size(),
                            "size parity: %s/%s", f.name.c_str(),
                            m.description.c_str());
                REQUIRE_CTX(equal_bytes(twin_out.data(), oracle_out.data(),
                                        twin_out.size()),
                            "byte parity: %s/%s", f.name.c_str(),
                            m.description.c_str());
            } else {
                /* A reject is always a defined status. */
                REQUIRE_CTX(twin_status == CUDEC_ERR_CORRUPT_INPUT ||
                                twin_status == CUDEC_ERR_OUTPUT_TOO_SMALL,
                            "undefined reject status %d: %s/%s",
                            static_cast<int>(twin_status), f.name.c_str(),
                            m.description.c_str());
                if (oracle_accepts) {
                    stricter_than_oracle++;
                } else {
                    rejected_total++;
                }
            }
        }
    }
    REQUIRE(mutant_total > 0);
    REQUIRE(rejected_total > 0);
    /* Golden bound over the deterministic corpus: the ONLY streams the
     * twin rejects while liblz4 accepts are the offset==0 family. This one
     * value (like the fixture digest in oracle_lz4) turns "offset==0 is the
     * sole divergence" from a printed number into an enforced invariant -
     * a regression that over-rejects a class of valid streams changes it
     * and reds the gate. Re-derive the divergence before touching it. */
    REQUIRE(stricter_than_oracle == 1);

    /* The last-LASTLITERALS match restriction, swept in both directions
     * (the mutation corpus cannot reach this class). */
    size_t last5_accepts = 0;
    size_t last5_rejects = 0;
    REQUIRE(SweepLastLiterals(&last5_accepts, &last5_rejects) == 0);
    REQUIRE(last5_accepts > 0);
    REQUIRE(last5_rejects > 0);

    /* The length-extension read-bound restriction, swept in both
     * directions (also unreachable by the mutation corpus). */
    size_t ext_accepts = 0;
    size_t ext_rejects = 0;
    REQUIRE(SweepExtensionReadBound(&ext_accepts, &ext_rejects) == 0);
    REQUIRE(ext_accepts > 0);
    REQUIRE(ext_rejects > 0);

    /* One crafted negative per ladder branch that liblz4 also rejects:
     * pins the twin's specific status AND that the stream is malformed. */
    const auto crafted = MakeCraftedNegatives();
    for (const auto& c : crafted) {
        std::vector<unsigned char> twin_out;
        REQUIRE_CTX(TwinDecode(c.stream, c.capacity, &twin_out) == c.expected,
                    "crafted %s", c.name);
        std::vector<unsigned char> oracle_out;
        REQUIRE_CTX(!OracleDecodes(c.stream, c.capacity, &oracle_out),
                    "crafted %s: oracle unexpectedly accepts", c.name);
    }

    /* The deliberate divergence, pinned explicitly: a non-terminal match
     * with offset==0 is spec-invalid; the twin rejects it fail-closed
     * even though liblz4 tolerates it (src/lz4_block.h). Oracle behavior
     * is intentionally NOT asserted here - liblz4 accepting it is the
     * whole reason this case is documented. */
    {
        const std::vector<unsigned char> zero_offset = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        std::vector<unsigned char> twin_out;
        REQUIRE(TwinDecode(zero_offset, 16, &twin_out) ==
                CUDEC_ERR_CORRUPT_INPUT);
    }

    /* The empty block (a lone zero token) is valid LZ4: both sides must
     * accept it with zero output bytes. */
    {
        std::vector<unsigned char> twin_out;
        REQUIRE(TwinDecode({0x00}, 16, &twin_out) == CUDEC_OK);
        REQUIRE(twin_out.empty());
        std::vector<unsigned char> oracle_out;
        REQUIRE(OracleDecodes({0x00}, 16, &oracle_out));
        REQUIRE(oracle_out.empty());
    }

    /* Capacity beyond the 64 KiB project convention: the frozen ABI
     * allows any size_t capacity, and the ladder must stay correct there
     * (the anti-pattern rule from masterplan section 9 - no field may be
     * narrowed to a convention the ABI does not enforce). */
    {
        const auto& f = fixtures.front();
        const size_t big_capacity = size_t{1} << 20;
        std::vector<unsigned char> twin_out;
        REQUIRE(TwinDecode(f.compressed, big_capacity, &twin_out) ==
                CUDEC_OK);
        std::vector<unsigned char> oracle_out;
        REQUIRE(OracleDecodes(f.compressed, big_capacity, &oracle_out));
        REQUIRE(twin_out.size() == oracle_out.size());
        REQUIRE(equal_bytes(twin_out.data(), oracle_out.data(),
                            twin_out.size()));
    }

    /* SIZE_MAX capacity, parser only (the twin cannot allocate that much
     * output, and the oracle casts capacity to int): drive the ladder
     * directly to prove the slack subtractions and the length accumulator
     * stay correct at the boundary the header comment names. A valid
     * stream must still decode to exactly its content size. */
    {
        const auto& f = fixtures.back();
        cudec_detail::Lz4Parser parser{f.compressed.data(),
                                       f.compressed.size(), SIZE_MAX};
        cudec_detail::Lz4Sequence seq;
        bool done = false;
        while (true) {
            const cudec_status status = parser.Next(&seq, &done);
            REQUIRE(status == CUDEC_OK);
            if (done) {
                break;
            }
        }
        REQUIRE(parser.dst_pos == f.original.size());
    }

    std::printf("PASS: %zu fixtures + %zu mutants in oracle parity "
                "(%zu oracle-rejected, %zu stricter-than-liblz4 on "
                "spec-invalid input); %zu crafted negatives; last-5 sweep "
                "%zu accept / %zu reject; empty block, offset==0 "
                "divergence, SIZE_MAX and beyond-convention capacity "
                "pinned; ext-read sweep %zu accept / %zu reject\n",
                fixtures.size(), mutant_total, rejected_total,
                stricter_than_oracle, crafted.size(), last5_accepts,
                last5_rejects, ext_accepts, ext_rejects);
    return 0;
}
