/* The LZ4 block sequence parser and its validation ladder - the
 * fail-closed core of the decoder, single-sourced for host and device
 * (masterplan section 9). The parser owns every check; callers own the
 * copies: the CPU twin test executes them sequentially, the M1 kernel
 * fans them out across a warp. Internal header, not part of the ABI.
 *
 * Ladder contract: every value decoded from the stream is validated
 * before first use as an address, length, or offset; lengths accumulate
 * in 64-bit with the remaining-capacity bound re-checked inside every
 * 255-continuation step (each step adds at most 255 after the check;
 * reaching a wrapping accumulator would need more continuation bytes than
 * the src buffer can hold, so under the decoder contract that the
 * src_size bytes are readable, no accumulator wraps for any caller sizes);
 * success is reported ONLY at exact stream consumption after a
 * literals-only tail.
 * Edge semantics are settled empirically by oracle parity - whenever
 * liblz4 rejects, this parser must reject (tests/parser_twin.cpp). */
#ifndef CUDEC_LZ4_BLOCK_H
#define CUDEC_LZ4_BLOCK_H

#include "cudec.h"

#include <stdint.h>

/* Guarded: src/snappy_block.h defines the same macro for the same reason,
 * and a device translation unit that decodes both formats includes both
 * headers. The definitions are identical, so an unguarded second one is
 * legal rather than an error - which is exactly why it would go unnoticed
 * if they ever stopped being identical. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* LZ4 block end-of-block parsing restrictions, from the reference decoder
 * (lz4.c, LZ4_decompress_safe non-partial path): a literal run whose end
 * lands within MFLIMIT bytes of the output capacity, OR within
 * (2 offset + 1 token + LASTLITERALS) bytes of the input end, must be the
 * terminal run and must consume the input exactly. Replicating this is
 * what gives bit-exact accept/reject parity with liblz4. */
constexpr uint64_t kLz4MinTrailingSlackDst = 12;  /* MFLIMIT */
constexpr uint64_t kLz4MinTrailingSlackSrc = 8;   /* 2 + 1 + LASTLITERALS */
constexpr uint64_t kLz4LastLiterals = 5;          /* LASTLITERALS */
constexpr uint64_t kLz4RunMask = 15;              /* RUN_MASK */
constexpr uint64_t kLz4MinMatch = 4;              /* MINMATCH */
/* The token's two nibbles: the high one is the literal length, the low one is
 * match_len - MINMATCH. Either nibble reading RUN_MASK means the length
 * continues in the extension bytes that follow. */
constexpr unsigned kLz4TokenLiteralShift = 4;
/* liblz4 caps a length extension's byte reads at a margin before the input
 * end (read_variable_length's ilimit): iend - RUN_MASK for literals,
 * iend - (LASTLITERALS - 1) for matches. A decoder reading extension bytes
 * up to iend instead would ACCEPT streams liblz4 rejects (fail-open). */
constexpr uint64_t kLz4LiteralReadMargin = kLz4RunMask;         /* 15 */
constexpr uint64_t kLz4MatchReadMargin = kLz4LastLiterals - 1;  /* 4 */

/* One parsed sequence, in absolute offsets. The literals live in src;
 * the match gathers from the already-written region of dst (match_len is
 * zero only for the literals-only tail). Executing a sequence means:
 * copy literals_len bytes src[literals_src..] -> dst[literals_dst..],
 * then match_len bytes dst[match_src + (i mod offset)] -> dst[match_dst + i]
 * (equivalently a sequential chasing copy; offset = match_dst - match_src). */
struct Lz4Sequence {
    uint64_t literals_src;
    uint64_t literals_dst;
    uint64_t literals_len;
    uint64_t match_src;
    uint64_t match_dst;
    uint64_t match_len;
};

struct Lz4Parser {
    const unsigned char* src;
    uint64_t src_size;
    uint64_t dst_capacity;
    uint64_t src_pos = 0;
    uint64_t dst_pos = 0;

    /* Accumulates an LZ4 255-terminated length extension onto *length.
     * Replicates liblz4's read_variable_length exactly for accept/reject
     * parity: `read_margin` is liblz4's ilimit margin before the input end
     * (each byte read must leave src_pos <= src_size - read_margin), and
     * `initial_check` mirrors its pre-read guard on the literal path. The
     * margin comparisons add the margin to the left side so they never
     * underflow, even for src_size < read_margin. cudec adds one bound of
     * its own - the remaining dst capacity, checked before each add of at
     * most 255 - so no continuation sequence can wrap the 64-bit
     * accumulator or pass a length the destination cannot hold; this only
     * makes cudec reject sooner, never accept where liblz4 rejects. */
    CUDEC_HOST_DEVICE cudec_status AccumulateLength(uint64_t* length,
                                                    uint64_t read_margin,
                                                    bool initial_check) {
        if (initial_check && src_pos + read_margin >= src_size) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        /* Fuel: the loop's termination otherwise rests entirely on the
         * guards below staying correct, and a decoder that fails to
         * terminate hangs the device - a fail-closed violation exactly like
         * an out-of-bounds read (nvCOMP 5.3 shipped that bug class in its
         * Snappy decoder). Every step consumes one src byte and the loop is
         * entered with src_pos <= src_size, so this budget is unreachable
         * for every input the guards admit: the accept set does not move,
         * and a future guard bug degrades to a reject instead of a hang. */
        uint64_t fuel = src_size - src_pos + 1;
        unsigned char byte;
        do {
            if (fuel-- == 0) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            if (src_pos >= src_size) {
                return CUDEC_ERR_CORRUPT_INPUT; /* extension byte missing */
            }
            if (*length > dst_capacity - dst_pos) {
                return CUDEC_ERR_OUTPUT_TOO_SMALL;
            }
            byte = src[src_pos++];
            *length += byte;
            if (src_pos + read_margin > src_size) {
                return CUDEC_ERR_CORRUPT_INPUT; /* read past liblz4's ilimit */
            }
        } while (byte == 255);
        return CUDEC_OK;
    }

    /* Parses the next sequence and advances both cursors past it. Exactly
     * one of three outcomes: CUDEC_OK with *sequence filled (execute it,
     * then call again); CUDEC_OK with *done set (exact-consumption
     * success - the ONLY success exit, after a literals-only tail); or a
     * reject status. Liveness: every call consumes at least the token
     * byte, so decode loops terminate on every input - asserted per call
     * over truncated, mutated, and crafted streams in
     * tests/termination.cpp, and backed by a fuel cap in every driver
     * loop so a regression rejects rather than hangs. */
    CUDEC_HOST_DEVICE cudec_status Next(Lz4Sequence* sequence, bool* done) {
        *done = false;
        if (src_pos >= src_size) {
            return CUDEC_ERR_CORRUPT_INPUT; /* a token must exist */
        }
        const unsigned char token = src[src_pos++];

        uint64_t literals_len = token >> kLz4TokenLiteralShift;
        if (literals_len == kLz4RunMask) {
            const cudec_status status = AccumulateLength(
                &literals_len, kLz4LiteralReadMargin, /*initial_check=*/true);
            if (status != CUDEC_OK) {
                return status;
            }
        }
        if (literals_len > src_size - src_pos) {
            return CUDEC_ERR_CORRUPT_INPUT; /* literals must exist */
        }
        if (literals_len > dst_capacity - dst_pos) {
            return CUDEC_ERR_OUTPUT_TOO_SMALL;
        }

        /* End-of-block rule (both slacks are non-negative given the two
         * checks above, so no subtraction underflows even for a
         * caller-supplied SIZE_MAX capacity). A terminal run must consume
         * the input exactly - a match cannot follow this close to either
         * end. */
        const uint64_t dst_slack = dst_capacity - dst_pos - literals_len;
        const uint64_t src_slack = src_size - src_pos - literals_len;
        const bool terminal = dst_slack < kLz4MinTrailingSlackDst ||
                              src_slack < kLz4MinTrailingSlackSrc;
        if (terminal) {
            if (src_pos + literals_len != src_size) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            /* liblz4's empty-output-buffer early-out, mirrored (issue #315).
             * With dstCapacity == 0 the reference accepts exactly one stream,
             * the single byte 0x00, and rejects every other input before it
             * looks at the token. Reaching this line with dst_capacity == 0
             * means literals_len == 0 (the capacity check above admits no
             * other length) and exact consumption of a one-byte stream, so
             * `token` IS that byte: without this, every token whose literal
             * nibble is zero satisfies the terminal rule vacuously and
             * fifteen streams the reference rejects are accepted here. Only
             * capacity zero is affected - at every other capacity liblz4
             * ignores a terminal token's match nibble exactly as this parser
             * does, so no other acceptance moves. */
            if (dst_capacity == 0 && token != 0) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            sequence->literals_src = src_pos;
            sequence->literals_dst = dst_pos;
            sequence->literals_len = literals_len;
            src_pos += literals_len;
            dst_pos += literals_len;
            sequence->match_src = 0;
            sequence->match_dst = dst_pos;
            sequence->match_len = 0;
            *done = true;
            return CUDEC_OK;
        }

        sequence->literals_src = src_pos;
        sequence->literals_dst = dst_pos;
        sequence->literals_len = literals_len;
        src_pos += literals_len;
        dst_pos += literals_len;

        /* Non-terminal: src_slack >= kLz4MinTrailingSlackSrc (8) here, so
         * the 2-byte offset and the match token are present. */
        const uint64_t offset = static_cast<uint64_t>(src[src_pos]) |
                                (static_cast<uint64_t>(src[src_pos + 1]) << 8);
        src_pos += 2;
        /* offset == 0 is invalid per the LZ4 block spec. liblz4 TOLERATES
         * it (a defined self-referential copy - lz4.c silences an msan
         * warning there rather than erroring), but cudec is deliberately
         * stricter: fail-closed on spec-invalid input outranks bug-for-bug
         * parity with the reference (prime directive 1). This is the one
         * point where cudec's accept set is a strict subset of liblz4's;
         * the twin test documents and bounds it. It is also load-bearing
         * for the M1 kernel: the closed-form gather reduces by
         * offset (i mod offset), so offset == 0 reaching the device copy
         * engine would be a modulo-by-zero. Never relax this to regain
         * bug-for-bug liblz4 parity. */
        if (offset == 0) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        if (offset > dst_pos) {
            return CUDEC_ERR_CORRUPT_INPUT; /* match source underflow */
        }

        uint64_t match_len = (token & kLz4RunMask) + kLz4MinMatch;
        if ((token & kLz4RunMask) == kLz4RunMask) {
            /* MINMATCH is already in match_len; the extension
             * accumulates on top with the same in-loop bound. */
            const cudec_status status = AccumulateLength(
                &match_len, kLz4MatchReadMargin, /*initial_check=*/false);
            if (status != CUDEC_OK) {
                return status;
            }
        }
        if (match_len > dst_capacity - dst_pos) {
            return CUDEC_ERR_OUTPUT_TOO_SMALL;
        }
        /* LZ4 parsing restriction: the last LASTLITERALS bytes of a block
         * are always literals, so a match may not end within them. liblz4
         * enforces this (cpy > oend - LASTLITERALS is _output_error); a
         * decoder that skipped it would ACCEPT streams liblz4 rejects -
         * the fail-open direction. No underflow: the non-terminal path
         * guarantees dst_capacity - dst_pos >= kLz4MinTrailingSlackDst
         * (12) > kLz4LastLiterals. */
        if (match_len > dst_capacity - dst_pos - kLz4LastLiterals) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        sequence->match_src = dst_pos - offset;
        sequence->match_dst = dst_pos;
        sequence->match_len = match_len;
        dst_pos += match_len;
        return CUDEC_OK;
    }
};

}  // namespace cudec_detail

#endif /* CUDEC_LZ4_BLOCK_H */
