/* The one element vocabulary every byte-oriented format parser hands the
 * chunk decoder (masterplan section 10, "The seam"). Internal header, not
 * part of the ABI.
 *
 * There is exactly one of these on purpose. The copy engine in
 * src/chunk_decode.cuh performs no input-derived bound check of its own -
 * all validation lives in the parser - so what stands between a parser bug
 * and an out-of-bounds write is that the engine has ONE contract a reviewer
 * can hold in their head, not one per format. A per-parser element type
 * was proposed at the #85 design panel and refused for that reason, and
 * this header is that refusal executed rather than restated.
 *
 * Executing a sequence means: copy literals_len bytes
 * src[literals_src..] -> dst[literals_dst..], then match_len bytes
 * dst[match_src + (i mod offset)] -> dst[match_dst + i], where
 * offset = match_dst - match_src and is at least 1 by every parser's
 * offset-zero refusal. Either half may be empty, which is what lets one
 * struct carry two formats' element shapes:
 *
 *   LZ4      a literal run followed by a match; both halves usually used,
 *            match_len zero only on the literals-only tail.
 *   Snappy   a literal OR a copy, never both: a literal fills the literals
 *            half and leaves match_len zero, a copy fills the match half
 *            and leaves literals_len zero.
 *
 * Absolute offsets, and 64-bit throughout. Snappy's declared length is a
 * varint32 so its positions provably fit in 32 bits, but the caller's
 * capacity is a size_t and mixing the two widths in a bound comparison is
 * where an overflow would hide.
 *
 * THE DEVICE CARRIES THEM AT THIS WIDTH TOO, MEASURED RATHER THAN INHERITED.
 * Issue #292 asked whether the kernel should hold a narrowed copy of this
 * struct. Both arms were built and run: narrowing the six fields to 32 bits
 * takes the Snappy instantiation from 56 registers to 48 and from 9 resident
 * blocks per SM to 10 on sm_86, and it costs 4-8% of decode throughput on
 * every recorded corpus while the parse-only ceiling does not move. The copy
 * loops index global memory, so a narrow field is widened again at the point
 * of use, and the occupancy the narrowing buys is occupancy this kernel was
 * not short of. docs/BENCHMARKS.md carries both arms and the mechanism. */
#ifndef CUDEC_DECODE_SEQUENCE_H
#define CUDEC_DECODE_SEQUENCE_H

#include <stdint.h>

/* Defined here rather than in each format header, so a translation unit
 * that decodes two formats cannot pick up two definitions that have quietly
 * stopped being identical. */
#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

struct DecodeSequence {
    uint64_t literals_src;
    uint64_t literals_dst;
    uint64_t literals_len;
    uint64_t match_src;
    uint64_t match_dst;
    uint64_t match_len;
};

}  // namespace cudec_detail

#endif /* CUDEC_DECODE_SEQUENCE_H */
