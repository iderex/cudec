/* XXH64 (Yann Collet's xxHash, 64-bit, seed zero) - the hash the Zstd frame
 * format uses for its content checksum. Single-sourced for host and device,
 * so the M5 kernel computes the same digest the host twin does. Internal
 * header, not part of the ABI.
 *
 * NO SEED PARAMETER, DELIBERATELY. RFC 8878 section 3.1.1 fixes the seed:
 * the checksum is "the result of the XXH64() hash function digesting the
 * original (decoded) data as input, and a seed of zero". A parameter that
 * every caller in this project would pass zero to is an argument nothing
 * tests, and the tree already made this choice once, in src/xxhash32.h for
 * the LZ4 frame's XXH32.
 *
 * The reads are byte-wise rather than a memcpy of eight bytes. The format
 * fixes the byte order and the host's is not fixed, the pointer is one the
 * caller chose and may be unaligned, and the same source has to compile for
 * a device where the memcpy shape buys nothing. Correctness first; this is
 * not on a measured path, and when it is, the measurement decides.
 *
 * Every loop is counted rather than a walk to a limit pointer, so the
 * termination argument is the count and not a comparison a reader has to
 * re-derive. The digits 32 and 64 arrive through named constants because a
 * width and a wave width look identical to a text scan (issue #211). */
#ifndef CUDEC_XXHASH64_H
#define CUDEC_XXHASH64_H

#include <stdint.h>

#ifndef CUDEC_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIP__)
#define CUDEC_HOST_DEVICE __host__ __device__
#else
#define CUDEC_HOST_DEVICE
#endif
#endif

namespace cudec_detail {

/* xxHash's five primes, in the reference's own order.
 *
 * These are the one part of this file that cannot be derived from anything
 * around them, and prime 1 in particular is reachable by every input longer
 * than three bytes and by none shorter. A transposed digit in it therefore
 * passes the published empty-input vector and fails on the first real byte,
 * which is how the wrong value got here and how the length sweep in the twin
 * caught it. */
constexpr uint64_t kXxh64Prime1 = 11400714785074694791ull;
constexpr uint64_t kXxh64Prime2 = 14029467366897019727ull;
constexpr uint64_t kXxh64Prime3 = 1609587929392839161ull;
constexpr uint64_t kXxh64Prime4 = 9650029242287828579ull;
constexpr uint64_t kXxh64Prime5 = 2870177450012600261ull;

/* The accumulator's width, and the four-accumulator stripe that width
 * implies. Bound to names on their own lines for the reason the header
 * comment gives. */
constexpr int kXxh64AccumulatorBits = 64;
constexpr uint64_t kXxh64StripeBytes = 32;
/* The avalanche's last shift, which is half the accumulator. Its own line,
 * for the same reason. */
constexpr int kXxh64FinalShift = 32;
constexpr uint64_t kXxh64LaneBytes = 8;
constexpr uint64_t kXxh64HalfLaneBytes = 4;

CUDEC_HOST_DEVICE inline uint64_t Xxh64Rotl(uint64_t x, int r) {
    return (x << r) | (x >> (kXxh64AccumulatorBits - r));
}

CUDEC_HOST_DEVICE inline uint64_t Xxh64Read64(const unsigned char* p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < kXxh64LaneBytes; i++) {
        value |= static_cast<uint64_t>(p[i]) << (8u * i);
    }
    return value;
}

CUDEC_HOST_DEVICE inline uint64_t Xxh64Read32(const unsigned char* p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < kXxh64HalfLaneBytes; i++) {
        value |= static_cast<uint64_t>(p[i]) << (8u * i);
    }
    return value;
}

CUDEC_HOST_DEVICE inline uint64_t Xxh64Round(uint64_t acc, uint64_t input) {
    acc += input * kXxh64Prime2;
    acc = Xxh64Rotl(acc, 31);
    acc *= kXxh64Prime1;
    return acc;
}

CUDEC_HOST_DEVICE inline uint64_t Xxh64MergeRound(uint64_t acc,
                                                  uint64_t value) {
    acc ^= Xxh64Round(0, value);
    acc = acc * kXxh64Prime1 + kXxh64Prime4;
    return acc;
}

/* The digest of `len` bytes at `data`, seed zero. */
CUDEC_HOST_DEVICE inline uint64_t Xxh64(const unsigned char* data,
                                        uint64_t len) {
    const uint64_t stripes = len / kXxh64StripeBytes;
    uint64_t offset = 0;
    uint64_t hash = 0;

    if (stripes != 0) {
        uint64_t v1 = kXxh64Prime1 + kXxh64Prime2;
        uint64_t v2 = kXxh64Prime2;
        uint64_t v3 = 0;
        /* Seed minus prime 1, in unsigned arithmetic, which is where the
         * reference's `seed - PRIME1` lands with a seed of zero. */
        uint64_t v4 = 0ull - kXxh64Prime1;
        for (uint64_t i = 0; i < stripes; i++) {
            v1 = Xxh64Round(v1, Xxh64Read64(data + offset));
            v2 = Xxh64Round(v2, Xxh64Read64(data + offset + kXxh64LaneBytes));
            v3 = Xxh64Round(
                v3, Xxh64Read64(data + offset + 2 * kXxh64LaneBytes));
            v4 = Xxh64Round(
                v4, Xxh64Read64(data + offset + 3 * kXxh64LaneBytes));
            offset += kXxh64StripeBytes;
        }
        hash = Xxh64Rotl(v1, 1) + Xxh64Rotl(v2, 7) + Xxh64Rotl(v3, 12) +
               Xxh64Rotl(v4, 18);
        hash = Xxh64MergeRound(hash, v1);
        hash = Xxh64MergeRound(hash, v2);
        hash = Xxh64MergeRound(hash, v3);
        hash = Xxh64MergeRound(hash, v4);
    } else {
        hash = kXxh64Prime5;
    }

    hash += len;

    /* The tail, in the reference's three widths: whole lanes, then one half
     * lane, then single bytes. Each is counted from what is left rather than
     * walked to a pointer. */
    const uint64_t tail_lanes = (len - offset) / kXxh64LaneBytes;
    for (uint64_t i = 0; i < tail_lanes; i++) {
        hash ^= Xxh64Round(0, Xxh64Read64(data + offset));
        hash = Xxh64Rotl(hash, 27) * kXxh64Prime1 + kXxh64Prime4;
        offset += kXxh64LaneBytes;
    }
    if (len - offset >= kXxh64HalfLaneBytes) {
        hash ^= Xxh64Read32(data + offset) * kXxh64Prime1;
        hash = Xxh64Rotl(hash, 23) * kXxh64Prime2 + kXxh64Prime3;
        offset += kXxh64HalfLaneBytes;
    }
    const uint64_t tail_bytes = len - offset;
    for (uint64_t i = 0; i < tail_bytes; i++) {
        hash ^= static_cast<uint64_t>(data[offset + i]) * kXxh64Prime5;
        hash = Xxh64Rotl(hash, 11) * kXxh64Prime1;
    }

    /* The avalanche. */
    hash ^= hash >> 33;
    hash *= kXxh64Prime2;
    hash ^= hash >> 29;
    hash *= kXxh64Prime3;
    hash ^= hash >> kXxh64FinalShift;
    return hash;
}

}  // namespace cudec_detail

#endif /* CUDEC_XXHASH64_H */
