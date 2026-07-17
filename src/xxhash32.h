/* Minimal XXH32 (Yann Collet's xxHash, 32-bit, seed 0) - the checksum the
 * LZ4 frame format uses for its header, block, and content integrity
 * fields. Implemented here because the library depends on nothing beyond
 * the CUDA toolkit; verified byte-for-byte against liblz4's XXH32 in the
 * frame twin test. Host-only, internal. */
#ifndef CUDEC_XXHASH32_H
#define CUDEC_XXHASH32_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cudec_detail {

inline uint32_t xxh_read32(const unsigned char* p) {
    uint32_t v;
    std::memcpy(&v, p, 4); /* little-endian hosts; the frame format is LE */
    return v;
}

inline uint32_t xxh_rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

inline uint32_t xxh_round(uint32_t acc, uint32_t input) {
    constexpr uint32_t kPrime2 = 2246822519u;
    constexpr uint32_t kPrime1 = 2654435761u;
    return xxh_rotl(acc + input * kPrime2, 13) * kPrime1;
}

inline uint32_t xxhash32(const void* data, size_t len) {
    constexpr uint32_t kPrime1 = 2654435761u;
    constexpr uint32_t kPrime2 = 2246822519u;
    constexpr uint32_t kPrime3 = 3266489917u;
    constexpr uint32_t kPrime4 = 668265263u;
    constexpr uint32_t kPrime5 = 374761393u;

    const unsigned char* p = static_cast<const unsigned char*>(data);
    const unsigned char* const end = p + len;
    uint32_t h;

    if (len >= 16) {
        const unsigned char* const limit = end - 16;
        uint32_t v1 = kPrime1 + kPrime2;
        uint32_t v2 = kPrime2;
        uint32_t v3 = 0;
        uint32_t v4 = static_cast<uint32_t>(0u - kPrime1);
        do {
            v1 = xxh_round(v1, xxh_read32(p));
            p += 4;
            v2 = xxh_round(v2, xxh_read32(p));
            p += 4;
            v3 = xxh_round(v3, xxh_read32(p));
            p += 4;
            v4 = xxh_round(v4, xxh_read32(p));
            p += 4;
        } while (p <= limit);
        h = xxh_rotl(v1, 1) + xxh_rotl(v2, 7) + xxh_rotl(v3, 12) +
            xxh_rotl(v4, 18);
    } else {
        h = kPrime5;
    }

    h += static_cast<uint32_t>(len);
    while (p + 4 <= end) {
        h += xxh_read32(p) * kPrime3;
        h = xxh_rotl(h, 17) * kPrime4;
        p += 4;
    }
    while (p < end) {
        h += (*p) * kPrime5;
        h = xxh_rotl(h, 11) * kPrime1;
        p++;
    }

    h ^= h >> 15;
    h *= kPrime2;
    h ^= h >> 13;
    h *= kPrime3;
    h ^= h >> 16;
    return h;
}

}  // namespace cudec_detail

#endif /* CUDEC_XXHASH32_H */
