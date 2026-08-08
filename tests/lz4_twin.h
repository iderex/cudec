/* The host reference driver for the LZ4 block parser: one copy, shared by
 * the CPU twin test (tests/parser_twin.cpp) and the differential fuzz target
 * (fuzz/fuzz_lz4_block.cpp), issue #140.
 *
 * It has to be one copy rather than two similar ones. The fuzz target and the
 * twin test assert the same property against the same parser, so a second
 * implementation could drift into asserting the property against a driver the
 * test net never runs - a fuzzer that goes green about code nobody ships.
 *
 * Both buffers are allocated tight, which is the load-bearing part on a
 * sanitized build: the stream copy puts an ASan redzone immediately after the
 * last stream byte, so a parser over-read past src_size reds instead of
 * landing in a vector's rounded-up slack, and the output buffer is exactly
 * `capacity` bytes so a write past the declared capacity reds the same way.
 *
 * The match copy is the chasing byte copy - reads may land on just-written
 * bytes, which is exactly LZ4's replication semantics for overlapping matches
 * (and equivalent to the kernel's closed-form modular gather). */
#ifndef CUDEC_TESTS_LZ4_TWIN_H
#define CUDEC_TESTS_LZ4_TWIN_H

#include "cudec.h"
#include "lz4_block.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace cudec_test {

struct Lz4TwinResult {
    cudec_status status;
    /* Decoded byte count, meaningful only when status == CUDEC_OK. */
    size_t size;
    /* Exactly `capacity` bytes, always allocated: on a reject the caller may
     * still want the partial writes the parser had already authorised. */
    std::unique_ptr<unsigned char[]> bytes;
};

inline Lz4TwinResult Lz4TwinDecode(const unsigned char* stream,
                                   size_t stream_size, size_t capacity) {
    Lz4TwinResult result{CUDEC_OK, 0,
                         std::make_unique<unsigned char[]>(capacity)};
    if (capacity != 0) {
        std::memset(result.bytes.get(), 0, capacity);
    }

    auto tight = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(tight.get(), stream, stream_size);
    }

    cudec_detail::Lz4Parser parser{tight.get(), stream_size, capacity};
    cudec_detail::Lz4Sequence sequence;
    bool done = false;
    while (true) {
        const cudec_status status = parser.Next(&sequence, &done);
        if (status != CUDEC_OK) {
            result.status = status;
            return result;
        }
        for (uint64_t i = 0; i < sequence.literals_len; i++) {
            result.bytes[sequence.literals_dst + i] =
                tight[sequence.literals_src + i];
        }
        for (uint64_t i = 0; i < sequence.match_len; i++) {
            result.bytes[sequence.match_dst + i] =
                result.bytes[sequence.match_src + i];
        }
        if (done) {
            break;
        }
    }
    result.size = static_cast<size_t>(parser.dst_pos);
    return result;
}

}  // namespace cudec_test

#endif /* CUDEC_TESTS_LZ4_TWIN_H */
