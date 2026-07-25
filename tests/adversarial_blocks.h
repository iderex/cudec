/* The hand-crafted hostile LZ4 block corpus, single-sourced (issue #72).
 * The seeded fixture/mutant corpus in fixtures.h covers "what a compressor
 * produced, then damaged"; this covers the shapes an attacker writes by
 * hand: maximal LSIC length-extension runs, the spec-invalid zero offset, a
 * match source past everything written, and matches whose source overlaps
 * their own destination.
 *
 * Header-only and shared by the host parser test (tests/termination.cpp) and
 * the device one (tests/termination_gpu.cu) so both drive the identical
 * bytes - a stream that terminates on the CPU twin but not in the kernel
 * would otherwise be invisible. Compression goes through fixtures.h's
 * Lz4CompressBlock, so nothing here needs liblz4's headers (nvcc never sees
 * lz4.h - tests/CMakeLists.txt keeps that path private on purpose). */
#ifndef CUDEC_TESTS_ADVERSARIAL_BLOCKS_H
#define CUDEC_TESTS_ADVERSARIAL_BLOCKS_H

#include "fixtures.h"

#include <string>
#include <vector>

struct AdversarialBlock {
    std::string name;
    std::vector<unsigned char> stream;
    /* The destination capacity a caller would plausibly pass for this
     * stream; the host test sweeps the ladder's other decision points on
     * top of it. */
    size_t dst_capacity;
};

/* A token asking for a literal-length extension followed by `run` bytes of
 * 0xFF: the LSIC continuation loop is the one place in the parser whose
 * iteration count is read from the stream itself. */
inline std::vector<unsigned char> Lz4LiteralExtensionRun(size_t run) {
    std::vector<unsigned char> s;
    s.push_back(0xF0); /* 15 literals -> read an extension */
    s.insert(s.end(), run, 0xFF);
    s.push_back(0x00);
    s.insert(s.end(), 64, 0x41);
    return s;
}

/* The same on the match length: four literals, offset 1 (the shortest
 * self-referential distance there is), then the 0xFF extension run. */
inline std::vector<unsigned char> Lz4MatchExtensionRun(size_t run) {
    std::vector<unsigned char> s = {0x4F, 0x41, 0x42, 0x43, 0x44, 0x01, 0x00};
    s.insert(s.end(), run, 0xFF);
    s.push_back(0x00);
    s.insert(s.end(), 64, 0x41);
    return s;
}

inline std::vector<AdversarialBlock> MakeAdversarialBlocks() {
    std::vector<AdversarialBlock> out;
    const auto add = [&out](std::string name,
                            std::vector<unsigned char> stream,
                            size_t dst_capacity) {
        out.push_back({std::move(name), std::move(stream), dst_capacity});
    };
    add("empty", {}, 64);
    add("lone-zero-token", {0x00}, 64);
    add("lone-max-token", {0xFF}, 64);
    add("all-ff-4096", std::vector<unsigned char>(4096, 0xFF), 1u << 20);
    add("all-zero-4096", std::vector<unsigned char>(4096, 0x00), 1u << 20);
    add("literal-extension-run-1", Lz4LiteralExtensionRun(1), 1u << 20);
    add("literal-extension-run-4096", Lz4LiteralExtensionRun(4096), 1u << 20);
    add("match-extension-run-1", Lz4MatchExtensionRun(1), 1u << 20);
    add("match-extension-run-4096", Lz4MatchExtensionRun(4096), 1u << 20);
    /* Zero offset: rejected by cudec (the one deliberate divergence from
     * liblz4) and the input that would be a modulo-by-zero in the kernel's
     * closed-form gather. */
    {
        std::vector<unsigned char> s = {0x40, 0x41, 0x42, 0x43, 0x44,
                                        0x00, 0x00};
        s.insert(s.end(), 64, 0x41);
        add("zero-offset", std::move(s), 1u << 20);
    }
    /* A match source before the start of the output. */
    {
        std::vector<unsigned char> s = {0x40, 0x41, 0x42, 0x43, 0x44,
                                        0xFF, 0xFF};
        s.insert(s.end(), 64, 0x41);
        add("offset-past-output", std::move(s), 1u << 20);
    }
    /* Genuinely self-referential matches, built by the oracle so they are
     * valid: a run-length block (every match at offset 1, lengths far past
     * it) and a two-byte period. These are the overlapping matches the
     * kernel resolves by modular gather. */
    add("self-referential-rle",
        Lz4CompressBlock(std::vector<unsigned char>(65536, 0x5A)), 65536);
    {
        std::vector<unsigned char> period2(65536, 0x00);
        for (size_t i = 1; i < period2.size(); i += 2) {
            period2[i] = 0xFF;
        }
        add("self-referential-period-2", Lz4CompressBlock(period2), 65536);
    }
    return out;
}

#endif /* CUDEC_TESTS_ADVERSARIAL_BLOCKS_H */
