/* The hand-crafted hostile Snappy corpus, single-sourced (issue #154) - the
 * sibling of adversarial_blocks.h, which does the same job for LZ4.
 *
 * The seeded fixture/mutant corpus in fixtures.h covers "what the compressor
 * produced, then damaged". This covers what a compressor never emits and an
 * attacker writes by hand: the wide length and offset classes, the copy forms
 * a compressor has no reason to choose, offsets reaching backwards past
 * everything written, declared lengths at the varint's 2^32 ceiling, and the
 * one arithmetic shape where cudec is deliberately stricter than the
 * reference.
 *
 * Header-only and driven by both sides so a stream that behaves on the host
 * and misbehaves in the kernel cannot hide: tests/snappy_block_device.cu
 * parses every one of them with the host compiler and with nvcc and compares
 * the element traces, and tests/gpu_fixture.cu drives the identical bytes
 * through the public batch entry against the oracle's verdict.
 *
 * Nothing here goes through a compressor, on purpose: every byte is chosen,
 * so a stream that stops exercising what its name says would have to be
 * edited rather than drift. The verdicts are the oracle's wherever a caller
 * asks for one; this header states no expected status of its own. */
#ifndef CUDEC_TESTS_ADVERSARIAL_SNAPPY_BLOCKS_H
#define CUDEC_TESTS_ADVERSARIAL_SNAPPY_BLOCKS_H

#include <cstddef>
#include <string>
#include <vector>

struct AdversarialSnappyBlock {
    std::string name;
    std::vector<unsigned char> stream;
    /* The destination capacity a caller would plausibly pass. It matters
     * here in a way it does not for LZ4: the declared length is compared
     * against exactly this value, and that comparison is the only place a
     * Snappy stream's own claim about its size is ever bounded. */
    size_t dst_capacity;
};

/* The little-endian base-128 preamble, written without a minimality rule so
 * the callers below can spell one length several ways. */
inline std::vector<unsigned char> SnappyPreamble(unsigned long long length) {
    std::vector<unsigned char> out;
    unsigned long long v = length;
    while (v >= 0x80) {
        out.push_back(static_cast<unsigned char>((v & 0x7f) | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<unsigned char>(v));
    return out;
}

inline void SnappyAppend(std::vector<unsigned char>* out,
                         const std::vector<unsigned char>& more) {
    out->insert(out->end(), more.begin(), more.end());
}

/* A literal in the inline-length class: the six tag bits hold length-1. */
inline std::vector<unsigned char> SnappyLiteral(const char* text,
                                                size_t size) {
    std::vector<unsigned char> out;
    out.push_back(static_cast<unsigned char>((size - 1) << 2));
    for (size_t i = 0; i < size; i++) {
        out.push_back(static_cast<unsigned char>(text[i]));
    }
    return out;
}

/* The one-byte-offset copy form. It carries the offset's high three bits in
 * the tag and cannot express a length below four. */
inline std::vector<unsigned char> SnappyCopy1(unsigned length,
                                              unsigned offset) {
    std::vector<unsigned char> out;
    out.push_back(static_cast<unsigned char>(((offset >> 8) << 5) |
                                             ((length - 4) << 2) | 1));
    out.push_back(static_cast<unsigned char>(offset & 0xff));
    return out;
}

inline std::vector<unsigned char> SnappyCopy2(unsigned length,
                                              unsigned offset) {
    std::vector<unsigned char> out;
    out.push_back(static_cast<unsigned char>(((length - 1) << 2) | 2));
    out.push_back(static_cast<unsigned char>(offset & 0xff));
    out.push_back(static_cast<unsigned char>((offset >> 8) & 0xff));
    return out;
}

/* The four-byte-offset form. No compressor emits it - snappy's own writer
 * never selects tag 3 - and the reference decoder takes it, so refusing it
 * would be over-strictness the oracle diff would catch. */
inline std::vector<unsigned char> SnappyCopy4(unsigned length,
                                              unsigned long long offset) {
    std::vector<unsigned char> out;
    out.push_back(static_cast<unsigned char>(((length - 1) << 2) | 3));
    for (int i = 0; i < 4; i++) {
        out.push_back(static_cast<unsigned char>((offset >> (8 * i)) & 0xff));
    }
    return out;
}

inline std::vector<AdversarialSnappyBlock> MakeAdversarialSnappyBlocks() {
    using Bytes = std::vector<unsigned char>;
    std::vector<AdversarialSnappyBlock> out;
    const auto add = [&out](std::string name, Bytes stream, size_t capacity) {
        out.push_back({std::move(name), std::move(stream), capacity});
    };
    const size_t kCap = 1u << 20;

    add("empty", {}, kCap);
    add("preamble-only-zero", SnappyPreamble(0), kCap);

    {
        Bytes s = SnappyPreamble(4);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("one-literal", s, kCap);
    }
    {
        Bytes s = SnappyPreamble(8);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy2(4, 4));
        add("literal-then-copy", s, kCap);
    }
    {
        /* Offset 1 is the pattern-repeating extreme: every byte of the copy
         * reads the byte the copy itself wrote one position earlier, which
         * is where a warp gather that is not a closed-form modulo diverges
         * from a sequential executor. */
        Bytes s = SnappyPreamble(64);
        SnappyAppend(&s, SnappyLiteral("a", 1));
        SnappyAppend(&s, SnappyCopy2(63, 1));
        add("rle-offset-one-63", s, kCap);
    }
    {
        /* The 64-byte cap of the two-byte form, at offset 2: an overlap that
         * is neither a pure RLE nor disjoint. */
        Bytes s = SnappyPreamble(66);
        SnappyAppend(&s, SnappyLiteral("ab", 2));
        SnappyAppend(&s, SnappyCopy2(64, 2));
        add("copy2-max-length-overlap", s, kCap);
    }
    {
        Bytes s = SnappyPreamble(8);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy2(4, 0));
        add("copy2-offset-zero", s, kCap);
    }
    {
        Bytes s = SnappyPreamble(8);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy1(4, 0));
        add("copy1-offset-zero", s, kCap);
    }
    {
        Bytes s = SnappyPreamble(8);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy4(4, 0));
        add("copy4-offset-zero", s, kCap);
    }
    {
        Bytes s = SnappyPreamble(8);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy2(4, 9));
        add("copy-offset-past-output-start", s, kCap);
    }
    {
        /* An offset far past anything written, in the only form wide enough
         * to express it. The bound is the bytes produced, never the window
         * convention. */
        Bytes s = SnappyPreamble(8);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy4(4, 0xFFFFFFFFull));
        add("copy4-offset-4g", s, kCap);
    }
    {
        /* Copy lengths 1 to 3 cannot be spelled in the narrow form and are
         * legal in the wide ones. A decoder that imposed a length floor
         * would be stricter than the reference here. */
        Bytes s = SnappyPreamble(5);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy2(1, 4));
        add("copy2-length-one", s, kCap);
    }
    {
        Bytes s = SnappyPreamble(8);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyCopy4(4, 4));
        add("copy4-ordinary", s, kCap);
    }
    {
        /* An offset above 65536, which needs the four-byte form and a
         * destination that has produced that much. Built as one literal run
         * long enough to reach back into. */
        Bytes s = SnappyPreamble(70000);
        s.push_back(0xF4); /* long-form literal, two length bytes follow */
        s.push_back(static_cast<unsigned char>((69995 - 1) & 0xff));
        s.push_back(static_cast<unsigned char>(((69995 - 1) >> 8) & 0xff));
        for (size_t i = 0; i < 69995; i++) {
            s.push_back(static_cast<unsigned char>('a' + i % 26));
        }
        SnappyAppend(&s, SnappyCopy4(5, 69000));
        add("copy4-offset-above-64k", s, kCap);
    }
    {
        /* A non-minimal preamble: five groups spelling a small number. The
         * reference accepts it, so cudec must too. */
        Bytes s = {0x84, 0x80, 0x80, 0x80, 0x00};
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("preamble-non-minimal-five-groups", s, kCap);
    }
    add("preamble-final-group-sixteen",
        std::vector<unsigned char>{0x80, 0x80, 0x80, 0x80, 0x10}, kCap);
    {
        /* The documented divergence: the four-byte literal length class
         * spelling 0xFFFFFFFF. The reference adds one in 32 bits, wraps to a
         * zero-length literal and accepts the stream; cudec refuses it,
         * because a length that exists only because an accumulator wrapped
         * is not a length. This is the ONE stream where the two verdicts
         * differ by design. */
        add("literal-length-wraps-to-zero",
            std::vector<unsigned char>{0x00, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF},
            kCap);
    }
    {
        /* Header past the end: the long-form literal's length bytes are
         * missing entirely. */
        Bytes s = SnappyPreamble(300);
        s.push_back(0xF0);
        add("literal-header-truncated", s, kCap);
    }
    {
        /* The payload is short of what the header declares. */
        Bytes s = SnappyPreamble(64);
        s.push_back(0xFC); /* 60 -> one length byte follows */
        s.push_back(63);
        for (int i = 0; i < 10; i++) {
            s.push_back('a');
        }
        add("literal-payload-truncated", s, kCap);
    }
    {
        /* Produces less than it declared: the source runs out first. */
        Bytes s = SnappyPreamble(16);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("under-production", s, kCap);
    }
    {
        /* Bytes after the declared length has been produced exactly. */
        Bytes s = SnappyPreamble(4);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        SnappyAppend(&s, SnappyLiteral("e", 1));
        add("trailing-element", s, kCap);
    }

    /* The capacity adversarials. The declared length is attacker-controlled
     * and reaches 2^32 - 1; the caller's capacity is the only truth it is
     * ever measured against, and nothing anywhere is sized from it. These
     * carry SMALL capacities on purpose: the point is that a 4 GiB claim is
     * refused at the preamble, before an element is parsed and before a byte
     * is written, rather than that a 4 GiB buffer happens to exist. */
    {
        Bytes s = SnappyPreamble(0xFFFFFFFFull);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("declared-4g-minus-one-capacity-4k", s, 4096);
    }
    {
        Bytes s = SnappyPreamble(0xFFFFFFFEull);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("declared-4g-minus-two-capacity-4k", s, 4096);
    }
    {
        Bytes s = SnappyPreamble(0x80000000ull);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("declared-2g-capacity-4k", s, 4096);
    }
    {
        /* The boundary from the other side: a declaration one byte above the
         * capacity, where refusing and accepting are one comparison apart. */
        Bytes s = SnappyPreamble(4097);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("declared-one-over-capacity", s, 4096);
    }
    {
        /* And exactly at it, which must NOT be refused for capacity: this
         * stream rejects for under-production instead, and a decoder with an
         * off-by-one at the capacity comparison reports the wrong one. */
        Bytes s = SnappyPreamble(4096);
        SnappyAppend(&s, SnappyLiteral("abcd", 4));
        add("declared-exactly-capacity", s, 4096);
    }

    return out;
}

#endif /* CUDEC_TESTS_ADVERSARIAL_SNAPPY_BLOCKS_H */
