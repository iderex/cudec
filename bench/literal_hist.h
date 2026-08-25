/* The literal-length distribution of a corpus, shared by both harnesses
 * (issue #165).
 *
 * Why it is one header and not two functions. The question it answers is a
 * COMPARISON - whether Snappy's literal runs are longer than LZ4's on the
 * same bytes - and a comparison between two histograms is worth nothing
 * unless both were bucketed the same way and printed the same way. Two
 * copies of the bucket edges is how that stops being true, silently, the
 * first time one of them is tuned.
 *
 * It walks the corpus with the shipped parser rather than with a reader of
 * its own, so the elements it counts are the elements the decoder executes.
 * Nothing here writes output: a parser hands back an element and the driver
 * is what would copy it, so the walk costs the parse alone.
 *
 * The bucket edges are powers of two around the one that matters. A wide
 * literal path copies 16 bytes at a time, so its trigger rate is the share
 * of literal BYTES sitting in literals of at least 16 - the share of
 * ELEMENTS is the wrong statistic and is reported beside it precisely so the
 * two cannot be confused. */
#ifndef CUDEC_BENCH_LITERAL_HIST_H
#define CUDEC_BENCH_LITERAL_HIST_H

#include "cudec.h"
#include "decode_sequence.h"

#include <cstdint>
#include <cstdio>

namespace cudec_bench {

constexpr unsigned kLiteralBuckets = 8;

/* Printed beside every row, so a reader never has to reconstruct the edges
 * from the code. */
inline const char* LiteralBucketName(unsigned bucket) {
    static const char* const kNames[kLiteralBuckets] = {
        "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-255", "256+"};
    return kNames[bucket];
}

/* The wide path's threshold, in bucket terms: buckets at or above this index
 * hold literals of at least 16 bytes. */
constexpr unsigned kLiteralWideBucket = 4;

struct LiteralHistogram {
    uint64_t streams = 0;
    uint64_t elements = 0;
    uint64_t literal_elements = 0;
    uint64_t literal_bytes = 0;
    uint64_t count[kLiteralBuckets] = {0};
    uint64_t bytes[kLiteralBuckets] = {0};
};

inline unsigned LiteralBucket(uint64_t length) {
    if (length < 2) {
        return 0;
    }
    if (length < 4) {
        return 1;
    }
    if (length < 8) {
        return 2;
    }
    if (length < 16) {
        return 3;
    }
    if (length < 32) {
        return 4;
    }
    if (length < 64) {
        return 5;
    }
    if (length < 256) {
        return 6;
    }
    return 7;
}

/* Walks one stream with the shipped parser and folds its literal lengths in.
 * Returns false if the parser rejects the stream, which for a corpus this
 * harness just built and verified is a defect rather than a datum, so every
 * caller must treat it as one instead of skipping the stream. */
template <class Parser>
bool AccumulateLiteralLengths(const unsigned char* src, uint64_t src_size,
                              uint64_t dst_capacity, LiteralHistogram* out) {
    Parser parser{src, src_size, dst_capacity};
    cudec_detail::DecodeSequence element;
    bool done = false;
    while (true) {
        if (parser.Next(&element, &done) != CUDEC_OK) {
            return false;
        }
        if (done) {
            break;
        }
        out->elements++;
        if (element.literals_len != 0) {
            const unsigned bucket = LiteralBucket(element.literals_len);
            out->literal_elements++;
            out->literal_bytes += element.literals_len;
            out->count[bucket]++;
            out->bytes[bucket] += element.literals_len;
        }
    }
    out->streams++;
    return true;
}

inline double LiteralShare(uint64_t part, uint64_t whole) {
    return whole == 0 ? 0.0
                      : 100.0 * static_cast<double>(part) /
                            static_cast<double>(whole);
}

inline void PrintLiteralHistogram(const LiteralHistogram& hist) {
    uint64_t wide_bytes = 0;
    uint64_t wide_count = 0;
    for (unsigned b = kLiteralWideBucket; b < kLiteralBuckets; b++) {
        wide_bytes += hist.bytes[b];
        wide_count += hist.count[b];
    }
    std::printf("- literal distribution: %llu streams, %llu elements, %llu "
                "of them literal (%.1f%%), %llu literal bytes\n",
                static_cast<unsigned long long>(hist.streams),
                static_cast<unsigned long long>(hist.elements),
                static_cast<unsigned long long>(hist.literal_elements),
                LiteralShare(hist.literal_elements, hist.elements),
                static_cast<unsigned long long>(hist.literal_bytes));
    for (unsigned b = 0; b < kLiteralBuckets; b++) {
        std::printf("- literals %-7s: %12llu elements (%5.1f%%), %14llu "
                    "bytes (%5.1f%%)\n",
                    LiteralBucketName(b),
                    static_cast<unsigned long long>(hist.count[b]),
                    LiteralShare(hist.count[b], hist.literal_elements),
                    static_cast<unsigned long long>(hist.bytes[b]),
                    LiteralShare(hist.bytes[b], hist.literal_bytes));
    }
    /* The trigger rate a 16-byte wide copy would actually see, in both
     * statistics, because the element share and the byte share answer
     * different questions and only the second one is the work. */
    std::printf("- wide-path trigger (literals >= 16 bytes): %.1f%% of "
                "literal elements, %.1f%% of literal bytes\n",
                LiteralShare(wide_count, hist.literal_elements),
                LiteralShare(wide_bytes, hist.literal_bytes));
}

}  // namespace cudec_bench

#endif /* CUDEC_BENCH_LITERAL_HIST_H */
