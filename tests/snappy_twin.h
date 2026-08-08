/* The host reference driver for the Snappy element parser: one copy, shared
 * by the CPU twin test (tests/snappy_parser_twin.cpp) and the differential
 * fuzz target (fuzz/fuzz_snappy_block.cpp), issue #90. The LZ4 pair took the
 * same shape first, in tests/lz4_twin.h (issue #140), and for the same
 * reason: two similar drivers can drift until the fuzz target is green about
 * a parser execution the test net never performs.
 *
 * What the driver carries with it, so neither caller has to remember it:
 *
 * The tight stream copy. A parse runs from an exactly-sized allocation, so an
 * over-read past src_size lands in an ASan redzone instead of in a vector's
 * rounded-up slack or in a fuzzer's input buffer.
 *
 * The fuel bound. A parser that stops making progress must red the run rather
 * than hang it. Every non-terminal call consumes at least one source byte, so
 * one call per byte plus the terminal one is more than a live parser needs.
 *
 * The bounds count. A verdict alone cannot prove a bounds guard: removing the
 * check that a long-form literal's length bytes are present still rejects
 * every crafted stream, because the cursor runs past the source end, the next
 * subtraction wraps, and a later branch catches it after the read has already
 * happened. What the guard prevents is that read, so that is what is counted,
 * separately from the status. Measured - three such removals left a
 * verdict-only suite green.
 *
 * The match copy is the chasing byte copy: reads may land on just-written
 * bytes, which is the pattern-repeating semantics an overlapping Snappy copy
 * has. */
#ifndef CUDEC_TESTS_SNAPPY_TWIN_H
#define CUDEC_TESTS_SNAPPY_TWIN_H

#include "cudec.h"
#include "snappy_block.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace cudec_test {

/* What a driver run reports beside its status. It is an out-parameter rather
 * than a return value because both callers accumulate across many runs: the
 * twin test requires the total to be zero at the end, and the fuzz target
 * traps on the first one. */
struct SnappyTwinObserver {
    /* Elements the parser handed back that the caller could not have
     * executed without leaving its buffers, cursors left outside their own
     * bounds, and parses that did not terminate. */
    size_t bounds_violations = 0;
    /* Which ladder branch refused last. Diagnostic: the returned status only
     * says that some rung refused, and several crafted negatives share one
     * status. */
    cudec_detail::SnappyReject last_reject = cudec_detail::kSnappyRejectNone;
};

/* Sequential reference execution of the parsed elements at a caller-chosen
 * capacity. `src` must already be an exactly-sized allocation. */
inline cudec_status SnappyTwinDrive(const unsigned char* src,
                                    uint64_t src_size, uint64_t capacity,
                                    std::vector<unsigned char>* out,
                                    SnappyTwinObserver* observer) {
    out->assign(static_cast<size_t>(capacity), 0);
    cudec_detail::SnappyParser parser{src, src_size, capacity};
    cudec_detail::SnappyElement element;
    bool done = false;
    uint64_t fuel = src_size + 2;
    while (true) {
        if (fuel-- == 0) {
            std::fprintf(stderr,
                         "the parser did not terminate on a %llu-byte "
                         "stream\n",
                         static_cast<unsigned long long>(src_size));
            observer->bounds_violations++;
            out->clear();
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        const cudec_status status = parser.Next(&element, &done);
        if (status != CUDEC_OK) {
            /* A rejected stream produces nothing, the same contract the
             * snappy oracle wrapper holds its caller to. */
            observer->last_reject = parser.reject;
            out->clear();
            return status;
        }
        /* Both cursors stay inside their buffers, the produced length stays
         * inside the declaration, and the element lies inside what the
         * parser says it produced. A literal's source range is inside the
         * stream; a copy reaches strictly backwards, which is what makes the
         * chasing read below defined. */
        const bool bounded =
            parser.src_pos <= parser.src_size &&
            parser.dst_pos <= parser.declared &&
            parser.declared <= capacity &&
            element.to + element.length <= parser.dst_pos &&
            (element.is_copy ? element.from < element.to
                             : element.from + element.length <=
                                   parser.src_pos);
        if (!bounded) {
            std::fprintf(stderr,
                         "the parser left its bounds: src_pos=%llu/%llu "
                         "dst_pos=%llu/%llu element from=%llu to=%llu len=%llu "
                         "copy=%d\n",
                         static_cast<unsigned long long>(parser.src_pos),
                         static_cast<unsigned long long>(parser.src_size),
                         static_cast<unsigned long long>(parser.dst_pos),
                         static_cast<unsigned long long>(parser.declared),
                         static_cast<unsigned long long>(element.from),
                         static_cast<unsigned long long>(element.to),
                         static_cast<unsigned long long>(element.length),
                         element.is_copy ? 1 : 0);
            observer->bounds_violations++;
            out->clear();
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        if (element.is_copy) {
            for (uint64_t i = 0; i < element.length; i++) {
                (*out)[static_cast<size_t>(element.to + i)] =
                    (*out)[static_cast<size_t>(element.from + i)];
            }
        } else {
            for (uint64_t i = 0; i < element.length; i++) {
                (*out)[static_cast<size_t>(element.to + i)] =
                    src[element.from + i];
            }
        }
        if (done) {
            break;
        }
    }
    out->resize(static_cast<size_t>(parser.dst_pos));
    return CUDEC_OK;
}

/* The whole-stream entry: size the destination from the stream's own
 * declaration, the way the snappy oracle wrapper does, so a mutant declaring
 * 4 GiB is refused on both sides for the same reason instead of reading as a
 * parity failure. `capacity_bound` is that shared refusal and belongs to the
 * caller, because the two callers allocate under different limits. */
inline cudec_status SnappyTwinDecode(const unsigned char* stream,
                                     size_t stream_size,
                                     uint64_t capacity_bound,
                                     std::vector<unsigned char>* out,
                                     SnappyTwinObserver* observer) {
    out->clear();
    auto tight = std::make_unique<unsigned char[]>(stream_size);
    if (stream_size != 0) {
        std::memcpy(tight.get(), stream, stream_size);
    }
    /* The declaration is read through the parser's own Begin, so the varint
     * is parsed once and the destination is sized from a value that has
     * already been compared against a capacity. */
    cudec_detail::SnappyParser probe{tight.get(), stream_size, capacity_bound};
    const cudec_status header = probe.Begin();
    if (header != CUDEC_OK) {
        observer->last_reject = probe.reject;
        return header;
    }
    return SnappyTwinDrive(tight.get(), stream_size, probe.declared, out,
                           observer);
}

}  // namespace cudec_test

#endif /* CUDEC_TESTS_SNAPPY_TWIN_H */
