/* Differential fuzz target over the whole host Zstd frame decode (issue #194),
 * the seventh target in fuzz/ and the fourth over the M5 surface. Where the
 * siblings each enter one unit, this one enters the path that runs all of them
 * in order - frame header, block header, literals section, sequence tables,
 * the interleaved sequence decode, the repeat-offset history and the sequence
 * execution - and asks libzstd about the same bytes.
 *
 * WHAT THAT ADDS, which is not more of the same. A unit target proves the unit
 * against the reference entry point that matches it. None of them can say
 * whether the STAGE ORDER is right, whether state that must survive a block
 * boundary does, or whether a section one unit accepted is one the next unit
 * reads at the offset the first one left. Those are properties of the
 * composition, and a composition is only observable at the frame.
 *
 * TWO PASSES, AND THEY REACH DIFFERENT HALVES OF THE PATH.
 *
 * The RAW pass hands the fuzzer's bytes over as a whole frame. That is the
 * only shape that reaches the frame header, the subset decisions, the content
 * checksum and the bytes-after-the-frame rule, and it is where the
 * fail-closed direction over a whole frame is asserted. It is also where
 * almost every input dies in the first six bytes, which is why it is not the
 * only pass.
 *
 * The STRUCTURE-AWARE pass holds a frame header, a block header and a Raw
 * literals section fixed and lets the fuzzer write the SEQUENCES SECTION. That
 * is the only way the repeat-offset rotation, the zero-literals index shift,
 * the minus-one rule, the offset-against-window bound and the sequence
 * execution are reached at all: they sit behind an entropy decode that random
 * bytes do not survive.
 *
 * THE DECLARED CONTENT SIZE IS WHAT MAKES THE SECOND PASS HARD, AND THE
 * REFERENCE IS WHAT SOLVES IT. The v1 subset (#102) decodes only frames that
 * declare their content size, and what a fuzzed sequence section regenerates
 * is not known until it has been decoded - so the envelope cannot declare it
 * up front without deciding its own answer. The way out is to ask the
 * reference twice. The body is first wrapped in a frame declaring NO content
 * size, which libzstd decodes happily and cudec declines as outside the
 * subset; the size that comes back is then written into a second frame with
 * the same body, and THAT frame is what both sides answer for. The rebuild is
 * proven rather than assumed: the reference is asked about the second frame
 * too, and its two answers must be identical.
 *
 * WHERE THE REFERENCE REFUSES THE FIRST FRAME there is no size to write down,
 * and no frame both sides can be asked the same question about. The pass still
 * runs the twin, over a frame declaring a fixed size, and asserts only that it
 * does not ACCEPT - an acceptance there would mean cudec regenerated exactly
 * that many bytes from a body libzstd refuses, which is a fail-open whatever
 * the size was. The byte comparison is absent in that arm and the sanitizers
 * are what is watching instead: this target is built with ASan and UBSan, so a
 * read outside the literals, a match reaching before the output, or a shift
 * past a type's width reds the run on its own without the reference having an
 * opinion. Said plainly rather than left to be inferred from the counters.
 *
 * WHICH DIRECTION IS ASSERTED. The fail-open direction always traps: cudec
 * accepting what libzstd refuses, or producing different bytes or a different
 * size. The reverse - cudec refusing what libzstd accepts - traps too unless
 * the status is UNSUPPORTED, which is the subset declining a legal frame
 * rather than calling it corrupt, with ONE declared exception. A buffer
 * holding a frame followed by anything else is one libzstd decodes and cudec
 * refuses as CORRUPT_INPUT at the trailing-bytes rule (section 12.4). That is
 * identified rather than guessed at - ZSTD_findFrameCompressedSize says where
 * the first frame ends, and the exception applies only where that is short of
 * the input - and it is counted rather than passed over.
 */
#include "cudec.h"
#include "zstd_twin_driver.h"

#include <zstd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using cudec_twin::Bytes;
using cudec_twin::Run;

/* The fuzzer's slice is capped below the block maximum so the block header's
 * own size field and the 128 KB bound are never the thing under test. */
constexpr size_t kMaxSection = 32 * 1024;
/* Larger than anything the envelope can regenerate, so a refusal is never
 * accidentally about room. */
constexpr uint64_t kDecodeCapacity = 1024ull * 1024ull;
/* The size the structure-aware pass declares when the reference gave it none.
 * Any legal value does: the arm asserts a refusal, and an acceptance at
 * exactly this size would be the fail-open it exists to catch. */
constexpr uint64_t kProbeContentSize = 4096;

void Trap(const char* what, const char* where, size_t size) {
    std::fprintf(stderr, "DIVERGENCE: %s; pass=%s input=%zu\n", what, where,
                 size);
    __builtin_trap();
}

void Append(Bytes* out, const unsigned char* data, size_t size) {
    out->insert(out->end(), data, data + size);
}

/* The literals the structure-aware envelope holds fixed, and the sequences
 * section written against them.
 *
 * Both were taken from ONE compressed block that libzstd wrote, so the seed
 * corpus can carry that section back in unchanged and decode: the literals are
 * what that block's Raw literals section regenerated, and the sequences are
 * the bytes that followed it. A section the fuzzer mutates then meets exactly
 * the literals its lengths were written against, which is what makes the
 * execution reachable rather than a lucky accident.
 *
 * The envelope's window is the 128 KB ceiling rather than the original frame's,
 * which only widens the bound a match is held to; the offsets are unchanged. */
const unsigned char kLiterals[] = {
    0x74, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b, 0x20, 0x62, 0x72,
    0x6f, 0x77, 0x6e, 0x20, 0x66, 0x6f, 0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70,
    0x73, 0x20, 0x6f, 0x76, 0x65, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x6c,
    0x61, 0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67, 0x61, 0x67, 0x61, 0x69, 0x6e,
    0x20, 0x61, 0x6e, 0x64, 0x20, 0x61, 0x67, 0x61, 0x69, 0x6e};

/* Section 3.1.1.3.1.1, Raw literals with Size_Format 01: a two-byte header
 * whose four high bits of the first byte carry the low four of the regenerated
 * size. Written out rather than a constant, so the header and the array above
 * cannot drift apart. */
Bytes RawLiteralsSection() {
    const uint32_t size = static_cast<uint32_t>(sizeof(kLiterals));
    Bytes out;
    out.push_back(static_cast<unsigned char>((1u << 2) | ((size & 0x0fu) << 4)));
    out.push_back(static_cast<unsigned char>(size >> 4));
    Append(&out, kLiterals, sizeof(kLiterals));
    return out;
}

/* Section 3.1.1.2: a last, compressed block around an already-assembled
 * body. */
void AppendCompressedBlock(Bytes* out, const Bytes& body) {
    const uint32_t header =
        static_cast<uint32_t>(body.size() << 3) | (2u << 1) | 1u;
    for (unsigned i = 0; i < 3; i++) {
        out->push_back(static_cast<unsigned char>(header >> (8u * i)));
    }
    Append(out, body.data(), body.size());
}

/* Section 3.1.1.1. Window_Descriptor 0x38 is exponent seven, mantissa zero, so
 * the window is 2^17 and the block maximum is the 128 KB ceiling rather than
 * the window. With a content size the descriptor's high bits select the
 * four-byte field, which spans every size this envelope can produce; without
 * one the descriptor is zero and cudec declines the frame as outside the
 * subset, which is exactly what that frame is for. */
Bytes StructuredFrame(const unsigned char* section, size_t size,
                      bool with_content_size, uint64_t content_size) {
    Bytes out{0x28, 0xb5, 0x2f, 0xfd};
    out.push_back(with_content_size ? 0x80 : 0x00);
    out.push_back(0x38);
    if (with_content_size) {
        for (unsigned i = 0; i < 4; i++) {
            out.push_back(static_cast<unsigned char>(content_size >> (8u * i)));
        }
    }
    Bytes body = RawLiteralsSection();
    Append(&body, section, size);
    AppendCompressedBlock(&out, body);
    return out;
}

/* What the reference says about one frame. */
struct Reference {
    bool ok = false;
    Bytes output;
};

Reference Decode(const Bytes& frame) {
    Reference out;
    Bytes buffer(static_cast<size_t>(kDecodeCapacity), 0);
    const size_t produced =
        ZSTD_decompress(buffer.data(), buffer.size(), frame.data(),
                        frame.size());
    if (ZSTD_isError(produced)) {
        return out;
    }
    buffer.resize(produced);
    out.ok = true;
    out.output = buffer;
    return out;
}

/* The declared exception, identified rather than guessed at: the input holds a
 * frame and then something else, which libzstd walks and the subset refuses.
 * A findFrameCompressedSize that errors is not this case - the first frame
 * itself is what libzstd could not measure - so it answers no. */
bool HoldsMoreThanOneFrame(const Bytes& input) {
    const size_t first = ZSTD_findFrameCompressedSize(input.data(),
                                                      input.size());
    if (ZSTD_isError(first)) {
        return false;
    }
    return first < input.size();
}

size_t g_trailing_bytes_exception = 0;
size_t g_execution_reached = 0;

/* The three assertions of issue #194 over one frame both sides were given.
 *
 * `strict_reject` is false only in the arm where the reference was asked about
 * a different frame, so "libzstd accepted this" is not a claim this call can
 * make; the fail-open direction is asserted in both. */
void Compare(const char* where, const Bytes& frame, const Reference& reference,
             bool strict_reject, size_t input_size) {
    const Run run = cudec_twin::DecodeFrame(frame, kDecodeCapacity);
    if (run.ok) {
        if (!reference.ok) {
            Trap("cudec accepted a frame libzstd refuses", where, input_size);
        }
        if (run.output.size() != reference.output.size()) {
            Trap("regenerated size differs from libzstd", where, input_size);
        }
#ifdef CUDEC_FUZZ_SELFTEST_BREAK
        /* The self-test twin: perturb the reference's own output so the byte
         * comparison below MUST fire on any accepted input. A binary built
         * this way that runs to its time limit instead of trapping is a
         * harness that has stopped comparing. */
        Bytes perturbed = reference.output;
        if (!perturbed.empty()) {
            perturbed[perturbed.size() - 1] ^= 0xff;
        }
        if (!run.output.empty() &&
            std::memcmp(run.output.data(), perturbed.data(),
                        run.output.size()) != 0) {
            Trap("regenerated bytes differ from libzstd", where, input_size);
        }
#else
        if (!run.output.empty() &&
            std::memcmp(run.output.data(), reference.output.data(),
                        run.output.size()) != 0) {
            Trap("regenerated bytes differ from libzstd", where, input_size);
        }
#endif
        /* The coverage evidence issue #194 asks for, printed rather than
         * counted in silence: a reader of the job's output should be able to
         * see that the sequence execution was entered under the pass that was
         * built to reach it, and a run where this never fires is a run that
         * decoded no sequence at all. */
        if (run.sequences > 0) {
            if (g_execution_reached == 0) {
                std::fprintf(stderr,
                             "NOTE: the sequence execution was entered; "
                             "pass=%s sequences=%zu blocks=%zu\n",
                             where, run.sequences, run.blocks);
            }
            g_execution_reached++;
        }
        return;
    }
    if (!strict_reject || !reference.ok) {
        return;
    }
    /* cudec refused what libzstd accepted. The subset declining a legal frame
     * says UNSUPPORTED; anything else is cudec calling a good frame corrupt,
     * which is the over-strictness this assertion exists to find. */
    if (run.status == CUDEC_ERR_UNSUPPORTED) {
        return;
    }
    if (run.stage == cudec_twin::kStageTrailingBytes &&
        HoldsMoreThanOneFrame(frame)) {
        /* Said once per run rather than counted in silence, for the reason the
         * note above gives: the exception is real, and a reader should be able
         * to see that it was taken. */
        if (g_trailing_bytes_exception == 0) {
            std::fprintf(stderr,
                         "NOTE: a buffer holding more than one frame was "
                         "refused at the trailing-bytes rule (section 12.4); "
                         "not a divergence\n");
        }
        g_trailing_bytes_exception++;
        return;
    }
    std::fprintf(stderr, "stage=%s status=%d rung=%d why=%s\n",
                 cudec_twin::kStageNames[run.stage],
                 static_cast<int>(run.status), run.rung, run.why.c_str());
    Trap("cudec refused a frame libzstd accepts, and not as UNSUPPORTED",
         where, input_size);
}

void RawPass(const unsigned char* data, size_t size) {
    Bytes frame(data, data + size);
    Compare("raw", frame, Decode(frame), true, size);
}

void StructuredPass(const unsigned char* data, size_t size) {
    /* First frame: no declared content size, which is what lets the reference
     * decode a body whose regenerated size nobody knows yet. */
    const Bytes sizeless = StructuredFrame(data, size, false, 0);
    const Reference measured = Decode(sizeless);
    if (!measured.ok) {
        /* No size to write down and no shared question to ask - but the two
         * sides still disagree about the BLOCK, and that is decidable without
         * one. libzstd refused a body; cudec is handed the same body inside a
         * frame declaring a fixed size, and either of two outcomes is the
         * fail-open direction. Accepting it outright means cudec regenerated
         * exactly that many bytes from a block libzstd calls corrupt. Refusing
         * it at the CONTENT-SIZE rule means the same thing one step later: that
         * rule runs after the block is decoded, so reaching it at all says the
         * block was decoded, and only the number it produced disagreed. Every
         * earlier stage is inside the block and is a refusal the two sides
         * agree on. */
        const Bytes probe = StructuredFrame(data, size, true,
                                            kProbeContentSize);
        const Run run = cudec_twin::DecodeFrame(probe, kDecodeCapacity);
        if (run.ok) {
            Trap("cudec accepted a block libzstd refuses", "unmeasured", size);
        }
        if (run.stage == cudec_twin::kStageContentSize) {
            std::fprintf(stderr, "stage=%s status=%d rung=%d why=%s\n",
                         cudec_twin::kStageNames[run.stage],
                         static_cast<int>(run.status), run.rung,
                         run.why.c_str());
            Trap("cudec decoded a block libzstd refuses, and stopped only on "
                 "the declared size", "unmeasured", size);
        }
        return;
    }
    const Bytes frame = StructuredFrame(data, size, true,
                                        measured.output.size());
    const Reference reference = Decode(frame);
    /* The rebuild is proven, not assumed: writing a content size into a frame
     * whose body did not change cannot change what the reference produces, and
     * a run where it did would mean the envelope is not what this file says it
     * is. */
    if (!reference.ok || reference.output != measured.output) {
        Trap("the rebuilt frame is not the one the size came from",
             "structured", size);
    }
    Compare("structured", frame, reference, true, size);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    size_t section_size = size;
    if (section_size > kMaxSection) {
        section_size = kMaxSection;
    }
    /* libFuzzer hands out a slice of a buffer sized to -max_len rather than to
     * this input, so a read past the input would land in that slack and stay
     * green. Both passes run over exactly-sized copies instead: the raw pass
     * builds one, and the structure-aware pass copies the slice into a frame
     * it assembles. */
    RawPass(data, section_size);
    StructuredPass(data, section_size);
    return 0;
}
