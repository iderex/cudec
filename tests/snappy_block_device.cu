/* The "single-source host and device" half of src/snappy_block.h, executed
 * rather than asserted (issue #171). The header is written __host__
 * __device__ and the CPU twin exercises only the host half, so without this
 * nothing in the tree ever hands it to nvcc: the property would be a comment.
 *
 * What runs: one thread parses each stream on the device and records a
 * trace - the final status, the produced length, the element count and a
 * digest over every element field in order. The host runs the same parser
 * over the same bytes and the two traces must be identical. That is the
 * single-source claim stated as a comparison, and it is also the
 * determinism claim at the granularity this header owns: the same input
 * produces the same element sequence on both.
 *
 * Not the device gate set (#154), which is a kernel's business. There is no
 * kernel yet; this is the header on a device, one thread, no warp
 * cooperation. */
#include "cudec.h"
#include "require.h"
#include "snappy_block.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

/* The whole observable behaviour of one parse, in a form two sides can
 * compare byte for byte. */
struct Trace {
    int status;
    int out_of_fuel;
    unsigned long long produced;
    unsigned long long elements;
    unsigned long long digest;
};

/* One warp per launch block. Named rather than written twice, so the two
 * places that need it cannot drift. */
constexpr unsigned kThreadsPerBlock = 32;

/* FNV-1a over the element fields, in order. Not crypto: a difference
 * detector that does not need the element sequence copied off the device. */
CUDEC_HOST_DEVICE inline unsigned long long Mix(unsigned long long hash,
                                                unsigned long long value) {
    for (unsigned i = 0; i < sizeof(value); i++) {
        hash ^= (value >> (8 * i)) & 0xFF;
        hash *= 1099511628211ull;
    }
    return hash;
}

CUDEC_HOST_DEVICE inline void RunParser(const unsigned char* src,
                                        unsigned long long src_size,
                                        unsigned long long capacity,
                                        Trace* trace) {
    cudec_detail::SnappyParser parser{src, src_size, capacity};
    cudec_detail::SnappyElement element;
    bool done = false;
    unsigned long long digest = 14695981039346656037ull;
    unsigned long long elements = 0;
    cudec_status status = CUDEC_OK;
    int out_of_fuel = 0;
    /* Fuel: every non-terminal call consumes at least one source byte, so
     * this budget is unreachable for a live parser - and a regression that
     * stopped consuming reds this test instead of holding the device. */
    unsigned long long fuel = src_size + 2;
    while (true) {
        if (fuel-- == 0) {
            out_of_fuel = 1;
            break;
        }
        status = parser.Next(&element, &done);
        if (status != CUDEC_OK) {
            break;
        }
        digest = Mix(digest, element.from);
        digest = Mix(digest, element.to);
        digest = Mix(digest, element.length);
        digest = Mix(digest, element.is_copy ? 1u : 0u);
        elements++;
        if (done) {
            break;
        }
    }
    trace->status = static_cast<int>(status);
    trace->out_of_fuel = out_of_fuel;
    trace->produced = parser.dst_pos;
    trace->elements = elements;
    trace->digest = digest;
}

__global__ void ParseOnDevice(const unsigned char* streams,
                              const unsigned long long* offsets,
                              const unsigned long long* sizes,
                              unsigned long long capacity, unsigned count,
                              Trace* traces) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    RunParser(streams + offsets[index], sizes[index], capacity,
              &traces[index]);
}

Bytes Preamble(unsigned length) {
    Bytes out;
    unsigned v = length;
    while (v >= 0x80) {
        out.push_back(static_cast<unsigned char>((v & 0x7f) | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<unsigned char>(v));
    return out;
}

void Append(Bytes* out, const Bytes& more) {
    out->insert(out->end(), more.begin(), more.end());
}

Bytes Literal(const char* text, size_t size) {
    Bytes out;
    out.push_back(static_cast<unsigned char>((size - 1) << 2));
    for (size_t i = 0; i < size; i++) {
        out.push_back(static_cast<unsigned char>(text[i]));
    }
    return out;
}

Bytes Copy2(unsigned length, unsigned offset) {
    Bytes out;
    out.push_back(static_cast<unsigned char>(((length - 1) << 2) | 2));
    out.push_back(static_cast<unsigned char>(offset & 0xff));
    out.push_back(static_cast<unsigned char>((offset >> 8) & 0xff));
    return out;
}

/* Valid streams, rejects from several rungs of the ladder, and the shapes
 * whose arithmetic differs between the two compilers if anything does: the
 * wide length classes and the wide offset form. */
std::vector<Bytes> MakeStreams() {
    std::vector<Bytes> out;
    out.push_back(Bytes{});                     /* no preamble */
    out.push_back(Preamble(0));                 /* a valid empty stream */
    {
        Bytes s = Preamble(4);
        Append(&s, Literal("abcd", 4));
        out.push_back(s);                       /* one literal */
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal("abcd", 4));
        Append(&s, Copy2(4, 4));
        out.push_back(s);                       /* literal then copy */
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal("ab", 2));
        Append(&s, Copy2(6, 1));
        out.push_back(s);                       /* an overlapping copy */
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal("abcd", 4));
        Append(&s, Copy2(4, 0));
        out.push_back(s);                       /* offset zero, refused */
    }
    {
        Bytes s = Preamble(8);
        Append(&s, Literal("abcd", 4));
        Append(&s, Copy2(4, 9));
        out.push_back(s);                       /* offset past the output */
    }
    out.push_back(Bytes{0x80, 0x80, 0x80, 0x80, 0x10}); /* varint overflow */
    out.push_back(Bytes{0x00, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF}); /* the wrap */
    {
        Bytes s = Preamble(300);
        s.push_back(0xF0); /* long-form literal, its length byte missing */
        out.push_back(s);
    }
    {
        Bytes s = Preamble(64);
        s.push_back(0xF4); /* two length bytes for a 64-byte literal */
        s.push_back(63);
        s.push_back(0);
        for (int i = 0; i < 64; i++) {
            s.push_back(static_cast<unsigned char>('a' + i % 26));
        }
        out.push_back(s);
    }
    return out;
}

}  // namespace

int main() {
    const std::vector<Bytes> streams = MakeStreams();
    const unsigned count = static_cast<unsigned>(streams.size());
    REQUIRE(count > 0);
    const unsigned long long capacity = 1u << 20;

    Bytes flat;
    std::vector<unsigned long long> offsets;
    std::vector<unsigned long long> sizes;
    for (const auto& s : streams) {
        offsets.push_back(flat.size());
        sizes.push_back(s.size());
        flat.insert(flat.end(), s.begin(), s.end());
    }
    /* One byte of slack, so a zero-length stream still has a valid base
     * pointer to hand the parser and the allocation is never zero-sized. */
    flat.push_back(0);

    unsigned char* d_streams = nullptr;
    unsigned long long* d_offsets = nullptr;
    unsigned long long* d_sizes = nullptr;
    Trace* d_traces = nullptr;
    REQUIRE_CUDA(cudaMalloc(&d_streams, flat.size()));
    REQUIRE_CUDA(cudaMalloc(&d_offsets, offsets.size() * sizeof(*d_offsets)));
    REQUIRE_CUDA(cudaMalloc(&d_sizes, sizes.size() * sizeof(*d_sizes)));
    REQUIRE_CUDA(cudaMalloc(&d_traces, count * sizeof(*d_traces)));
    REQUIRE_CUDA(cudaMemcpy(d_streams, flat.data(), flat.size(),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(d_offsets, offsets.data(),
                            offsets.size() * sizeof(*d_offsets),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(d_sizes, sizes.data(),
                            sizes.size() * sizeof(*d_sizes),
                            cudaMemcpyHostToDevice));
    /* Poisoned, so a trace the kernel never wrote cannot pass as a match. */
    REQUIRE_CUDA(cudaMemset(d_traces, 0xFF, count * sizeof(*d_traces)));

    ParseOnDevice<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock>>>(d_streams, d_offsets, d_sizes,
                                        capacity, count, d_traces);
    REQUIRE_CUDA(cudaGetLastError());
    REQUIRE_CUDA(cudaDeviceSynchronize());

    std::vector<Trace> device_traces(count);
    REQUIRE_CUDA(cudaMemcpy(device_traces.data(), d_traces,
                            count * sizeof(Trace), cudaMemcpyDeviceToHost));

    size_t accepted = 0;
    for (unsigned i = 0; i < count; i++) {
        Trace host_trace;
        RunParser(flat.data() + offsets[i], sizes[i], capacity, &host_trace);
        const Trace& device_trace = device_traces[i];
        REQUIRE_CTX(host_trace.status == device_trace.status,
                    "stream %u: host status %d, device status %d", i,
                    host_trace.status, device_trace.status);
        REQUIRE_CTX(host_trace.produced == device_trace.produced,
                    "stream %u: produced %llu vs %llu", i, host_trace.produced,
                    device_trace.produced);
        REQUIRE_CTX(host_trace.elements == device_trace.elements,
                    "stream %u: %llu elements vs %llu", i, host_trace.elements,
                    device_trace.elements);
        REQUIRE_CTX(host_trace.digest == device_trace.digest,
                    "stream %u: element digest %llx vs %llx", i,
                    host_trace.digest, device_trace.digest);
        /* The fuel budget must never be what ended a parse, on either side. */
        REQUIRE_CTX(host_trace.out_of_fuel == 0,
                    "stream %u did not terminate on the host", i);
        REQUIRE_CTX(device_trace.out_of_fuel == 0,
                    "stream %u did not terminate on the device", i);
        if (host_trace.status == CUDEC_OK) {
            accepted++;
        }
    }
    /* Both directions, so the comparison is not vacuous on a corpus that
     * only rejects or only accepts. */
    REQUIRE(accepted > 0);
    REQUIRE(accepted < count);

    REQUIRE_CUDA(cudaFree(d_streams));
    REQUIRE_CUDA(cudaFree(d_offsets));
    REQUIRE_CUDA(cudaFree(d_sizes));
    REQUIRE_CUDA(cudaFree(d_traces));

    std::printf("PASS: %u streams parsed identically by the host compiler and "
                "nvcc from one header (%zu accepted, %zu rejected)\n",
                count, accepted, count - accepted);
    return 0;
}
