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
 * The bytes are the single-sourced hostile corpus of
 * tests/adversarial_snappy_blocks.h (issue #154), which the whole-batch
 * device gate drives through the public entry as well - so a stream that
 * behaves under one thread and misbehaves under a warp has two places to be
 * caught rather than one.
 *
 * This is not itself the device gate set: one thread, no warp cooperation,
 * no kernel. What it owns is the claim that one header compiled twice
 * decides identically. */
#include "adversarial_snappy_blocks.h"
#include "cudec.h"
#include "require.h"
#include "snappy_block.h"

#include "vendor_rt_test.h"

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
    cudec_detail::DecodeSequence element;
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
        digest = Mix(digest, element.literals_src);
        digest = Mix(digest, element.literals_dst);
        digest = Mix(digest, element.literals_len);
        digest = Mix(digest, element.match_src);
        digest = Mix(digest, element.match_dst);
        digest = Mix(digest, element.match_len);
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
                              const unsigned long long* capacities,
                              unsigned count, Trace* traces) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    RunParser(streams + offsets[index], sizes[index], capacities[index],
              &traces[index]);
}

}  // namespace

int main() {
    const std::vector<AdversarialSnappyBlock> blocks =
        MakeAdversarialSnappyBlocks();
    const unsigned count = static_cast<unsigned>(blocks.size());
    REQUIRE(count > 0);

    Bytes flat;
    std::vector<unsigned long long> offsets;
    std::vector<unsigned long long> sizes;
    std::vector<unsigned long long> caps;
    for (const auto& b : blocks) {
        offsets.push_back(flat.size());
        sizes.push_back(b.stream.size());
        caps.push_back(b.dst_capacity);
        flat.insert(flat.end(), b.stream.begin(), b.stream.end());
    }
    /* One byte of slack, so a zero-length stream still has a valid base
     * pointer to hand the parser and the allocation is never zero-sized. */
    flat.push_back(0);

    unsigned char* d_streams = nullptr;
    unsigned long long* d_offsets = nullptr;
    unsigned long long* d_sizes = nullptr;
    unsigned long long* d_caps = nullptr;
    Trace* d_traces = nullptr;
    REQUIRE_RT(cudec_rt::device_malloc(&d_streams, flat.size()));
    REQUIRE_RT(cudec_rt::device_malloc(&d_offsets,
                                       offsets.size() * sizeof(*d_offsets)));
    REQUIRE_RT(
        cudec_rt::device_malloc(&d_sizes, sizes.size() * sizeof(*d_sizes)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_caps, caps.size() * sizeof(*d_caps)));
    REQUIRE_RT(cudec_rt::device_malloc(&d_traces, count * sizeof(*d_traces)));
    REQUIRE_RT(cudec_rt::memcpy(d_streams, flat.data(), flat.size(),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_offsets, offsets.data(),
                            offsets.size() * sizeof(*d_offsets),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_sizes, sizes.data(),
                            sizes.size() * sizeof(*d_sizes),
                            cudec_rt::memcpy_h2d));
    REQUIRE_RT(cudec_rt::memcpy(d_caps, caps.data(),
                            caps.size() * sizeof(*d_caps),
                            cudec_rt::memcpy_h2d));
    /* Poisoned, so a trace the kernel never wrote cannot pass as a match. */
    REQUIRE_RT(
        cudec_rt::device_memset(d_traces, 0xFF, count * sizeof(*d_traces)));

    ParseOnDevice<<<(count + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock>>>(d_streams, d_offsets, d_sizes,
                                        d_caps, count, d_traces);
    REQUIRE_RT(cudec_rt::get_last_error());
    REQUIRE_RT(cudec_rt::device_synchronize());

    std::vector<Trace> device_traces(count);
    REQUIRE_RT(cudec_rt::memcpy(device_traces.data(), d_traces,
                            count * sizeof(Trace), cudec_rt::memcpy_d2h));

    size_t accepted = 0;
    for (unsigned i = 0; i < count; i++) {
        Trace host_trace;
        RunParser(flat.data() + offsets[i], sizes[i], caps[i], &host_trace);
        const Trace& device_trace = device_traces[i];
        const char* name = blocks[i].name.c_str();
        REQUIRE_CTX(host_trace.status == device_trace.status,
                    "%s: host status %d, device status %d", name,
                    host_trace.status, device_trace.status);
        REQUIRE_CTX(host_trace.produced == device_trace.produced,
                    "%s: produced %llu vs %llu", name, host_trace.produced,
                    device_trace.produced);
        REQUIRE_CTX(host_trace.elements == device_trace.elements,
                    "%s: %llu elements vs %llu", name, host_trace.elements,
                    device_trace.elements);
        REQUIRE_CTX(host_trace.digest == device_trace.digest,
                    "%s: element digest %llx vs %llx", name,
                    host_trace.digest, device_trace.digest);
        /* The fuel budget must never be what ended a parse, on either side. */
        REQUIRE_CTX(host_trace.out_of_fuel == 0,
                    "%s did not terminate on the host", name);
        REQUIRE_CTX(device_trace.out_of_fuel == 0,
                    "%s did not terminate on the device", name);
        if (host_trace.status == CUDEC_OK) {
            accepted++;
        }
    }
    /* Both directions, so the comparison is not vacuous on a corpus that
     * only rejects or only accepts. */
    REQUIRE(accepted > 0);
    REQUIRE(accepted < count);

    REQUIRE_RT(cudec_rt::device_free(d_streams));
    REQUIRE_RT(cudec_rt::device_free(d_offsets));
    REQUIRE_RT(cudec_rt::device_free(d_sizes));
    REQUIRE_RT(cudec_rt::device_free(d_caps));
    REQUIRE_RT(cudec_rt::device_free(d_traces));

    std::printf("PASS: %u streams parsed identically by the host compiler and "
                "nvcc from one header (%zu accepted, %zu rejected)\n",
                count, accepted, count - accepted);
    return 0;
}
