/* The wave-width parameter, proved where nothing else can prove it (issue
 * #241).
 *
 * WHAT THIS TEST IS FOR, AND WHY IT IS NOT COVERED BY THE REST OF THE SUITE.
 * The decode kernel family is now `template <..., int WaveSize>`, and every
 * lane stride and geometry constant is derived from that parameter. The
 * hardware here is NVIDIA, so every other test in this tree exercises exactly
 * one width - a wave64 instantiation that fails to compile, or a constant
 * that quietly stayed a wave32 one, is invisible to all of them. This test
 * makes the second instantiation a build product, so the failure surfaces on
 * the machine that cannot run it.
 *
 * THE WAVE64 KERNEL IS INSTANTIATED AND NEVER LAUNCHED, and that is the
 * point rather than a limitation. Launching it on a wave32 device would be a
 * geometry the kernel's own guard refuses (a block that is not a whole
 * number of waves), so the run would prove nothing about wave64 and would
 * report a guard firing as a pass. What is claimed here is exactly what an
 * NVIDIA machine can claim: the width-parameterised source compiles at both
 * widths and derives the block shape from the parameter. Whether a wave64
 * device decodes correctly is #212 and the AMD validation ladder, and this
 * file asserts nothing about it.
 *
 * THE SHIPPED WIDTH IS RUN, because a compile-only test would pass over a
 * kernel whose strides had been ported wrong in a way that still compiles. A
 * chunk large enough to need several strides through the copy loops is
 * decoded through the direct launch, at the block shape derived from the
 * parameter, and its bytes are compared. */
#include "chunk_decode.cuh"
#include "cudec.h"
#include "lz4_block.h"
#include "require.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/* The two widths the family is written for. The second is not this machine's
 * and is named here rather than at the instantiations below, so the digits
 * appear once and bound to a name (issue #211). */
constexpr int kWave32 = 32;
constexpr int kWave64 = 64;

/* The block shape follows the parameter rather than a constant. Asserted
 * rather than commented: this is the arithmetic every launch's thread count
 * comes out of, and a shape that stopped tracking the width would give a
 * wave64 device a block of the wrong number of waves. */
static_assert(cudec_detail::kBlockThreadsFor<kWave32> ==
                  cudec_detail::kBlockWarps * kWave32,
              "the block shape must be waves-per-block times the wave width");
static_assert(cudec_detail::kBlockThreadsFor<kWave64> ==
                  cudec_detail::kBlockWarps * kWave64,
              "the block shape must be waves-per-block times the wave width");
static_assert(cudec_detail::kBlockThreadsFor<kWave64> !=
                  cudec_detail::kBlockThreadsFor<kWave32>,
              "a block shape that does not move with the width is a constant "
              "wearing the parameter's name");
static_assert(cudec_detail::kCudaWaveSize == kWave32,
              "the CUDA build instantiates one width, and it is this one");

/* Both instantiations, asked about through the driver rather than merely
 * named. A __global__ function nothing references produces no code, so a
 * wave64 body that does not compile would go unnoticed; and asking the
 * driver for each one's attributes proves the kernel is in the binary rather
 * than proving that a symbol could be spelled.
 *
 * The attribute read is also the check itself. `maxThreadsPerBlock` is what
 * __launch_bounds__ put on the kernel, and that bound is written in terms of
 * the parameter, so a launch bound that stopped following the width shows up
 * here as a number that did not move. That is the one thing about the wave64
 * kernel an NVIDIA device can actually answer. */
int BothInstantiationsAreInTheBinary() {
    /* REQUIRE returns 1 on failure, so this reports rather than aborts and
     * main turns a non-zero answer into the test's own failure. */
    using cudec_detail::chunk_decode_batch;
    using cudec_detail::Lz4Parser;
    cudaFuncAttributes at_wave32;
    cudaFuncAttributes at_wave64;
    REQUIRE_CUDA(cudaFuncGetAttributes(
        &at_wave32, chunk_decode_batch<Lz4Parser, false, kWave32>));
    REQUIRE_CUDA(cudaFuncGetAttributes(
        &at_wave64, chunk_decode_batch<Lz4Parser, false, kWave64>));
    REQUIRE(at_wave32.maxThreadsPerBlock ==
            static_cast<int>(cudec_detail::kBlockThreadsFor<kWave32>));
    REQUIRE(at_wave64.maxThreadsPerBlock ==
            static_cast<int>(cudec_detail::kBlockThreadsFor<kWave64>));
    return 0;
}

/* The literal length a token's own 4-bit field can hold; anything longer is
 * spelled with the extension bytes below. */
constexpr size_t kLz4TokenLiteralMax = 15;

/* A literals-only LZ4 block long enough that the copy loop takes several
 * strides at either width: a wave64 stride on a wave32 launch would leave
 * every byte between the two strides unwritten, and a wave32 stride is the
 * shipped one. The payload below is well past the token's own field, so the
 * length extension is exercised too rather than a token the parser could
 * special-case away.
 *
 * The caller's payload is longer than kLz4TokenLiteralMax by construction,
 * which is why this carries no check of its own: a fixture builder that
 * cannot fail is better than one whose failure path nothing runs. */
std::vector<unsigned char> LiteralsOnlyBlock(const std::vector<unsigned char>&
                                                 payload) {
    std::vector<unsigned char> block;
    const size_t n = payload.size();
    block.push_back(0xF0); /* the token's literal-length field, all ones */
    size_t remaining = n - kLz4TokenLiteralMax;
    while (remaining >= 255) {
        block.push_back(0xFF);
        remaining -= 255;
    }
    block.push_back(static_cast<unsigned char>(remaining));
    block.insert(block.end(), payload.begin(), payload.end());
    return block;
}

}  // namespace

int main() {
    REQUIRE(BothInstantiationsAreInTheBinary() == 0);

    /* Comfortably past the token's own literal-length field and past a
     * single stride of either width, so both the extension bytes and the
     * strided copy are exercised. */
    std::vector<unsigned char> payload(300);
    static_assert(300 > kLz4TokenLiteralMax,
                  "the block builder below spells the length extension "
                  "unconditionally");
    for (size_t i = 0; i < payload.size(); i++) {
        payload[i] = static_cast<unsigned char>(i * 7 + 1);
    }
    const std::vector<unsigned char> block = LiteralsOnlyBlock(payload);

    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    const void** d_src_ptrs = nullptr;
    void** d_dst_ptrs = nullptr;
    size_t* d_sizes = nullptr;
    size_t* d_caps = nullptr;
    cudec_chunk_result* d_results = nullptr;
    REQUIRE_CUDA(cudaMalloc(&d_src, block.size()));
    REQUIRE_CUDA(cudaMalloc(&d_dst, payload.size()));
    REQUIRE_CUDA(cudaMalloc(&d_src_ptrs, sizeof(void*)));
    REQUIRE_CUDA(cudaMalloc(&d_dst_ptrs, sizeof(void*)));
    REQUIRE_CUDA(cudaMalloc(&d_sizes, sizeof(size_t)));
    REQUIRE_CUDA(cudaMalloc(&d_caps, sizeof(size_t)));
    REQUIRE_CUDA(cudaMalloc(&d_results, sizeof(cudec_chunk_result)));
    REQUIRE_CUDA(cudaMemcpy(d_src, block.data(), block.size(),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemset(d_dst, 0, payload.size()));

    const void* h_src_ptr = d_src;
    void* h_dst_ptr = d_dst;
    const size_t h_size = block.size();
    const size_t h_cap = payload.size();
    REQUIRE_CUDA(cudaMemcpy(d_src_ptrs, &h_src_ptr, sizeof(void*),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(cudaMemcpy(d_dst_ptrs, &h_dst_ptr, sizeof(void*),
                            cudaMemcpyHostToDevice));
    REQUIRE_CUDA(
        cudaMemcpy(d_sizes, &h_size, sizeof(size_t), cudaMemcpyHostToDevice));
    REQUIRE_CUDA(
        cudaMemcpy(d_caps, &h_cap, sizeof(size_t), cudaMemcpyHostToDevice));

    /* The launch geometry is derived from the parameter, exactly as the
     * shipped entry derives it. Hoisted into a named constant because the
     * template argument's closing angle bracket would otherwise sit against
     * the launch syntax's own. */
    constexpr unsigned kThreads =
        cudec_detail::kBlockThreadsFor<cudec_detail::kCudaWaveSize>;
    cudec_detail::chunk_decode_batch<cudec_detail::Lz4Parser, false,
                                     cudec_detail::kCudaWaveSize>
        <<<1, kThreads>>>(d_src_ptrs, d_sizes, d_dst_ptrs, d_caps, 1,
                          d_results);
    REQUIRE_CUDA(cudaGetLastError());
    REQUIRE_CUDA(cudaDeviceSynchronize());

    cudec_chunk_result result;
    REQUIRE_CUDA(cudaMemcpy(&result, d_results, sizeof(result),
                            cudaMemcpyDeviceToHost));
    REQUIRE(result.status == CUDEC_OK);
    REQUIRE(result.bytes_written == payload.size());

    std::vector<unsigned char> out(payload.size(), 0);
    REQUIRE_CUDA(cudaMemcpy(out.data(), d_dst, out.size(),
                            cudaMemcpyDeviceToHost));
    REQUIRE(std::memcmp(out.data(), payload.data(), payload.size()) == 0);

    std::printf("PASS: both wave widths instantiate; the shipped width "
                "decodes %zu bytes at %u threads per block\n",
                payload.size(), kThreads);
    return 0;
}
