#include "cudec.h"
#include "chunk_decode.cuh"
#include "batch_limits.h"
#include "gdeflate_decode.cuh"
#include "lz4_block.h"
#include "snappy_block.h"
#include "zstd_decode.cuh"

#include "vendor_rt.h"

namespace {

/* The launches, written once per kernel family and once for both widths.
 * The grid is a whole number of blocks and each block is kBlockWarps waves at
 * whichever width was selected, so the geometry the kernels refuse - a block
 * that is not a wave multiple - cannot be produced by either arm. A launcher
 * is a type with one static template rather than a function template,
 * because the submission below is written once over the launcher and a
 * function template cannot be a template argument. */
template <class Parser>
struct ChunkLaunch {
    template <int WaveSize>
    static void Run(const void* const* d_src_ptrs, const size_t* d_src_sizes,
                    void* const* d_dst_ptrs, const size_t* d_dst_capacities,
                    size_t chunk_count, cudec_chunk_result* d_results,
                    cudec_stream_t stream) {
        cudec_detail::chunk_decode_batch<Parser, false, WaveSize>
            <<<cudec_detail::decode_grid_blocks(chunk_count),
               cudec_detail::kBlockThreadsFor<WaveSize>, 0,
               cudec_rt::stream_from_abi(stream)>>>(
                d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities,
                chunk_count, d_results);
    }
};

/* One team per page rather than one lane-lockstep parse per chunk, and the
 * same block shape: kBlockWarps teams to a block, so the grid arithmetic the
 * chunk decoder settled serves the page decoder unchanged. */
struct GDeflateLaunch {
    template <int WaveSize>
    static void Run(const void* const* d_src_ptrs, const size_t* d_src_sizes,
                    void* const* d_dst_ptrs, const size_t* d_dst_capacities,
                    size_t chunk_count, cudec_chunk_result* d_results,
                    cudec_stream_t stream) {
        cudec_detail::gdeflate_decode_batch<WaveSize>
            <<<cudec_detail::decode_grid_blocks(chunk_count),
               cudec_detail::kBlockThreadsFor<WaveSize>, 0,
               cudec_rt::stream_from_abi(stream)>>>(
                d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities,
                chunk_count, d_results);
    }
};

/* One BLOCK per frame rather than one wave, and the grid is one block per
 * frame: a Zstd frame's entropy decode is serial per stream, so what a frame
 * gets is a block whose shared memory holds its table set
 * (docs/MASTERPLAN.md section 14.2). The width is the kernel's own constant
 * rather than the wave-derived one the two launchers above use, because 14.2
 * derives it from the table set against an SM's shared memory and not from a
 * wave count - it is 128 threads at either wave width, which is four waves on
 * one and two on the other. */
struct ZstdLaunch {
    template <int WaveSize>
    static void Run(const void* const* d_src_ptrs, const size_t* d_src_sizes,
                    void* const* d_dst_ptrs, const size_t* d_dst_capacities,
                    size_t chunk_count, cudec_chunk_result* d_results,
                    cudec_stream_t stream) {
        cudec_detail::zstd_decode_batch<WaveSize>
            <<<cudec_detail::zstd_grid_blocks(chunk_count),
               cudec_detail::kZstdBlockThreads, 0,
               cudec_rt::stream_from_abi(stream)>>>(
                d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities,
                chunk_count, d_results);
    }
};

/* The entries differ in which launcher they name and in nothing else, so the
 * body they share is written once. Templated rather than handed a function
 * pointer: a launch through an indirect call would put the kernel address in
 * device memory and cost the compiler the inlining the parser depends on. */
template <class Launch>
cudec_status submit_batch(const void* const* d_src_ptrs,
                          const size_t* d_src_sizes, void* const* d_dst_ptrs,
                          const size_t* d_dst_capacities, size_t chunk_count,
                          cudec_chunk_result* d_results,
                          cudec_stream_t stream) {
    const cudec_status valid = cudec_detail::validate_batch_args(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results);
    if (valid != CUDEC_OK) {
        return valid;
    }

    /* Drain any error already pending on this thread so the post-launch
     * check reports this submission alone; the header documents that the
     * call consumes the pending error state. Before the width query rather
     * than after it, so a fault this call inherits cannot be reported as the
     * query's own. */
    (void)cudec_rt::get_last_error();

    /* The width is resolved per submission rather than once, and the reason is
     * correctness rather than caution: a process may change device between two
     * submissions, and a width cached for the first one is a launch geometry
     * chosen for the wrong GPU. On a backend that fixes the width this costs a
     * constant; on one that does not it is an attribute read against a kernel
     * launch, and that ratio is NOT MEASURED - no device was reachable when
     * this landed, and no backend that would pay it has a compiler here. */
    int reported_width = 0;
    if (cudec_rt::wave_width_for_launch(&reported_width) != cudec_rt::success) {
        return CUDEC_ERR_CUDA;
    }

    /* Both instantiations are emitted on both backends, and on CUDA the
     * wave64 arm is unreachable by construction - no CUDA device reports 64.
     * It is emitted anyway because a compiler is the only thing that can say
     * the second half of the width family still builds, and the CUDA
     * toolchain is the one this project's gate actually runs. Stripping it
     * where it cannot be launched would mean the wave64 kernel is first
     * compiled by whoever first has ROCm, which is the opposite of a lock.
     * #212 holds the same property for the HIP binary. */
    switch (cudec_detail::select_wave_instantiation(reported_width)) {
        case cudec_detail::WaveInstantiation::kWave32:
            Launch::template Run<cudec_detail::kWaveWidth32>(
                d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities,
                chunk_count, d_results, stream);
            break;
        case cudec_detail::WaveInstantiation::kWave64:
            Launch::template Run<cudec_detail::kWaveWidth64>(
                d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities,
                chunk_count, d_results, stream);
            break;
        case cudec_detail::WaveInstantiation::kUnsupported:
            return CUDEC_ERR_UNSUPPORTED;
    }
    return cudec_rt::get_last_error() == cudec_rt::success ? CUDEC_OK
                                                           : CUDEC_ERR_CUDA;
}

}  // namespace

cudec_status cudec_lz4_decompress_batch(const void* const* d_src_ptrs,
                                        const size_t* d_src_sizes,
                                        void* const* d_dst_ptrs,
                                        const size_t* d_dst_capacities,
                                        size_t chunk_count,
                                        cudec_chunk_result* d_results,
                                        cudec_stream_t stream) {
    return submit_batch<ChunkLaunch<cudec_detail::Lz4Parser> >(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results, stream);
}

cudec_status cudec_snappy_decompress_batch(const void* const* d_src_ptrs,
                                           const size_t* d_src_sizes,
                                           void* const* d_dst_ptrs,
                                           const size_t* d_dst_capacities,
                                           size_t chunk_count,
                                           cudec_chunk_result* d_results,
                                           cudec_stream_t stream) {
    return submit_batch<ChunkLaunch<cudec_detail::SnappyParser> >(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results, stream);
}

/* The GDeflate entry: the frozen contract with its kernel behind it (issues
 * #216, #214). It shares the validator, the width query and the post-launch
 * check with the two above, so the reject classes and the launch discipline
 * cannot drift apart between the families. */
cudec_status cudec_gdeflate_decompress_batch(const void* const* d_src_ptrs,
                                             const size_t* d_src_sizes,
                                             void* const* d_dst_ptrs,
                                             const size_t* d_dst_capacities,
                                             size_t chunk_count,
                                             cudec_chunk_result* d_results,
                                             cudec_stream_t stream) {
    return submit_batch<GDeflateLaunch>(d_src_ptrs, d_src_sizes, d_dst_ptrs,
                                        d_dst_capacities, chunk_count,
                                        d_results, stream);
}

/* The Zstd entry with its kernel behind it (issues #427, #203). It shares the
 * validator, the width query and the post-launch check with the three above,
 * so the reject classes and the launch discipline cannot drift apart between
 * the families; what it does not share is the launch geometry, which is
 * ZstdLaunch's and is a block per frame.
 *
 * tests/launch_fail.cpp read the not-implemented answer this entry used to
 * give as the evidence that it touched the runtime only through a validator
 * that touches it not at all. That evidence is spent: with a kernel behind it
 * the entry queries the width and launches, so with no visible device it
 * answers CUDEC_ERR_CUDA exactly as the three entries beside it do, and that
 * is the line in that test which changes meaning today. */
cudec_status cudec_zstd_decompress_batch(const void* const* d_src_ptrs,
                                         const size_t* d_src_sizes,
                                         void* const* d_dst_ptrs,
                                         const size_t* d_dst_capacities,
                                         size_t chunk_count,
                                         cudec_chunk_result* d_results,
                                         cudec_stream_t stream) {
    return submit_batch<ZstdLaunch>(d_src_ptrs, d_src_sizes, d_dst_ptrs,
                                    d_dst_capacities, chunk_count, d_results,
                                    stream);
}
