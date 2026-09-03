#include "cudec.h"
#include "chunk_decode.cuh"
#include "batch_limits.h"
#include "lz4_block.h"
#include "snappy_block.h"

#include "vendor_rt.h"

namespace {

/* The launch, written once for both widths. The grid is a whole number of
 * blocks and each block is kBlockWarps waves at whichever width was selected,
 * so the geometry the kernel refuses - a block that is not a wave multiple -
 * cannot be produced by either arm. */
template <class Parser, int WaveSize>
void launch_decode(const void* const* d_src_ptrs, const size_t* d_src_sizes,
                   void* const* d_dst_ptrs, const size_t* d_dst_capacities,
                   size_t chunk_count, cudec_chunk_result* d_results,
                   cudec_stream_t stream) {
    cudec_detail::chunk_decode_batch<Parser, false, WaveSize>
        <<<cudec_detail::decode_grid_blocks(chunk_count),
           cudec_detail::kBlockThreadsFor<WaveSize>, 0,
           cudec_rt::stream_from_abi(stream)>>>(
            d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
            d_results);
}

/* The two entries differ in one template argument and in nothing else, so
 * the body they share is written once. Templated rather than handed a
 * function pointer: a launch through an indirect call would put the kernel
 * address in device memory and cost the compiler the inlining the parser
 * depends on. */
template <class Parser>
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
            launch_decode<Parser, cudec_detail::kWaveWidth32>(
                d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities,
                chunk_count, d_results, stream);
            break;
        case cudec_detail::WaveInstantiation::kWave64:
            launch_decode<Parser, cudec_detail::kWaveWidth64>(
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
    return submit_batch<cudec_detail::Lz4Parser>(
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
    return submit_batch<cudec_detail::SnappyParser>(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results, stream);
}

/* The GDeflate and Zstd entries are the frozen contracts without their
 * kernels behind them yet (issues #216 and #427). Each shares the validator
 * with the two above rather than growing its own, so the reject classes
 * cannot drift apart while they are being frozen, and then stops: no launch,
 * and deliberately no cudec_rt::get_last_error() drain either. Draining is how the entries above
 * buy a
 * post-launch check that reports their own submission; with nothing
 * submitted there is nothing to report, and consuming a caller's pending
 * error on the way to answering "not built here" would be this entry
 * altering CUDA state it never used.
 *
 * That absence is what tests/launch_fail.cpp reads: with no visible device
 * an entry that made any CUDA call would answer CUDEC_ERR_CUDA, so the
 * not-implemented answer there is the evidence that this path touches the
 * runtime at all only through the validator, which touches it not at
 * all. */
cudec_status cudec_gdeflate_decompress_batch(const void* const* d_src_ptrs,
                                             const size_t* d_src_sizes,
                                             void* const* d_dst_ptrs,
                                             const size_t* d_dst_capacities,
                                             size_t chunk_count,
                                             cudec_chunk_result* d_results,
                                             cudec_stream_t stream) {
    (void)stream;
    const cudec_status valid = cudec_detail::validate_batch_args(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results);
    if (valid != CUDEC_OK) {
        return valid;
    }
    return CUDEC_ERR_NOT_IMPLEMENTED;
}

/* The Zstd entry, the same shape and for the same reasons - the comment above
 * covers both, and the unit each accepts is the only thing that differs
 * between them. */
cudec_status cudec_zstd_decompress_batch(const void* const* d_src_ptrs,
                                         const size_t* d_src_sizes,
                                         void* const* d_dst_ptrs,
                                         const size_t* d_dst_capacities,
                                         size_t chunk_count,
                                         cudec_chunk_result* d_results,
                                         cudec_stream_t stream) {
    (void)stream;
    const cudec_status valid = cudec_detail::validate_batch_args(
        d_src_ptrs, d_src_sizes, d_dst_ptrs, d_dst_capacities, chunk_count,
        d_results);
    if (valid != CUDEC_OK) {
        return valid;
    }
    return CUDEC_ERR_NOT_IMPLEMENTED;
}
