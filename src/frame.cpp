/* LZ4 frame (.lz4 container) decode: host-side frame parse + the GPU block
 * batch decoder (cudec_lz4_decompress_batch) for the compressed blocks,
 * then assembly and checksum verification. Host orchestration on top of
 * the device engine - masterplan section 3 (M2). Supported subset:
 * block-independent frames; the header/block/content checksums and the
 * optional declared content size are verified fail-closed. Frame spec
 * (public): lz4_Frame_format.md. */
#include "cudec.h"
#include "cuda_raii.h"
#include "xxhash32.h"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kLz4FrameMagic = 0x184D2204u;

uint32_t Read32LE(const unsigned char* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v; /* the frame format and the target hosts are little-endian */
}

uint64_t Read64LE(const unsigned char* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

/* A parsed data block: an offset/length into the frame, and whether it is
 * stored uncompressed (a straight copy) or LZ4-compressed (GPU-decoded). */
struct FrameBlock {
    size_t src_off;
    size_t src_len;
    bool uncompressed;
};

/* Decodes the compressed blocks in one device batch and assembles the whole
 * frame output (compressed decoded bytes + uncompressed blocks copied
 * verbatim) into `out`, bounded by dst_capacity, writing the produced size
 * to *total_out. Every CUDA failure maps to a defined status; a block the
 * decoder rejects makes the whole frame CORRUPT_INPUT.
 *
 * Staging is a single source buffer and a single destination buffer (not a
 * cudaMalloc per block): device memory stays bounded, an oversized hostile
 * frame fails fast and cleanly on one allocation instead of a per-block
 * allocation storm, and the shape is closer to what the pinned-host
 * streaming path (issue #24) needs. #24 still supersedes it (overlap,
 * pinned memory, per-block dst sizing). */
cudec_status DecodeAndAssemble(const unsigned char* frame,
                               const std::vector<FrameBlock>& blocks,
                               size_t block_max, unsigned char* out,
                               size_t dst_capacity, size_t* total_out) {
#define FRAME_CUDA(call) CUDEC_CUDA_CHECK(call, return CUDEC_ERR_CUDA)

    std::vector<size_t> cidx;
    size_t total_src = 0;
    for (size_t i = 0; i < blocks.size(); i++) {
        if (!blocks[i].uncompressed) {
            cidx.push_back(i);
            total_src += blocks[i].src_len;
        }
    }
    const size_t n = cidx.size();

    /* Decoded output size per block; uncompressed blocks use src_len. */
    std::vector<size_t> decoded_len(blocks.size(), 0);

    /* One-shot owners: each buffer is allocated once here and freed on every
     * scope exit. The shared grow-only DevBuf serves this by a single ensure()
     * from cap 0 - every size below is non-zero (guarded by n != 0), so the
     * first ensure allocates exactly once, identical to a bare cudaMalloc. */
    cudec_cuda::DevBuf d_src, d_dst, dd_meta, dd_res;
    if (n != 0) {
        /* One destination slot of block_max per compressed block. Guard the
         * product against size_t overflow before asking the driver. */
        if (block_max != 0 && n > SIZE_MAX / block_max) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        FRAME_CUDA(d_src.ensure(total_src));
        FRAME_CUDA(d_dst.ensure(n * block_max));

        std::vector<const void*> h_src(n);
        std::vector<void*> h_dst(n);
        std::vector<size_t> h_ssz(n), h_dcp(n);
        {
            /* Gather the (non-contiguous) compressed block sources into one
             * host staging buffer and copy them up in a single transfer. */
            std::vector<unsigned char> stage(total_src);
            size_t so = 0;
            for (size_t k = 0; k < n; k++) {
                const FrameBlock& b = blocks[cidx[k]];
                std::memcpy(stage.data() + so, frame + b.src_off, b.src_len);
                h_src[k] = static_cast<unsigned char*>(d_src.p) + so;
                h_ssz[k] = b.src_len;
                h_dst[k] = static_cast<unsigned char*>(d_dst.p) + k * block_max;
                h_dcp[k] = block_max;
                so += b.src_len;
            }
            FRAME_CUDA(cudaMemcpy(d_src.p, stage.data(), total_src,
                                  cudaMemcpyHostToDevice));
        }

        /* The four metadata arrays go up in ONE transfer out of one packed
         * allocation, in the layout src/stream.cpp already uses:
         * [src_ptrs][src_sizes][dst_ptrs][dst_caps], each n elements wide at
         * a uniform stride. Four separate uploads is four submissions per
         * decode where one will do, and this path pays that per frame. The
         * stride is written in units of void* for every section, including
         * the two that hold size_t, so the assert below is what keeps that
         * from being a silent misalignment on a host where the two differ. */
        static_assert(sizeof(size_t) == sizeof(void*),
                      "the packed metadata gives every section a void*-sized "
                      "stride; a host where size_t is narrower would "
                      "misalign the two size sections");
        /* Packing multiplies the request by four, so the product gets the
         * same treatment n * block_max already gets above: checked before
         * the driver sees it, rejected rather than wrapped. */
        if (n > SIZE_MAX / (4 * sizeof(void*))) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        const size_t meta_stride = n * sizeof(void*);
        FRAME_CUDA(dd_meta.ensure(4 * meta_stride));
        FRAME_CUDA(dd_res.ensure(n * sizeof(cudec_chunk_result)));

        /* One host staging buffer in the same layout, filled section by
         * section, so the upload is a straight copy of it. */
        std::vector<unsigned char> h_meta(4 * meta_stride);
        std::memcpy(h_meta.data(), h_src.data(), meta_stride);
        std::memcpy(h_meta.data() + meta_stride, h_ssz.data(), meta_stride);
        std::memcpy(h_meta.data() + 2 * meta_stride, h_dst.data(),
                    meta_stride);
        std::memcpy(h_meta.data() + 3 * meta_stride, h_dcp.data(),
                    meta_stride);
        FRAME_CUDA(cudaMemcpy(dd_meta.p, h_meta.data(), 4 * meta_stride,
                              cudaMemcpyHostToDevice));

        unsigned char* const d_meta = static_cast<unsigned char*>(dd_meta.p);
        const cudec_status launched = cudec_lz4_decompress_batch(
            reinterpret_cast<const void* const*>(d_meta),
            reinterpret_cast<const size_t*>(d_meta + meta_stride),
            reinterpret_cast<void* const*>(d_meta + 2 * meta_stride),
            reinterpret_cast<const size_t*>(d_meta + 3 * meta_stride), n,
            static_cast<cudec_chunk_result*>(dd_res.p), nullptr);
        if (launched != CUDEC_OK) {
            return launched;
        }
        FRAME_CUDA(cudaDeviceSynchronize());

        std::vector<cudec_chunk_result> res(n);
        FRAME_CUDA(cudaMemcpy(res.data(), dd_res.p,
                              n * sizeof(cudec_chunk_result),
                              cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < n; k++) {
            /* Any per-block failure means the frame's block data is corrupt.
             * The per-block dst capacity is our internal block_max, never the
             * caller's dst, so a block-level OUTPUT_TOO_SMALL is an over-long
             * corrupt block - map every non-OK block to CORRUPT_INPUT rather
             * than leaking the internal capacity as the caller's status. */
            if (res[k].status != CUDEC_OK ||
                res[k].bytes_written > block_max) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            decoded_len[cidx[k]] = res[k].bytes_written;
        }
    }

    /* Compute the assembled layout and total size, and check the caller's
     * capacity BEFORE writing a single byte - so a too-small buffer yields
     * OUTPUT_TOO_SMALL with no partial output left in `out`. */
    size_t total = 0;
    for (size_t i = 0; i < blocks.size(); i++) {
        const size_t len =
            blocks[i].uncompressed ? blocks[i].src_len : decoded_len[i];
        if (len > dst_capacity - total) {
            return CUDEC_ERR_OUTPUT_TOO_SMALL;
        }
        total += len;
    }

    /* Place each block at its prefix offset: uncompressed blocks copied from
     * the frame, compressed blocks copied from the device.
     *
     * Each compressed block used to cost its own blocking cudaMemcpy, which is
     * one host stall per block; the block-count sweep in docs/BENCHMARKS.md is
     * what made that visible, at roughly 34 microseconds a block. The copies
     * are now issued on one stream and drained once, and consecutive blocks
     * that are contiguous on BOTH sides are merged into a single copy first,
     * so the submission count follows the frame's layout instead of its block
     * count.
     *
     * The merge is what makes this pay, and what makes it pay is a property of
     * the format rather than of the corpus: every data block of a frame except
     * the last decodes to exactly block_max, which is the device stride, so a
     * run of full blocks is contiguous in device memory; and compressed blocks
     * land at consecutive output offsets wherever no uncompressed block
     * separates them. Both are tested per block below rather than assumed, so
     * a frame that breaks either just gets more copies, never wrong bytes.
     *
     * Every destination range written here is disjoint from every other one -
     * they are prefix offsets into `out` - so the copies may complete in any
     * order, and the interleaved host memcpy for an uncompressed block touches
     * bytes no copy is targeting. That disjointness is what makes the order
     * free; nothing here depends on a D2H into pageable memory happening to be
     * synchronous.
     *
     * The drain has to happen before d_dst is freed - an in-flight copy out of
     * freed device memory is a use-after-free that does not fail loudly - so
     * the guard below runs it on EVERY exit path from here on, including the
     * error returns, and the ordinary path checks its status as well. */
    cudec_cuda::StreamOwner asm_stream;
    if (n != 0) {
        FRAME_CUDA(asm_stream.create());
    }
    struct AsmDrain {
        cudaStream_t s;
        ~AsmDrain() {
            if (s != nullptr) {
                (void)cudaStreamSynchronize(s);
            }
        }
    } asm_drain{asm_stream.s};

    /* The open run of merged blocks; run_len == 0 means no run is open.
     * Neither sum below can overflow: a run that started at block k0 holds at
     * most one block_max per block, so run_dev + run_len is at most
     * n * block_max, which the guard above already refused an overflow of; and
     * run_out + run_len is at most `total`, which the capacity walk computed
     * without one. */
    size_t run_dev = 0;
    size_t run_out = 0;
    size_t run_len = 0;
    size_t off = 0;
    size_t k = 0;
    for (size_t i = 0; i < blocks.size(); i++) {
        if (blocks[i].uncompressed) {
            if (blocks[i].src_len != 0) {
                std::memcpy(out + off, frame + blocks[i].src_off,
                            blocks[i].src_len);
            }
            off += blocks[i].src_len;
        } else {
            const size_t len = decoded_len[i];
            const size_t dev = k * block_max;
            if (run_len != 0 && run_dev + run_len == dev &&
                run_out + run_len == off) {
                run_len += len;
            } else {
                if (run_len != 0) {
                    FRAME_CUDA(cudaMemcpyAsync(
                        out + run_out,
                        static_cast<unsigned char*>(d_dst.p) + run_dev, run_len,
                        cudaMemcpyDeviceToHost, asm_stream.s));
                }
                run_dev = dev;
                run_out = off;
                run_len = len;
            }
            off += len;
            k++;
        }
    }
    if (run_len != 0) {
        FRAME_CUDA(cudaMemcpyAsync(
            out + run_out, static_cast<unsigned char*>(d_dst.p) + run_dev,
            run_len, cudaMemcpyDeviceToHost, asm_stream.s));
    }
    if (n != 0) {
        FRAME_CUDA(cudaStreamSynchronize(asm_stream.s));
    }

    *total_out = total;
    return CUDEC_OK;
#undef FRAME_CUDA
}

/* The frame decode proper; wrapped by the extern "C" entry point in a
 * try/catch so a host allocation failure can never cross the C ABI. */
cudec_status DecodeFrame(const unsigned char* f, size_t frame_size,
                         unsigned char* out, size_t dst_capacity,
                         size_t* bytes_written) {
    /* Magic (4) + FLG (1) + BD (1) + HC (1) is the minimum header. */
    if (frame_size < 7 || Read32LE(f) != kLz4FrameMagic) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const unsigned flg = f[4];
    const unsigned bd = f[5];
    if (((flg >> 6) & 3) != 1 || (flg & 2) != 0) {
        return CUDEC_ERR_CORRUPT_INPUT; /* version / reserved bit */
    }
    const bool block_independent = (flg >> 5) & 1;
    const bool block_checksum = (flg >> 4) & 1;
    const bool content_size = (flg >> 3) & 1;
    const bool content_checksum = (flg >> 2) & 1;
    const bool dict_id = flg & 1;
    if ((bd & 0x8F) != 0) {
        return CUDEC_ERR_CORRUPT_INPUT; /* BD reserved bits */
    }
    const unsigned bmax = (bd >> 4) & 7;
    if (bmax < 4 || bmax > 7) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    /* The frame format's smallest block-max, 64 KiB, doubled twice per BD
     * step. Named for the reason issue #211 gives: digits in an expression
     * cannot be told from a lane count by anything that reads this tree. */
    constexpr size_t kBlockMaxKiB = 64;
    const size_t block_max = (kBlockMaxKiB << 10) << ((bmax - 4) * 2);
    if (!block_independent || dict_id) {
        return CUDEC_ERR_UNSUPPORTED; /* linked blocks / dictionaries */
    }

    /* Header checksum covers FLG..end-of-descriptor. */
    size_t pos = 6;
    uint64_t declared_content_size = 0;
    if (content_size) {
        if (frame_size < pos + 8) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        declared_content_size = Read64LE(f + pos);
        pos += 8;
    }
    if (frame_size < pos + 1) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    const unsigned hc = f[pos];
    if (((cudec_detail::xxhash32(f + 4, pos - 4) >> 8) & 0xFF) != hc) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }
    pos += 1;

    std::vector<FrameBlock> blocks;
    /* Fuel: every step consumes at least the 4 block-size bytes, so a frame
     * of frame_size bytes admits at most frame_size / 4 + 1 steps - a budget
     * no frame the guards below admit can reach. It makes a future bounds
     * bug a rejected frame instead of a spinning host thread. */
    uint64_t fuel = frame_size / 4 + 1;
    bool end_mark = false;
    while (fuel-- != 0) {
        if (frame_size < pos + 4) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        const uint32_t bs = Read32LE(f + pos);
        pos += 4;
        if (bs == 0) {
            end_mark = true;
            break;
        }
        const bool uncompressed = (bs >> 31) & 1;
        const size_t blen = bs & 0x7FFFFFFFu;
        if (blen == 0 || blen > block_max || frame_size < pos + blen) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        if (block_checksum) {
            if (frame_size < pos + blen + 4) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }
            if (cudec_detail::xxhash32(f + pos, blen) !=
                Read32LE(f + pos + blen)) {
                return CUDEC_ERR_CORRUPT_INPUT;
            }
        }
        blocks.push_back({pos, blen, uncompressed});
        pos += blen + (block_checksum ? 4 : 0);
    }
    if (!end_mark) {
        return CUDEC_ERR_CORRUPT_INPUT; /* fuel exhausted: no end mark */
    }

    size_t total = 0;
    const cudec_status decode_status =
        DecodeAndAssemble(f, blocks, block_max, out, dst_capacity, &total);
    if (decode_status != CUDEC_OK) {
        return decode_status;
    }

    /* A declared content size (FLG bit 3) must equal the produced size -
     * liblz4 rejects a frame whose declared size does not match (too large
     * or too small); cudec rejects it too (oracle parity). */
    if (content_size && declared_content_size != total) {
        return CUDEC_ERR_CORRUPT_INPUT;
    }

    if (content_checksum) {
        if (frame_size < pos + 4) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
        if (cudec_detail::xxhash32(out, total) != Read32LE(f + pos)) {
            return CUDEC_ERR_CORRUPT_INPUT;
        }
    }

    *bytes_written = total;
    return CUDEC_OK;
}

}  // namespace

cudec_status cudec_lz4f_decompress(const void* frame_v, size_t frame_size,
                                   void* dst_v, size_t dst_capacity,
                                   size_t* bytes_written) {
    if (bytes_written != nullptr) {
        *bytes_written = 0;
    }
    if (frame_v == nullptr || bytes_written == nullptr ||
        (dst_v == nullptr && dst_capacity != 0)) {
        return CUDEC_ERR_INVALID_ARGUMENT;
    }
    try {
        return DecodeFrame(static_cast<const unsigned char*>(frame_v),
                           frame_size, static_cast<unsigned char*>(dst_v),
                           dst_capacity, bytes_written);
    } catch (...) {
        /* A host allocation failed (e.g. std::bad_alloc driven by a hostile
         * block count). Never let an exception cross the C ABI: report a
         * defined resource error, no output presented as success. */
        *bytes_written = 0;
        return CUDEC_ERR_CUDA;
    }
}
