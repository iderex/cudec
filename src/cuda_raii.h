/* Shared host-side CUDA RAII owners and the cudaSuccess-check macro for the
 * CUDA host-orchestration translation units (frame.cpp, stream.cpp, and the
 * Snappy/GDeflate host paths to come). INTERNAL - device-independent host glue,
 * never part of the public C ABI in include/cudec.h.
 *
 * Each owner owns exactly one CUDA handle, is non-copyable, and frees it on
 * EVERY scope exit - including the hostile-input reject paths, not just on
 * success. A leak on the expected corrupt-input path is a device/pinned-memory
 * exhaustion DoS in a library entry point. Cleanup errors are discarded
 * deliberately: they are not actionable and must not mask the real decode
 * status. On an allocation failure the owner holds nothing (null, cap 0), so
 * the destructor never double-frees; the enclosing caller decides whether the
 * failure poisons a reusable context.
 *
 * The member types have external linkage (this is a named namespace, not an
 * anonymous one) so an ABI-visible struct built from them - cudec_stream_ctx -
 * triggers no subobject-linkage diagnostic. */
#ifndef CUDEC_CUDA_RAII_H
#define CUDEC_CUDA_RAII_H

#include <cuda_runtime.h>

#include <cstddef>

/* Evaluate a CUDA runtime `call` exactly once; if it did not return
 * cudaSuccess, run `on_failure`. `on_failure` maps the fault to a defined
 * cudec_status return (and, for the reusable stream context, poisons it first
 * so only destruction is valid afterwards). The single expansion is why `call`
 * is never evaluated twice - the failure policy is the caller's, the check is
 * single-sourced here. */
#define CUDEC_CUDA_CHECK(call, on_failure) \
    do {                                   \
        if ((call) != cudaSuccess) {       \
            on_failure;                    \
        }                                  \
    } while (0)

namespace cudec_cuda {

/* Grow-only device buffer. ensure() reuses the existing allocation when it is
 * already large enough (the amortization); otherwise it frees the old one and
 * allocates the larger size. On failure the owner holds nothing (null, cap 0),
 * so the destructor never double-frees - the enclosing context is poisoned by
 * the caller and only its destruction is valid afterwards. All grows happen
 * before any staging in a decode, so no live buffer is ever reallocated
 * mid-call. A one-shot caller simply calls ensure() once on a fresh owner: the
 * first ensure of a non-zero size allocates exactly once. */
struct DevBuf {
    void* p = nullptr;
    size_t cap = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() {
        if (p != nullptr) {
            (void)cudaFree(p);
        }
    }
    cudaError_t ensure(size_t bytes) {
        if (bytes <= cap) {
            return cudaSuccess;
        }
        if (p != nullptr) {
            (void)cudaFree(p);
            p = nullptr;
            cap = 0;
        }
        const cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) {
            p = nullptr;
            return e;
        }
        cap = bytes;
        return cudaSuccess;
    }
};

/* Grow-only pinned host buffer, same contract as DevBuf. */
struct PinnedBuf {
    void* p = nullptr;
    size_t cap = 0;
    PinnedBuf() = default;
    PinnedBuf(const PinnedBuf&) = delete;
    PinnedBuf& operator=(const PinnedBuf&) = delete;
    ~PinnedBuf() {
        if (p != nullptr) {
            (void)cudaFreeHost(p);
        }
    }
    cudaError_t ensure(size_t bytes) {
        if (bytes <= cap) {
            return cudaSuccess;
        }
        if (p != nullptr) {
            (void)cudaFreeHost(p);
            p = nullptr;
            cap = 0;
        }
        const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
        if (e != cudaSuccess) {
            p = nullptr;
            return e;
        }
        cap = bytes;
        return cudaSuccess;
    }
};

struct StreamOwner {
    cudaStream_t s = nullptr;
    StreamOwner() = default;
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    ~StreamOwner() {
        if (s != nullptr) {
            (void)cudaStreamDestroy(s);
        }
    }
    cudaError_t create() {
        /* Non-blocking so a wave does not depend on the legacy default stream
         * staying idle (a caller with default-stream work in the same context
         * would otherwise serialize every wave). */
        return cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    }
};

struct EventOwner {
    cudaEvent_t e = nullptr;
    EventOwner() = default;
    EventOwner(const EventOwner&) = delete;
    EventOwner& operator=(const EventOwner&) = delete;
    ~EventOwner() {
        if (e != nullptr) {
            (void)cudaEventDestroy(e);
        }
    }
    cudaError_t create() { return cudaEventCreate(&e); }
};

}  // namespace cudec_cuda

#endif  // CUDEC_CUDA_RAII_H
