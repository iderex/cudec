/* Shared host-side device-runtime RAII owners and the success-check macro for
 * the host-orchestration translation units (frame.cpp, stream.cpp, and the
 * Snappy/GDeflate host paths to come). INTERNAL - device-independent host glue,
 * never part of the public C ABI in include/cudec.h.
 *
 * Each owner owns exactly one device-runtime handle, is non-copyable, and
 * frees it on EVERY scope exit - including the hostile-input reject paths,
 * not just on success. A leak on the expected corrupt-input path is a
 * device/pinned-memory exhaustion DoS in a library entry point. Cleanup
 * errors are discarded deliberately: they are not actionable and must not
 * mask the real decode status. On an allocation failure the owner holds
 * nothing (null, cap 0), so the destructor never double-frees; the enclosing
 * caller decides whether the failure poisons a reusable context.
 *
 * The member types have external linkage (this is a named namespace, not an
 * anonymous one) so an ABI-visible struct built from them - cudec_stream_ctx -
 * triggers no subobject-linkage diagnostic. */
#ifndef CUDEC_VENDOR_RAII_H
#define CUDEC_VENDOR_RAII_H

#include "vendor_rt.h"

#include <cstddef>

/* Evaluate a device-runtime `call` exactly once; if it did not return
 * cudec_rt::success, run `on_failure`. `on_failure` maps the fault to a defined
 * cudec_status return (and, for the reusable stream context, poisons it first
 * so only destruction is valid afterwards). The single expansion is why `call`
 * is never evaluated twice - the failure policy is the caller's, the check is
 * single-sourced here. */
#define CUDEC_RT_CHECK(call, on_failure)   \
    do {                                   \
        if ((call) != cudec_rt::success) { \
            on_failure;                    \
        }                                  \
    } while (0)

namespace cudec_rt {

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
            (void)cudec_rt::device_free(p);
        }
    }
    cudec_rt::error_t ensure(size_t bytes) {
        if (bytes <= cap) {
            return cudec_rt::success;
        }
        if (p != nullptr) {
            (void)cudec_rt::device_free(p);
            p = nullptr;
            cap = 0;
        }
        const cudec_rt::error_t e = cudec_rt::device_malloc(&p, bytes);
        if (e != cudec_rt::success) {
            p = nullptr;
            return e;
        }
        cap = bytes;
        return cudec_rt::success;
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
            (void)cudec_rt::host_free(p);
        }
    }
    cudec_rt::error_t ensure(size_t bytes) {
        if (bytes <= cap) {
            return cudec_rt::success;
        }
        if (p != nullptr) {
            (void)cudec_rt::host_free(p);
            p = nullptr;
            cap = 0;
        }
        const cudec_rt::error_t e = cudec_rt::host_alloc(&p, bytes);
        if (e != cudec_rt::success) {
            p = nullptr;
            return e;
        }
        cap = bytes;
        return cudec_rt::success;
    }
};

struct StreamOwner {
    cudec_rt::stream_t s = nullptr;
    StreamOwner() = default;
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    ~StreamOwner() {
        if (s != nullptr) {
            (void)cudec_rt::stream_destroy(s);
        }
    }
    cudec_rt::error_t create() {
        /* Non-blocking so a wave does not depend on the legacy default stream
         * staying idle (a caller with default-stream work in the same context
         * would otherwise serialize every wave). */
        return cudec_rt::stream_create_nonblocking(&s);
    }
};

struct EventOwner {
    cudec_rt::event_t e = nullptr;
    EventOwner() = default;
    EventOwner(const EventOwner&) = delete;
    EventOwner& operator=(const EventOwner&) = delete;
    ~EventOwner() {
        if (e != nullptr) {
            (void)cudec_rt::event_destroy(e);
        }
    }
    cudec_rt::error_t create() { return cudec_rt::event_create(&e); }
};

}  // namespace cudec_rt

#endif  // CUDEC_VENDOR_RAII_H
