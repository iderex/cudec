/* The vendor runtime seam: the ONE translation unit in src/ that names a
 * backend runtime. Every other file under src/ reaches the device runtime
 * through cudec_rt:: and never spells cudaXxx or hipXxx, so the host
 * orchestration (frame.cpp, stream.cpp, the RAII owners in vendor_raii.h) is
 * one set of sources compiled for either backend - masterplan section 15.
 *
 * WHY A MAPPING TABLE AND NOT A SET OF #defines OVER THE CALL SITES. The
 * llama.cpp vendors/hip.h pattern redefines the CUDA names to the HIP ones and
 * leaves the callers spelling cudaMalloc on both backends. That reads as CUDA
 * source that secretly is not, and it makes "which calls does the port owe?"
 * unanswerable without compiling. Here the two arms are two columns of the
 * same table: what the port owes is the length of the table, a reader compares
 * the columns by eye, and a configure-time rule (tests/CMakeLists.txt, issue
 * #240) reds the build when one column grows an entry the other does not.
 *
 * WHAT THIS HEADER DOES NOT CLAIM, AND THE COMPILER HALF HAS EXPIRED. It said
 * no hipcc had ever compiled the HIP arm, because no ROCm toolchain exists on
 * the machine it landed from. A hipcc compiles this arm on every pull request
 * now: the CI hip job configures CUDEC_ENABLE_HIP in a digest-pinned ROCm
 * image and builds the library's translation units, which include this header
 * (issue #210). So a name this table spells wrongly enough to break
 * compilation reds there, and the absence of a toolchain on any one machine is
 * no longer what decides it.
 *
 * WHAT THE COMPILER DID NOT DECIDE IS THE HALF THAT MATTERS. The mapping is
 * proven CONSISTENT - both arms define the same operations, the CUDA arm
 * builds and passes the whole gate, and a configure-time rule reds when one
 * column grows an entry the other does not - and is still NOT proven CORRECT:
 * a name mapped to the WRONG HIP entry point of the right shape compiles clean
 * on both backends and reds only where the call runs. No AMD device has ever
 * executed one line of it. That answer comes from hardware, which is issue
 * #415 and the runbook at docs/AMD-VALIDATION.md; a compiler was never going
 * to give it.
 *
 * INTERNAL - never part of the public C ABI in include/cudec.h. The public
 * status CUDEC_ERR_CUDA keeps its name on both backends: it is an ABI
 * constant a caller compiled against, not a spelling this seam may move. */
#ifndef CUDEC_VENDOR_RT_H
#define CUDEC_VENDOR_RT_H

#include "cudec.h"

#include <cstddef>

/* __HIP_PLATFORM_AMD__ is defined by hipcc when it targets AMD, and by nothing
 * else - so the CUDA arm is the default and a host compiler that knows about
 * neither backend still lands on a real runtime header rather than on an empty
 * shim. Fail-closed: there is no third arm that stubs the calls out. */
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>

#define CUDEC_RT_ERROR_T hipError_t
#define CUDEC_RT_STREAM_T hipStream_t
#define CUDEC_RT_EVENT_T hipEvent_t
#define CUDEC_RT_MEMCPY_KIND_T hipMemcpyKind
#define CUDEC_RT_SUCCESS hipSuccess
#define CUDEC_RT_MEMCPY_H2D hipMemcpyHostToDevice
#define CUDEC_RT_MEMCPY_D2H hipMemcpyDeviceToHost
#define CUDEC_RT_STREAM_NONBLOCKING hipStreamNonBlocking
#define CUDEC_RT_HOST_ALLOC_DEFAULT hipHostMallocDefault
#define CUDEC_RT_MALLOC hipMalloc
#define CUDEC_RT_FREE hipFree
#define CUDEC_RT_HOST_ALLOC hipHostMalloc
#define CUDEC_RT_HOST_FREE hipHostFree
#define CUDEC_RT_MEMCPY hipMemcpy
#define CUDEC_RT_MEMCPY_ASYNC hipMemcpyAsync
#define CUDEC_RT_STREAM_CREATE_WITH_FLAGS hipStreamCreateWithFlags
#define CUDEC_RT_STREAM_DESTROY hipStreamDestroy
#define CUDEC_RT_STREAM_SYNCHRONIZE hipStreamSynchronize
#define CUDEC_RT_EVENT_CREATE hipEventCreate
#define CUDEC_RT_EVENT_DESTROY hipEventDestroy
#define CUDEC_RT_EVENT_RECORD hipEventRecord
#define CUDEC_RT_EVENT_SYNCHRONIZE hipEventSynchronize
#define CUDEC_RT_DEVICE_SYNCHRONIZE hipDeviceSynchronize
#define CUDEC_RT_GET_LAST_ERROR hipGetLastError
#define CUDEC_RT_GET_DEVICE hipGetDevice
#define CUDEC_RT_DEVICE_GET_ATTRIBUTE hipDeviceGetAttribute
#define CUDEC_RT_ATTR_WAVE_WIDTH hipDeviceAttributeWarpSize
#define CUDEC_RT_FIXED_WAVE_WIDTH 0
/* The mask that names every lane of a wave, in the width the backend's
 * collectives take it: sixty-four bits here, where a wave may be sixty-four
 * lanes wide. A kernel names its participants with this and never with a
 * literal, so one spelling is right on both backends. */
#define CUDEC_RT_WAVE_FULL_MASK 0xffffffffffffffffull

#else
#include <cuda_runtime.h>

#define CUDEC_RT_ERROR_T cudaError_t
#define CUDEC_RT_STREAM_T cudaStream_t
#define CUDEC_RT_EVENT_T cudaEvent_t
#define CUDEC_RT_MEMCPY_KIND_T cudaMemcpyKind
#define CUDEC_RT_SUCCESS cudaSuccess
#define CUDEC_RT_MEMCPY_H2D cudaMemcpyHostToDevice
#define CUDEC_RT_MEMCPY_D2H cudaMemcpyDeviceToHost
#define CUDEC_RT_STREAM_NONBLOCKING cudaStreamNonBlocking
#define CUDEC_RT_HOST_ALLOC_DEFAULT cudaHostAllocDefault
#define CUDEC_RT_MALLOC cudaMalloc
#define CUDEC_RT_FREE cudaFree
#define CUDEC_RT_HOST_ALLOC cudaHostAlloc
#define CUDEC_RT_HOST_FREE cudaFreeHost
#define CUDEC_RT_MEMCPY cudaMemcpy
#define CUDEC_RT_MEMCPY_ASYNC cudaMemcpyAsync
#define CUDEC_RT_STREAM_CREATE_WITH_FLAGS cudaStreamCreateWithFlags
#define CUDEC_RT_STREAM_DESTROY cudaStreamDestroy
#define CUDEC_RT_STREAM_SYNCHRONIZE cudaStreamSynchronize
#define CUDEC_RT_EVENT_CREATE cudaEventCreate
#define CUDEC_RT_EVENT_DESTROY cudaEventDestroy
#define CUDEC_RT_EVENT_RECORD cudaEventRecord
#define CUDEC_RT_EVENT_SYNCHRONIZE cudaEventSynchronize
#define CUDEC_RT_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define CUDEC_RT_GET_LAST_ERROR cudaGetLastError
#define CUDEC_RT_GET_DEVICE cudaGetDevice
#define CUDEC_RT_DEVICE_GET_ATTRIBUTE cudaDeviceGetAttribute
#define CUDEC_RT_ATTR_WAVE_WIDTH cudaDevAttrWarpSize
#define CUDEC_RT_FIXED_WAVE_WIDTH 32
#define CUDEC_RT_WAVE_FULL_MASK 0xffffffffu

#endif

namespace cudec_rt {

using error_t = CUDEC_RT_ERROR_T;
using stream_t = CUDEC_RT_STREAM_T;
using event_t = CUDEC_RT_EVENT_T;
using memcpy_kind_t = CUDEC_RT_MEMCPY_KIND_T;

inline constexpr error_t success = CUDEC_RT_SUCCESS;
inline constexpr memcpy_kind_t memcpy_h2d = CUDEC_RT_MEMCPY_H2D;
inline constexpr memcpy_kind_t memcpy_d2h = CUDEC_RT_MEMCPY_D2H;

/* One thin inline per operation, so the seam costs nothing at run time and a
 * call site cannot pass a flag the port has not mapped. The two allocation
 * pairs hide their flag argument on purpose: the pinned flag is the only one
 * this library ever wants, and the backends spell it differently. */
inline error_t device_malloc(void** p, size_t bytes) {
    return CUDEC_RT_MALLOC(p, bytes);
}
inline error_t device_free(void* p) { return CUDEC_RT_FREE(p); }
inline error_t host_alloc(void** p, size_t bytes) {
    return CUDEC_RT_HOST_ALLOC(p, bytes, CUDEC_RT_HOST_ALLOC_DEFAULT);
}
inline error_t host_free(void* p) { return CUDEC_RT_HOST_FREE(p); }
inline error_t memcpy(void* dst, const void* src, size_t bytes,
                      memcpy_kind_t kind) {
    return CUDEC_RT_MEMCPY(dst, src, bytes, kind);
}
inline error_t memcpy_async(void* dst, const void* src, size_t bytes,
                            memcpy_kind_t kind, stream_t s) {
    return CUDEC_RT_MEMCPY_ASYNC(dst, src, bytes, kind, s);
}
inline error_t stream_create_nonblocking(stream_t* s) {
    return CUDEC_RT_STREAM_CREATE_WITH_FLAGS(s, CUDEC_RT_STREAM_NONBLOCKING);
}
inline error_t stream_destroy(stream_t s) {
    return CUDEC_RT_STREAM_DESTROY(s);
}
inline error_t stream_synchronize(stream_t s) {
    return CUDEC_RT_STREAM_SYNCHRONIZE(s);
}
inline error_t event_create(event_t* e) { return CUDEC_RT_EVENT_CREATE(e); }
inline error_t event_destroy(event_t e) { return CUDEC_RT_EVENT_DESTROY(e); }
inline error_t event_record(event_t e, stream_t s) {
    return CUDEC_RT_EVENT_RECORD(e, s);
}
inline error_t event_synchronize(event_t e) {
    return CUDEC_RT_EVENT_SYNCHRONIZE(e);
}
inline error_t device_synchronize() { return CUDEC_RT_DEVICE_SYNCHRONIZE(); }
inline error_t get_last_error() { return CUDEC_RT_GET_LAST_ERROR(); }

/* THE ONE PLACE THE PUBLIC STREAM HANDLE MEETS THE RUNTIME'S (masterplan
 * section 15.6). cudec_stream_t is the CUDA driver's stream pointer on both
 * backends, so the public header carries no backend define and a consumer
 * built against it yesterday is not rebuilt for the port; a HIP caller passes
 * its stream through that typedef with a cast, and this is where the pointer
 * gets its runtime type back, once, on the way into a launch. On CUDA the two
 * types are one type and the cast is the identity. */
inline stream_t stream_from_abi(cudec_stream_t s) {
    return reinterpret_cast<stream_t>(s);
}
/* The inverse, for the library's own host paths (frame.cpp, stream.cpp) that
 * hand a stream they created to the batch entry, and for the harness and the
 * bench, which do the same. Both are pointers to the driver's stream object,
 * so nothing is lost in either direction. */
inline cudec_stream_t abi_stream(stream_t s) {
    return reinterpret_cast<cudec_stream_t>(s);
}

/* WHETHER THE BACKEND FIXES THE WAVE WIDTH, and zero where it does not. This
 * is the one difference between the two backends that reaches past the mapping
 * table, so it is written here as a table entry rather than as a preprocessor
 * branch at the dispatch. Every CUDA device that exists reports 32; on HIP the
 * width is 32 on RDNA and 64 on CDNA and only the device knows which. */
inline constexpr int fixed_wave_width = CUDEC_RT_FIXED_WAVE_WIDTH;

/* The one composite in this file, and the exception the paragraph above is
 * written against. The device's wave width is ONE question that both runtimes
 * answer in two calls, so splitting it across the seam would put the same
 * two-step at every call site and give each of them its own chance to skip the
 * error check on the first. The attribute is read rather than the whole
 * property struct: it names the same field of the same runtime, it costs no
 * kilobyte-sized copy, and it is per-device, so a caller that changed device
 * between two submissions gets its own device's width instead of a cached
 * process-wide one. On failure *out is left alone - the caller has a status to
 * act on and must not read a width nobody wrote (issue #242). */
inline error_t device_wave_width(int* out) {
    int device = 0;
    const error_t e = CUDEC_RT_GET_DEVICE(&device);
    if (e != success) {
        return e;
    }
    return CUDEC_RT_DEVICE_GET_ATTRIBUTE(out, CUDEC_RT_ATTR_WAVE_WIDTH, device);
}

/* The width a launch is built for, which is what the dispatch asks. On a
 * backend that fixes the width this is a constant and the runtime is never
 * touched - so a submission on such a backend gains no call that can fail, and
 * the GPU-less launch-failure test keeps reaching the launch it is about. On a
 * backend that does not, the query is the answer and its failure is the
 * caller's, never a default. */
inline error_t wave_width_for_launch(int* out) {
    if (fixed_wave_width != 0) {
        *out = fixed_wave_width;
        return success;
    }
    return device_wave_width(out);
}

}  // namespace cudec_rt

#endif  // CUDEC_VENDOR_RT_H
