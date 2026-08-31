/* The harness's half of the vendor seam: the ONE file under tests/ that names
 * a backend runtime, and the counterpart of src/vendor_rt.h (issue #240).
 *
 * WHY THE HARNESS NEEDS A SECOND TABLE AND NOT A WIDER FIRST ONE. The
 * validation ladder drives the device itself - it allocates, poisons, copies
 * back and asks the driver questions the library never asks - so it reaches
 * for operations the shipped code has no caller for: a device memset, a
 * default-flags stream, a planted device error, the free-memory query, an
 * event created with timing off, and a kernel's own attributes. Putting those
 * into src/vendor_rt.h would grow the internal library header with surface
 * nothing in src/ calls, and the src/-only rule that keeps every other file
 * under src/ off the runtime would then be guarding a table half of which
 * exists for the tests. So the seam is split by who calls it: this file adds
 * what only a test needs, includes src/vendor_rt.h rather than restating one
 * line of it, and is the single exemption of the tests/-side rule that
 * mirrors #240's (issue #243).
 *
 * WHAT THIS FILE DOES NOT CLAIM. No hipcc has compiled the HIP arm below, for
 * the same reason src/vendor_rt.h states about its own: no ROCm toolchain
 * exists on the machine this landed from. The two arms are proven CONSISTENT
 * - the configure-time rule in tests/CMakeLists.txt reds when one arm carries
 * an entry the other does not - and are NOT proven CORRECT. Issue #210 is the
 * route to the compiler that would decide it.
 *
 * INTERNAL to the test harness; nothing here is part of the public C ABI. */
#ifndef CUDEC_TESTS_VENDOR_RT_TEST_H
#define CUDEC_TESTS_VENDOR_RT_TEST_H

#include "vendor_rt.h"

/* The same two-column shape as src/vendor_rt.h, and for the same reason: what
 * the port owes is the length of the table, a reader compares the columns by
 * eye, and the configure-time rule reds a column that grew alone. */
#if defined(__HIP_PLATFORM_AMD__)

#define CUDEC_RT_MEMSET hipMemset
#define CUDEC_RT_STREAM_CREATE hipStreamCreate
#define CUDEC_RT_SET_DEVICE hipSetDevice
#define CUDEC_RT_PEEK_AT_LAST_ERROR hipPeekAtLastError
#define CUDEC_RT_MEM_GET_INFO hipMemGetInfo
#define CUDEC_RT_EVENT_CREATE_WITH_FLAGS hipEventCreateWithFlags
#define CUDEC_RT_EVENT_DISABLE_TIMING hipEventDisableTiming
#define CUDEC_RT_EVENT_QUERY hipEventQuery
#define CUDEC_RT_ERROR_NOT_READY hipErrorNotReady
#define CUDEC_RT_GET_ERROR_STRING hipGetErrorString
#define CUDEC_RT_FUNC_ATTRIBUTES_T hipFuncAttributes
#define CUDEC_RT_FUNC_GET_ATTRIBUTES hipFuncGetAttributes

#else

#define CUDEC_RT_MEMSET cudaMemset
#define CUDEC_RT_STREAM_CREATE cudaStreamCreate
#define CUDEC_RT_SET_DEVICE cudaSetDevice
#define CUDEC_RT_PEEK_AT_LAST_ERROR cudaPeekAtLastError
#define CUDEC_RT_MEM_GET_INFO cudaMemGetInfo
#define CUDEC_RT_EVENT_CREATE_WITH_FLAGS cudaEventCreateWithFlags
#define CUDEC_RT_EVENT_DISABLE_TIMING cudaEventDisableTiming
#define CUDEC_RT_EVENT_QUERY cudaEventQuery
#define CUDEC_RT_ERROR_NOT_READY cudaErrorNotReady
#define CUDEC_RT_GET_ERROR_STRING cudaGetErrorString
#define CUDEC_RT_FUNC_ATTRIBUTES_T cudaFuncAttributes
#define CUDEC_RT_FUNC_GET_ATTRIBUTES cudaFuncGetAttributes

#endif

namespace cudec_rt {

using func_attributes_t = CUDEC_RT_FUNC_ATTRIBUTES_T;

inline constexpr error_t error_not_ready = CUDEC_RT_ERROR_NOT_READY;

/* One thin inline per operation, as in src/vendor_rt.h, so a call site cannot
 * pass a flag the port has not mapped. */
inline error_t device_memset(void* p, int value, size_t bytes) {
    return CUDEC_RT_MEMSET(p, value, bytes);
}

/* The typed allocation the harness writes, and the one place a cast for it
 * lives. The seam's own device_malloc takes void** because the library only
 * ever allocates into a void*; a test allocates into the pointer type it will
 * dereference, and CUDA's cudaMalloc is a template that hid the conversion.
 * Spelling the conversion once here keeps sixty call sites free of a
 * reinterpret_cast each, and the non-template overload above still wins for a
 * void** so this adds no second path for the library's own shape. */
template <class T>
inline error_t device_malloc(T** p, size_t bytes) {
    void* raw = 0;
    const error_t e = device_malloc(&raw, bytes);
    if (e == success) {
        *p = static_cast<T*>(raw);
    }
    return e;
}

/* The DEFAULT-flags stream, which is a different object from the seam's
 * non-blocking one and is deliberately not folded into it: a test that wants
 * the ordering the default stream imposes must not silently get a stream that
 * does not impose it. */
inline error_t stream_create(stream_t* s) { return CUDEC_RT_STREAM_CREATE(s); }

inline error_t set_device(int device) { return CUDEC_RT_SET_DEVICE(device); }
inline error_t peek_at_last_error() { return CUDEC_RT_PEEK_AT_LAST_ERROR(); }
inline error_t mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    return CUDEC_RT_MEM_GET_INFO(free_bytes, total_bytes);
}

/* Timing off, because the only question asked of these events is whether the
 * work behind them has retired; a timing event costs a synchronisation the
 * deadline loop exists to avoid. */
inline error_t event_create_untimed(event_t* e) {
    return CUDEC_RT_EVENT_CREATE_WITH_FLAGS(e, CUDEC_RT_EVENT_DISABLE_TIMING);
}
inline error_t event_query(event_t e) { return CUDEC_RT_EVENT_QUERY(e); }

inline const char* error_string(error_t e) {
    return CUDEC_RT_GET_ERROR_STRING(e);
}

/* The entry is taken as a plain pointer rather than through the CUDA-only
 * function-template overload, because that overload has no HIP counterpart and
 * the pointer form is what both runtimes actually declare. */
inline error_t func_get_attributes(func_attributes_t* attr, const void* entry) {
    return CUDEC_RT_FUNC_GET_ATTRIBUTES(attr, entry);
}

}  // namespace cudec_rt

#endif /* CUDEC_TESTS_VENDOR_RT_TEST_H */
