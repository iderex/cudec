/* cudec - open-source GPU decompression for the standard formats.
 *
 * Public C ABI. Everything in this header is C-compatible; the library
 * never throws across this boundary and never writes output it did not
 * fully validate.
 */
#ifndef CUDEC_H
#define CUDEC_H

#ifdef __cplusplus
extern "C" {
#endif

#define CUDEC_VERSION_MAJOR 0
#define CUDEC_VERSION_MINOR 0
#define CUDEC_VERSION_PATCH 1

/* Returns the runtime library version as (major * 10000 +
 * minor * 100 + patch), for ABI sanity checks against these macros. */
int cudec_version(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDEC_H */
