/* Decode a batch of independent LZ4 blocks on the GPU with cudec.
 *
 * Builds against the public header and the CUDA runtime, and nothing else:
 * no test helper, no cudec source, no fixture header. If this file stops
 * compiling or stops linking, the public surface stopped being enough to
 * drive the batch entry point with, which is the property CI reads it for
 * (issue #159).
 *
 *   example_decode_batch
 *
 * It takes no arguments. The corpus is four blocks built into this file,
 * one of them deliberately corrupt, so the per-chunk fail-closed contract
 * is shown running rather than asserted in a comment: the corrupt chunk
 * reports a defined error with bytes_written == 0 while its neighbours
 * decode normally. Chunks are independent - one bad chunk does not spoil
 * the batch and does not fail the call.
 *
 * The parts a first-time caller gets wrong, all of them visible below:
 *
 *   - every array argument is DEVICE memory holding chunk_count entries,
 *     and the pointers inside the source and destination arrays are
 *     themselves device pointers, so the host builds pointer tables and
 *     uploads them;
 *   - the results array must be 16-byte aligned, which any cudaMalloc
 *     allocation is;
 *   - the call is ASYNCHRONOUS on the stream. Its synchronous return value
 *     covers argument validation and launch submission only, so the
 *     results are read back after the stream has been synchronized;
 *   - the outcome of each chunk lives in its own cudec_chunk_result, not
 *     in the call's return value.
 */

#include <cudec.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

#define CHUNK_COUNT 4

/* One hand-written LZ4 block, laid out field by field so the bytes can be
 * read against the format rather than trusted. It carries a literal run, an
 * overlapping match (offset 8, length 16, so the match reads bytes the same
 * match is writing), and the literals-only tail every LZ4 block ends with.
 *
 *   0x8C   token: 8 literals (high nibble), match length 12 + 4 = 16
 *   ...    the 8 literal bytes
 *   08 00  match offset 8, little endian
 *   0xE0   token: 14 literals, no match - the terminal run
 *   ...    the 14 literal bytes
 */
static const unsigned char kBlock[] = {
    0x8C, 'G',  'P',  'U',  ' ', 'L', 'Z', '4', ' ',  0x08,
    0x00, 0xE0, 'b',  'a',  't', 'c', 'h', ' ', 'd',  'e',
    'c',  'o',  'd',  'e',  '.', '\n'};

/* The same block with the match offset changed from 8 to 255, which reaches
 * further back than the 8 bytes produced so far. That is a defined reject
 * (a match may never read before the start of its own output), not a
 * crash and not a guess. */
static const unsigned char kCorruptBlock[] = {
    0x8C, 'G',  'P',  'U',  ' ', 'L', 'Z', '4', ' ',  0xFF,
    0x00, 0xE0, 'b',  'a',  't', 'c', 'h', ' ', 'd',  'e',
    'c',  'o',  'd',  'e',  '.', '\n'};

/* What kBlock decodes to: the 8 literals, then the same 8 bytes twice more
 * out of the overlapping match, then the tail. */
static const char kExpected[] = "GPU LZ4 GPU LZ4 GPU LZ4 batch decode.\n";
#define DECODED_SIZE (sizeof kExpected - 1)

static const char* status_text(cudec_status status) {
    switch (status) {
        case CUDEC_OK:
            return "ok";
        case CUDEC_ERR_INVALID_ARGUMENT:
            return "invalid argument";
        case CUDEC_ERR_CORRUPT_INPUT:
            return "corrupt input: malformed block";
        case CUDEC_ERR_OUTPUT_TOO_SMALL:
            return "output buffer too small";
        case CUDEC_ERR_CUDA:
            return "a CUDA device or host resource failure";
        case CUDEC_ERR_NOT_IMPLEMENTED:
            return "not implemented";
        case CUDEC_ERR_UNSUPPORTED:
            return "valid input, but not a subset cudec decodes";
    }
    /* No default label, so a status added to the header reds this switch
     * under -Wswitch rather than falling through to a wrong message. */
    return "unknown status";
}

static int cuda_failed(cudaError_t error, const char* what) {
    if (error == cudaSuccess) {
        return 0;
    }
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    return 1;
}

int main(void) {
    /* Every pointer the cleanup path frees is declared and null-initialised
     * up front, so a failure anywhere below can jump straight to it and free
     * exactly the allocations that exist - cudaFree on a null pointer is a
     * no-op. The plain arrays below carry no initialiser because each is
     * filled before it is read. */
    const unsigned char* host_sources[CHUNK_COUNT];
    size_t host_source_sizes[CHUNK_COUNT];
    const void* device_sources[CHUNK_COUNT] = {NULL, NULL, NULL, NULL};
    void* device_destinations[CHUNK_COUNT] = {NULL, NULL, NULL, NULL};
    size_t host_capacities[CHUNK_COUNT];
    cudec_chunk_result results[CHUNK_COUNT];
    unsigned char decoded[DECODED_SIZE];

    const void** d_src_ptrs = NULL;
    size_t* d_src_sizes = NULL;
    void** d_dst_ptrs = NULL;
    size_t* d_dst_capacities = NULL;
    cudec_chunk_result* d_results = NULL;
    cudaStream_t stream = NULL;

    cudec_status status = CUDEC_OK;
    int failures = 0;
    int i = 0;

    /* Three good chunks around one corrupt one: the neighbours are what
     * make per-chunk isolation visible. */
    for (i = 0; i < CHUNK_COUNT; i++) {
        if (i == 1) {
            host_sources[i] = kCorruptBlock;
            host_source_sizes[i] = sizeof kCorruptBlock;
        } else {
            host_sources[i] = kBlock;
            host_source_sizes[i] = sizeof kBlock;
        }
        host_capacities[i] = DECODED_SIZE;
    }

    for (i = 0; i < CHUNK_COUNT; i++) {
        void* src = NULL;
        if (cuda_failed(cudaMalloc(&src, host_source_sizes[i]),
                        "cudaMalloc source")) {
            failures = 1;
            goto cleanup;
        }
        device_sources[i] = src;
        if (cuda_failed(cudaMemcpy(src, host_sources[i], host_source_sizes[i],
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy source")) {
            failures = 1;
            goto cleanup;
        }
        if (cuda_failed(cudaMalloc(&device_destinations[i], DECODED_SIZE),
                        "cudaMalloc destination")) {
            failures = 1;
            goto cleanup;
        }
    }

    /* The four argument arrays and the result array live in device memory,
     * and the source and destination arrays hold device pointers. */
    if (cuda_failed(cudaMalloc((void**)&d_src_ptrs, sizeof device_sources),
                    "cudaMalloc source table") ||
        cuda_failed(cudaMalloc((void**)&d_src_sizes, sizeof host_source_sizes),
                    "cudaMalloc size table") ||
        cuda_failed(
            cudaMalloc((void**)&d_dst_ptrs, sizeof device_destinations),
            "cudaMalloc destination table") ||
        cuda_failed(
            cudaMalloc((void**)&d_dst_capacities, sizeof host_capacities),
            "cudaMalloc capacity table") ||
        cuda_failed(cudaMalloc((void**)&d_results, sizeof results),
                    "cudaMalloc result array")) {
        failures = 1;
        goto cleanup;
    }

    if (cuda_failed(cudaMemcpy(d_src_ptrs, device_sources,
                               sizeof device_sources, cudaMemcpyHostToDevice),
                    "cudaMemcpy source table") ||
        cuda_failed(cudaMemcpy(d_src_sizes, host_source_sizes,
                               sizeof host_source_sizes,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy size table") ||
        cuda_failed(cudaMemcpy(d_dst_ptrs, device_destinations,
                               sizeof device_destinations,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy destination table") ||
        cuda_failed(cudaMemcpy(d_dst_capacities, host_capacities,
                               sizeof host_capacities,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy capacity table")) {
        failures = 1;
        goto cleanup;
    }

    if (cuda_failed(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        failures = 1;
        goto cleanup;
    }

    /* cudec_stream_t and cudaStream_t are the same pointer type, so the
     * stream is passed straight through. */
    status = cudec_lz4_decompress_batch(d_src_ptrs, d_src_sizes, d_dst_ptrs,
                                        d_dst_capacities, CHUNK_COUNT,
                                        d_results, stream);
    /* This status is about the call, never about the data: a corrupt chunk
     * does not fail it. Only argument validation and launch submission are
     * covered here. */
    if (status != CUDEC_OK) {
        fprintf(stderr, "batch submission rejected: %s (cudec_status %d)\n",
                status_text(status), (int)status);
        failures = 1;
        goto cleanup;
    }

    /* Asynchronous: the results are not readable until the stream has run
     * the launch to completion. */
    if (cuda_failed(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        failures = 1;
        goto cleanup;
    }

    if (cuda_failed(cudaMemcpy(results, d_results, sizeof results,
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy results")) {
        failures = 1;
        goto cleanup;
    }

    for (i = 0; i < CHUNK_COUNT; i++) {
        const cudec_status chunk_status = (cudec_status)results[i].status;
        printf("chunk %d: %s (cudec_status %d), bytes_written %llu\n", i,
               status_text(chunk_status), (int)chunk_status,
               (unsigned long long)results[i].bytes_written);

        if (i == 1) {
            /* The corrupt chunk. A defined error and no output presented as
             * valid - that is the whole of the fail-closed contract. */
            if (chunk_status == CUDEC_OK || results[i].bytes_written != 0) {
                fprintf(stderr,
                        "chunk %d was expected to be rejected with zero "
                        "bytes written\n",
                        i);
                failures = 1;
            }
            continue;
        }

        if (chunk_status != CUDEC_OK ||
            results[i].bytes_written != DECODED_SIZE) {
            fprintf(stderr, "chunk %d was expected to decode %llu bytes\n", i,
                    (unsigned long long)DECODED_SIZE);
            failures = 1;
            continue;
        }

        if (cuda_failed(cudaMemcpy(decoded, device_destinations[i],
                                   DECODED_SIZE, cudaMemcpyDeviceToHost),
                        "cudaMemcpy decoded output")) {
            failures = 1;
            continue;
        }
        if (memcmp(decoded, kExpected, DECODED_SIZE) != 0) {
            fprintf(stderr, "chunk %d decoded to unexpected bytes\n", i);
            failures = 1;
        }
    }

cleanup:
    if (stream != NULL) {
        cudaStreamDestroy(stream);
    }
    cudaFree(d_results);
    cudaFree(d_dst_capacities);
    cudaFree(d_dst_ptrs);
    cudaFree(d_src_sizes);
    cudaFree(d_src_ptrs);
    for (i = 0; i < CHUNK_COUNT; i++) {
        cudaFree(device_destinations[i]);
        cudaFree((void*)device_sources[i]);
    }

    if (failures != 0) {
        return 1;
    }
    printf("%d chunks decoded, 1 rejected as expected\n", CHUNK_COUNT - 1);
    return 0;
}
