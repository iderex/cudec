/* Decode a .lz4 frame on the GPU with cudec.
 *
 * Builds against the installed public header and libc, and nothing else: no
 * test helper, no CUDA header, no cudec source. If this file stops compiling
 * or stops linking, the public surface stopped being enough to write a
 * consumer with, which is the property CI reads it for (issue #158).
 *
 *   decode_frame <input.lz4> <output>
 *
 * The frame must be block-INDEPENDENT (LZ4F_blockIndependent); the header
 * says what else is unsupported and what each error means.
 */

#include <cudec.h>

#include <stdio.h>
#include <stdlib.h>

static const char* status_text(cudec_status status) {
    switch (status) {
        case CUDEC_OK:
            return "ok";
        case CUDEC_ERR_INVALID_ARGUMENT:
            return "invalid argument";
        case CUDEC_ERR_CORRUPT_INPUT:
            return "corrupt input: malformed frame or checksum mismatch";
        case CUDEC_ERR_OUTPUT_TOO_SMALL:
            return "output buffer too small";
        case CUDEC_ERR_CUDA:
            return "a CUDA device or host resource failure";
        case CUDEC_ERR_NOT_IMPLEMENTED:
            return "not implemented";
        case CUDEC_ERR_UNSUPPORTED:
            return "valid frame, but not a subset cudec decodes";
    }
    /* No default label, so a status added to the header reds this switch
     * under -Wswitch rather than falling through to a wrong message. */
    return "unknown status";
}

/* Reads the whole file into a malloc'd buffer. Returns NULL and reports why
 * on any failure, including a file that does not fit in memory. */
static unsigned char* read_file(const char* path, size_t* out_size) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "cannot seek %s\n", path);
        fclose(file);
        return NULL;
    }
    long end = ftell(file);
    if (end < 0) {
        fprintf(stderr, "cannot size %s\n", path);
        fclose(file);
        return NULL;
    }
    rewind(file);

    size_t size = (size_t)end;
    /* malloc(0) may return NULL without failing, and an empty file is not a
     * frame anyway; one byte keeps the NULL below unambiguous. */
    unsigned char* buffer = (unsigned char*)malloc(size + 1);
    if (buffer == NULL) {
        fprintf(stderr, "out of memory reading %s\n", path);
        fclose(file);
        return NULL;
    }
    if (size != 0 && fread(buffer, 1, size, file) != size) {
        fprintf(stderr, "short read on %s\n", path);
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *out_size = size;
    return buffer;
}

static int write_file(const char* path, const unsigned char* data,
                      size_t size) {
    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s for writing\n", path);
        return 0;
    }
    if (size != 0 && fwrite(data, 1, size, file) != size) {
        fprintf(stderr, "short write on %s\n", path);
        fclose(file);
        return 0;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "cannot close %s\n", path);
        return 0;
    }
    return 1;
}

/* A frame carries no size the caller can trust, so the output buffer is
 * grown against the decoder's own answer: CUDEC_ERR_OUTPUT_TOO_SMALL is a
 * defined status with no partial output behind it, which makes retrying with
 * a larger buffer safe and makes the growth loop the shortest honest way to
 * size the destination. The cap is what keeps a hostile frame from walking
 * this consumer up to the machine's memory limit. */
#define DECODE_CAPACITY_CAP ((size_t)1 << 30)

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.lz4> <output>\n", argv[0]);
        return 2;
    }

    size_t frame_size = 0;
    unsigned char* frame = read_file(argv[1], &frame_size);
    if (frame == NULL) {
        return 1;
    }

    size_t capacity = 4 * frame_size + 1024;
    if (capacity > DECODE_CAPACITY_CAP) {
        capacity = DECODE_CAPACITY_CAP;
    }

    unsigned char* out = NULL;
    size_t written = 0;
    cudec_status status = CUDEC_ERR_OUTPUT_TOO_SMALL;
    while (status == CUDEC_ERR_OUTPUT_TOO_SMALL) {
        free(out);
        out = (unsigned char*)malloc(capacity);
        if (out == NULL) {
            fprintf(stderr, "out of memory: %zu bytes\n", capacity);
            free(frame);
            return 1;
        }
        status = cudec_lz4f_decompress(frame, frame_size, out, capacity,
                                       &written);
        if (status != CUDEC_ERR_OUTPUT_TOO_SMALL) {
            break;
        }
        if (capacity >= DECODE_CAPACITY_CAP) {
            fprintf(stderr, "decoded output exceeds the %zu byte cap this "
                            "example allows\n",
                    (size_t)DECODE_CAPACITY_CAP);
            free(out);
            free(frame);
            return 1;
        }
        capacity = capacity > DECODE_CAPACITY_CAP / 2 ? DECODE_CAPACITY_CAP
                                                      : capacity * 2;
    }
    free(frame);

    if (status != CUDEC_OK) {
        fprintf(stderr, "%s: %s (cudec_status %d)\n", argv[1],
                status_text(status), (int)status);
        free(out);
        return 1;
    }

    if (!write_file(argv[2], out, written)) {
        free(out);
        return 1;
    }
    free(out);

    printf("%s -> %s, %zu bytes\n", argv[1], argv[2], written);
    return 0;
}
