/**
 * @file
 * A writer for the `.tdump` format shared with the Python reference, float32 records only.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Internal
#include "io/tdump.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

#define VERSION 1u
#define DTYPE_FLOAT32 1u

// Both targets are little-endian, like the format, so float data is written as it sits in memory
_Static_assert(
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
    "tdump writes float data in native byte order"
);

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Writes an unsigned integer in little-endian byte order.
 *
 * @param file The stream.
 * @param value The integer.
 * @param size The width in bytes, at most 4.
 * @return 0 on success, nonzero on failure.
 */
static int write_uint(FILE *file, uint32_t value, size_t size) {
    unsigned char bytes[4];
    for (size_t i = 0; i < size; i++) {
        bytes[i] = (unsigned char)(value >> (8 * i));
    }
    return fwrite(bytes, 1, size, file) != size;
}

int tdump_open(struct tdump *dump, const char *path) {
    dump->file = fopen(path, "wb");
    if (dump->file == NULL) {
        return 1;
    }
    if (fwrite("TDMP", 1, 4, dump->file) != 4 ||
        write_uint(dump->file, VERSION, 4)) {
        fclose(dump->file);
        dump->file = NULL;
        return 1;
    }
    return 0;
}

int tdump_write(
    struct tdump *dump,
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims
) {
    size_t name_size = strlen(name);
    if (name_size > UINT16_MAX || ndim < 0 || ndim > UINT8_MAX) {
        return 1;
    }

    size_t count = 1;
    for (int i = 0; i < ndim; i++) {
        count *= dims[i];
    }

    if (write_uint(dump->file, (uint32_t)name_size, 2) ||
        fwrite(name, 1, name_size, dump->file) != name_size ||
        write_uint(dump->file, DTYPE_FLOAT32, 1) ||
        write_uint(dump->file, (uint32_t)ndim, 1)) {
        return 1;
    }
    for (int i = 0; i < ndim; i++) {
        if (write_uint(dump->file, dims[i], 4)) {
            return 1;
        }
    }
    return fwrite(data, sizeof(float), count, dump->file) != count;
}

int tdump_close(struct tdump *dump) {
    int failed = fflush(dump->file) != 0;
    failed |= fclose(dump->file) != 0;
    dump->file = NULL;
    return failed;
}
