/**
 * @file
 * A reader for the `.rows` series format that `prepare-rows` writes.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal
#include "io/rows.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The format version the reader accepts.
#define VERSION 1u
/// The bytes every file starts with.
#define MAGIC "ROWS"
/// The length of `MAGIC`, without the terminator.
#define MAGIC_SIZE 4
/// The bytes in one stored integer.
#define UINT32_SIZE 4

// Both targets are little-endian, like the format, so doubles are read as they sit in memory
_Static_assert(
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
    "rows reads float data in native byte order"
);

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Reads an unsigned 32-bit integer in little-endian byte order.
 *
 * @param file The stream.
 * @param value Receives the integer.
 * @return 0 on success, nonzero on a short read.
 */
static int read_uint32(FILE *file, uint32_t *value) {
    unsigned char bytes[UINT32_SIZE];
    if (fread(bytes, 1, UINT32_SIZE, file) != UINT32_SIZE) {
        return 1;
    }
    *value = 0;
    for (size_t i = 0; i < UINT32_SIZE; i++) {
        *value |= (uint32_t)bytes[i] << (8 * i);
    }
    return 0;
}

/**
 * Reads the magic, version, channel and row counts.
 *
 * @param reader The reader whose file is open and positioned at the start.
 * @return 0 on success, nonzero if the header is short or not a version 1 `.rows` file.
 */
static int read_header(struct rows_reader *reader) {
    char magic[MAGIC_SIZE];
    uint32_t version;
    if (fread(magic, 1, MAGIC_SIZE, reader->file) != MAGIC_SIZE ||
        memcmp(magic, MAGIC, MAGIC_SIZE) != 0 ||
        read_uint32(reader->file, &version) || version != VERSION ||
        read_uint32(reader->file, &reader->channels) ||
        read_uint32(reader->file, &reader->rows)) {
        return 1;
    }
    return reader->channels == 0;
}

int rows_open(struct rows_reader *reader, const char *path) {
    reader->file = fopen(path, "rb");
    reader->mean = NULL;
    reader->std = NULL;
    reader->next = 0;
    if (reader->file == NULL) {
        return 1;
    }

    // Step 1: the header
    if (read_header(reader)) {
        rows_close(reader);
        return 1;
    }

    // Step 2: mean and std share one allocation, mean first
    size_t channels = reader->channels;
    reader->mean = malloc(2 * channels * sizeof(double));
    if (reader->mean == NULL) {
        rows_close(reader);
        return 1;
    }
    reader->std = reader->mean + channels;
    if (fread(reader->mean, sizeof(double), channels, reader->file) !=
            channels ||
        fread(reader->std, sizeof(double), channels, reader->file) !=
            channels) {
        rows_close(reader);
        return 1;
    }
    return 0;
}

int rows_next(struct rows_reader *reader, double *row) {
    if (reader->next >= reader->rows) {
        return 1;
    }
    size_t channels = reader->channels;
    if (fread(row, sizeof(double), channels, reader->file) != channels) {
        return 1;
    }
    reader->next++;
    return 0;
}

void rows_close(struct rows_reader *reader) {
    if (reader->file != NULL) {
        fclose(reader->file);
        reader->file = NULL;
    }
    // `std` points into the same allocation as `mean`
    free(reader->mean);
    reader->mean = NULL;
    reader->std = NULL;
}
