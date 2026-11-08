/**
 * @file
 * A writer for the `.tdump` format shared with the Python reference, float32 records only.
 */

#ifndef TDUMP_H
#define TDUMP_H

// Built-in
#include <stdint.h>
#include <stdio.h>

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// An open `.tdump` file.
struct tdump {
    FILE *file; ///< The underlying stream.
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Opens a file for writing and writes the header.
 *
 * @param dump The writer to set up.
 * @param path The file to write. It is truncated if it exists.
 * @return 0 on success, nonzero on failure.
 */
int tdump_open(struct tdump *dump, const char *path);

/**
 * Appends one float32 record.
 *
 * @param dump An open writer.
 * @param name The record name, at most 65535 bytes.
 * @param data The values, row-major.
 * @param ndim The number of dimensions, at most 255.
 * @param dims The size of each dimension.
 * @return 0 on success, nonzero on failure.
 */
int tdump_write(
    struct tdump *dump,
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims
);

/**
 * Flushes and closes the file.
 *
 * @param dump An open writer.
 * @return 0 on success, nonzero if any write failed.
 */
int tdump_close(struct tdump *dump);

#endif
