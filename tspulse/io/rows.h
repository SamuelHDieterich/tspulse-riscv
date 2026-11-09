/**
 * @file
 * A reader for the `.rows` series format that `prepare-rows` writes.
 *
 * Little-endian, with no padding: the 4 bytes `ROWS`, then `uint32` version (1), channels C and rows N, then `float64` mean[C], `float64` std[C] and N rows of C `float64` raw values.
 */

#ifndef ROWS_H
#define ROWS_H

// Built-in
#include <stdint.h>
#include <stdio.h>

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// An open `.rows` file, positioned at the next row to read.
struct rows_reader {
    FILE *file;        ///< The underlying stream.
    uint32_t channels; ///< Values per row.
    uint32_t rows;     ///< Rows in the file.
    uint32_t next;     ///< Index of the next row `rows_next` returns.
    double *mean;      ///< Per-channel average to subtract, `channels` values.
    double *std;       ///< Per-channel spread to divide by, `channels` values.
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Opens a file and reads its header, mean and std.
 *
 * @param reader The reader to set up.
 * @param path The file to read.
 * @return 0 on success, nonzero on failure. The reader holds nothing then.
 */
int rows_open(struct rows_reader *reader, const char *path);

/**
 * Reads the next row.
 *
 * @param reader An open reader.
 * @param row Receives `channels` raw values.
 * @return 0 on success, nonzero at the end of the rows or on a short read.
 */
int rows_next(struct rows_reader *reader, double *row);

/**
 * Closes the file and frees the mean and std.
 *
 * @param reader An open reader.
 */
void rows_close(struct rows_reader *reader);

#endif
