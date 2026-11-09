/**
 * @file
 * Swaps the two dimensions of a matrix, as `torch.transpose` does on the last two.
 */

#ifndef TRANSPOSE_H
#define TRANSPOSE_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Writes the transpose of `x` into `y`.
 *
 * @param x The input, `rows` rows of `cols` values.
 * @param y Receives `cols` rows of `rows` values. It must not overlap `x`.
 * @param rows Rows in `x`.
 * @param cols Values per row of `x`.
 */
void transpose(const float *restrict x, float *restrict y, int rows, int cols);

#endif
