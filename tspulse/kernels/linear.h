/**
 * @file
 * PyTorch's Linear: every row times the transposed weights, plus the bias.
 */

#ifndef LINEAR_H
#define LINEAR_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Applies a Linear layer to each row of `x`.
 *
 * @param x The input, `rows` rows of `in` values.
 * @param w The weights, `out` rows of `in` values, as PyTorch stores them.
 * @param b The bias, `out` values.
 * @param y Receives `rows` rows of `out` values. It must not overlap `x`.
 * @param rows Rows in `x`.
 * @param in Values per input row.
 * @param out Values per output row.
 */
void linear(
    const float *restrict x,
    const float *restrict w,
    const float *restrict b,
    float *restrict y,
    int rows,
    int in,
    int out
);

#endif
