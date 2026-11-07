/**
 * @file
 * The matrix multiply of a PyTorch Linear layer.
 */

#ifndef MATMUL_H
#define MATMUL_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Computes y = x · wᵀ + b, row by row.
 *
 * All matrices are row-major, and w keeps PyTorch's [out, in] layout. The buffers must not overlap.
 *
 * @param x The input, [rows, in].
 * @param w The weights, [out, in].
 * @param b The bias, [out].
 * @param y The output, [rows, out].
 * @param rows The input rows.
 * @param in The layer's input size.
 * @param out The layer's output size.
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
