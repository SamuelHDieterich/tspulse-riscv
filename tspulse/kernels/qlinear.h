/**
 * @file
 * PyTorch's Linear on int8 weights with one float32 scale per output row, for the mixers of a `QUANT` build.
 *
 * Each `QUANT` value compiles only its own file in `kernels/qlinear/`.
 */

#ifndef QLINEAR_H
#define QLINEAR_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Applies a Linear layer with int8 weights to each row of `x`, in float32 math.
 *
 * Every weight is turned back into float as `scale[o] * (float)w[o][i]` before it is used, so the result equals `linear` on those float weights, bit for bit.
 *
 * @param x The input, `rows` rows of `in` values.
 * @param w The int8 weights, `out` rows of `in` values, as PyTorch stores them.
 * @param scale The scale of each weight row, `out` values.
 * @param b The bias, `out` values.
 * @param y Receives `rows` rows of `out` values. It must not overlap `x`.
 * @param rows Rows in `x`.
 * @param in Values per input row.
 * @param out Values per output row.
 */
void linear_w8a32(
    const float *restrict x,
    const int8_t *restrict w,
    const float *restrict scale,
    const float *restrict b,
    float *restrict y,
    int rows,
    int in,
    int out
);

#endif
