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

// Internal
#include "model/sizes.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The largest int8 value used, as the export does: the range is symmetric, so -128 never appears.
#define QUANT_MAX 127

/// The most input rows of a mixer Linear: the feature mixer's 138 patches.
#define QUANT_ROWS_CAPACITY TOTAL_PATCHES

/// The most values in a mixer Linear's input: 24 rows of 276 in the patch mixer, or 138 rows of 48 in the feature mixer.
#define QUANT_INPUT_CAPACITY                                                   \
    (D_MODEL * PATCH_MLP_HIDDEN > TOTAL_PATCHES * FEATURE_MLP_HIDDEN           \
         ? D_MODEL * PATCH_MLP_HIDDEN                                          \
         : TOTAL_PATCHES * FEATURE_MLP_HIDDEN)

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

/**
 * Applies a Linear layer with int8 weights to each row of `x`, with int8 activations and int32 sums.
 *
 * Each input row is quantized by `quantize_rows` with its own scale, and each output is `b[o] + (sx[r] * scale[o]) * (float)acc`, where `acc` is the exact integer sum of the products.
 * The kernel quantizes into its own static scratch, which holds the largest mixer input, 24 rows of 276 or 138 rows of 48: `rows` must be at most `QUANT_ROWS_CAPACITY` (138) and `rows · in` at most `QUANT_INPUT_CAPACITY`, or the call writes past the scratch.
 * The static scratch also means two calls must not run at the same time.
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
void linear_w8a8(
    const float *restrict x,
    const int8_t *restrict w,
    const float *restrict scale,
    const float *restrict b,
    float *restrict y,
    int rows,
    int in,
    int out
);

/**
 * Quantizes each row of `x` to int8 with one scale per row, `max|x| / 127`.
 *
 * A row whose largest magnitude is too small for `127 / max|x|` to be a finite float (below `127 / FLT_MAX`), an all-zero row included, gets all-zero values and a scale of 1.
 * Inputs must be finite: `fmaxf` skips a NaN, and the clamp turns a NaN product into -127.
 *
 * @param x The input, `rows` rows of `in` values.
 * @param qx Receives the int8 values, `rows` rows of `in`.
 * @param sx Receives each row's scale, `rows` values.
 * @param rows Rows in `x`.
 * @param in Values per row.
 */
void quantize_rows(
    const float *restrict x,
    int8_t *restrict qx,
    float *restrict sx,
    int rows,
    int in
);

#endif
