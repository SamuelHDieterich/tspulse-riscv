/**
 * @file
 * PyTorch's Linear on int8 weights and int8 activations, with int32 sums, by the same 4 × 4 blocking as `kernels/linear/blocked.c`.
 *
 * Each input row is quantized at run time with its own scale, `max|x| / 127`, and each weight row carries the scale the export gave it.
 * The products are summed as integers, which is exact, so the order of the sums doesn't change the result, and only the final scaling to float rounds.
 * That is the arithmetic an int8 vector kernel would do, so a vector build that keeps `quantize_rows` and computes the final `b + (sx · s) · acc` as one fused multiply-add gives the same outputs.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>
#include <stdint.h>

// Internal
#include "kernels/qlinear.h"
#include "model/sizes.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The larger of two sizes, for the capacities below.
#define LARGER(a, b) ((a) > (b) ? (a) : (b))

/// The longest mixer input row, over every mixer Linear's input length: the patch mixer's, the feature mixer's and the register mixer's.
#define IN_CAPACITY                                                            \
    LARGER(                                                                    \
        LARGER(TOTAL_PATCHES, PATCH_MLP_HIDDEN),                               \
        LARGER(                                                                \
            LARGER(D_MODEL, FEATURE_MLP_HIDDEN),                               \
            LARGER(REGISTER_TOKENS, REGISTER_MLP_HIDDEN)                       \
        )                                                                      \
    )

/// 127 / FLT_MAX, rounded up to the next float.
/// A row whose largest magnitude m is at least this has a finite 127 / m, so a smaller one, zero included, is set to zero instead of being divided.
#define QUANT_MIN_RANGE 0x1.fc0002p-122f

/// 1.5 · 2^23. Adding it to a float v with |v| < 2^22 leaves no bits below 1, so the add rounds v to the nearest integer, ties to even, and subtracting it again gives that integer exactly.
/// https://en.wikipedia.org/wiki/Floating-point_arithmetic#Rounding_modes
#define ROUND_MAGIC 12582912.0f

// An int32 sum of `in` products of two int8 values is at most in · 127 · 127, 4,451,604 for the longest row, under 2^24, so converting it to float is exact.
// A longer input row would round that conversion, and the result would no longer be the exact sum scaled once.
_Static_assert(
    IN_CAPACITY * QUANT_MAX * QUANT_MAX < (1 << 24),
    "an int32 sum of int8 products must convert to float exactly"
);

/// The block's size, as in `blocked.c`.
/// An enum rather than #define, because the `GCC unroll` pragma doesn't expand macros but does read an enum constant.
enum block_size {
    BLOCK_ROWS = 4, ///< Rows per block.
    BLOCK_OUTS = 4, ///< Outputs per block.
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// sx = max|x| / 127, qx = round(x · (127 / max|x|))
// https://en.wikipedia.org/wiki/Quantization_(signal_processing)
// Per-row (per-token) dynamic scales: Xiao et al. 2023, SmoothQuant, https://arxiv.org/abs/2211.10438
void quantize_rows(
    const float *restrict x,
    int8_t *restrict qx,
    float *restrict sx,
    int rows,
    int in
) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * in;
        int8_t *qr = qx + r * in;

        // Step 1: the row's largest magnitude, exact in any order
        float m = 0.0f;
        for (int i = 0; i < in; i++) {
            m = fmaxf(m, fabsf(xr[i]));
        }

        // Step 2: the row's scale, and its inverse to multiply by
        // A row too small to divide by can't use 127 / m, since 0 · inf would give NaN and an overflowed 127 / m converts to a different integer on x86 and on RISC-V
        // It gets scale 1 and k = 0 instead, so each finite x · 0 is ±0, and the rounding below turns every value into 0
        float k;
        if (m < QUANT_MIN_RANGE) {
            sx[r] = 1.0f;
            k = 0.0f;
        } else {
            sx[r] = m / (float)QUANT_MAX;
            k = (float)QUANT_MAX / m;
        }

        // Step 3: round each value, by plain float operations, since lrintf and its kin are libm calls on rv64
        for (int i = 0; i < in; i++) {
            // |x · k| can exceed 127 by a rounding step, so clamp before converting
            float v =
                fminf(fmaxf(xr[i] * k, -(float)QUANT_MAX), (float)QUANT_MAX);
            // Round to nearest, ties to even, as the export's Python round does; (int32_t)(v + 0.5f) would send 0.49999997 to 1
            v = (v + ROUND_MAGIC) - ROUND_MAGIC;
            qr[i] = (int8_t)(int32_t)v;
        }
    }
}

/**
 * Computes the 4 × 4 block of `y` that starts at row `r` and output `o`, from the quantized input.
 *
 * @param qx The quantized input, rows of `in` values.
 * @param sx Each input row's scale.
 * @param w The int8 weights, `out` rows of `in` values.
 * @param scale The scale of each weight row, `out` values.
 * @param b The bias, `out` values.
 * @param y Receives rows of `out` values.
 * @param in Values per input row.
 * @param out Values per output row.
 * @param r The block's first row.
 * @param o The block's first output.
 */
static void block_4x4(
    const int8_t *restrict qx,
    const float *restrict sx,
    const int8_t *restrict w,
    const float *restrict scale,
    const float *restrict b,
    float *restrict y,
    int in,
    int out,
    int r,
    int o
) {
    // Every loop over k or m is unrolled by its `GCC unroll` pragma: at -O2 GCC would leave these 4-step loops rolled and keep the arrays in stack memory

    // Step 1: point at the block's input rows and weight rows, and start each sum from zero
    const int8_t *xr[BLOCK_ROWS];
    const int8_t *wo[BLOCK_OUTS];
    int32_t acc[BLOCK_ROWS][BLOCK_OUTS];
#pragma GCC unroll BLOCK_ROWS
    for (int k = 0; k < BLOCK_ROWS; k++) {
        xr[k] = qx + (r + k) * in;
    }
#pragma GCC unroll BLOCK_OUTS
    for (int m = 0; m < BLOCK_OUTS; m++) {
        wo[m] = w + (o + m) * in;
    }
#pragma GCC unroll BLOCK_ROWS
    for (int k = 0; k < BLOCK_ROWS; k++) {
#pragma GCC unroll BLOCK_OUTS
        for (int m = 0; m < BLOCK_OUTS; m++) {
            acc[k][m] = 0;
        }
    }

    // Step 2: add every product into its integer sum
    for (int i = 0; i < in; i++) {
        int32_t xi[BLOCK_ROWS];
        int32_t wi[BLOCK_OUTS];
#pragma GCC unroll BLOCK_ROWS
        for (int k = 0; k < BLOCK_ROWS; k++) {
            xi[k] = xr[k][i];
        }
#pragma GCC unroll BLOCK_OUTS
        for (int m = 0; m < BLOCK_OUTS; m++) {
            wi[m] = wo[m][i];
        }
#pragma GCC unroll BLOCK_ROWS
        for (int k = 0; k < BLOCK_ROWS; k++) {
#pragma GCC unroll BLOCK_OUTS
            for (int m = 0; m < BLOCK_OUTS; m++) {
                acc[k][m] += xi[k] * wi[m];
            }
        }
    }

    // Step 3: scale each sum back to float once and add the bias, from the block's first value in y; (float)acc is exact, since |acc| < 2^24
    float *y_block = y + r * out + o;
#pragma GCC unroll BLOCK_ROWS
    for (int k = 0; k < BLOCK_ROWS; k++) {
#pragma GCC unroll BLOCK_OUTS
        for (int m = 0; m < BLOCK_OUTS; m++) {
            y_block[k * out + m] =
                b[o + m] + (sx[r + k] * scale[o + m]) * (float)acc[k][m];
        }
    }
}

/**
 * Computes one value of `y`, at row `r` and output `o`, by the plain loop over the quantized input.
 *
 * @param qx The quantized input, rows of `in` values.
 * @param sx Each input row's scale.
 * @param w The int8 weights, `out` rows of `in` values.
 * @param scale The scale of each weight row, `out` values.
 * @param b The bias, `out` values.
 * @param y Receives rows of `out` values.
 * @param in Values per input row.
 * @param out Values per output row.
 * @param r The value's row.
 * @param o The value's output.
 */
static void single(
    const int8_t *restrict qx,
    const float *restrict sx,
    const int8_t *restrict w,
    const float *restrict scale,
    const float *restrict b,
    float *restrict y,
    int in,
    int out,
    int r,
    int o
) {
    const int8_t *xr = qx + r * in;
    const int8_t *wo = w + o * in;
    int32_t acc = 0;
    for (int i = 0; i < in; i++) {
        acc += (int32_t)xr[i] * (int32_t)wo[i];
    }
    y[r * out + o] = b[o] + (sx[r] * scale[o]) * (float)acc;
}

// y[r][o] = b[o] + (sx[r] · s[o]) · Σ_i qx[r][i] · q[o][i]
// https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
// int8 products with int32 sums and per-channel weight scales: Nagel et al. 2021, A White Paper on Neural Network Quantization, https://arxiv.org/abs/2106.08295
void linear_w8a8(
    const float *restrict x,
    const int8_t *restrict w,
    const float *restrict scale,
    const float *restrict b,
    float *restrict y,
    int rows,
    int in,
    int out
) {
    static int8_t qx[QUANT_INPUT_CAPACITY];
    static float sx[QUANT_ROWS_CAPACITY];

    // Step 1: quantize every input row once, since each feeds all the outputs
    quantize_rows(x, qx, sx, rows, in);

    // The rows and outputs that fill whole blocks
    int block_rows_end = rows - rows % BLOCK_ROWS;
    int block_outs_end = out - out % BLOCK_OUTS;

    for (int r = 0; r < block_rows_end; r += BLOCK_ROWS) {
        // Step 2: the whole 4 × 4 blocks of this group of rows
        for (int o = 0; o < block_outs_end; o += BLOCK_OUTS) {
            block_4x4(qx, sx, w, scale, b, y, in, out, r, o);
        }

        // Step 3: the outputs left over past the last whole block, one value at a time
        for (int k = 0; k < BLOCK_ROWS; k++) {
            for (int o = block_outs_end; o < out; o++) {
                single(qx, sx, w, scale, b, y, in, out, r + k, o);
            }
        }
    }

    // Step 4: the rows left over past the last whole block, one value at a time
    for (int r = block_rows_end; r < rows; r++) {
        for (int o = 0; o < out; o++) {
            single(qx, sx, w, scale, b, y, in, out, r, o);
        }
    }
}
