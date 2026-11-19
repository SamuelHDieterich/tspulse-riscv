/**
 * @file
 * PyTorch's Linear on int8 weights with float32 math, by the same 4 × 4 register blocking as `kernels/linear/blocked.c`.
 *
 * Each weight is loaded as one int8 byte and turned back into float as `scale[o] * (float)q` just before its multiply-adds, so the weights take a quarter of the memory and the math stays float.
 * That product is rounded to float once, as the exported `.tdump` does, and then feeds the same chain of multiply-adds from the bias as `blocked.c`, so the result equals `blocked.c` run on the dequantized weights, bit for bit.
 * Like `blocked.c`, that holds only while no flag lets the compiler reorder float math, such as `-ffast-math` or `-fassociative-math`.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>

// Internal
#include "kernels/qlinear.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The block's size, as in `blocked.c`: 4 × 4 sums plus the 4 + 4 values loaded per input, and the 4 row scales, take 28 of RV64's 32 float registers.
/// An enum rather than #define, because the `GCC unroll` pragma doesn't expand macros but does read an enum constant.
enum block_size {
    BLOCK_ROWS = 4, ///< Rows per block.
    BLOCK_OUTS = 4, ///< Outputs per block.
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Computes the 4 × 4 block of `y` that starts at row `r` and output `o`.
 *
 * @param x The input, rows of `in` values.
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
    const float *restrict x,
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

    // Step 1: point at the block's input rows and weight rows, load the rows' scales, and start each sum from its bias
    const float *xr[BLOCK_ROWS];
    const int8_t *wo[BLOCK_OUTS];
    float so[BLOCK_OUTS];
    float acc[BLOCK_ROWS][BLOCK_OUTS];
#pragma GCC unroll BLOCK_ROWS
    for (int k = 0; k < BLOCK_ROWS; k++) {
        xr[k] = x + (r + k) * in;
    }
#pragma GCC unroll BLOCK_OUTS
    for (int m = 0; m < BLOCK_OUTS; m++) {
        wo[m] = w + (o + m) * in;
        so[m] = scale[o + m];
    }
#pragma GCC unroll BLOCK_ROWS
    for (int k = 0; k < BLOCK_ROWS; k++) {
#pragma GCC unroll BLOCK_OUTS
        for (int m = 0; m < BLOCK_OUTS; m++) {
            acc[k][m] = b[o + m];
        }
    }

    // Step 2: walk the inputs in order, so each sum adds its products in the same order as the plain loop
    for (int i = 0; i < in; i++) {
        float xi[BLOCK_ROWS];
        float wi[BLOCK_OUTS];
#pragma GCC unroll BLOCK_ROWS
        for (int k = 0; k < BLOCK_ROWS; k++) {
            xi[k] = xr[k][i];
        }
#pragma GCC unroll BLOCK_OUTS
        for (int m = 0; m < BLOCK_OUTS; m++) {
            // The dequantized weight, rounded to float once, and shared by the four rows' sums
            wi[m] = so[m] * (float)wo[m][i];
        }
#pragma GCC unroll BLOCK_ROWS
        for (int k = 0; k < BLOCK_ROWS; k++) {
#pragma GCC unroll BLOCK_OUTS
            for (int m = 0; m < BLOCK_OUTS; m++) {
                // Each sum takes its own product, which GCC fuses into one multiply-add as in the plain loop
                acc[k][m] += xi[k] * wi[m];
            }
        }
    }

    // Step 3: store the finished sums, from the block's first value in y
    float *y_block = y + r * out + o;
#pragma GCC unroll BLOCK_ROWS
    for (int k = 0; k < BLOCK_ROWS; k++) {
#pragma GCC unroll BLOCK_OUTS
        for (int m = 0; m < BLOCK_OUTS; m++) {
            y_block[k * out + m] = acc[k][m];
        }
    }
}

/**
 * Computes one value of `y`, at row `r` and output `o`, by the plain loop.
 *
 * @param x The input, rows of `in` values.
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
    const float *restrict x,
    const int8_t *restrict w,
    const float *restrict scale,
    const float *restrict b,
    float *restrict y,
    int in,
    int out,
    int r,
    int o
) {
    const float *xr = x + r * in;
    const int8_t *wo = w + o * in;
    float so = scale[o];
    float acc = b[o];
    for (int i = 0; i < in; i++) {
        // The dequantized weight is rounded to float before the multiply-add, as in the block
        float wi = so * (float)wo[i];
        acc += xr[i] * wi;
    }
    y[r * out + o] = acc;
}

// y[r][o] = b[o] + Σ_i x[r][i] · (s[o] · q[o][i])
// https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
// Per-row weight quantization: Nagel et al. 2021, A White Paper on Neural Network Quantization, https://arxiv.org/abs/2106.08295
void linear_w8a32(
    const float *restrict x,
    const int8_t *restrict w,
    const float *restrict scale,
    const float *restrict b,
    float *restrict y,
    int rows,
    int in,
    int out
) {
    // The rows and outputs that fill whole blocks
    int block_rows_end = rows - rows % BLOCK_ROWS;
    int block_outs_end = out - out % BLOCK_OUTS;

    for (int r = 0; r < block_rows_end; r += BLOCK_ROWS) {
        // Step 1: the whole 4 × 4 blocks of this group of rows
        for (int o = 0; o < block_outs_end; o += BLOCK_OUTS) {
            block_4x4(x, w, scale, b, y, in, out, r, o);
        }

        // Step 2: the outputs left over past the last whole block, one value at a time
        for (int k = 0; k < BLOCK_ROWS; k++) {
            for (int o = block_outs_end; o < out; o++) {
                single(x, w, scale, b, y, in, out, r + k, o);
            }
        }
    }

    // Step 3: the rows left over past the last whole block, one value at a time
    for (int r = block_rows_end; r < rows; r++) {
        for (int o = 0; o < out; o++) {
            single(x, w, scale, b, y, in, out, r, o);
        }
    }
}
