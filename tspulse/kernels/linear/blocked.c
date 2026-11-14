/**
 * @file
 * PyTorch's Linear by register blocking: 4 rows times 4 outputs at a time, with the same result as the plain loop, bit for bit.
 *
 * Each 4 × 4 block keeps its 16 sums in registers. Every input and weight loaded feeds four multiply-adds instead of one, and the 16 independent sums let one multiply-add start while the others are still in flight.
 *
 * Every sum is still the plain loop's expression: it starts from its bias and adds x * w over the inputs in order, so GCC fuses it into the same chain of multiply-adds and every output keeps the plain loop's rounding.
 * That holds only while each product is used once, by its own sum, and never shared or combined across sums, and while no flag lets the compiler reorder float math, such as `-ffast-math` or `-fassociative-math`.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/linear.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The block's size. 4 × 4 sums plus the 4 + 4 values loaded per input take 24 of RV64's 32 float registers, so nothing spills to memory.
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
 * @param w The weights, `out` rows of `in` values.
 * @param b The bias, `out` values.
 * @param y Receives rows of `out` values.
 * @param in Values per input row.
 * @param out Values per output row.
 * @param r The block's first row.
 * @param o The block's first output.
 */
static void block_4x4(
    const float *restrict x,
    const float *restrict w,
    const float *restrict b,
    float *restrict y,
    int in,
    int out,
    int r,
    int o
) {
    // Every loop over k or m is unrolled by its `GCC unroll` pragma: GCC -O2 leaves these 4-step loops rolled, and then keeps xr, wo, acc, xi and wi in stack memory instead of registers

    // Step 1: point at the block's input rows and weight rows, and start each sum from its bias
    const float *xr[BLOCK_ROWS];
    const float *wo[BLOCK_OUTS];
    float acc[BLOCK_ROWS][BLOCK_OUTS];
#pragma GCC unroll BLOCK_ROWS
    for (int k = 0; k < BLOCK_ROWS; k++) {
        xr[k] = x + (r + k) * in;
    }
#pragma GCC unroll BLOCK_OUTS
    for (int m = 0; m < BLOCK_OUTS; m++) {
        wo[m] = w + (o + m) * in;
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
            wi[m] = wo[m][i];
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
 * @param w The weights, `out` rows of `in` values.
 * @param b The bias, `out` values.
 * @param y Receives rows of `out` values.
 * @param in Values per input row.
 * @param out Values per output row.
 * @param r The value's row.
 * @param o The value's output.
 */
static void single(
    const float *restrict x,
    const float *restrict w,
    const float *restrict b,
    float *restrict y,
    int in,
    int out,
    int r,
    int o
) {
    const float *xr = x + r * in;
    const float *wo = w + o * in;
    float acc = b[o];
    for (int i = 0; i < in; i++) {
        acc += xr[i] * wo[i];
    }
    y[r * out + o] = acc;
}

// y[r][o] = b[o] + sum_i x[r][i] * w[o][i]
// https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linear(
    const float *restrict x,
    const float *restrict w,
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
            block_4x4(x, w, b, y, in, out, r, o);
        }

        // Step 2: the outputs left over past the last whole block, one value at a time
        for (int k = 0; k < BLOCK_ROWS; k++) {
            for (int o = block_outs_end; o < out; o++) {
                single(x, w, b, y, in, out, r + k, o);
            }
        }
    }

    // Step 3: the rows left over past the last whole block, one value at a time
    for (int r = block_rows_end; r < rows; r++) {
        for (int o = 0; o < out; o++) {
            single(x, w, b, y, in, out, r, o);
        }
    }
}
