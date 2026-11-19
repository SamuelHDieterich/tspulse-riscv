/**
 * @file
 * Checks the quantized Linear kernel of the build's `QUANT` value on the mixers' shapes, bit for bit.
 *
 * On a w8a32 build, `linear_w8a32` must equal `kernels/linear/blocked.c`'s `linear` run on the dequantized weights, `scale[o] * (float)q[o][i]`.
 * On a w8a8 build, `linear_w8a8` must equal a plain triple loop over the input quantized by the same `quantize_rows`, scaled back by the same expression; the integer sums are exact, so their order can't make the two differ.
 * Its first input row is all zeros, which must quantize to zeros and give `y == b` exactly, and its second holds one large outlier.
 * Inputs, weights, scales and biases come from a fixed linear congruential generator rather than `rand()`, whose sequence differs between glibc and newlib, so the host and rv64 check the same numbers.
 *
 * Prints one line per shape, `pass` or the first differing index, and exits 1 on any difference.
 *
 * Usage: check-qlinear
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Internal
#include "kernels/linear.h"
#include "kernels/qlinear.h"
#include "model/sizes.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The largest mixer weight: the patch MLP's 276 × 138.
#define WEIGHT_CAPACITY (PATCH_MLP_HIDDEN * TOTAL_PATCHES)

/// The largest number of outputs, the patch MLP's hidden layer.
#define OUT_CAPACITY PATCH_MLP_HIDDEN

/// The generator's multiplier, from Numerical Recipes.
#define LCG_MULTIPLIER 1664525u

/// The generator's increment, from Numerical Recipes.
#define LCG_INCREMENT 1013904223u

/// The generator's starting state.
#define LCG_SEED 12345u

/// Drops the state's 8 low bits, whose period is short, and keeps the top 24.
#define LCG_TOP_BITS_SHIFT 8

/// The top 24 bits of a state, scaled by 2^-24, give a float in [0, 1) exactly.
#define LCG_FLOAT_STEP 0x1p-24f

/// The smallest scale drawn. Scales are drawn in [SCALE_LOW, 2 · SCALE_LOW), so 127 · scale, the largest weight a row can hold, lies in [0.5, 1).
#define SCALE_LOW (0.5f / QUANT_MAX)

/// The w8a8 check's input row that is all zeros.
#define ZERO_ROW 0

/// The w8a8 check's input row that holds one outlier.
#define OUTLIER_ROW 1

/// The outlier, far above the other inputs in [-1, 1), so the row's scale rounds all of them to 0 and the outlier to 127.
#define OUTLIER_VALUE 1000.0f

/// Number of elements in an array.
#define ARRAY_LEN(array) ((int)(sizeof(array) / sizeof((array)[0])))

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// One Linear call's shape, as the mixers make it.
struct shape {
    const char *name; ///< Which mixer Linear it is.
    int rows;         ///< Rows in the input.
    int in;           ///< Values per input row.
    int out;          ///< Values per output row.
};

/// Every mixer Linear shape: the patch mixer over 138 patches, the feature mixer, and the register mixer over 10 patches.
static const struct shape SHAPES[] = {
    {"patch fc1", D_MODEL, TOTAL_PATCHES, PATCH_MLP_HIDDEN},
    {"patch fc2", D_MODEL, PATCH_MLP_HIDDEN, TOTAL_PATCHES},
    {"patch gate", D_MODEL, TOTAL_PATCHES, TOTAL_PATCHES},
    {"feature fc1", TOTAL_PATCHES, D_MODEL, FEATURE_MLP_HIDDEN},
    {"feature fc2", TOTAL_PATCHES, FEATURE_MLP_HIDDEN, D_MODEL},
    {"register fc1", D_MODEL, REGISTER_TOKENS, REGISTER_MLP_HIDDEN},
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Advances the generator one step.
 *
 * state = a · state + c (mod 2^32)
 * https://en.wikipedia.org/wiki/Linear_congruential_generator
 *
 * @param state The generator's state, updated in place.
 * @return The new state.
 */
static uint32_t lcg_next(uint32_t *state) {
    // Unsigned arithmetic wraps modulo 2^32 by definition, which is the generator's modulus
    *state = LCG_MULTIPLIER * *state + LCG_INCREMENT;
    return *state;
}

/**
 * Draws a float in [0, 1).
 *
 * @param state The generator's state.
 * @return The value.
 */
static float lcg_unit(uint32_t *state) {
    return (float)(lcg_next(state) >> LCG_TOP_BITS_SHIFT) * LCG_FLOAT_STEP;
}

/**
 * Draws a float in [-1, 1).
 *
 * @param state The generator's state.
 * @return The value.
 */
static float lcg_signed(uint32_t *state) {
    return 2.0f * lcg_unit(state) - 1.0f;
}

/**
 * Draws an int8 weight in [-127, 127].
 *
 * @param state The generator's state.
 * @return The value.
 */
static int8_t lcg_weight(uint32_t *state) {
    uint32_t draw = lcg_next(state) >> LCG_TOP_BITS_SHIFT;
    return (int8_t)((int32_t)(draw % (2 * QUANT_MAX + 1)) - QUANT_MAX);
}

/**
 * Compares two outputs bit for bit and prints the shape's result.
 *
 * @param shape The shape the outputs belong to.
 * @param got The kernel's output.
 * @param expected The reference output.
 * @param count Values in each output.
 * @return 0 when they are identical, 1 otherwise.
 */
static int report(
    const struct shape *shape,
    const float *got,
    const float *expected,
    int count
) {
    if (memcmp(got, expected, (size_t)count * sizeof(float)) == 0) {
        printf(
            "%s %dx%d->%d: pass\n", shape->name, shape->rows, shape->in,
            shape->out
        );
        return 0;
    }

    // memcmp only says that they differ, so find the first value that does
    int index = 0;
    while (memcmp(&got[index], &expected[index], sizeof(float)) == 0) {
        index++;
    }
    printf(
        "%s %dx%d->%d: differs at index %d, got %.9g, expected %.9g\n",
        shape->name, shape->rows, shape->in, shape->out, index,
        (double)got[index], (double)expected[index]
    );
    return 1;
}

/**
 * Checks that a shape fits the check's buffers, and prints why when it doesn't.
 *
 * @param shape The shape to check.
 * @return 1 when it fits, 0 otherwise.
 */
static int fits(const struct shape *shape) {
    if (shape->rows * shape->in > QUANT_INPUT_CAPACITY ||
        shape->rows * shape->out > QUANT_INPUT_CAPACITY ||
        shape->out * shape->in > WEIGHT_CAPACITY || shape->out > OUT_CAPACITY ||
        shape->rows > QUANT_ROWS_CAPACITY) {
        printf("%s: shape exceeds the check's buffers\n", shape->name);
        return 0;
    }
    return 1;
}

/**
 * Draws the inputs, the int8 weights, their row scales and the bias for one shape.
 *
 * @param shape The shape to draw for.
 * @param state The generator's state.
 * @param x Receives `rows` rows of `in` inputs.
 * @param q Receives `out` rows of `in` int8 weights.
 * @param scale Receives `out` row scales.
 * @param bias Receives `out` biases.
 */
static void draw(
    const struct shape *shape,
    uint32_t *state,
    float *x,
    int8_t *q,
    float *scale,
    float *bias
) {
    for (int j = 0; j < shape->rows * shape->in; j++) {
        x[j] = lcg_signed(state);
    }
    for (int o = 0; o < shape->out; o++) {
        scale[o] = SCALE_LOW * (1.0f + lcg_unit(state));
        bias[o] = lcg_signed(state);
        for (int i = 0; i < shape->in; i++) {
            q[o * shape->in + i] = lcg_weight(state);
        }
    }
}

#if defined(TSPULSE_QUANT_W8A32)
/**
 * Checks `linear_w8a32` on one shape against `linear` on the dequantized weights.
 *
 * @param shape The shape to check.
 * @param state The generator's state.
 * @return 0 on a match, 1 otherwise.
 */
static int check_w8a32(const struct shape *shape, uint32_t *state) {
    static float x[QUANT_INPUT_CAPACITY];
    static int8_t q[WEIGHT_CAPACITY];
    static float dequantized[WEIGHT_CAPACITY];
    static float scale[OUT_CAPACITY];
    static float bias[OUT_CAPACITY];
    static float got[QUANT_INPUT_CAPACITY];
    static float expected[QUANT_INPUT_CAPACITY];

    // Step 1: refuse a shape the buffers can't hold, instead of writing past them
    if (!fits(shape)) {
        return 1;
    }

    // Step 2: draw the inputs, the int8 weights, their row scales and the bias
    draw(shape, state, x, q, scale, bias);

    // Step 3: the float weights the model is meant to use, by the same product as the kernel
    for (int o = 0; o < shape->out; o++) {
        for (int i = 0; i < shape->in; i++) {
            dequantized[o * shape->in + i] =
                scale[o] * (float)q[o * shape->in + i];
        }
    }

    // Step 4: run both kernels and compare
    linear_w8a32(x, q, scale, bias, got, shape->rows, shape->in, shape->out);
    linear(x, dequantized, bias, expected, shape->rows, shape->in, shape->out);
    return report(shape, got, expected, shape->rows * shape->out);
}
#endif

#if defined(TSPULSE_QUANT_W8A8)
/**
 * Checks `linear_w8a8` on one shape against a plain triple loop over the same quantized input, and checks the all-zero row on its own.
 *
 * @param shape The shape to check.
 * @param state The generator's state.
 * @return 0 on a match, 1 otherwise.
 */
static int check_w8a8(const struct shape *shape, uint32_t *state) {
    static float x[QUANT_INPUT_CAPACITY];
    static int8_t q[WEIGHT_CAPACITY];
    static float scale[OUT_CAPACITY];
    static float bias[OUT_CAPACITY];
    static int8_t qx[QUANT_INPUT_CAPACITY];
    static float sx[QUANT_ROWS_CAPACITY];
    static float got[QUANT_INPUT_CAPACITY];
    static float expected[QUANT_INPUT_CAPACITY];
    static const int8_t zeros[PATCH_MLP_HIDDEN];

    // Step 1: refuse a shape the buffers can't hold, or too short for the edge rows
    if (!fits(shape)) {
        return 1;
    }
    if (shape->rows <= OUTLIER_ROW || shape->in > ARRAY_LEN(zeros)) {
        printf("%s: shape can't hold the w8a8 edge rows\n", shape->name);
        return 1;
    }

    // Step 2: draw the inputs, the int8 weights, their row scales and the bias, then set the edge rows
    draw(shape, state, x, q, scale, bias);
    float *zero_row = x + ZERO_ROW * shape->in;
    for (int i = 0; i < shape->in; i++) {
        zero_row[i] = 0.0f;
    }
    x[OUTLIER_ROW * shape->in + shape->in / 2] = OUTLIER_VALUE;

    // Step 3: the reference, a plain triple loop over the input quantized by the kernel's own quantize_rows
    // y[r][o] = b[o] + (sx[r] · s[o]) · Σ_i qx[r][i] · q[o][i], with the kernel's operation order and the same compiler flags, so both fuse it into one multiply-add or neither does, and the float rounding matches too
    quantize_rows(x, qx, sx, shape->rows, shape->in);
    for (int r = 0; r < shape->rows; r++) {
        for (int o = 0; o < shape->out; o++) {
            int32_t acc = 0;
            for (int i = 0; i < shape->in; i++) {
                acc += (int32_t)qx[r * shape->in + i] *
                       (int32_t)q[o * shape->in + i];
            }
            expected[r * shape->out + o] =
                bias[o] + (sx[r] * scale[o]) * (float)acc;
        }
    }

    // Step 4: the outlier row on its own: its scale must come from the outlier, which quantizes to 127 while every other value rounds to 0
    const int8_t *outlier_qx = qx + OUTLIER_ROW * shape->in;
    int outlier_at = shape->in / 2;
    int outlier_row_ok = outlier_qx[outlier_at] == QUANT_MAX;
    for (int i = 0; i < shape->in; i++) {
        if (i != outlier_at && outlier_qx[i] != 0) {
            outlier_row_ok = 0;
        }
    }
    if (!outlier_row_ok) {
        printf(
            "%s: the outlier row didn't quantize to one 127 and zeros\n",
            shape->name
        );
        return 1;
    }

    // Step 5: the kernel, and the all-zero row on its own: it must quantize to zeros and give the bias exactly, whatever the reference says
    linear_w8a8(x, q, scale, bias, got, shape->rows, shape->in, shape->out);
    int zero_row_ok =
        memcmp(qx + ZERO_ROW * shape->in, zeros, (size_t)shape->in) == 0 &&
        memcmp(
            got + ZERO_ROW * shape->out, bias,
            (size_t)shape->out * sizeof(float)
        ) == 0;
    if (!zero_row_ok) {
        printf(
            "%s: the all-zero row didn't quantize to zeros or give the bias\n",
            shape->name
        );
        return 1;
    }

    // Step 6: compare the whole output with the reference
    return report(shape, got, expected, shape->rows * shape->out);
}
#endif

// ░█▄█░█▀█░▀█▀░█▀█
// ░█░█░█▀█░░█░░█░█
// ░▀░▀░▀░▀░▀▀▀░▀░▀

int main(void) {
    uint32_t state = LCG_SEED;
    int failures = 0;
    for (int s = 0; s < ARRAY_LEN(SHAPES); s++) {
#if defined(TSPULSE_QUANT_W8A32)
        failures += check_w8a32(&SHAPES[s], &state);
#elif defined(TSPULSE_QUANT_W8A8)
        failures += check_w8a8(&SHAPES[s], &state);
#endif
    }
    return failures == 0 ? 0 : 1;
}
