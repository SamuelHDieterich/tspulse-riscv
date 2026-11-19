/**
 * @file
 * Checks the quantized Linear kernel of the build's `QUANT` value on the mixers' shapes, bit for bit.
 *
 * `linear_w8a32` must equal `kernels/linear/blocked.c`'s `linear` run on the dequantized weights, `scale[o] * (float)q[o][i]`.
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

/// The largest input or output of a mixer Linear: 24 rows of 276 in the patch mixer, or 138 rows of 48 in the feature mixer.
#define ACTIVATION_CAPACITY                                                    \
    (D_MODEL * PATCH_MLP_HIDDEN > TOTAL_PATCHES * FEATURE_MLP_HIDDEN           \
         ? D_MODEL * PATCH_MLP_HIDDEN                                          \
         : TOTAL_PATCHES * FEATURE_MLP_HIDDEN)

/// The largest mixer weight: the patch MLP's 276 × 138.
#define WEIGHT_CAPACITY (PATCH_MLP_HIDDEN * TOTAL_PATCHES)

/// The largest number of outputs, the patch MLP's hidden layer.
#define OUT_CAPACITY PATCH_MLP_HIDDEN

/// The int8 range the export uses, symmetric, so -128 never appears.
#define QUANT_MAX 127

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
 * Checks `linear_w8a32` on one shape against `linear` on the dequantized weights.
 *
 * @param shape The shape to check.
 * @param state The generator's state.
 * @return 0 on a match, 1 otherwise.
 */
static int check_w8a32(const struct shape *shape, uint32_t *state) {
    static float x[ACTIVATION_CAPACITY];
    static int8_t q[WEIGHT_CAPACITY];
    static float dequantized[WEIGHT_CAPACITY];
    static float scale[OUT_CAPACITY];
    static float bias[OUT_CAPACITY];
    static float got[ACTIVATION_CAPACITY];
    static float expected[ACTIVATION_CAPACITY];

    // Step 1: refuse a shape the buffers can't hold, instead of writing past them
    if (shape->rows * shape->in > ACTIVATION_CAPACITY ||
        shape->rows * shape->out > ACTIVATION_CAPACITY ||
        shape->out * shape->in > WEIGHT_CAPACITY || shape->out > OUT_CAPACITY) {
        printf("%s: shape exceeds the check's buffers\n", shape->name);
        return 1;
    }

    // Step 2: draw the inputs, the int8 weights, their row scales and the bias
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

// ░█▄█░█▀█░▀█▀░█▀█
// ░█░█░█▀█░░█░░█░█
// ░▀░▀░▀░▀░▀▀▀░▀░▀

int main(void) {
    uint32_t state = LCG_SEED;
    int failures = 0;
    for (int s = 0; s < ARRAY_LEN(SHAPES); s++) {
        failures += check_w8a32(&SHAPES[s], &state);
    }
    return failures == 0 ? 0 : 1;
}
