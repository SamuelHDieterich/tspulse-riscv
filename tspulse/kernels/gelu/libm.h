/**
 * @file
 * GELU in its exact form through the math library's `erff`, as the baseline computes it.
 */

#ifndef GELU_LIBM_H
#define GELU_LIBM_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))), the default `approximate='none'`.
 * https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
 *
 * @param x The input.
 * @return GELU(x).
 */
static inline float gelu_libm(float x) {
    // M_SQRT1_2 is a double, so it is cast to keep the product in float
    return 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2));
}

#endif
