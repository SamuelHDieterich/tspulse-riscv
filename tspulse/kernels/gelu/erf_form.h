/**
 * @file
 * The part of GELU's erf form that the two polynomial approximations of `erf` share.
 */

#ifndef GELU_ERF_FORM_H
#define GELU_ERF_FORM_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░░▀░░▀▀▀

/// 1 / sqrt(2), as a float so that the product with x stays in float.
#define GELU_ERF_SQRT1_2 ((float)M_SQRT1_2)

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * GELU(x) from erf(|x| / sqrt(2)), using that erf is odd.
 *
 * GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))), and erf(-z) = -erf(z), so the approximations only need z >= 0.
 * https://en.wikipedia.org/wiki/Error_function#Properties
 *
 * @param x The input.
 * @param erf_magnitude erf(|x| / sqrt(2)), which is never negative.
 * @return GELU(x).
 */
static inline float gelu_from_erf_magnitude(float x, float erf_magnitude) {
    float erf_signed = x < 0.0f ? -erf_magnitude : erf_magnitude;
    return 0.5f * x * (1.0f + erf_signed);
}

#endif
