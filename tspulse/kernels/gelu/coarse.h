/**
 * @file
 * GELU with `erf` from Abramowitz and Stegun 7.1.27, about 7e-4 absolute.
 */

#ifndef GELU_COARSE_H
#define GELU_COARSE_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/gelu/erf_form.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░░▀░░▀▀▀

/// Coefficients of the degree-4 polynomial, from a1 (of z) to a4 (of z^4).
#define GELU_COARSE_A1 0.278393f
#define GELU_COARSE_A2 0.230389f
#define GELU_COARSE_A3 0.000972f
#define GELU_COARSE_A4 0.078108f

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * GELU(x) with erf(z) = 1 - 1 / (1 + a1 z + ... + a4 z^4)^4 for z >= 0.
 *
 * The formula is off by at most 5e-4 in `erf`, which is about 7e-4 in GELU.
 * Abramowitz and Stegun, Handbook of Mathematical Functions, 7.1.27.
 * https://en.wikipedia.org/wiki/Abramowitz_and_Stegun
 *
 * @param x The input.
 * @return GELU(x).
 */
static inline float gelu_coarse(float x) {
    // Step 1: z = |x| / sqrt(2)
    float z = (x < 0.0f ? -x : x) * GELU_ERF_SQRT1_2;

    // Step 2: the polynomial by Horner's rule
    float p = GELU_COARSE_A4;
    p = p * z + GELU_COARSE_A3;
    p = p * z + GELU_COARSE_A2;
    p = p * z + GELU_COARSE_A1;
    p = p * z + 1.0f;

    // Step 3: p^4 by two squarings
    float p4 = p * p;
    p4 = p4 * p4;

    // Step 4: erf(z) = 1 - 1 / p^4
    return gelu_from_erf_magnitude(x, 1.0f - 1.0f / p4);
}

#endif
