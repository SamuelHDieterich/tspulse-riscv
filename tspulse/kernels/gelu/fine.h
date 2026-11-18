/**
 * @file
 * GELU with `erf` from Abramowitz and Stegun 7.1.28, about 9e-7 absolute over [-10, 10], close to float's own rounding there.
 */

#ifndef GELU_FINE_H
#define GELU_FINE_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/gelu/erf_form.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░░▀░░▀▀▀

/// Coefficients of the degree-6 polynomial, from a1 (of z) to a6 (of z^6).
#define GELU_FINE_A1 0.0705230784f
#define GELU_FINE_A2 0.0422820123f
#define GELU_FINE_A3 0.0092705272f
#define GELU_FINE_A4 0.0001520143f
#define GELU_FINE_A5 0.0002765672f
#define GELU_FINE_A6 0.0000430638f

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * GELU(x) with erf(z) = 1 - 1 / (1 + a1 z + ... + a6 z^6)^16 for z >= 0.
 *
 * The formula is exact to 3e-7 in `erf`; rounding in the 16th power and the division brings the float result to about 1.9e-6 in `erf` and 9e-7 in GELU over [-10, 10], where rounding the result itself costs about 4.5e-7.
 * Abramowitz and Stegun, Handbook of Mathematical Functions, 7.1.28.
 * https://en.wikipedia.org/wiki/Abramowitz_and_Stegun
 *
 * @param x The input.
 * @return GELU(x).
 */
static inline float gelu_fine(float x) {
    // Step 1: z = |x| / sqrt(2)
    float z = (x < 0.0f ? -x : x) * GELU_ERF_SQRT1_2;

    // Step 2: the polynomial by Horner's rule
    float p = GELU_FINE_A6;
    p = p * z + GELU_FINE_A5;
    p = p * z + GELU_FINE_A4;
    p = p * z + GELU_FINE_A3;
    p = p * z + GELU_FINE_A2;
    p = p * z + GELU_FINE_A1;
    p = p * z + 1.0f;

    // Step 3: p^16 by four squarings
    float p16 = p * p;
    p16 = p16 * p16;
    p16 = p16 * p16;
    p16 = p16 * p16;

    // Step 4: erf(z) = 1 - 1 / p^16, which is 1 when p^16 overflows
    return gelu_from_erf_magnitude(x, 1.0f - 1.0f / p16);
}

#endif
