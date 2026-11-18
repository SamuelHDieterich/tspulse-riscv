/**
 * @file
 * The exponential by Cephes' `expf`: range reduction and a degree-5 polynomial, within a few ulp.
 */

#ifndef EXP_FINE_H
#define EXP_FINE_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/exp/reduce.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Coefficients of the polynomial, highest degree first.
#define EXP_FINE_P0 1.9875691500e-4f
#define EXP_FINE_P1 1.3981999507e-3f
#define EXP_FINE_P2 8.3334519073e-3f
#define EXP_FINE_P3 4.1665795894e-2f
#define EXP_FINE_P4 1.6666665459e-1f
#define EXP_FINE_P5 5.0000001201e-1f

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * e^x for any x, clamped to the range where the result is a finite normal float.
 *
 * With x = n * ln(2) + r from `exp_reduce`: e^x = 2^n * e^r, and e^r = 1 + r + r^2 * P(r) with P of degree 5.
 * https://www.netlib.org/cephes/ (single/expf.c)
 *
 * @param x The exponent.
 * @return e^x.
 */
static inline float exp_fine(float x) {
    // Step 1: x = n * ln(2) + r
    int n;
    float r;
    exp_reduce(x, &n, &r);

    // Step 2: e^r by Horner's rule
    float p = EXP_FINE_P0;
    p = p * r + EXP_FINE_P1;
    p = p * r + EXP_FINE_P2;
    p = p * r + EXP_FINE_P3;
    p = p * r + EXP_FINE_P4;
    p = p * r + EXP_FINE_P5;
    float exp_r = p * (r * r) + r + 1.0f;

    // Step 3: scale by 2^n
    return exp_scale(exp_r, n);
}

#endif
