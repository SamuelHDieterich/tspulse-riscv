/**
 * @file
 * The exponential by the same range reduction as `fine.h` with a degree-3 Taylor polynomial, within 8e-4 relative.
 */

#ifndef EXP_COARSE_H
#define EXP_COARSE_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/exp/reduce.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Taylor coefficients 1/2 and 1/6.
#define EXP_COARSE_HALF 0.5f
#define EXP_COARSE_SIXTH 0.16666667f

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * e^x for any x, clamped to the range where the result is a finite normal float.
 *
 * With x = n * ln(2) + r from `exp_reduce`: e^x = 2^n * e^r, and e^r = 1 + r + r^2/2 + r^3/6, off by about r^4/24, so at most 8e-4 relative.
 * https://en.wikipedia.org/wiki/Taylor_series
 *
 * @param x The exponent.
 * @return e^x.
 */
static inline float exp_coarse(float x) {
    // Step 1: x = n * ln(2) + r
    int n;
    float r;
    exp_reduce(x, &n, &r);

    // Step 2: e^r by Horner's rule
    float exp_r =
        ((EXP_COARSE_SIXTH * r + EXP_COARSE_HALF) * r + 1.0f) * r + 1.0f;

    // Step 3: scale by 2^n
    return exp_scale(exp_r, n);
}

#endif
