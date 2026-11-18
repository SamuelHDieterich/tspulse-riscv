/**
 * @file
 * The range reduction both polynomial exponentials share: x = n * ln(2) + r, so that e^x = 2^n * e^r.
 */

#ifndef EXP_REDUCE_H
#define EXP_REDUCE_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>
#include <string.h>

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Below this, e^x falls under the smallest normal float and 2^n would leave the exponent field.
#define EXP_REDUCE_LOWEST -87.3f

/// Above this, n would reach 128, whose exponent bits encode infinity.
#define EXP_REDUCE_HIGHEST 88.3f

/// log2(e), to turn x into a count of powers of two.
#define EXP_REDUCE_LOG2E 1.44269504088896341f

/// Cody-Waite split of ln(2): the high part has few mantissa bits, so n times it is exact.
#define EXP_REDUCE_LN2_HIGH 0.693359375f

/// The rest of ln(2), negative because the high part overshoots.
#define EXP_REDUCE_LN2_LOW -2.12194440e-4f

/// Bias of a float's exponent field.
#define EXP_REDUCE_EXPONENT_BIAS 127

/// Mantissa bits of a float, which is where the exponent field starts.
#define EXP_REDUCE_MANTISSA_BITS 23

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Splits x into n and r with x = n * ln(2) + r and |r| <= ln(2) / 2, after clamping x to the range where e^x is a finite normal float.
 *
 * n = round(x * log2(e)), and r is found by Cody-Waite reduction.
 * https://www.netlib.org/cephes/ (single/expf.c)
 * https://en.wikipedia.org/wiki/Range_reduction
 *
 * @param x The exponent.
 * @param n Receives the power of two.
 * @param r Receives what is left of x.
 */
static inline void exp_reduce(float x, int *n, float *r) {
    // Step 1: clamp, so the conversion below stays in range, and NaN falls to the low end
    if (!(x >= EXP_REDUCE_LOWEST)) {
        x = EXP_REDUCE_LOWEST;
    }
    if (x > EXP_REDUCE_HIGHEST) {
        x = EXP_REDUCE_HIGHEST;
    }

    // Step 2: n, rounded half away from zero by adding 0.5 by sign, since floorf and roundf would call the library on rv64
    float scaled = x * EXP_REDUCE_LOG2E;
    *n = (int)(scaled + (scaled < 0.0f ? -0.5f : 0.5f));

    // Step 3: r, in two subtractions: n times the high part is exact, so only the small product with the low part rounds
    float fn = (float)*n;
    *r = x - fn * EXP_REDUCE_LN2_HIGH;
    *r = *r - fn * EXP_REDUCE_LN2_LOW;
}

/**
 * value * 2^n, with 2^n built by writing n + 127 into the exponent field.
 *
 * @param value The factor to scale.
 * @param n The power of two, from `exp_reduce`.
 * @return value * 2^n.
 */
static inline float exp_scale(float value, int n) {
    uint32_t bits = (uint32_t)(n + EXP_REDUCE_EXPONENT_BIAS)
                    << EXP_REDUCE_MANTISSA_BITS;
    float two_n;
    memcpy(&two_n, &bits, sizeof two_n);

    return value * two_n;
}

#endif
