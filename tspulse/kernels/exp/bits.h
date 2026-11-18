/**
 * @file
 * The exponential by Schraudolph's trick: one multiply-add and a reinterpretation of the bits, within 4% relative.
 */

#ifndef EXP_BITS_H
#define EXP_BITS_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>
#include <string.h>

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Below this, the integer would fall under the exponent field's smallest value.
#define EXP_BITS_LOWEST -87.0f

/// Above this, the integer would pass the exponent field's largest value.
#define EXP_BITS_HIGHEST 88.0f

/// 2^23 / ln(2): one unit of x moves the float's exponent by 1/ln(2), and the exponent starts at bit 23.
#define EXP_BITS_SCALE 12102203.0f

/// 127 * 2^23 less a correction that minimises the RMS relative error, which leaves it between -3.94% and +1.97%.
#define EXP_BITS_OFFSET 1064866805.0f

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * e^x for any x, clamped so the integer stays inside the exponent field.
 *
 * A float's bits, read as an integer, are roughly 2^23 * (log2(value) + 127), so a value of e^x has bits 2^23 * (x / ln(2) + 127).
 * Schraudolph, "A Fast, Compact Approximation of the Exponential Function", Neural Computation 11(4), 1999.
 * https://doi.org/10.1162/089976699300016467
 *
 * @param x The exponent.
 * @return e^x, up to 4%.
 */
static inline float exp_bits(float x) {
    // Step 1: clamp, so the conversion below stays in range, and NaN falls to the low end
    if (!(x >= EXP_BITS_LOWEST)) {
        x = EXP_BITS_LOWEST;
    }
    if (x > EXP_BITS_HIGHEST) {
        x = EXP_BITS_HIGHEST;
    }

    // Step 2: the integer whose bits are the answer
    uint32_t bits = (uint32_t)(int32_t)(EXP_BITS_SCALE * x + EXP_BITS_OFFSET);
    float result;
    memcpy(&result, &bits, sizeof result);

    return result;
}

#endif
