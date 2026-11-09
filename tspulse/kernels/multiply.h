/**
 * @file
 * Element-wise product of two arrays, which the gated attention uses to weigh its input.
 */

#ifndef MULTIPLY_H
#define MULTIPLY_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Writes `a * b` into `y`, value by value.
 *
 * @param a The first array, `n` values.
 * @param b The second array, `n` values.
 * @param y Receives the products. It may be `a` or `b`.
 * @param n Values in each array.
 */
void multiply(const float *a, const float *b, float *y, int n);

#endif
