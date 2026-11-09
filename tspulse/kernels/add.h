/**
 * @file
 * Element-wise sum of two arrays, which the mixers use for their residual connections.
 */

#ifndef ADD_H
#define ADD_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Writes `a + b` into `y`, value by value.
 *
 * @param a The first array, `n` values.
 * @param b The second array, `n` values.
 * @param y Receives the sums. It may be `a` or `b`.
 * @param n Values in each array.
 */
void add(const float *a, const float *b, float *y, int n);

#endif
