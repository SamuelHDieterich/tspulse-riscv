/**
 * @file
 * Pairwise summation: sums an array by halves, so rounding errors grow with log n instead of n.
 */

#ifndef SUM_H
#define SUM_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Sums `n` values by splitting them into halves until a block is small enough to add in order.
 *
 * @param x The values.
 * @param n Values in `x`.
 * @return Their sum.
 */
float pairwise_sum(const float *x, int n);

#endif
