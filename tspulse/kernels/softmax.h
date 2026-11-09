/**
 * @file
 * PyTorch's Softmax over the last dimension.
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Turns each row of `x` into probabilities that sum to 1, in place.
 *
 * @param x The input, `rows` rows of `features` values, overwritten with the probabilities.
 * @param rows Rows in `x`.
 * @param features Values per row.
 */
void softmax(float *x, int rows, int features);

#endif
