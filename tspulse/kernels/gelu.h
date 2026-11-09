/**
 * @file
 * PyTorch's GELU in its exact form, through the error function.
 */

#ifndef GELU_H
#define GELU_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Applies GELU to every value, in place.
 *
 * @param x The values, overwritten with their GELU.
 * @param n Values in `x`.
 */
void gelu(float *x, int n);

#endif
