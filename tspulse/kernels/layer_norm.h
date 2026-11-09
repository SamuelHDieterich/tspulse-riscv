/**
 * @file
 * PyTorch's LayerNorm over the last dimension, with a learned scale and shift.
 */

#ifndef LAYER_NORM_H
#define LAYER_NORM_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Normalizes each row of `x` to zero mean and unit variance, then scales and shifts it.
 *
 * @param x The input, `rows` rows of `features` values.
 * @param weight The scale, `features` values.
 * @param bias The shift, `features` values.
 * @param y Receives the normalized rows. It may be `x` itself.
 * @param rows Rows in `x`.
 * @param features Values per row, at most 3312.
 */
void layer_norm(
    const float *x,
    const float *weight,
    const float *bias,
    float *y,
    int rows,
    int features
);

#endif
