/**
 * @file
 * TSPulse's RevIN scaler: normalizes a series over its observed rows only, then applies a learned affine.
 */

#ifndef REVIN_H
#define REVIN_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Normalizes a series by the mean and spread of its observed rows.
 *
 * Hidden rows keep their value, the mask token, unnormalized, but the affine applies to every row.
 *
 * @param x The series, `n` values.
 * @param observed 1.0 where a row is observed, 0.0 where it is hidden, `n` values.
 * @param affine_weight The learned scale applied after normalizing.
 * @param affine_bias The learned shift applied after normalizing.
 * @param y Receives the normalized series, `n` values. It must not overlap `x`.
 * @param loc Receives the mean of the observed rows.
 * @param scale Receives their spread, at least the square root of `minimum_scale`.
 * @param n Values in the series, at most 512.
 */
void revin(
    const float *x,
    const float *observed,
    float affine_weight,
    float affine_bias,
    float *y,
    float *loc,
    float *scale,
    int n
);

/**
 * Undoes `revin`: removes the affine, then restores the mean and spread.
 *
 * @param x The normalized series, `n` values.
 * @param affine_weight The learned scale `revin` applied.
 * @param affine_bias The learned shift `revin` applied.
 * @param loc The mean `revin` returned.
 * @param scale The spread `revin` returned.
 * @param y Receives the series in its original units. It may be `x` itself.
 * @param n Values in the series.
 */
void revin_inverse(
    const float *x,
    float affine_weight,
    float affine_bias,
    float loc,
    float scale,
    float *y,
    int n
);

#endif
