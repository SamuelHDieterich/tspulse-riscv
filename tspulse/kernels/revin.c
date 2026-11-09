/**
 * @file
 * TSPulse's RevIN scaler: normalizes a series over its observed rows only, then applies a learned affine.
 *
 * Built without fused multiply-add, because PyTorch rounds RevIN's multiply and add separately and fused they push the FFT's base component on YAHOO w908 past its allowance.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// Internal
#include "kernels/revin.h"
#include "kernels/sum.h"
#include "model/config.h"
#include "model/sizes.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// With m[t] = 1 for observed rows and 0 for hidden ones:
// count = max(sum_t m[t], 1)
// loc = sum_t x[t] * m[t] / count
// var = sum_t ((x[t] - loc) * m[t])^2 / count
// scale = sqrt(max(var, minimum_scale))
// y[t] = ((x[t] - loc) / scale if m[t] else x[t]) * affine_weight + affine_bias
// Reversible instance normalization, Kim et al., ICLR 2022: https://openreview.net/forum?id=cGDAkQo1C0p
void revin(
    const float *x,
    const float *observed,
    float affine_weight,
    float affine_bias,
    float *y,
    float *loc,
    float *scale,
    int n
) {
    // The terms of each sum are stored first, so they can be added pairwise
    static float terms[CONTEXT_LENGTH];

    // Step 1: the mean of the observed rows. The count is at least 1, so a fully hidden series doesn't divide by zero
    float count = fmaxf(pairwise_sum(observed, n), 1.0f);
    for (int t = 0; t < n; t++) {
        terms[t] = x[t] * observed[t];
    }
    float mean = pairwise_sum(terms, n) / count;

    // Step 2: their variance, floored so a flat series isn't divided by zero
    for (int t = 0; t < n; t++) {
        float centered = (x[t] - mean) * observed[t];
        terms[t] = centered * centered;
    }
    float spread = sqrtf(fmaxf(pairwise_sum(terms, n) / count, MINIMUM_SCALE));

    // Step 3: normalize the observed rows only. The hidden rows hold the learned mask token, which the model expects as it is
    for (int t = 0; t < n; t++) {
        float normalized = observed[t] != 0.0f ? (x[t] - mean) / spread : x[t];
        y[t] = normalized * affine_weight + affine_bias;
    }

    *loc = mean;
    *scale = spread;
}

// y[t] = (x[t] - affine_bias) / affine_weight * scale + loc
// Reversible instance normalization, Kim et al., ICLR 2022: https://openreview.net/forum?id=cGDAkQo1C0p
void revin_inverse(
    const float *x,
    float affine_weight,
    float affine_bias,
    float loc,
    float scale,
    float *y,
    int n
) {
    for (int t = 0; t < n; t++) {
        y[t] = (x[t] - affine_bias) / affine_weight * scale + loc;
    }
}
