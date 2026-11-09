/**
 * @file
 * PyTorch's LayerNorm over the last dimension, with a learned scale and shift.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// Internal
#include "kernels/layer_norm.h"
#include "kernels/sum.h"
#include "model/config.h"
#include "model/sizes.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// mean = sum_i x[i] / n
// var = sum_i (x[i] - mean)^2 / n, the biased variance, as PyTorch uses
// y[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
// https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
void layer_norm(
    const float *x,
    const float *weight,
    const float *bias,
    float *y,
    int rows,
    int features
) {
    // The squares are stored first, so they can be added pairwise
    static float squares[FLAT_LENGTH];

    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * features;
        float *yr = y + r * features;

        // Step 1: the mean
        float mean = pairwise_sum(xr, features) / (float)features;

        // Step 2: the variance, in a second pass so no large squares are subtracted
        for (int i = 0; i < features; i++) {
            float centered = xr[i] - mean;
            squares[i] = centered * centered;
        }
        float spread =
            sqrtf(pairwise_sum(squares, features) / (float)features + NORM_EPS);

        // Step 3: normalize, scale and shift. y may be x, and each value is read before it is written
        for (int i = 0; i < features; i++) {
            yr[i] = (xr[i] - mean) / spread * weight[i] + bias[i];
        }
    }
}
