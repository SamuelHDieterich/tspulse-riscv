/**
 * @file
 * PyTorch's Softmax over the last dimension.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// Internal
#include "kernels/softmax.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// softmax(x)[i] = exp(x[i] - max_j x[j]) / sum_k exp(x[k] - max_j x[j])
// Subtracting the maximum leaves the result unchanged and keeps exp from overflowing
// https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html
void softmax(float *x, int rows, int features) {
    for (int r = 0; r < rows; r++) {
        float *xr = x + r * features;

        // Step 1: the maximum
        float max = xr[0];
        for (int i = 1; i < features; i++) {
            max = fmaxf(max, xr[i]);
        }

        // Step 2: the exponentials and their sum
        float sum = 0.0f;
        for (int i = 0; i < features; i++) {
            xr[i] = expf(xr[i] - max);
            sum += xr[i];
        }

        // Step 3: divide by the sum
        for (int i = 0; i < features; i++) {
            xr[i] /= sum;
        }
    }
}
