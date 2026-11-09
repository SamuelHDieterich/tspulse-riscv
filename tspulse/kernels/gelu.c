/**
 * @file
 * PyTorch's GELU in its exact form, through the error function.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// Internal
#include "kernels/gelu.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))), the default `approximate='none'`
// https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
void gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        // M_SQRT1_2 is a double, so it is cast to keep the product in float
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * (float)M_SQRT1_2));
    }
}
