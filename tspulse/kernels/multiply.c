/**
 * @file
 * Element-wise product of two arrays, which the gated attention uses to weigh its input.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/multiply.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// y[i] = a[i] * b[i]
// https://docs.pytorch.org/docs/stable/generated/torch.mul.html
void multiply(const float *a, const float *b, float *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}
