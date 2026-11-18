/**
 * @file
 * PyTorch's tanh form of GELU, which differs from the erf form by up to 4.7e-4.
 */

#ifndef GELU_TANH_H
#define GELU_TANH_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
// Always the fine exp, so `GELU=tanh` measures the formula alone, whatever `EXP` is
#include "kernels/exp/fine.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░░▀░░▀▀▀

/// 2 * sqrt(2 / pi), the scale inside the sigmoid.
#define GELU_TANH_SCALE 1.5957691216f

/// Coefficient of the cubic term.
#define GELU_TANH_CUBIC 0.044715f

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * GELU(x) = x * sigmoid(2 * sqrt(2 / pi) * (x + 0.044715 * x^3)).
 *
 * That is PyTorch's `approximate='tanh'`, 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))), rewritten through tanh(u) = 2 * sigmoid(2u) - 1.
 * https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
 * https://en.wikipedia.org/wiki/Sigmoid_function
 *
 * @param x The input.
 * @return GELU(x) by the tanh form.
 */
static inline float gelu_tanh(float x) {
    // Step 1: the sigmoid's argument
    float u = GELU_TANH_SCALE * (x + GELU_TANH_CUBIC * x * x * x);

    // Step 2: sigmoid(u) = 1 / (1 + e^-u), where exp_fine clamps a large |u|
    return x / (1.0f + exp_fine(-u));
}

#endif
