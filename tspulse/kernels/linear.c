/**
 * @file
 * PyTorch's Linear: every row times the transposed weights, plus the bias.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/linear.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// y[r][o] = b[o] + sum_i x[r][i] * w[o][i]
// https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linear(
    const float *restrict x,
    const float *restrict w,
    const float *restrict b,
    float *restrict y,
    int rows,
    int in,
    int out
) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * in;
        for (int o = 0; o < out; o++) {
            // w is [out, in], so each output is a dot product over contiguous memory
            const float *wo = w + o * in;
            float acc = b[o];

            // Float addition isn't associative, so the sum runs in input order from the bias
            for (int i = 0; i < in; i++) {
                acc += xr[i] * wo[i];
            }
            y[r * out + o] = acc;
        }
    }
}
