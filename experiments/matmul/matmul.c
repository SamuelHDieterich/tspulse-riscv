/**
 * @file
 * A direct port of PyTorch's Linear, with no loop reordering or blocking.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "matmul.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

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

            // Float addition isn't associative, so GCC can only vectorize this as an ordered reduction (vfredosum on RVV)
            for (int i = 0; i < in; i++) {
                acc += xr[i] * wo[i];
            }
            y[r * out + o] = acc;
        }
    }
}
