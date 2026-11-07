/**
 * @file
 * A 32-bit xorshift generator, mapped to floats.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "random.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Advances Marsaglia's 13/17/5 xorshift, which cycles through every non-zero 32-bit state.
 *
 * @param state The state, advanced in place.
 * @return The new state.
 */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

void fill_random(float *buffer, int count, uint32_t *state) {
    for (int i = 0; i < count; i++) {
        // The top 24 bits fit a float's mantissa exactly, so every build rounds them the same way
        buffer[i] =
            (float)(xorshift32(state) >> 8) * (1.0f / 8388608.0f) - 1.0f;
    }
}
