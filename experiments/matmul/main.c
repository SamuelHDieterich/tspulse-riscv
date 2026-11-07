/**
 * @file
 * Runs one TSPulse pass's matrix multiplies and counts their instructions.
 *
 * Prints one line per shape and a total, in the format `spike-matmul` parses:
 *
 *     shape rows=24 in=138 out=276 calls=10 macs=9141120 instret=<n> abssum=<x>
 *     total macs=27186432 instret=<n>
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>
#include <stdio.h>

// Internal
#include "counter.h"
#include "matmul.h"
#include "random.h"
#include "shapes.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

// Every shape starts from this seed, so every build sees the same numbers
#define SEED 2026u

// ░█▀▄░█░█░█▀▀░█▀▀░█▀▀░█▀▄░█▀▀
// ░█▀▄░█░█░█▀▀░█▀▀░█▀▀░█▀▄░▀▀█
// ░▀▀░░▀▀▀░▀░░░▀░░░▀▀▀░▀░▀░▀▀▀

// Static and sized for the largest shape, so every shape reuses them
static float x[MAX_ROWS * MAX_IN];
static float w[MAX_OUT * MAX_IN];
static float b[MAX_OUT];
static float y[MAX_ROWS * MAX_OUT];

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Whether a shape fits the static buffers.
 *
 * @param shape The shape.
 * @return 1 if it fits, 0 otherwise.
 */
static int fits(const struct shape *shape) {
    return shape->rows <= MAX_ROWS && shape->in <= MAX_IN &&
           shape->out <= MAX_OUT;
}

/**
 * Sums the absolute values of a buffer.
 *
 * Absolute values can't cancel out to near zero, so a relative comparison against this sum stays meaningful.
 *
 * @param values The buffer.
 * @param count How many values to sum.
 * @return The sum, in double precision.
 */
static double abs_sum(const float *values, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += values[i] < 0.0f ? -values[i] : values[i];
    }
    return sum;
}

// ░█▄█░█▀█░▀█▀░█▀█
// ░█░█░█▀█░░█░░█░█
// ░▀░▀░▀░▀░▀▀▀░▀░▀

int main(void) {
    uint64_t total_macs = 0;
    uint64_t total_instret = 0;

    for (int s = 0; s < SHAPE_COUNT; s++) {
        const struct shape *shape = &SHAPES[s];

        // Step 1: refuse a shape that would overflow the static buffers
        if (!fits(shape)) {
            fprintf(
                stderr, "shape %d is larger than MAX_ROWS, MAX_IN or MAX_OUT\n",
                s
            );
            return 1;
        }

        // Step 2: fill the inputs from the same seed
        uint32_t state = SEED;
        // clang-format off
        fill_random(x, shape->rows  * shape->in,    &state);
        fill_random(w, shape->out   * shape->in,    &state);
        fill_random(b, shape->out,                  &state);
        // clang-format on

        // Step 3: count the instructions of this shape's calls, and nothing else
        uint64_t start = read_instret();
        for (int c = 0; c < shape->calls; c++) {
            linear(x, w, b, y, shape->rows, shape->in, shape->out);
            // Stops the compiler merging the repeated calls, even with LTO
            __asm__ volatile("" ::: "memory");
        }
        uint64_t instret = read_instret() - start;

        // Step 4: print the shape's line, with the sum the host build is checked against
        uint64_t macs =
            (uint64_t)shape->rows * shape->in * shape->out * shape->calls;
        total_macs += macs;
        total_instret += instret;

        printf(
            "shape rows=%d in=%d out=%d calls=%d macs=%llu instret=%llu "
            "abssum=%.9g\n",
            shape->rows, shape->in, shape->out, shape->calls,
            (unsigned long long)macs, (unsigned long long)instret,
            abs_sum(y, shape->rows * shape->out)
        );
    }

    // Step 5: print the whole pass's totals
    printf(
        "total macs=%llu instret=%llu\n", (unsigned long long)total_macs,
        (unsigned long long)total_instret
    );
    return 0;
}
