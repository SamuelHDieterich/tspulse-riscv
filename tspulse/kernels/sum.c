/**
 * @file
 * Pairwise summation: sums an array by halves, so rounding errors grow with log n instead of n.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "kernels/sum.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Blocks of at most 8 values are added in order. That adds at most 7 roundings per block, which keeps the error close to pure pairwise summation, while the recursion stops before its calls cost more than the additions.
#define BLOCK 8

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// sum(x[0 .. n)) = sum(x[0 .. n/2)) + sum(x[n/2 .. n)), down to blocks of at most 8 added in order
// The error grows as O(eps * log n), against O(eps * n) for a plain running sum
// https://en.wikipedia.org/wiki/Pairwise_summation
float pairwise_sum(const float *x, int n) {
    if (n <= BLOCK) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += x[i];
        }
        return sum;
    }
    int half = n / 2;
    return pairwise_sum(x, half) + pairwise_sum(x + half, n - half);
}
