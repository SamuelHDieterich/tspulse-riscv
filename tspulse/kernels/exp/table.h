/**
 * @file
 * The exponential from a table over [-16, 0] and linear interpolation, about 3e-5 relative.
 */

#ifndef EXP_TABLE_H
#define EXP_TABLE_H

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Lowest input in the table; below it e^x is under 1.2e-7 and the result is 0.
#define EXP_TABLE_LOWEST -16.0f

/// Nodes per unit of x, so the step is 1/64.
#define EXP_TABLE_PER_UNIT 64.0f

/// Nodes from -16 to 0 inclusive: 16 * 64 + 1.
#define EXP_TABLE_NODES 1025

/// The last node, e^0.
#define EXP_TABLE_LAST (EXP_TABLE_NODES - 1)

/// Softmax subtracts the row's maximum, so its inputs never go above this.
#define EXP_TABLE_HIGHEST 0.0f

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/// e^x at each node, x = -16 + i/64. Filled by `exp_table_fill`.
extern float exp_table_nodes[EXP_TABLE_NODES];

/**
 * Fills the table from the program's own `expf`. Call it once, before `exp_table`.
 */
void exp_table_fill(void);

/**
 * e^x on [-16, 0] by linear interpolation between neighbouring nodes.
 *
 * The error of interpolating at step h is at most h^2/8 * max|f''|, which is 3e-5 relative for h = 1/64.
 * https://en.wikipedia.org/wiki/Linear_interpolation#Interpolation_error:_Linear_interpolation_error
 *
 * @param x The exponent, at most 0.
 * @return e^x, and 0 below -16.
 */
static inline float exp_table(float x) {
    // Step 1: the ends, where the interpolation below would read past the nodes; NaN lands in the first branch
    if (!(x >= EXP_TABLE_LOWEST)) {
        return 0.0f;
    }
    if (x >= EXP_TABLE_HIGHEST) {
        return exp_table_nodes[EXP_TABLE_LAST];
    }

    // Step 2: the node at or below x, kept one under the last so the next one exists, because x + 16 rounds up to 16 in float for an x just below 0 (within about 4.8e-7), which would put the index on the last node
    float position = (x - EXP_TABLE_LOWEST) * EXP_TABLE_PER_UNIT;
    int node = (int)position;
    if (node > EXP_TABLE_LAST - 1) {
        node = EXP_TABLE_LAST - 1;
    }

    // Step 3: interpolate between the node and the next
    float fraction = position - (float)node;
    float low = exp_table_nodes[node];
    float high = exp_table_nodes[node + 1];
    return low + fraction * (high - low);
}

#endif
