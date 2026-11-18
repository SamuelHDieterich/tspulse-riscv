/**
 * @file
 * GELU from a table over [-8, 8] and linear interpolation, about 1e-4 absolute.
 */

#ifndef GELU_TABLE_H
#define GELU_TABLE_H

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░░▀░░▀▀▀

/// Lowest input in the table; below it GELU is under 1e-14 and the result is 0.
#define GELU_TABLE_LOWEST -8.0f

/// Highest input in the table; from it up GELU(x) is x to within 1e-15.
#define GELU_TABLE_HIGHEST 8.0f

/// Nodes per unit of x, so the step is 1/32.
#define GELU_TABLE_PER_UNIT 32.0f

/// Nodes from -8 to 8 inclusive: 16 * 32 + 1.
#define GELU_TABLE_NODES 513

/// The last node, GELU(8).
#define GELU_TABLE_LAST (GELU_TABLE_NODES - 1)

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/// GELU at each node, x = -8 + i/32. Filled by `gelu_table_fill`.
extern float gelu_table_nodes[GELU_TABLE_NODES];

/**
 * Fills the table from the program's own `erff`. Call it once, before `gelu_table`.
 */
void gelu_table_fill(void);

/**
 * GELU on [-8, 8] by linear interpolation between neighbouring nodes.
 *
 * The error of interpolating at step h is at most h^2/8 * max|GELU''|, which is about 1e-4 for h = 1/32.
 * https://en.wikipedia.org/wiki/Linear_interpolation#Interpolation_error:_Linear_interpolation_error
 *
 * @param x The input.
 * @return GELU(x), 0 below -8 and x from 8 up.
 */
static inline float gelu_table(float x) {
    // Step 1: the ends, where the interpolation below would read past the nodes; NaN lands in the first branch
    if (!(x >= GELU_TABLE_LOWEST)) {
        return 0.0f;
    }
    if (x >= GELU_TABLE_HIGHEST) {
        return x;
    }

    // Step 2: the node at or below x, kept one under the last so the next one exists, because x + 8 rounds up to 16 for the float just below 8
    float position = (x - GELU_TABLE_LOWEST) * GELU_TABLE_PER_UNIT;
    int node = (int)position;
    if (node > GELU_TABLE_LAST - 1) {
        node = GELU_TABLE_LAST - 1;
    }

    // Step 3: interpolate between the node and the next
    float fraction = position - (float)node;
    float low = gelu_table_nodes[node];
    float high = gelu_table_nodes[node + 1];
    return low + fraction * (high - low);
}

#endif
