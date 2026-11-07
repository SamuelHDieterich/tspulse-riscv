/**
 * @file
 * The matrix multiply shapes of one TSPulse pass.
 */

#ifndef SHAPES_H
#define SHAPES_H

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/** One Linear call's sizes, and how often one pass makes it. */
struct shape {
    int rows;  ///< The input rows.
    int in;    ///< The layer's input size.
    int out;   ///< The layer's output size.
    int calls; ///< How many times one pass makes this call.
};

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/** tspulse-r1's Linear calls for one pass of one channel. */
// clang-format off
static const struct shape SHAPES[] = {
// rows in   out  calls
  {24,  138, 276, 10},
  {24,  276, 138, 10},
  {24,  138, 138, 10},
  {138, 24,  48,  10},
  {138, 48,  24,  10},
  {138, 24,  24,  10},
  {10,  24,  48,  6},
  {10,  48,  24,  6},
  {1,   240, 256, 1},
  {10,  24,  24,  6},
  {24,  10,  20,  6},
  {24,  20,  10,  6},
  {64,  8,   24,  2},
  {64,  24,  8,   2},
  {24,  10,  10,  6},
  {1,   240, 16,  1},
};
// clang-format on

#define SHAPE_COUNT ((int)(sizeof SHAPES / sizeof SHAPES[0]))

// The largest sizes in SHAPES, which size the static buffers
#define MAX_ROWS 138
#define MAX_IN 276
#define MAX_OUT 276

#endif
