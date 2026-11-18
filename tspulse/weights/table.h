/**
 * @file
 * A name, pointer and shape for every weight, for verification programs only.
 *
 * The model program reads the weights through `TSPULSE_WEIGHTS` and leaves the table out of the build.
 */

#ifndef TABLE_H
#define TABLE_H

// Built-in
#include <stdint.h>

// Internal
#include "weights/quant.h"

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// One weight tensor.
/// In an int8 build, a quantized weight has `data` NULL and fills `q` and `scale`.
struct weight_entry {
    const char *name;  ///< The PyTorch tensor name.
    const float *data; ///< The float32 values, row-major, else NULL.
    int ndim;          ///< The number of dimensions.
    uint32_t dims[4];  ///< The size of each dimension.
#ifdef TSPULSE_QUANT_INT8
    const int8_t *q;    ///< A quantized weight's int8 values, else NULL.
    const float *scale; ///< Its scale per output row, else NULL.
#endif
};

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Every weight, in the order of the safetensors data.
extern const struct weight_entry WEIGHT_TABLE[];

/// The number of entries in `WEIGHT_TABLE`.
extern const int WEIGHT_COUNT;

#endif
