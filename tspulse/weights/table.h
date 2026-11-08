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

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// One weight tensor.
struct weight_entry {
    const char *name;  ///< The PyTorch tensor name.
    const float *data; ///< The values, row-major.
    int ndim;          ///< The number of dimensions.
    uint32_t dims[4];  ///< The size of each dimension.
};

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Every weight, in the order of the safetensors data.
extern const struct weight_entry WEIGHT_TABLE[];

/// The number of entries in `WEIGHT_TABLE`.
extern const int WEIGHT_COUNT;

#endif
