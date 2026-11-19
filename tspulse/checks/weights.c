/**
 * @file
 * Writes every built-in weight to a `.tdump` file, to compare against the reference export-weights writes.
 *
 * In a `QUANT` build, a quantized weight is written as the float values the model uses, `scale[o] * (float)q[o][i]`, which `weights_int8.tdump` holds too.
 *
 * Usage: check-weights OUTPUT
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdio.h>

// Internal
#include "io/tdump.h"
#include "model/sizes.h"
#include "weights/quant.h"
#include "weights/table.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

#ifdef TSPULSE_QUANT_INT8
/// The largest quantized weight: the patch MLP's fc1, 276 rows of 138, or fc2, 138 rows of 276.
#define DEQUANTIZED_CAPACITY (PATCH_MLP_HIDDEN * TOTAL_PATCHES)
#endif

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

#ifdef TSPULSE_QUANT_INT8
/**
 * Turns a quantized weight back into the float values the model uses.
 *
 * Each value is `scale[o] * (float)q[o][i]`, the same float product the kernel computes, so the dump matches the export's bit for bit only when both read the same int8 values and scales.
 *
 * @param entry A table entry with `q` and `scale`, of 2 dimensions, rows first.
 * @return The values, valid until the next call, or NULL when the tensor is not 2-D or holds more than `DEQUANTIZED_CAPACITY` values.
 */
static const float *dequantize(const struct weight_entry *entry) {
    static float values[DEQUANTIZED_CAPACITY];
    uint32_t rows = entry->dims[0];
    uint32_t columns = entry->dims[1];
    if (entry->ndim != 2 || rows * columns > DEQUANTIZED_CAPACITY) {
        return NULL;
    }
    for (uint32_t o = 0; o < rows; o++) {
        for (uint32_t i = 0; i < columns; i++) {
            values[o * columns + i] =
                entry->scale[o] * (float)entry->q[o * columns + i];
        }
    }
    return values;
}
#endif

// ░█▄█░█▀█░▀█▀░█▀█
// ░█░█░█▀█░░█░░█░█
// ░▀░▀░▀░▀░▀▀▀░▀░▀

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 1;
    }

    struct tdump dump;
    if (tdump_open(&dump, argv[1])) {
        fprintf(stderr, "cannot write %s\n", argv[1]);
        return 1;
    }

    for (int i = 0; i < WEIGHT_COUNT; i++) {
        const struct weight_entry *entry = &WEIGHT_TABLE[i];
        const float *values = entry->data;
#ifdef TSPULSE_QUANT_INT8
        if (entry->q != NULL) {
            values = dequantize(entry);
            if (values == NULL) {
                fprintf(
                    stderr,
                    "%s is not a 2-D tensor that fits the dequantizing "
                    "buffer\n",
                    entry->name
                );
                tdump_close(&dump);
                return 1;
            }
        }
#endif
        if (tdump_write(&dump, entry->name, values, entry->ndim, entry->dims)) {
            fprintf(stderr, "cannot write %s to %s\n", entry->name, argv[1]);
            tdump_close(&dump);
            return 1;
        }
    }

    if (tdump_close(&dump)) {
        fprintf(stderr, "cannot finish %s\n", argv[1]);
        return 1;
    }
    printf("wrote %d tensors\n", WEIGHT_COUNT);
    return 0;
}
