/**
 * @file
 * Writes every built-in weight to a `.tdump` file, to compare against the reference export-weights writes.
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
#include "weights/table.h"

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
        if (tdump_write(
                &dump, entry->name, entry->data, entry->ndim, entry->dims
            )) {
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
