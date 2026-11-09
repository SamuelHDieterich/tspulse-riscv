/**
 * @file
 * The last 513 rows of a series, kept per channel and handed out standardized.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Internal
#include "model/history.h"
#include "model/sizes.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

int history_init(
    struct history *history,
    uint32_t channels,
    const double *mean,
    const double *std
) {
    history->rings = malloc((size_t)channels * WINDOW_LENGTH * sizeof(double));
    history->mean = mean;
    history->std = std;
    history->channels = channels;
    history->pushed = 0;
    return history->rings == NULL;
}

void history_push(struct history *history, const double *row) {
    size_t slot = history->pushed % WINDOW_LENGTH;
    for (uint32_t c = 0; c < history->channels; c++) {
        history->rings[(size_t)c * WINDOW_LENGTH + slot] = row[c];
    }
    history->pushed++;
}

int history_full(const struct history *history) {
    return history->pushed >= WINDOW_LENGTH;
}

void history_window(
    const struct history *history, uint32_t channel, float out[WINDOW_LENGTH]
) {
    const double *ring = history->rings + (size_t)channel * WINDOW_LENGTH;

    // The next row to be overwritten is the oldest one
    size_t oldest = history->pushed % WINDOW_LENGTH;

    // z = (x - mean) / std, in double and rounded once to float
    // This is what scikit-learn's StandardScaler.transform does in float64 before the dataset casts to float32
    // https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    double mean = history->mean[channel];
    double std = history->std[channel];
    for (size_t i = 0; i < WINDOW_LENGTH; i++) {
        double raw = ring[(oldest + i) % WINDOW_LENGTH];
        out[i] = (float)((raw - mean) / std);
    }
}

void history_free(struct history *history) {
    free(history->rings);
    history->rings = NULL;
}
