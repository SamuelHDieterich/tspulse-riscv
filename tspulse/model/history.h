/**
 * @file
 * The last 513 rows of a series, kept per channel and handed out standardized.
 */

#ifndef HISTORY_H
#define HISTORY_H

// Built-in
#include <stddef.h>
#include <stdint.h>

// Internal
#include "model/sizes.h"

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// A ring of the most recent raw rows, one ring per channel.
struct history {
    // clang-format off: one comment column for the whole struct, which clang-format splits at each line without a trailing comment
    double *rings;     ///< `channels` rings of `WINDOW_LENGTH` raw values each.
    /// Per-channel average to subtract, owned by the caller.
    const double *mean;
    /// Per-channel spread to divide by, owned by the caller.
    const double *std;
    uint32_t channels; ///< Values per row.
    uint64_t pushed;   ///< Rows pushed so far.
    // clang-format on
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Allocates the rings, which start empty.
 *
 * @param history The history to set up.
 * @param channels Values per row.
 * @param mean Per-channel average, which must outlive the history.
 * @param std Per-channel spread, which must outlive the history.
 * @return 0 on success, nonzero if the memory is not available.
 */
int history_init(
    struct history *history,
    uint32_t channels,
    const double *mean,
    const double *std
);

/**
 * Appends one raw row, dropping the oldest once 513 are held.
 *
 * @param history An initialized history.
 * @param row `channels` raw values.
 */
void history_push(struct history *history, const double *row);

/**
 * Whether 513 rows have been pushed, so a window can be read.
 *
 * @param history An initialized history.
 * @return Nonzero when `history_window` may be called.
 */
int history_full(const struct history *history);

/**
 * Standardizes one channel of the last 513 rows, oldest first.
 *
 * @param history A full history.
 * @param channel The channel to read.
 * @param out Receives `(row - mean) / std` for each row, rounded to float.
 */
void history_window(
    const struct history *history, uint32_t channel, float out[WINDOW_LENGTH]
);

/**
 * Frees the rings.
 *
 * @param history An initialized history.
 */
void history_free(struct history *history);

#endif
