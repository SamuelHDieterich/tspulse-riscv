/**
 * @file
 * One window through the model: its input, and the three errors it yields.
 */

#ifndef WINDOW_H
#define WINDOW_H

// Internal
#include "io/tdump.h"
#include "model/history.h"

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// The errors of one window, one per head.
struct window_errors {
    /// Mean squared error of the stitched time reconstruction over offsets 416 to 511 and every channel.
    float time;
    /// Mean squared error of the stitched reconstruction through the FFT over the same offsets and channels.
    float fft;
    /// Squared error of the first forecast row against row 512, averaged over the channels.
    float forecast;
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Runs the model on the window the history holds, one channel at a time, and scores its three heads.
 *
 * Each channel goes through the plain pass and the 12 hidden passes before the next channel starts.
 * In the dump build, every channel's standardized input is written to `input.window`, a (513, channels) record, each pass writes its records under its copy's path, and the errors follow as `error.time`, `error.fft` and `error.forecast`, one value each.
 *
 * @param history A full history.
 * @param dump The dump to write into, or NULL to write nothing. The clean build never writes.
 * @param errors Receives the window's errors.
 * @return 0 on success, nonzero if a dump write failed.
 */
int window_run(
    const struct history *history,
    struct tdump *dump,
    struct window_errors *errors
);

#endif
