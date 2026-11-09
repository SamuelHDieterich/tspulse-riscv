/**
 * @file
 * One window through the model: its input, and the three errors it yields.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>

// Internal
#include "io/tdump.h"
#include "model/history.h"
#include "model/sizes.h"
#include "model/window.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

#define WINDOW_NDIM 2
#define CHANNEL_AXIS (WINDOW_NDIM - 1)

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

int window_run(
    const struct history *history,
    struct tdump *dump,
    struct window_errors *errors
) {
    // Working memory is sized for one channel, so it doesn't grow with the channel count
    static float input[WINDOW_LENGTH];

    errors->time = 0.0f;
    errors->fft = 0.0f;
    errors->forecast = 0.0f;

    // The clean build has no dump code
#ifndef TSPULSE_DUMP
    (void)dump;
#endif

    for (uint32_t channel = 0; channel < history->channels; channel++) {
        // Step 1: the channel's standardized window
        history_window(history, channel, input);

#ifdef TSPULSE_DUMP
        // Step 2: the dump, which takes the channels one at a time and puts them last, as the reference's (513, C) record does
        if (dump != NULL) {
            const uint32_t dims[WINDOW_NDIM] = {
                WINDOW_LENGTH, history->channels
            };
            tdump_begin_channel(dump, channel, history->channels);
            if (tdump_write_channel(
                    dump, "input.window", input, WINDOW_NDIM, dims, CHANNEL_AXIS
                )) {
                return 1;
            }
        }
#endif
    }
    return 0;
}
