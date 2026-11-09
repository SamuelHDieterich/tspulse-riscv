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
#include "io/dump.h"
#include "io/tdump.h"
#include "model/history.h"
#include "model/sizes.h"
#include "model/window.h"
#include "modules/backbone.h"
#include "weights/weights.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The record path of the pass that hides nothing.
#define PLAIN_COPY "plain/"

/// The record path of the pass that hides patch `i`, from 00 to 11.
#define HIDDEN_COPY "hidden%02d/"

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
    static struct backbone_output output;

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

        // Step 2: the dump takes the channels one at a time, and every channel writes the same records in the same order
        DUMP_BEGIN_CHANNEL(dump, channel, history->channels);
        DUMP(
            "input.window", input, SERIES_NDIM,
            DIMS(WINDOW_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
        );

        // Step 3: the plain pass, then one pass per hidden patch
        for (int copy = 0; copy < COPIES; copy++) {
            int hidden_patch = copy == 0 ? NO_HIDDEN_PATCH : copy - 1;
            if (hidden_patch == NO_HIDDEN_PATCH) {
                DUMP_PUSH(PLAIN_COPY);
            } else {
                DUMP_PUSH(HIDDEN_COPY, hidden_patch);
            }
            DUMP_PUSH("backbone");
            backbone(&TSPULSE_WEIGHTS, input, hidden_patch, &output);
            DUMP_POP();
            DUMP_POP();
        }
    }

#ifdef TSPULSE_DUMP
    return dump_failed();
#else
    return 0;
#endif
}
