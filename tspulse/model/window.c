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
#include "kernels/sum.h"
#include "model/history.h"
#include "model/pass.h"
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

/// The window's row the forecast is scored against: the one right after the past.
#define FORECAST_TARGET_ROW CONTEXT_LENGTH

/// The forecast row that is scored: the first, which predicts the target row.
#define SCORED_FORECAST_ROW 0

/// Dimensions of an error record, which holds one value.
#define ERROR_NDIM 1

/// Values in an error record.
#define ERROR_LENGTH 1

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Stores the squared errors of the patch a hidden pass hid, where the stitched reconstruction takes that patch from.
 *
 * @param input The channel's standardized window, of which rows 416 to 511 are scored.
 * @param reconstruction The pass's reconstruction, 512 values.
 * @param hidden_patch The patch the pass hid, from 0 to 11.
 * @param squares Receives the 8 squared errors of that patch, at offset `8 * hidden_patch` of the 96 scored rows.
 */
static void stitch_squares(
    const float input[WINDOW_LENGTH],
    const float reconstruction[CONTEXT_LENGTH],
    int hidden_patch,
    float squares[HIDDEN_ROWS]
) {
    int first = hidden_patch * PATCH_LENGTH;
    for (int j = first; j < first + PATCH_LENGTH; j++) {
        float difference =
            input[FIRST_HIDDEN_ROW + j] - reconstruction[FIRST_HIDDEN_ROW + j];
        squares[j] = difference * difference;
    }
}

#ifdef TSPULSE_DUMP
/**
 * Writes the window's three errors as whole `(1,)` records, after every channel's records.
 *
 * @param dump The open writer.
 * @param errors The window's errors.
 * @return 0 on success, nonzero if a write failed.
 */
static int
write_errors(struct tdump *dump, const struct window_errors *errors) {
    int failed = 0;
    failed |= tdump_write(
        dump, "error.time", &errors->time, ERROR_NDIM, DIMS(ERROR_LENGTH)
    );
    failed |= tdump_write(
        dump, "error.fft", &errors->fft, ERROR_NDIM, DIMS(ERROR_LENGTH)
    );
    failed |= tdump_write(
        dump, "error.forecast", &errors->forecast, ERROR_NDIM,
        DIMS(ERROR_LENGTH)
    );
    return failed;
}
#endif

// The squared error of each scored value, averaged over the scored values of every channel, as `nn.MSELoss` followed by `torch.mean`:
// time = sum_c sum_{t=416}^{511} (x[t, c] - stitched_time[t, c])^2 / (96 * C)
// fft = sum_c sum_{t=416}^{511} (x[t, c] - stitched_fft[t, c])^2 / (96 * C)
// forecast = sum_c (x[512, c] - forecast[0, c])^2 / C
// x is the standardized window, and both reconstructions and the forecast are back in its units after the inverse RevIN
// https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
int window_run(
    const struct history *history,
    struct tdump *dump,
    struct window_errors *errors
) {
    // Working memory is sized for one channel, so it doesn't grow with the channel count
    static float input[WINDOW_LENGTH];
    static struct pass_output output;
    static float time_squares[HIDDEN_ROWS];
    static float fft_squares[HIDDEN_ROWS];
    float time_total = 0.0f;
    float fft_total = 0.0f;
    float forecast_total = 0.0f;

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

        // Step 3: the plain pass, which scores the forecast, then one pass per hidden patch, each scoring the patch it hid
        for (int copy = 0; copy < COPIES; copy++) {
            int hidden_patch = copy == 0 ? NO_HIDDEN_PATCH : copy - 1;
            if (hidden_patch == NO_HIDDEN_PATCH) {
                DUMP_PUSH(PLAIN_COPY);
            } else {
                DUMP_PUSH(HIDDEN_COPY, hidden_patch);
            }
            pass(&TSPULSE_WEIGHTS, input, hidden_patch, &output);
            DUMP_POP();

            if (hidden_patch == NO_HIDDEN_PATCH) {
                float difference = input[FORECAST_TARGET_ROW] -
                                   output.forecast[SCORED_FORECAST_ROW];
                forecast_total += difference * difference;
            } else {
                stitch_squares(
                    input, output.reconstruction, hidden_patch, time_squares
                );
                stitch_squares(
                    input, output.reconstruction_from_fft, hidden_patch,
                    fft_squares
                );
            }
        }

        // Step 4: the channel's share of the reconstruction errors, summed pairwise within a channel, then added channel by channel
        time_total += pairwise_sum(time_squares, HIDDEN_ROWS);
        fft_total += pairwise_sum(fft_squares, HIDDEN_ROWS);
    }

    // Step 5: the means over every scored value
    float scored = (float)(HIDDEN_ROWS * history->channels);
    errors->time = time_total / scored;
    errors->fft = fft_total / scored;
    errors->forecast = forecast_total / (float)history->channels;

#ifdef TSPULSE_DUMP
    int failed = dump_failed();
    if (dump != NULL) {
        failed |= write_errors(dump, errors);
    }
    return failed;
#else
    return 0;
#endif
}
