/**
 * @file
 * One forward pass of one channel for one copy: the backbone, the decoder, and the heads that copy's error needs.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <string.h>

// Internal
#include "io/dump.h"
#include "kernels/revin.h"
#include "model/pass.h"
#include "model/sizes.h"
#include "modules/backbone.h"
#include "modules/decoder.h"
#include "modules/heads.h"
#include "weights/weights.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Rows of a per-channel statistic such as `loc`.
#define STATISTIC_ROWS 1

/// Where the decoder's time rows start: the first 64 of its 138 rows.
#define TIME_ROWS_OFFSET 0

/// Where its FFT rows start, right after the time rows.
#define FFT_ROWS_OFFSET (NUM_PATCHES * D_MODEL)

/// Where the register tokens start: the last 10 rows.
#define REGISTER_ROWS_OFFSET (ENCODED_PATCHES * D_MODEL)

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * The plain pass's head: the forecast from the decoder's register tokens, then back to the series' units.
 *
 * @param weights The model's weights.
 * @param decoded The decoder's output, 138 rows of 24 values.
 * @param model The backbone's output, for RevIN's statistics.
 * @param forecast Receives the 16 forecast rows in the series' units.
 * @param scaled_forecast Receives the same rows as RevIN's scaled values, before the inverse.
 */
static void forecast_head(
    const struct tspulse_weights *weights,
    const float decoded[FLAT_LENGTH],
    const struct backbone_output *model,
    float forecast[PREDICTION_LENGTH],
    float scaled_forecast[PREDICTION_LENGTH]
) {
    const float affine_weight = weights->backbone.scaler.affine_weight[0][0][0];
    const float affine_bias = weights->backbone.scaler.affine_bias[0][0][0];

    DUMP_PUSH("forecast_mapping");
    forecast_mapping(
        weights->decoder_with_head.forecast_mapping._0.mixers,
        weights->decoder_with_head.forecast_mapping._2.weight,
        weights->decoder_with_head.forecast_mapping._2.bias,
        weights->decoder_with_head.forecast_mapping._3.weight,
        weights->decoder_with_head.forecast_mapping._3.bias,
        decoded + REGISTER_ROWS_OFFSET, scaled_forecast
    );
    DUMP_POP();

    // The forecast is transposed to time first, which for one channel leaves it where it is
    DUMP(
        "forecast_output", scaled_forecast, SERIES_NDIM,
        DIMS(PREDICTION_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
    );
    revin_inverse(
        scaled_forecast, affine_weight, affine_bias, model->loc, model->scale,
        forecast, PREDICTION_LENGTH
    );
}

/**
 * A hidden pass's heads: the time reconstruction and the reconstruction through the FFT, then both back to the series' units.
 *
 * @param weights The model's weights.
 * @param decoded The decoder's output, 138 rows of 24 values.
 * @param model The backbone's output, for the FFT maxima and RevIN's statistics.
 * @param output Receives both reconstructions in the series' units.
 * @param fft_output Receives the FFT head's own output, the normalized bins.
 * @param scaled_time Receives the time reconstruction before the inverse RevIN.
 * @param scaled_fft Receives the reconstruction through the FFT before the inverse RevIN.
 */
static void reconstruction_heads(
    const struct tspulse_weights *weights,
    const float decoded[FLAT_LENGTH],
    const struct backbone_output *model,
    struct pass_output *output,
    float fft_output[CONTEXT_LENGTH],
    float scaled_time[CONTEXT_LENGTH],
    float scaled_fft[CONTEXT_LENGTH]
) {
    const float affine_weight = weights->backbone.scaler.affine_weight[0][0][0];
    const float affine_bias = weights->backbone.scaler.affine_bias[0][0][0];

    // Step 1: the FFT head on the decoder's FFT rows, then the time head on its time rows, in PyTorch's order
    DUMP_PUSH("fft_head");
    reconstruction_head(
        &weights->decoder_with_head.fft_head, decoded + FFT_ROWS_OFFSET,
        fft_output
    );
    DUMP_POP();
    DUMP_PUSH("time_head");
    reconstruction_head(
        &weights->decoder_with_head.time_head, decoded + TIME_ROWS_OFFSET,
        scaled_time
    );
    DUMP_POP();
    DUMP(
        "reconstruction_outputs", scaled_time, SERIES_NDIM,
        DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "reconstruction_fft_outputs", fft_output, SERIES_NDIM,
        DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
    );

    // Step 2: the FFT head's bins back to a series
    fft_reconstruction(
        fft_output, model->fft_base_component, model->fft_real_max,
        model->fft_imag_max, scaled_fft
    );
    DUMP(
        "reconstructed_ts_from_fft", scaled_fft, SERIES_NDIM,
        DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
    );

    // Step 3: both reconstructions back to the series' units
    revin_inverse(
        scaled_time, affine_weight, affine_bias, model->loc, model->scale,
        output->reconstruction, CONTEXT_LENGTH
    );
    revin_inverse(
        scaled_fft, affine_weight, affine_bias, model->loc, model->scale,
        output->reconstruction_from_fft, CONTEXT_LENGTH
    );
}

void pass(
    const struct tspulse_weights *weights,
    const float past[CONTEXT_LENGTH],
    int hidden_patch,
    struct pass_output *output
) {
    static struct backbone_output model;
    static float decoded[FLAT_LENGTH];
    static float scaled_forecast[PREDICTION_LENGTH];
    static float fft_output[CONTEXT_LENGTH];
    static float scaled_time[CONTEXT_LENGTH];
    static float scaled_fft[CONTEXT_LENGTH];
    int plain = hidden_patch == NO_HIDDEN_PATCH;

    // Step 1: the backbone
    DUMP_PUSH("backbone");
    backbone(weights, past, hidden_patch, &model);
    DUMP_POP();

    // Step 2: the decoder, on a copy, since the backbone's state is also an output field
    memcpy(decoded, model.last_hidden_state, sizeof(decoded));
    DUMP_PUSH("decoder_with_head");
    DUMP_PUSH("decoder");
    decoder(weights->decoder_with_head.decoder.decoder_block.mixers, decoded);
    DUMP_POP();

    // Step 3: only the heads this copy's error reads. The baseline skips the rest, and so does this pass
    if (plain) {
        forecast_head(
            weights, decoded, &model, output->forecast, scaled_forecast
        );
    } else {
        reconstruction_heads(
            weights, decoded, &model, output, fft_output, scaled_time,
            scaled_fft
        );
    }

    // Step 4: the decoder's whole output, flattened, which leaves it where it is
    DUMP(
        "decoder_hidden_state", decoded, FLAT_NDIM,
        DIMS(DUMP_CHANNELS, FLAT_LENGTH), CHANNEL_FIRST
    );
    DUMP_POP();

    // Step 5: the fields of `TSPulseForReconstructionOutput`
    DUMP_PUSH("output");
    if (plain) {
        DUMP(
            "forecast_output", output->forecast, SERIES_NDIM,
            DIMS(PREDICTION_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
        );
    } else {
        DUMP(
            "reconstruction_outputs", output->reconstruction, SERIES_NDIM,
            DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
        );
        DUMP(
            "fft_reconstruction_outputs", fft_output, SERIES_NDIM,
            DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
        );
        DUMP(
            "reconstructed_ts_from_fft", output->reconstruction_from_fft,
            SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
        );
    }
    DUMP(
        "backbone_hidden_state", model.last_hidden_state, FLAT_NDIM,
        DIMS(DUMP_CHANNELS, FLAT_LENGTH), CHANNEL_FIRST
    );
    DUMP(
        "masked_past_values", model.masked_past_values, SERIES_NDIM,
        DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "past_values", past, SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP(
        "decoder_hidden_state", decoded, FLAT_NDIM,
        DIMS(DUMP_CHANNELS, FLAT_LENGTH), CHANNEL_FIRST
    );
    DUMP(
        "mask", model.mask, SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP(
        "loc", &model.loc, SERIES_NDIM, DIMS(STATISTIC_ROWS, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP(
        "scale", &model.scale, SERIES_NDIM, DIMS(STATISTIC_ROWS, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP_POP();
}
