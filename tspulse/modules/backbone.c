/**
 * @file
 * The backbone, `TSPulseModel`: masking, scaling, the time and FFT embeddings, and the encoder.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// Internal
#include "io/dump.h"
#include "kernels/fft.h"
#include "kernels/revin.h"
#include "model/config.h"
#include "model/sizes.h"
#include "modules/backbone.h"
#include "modules/block.h"
#include "modules/embedding.h"
#include "weights/types.h"
#include "weights/weights.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Rows of a per-channel statistic such as `loc`.
#define STATISTIC_ROWS 1

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * The part of `get_fft` the baseline uses: the DFT, its last bin set aside, and each part divided by its largest magnitude.
 *
 * The magnitudes, their `log1p` and the softmax over them feed only a training loss, so they are not computed.
 *
 * @param scaled The scaled, masked series, 512 values.
 * @param features Receives the real parts of bins 0 to 255 divided by their maximum, then the imaginary parts divided by theirs.
 * @param output Receives the base component and the two maxima.
 */
static void get_fft(
    const float scaled[CONTEXT_LENGTH],
    float features[CONTEXT_LENGTH],
    struct backbone_output *output
) {
    static float real[FFT_BINS];
    static float imag[FFT_BINS];

    // Step 1: bins 0 to 256. The config's `fft_remove_component` is "last", so bin 256 is set aside as the base component
    fft_forward(scaled, real, imag);
    output->fft_base_component = real[FFT_KEPT_BINS];

    // Step 2: the largest magnitude of each part, plus `minimum_scale` so a flat series doesn't divide by zero
    // real_max = max_k |Re X[k]| + epsilon, imag_max = max_k |Im X[k]| + epsilon, over k = 0 to 255
    float real_max = 0.0f;
    float imag_max = 0.0f;
    for (int k = 0; k < FFT_KEPT_BINS; k++) {
        real_max = fmaxf(real_max, fabsf(real[k]));
        imag_max = fmaxf(imag_max, fabsf(imag[k]));
    }
    real_max += MINIMUM_SCALE;
    imag_max += MINIMUM_SCALE;
    output->fft_real_max = real_max;
    output->fft_imag_max = imag_max;

    // Step 3: the normalized real parts, then the imaginary parts
    for (int k = 0; k < FFT_KEPT_BINS; k++) {
        features[k] = real[k] / real_max;
        features[FFT_KEPT_BINS + k] = imag[k] / imag_max;
    }
}

void time_masker(
    const float mask_token[MASK_BLOCK_LENGTH],
    const float past[CONTEXT_LENGTH],
    int hidden_patch,
    float masked[CONTEXT_LENGTH],
    float mask[CONTEXT_LENGTH]
) {
    // Pass i hides rows 416 + 8i to 423 + 8i. The plain pass hides nothing, so its range is empty
    int first = CONTEXT_LENGTH;
    int last = CONTEXT_LENGTH;
    if (hidden_patch != NO_HIDDEN_PATCH) {
        first = FIRST_HIDDEN_ROW + hidden_patch * PATCH_LENGTH;
        last = first + PATCH_LENGTH;
    }

    for (int t = 0; t < CONTEXT_LENGTH; t++) {
        int hidden = t >= first && t < last;
        mask[t] = hidden ? 1.0f : 0.0f;
        masked[t] = hidden ? mask_token[t % MASK_BLOCK_LENGTH] : past[t];
    }

    // The module returns a tuple: the masked input, then the mask
    DUMP(
        "0", masked, SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP(
        "1", mask, SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS),
        CHANNEL_LAST
    );
}

void backbone(
    const struct tspulse_weights *weights,
    const float past[CONTEXT_LENGTH],
    int hidden_patch,
    struct backbone_output *output
) {
    static float observed[CONTEXT_LENGTH];
    static float scaled[CONTEXT_LENGTH];
    static float fft_features[CONTEXT_LENGTH];
    static float encoded[ENCODED_PATCHES * D_MODEL];
    float *hidden = output->last_hidden_state;

    // Step 1: hide the pass's patch behind the mask token
    DUMP_PUSH("time_masker");
    time_masker(
        weights->backbone.time_masker.mask_token, past, hidden_patch,
        output->masked_past_values, output->mask
    );
    DUMP_POP();

    // Step 2: RevIN over the rows left observed. Its tuple is the scaled series, the mean and the spread
    for (int t = 0; t < CONTEXT_LENGTH; t++) {
        observed[t] = 1.0f - output->mask[t];
    }
    revin(
        output->masked_past_values, observed,
        weights->backbone.scaler.affine_weight[0][0][0],
        weights->backbone.scaler.affine_bias[0][0][0], scaled, &output->loc,
        &output->scale, CONTEXT_LENGTH
    );
    DUMP(
        "scaler.0", scaled, SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP(
        "scaler.1", &output->loc, SERIES_NDIM,
        DIMS(STATISTIC_ROWS, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "scaler.2", &output->scale, SERIES_NDIM,
        DIMS(STATISTIC_ROWS, DUMP_CHANNELS), CHANNEL_LAST
    );

    // Step 3: the time embedding. Patching one channel's 512 values into 64 rows of 8 leaves them where they are
    DUMP(
        "patching", scaled, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, NUM_PATCHES, PATCH_LENGTH), CHANNEL_FIRST
    );
    DUMP_PUSH("time_encoding");
    patch_encoding(&weights->backbone.time_encoding, scaled, encoded);
    DUMP_POP();

    // Step 4: the FFT embedding, from the same scaled and masked series. The second run of `patching` is named `patching#2`
    get_fft(scaled, fft_features, output);
    DUMP(
        "patching#2", fft_features, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, NUM_PATCHES, PATCH_LENGTH), CHANNEL_FIRST
    );
    DUMP_PUSH("fft_encoding");
    patch_encoding(
        &weights->backbone.fft_encoding, fft_features,
        encoded + NUM_PATCHES * D_MODEL
    );
    DUMP_POP();

    // Step 5: the register tokens after the time and FFT patches, then the norm over all 3312 values
    DUMP_PUSH("add_tokens");
    add_tokens(weights->backbone.add_tokens.patch_tokens, encoded, hidden);
    DUMP_POP();
    DUMP_PUSH("base_norm");
    base_norm(
        weights->backbone.base_norm.weight, weights->backbone.base_norm.bias,
        hidden
    );
    DUMP_POP();

    // Step 6: the encoder
    DUMP_PUSH("encoder_block");
    tspulse_block(
        weights->backbone.encoder_block.mixers, ENCODER_LAYERS, hidden
    );
    DUMP_POP();

    // Step 7: the backbone's output fields. The flattened state holds the same values as the hidden state
    DUMP(
        "last_hidden_state", hidden, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, TOTAL_PATCHES, D_MODEL), CHANNEL_FIRST
    );
    DUMP(
        "last_hidden_flatten_state", hidden, FLAT_NDIM,
        DIMS(DUMP_CHANNELS, FLAT_LENGTH), CHANNEL_FIRST
    );
    DUMP(
        "masked_past_values", output->masked_past_values, SERIES_NDIM,
        DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "past_values_fft", fft_features, SERIES_NDIM,
        DIMS(CONTEXT_LENGTH, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "loc", &output->loc, SERIES_NDIM, DIMS(STATISTIC_ROWS, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP(
        "scale", &output->scale, SERIES_NDIM,
        DIMS(STATISTIC_ROWS, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "mask", output->mask, SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS),
        CHANNEL_LAST
    );
    DUMP(
        "fft_base_component.real", &output->fft_base_component, SERIES_NDIM,
        DIMS(STATISTIC_ROWS, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "fft_real_max", &output->fft_real_max, SERIES_NDIM,
        DIMS(STATISTIC_ROWS, DUMP_CHANNELS), CHANNEL_LAST
    );
    DUMP(
        "fft_imag_max", &output->fft_imag_max, SERIES_NDIM,
        DIMS(STATISTIC_ROWS, DUMP_CHANNELS), CHANNEL_LAST
    );
}
