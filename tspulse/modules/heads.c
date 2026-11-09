/**
 * @file
 * The heads after the decoder: the forecast from the register tokens, the time reconstruction, and the reconstruction through the FFT.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <string.h>

// Internal
#include "io/dump.h"
#include "kernels/fft.h"
#include "kernels/layer_norm.h"
#include "kernels/linear.h"
#include "model/sizes.h"
#include "modules/heads.h"
#include "modules/mixer.h"
#include "weights/types.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The one row the forecast head's LayerNorm and Linear see per channel.
#define FORECAST_ROWS 1

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

void forecast_mapping(
    const struct mixer_layer_10 mixers[REGISTER_MIXER_LAYERS],
    const float norm_weight[REGISTER_FLAT_LENGTH],
    const float norm_bias[REGISTER_FLAT_LENGTH],
    const float linear_weight[PREDICTION_LENGTH][REGISTER_FLAT_LENGTH],
    const float linear_bias[PREDICTION_LENGTH],
    const float registers[REGISTER_FLAT_LENGTH],
    float forecast[PREDICTION_LENGTH]
) {
    static float tokens[REGISTER_FLAT_LENGTH];

    // Step 1: the mixer layers, on a copy so the decoder's output stays whole for its own record
    memcpy(tokens, registers, sizeof(tokens));
    DUMP_PUSH("0");
    for (int i = 0; i < REGISTER_MIXER_LAYERS; i++) {
        DUMP_PUSH("mixers.%d", i);
        mixer_layer_10(&mixers[i], tokens);
        DUMP_POP();
    }

    // This block is built with `return_tuple=False`, so its output is recorded whole, not as part 0
    DUMP(
        "", tokens, NETWORK_NDIM, DIMS(DUMP_CHANNELS, REGISTER_TOKENS, D_MODEL),
        CHANNEL_FIRST
    );
    DUMP_POP();

    // Step 2: flatten the 10 rows of 24 into one row of 240, which leaves them where they are
    DUMP(
        "1", tokens, FLAT_NDIM, DIMS(DUMP_CHANNELS, REGISTER_FLAT_LENGTH),
        CHANNEL_FIRST
    );

    // Step 3: normalize the 240 values as one row
    layer_norm(
        tokens, norm_weight, norm_bias, tokens, FORECAST_ROWS,
        REGISTER_FLAT_LENGTH
    );
    DUMP(
        "2", tokens, FLAT_NDIM, DIMS(DUMP_CHANNELS, REGISTER_FLAT_LENGTH),
        CHANNEL_FIRST
    );

    // Step 4: the Linear to the 16 forecast rows
    linear(
        tokens, &linear_weight[0][0], linear_bias, forecast, FORECAST_ROWS,
        REGISTER_FLAT_LENGTH, PREDICTION_LENGTH
    );
    DUMP(
        "3", forecast, FLAT_NDIM, DIMS(DUMP_CHANNELS, PREDICTION_LENGTH),
        CHANNEL_FIRST
    );
    DUMP(
        "", forecast, FLAT_NDIM, DIMS(DUMP_CHANNELS, PREDICTION_LENGTH),
        CHANNEL_FIRST
    );
}

void reconstruction_head(
    const struct reconstruction_head *weights,
    const float patches[NUM_PATCHES * D_MODEL],
    float series[CONTEXT_LENGTH]
) {
    // Step 1: dropout does nothing at inference, so its record is the input
    DUMP(
        "dropout_layer", patches, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, NUM_PATCHES, D_MODEL), CHANNEL_FIRST
    );

    // Step 2: each patch's 24 features to its 8 values
    linear(
        patches, &weights->base_reconstruction_block.weight[0][0],
        weights->base_reconstruction_block.bias, series, NUM_PATCHES, D_MODEL,
        PATCH_LENGTH
    );
    DUMP(
        "base_reconstruction_block", series, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, NUM_PATCHES, PATCH_LENGTH), CHANNEL_FIRST
    );

    // Step 3: 64 patches of 8 flattened to 512 values, which leaves them where they are, then transposed to time first
    DUMP(
        "flatten", series, FLAT_NDIM, DIMS(DUMP_CHANNELS, CONTEXT_LENGTH),
        CHANNEL_FIRST
    );
    DUMP(
        "", series, SERIES_NDIM, DIMS(CONTEXT_LENGTH, DUMP_CHANNELS),
        CHANNEL_LAST
    );
}

void fft_reconstruction(
    const float fft_output[CONTEXT_LENGTH],
    float base_component,
    float real_max,
    float imag_max,
    float series[CONTEXT_LENGTH]
) {
    static float real[FFT_BINS];
    static float imag[FFT_BINS];

    // Step 1: undo the backbone's division of each part of `torch.fft.rfft`'s bins by its maximum, in `get_fft`
    // Re X[k] = output[k] * real_max, Im X[k] = output[256 + k] * imag_max, for k = 0 to 255
    // https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html
    for (int k = 0; k < FFT_KEPT_BINS; k++) {
        real[k] = fft_output[k] * real_max;
        imag[k] = fft_output[FFT_KEPT_BINS + k] * imag_max;
    }

    // Step 2: the config's `fft_remove_component` is "last", so the base component goes back as bin 256. Its imaginary part is never read, as irfft ignores it
    real[FFT_KEPT_BINS] = base_component;
    imag[FFT_KEPT_BINS] = 0.0f;

    // Step 3: the inverse DFT with n = 512, which also ignores the head's imaginary part of bin 0
    // https://docs.pytorch.org/docs/stable/generated/torch.fft.irfft.html
    fft_inverse(real, imag, series);
}
