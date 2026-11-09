/**
 * @file
 * The heads after the decoder: the forecast from the register tokens, the time reconstruction, and the reconstruction through the FFT.
 *
 * In the dump build, each function writes its parts and its own output under the path its caller pushed.
 */

#ifndef HEADS_H
#define HEADS_H

// Internal
#include "model/sizes.h"
#include "weights/types.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * `forecast_mapping`: 3 mixer layers over the register tokens, then flatten, LayerNorm and a Linear to the forecast.
 *
 * Its PyTorch form is an `nn.Sequential`, so its parts are recorded as `0` to `3`, with the flatten as `1`.
 *
 * @param mixers The 3 mixer layers over 10 tokens, part `0`.
 * @param norm_weight The LayerNorm's scale over 240 values, part `2`.
 * @param norm_bias The LayerNorm's shift.
 * @param linear_weight The Linear's weights from 240 values to 16, part `3`.
 * @param linear_bias The Linear's bias.
 * @param registers The decoder's last 10 rows of 24 values.
 * @param forecast Receives the 16 forecast rows, still scaled as RevIN left them.
 */
void forecast_mapping(
    const struct mixer_layer_10 mixers[REGISTER_MIXER_LAYERS],
    const float norm_weight[REGISTER_FLAT_LENGTH],
    const float norm_bias[REGISTER_FLAT_LENGTH],
    const float linear_weight[PREDICTION_LENGTH][REGISTER_FLAT_LENGTH],
    const float linear_bias[PREDICTION_LENGTH],
    const float registers[REGISTER_FLAT_LENGTH],
    float forecast[PREDICTION_LENGTH]
);

/**
 * `TSPulseForReconstructionHead`: one Linear from each patch's 24 features to its 8 values, flattened to a series.
 *
 * @param weights The head's weights.
 * @param patches 64 rows of 24 values: the decoder's time rows for the time head, its FFT rows for the FFT head.
 * @param series Receives the 512 values the head reconstructs.
 */
void reconstruction_head(
    const struct reconstruction_head *weights,
    const float patches[NUM_PATCHES * D_MODEL],
    float series[CONTEXT_LENGTH]
);

/**
 * The series behind the FFT head's output: the bins scaled back by the maxima, the base component appended as bin 256, and the inverse DFT.
 *
 * @param fft_output The FFT head's 512 values: the normalized real parts of bins 0 to 255, then the normalized imaginary parts.
 * @param base_component The real part of bin 256, which the backbone set aside.
 * @param real_max The backbone's largest real magnitude plus epsilon, which the real parts were divided by.
 * @param imag_max The backbone's largest imaginary magnitude plus epsilon, which the imaginary parts were divided by.
 * @param series Receives the 512 values, still scaled as RevIN left them.
 */
void fft_reconstruction(
    const float fft_output[CONTEXT_LENGTH],
    float base_component,
    float real_max,
    float imag_max,
    float series[CONTEXT_LENGTH]
);

#endif
