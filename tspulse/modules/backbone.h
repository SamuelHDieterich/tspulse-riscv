/**
 * @file
 * The backbone, `TSPulseModel`: masking, scaling, the time and FFT embeddings, and the encoder.
 *
 * In the dump build, each function writes its parts and its own output under the path its caller pushed.
 */

#ifndef BACKBONE_H
#define BACKBONE_H

// Internal
#include "model/sizes.h"
#include "weights/types.h"
#include "weights/weights.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The `hidden_patch` of the plain pass, which hides nothing.
#define NO_HIDDEN_PATCH (-1)

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// What the backbone hands the decoder and the heads, named after the fields of `TSPulseModelOutput`.
struct backbone_output {
    /// The encoder's output, 138 rows of 24 values.
    float last_hidden_state[FLAT_LENGTH];
    /// The input with the hidden patch replaced by the mask token.
    float masked_past_values[CONTEXT_LENGTH];
    /// 1.0 on the hidden rows, 0.0 elsewhere.
    float mask[CONTEXT_LENGTH];
    float loc;   ///< The mean RevIN removed.
    float scale; ///< The spread RevIN divided by.
    /// The real part of DFT bin 256, which the FFT embedding leaves out.
    float fft_base_component;
    /// The largest magnitude among the real parts of bins 0 to 255, plus epsilon.
    float fft_real_max;
    /// The largest magnitude among their imaginary parts, plus epsilon.
    float fft_imag_max;
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * `TSPulseMasking` with user masks: replaces the hidden rows with the learned mask token.
 *
 * @param mask_token The learned token, 8 values, of which row `t` takes value `t mod 8`.
 * @param past The input, 512 values.
 * @param hidden_patch Which of the last 12 patches to hide, from 0 to 11, or `NO_HIDDEN_PATCH`.
 * @param masked Receives the masked input, 512 values.
 * @param mask Receives 1.0 on the hidden rows and 0.0 elsewhere, 512 values.
 */
void time_masker(
    const float mask_token[MASK_BLOCK_LENGTH],
    const float past[CONTEXT_LENGTH],
    int hidden_patch,
    float masked[CONTEXT_LENGTH],
    float mask[CONTEXT_LENGTH]
);

/**
 * `TSPulseBlock` as the encoder: the 8 mixer layers over 138 patches, one after another.
 *
 * @param layers The layers' weights.
 * @param hidden The hidden state, 138 rows of 24 values, updated in place.
 */
void encoder_block(
    const struct mixer_layer_138 layers[ENCODER_LAYERS],
    float hidden[FLAT_LENGTH]
);

/**
 * `TSPulseModel`: runs one pass of one channel through the backbone.
 *
 * @param weights The model's weights, of which the backbone's are used.
 * @param past The channel's standardized past, 512 values.
 * @param hidden_patch Which of the last 12 patches this pass hides, from 0 to 11, or `NO_HIDDEN_PATCH`.
 * @param output Receives the encoder's output and what the heads need to undo the scaling.
 */
void backbone(
    const struct tspulse_weights *weights,
    const float past[CONTEXT_LENGTH],
    int hidden_patch,
    struct backbone_output *output
);

#endif
