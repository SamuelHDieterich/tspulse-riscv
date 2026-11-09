/**
 * @file
 * One forward pass of one channel for one copy: the backbone, the decoder, and the heads that copy's error needs.
 */

#ifndef PASS_H
#define PASS_H

// Internal
#include "model/sizes.h"
#include "weights/weights.h"

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// What one pass yields toward the window's errors, in the series' standardized units.
struct pass_output {
    /// The time head's reconstruction, filled by the hidden passes only.
    float reconstruction[CONTEXT_LENGTH];
    /// The reconstruction through the FFT head, filled by the hidden passes only.
    float reconstruction_from_fft[CONTEXT_LENGTH];
    /// The forecast of the rows after the past, filled by the plain pass only.
    float forecast[PREDICTION_LENGTH];
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * `TSPulseForReconstruction`: runs one channel through the model for one copy.
 *
 * The plain pass only feeds the forecast error, so it runs the forecast head and not the reconstruction heads.
 * A hidden pass only feeds the reconstruction errors, so it runs the time and FFT heads and not the forecast head.
 * In the dump build, it writes the records of `backbone`, `decoder_with_head` and `output` under the path its caller pushed.
 *
 * @param weights The model's weights.
 * @param past The channel's standardized past, 512 values.
 * @param hidden_patch Which of the last 12 patches this pass hides, from 0 to 11, or `NO_HIDDEN_PATCH` for the plain pass.
 * @param output Receives the heads' outputs this copy computes, back in the series' units.
 */
void pass(
    const struct tspulse_weights *weights,
    const float past[CONTEXT_LENGTH],
    int hidden_patch,
    struct pass_output *output
);

#endif
