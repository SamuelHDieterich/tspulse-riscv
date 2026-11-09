/**
 * @file
 * The parts of a TSMixer layer: the MLP, the gated attention, the patch and feature mixers, and the layer that chains them.
 *
 * Each function is named after the PyTorch module it ports, with the size its weight type carries.
 * Inputs are one channel's values, row-major, in the layout the PyTorch module sees without its batch and channel dimensions.
 * In the dump build, each function writes its parts and its own output under the path its caller pushed.
 */

#ifndef MIXER_H
#define MIXER_H

// Internal
#include "weights/types.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * `TSPulseMLP` over 10 features: fc1, GELU, fc2. Dropout does nothing at inference.
 *
 * @param weights The MLP's weights.
 * @param x The input, `rows` rows of 10 values.
 * @param y Receives `rows` rows of 10 values. It must not overlap `x`.
 * @param rows Rows in `x`, at most 24.
 */
void mlp_10(const struct mlp_10 *weights, const float *x, float *y, int rows);

/**
 * `TSPulseMLP` over 24 features: fc1, GELU, fc2. Dropout does nothing at inference.
 *
 * @param weights The MLP's weights.
 * @param x The input, `rows` rows of 24 values.
 * @param y Receives `rows` rows of 24 values. It must not overlap `x`.
 * @param rows Rows in `x`, at most 138.
 */
void mlp_24(const struct mlp_24 *weights, const float *x, float *y, int rows);

/**
 * `TSPulseMLP` over 138 features: fc1, GELU, fc2. Dropout does nothing at inference.
 *
 * @param weights The MLP's weights.
 * @param x The input, `rows` rows of 138 values.
 * @param y Receives `rows` rows of 138 values. It must not overlap `x`.
 * @param rows Rows in `x`, at most 24.
 */
void mlp_138(const struct mlp_138 *weights, const float *x, float *y, int rows);

/**
 * `TSPulseGatedAttention` over 10 features: the input times a softmax of a Linear of itself.
 *
 * @param weights The gate's weights.
 * @param x The input, `rows` rows of 10 values.
 * @param y Receives `rows` rows of 10 values. It must not overlap `x`.
 * @param rows Rows in `x`, at most 24.
 */
void gated_attention_10(
    const struct gated_attention_10 *weights, const float *x, float *y, int rows
);

/**
 * `TSPulseGatedAttention` over 24 features: the input times a softmax of a Linear of itself.
 *
 * @param weights The gate's weights.
 * @param x The input, `rows` rows of 24 values.
 * @param y Receives `rows` rows of 24 values. It must not overlap `x`.
 * @param rows Rows in `x`, at most 138.
 */
void gated_attention_24(
    const struct gated_attention_24 *weights, const float *x, float *y, int rows
);

/**
 * `TSPulseGatedAttention` over 138 features: the input times a softmax of a Linear of itself.
 *
 * @param weights The gate's weights.
 * @param x The input, `rows` rows of 138 values.
 * @param y Receives `rows` rows of 138 values. It must not overlap `x`.
 * @param rows Rows in `x`, at most 24.
 */
void gated_attention_138(
    const struct gated_attention_138 *weights,
    const float *x,
    float *y,
    int rows
);

/**
 * `PatchMixerBlock` over 10 patches: mixes each feature across the patches, with a residual connection.
 *
 * @param weights The mixer's weights.
 * @param hidden The hidden state, 10 rows of 24 values, updated in place.
 */
void patch_mixer_10(const struct patch_mixer_10 *weights, float *hidden);

/**
 * `PatchMixerBlock` over 138 patches: mixes each feature across the patches, with a residual connection.
 *
 * @param weights The mixer's weights.
 * @param hidden The hidden state, 138 rows of 24 values, updated in place.
 */
void patch_mixer_138(const struct patch_mixer_138 *weights, float *hidden);

/**
 * `FeatureMixerBlock`: mixes each patch's 24 features, with a residual connection.
 *
 * @param weights The mixer's weights.
 * @param hidden The hidden state, `rows` rows of 24 values, updated in place.
 * @param rows Patches in the hidden state, at most 138.
 */
void feature_mixer(
    const struct feature_mixer *weights, float *hidden, int rows
);

/**
 * `TSPulseLayer` over 10 patches: the patch mixer, then the feature mixer.
 *
 * @param weights The layer's weights.
 * @param hidden The hidden state, 10 rows of 24 values, updated in place.
 */
void mixer_layer_10(const struct mixer_layer_10 *weights, float *hidden);

/**
 * `TSPulseLayer` over 138 patches: the patch mixer, then the feature mixer.
 *
 * @param weights The layer's weights.
 * @param hidden The hidden state, 138 rows of 24 values, updated in place.
 */
void mixer_layer_138(const struct mixer_layer_138 *weights, float *hidden);

#endif
