/**
 * @file
 * The parts of a TSMixer layer: the MLP, the gated attention, the patch and feature mixers, and the layer that chains them.
 *
 * The sized functions share one generic body per module, which takes the weights as plain arrays and the sizes as arguments.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "modules/mixer.h"
#include "io/dump.h"
#include "kernels/add.h"
#include "kernels/gelu.h"
#include "kernels/layer_norm.h"
#include "kernels/linear.h"
#include "kernels/multiply.h"
#include "kernels/softmax.h"
#include "kernels/transpose.h"
#include "model/sizes.h"
#include "weights/types.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The largest MLP hidden layer: 24 rows of 276 in the patch mixer, or 138 rows of 48 in the feature mixer.
#define MLP_SCRATCH                                                            \
    (D_MODEL * PATCH_MLP_HIDDEN > TOTAL_PATCHES * FEATURE_MLP_HIDDEN           \
         ? D_MODEL * PATCH_MLP_HIDDEN                                          \
         : TOTAL_PATCHES * FEATURE_MLP_HIDDEN)

/// The largest hidden state, 138 rows of 24 or its transpose.
#define STATE_SCRATCH FLAT_LENGTH

/// The register tokens' hidden state, 10 rows of 24 or its transpose.
#define REGISTER_SCRATCH (REGISTER_TOKENS * D_MODEL)

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * The body of every `TSPulseMLP`: fc1, GELU, fc2.
 *
 * @param fc1_weight fc1's weights, `hidden_features` rows of `features`.
 * @param fc1_bias fc1's bias, `hidden_features` values.
 * @param fc2_weight fc2's weights, `features` rows of `hidden_features`.
 * @param fc2_bias fc2's bias, `features` values.
 * @param x The input, `rows` rows of `features` values.
 * @param y Receives `rows` rows of `features` values. It must not overlap `x`.
 * @param rows Rows in `x`.
 * @param features Values per row of `x` and `y`.
 * @param hidden_features Values per row of the hidden layer.
 */
static void
mlp(const float *fc1_weight,
    const float *fc1_bias,
    const float *fc2_weight,
    const float *fc2_bias,
    const float *x,
    float *y,
    int rows,
    int features,
    int hidden_features) {
    static float hidden[MLP_SCRATCH];

    // Step 1: fc1
    linear(x, fc1_weight, fc1_bias, hidden, rows, features, hidden_features);
    DUMP(
        "fc1", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, hidden_features),
        CHANNEL_FIRST
    );

    // Step 2: GELU. Dropout does nothing at inference, so its record holds the same values
    gelu(hidden, rows * hidden_features);
    DUMP(
        "dropout1", hidden, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, rows, hidden_features), CHANNEL_FIRST
    );

    // Step 3: fc2, whose output is also the second dropout's and the MLP's
    linear(hidden, fc2_weight, fc2_bias, y, rows, hidden_features, features);
    DUMP(
        "fc2", y, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, features),
        CHANNEL_FIRST
    );
    DUMP(
        "dropout2", y, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, features),
        CHANNEL_FIRST
    );
    DUMP(
        "", y, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, features), CHANNEL_FIRST
    );
}

/**
 * The body of every `TSPulseGatedAttention`: y = x * softmax(Linear(x)), the softmax over each row.
 *
 * @param weight The Linear's weights, `features` rows of `features`.
 * @param bias The Linear's bias, `features` values.
 * @param x The input, `rows` rows of `features` values.
 * @param y Receives `rows` rows of `features` values. It must not overlap `x`.
 * @param rows Rows in `x`.
 * @param features Values per row.
 */
static void gated_attention(
    const float *weight,
    const float *bias,
    const float *x,
    float *y,
    int rows,
    int features
) {
    // y doubles as the gate's buffer, so the gate needs no memory of its own
    linear(x, weight, bias, y, rows, features, features);
    DUMP(
        "attn_layer", y, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, features),
        CHANNEL_FIRST
    );
    softmax(y, rows, features);
    DUMP(
        "attn_activation_layer", y, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, rows, features), CHANNEL_FIRST
    );
    multiply(x, y, y, rows * features);
    DUMP(
        "", y, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, features), CHANNEL_FIRST
    );
}

/**
 * A mixer's `TSPulseNormLayer`, which wraps a LayerNorm over the 24 features called `norm`.
 *
 * @param weights The norm layer's weights.
 * @param x The hidden state, `rows` rows of 24 values.
 * @param y Receives the normalized state. It must not overlap `x`.
 * @param rows Rows in `x`.
 */
static void mixer_norm(
    const struct norm_layer *weights, const float *x, float *y, int rows
) {
    DUMP_PUSH("norm");
    layer_norm(x, weights->norm.weight, weights->norm.bias, y, rows, D_MODEL);

    // The wrapper returns the LayerNorm's output as it is, so both records hold the same values
    DUMP(
        "norm", y, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, D_MODEL),
        CHANNEL_FIRST
    );
    DUMP(
        "", y, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, D_MODEL), CHANNEL_FIRST
    );
    DUMP_POP();
}

/**
 * The patch mixer's first half: normalize, then transpose so each row holds one feature across all patches.
 *
 * @param norm The norm layer's weights.
 * @param hidden The hidden state, `patches` rows of 24 values.
 * @param normed Scratch for the normalized state, `patches` rows of 24 values.
 * @param transposed Receives 24 rows of `patches` values.
 * @param patches Rows in the hidden state.
 */
static void patch_mixer_begin(
    const struct norm_layer *norm,
    const float *hidden,
    float *normed,
    float *transposed,
    int patches
) {
    mixer_norm(norm, hidden, normed, patches);
    transpose(normed, transposed, patches, D_MODEL);
}

/**
 * The patch mixer's second half: transpose back, then add the residual.
 *
 * @param gated The gated MLP output, 24 rows of `patches` values.
 * @param scratch Scratch for the transposed output, `patches` rows of 24 values.
 * @param hidden The hidden state, `patches` rows of 24 values, which receives the sum.
 * @param patches Rows in the hidden state.
 */
static void patch_mixer_end(
    const float *gated, float *scratch, float *hidden, int patches
) {
    transpose(gated, scratch, D_MODEL, patches);
    add(scratch, hidden, hidden, patches * D_MODEL);
    DUMP(
        "", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, patches, D_MODEL),
        CHANNEL_FIRST
    );
}

void mlp_10(const struct mlp_10 *weights, const float *x, float *y, int rows) {
    mlp(&weights->fc1.weight[0][0], weights->fc1.bias,
        &weights->fc2.weight[0][0], weights->fc2.bias, x, y, rows,
        REGISTER_TOKENS, REGISTER_MLP_HIDDEN);
}

void mlp_24(const struct mlp_24 *weights, const float *x, float *y, int rows) {
    mlp(&weights->fc1.weight[0][0], weights->fc1.bias,
        &weights->fc2.weight[0][0], weights->fc2.bias, x, y, rows, D_MODEL,
        FEATURE_MLP_HIDDEN);
}

void mlp_138(
    const struct mlp_138 *weights, const float *x, float *y, int rows
) {
    mlp(&weights->fc1.weight[0][0], weights->fc1.bias,
        &weights->fc2.weight[0][0], weights->fc2.bias, x, y, rows,
        TOTAL_PATCHES, PATCH_MLP_HIDDEN);
}

void gated_attention_10(
    const struct gated_attention_10 *weights, const float *x, float *y, int rows
) {
    gated_attention(
        &weights->attn_layer.weight[0][0], weights->attn_layer.bias, x, y, rows,
        REGISTER_TOKENS
    );
}

void gated_attention_24(
    const struct gated_attention_24 *weights, const float *x, float *y, int rows
) {
    gated_attention(
        &weights->attn_layer.weight[0][0], weights->attn_layer.bias, x, y, rows,
        D_MODEL
    );
}

void gated_attention_138(
    const struct gated_attention_138 *weights,
    const float *x,
    float *y,
    int rows
) {
    gated_attention(
        &weights->attn_layer.weight[0][0], weights->attn_layer.bias, x, y, rows,
        TOTAL_PATCHES
    );
}

void patch_mixer_10(const struct patch_mixer_10 *weights, float *hidden) {
    static float normed[REGISTER_SCRATCH];
    static float transposed[REGISTER_SCRATCH];
    static float mixed[REGISTER_SCRATCH];
    static float gated[REGISTER_SCRATCH];

    // Step 1: normalize, and put the patches last so the MLP mixes across them
    patch_mixer_begin(
        &weights->norm, hidden, normed, transposed, REGISTER_TOKENS
    );

    // Step 2: the MLP and the gate over the patches
    DUMP_PUSH("mlp");
    mlp_10(&weights->mlp, transposed, mixed, D_MODEL);
    DUMP_POP();
    DUMP_PUSH("gating_block");
    gated_attention_10(&weights->gating_block, mixed, gated, D_MODEL);
    DUMP_POP();

    // Step 3: back to patches first, plus the residual. normed is free again, so it holds the transpose
    patch_mixer_end(gated, normed, hidden, REGISTER_TOKENS);
}

void patch_mixer_138(const struct patch_mixer_138 *weights, float *hidden) {
    static float normed[STATE_SCRATCH];
    static float transposed[STATE_SCRATCH];
    static float mixed[STATE_SCRATCH];
    static float gated[STATE_SCRATCH];

    // Step 1: normalize, and put the patches last so the MLP mixes across them
    patch_mixer_begin(
        &weights->norm, hidden, normed, transposed, TOTAL_PATCHES
    );

    // Step 2: the MLP and the gate over the patches
    DUMP_PUSH("mlp");
    mlp_138(&weights->mlp, transposed, mixed, D_MODEL);
    DUMP_POP();
    DUMP_PUSH("gating_block");
    gated_attention_138(&weights->gating_block, mixed, gated, D_MODEL);
    DUMP_POP();

    // Step 3: back to patches first, plus the residual. normed is free again, so it holds the transpose
    patch_mixer_end(gated, normed, hidden, TOTAL_PATCHES);
}

void feature_mixer(
    const struct feature_mixer *weights, float *hidden, int rows
) {
    static float normed[STATE_SCRATCH];
    static float mixed[STATE_SCRATCH];
    static float gated[STATE_SCRATCH];

    // Step 1: normalize
    mixer_norm(&weights->norm, hidden, normed, rows);

    // Step 2: the MLP and the gate over each patch's features
    DUMP_PUSH("mlp");
    mlp_24(&weights->mlp, normed, mixed, rows);
    DUMP_POP();
    DUMP_PUSH("gating_block");
    gated_attention_24(&weights->gating_block, mixed, gated, rows);
    DUMP_POP();

    // Step 3: the residual
    add(gated, hidden, hidden, rows * D_MODEL);
    DUMP(
        "", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, rows, D_MODEL),
        CHANNEL_FIRST
    );
}

void mixer_layer_10(const struct mixer_layer_10 *weights, float *hidden) {
    DUMP_PUSH("patch_mixer");
    patch_mixer_10(&weights->patch_mixer, hidden);
    DUMP_POP();
    DUMP_PUSH("feature_mixer");
    feature_mixer(&weights->feature_mixer, hidden, REGISTER_TOKENS);
    DUMP_POP();
    DUMP(
        "", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, REGISTER_TOKENS, D_MODEL),
        CHANNEL_FIRST
    );
}

void mixer_layer_138(const struct mixer_layer_138 *weights, float *hidden) {
    DUMP_PUSH("patch_mixer");
    patch_mixer_138(&weights->patch_mixer, hidden);
    DUMP_POP();
    DUMP_PUSH("feature_mixer");
    feature_mixer(&weights->feature_mixer, hidden, TOTAL_PATCHES);
    DUMP_POP();
    DUMP(
        "", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, TOTAL_PATCHES, D_MODEL),
        CHANNEL_FIRST
    );
}
