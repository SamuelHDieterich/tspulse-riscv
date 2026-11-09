/**
 * @file
 * The embedding modules: the patch encoding, the register tokens, and the norm over the whole embedding.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <string.h>

// Internal
#include "io/dump.h"
#include "kernels/layer_norm.h"
#include "kernels/linear.h"
#include "model/sizes.h"
#include "modules/embedding.h"
#include "weights/types.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The one row `base_norm` normalizes.
#define BASE_NORM_ROWS 1

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

void patch_encoding(
    const struct patch_encoding *weights,
    const float patches[NUM_PATCHES * PATCH_LENGTH],
    float encoded[NUM_PATCHES * D_MODEL]
) {
    linear(
        patches, &weights->patcher.weight[0][0], weights->patcher.bias, encoded,
        NUM_PATCHES, PATCH_LENGTH, D_MODEL
    );

    // The model has no positional encoding, so the module's output is the Linear's
    DUMP(
        "patcher", encoded, NETWORK_NDIM,
        DIMS(DUMP_CHANNELS, NUM_PATCHES, D_MODEL), CHANNEL_FIRST
    );
    DUMP(
        "", encoded, NETWORK_NDIM, DIMS(DUMP_CHANNELS, NUM_PATCHES, D_MODEL),
        CHANNEL_FIRST
    );
}

void add_tokens(
    const float tokens[REGISTER_TOKENS][D_MODEL],
    const float encoded[ENCODED_PATCHES * D_MODEL],
    float hidden[FLAT_LENGTH]
) {
    memcpy(hidden, encoded, ENCODED_PATCHES * D_MODEL * sizeof(float));
    memcpy(
        hidden + ENCODED_PATCHES * D_MODEL, &tokens[0][0],
        REGISTER_TOKENS * D_MODEL * sizeof(float)
    );
    DUMP(
        "", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, TOTAL_PATCHES, D_MODEL),
        CHANNEL_FIRST
    );
}

void base_norm(
    const float weight[FLAT_LENGTH],
    const float bias[FLAT_LENGTH],
    float hidden[FLAT_LENGTH]
) {
    // The 138 rows of 24 are normalized as one row of 3312, as PyTorch reshapes them
    layer_norm(hidden, weight, bias, hidden, BASE_NORM_ROWS, FLAT_LENGTH);
    DUMP(
        "", hidden, FLAT_NDIM, DIMS(DUMP_CHANNELS, FLAT_LENGTH), CHANNEL_FIRST
    );
}
