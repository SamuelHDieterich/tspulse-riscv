/**
 * @file
 * The embedding modules: the patch encoding, the register tokens, and the norm over the whole embedding.
 *
 * In the dump build, each function writes its parts and its own output under the path its caller pushed.
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

// Internal
#include "model/sizes.h"
#include "weights/types.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * `TSPulseEncoding`: embeds each patch of 8 values into 24 features with one Linear, called `patcher`.
 *
 * @param weights The encoding's weights.
 * @param patches The input, 64 patches of 8 values.
 * @param encoded Receives 64 rows of 24 values.
 */
void patch_encoding(
    const struct patch_encoding *weights,
    const float patches[NUM_PATCHES * PATCH_LENGTH],
    float encoded[NUM_PATCHES * D_MODEL]
);

/**
 * `TSPulseAddLearnableRegisterTokens`: appends the 10 learned rows after the embedded patches.
 *
 * @param tokens The learned rows, 10 of 24 values.
 * @param encoded The time patches then the FFT patches, 128 rows of 24 values.
 * @param hidden Receives 138 rows of 24 values.
 */
void add_tokens(
    const float tokens[REGISTER_TOKENS][D_MODEL],
    const float encoded[ENCODED_PATCHES * D_MODEL],
    float hidden[FLAT_LENGTH]
);

/**
 * `base_norm`: a LayerNorm over all 3312 values of the hidden state at once.
 *
 * @param weight The scale, 3312 values.
 * @param bias The shift, 3312 values.
 * @param hidden The hidden state, 138 rows of 24 values, normalized in place.
 */
void base_norm(
    const float weight[FLAT_LENGTH],
    const float bias[FLAT_LENGTH],
    float hidden[FLAT_LENGTH]
);

#endif
