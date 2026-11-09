/**
 * @file
 * `TSPulseBlock`: a stack of mixer layers over the 138 patches, which both the encoder and the decoder are.
 *
 * In the dump build, it writes its layers and its own output under the path its caller pushed.
 */

#ifndef BLOCK_H
#define BLOCK_H

// Internal
#include "model/sizes.h"
#include "weights/types.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * `TSPulseBlock`: the mixer layers over 138 patches, one after another.
 *
 * @param layers The block's layers, `count` of them.
 * @param count Layers in the block: 8 in the encoder, 2 in the decoder.
 * @param hidden The hidden state, 138 rows of 24 values, updated in place.
 */
void tspulse_block(
    const struct mixer_layer_138 *layers, int count, float hidden[FLAT_LENGTH]
);

#endif
