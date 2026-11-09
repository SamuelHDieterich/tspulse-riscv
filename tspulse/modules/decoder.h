/**
 * @file
 * The decoder, `TSPulseDecoder`: two more mixer layers over the encoder's 138 patches.
 *
 * In the dump build, it writes its parts and its own output under the path its caller pushed.
 */

#ifndef DECODER_H
#define DECODER_H

// Internal
#include "model/sizes.h"
#include "weights/types.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * `TSPulseDecoder`: its `decoder_block`, the 2 mixer layers over 138 patches, one after another.
 *
 * @param layers The decoder block's layers.
 * @param hidden The hidden state, 138 rows of 24 values, updated in place.
 */
void decoder(
    const struct mixer_layer_138 layers[DECODER_LAYERS],
    float hidden[FLAT_LENGTH]
);

#endif
