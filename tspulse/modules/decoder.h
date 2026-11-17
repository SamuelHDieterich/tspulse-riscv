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
#include "modules/mixer.h"
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

#ifdef TSPULSE_OUTPUTS_READ
/**
 * The decoder, with the last layer's feature mixer run only on the given rows.
 *
 * @param layers The decoder block's layers.
 * @param hidden The hidden state, 138 rows of 24 values, updated in place. Afterwards only the rows in `spans` hold the decoder's output.
 * @param spans The runs of rows the caller reads.
 * @param span_count Runs in `spans`.
 */
void decoder_spans(
    const struct mixer_layer_138 layers[DECODER_LAYERS],
    float hidden[FLAT_LENGTH],
    const struct row_span *spans,
    int span_count
);
#endif

#endif
