/**
 * @file
 * `TSPulseBlock`: a stack of mixer layers over the 138 patches, which both the encoder and the decoder are.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "modules/block.h"
#include "io/dump.h"
#include "model/sizes.h"
#include "modules/mixer.h"
#include "weights/types.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

void tspulse_block(
    const struct mixer_layer_138 *layers, int count, float hidden[FLAT_LENGTH]
) {
    for (int i = 0; i < count; i++) {
        DUMP_PUSH("mixers.%d", i);
        mixer_layer_138(&layers[i], hidden);
        DUMP_POP();
    }

    // The block returns a tuple of the output and the hidden states it was not asked to keep, so only part 0 exists
    DUMP(
        "0", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, TOTAL_PATCHES, D_MODEL),
        CHANNEL_FIRST
    );
}
