/**
 * @file
 * The decoder, `TSPulseDecoder`: two more mixer layers over the encoder's 138 patches.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#include "modules/decoder.h"
#include "io/dump.h"
#include "model/sizes.h"
#include "modules/block.h"
#include "weights/types.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

void decoder(
    const struct mixer_layer_138 layers[DECODER_LAYERS],
    float hidden[FLAT_LENGTH]
) {
    DUMP_PUSH("decoder_block");
    tspulse_block(layers, DECODER_LAYERS, hidden);
    DUMP_POP();

    // TSPulseDecoder also returns a tuple, whose part 0 is the block's output
    DUMP(
        "0", hidden, NETWORK_NDIM, DIMS(DUMP_CHANNELS, TOTAL_PATCHES, D_MODEL),
        CHANNEL_FIRST
    );
}
