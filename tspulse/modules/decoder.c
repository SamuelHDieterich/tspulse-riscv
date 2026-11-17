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

#ifdef TSPULSE_OUTPUTS_READ
void decoder_spans(
    const struct mixer_layer_138 layers[DECODER_LAYERS],
    float hidden[FLAT_LENGTH],
    const struct row_span *spans,
    int span_count
) {
    const int last = DECODER_LAYERS - 1;

    // Step 1: every layer but the last, whole
    DUMP_PUSH("decoder_block");
    tspulse_block(layers, last, hidden);

    // Step 2: the last layer, with its feature mixer on the rows the caller reads
    DUMP_PUSH("mixers.%d", last);
    mixer_layer_138_spans(&layers[last], hidden, spans, span_count);
    DUMP_POP();
    DUMP_POP();
}
#endif
