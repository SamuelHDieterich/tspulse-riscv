/**
 * @file
 * The sizes of the model and its input, shared by every file that handles a window.
 */

#ifndef SIZES_H
#define SIZES_H

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Rows the model reads as its past.
#define CONTEXT_LENGTH 512

/// Rows in a window: the past, then the row the forecast head is scored against.
#define WINDOW_LENGTH (CONTEXT_LENGTH + 1)

/// Values in one patch, for the time series and for its spectrum alike.
#define PATCH_LENGTH 8

/// Patches cut from 512 values.
#define NUM_PATCHES (CONTEXT_LENGTH / PATCH_LENGTH)

/// Features each patch is embedded into.
#define D_MODEL 24

/// Learned rows appended after the time and FFT patches.
#define REGISTER_TOKENS 10

/// Patches out of the two embeddings: the time patches, then the FFT patches.
#define ENCODED_PATCHES (2 * NUM_PATCHES)

/// Rows the encoder mixes: the embedded patches, then the register tokens.
#define TOTAL_PATCHES (ENCODED_PATCHES + REGISTER_TOKENS)

/// The encoder's hidden state flattened to one row, which `base_norm` normalizes.
#define FLAT_LENGTH (TOTAL_PATCHES * D_MODEL)

/// How much wider an MLP's hidden layer is than its input.
#define EXPANSION_FACTOR 2

/// Hidden width of the patch mixer's MLP over the encoder's 138 patches.
#define PATCH_MLP_HIDDEN (TOTAL_PATCHES * EXPANSION_FACTOR)

/// Hidden width of the patch mixer's MLP over the 10 register tokens.
#define REGISTER_MLP_HIDDEN (REGISTER_TOKENS * EXPANSION_FACTOR)

/// Hidden width of the feature mixer's MLP.
#define FEATURE_MLP_HIDDEN (D_MODEL * EXPANSION_FACTOR)

/// Mixer layers in the encoder.
#define ENCODER_LAYERS 8

/// Bins of a real DFT over 512 values, from 0 to 256.
#define FFT_BINS (CONTEXT_LENGTH / 2 + 1)

/// Bins the model keeps: the last one is set aside as the base component.
#define FFT_KEPT_BINS (FFT_BINS - 1)

/// Length of the learned mask token, which repeats every 8 rows.
#define MASK_BLOCK_LENGTH 8

/// Passes that each hide one patch of the last 96 rows.
#define HIDDEN_COPIES 12

/// Passes per window: the plain one, then the hidden ones.
#define COPIES (1 + HIDDEN_COPIES)

/// The first row the first hidden pass hides. Pass `i` hides rows `416 + 8i` to `423 + 8i`.
#define FIRST_HIDDEN_ROW (CONTEXT_LENGTH - HIDDEN_COPIES * PATCH_LENGTH)

#endif
