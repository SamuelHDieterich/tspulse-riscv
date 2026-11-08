/**
 * @file
 * The TSPulse weights, as one struct that mirrors the PyTorch module tree.
 *
 * Member names follow the PyTorch tensor names, and array dimensions equal the PyTorch shapes, so a Linear weight keeps its [out, in] layout.
 * The `mixers` lists are arrays, so module code can loop over them.
 * The entries of `forecast_mapping` are different modules, so they are members named `_0`, `_2` and `_3`.
 */

#ifndef WEIGHTS_H
#define WEIGHTS_H

// Internal
#include "weights/types.h"

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// Every weight of the model, in the layout of the PyTorch module tree.
struct tspulse_weights {
    struct {
        struct {
            float patch_tokens[10][24];
        } add_tokens;
        struct {
            float weight[3312];
            float bias[3312];
        } base_norm;
        struct {
            struct mixer_layer_138 mixers[8];
        } encoder_block;
        struct patch_encoding fft_encoding;
        struct {
            float affine_weight[1][1][1];
            float affine_bias[1][1][1];
        } scaler;
        struct patch_encoding time_encoding;
        struct {
            float mask_token[8];
        } time_masker;
    } backbone;
    struct {
        struct {
            struct {
                struct mixer_layer_138 mixers[2];
            } decoder_block;
        } decoder;
        struct reconstruction_head fft_head;
        struct {
            struct {
                struct mixer_layer_10 mixers[3];
            } _0;
            struct {
                float weight[240];
                float bias[240];
            } _2;
            struct {
                float weight[16][240];
                float bias[16];
            } _3;
        } forecast_mapping;
        struct reconstruction_head time_head;
    } decoder_with_head;
};

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The weights compiled into the program, in read-only data.
extern const struct tspulse_weights TSPULSE_WEIGHTS;

#endif
