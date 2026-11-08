/**
 * @file
 * The weight types of the TSPulse modules, one struct per module kind and size.
 *
 * Member names follow the PyTorch tensor names, and array dimensions equal the PyTorch shapes, so a Linear weight keeps its [out, in] layout.
 */

#ifndef TYPES_H
#define TYPES_H

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// An nn.LayerNorm over 24 features.
struct layer_norm_24 {
    float weight[24];
    float bias[24];
};

/// The norm wrapper of a mixer, which holds an nn.LayerNorm called `norm`.
struct norm_layer {
    struct layer_norm_24 norm;
};

/// A gated attention block over 10 features.
struct gated_attention_10 {
    struct {
        float weight[10][10];
        float bias[10];
    } attn_layer;
};

/// A gated attention block over 24 features.
struct gated_attention_24 {
    struct {
        float weight[24][24];
        float bias[24];
    } attn_layer;
};

/// A gated attention block over 138 features.
struct gated_attention_138 {
    struct {
        float weight[138][138];
        float bias[138];
    } attn_layer;
};

/// A two-layer MLP from 10 features to 20 and back.
struct mlp_10 {
    struct {
        float weight[20][10];
        float bias[20];
    } fc1;
    struct {
        float weight[10][20];
        float bias[10];
    } fc2;
};

/// A two-layer MLP from 24 features to 48 and back.
struct mlp_24 {
    struct {
        float weight[48][24];
        float bias[48];
    } fc1;
    struct {
        float weight[24][48];
        float bias[24];
    } fc2;
};

/// A two-layer MLP from 138 features to 276 and back.
struct mlp_138 {
    struct {
        float weight[276][138];
        float bias[276];
    } fc1;
    struct {
        float weight[138][276];
        float bias[138];
    } fc2;
};

/// A patch mixer over 10 patches.
struct patch_mixer_10 {
    struct norm_layer norm;
    struct mlp_10 mlp;
    struct gated_attention_10 gating_block;
};

/// A patch mixer over 138 patches.
struct patch_mixer_138 {
    struct norm_layer norm;
    struct mlp_138 mlp;
    struct gated_attention_138 gating_block;
};

/// A feature mixer over 24 features.
struct feature_mixer {
    struct norm_layer norm;
    struct mlp_24 mlp;
    struct gated_attention_24 gating_block;
};

/// A mixer layer over 10 patches.
struct mixer_layer_10 {
    struct patch_mixer_10 patch_mixer;
    struct feature_mixer feature_mixer;
};

/// A mixer layer over 138 patches.
struct mixer_layer_138 {
    struct patch_mixer_138 patch_mixer;
    struct feature_mixer feature_mixer;
};

/// A Linear layer from 8 features to 24.
struct linear_8_24 {
    float weight[24][8];
    float bias[24];
};

/// A Linear layer from 24 features to 8.
struct linear_24_8 {
    float weight[8][24];
    float bias[8];
};

/// A patch embedding, whose Linear is called `patcher`.
struct patch_encoding {
    struct linear_8_24 patcher;
};

/// A reconstruction head, whose output Linear is `base_reconstruction_block`.
struct reconstruction_head {
    struct linear_24_8 base_reconstruction_block;
};

#endif
