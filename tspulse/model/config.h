/**
 * @file
 * The settings from the model's `config.json` that aren't sizes, since the C reads no config.
 */

#ifndef CONFIG_H
#define CONFIG_H

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// `minimum_scale`: the floor on RevIN's variance, and the epsilon added to the FFT maxima.
#define MINIMUM_SCALE 0.001f

/// `norm_eps`: the epsilon every LayerNorm adds to the variance.
#define NORM_EPS 1e-5f

#endif
