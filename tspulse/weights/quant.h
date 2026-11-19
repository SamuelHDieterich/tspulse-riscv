/**
 * @file
 * Which weight format the build uses: float32 everywhere, or int8 mixer weights with one scale per output row.
 *
 * The int8 format is on when `TSPULSE_QUANT_W8A32` or `TSPULSE_QUANT_W8A8` is defined, and every file tests `TSPULSE_QUANT_INT8` instead of the two.
 */

#ifndef QUANT_H
#define QUANT_H

// Built-in
// NULL, which the int8 weight table's rows use
#include <stddef.h>
#include <stdint.h>

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

#if defined(TSPULSE_QUANT_W8A32) || defined(TSPULSE_QUANT_W8A8)
#define TSPULSE_QUANT_INT8
/// The type of a mixer Linear's weights.
typedef int8_t mixer_weight;
/// A mixer Linear's scales, one per output row.
#define MIXER_SCALE(member) ((member).scale)
#else
/// The type of a mixer Linear's weights.
typedef float mixer_weight;
/// A float mixer Linear has no scales.
#define MIXER_SCALE(member) NULL
#endif

#endif
