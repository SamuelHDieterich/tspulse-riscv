/**
 * @file
 * The real FFT over 512 values and its inverse, by radix-2 Cooley–Tukey on one table of cosines.
 */

#ifndef FFT_H
#define FFT_H

// Internal
#include "model/sizes.h"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Fills the cosine table. Call it once, before any other function here.
 */
void fft_setup(void);

/**
 * The DFT of a real series, as `torch.fft.rfft` returns it: bins 0 to 256.
 *
 * @param x The series, 512 values.
 * @param real Receives the real part of each bin.
 * @param imag Receives the imaginary part of each bin.
 */
void fft_forward(
    const float x[CONTEXT_LENGTH], float real[FFT_BINS], float imag[FFT_BINS]
);

/**
 * The real series whose DFT is the given bins, as `torch.fft.irfft` with n=512 returns it.
 *
 * The imaginary parts of bins 0 and 256 are ignored, as `irfft` ignores them.
 *
 * @param real The real part of bins 0 to 256.
 * @param imag The imaginary part of bins 0 to 256.
 * @param x Receives the series, 512 values.
 */
void fft_inverse(
    const float real[FFT_BINS],
    const float imag[FFT_BINS],
    float x[CONTEXT_LENGTH]
);

#endif
