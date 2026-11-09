/**
 * @file
 * The real FFT over 512 values and its inverse, by radix-2 Cooley–Tukey on one table of cosines.
 *
 * Both directions run a complex FFT of 512 points: the forward one with a zero imaginary part, the inverse one on the full spectrum rebuilt from its conjugate symmetry.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>

// Internal
#include "kernels/fft.h"
#include "model/sizes.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Points of the transform.
#define FFT_SIZE CONTEXT_LENGTH

/// Halvings from 512 points down to 1: 512 = 2^9.
#define FFT_STAGES 9

_Static_assert(
    (1 << FFT_STAGES) == FFT_SIZE, "FFT_STAGES must be log2 of FFT_SIZE"
);

/// A quarter turn in table steps: sin(2*pi*j/N) = cos(2*pi*(j - N/4)/N).
#define QUARTER_TURN (FFT_SIZE / 4)

/// The highest bin of a real series, whose imaginary part `irfft` ignores.
#define NYQUIST_BIN (FFT_SIZE / 2)

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// The sign of the twiddles' angle, which is all that separates the two directions.
enum fft_direction {
    FFT_FORWARD, ///< Twiddles e^(-2*pi*i*j/N), giving the spectrum.
    FFT_INVERSE, ///< Twiddles e^(+2*pi*i*j/N), giving N times the series.
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/// cos(2*pi*j/512) for j = 0 to 511, rounded to float once from a double. `fft_setup` fills it at start and nothing writes it after.
static float cosines[FFT_SIZE];

/**
 * cos(2*pi*j/512) for any non-negative j.
 *
 * The angle repeats every 512 steps, so j is reduced mod 512 before indexing, which lands any index in the table.
 *
 * @param j The angle in steps of 2*pi/512.
 * @return The cosine.
 */
static float cosine(int j) { return cosines[j % FFT_SIZE]; }

/**
 * sin(2*pi*j/512) for any non-negative j, read from the cosine table a quarter turn back.
 *
 * @param j The angle in steps of 2*pi/512.
 * @return The sine.
 */
static float sine(int j) {
    return cosines[(j + FFT_SIZE - QUARTER_TURN) % FFT_SIZE];
}

/**
 * Reverses the order of the 9 low bits of an index, such as 1 = 000000001 into 256 = 100000000.
 *
 * @param index An index from 0 to 511.
 * @return The index with its bits reversed.
 */
static int bit_reverse(int index) {
    int reversed = 0;
    for (int bit = 0; bit < FFT_STAGES; bit++) {
        reversed = (reversed << 1) | ((index >> bit) & 1);
    }
    return reversed;
}

/**
 * An in-place complex FFT of 512 points, iterative radix-2 decimation in time.
 *
 * @param real The real parts, replaced by those of the transform.
 * @param imag The imaginary parts, replaced by those of the transform.
 * @param direction Which sign the twiddles' angle takes.
 */
static void fft_complex(
    float real[FFT_SIZE], float imag[FFT_SIZE], enum fft_direction direction
) {
    // Step 1: put the input in bit-reversed order, so each stage combines neighbouring blocks
    for (int i = 0; i < FFT_SIZE; i++) {
        int j = bit_reverse(i);
        if (i < j) {
            float swap_real = real[i];
            float swap_imag = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = swap_real;
            imag[j] = swap_imag;
        }
    }

    // Step 2: 9 stages, each merging pairs of transforms of `half` points into one of `span` points
    // X[k] = E[k] + W^k * O[k] and X[k + span/2] = E[k] - W^k * O[k]
    // W = e^(-2*pi*i/span) for the forward transform, and e^(+2*pi*i/span) for the inverse
    // W^k for a span of `span` is entry k * (512 / span) of the 512-point table
    float sign = direction == FFT_FORWARD ? -1.0f : 1.0f;
    for (int span = 2; span <= FFT_SIZE; span *= 2) {
        int half = span / 2;
        int stride = FFT_SIZE / span;
        for (int start = 0; start < FFT_SIZE; start += span) {
            for (int k = 0; k < half; k++) {
                float twiddle_real = cosine(k * stride);
                float twiddle_imag = sign * sine(k * stride);
                int even = start + k;
                int odd = even + half;

                // W^k * O[k], a complex product
                float product_real =
                    twiddle_real * real[odd] - twiddle_imag * imag[odd];
                float product_imag =
                    twiddle_real * imag[odd] + twiddle_imag * real[odd];

                real[odd] = real[even] - product_real;
                imag[odd] = imag[even] - product_imag;
                real[even] = real[even] + product_real;
                imag[even] = imag[even] + product_imag;
            }
        }
    }
}

void fft_setup(void) {
    // The table is built in double so each entry is the float nearest the true cosine
    for (int j = 0; j < FFT_SIZE; j++) {
        cosines[j] = (float)cos(2.0 * M_PI * j / FFT_SIZE);
    }
}

// X[k] = sum_t x[t] * e^(-2*pi*i*k*t/N), for k = 0 to N/2, computed by radix-2 Cooley–Tukey in N/2 * log2(N) butterflies
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
void fft_forward(
    const float x[CONTEXT_LENGTH], float real[FFT_BINS], float imag[FFT_BINS]
) {
    static float work_real[FFT_SIZE];
    static float work_imag[FFT_SIZE];

    // Step 1: the series as complex values with no imaginary part
    for (int t = 0; t < FFT_SIZE; t++) {
        work_real[t] = x[t];
        work_imag[t] = 0.0f;
    }

    // Step 2: the transform, of which bins 0 to 256 are kept, since a real series' upper bins are conjugates of these
    fft_complex(work_real, work_imag, FFT_FORWARD);
    for (int k = 0; k < FFT_BINS; k++) {
        real[k] = work_real[k];
        imag[k] = work_imag[k];
    }
}

// x[t] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(+2*pi*i*k*t/N), with X[N - k] = conj(X[k]) for a real series
// https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Inverse_transform
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
void fft_inverse(
    const float real[FFT_BINS],
    const float imag[FFT_BINS],
    float x[CONTEXT_LENGTH]
) {
    static float work_real[FFT_SIZE];
    static float work_imag[FFT_SIZE];

    // Step 1: bins 0 to 256 as given, but bins 0 and 256 of a real series are real, so their imaginary parts are dropped as irfft drops them
    for (int k = 0; k < FFT_BINS; k++) {
        work_real[k] = real[k];
        work_imag[k] = imag[k];
    }
    work_imag[0] = 0.0f;
    work_imag[NYQUIST_BIN] = 0.0f;

    // Step 2: bins 257 to 511, the conjugates of bins 255 down to 1
    for (int k = 1; k < NYQUIST_BIN; k++) {
        work_real[FFT_SIZE - k] = real[k];
        work_imag[FFT_SIZE - k] = -imag[k];
    }

    // Step 3: the transform with the opposite angle, whose real part is N times the series
    fft_complex(work_real, work_imag, FFT_INVERSE);
    for (int t = 0; t < FFT_SIZE; t++) {
        x[t] = work_real[t] / (float)FFT_SIZE;
    }
}
