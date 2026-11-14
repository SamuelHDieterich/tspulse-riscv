/**
 * @file
 * Scores a series with the TSPulse model, one window per row after the first 512.
 *
 * Usage: tspulse ROWS ERRORS [--windows START:STOP]
 *
 * ROWS is a `.rows` file from `prepare-rows`. ERRORS receives a CSV with one line per window.
 * Window `w` covers rows `w` to `w + 512`, and the series has `rows - 512` windows.
 * START and STOP count windows, and STOP is not run. Windows before START are read but not run.
 *
 * The RISC-V build adds `--counts PATH`, which receives a CSV with the instructions each window's model run took.
 *
 * The dump build adds `--dump-window W --dump PATH`, which writes window W's inner outputs to a `.tdump` file.
 *
 * The profile build adds `--profile PATH`, which receives a CSV with the instructions each module took in each window.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal
#include "io/instret.h"
#include "io/profile.h"
#include "io/rows.h"
#include "io/tdump.h"
#include "kernels/fft.h"
#include "model/history.h"
#include "model/sizes.h"
#include "model/window.h"

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// What the command line asked for.
struct options {
    // clang-format off: one comment column for the whole struct, which clang-format splits at each line without a trailing comment
    const char *rows_path;   ///< The `.rows` file to read.
    const char *errors_path; ///< The CSV to write.
#ifdef __riscv
    const char *counts_path; ///< The CSV of instruction counts to write, NULL when not asked for.
#endif
    uint32_t start;          ///< The first window to run.
    uint32_t stop;           ///< One past the last window to run.
    int has_stop;            ///< Whether `--windows` gave a stop, else all windows run.
#ifdef TSPULSE_DUMP
    uint32_t dump_window;    ///< The window whose inner outputs are dumped.
    const char *dump_path;   ///< The `.tdump` file to write.
    int has_dump_window;     ///< Whether `--dump-window` was given.
#endif
#ifdef TSPULSE_PROFILE
    const char *profile_path; ///< The CSV of per-module instruction counts to write, NULL when not asked for.
#endif
    // clang-format on
};

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The option that picks the range of windows.
#define OPTION_WINDOWS "--windows"
/// The option that picks the window whose inner outputs are dumped.
#define OPTION_DUMP_WINDOW "--dump-window"
/// The option that names the `.tdump` file.
#define OPTION_DUMP "--dump"
#ifdef TSPULSE_PROFILE
/// The option that names the per-module instruction counts CSV.
#define OPTION_PROFILE "--profile"
/// The first line of the profile CSV.
#define PROFILE_HEADER "window,path,depth,instructions,calls\n"
#endif
#ifdef __riscv
/// The option that names the instruction counts CSV.
#define OPTION_COUNTS "--counts"
/// The first line of the counts CSV.
#define COUNTS_HEADER "window,instructions\n"
/// One line of the counts CSV.
#define COUNTS_LINE "%u,%" PRIu64 "\n"
#endif
/// The base the window numbers are parsed in.
#define DECIMAL_BASE 10
/// The first line of the errors CSV.
#define ERRORS_HEADER "window,time,fft,forecast\n"

// Nine significant digits are enough to read any float32 back exactly
#define ERRORS_LINE "%u,%.9g,%.9g,%.9g\n"

// The usage line lists only the options this build has
#ifdef __riscv
#define USAGE_COUNTS " [--counts PATH]"
#else
#define USAGE_COUNTS ""
#endif
#ifdef TSPULSE_DUMP
#define USAGE_DUMP " [--dump-window W --dump PATH]"
#else
#define USAGE_DUMP ""
#endif
#ifdef TSPULSE_PROFILE
#define USAGE_PROFILE " [--profile PATH]"
#else
#define USAGE_PROFILE ""
#endif
#define USAGE                                                                  \
    "usage: %s ROWS ERRORS [--windows START:STOP]" USAGE_COUNTS                \
        USAGE_DUMP USAGE_PROFILE "\n"

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Parses a non-negative decimal integer that fills the whole text.
 *
 * @param text The text to parse.
 * @param end The character the number must stop at, either ':' or the end of the text.
 * @param value Receives the number.
 * @param next Receives the position after the number.
 * @return 0 on success, nonzero if the text holds no number or the number is too big.
 */
static int
parse_uint(const char *text, char end, uint32_t *value, const char **next) {
    if (text[0] < '0' || text[0] > '9') {
        return 1;
    }
    char *stop;
    errno = 0;
    unsigned long parsed = strtoul(text, &stop, DECIMAL_BASE);
    if (errno != 0 || parsed > UINT32_MAX || *stop != end) {
        return 1;
    }
    *value = (uint32_t)parsed;
    *next = stop;
    return 0;
}

/**
 * Parses `START:STOP`.
 *
 * @param text The text after `--windows`.
 * @param options Receives the start and stop.
 * @return 0 on success, nonzero if the text is malformed or the range is empty.
 */
static int parse_range(const char *text, struct options *options) {
    const char *next;
    if (parse_uint(text, ':', &options->start, &next) ||
        parse_uint(next + 1, '\0', &options->stop, &next)) {
        return 1;
    }
    options->has_stop = 1;
    return options->start >= options->stop;
}

/**
 * Parses the command line.
 *
 * @param argc The argument count.
 * @param argv The arguments.
 * @param options Receives what was asked for.
 * @return 0 on success, nonzero if an argument is missing, unknown or malformed.
 */
static int parse_args(int argc, char **argv, struct options *options) {
    if (argc < 3) {
        return 1;
    }
    memset(options, 0, sizeof(*options));
    options->rows_path = argv[1];
    options->errors_path = argv[2];

    for (int i = 3; i < argc; i += 2) {
        // Every option takes one value
        if (i + 1 >= argc) {
            return 1;
        }
        const char *value = argv[i + 1];
        if (strcmp(argv[i], OPTION_WINDOWS) == 0) {
            if (parse_range(value, options)) {
                return 1;
            }
#ifdef __riscv
        } else if (strcmp(argv[i], OPTION_COUNTS) == 0) {
            options->counts_path = value;
#endif
#ifdef TSPULSE_DUMP
        } else if (strcmp(argv[i], OPTION_DUMP_WINDOW) == 0) {
            const char *next;
            if (parse_uint(value, '\0', &options->dump_window, &next)) {
                return 1;
            }
            options->has_dump_window = 1;
        } else if (strcmp(argv[i], OPTION_DUMP) == 0) {
            options->dump_path = value;
#endif
#ifdef TSPULSE_PROFILE
        } else if (strcmp(argv[i], OPTION_PROFILE) == 0) {
            options->profile_path = value;
#endif
        } else {
            return 1;
        }
    }

#ifdef TSPULSE_DUMP
    // The window and the file only mean something together
    return options->has_dump_window != (options->dump_path != NULL);
#else
    return 0;
#endif
}

/**
 * Reads the series and runs the windows in range.
 *
 * @param options What the command line asked for.
 * @param reader An open `.rows` file.
 * @param errors The CSV, with its header already written.
 * @param counts The instruction counts CSV with its header already written, NULL when none was asked for. The host build passes NULL.
 * @param profile The profile CSV with its header already written, NULL when none was asked for. Only the profile build uses it.
 * @param dump The dump to write into, NULL when none was asked for. The clean build passes NULL.
 * @param dumped Receives whether the dump window was run.
 * @return 0 on success, nonzero on a read or write failure.
 */
static int run_series(
    const struct options *options,
    struct rows_reader *reader,
    FILE *errors,
    FILE *counts,
    FILE *profile,
    struct tdump *dump,
    int *dumped
) {
    // Step 1: the history, which holds the 513 rows of the window being run
    struct history history;
    if (history_init(&history, reader->channels, reader->mean, reader->std)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    double *row = malloc(reader->channels * sizeof(double));
    if (row == NULL) {
        fprintf(stderr, "out of memory\n");
        history_free(&history);
        return 1;
    }

    // Step 2: one row at a time. A row completes window `rows read - 513`, and windows before START are only read
    uint32_t stop = options->has_stop ? options->stop : UINT32_MAX;
    int failed = 0;
    for (uint32_t r = 0; r < reader->rows && !failed; r++) {
        if (rows_next(reader, row)) {
            fprintf(stderr, "short read at row %u\n", r);
            failed = 1;
            break;
        }
        history_push(&history, row);
        if (!history_full(&history)) {
            continue;
        }
        uint32_t window = r - CONTEXT_LENGTH;
        if (window < options->start) {
            continue;
        }
        if (window >= stop) {
            break;
        }

        // Step 3: the window, dumped only when it is the one asked for
        struct tdump *window_dump = NULL;
#ifdef TSPULSE_DUMP
        if (options->has_dump_window && window == options->dump_window) {
            window_dump = dump;
            *dumped = 1;
        }
#endif
        // Only the model work is measured: reading rows and writing results is file I/O through the proxy kernel
#ifdef __riscv
        uint64_t before = instret_read();
#endif
        struct window_errors result;
        int run_failed = window_run(&history, window_dump, &result);
#ifdef __riscv
        uint64_t after = instret_read();
#endif
        if (run_failed) {
            fprintf(stderr, "cannot dump window %u\n", window);
            failed = 1;
            break;
        }

        // Step 4: the instruction count, when asked for
#ifdef __riscv
        if (counts != NULL &&
            fprintf(counts, COUNTS_LINE, window, after - before) < 0) {
            fprintf(stderr, "cannot write the counts\n");
            failed = 1;
            break;
        }
#endif

        // Step 5: the module counts, when asked for. The window's count is the same one `--counts` writes, and profile_write takes the hooks' cost out of it
#ifdef TSPULSE_PROFILE
        if (profile != NULL && profile_write(profile, window, after - before)) {
            fprintf(
                stderr, "cannot write the profile, or a hook overflowed or was "
                        "unpaired\n"
            );
            failed = 1;
            break;
        }
#endif

        // Step 6: the errors; %.9g promotes the floats to double, which the model itself never does
        if (fprintf(
                errors, ERRORS_LINE, window, (double)result.time,
                (double)result.fft, (double)result.forecast
            ) < 0) {
            fprintf(stderr, "cannot write the errors\n");
            failed = 1;
        }
    }

    free(row);
    history_free(&history);
#ifndef TSPULSE_DUMP
    // The clean build has no dump code
    (void)dump;
    (void)dumped;
#endif
#ifndef __riscv
    // The host build has no counter
    (void)counts;
#endif
#ifndef TSPULSE_PROFILE
    // Only the profile build has hooks that count
    (void)profile;
#endif
    return failed;
}

// ░█▄█░█▀█░▀█▀░█▀█
// ░█░█░█▀█░░█░░█░█
// ░▀░▀░▀░▀░▀▀▀░▀░▀

int main(int argc, char **argv) {
    // Step 1: the command line
    struct options options;
    if (parse_args(argc, argv, &options)) {
        fprintf(stderr, USAGE, argv[0]);
        return 1;
    }

    // Step 2: the series, and the range of windows it allows
    struct rows_reader reader;
    if (rows_open(&reader, options.rows_path)) {
        fprintf(stderr, "cannot read %s\n", options.rows_path);
        return 1;
    }
    if (reader.rows <= CONTEXT_LENGTH) {
        fprintf(
            stderr, "%s has %u rows, a window needs %d\n", options.rows_path,
            reader.rows, WINDOW_LENGTH
        );
        rows_close(&reader);
        return 1;
    }
    uint32_t windows = reader.rows - CONTEXT_LENGTH;
    if (options.start >= windows ||
        (options.has_stop && options.stop > windows)) {
        fprintf(
            stderr, "windows are 0 to %u, so STOP is at most %u\n", windows - 1,
            windows
        );
        rows_close(&reader);
        return 1;
    }

    // Step 3: the outputs
    FILE *errors = fopen(options.errors_path, "w");
    if (errors == NULL || fputs(ERRORS_HEADER, errors) == EOF) {
        fprintf(stderr, "cannot write %s\n", options.errors_path);
        if (errors != NULL) {
            fclose(errors);
        }
        rows_close(&reader);
        return 1;
    }
    FILE *counts = NULL;
#ifdef __riscv
    if (options.counts_path != NULL) {
        counts = fopen(options.counts_path, "w");
        if (counts == NULL || fputs(COUNTS_HEADER, counts) == EOF) {
            fprintf(stderr, "cannot write %s\n", options.counts_path);
            if (counts != NULL) {
                fclose(counts);
            }
            fclose(errors);
            rows_close(&reader);
            return 1;
        }
    }
#endif
    FILE *profile = NULL;
#ifdef TSPULSE_PROFILE
    if (options.profile_path != NULL) {
        profile = fopen(options.profile_path, "w");
        if (profile == NULL || fputs(PROFILE_HEADER, profile) == EOF) {
            fprintf(stderr, "cannot write %s\n", options.profile_path);
            if (profile != NULL) {
                fclose(profile);
            }
            if (counts != NULL) {
                fclose(counts);
            }
            fclose(errors);
            rows_close(&reader);
            return 1;
        }
    }
#endif
    struct tdump *dump_target = NULL;
#ifdef TSPULSE_DUMP
    struct tdump dump;
    if (options.has_dump_window) {
        if (tdump_open(&dump, options.dump_path)) {
            fprintf(stderr, "cannot write %s\n", options.dump_path);
            if (counts != NULL) {
                fclose(counts);
            }
            fclose(errors);
            rows_close(&reader);
            return 1;
        }
        dump_target = &dump;
    }
#endif

    // Step 4: the series. Every window reads the FFT's cosine table, so it is filled first
    fft_setup();
    int dumped = 0;
    int failed = run_series(
        &options, &reader, errors, counts, profile, dump_target, &dumped
    );

    // Step 5: close everything, and check that the dump window was reached
    if (fclose(errors) != 0) {
        fprintf(stderr, "cannot finish %s\n", options.errors_path);
        failed = 1;
    }
#ifdef __riscv
    if (counts != NULL && fclose(counts) != 0) {
        fprintf(stderr, "cannot finish %s\n", options.counts_path);
        failed = 1;
    }
#endif
#ifdef TSPULSE_PROFILE
    if (profile != NULL && fclose(profile) != 0) {
        fprintf(stderr, "cannot finish %s\n", options.profile_path);
        failed = 1;
    }
#endif
#ifdef TSPULSE_DUMP
    if (dump_target != NULL && tdump_close(&dump)) {
        fprintf(stderr, "cannot finish %s\n", options.dump_path);
        failed = 1;
    }
    if (options.has_dump_window && !dumped && !failed) {
        fprintf(stderr, "window %u was not run\n", options.dump_window);
        failed = 1;
    }
#endif
    rows_close(&reader);
    return failed;
}
