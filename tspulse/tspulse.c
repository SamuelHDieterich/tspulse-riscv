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
 * The dump build adds `--dump-window W --dump PATH`, which writes window W's inner outputs to a `.tdump` file.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal
#include "io/rows.h"
#include "io/tdump.h"
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
    uint32_t start;          ///< The first window to run.
    uint32_t stop;           ///< One past the last window to run.
    int has_stop;            ///< Whether `--windows` gave a stop, else all windows run.
#ifdef TSPULSE_DUMP
    uint32_t dump_window;    ///< The window whose inner outputs are dumped.
    const char *dump_path;   ///< The `.tdump` file to write.
    int has_dump_window;     ///< Whether `--dump-window` was given.
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
/// The base the window numbers are parsed in.
#define DECIMAL_BASE 10
/// The first line of the errors CSV.
#define ERRORS_HEADER "window,time,fft,forecast\n"

// Nine significant digits are enough to read any float32 back exactly
#define ERRORS_LINE "%u,%.9g,%.9g,%.9g\n"

#ifdef TSPULSE_DUMP
#define USAGE                                                                  \
    "usage: %s ROWS ERRORS [--windows START:STOP] [--dump-window W --dump "    \
    "PATH]\n"
#else
#define USAGE "usage: %s ROWS ERRORS [--windows START:STOP]\n"
#endif

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
 * @param dump The dump to write into, NULL when none was asked for. The clean build passes NULL.
 * @param dumped Receives whether the dump window was run.
 * @return 0 on success, nonzero on a read or write failure.
 */
static int run_series(
    const struct options *options,
    struct rows_reader *reader,
    FILE *errors,
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
        struct window_errors result;
        if (window_run(&history, window_dump, &result)) {
            fprintf(stderr, "cannot dump window %u\n", window);
            failed = 1;
            break;
        }

        // Step 4: the errors; %.9g promotes the floats to double, which the model itself never does
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
    struct tdump *dump_target = NULL;
#ifdef TSPULSE_DUMP
    struct tdump dump;
    if (options.has_dump_window) {
        if (tdump_open(&dump, options.dump_path)) {
            fprintf(stderr, "cannot write %s\n", options.dump_path);
            fclose(errors);
            rows_close(&reader);
            return 1;
        }
        dump_target = &dump;
    }
#endif

    // Step 4: the series
    int dumped = 0;
    int failed = run_series(&options, &reader, errors, dump_target, &dumped);

    // Step 5: close everything, and check that the dump window was reached
    if (fclose(errors) != 0) {
        fprintf(stderr, "cannot finish %s\n", options.errors_path);
        failed = 1;
    }
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
