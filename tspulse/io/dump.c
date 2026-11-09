/**
 * @file
 * Hooks that write the model's inner outputs to a `.tdump` file, named as the PyTorch reference names them.
 *
 * Only the dump build compiles this file.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Internal
#include "io/dump.h"
#include "io/tdump.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Room for the longest record name with margin, such as `hidden11/backbone.encoder_block.mixers.7.feature_mixer.gating_block.attn_activation_layer`.
#define NAME_CAPACITY 256

/// Path segments deep enough for the deepest module.
#define DEPTH_CAPACITY 16

/// The separator between the copy and the module path.
#define COPY_SEPARATOR '/'

/// The separator between module path segments.
#define SEGMENT_SEPARATOR "."

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// Where the hooks write, and the path that names the next record.
struct dump_state {
    // clang-format off: one comment column for the whole struct, which clang-format splits at each line without a trailing comment
    struct tdump *target;     ///< The writer, or NULL when nothing is dumped.
    uint32_t channels;        ///< The window's channel count.
    int failed;               ///< Whether a write or a path failed.
    char path[NAME_CAPACITY]; ///< The current path, NUL-terminated.
    /// The length of the path before each pushed segment, to restore on pop.
    size_t marks[DEPTH_CAPACITY];
    int depth;                ///< Segments pushed.
    // clang-format on
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/// Where the hooks write and the current path. The hooks are called from deep in the module code, which takes no dump argument, so their state is file-wide.
static struct dump_state state;

/**
 * Whether a new segment joins the path with a dot.
 *
 * @param length The path's length.
 * @return Nonzero unless the path is empty or ends with the copy separator.
 */
static int needs_separator(size_t length) {
    return length > 0 && state.path[length - 1] != COPY_SEPARATOR;
}

void dump_begin_channel(
    struct tdump *target, uint32_t channel, uint32_t channels
) {
    state.target = target;
    state.channels = channels;
    state.path[0] = '\0';
    state.depth = 0;
    if (channel == 0) {
        state.failed = 0;
    }
    if (target != NULL) {
        tdump_begin_channel(target, channel, channels);
    }
}

void dump_push(const char *format, ...) {
    if (state.target == NULL) {
        return;
    }
    if (state.depth >= DEPTH_CAPACITY) {
        state.failed = 1;
        return;
    }

    // Step 1: remember where the segment starts, so pop can cut it off
    size_t length = strlen(state.path);
    state.marks[state.depth++] = length;

    // Step 2: the separator, then the formatted segment
    int written = 0;
    if (needs_separator(length)) {
        written = snprintf(
            state.path + length, NAME_CAPACITY - length, SEGMENT_SEPARATOR
        );
        length += (size_t)written;
    }
    va_list args;
    va_start(args, format);
    written =
        vsnprintf(state.path + length, NAME_CAPACITY - length, format, args);
    va_end(args);

    // A truncated path would write a record under the wrong name
    if (written < 0 || (size_t)written >= NAME_CAPACITY - length) {
        state.failed = 1;
    }
}

void dump_pop(void) {
    if (state.target == NULL || state.depth == 0) {
        return;
    }
    state.path[state.marks[--state.depth]] = '\0';
}

void dump_record(
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims,
    int channel_axis
) {
    if (state.target == NULL) {
        return;
    }

    // Step 1: the full name, the path alone when the name is empty
    char full[NAME_CAPACITY];
    size_t length = strlen(state.path);
    int written;
    if (name[0] == '\0') {
        written = snprintf(full, NAME_CAPACITY, "%s", state.path);
    } else if (needs_separator(length)) {
        written = snprintf(
            full, NAME_CAPACITY, "%s" SEGMENT_SEPARATOR "%s", state.path, name
        );
    } else {
        written = snprintf(full, NAME_CAPACITY, "%s%s", state.path, name);
    }
    if (written < 0 || written >= NAME_CAPACITY) {
        state.failed = 1;
        return;
    }

    // Step 2: the values
    if (tdump_write_channel(
            state.target, full, data, ndim, dims, channel_axis
        )) {
        state.failed = 1;
    }
}

uint32_t dump_channels(void) { return state.channels; }

int dump_failed(void) { return state.failed; }
