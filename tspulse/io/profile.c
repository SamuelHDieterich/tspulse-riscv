/**
 * @file
 * Counts the instructions each PyTorch module takes, from `instret` reads at the dump hooks.
 *
 * Each hook reads `instret` first and last, so everything between the two reads is the hook's own cost.
 * That cost is added to a running total, and a module's count loses the share of that total accumulated while it was open.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <ctype.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Internal
#include "io/instret.h"
#include "io/profile.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Room for the longest module path with margin, such as `hidden/backbone.encoder_block.mixers.7.feature_mixer.gating_block.attn_activation_layer`.
#define PATH_CAPACITY 256

/// Path segments deep enough for the deepest module.
#define DEPTH_CAPACITY 16

/// Modules the table can hold, with room to spare over the paths one window opens.
#define ROW_CAPACITY 512

/// The separator between the copy and the module path, which ends a copy's segment.
#define COPY_SEPARATOR '/'

/// The separator between module path segments.
#define SEGMENT_SEPARATOR "."

/// The path of the row that holds the window's count with every hook's cost subtracted.
#define WINDOW_PATH "window"

/// The path of the row that holds the window's count with nothing subtracted.
#define RAW_PATH "raw"

/// The depth of the two rows that cover the whole window.
#define ROOT_DEPTH 0

/// One line of the profile CSV.
#define ROW_LINE "%u,%s,%d,%" PRIu64 ",%" PRIu64 "\n"

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// A module that is open: where its segment starts in the path, and the counters at the end of its push.
struct frame {
    // clang-format off: one comment column for the whole struct, which clang-format splits at each line without a trailing comment
    size_t mark;            ///< The length of the path before the segment, to restore on pop.
    uint64_t start;         ///< `instret` at the end of the push.
    uint64_t hook_at_start; ///< The hook cost total at the end of the push.
    size_t row;             ///< The table row of the module's path, or ROW_CAPACITY when the table was full.
    // clang-format on
};

/// One module's row of the table.
struct row {
    // clang-format off: one comment column for the whole struct, which clang-format splits at each line without a trailing comment
    char path[PATH_CAPACITY]; ///< The module's path.
    int depth;                ///< Pushes open when the module was entered.
    uint64_t instructions;    ///< Instructions inside the module, children included, hooks subtracted.
    uint64_t calls;           ///< Times the module was entered.
    // clang-format on
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// The hooks are called from deep in the module code, which takes no profile argument, so their state is file-wide

/// The current path, NUL-terminated.
static char path[PATH_CAPACITY];
/// The modules entered since the last write.
static struct row rows[ROW_CAPACITY];
/// Rows in use.
static size_t row_count;
/// The modules open now, outermost first.
static struct frame frames[DEPTH_CAPACITY];
/// Modules open.
static int depth;
/// Instructions the hooks spent on themselves since the last write.
static uint64_t hook_cost;
/// Hook calls since the last write.
static uint64_t hook_calls;
/// Whether a path, the stack or the table overflowed, or a pop had no push, since the last write.
static int failed;

/**
 * Cuts the digits off a copy's segment, so `hidden03/` becomes `hidden/`.
 *
 * @param segment A formatted segment, changed in place if it ends with the copy separator.
 */
static void collapse_copy(char *segment) {
    size_t length = strlen(segment);
    if (length == 0 || segment[length - 1] != COPY_SEPARATOR) {
        return;
    }
    size_t end = length - 1;
    while (end > 0 && isdigit((unsigned char)segment[end - 1])) {
        end--;
    }
    segment[end] = COPY_SEPARATOR;
    segment[end + 1] = '\0';
}

/**
 * Finds the row of a path, adding it when it is new.
 *
 * @param module_path The module's path.
 * @param module_depth Pushes open when the module is entered.
 * @return The row's index, or ROW_CAPACITY when the table is full.
 */
static size_t find_row(const char *module_path, int module_depth) {
    for (size_t i = 0; i < row_count; i++) {
        if (strcmp(rows[i].path, module_path) == 0) {
            return i;
        }
    }
    if (row_count >= ROW_CAPACITY) {
        failed = 1;
        return ROW_CAPACITY;
    }
    struct row *added = &rows[row_count];
    strcpy(added->path, module_path);
    added->depth = module_depth;
    return row_count++;
}

void profile_push(const char *format, ...) {
    // Step 1: the first read, so everything up to the last one is this hook's cost
    uint64_t entered = instret_read();
    hook_calls++;
    if (depth >= DEPTH_CAPACITY) {
        failed = 1;
        return;
    }

    // Step 2: the formatted segment, with a copy's digits cut off
    char segment[PATH_CAPACITY];
    va_list args;
    va_start(args, format);
    int written = vsnprintf(segment, sizeof(segment), format, args);
    va_end(args);
    if (written < 0 || (size_t)written >= sizeof(segment)) {
        failed = 1;
        return;
    }
    collapse_copy(segment);

    // Step 3: the path, joined with a dot unless it is empty or ends with the copy's separator
    size_t length = strlen(path);
    int needs_separator = length > 0 && path[length - 1] != COPY_SEPARATOR;
    size_t added = strlen(segment) + (needs_separator ? 1 : 0);
    if (length + added >= PATH_CAPACITY) {
        failed = 1;
        return;
    }
    if (needs_separator) {
        strcpy(path + length, SEGMENT_SEPARATOR);
    }
    strcat(path, segment);

    // Step 4: the module's row, and the frame that pop closes
    struct frame *frame = &frames[depth++];
    frame->mark = length;
    frame->row = find_row(path, depth);
    if (frame->row < ROW_CAPACITY) {
        rows[frame->row].calls++;
    }

    // Step 5: the last read. The module's count starts there, after this hook's cost went into the total
    uint64_t left = instret_read();
    hook_cost += left - entered;
    frame->hook_at_start = hook_cost;
    frame->start = left;
}

void profile_pop(void) {
    // Step 1: the first read, which ends the module's count
    uint64_t entered = instret_read();
    hook_calls++;
    if (depth == 0) {
        failed = 1;
        return;
    }

    // Step 2: the module's instructions, less every hook that ran while it was open
    struct frame *frame = &frames[--depth];
    uint64_t hooks_inside = hook_cost - frame->hook_at_start;
    if (frame->row < ROW_CAPACITY) {
        rows[frame->row].instructions += entered - frame->start - hooks_inside;
    }
    path[frame->mark] = '\0';

    // Step 3: the last read, and this hook's cost into the total
    hook_cost += instret_read() - entered;
}

int profile_write(FILE *out, uint32_t window, uint64_t raw) {
    // Step 1: the two rows that cover the whole window
    int bad = failed || depth != 0;
    bad |= fprintf(
               out, ROW_LINE, window, WINDOW_PATH, ROOT_DEPTH, raw - hook_cost,
               (uint64_t)1
           ) < 0;
    bad |=
        fprintf(out, ROW_LINE, window, RAW_PATH, ROOT_DEPTH, raw, hook_calls) <
        0;

    // Step 2: the modules, a copy's row without its trailing separator
    for (size_t i = 0; i < row_count; i++) {
        char name[PATH_CAPACITY];
        strcpy(name, rows[i].path);
        size_t length = strlen(name);
        if (length > 0 && name[length - 1] == COPY_SEPARATOR) {
            name[length - 1] = '\0';
        }
        bad |= fprintf(
                   out, ROW_LINE, window, name, rows[i].depth,
                   rows[i].instructions, rows[i].calls
               ) < 0;
    }

    // Step 3: clear everything for the next window
    memset(rows, 0, sizeof(rows[0]) * row_count);
    path[0] = '\0';
    row_count = 0;
    depth = 0;
    hook_cost = 0;
    hook_calls = 0;
    failed = 0;
    return bad;
}
