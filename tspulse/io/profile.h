/**
 * @file
 * Hooks that count the instructions each PyTorch module takes, by reading `instret` where the dump build writes records.
 *
 * Only the profile build compiles `profile.c` and calls these hooks, and only for rv64.
 * A module's path is the copy, `hidden` or `plain`, then a `/`, then the dot-joined PyTorch names, such as `hidden/backbone.encoder_block`.
 * The twelve hidden copies share one path, so their rows add up.
 * Each row holds the instructions spent inside the module, children included, with the cost of every hook subtracted.
 */

#ifndef PROFILE_H
#define PROFILE_H

// Built-in
#include <stdint.h>
#include <stdio.h>

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Appends a path segment, formatted as by printf, and starts counting the module's instructions.
 *
 * A segment ending in `/` names a copy, and loses its digits so that every hidden copy lands on `hidden/`.
 *
 * @param format A printf format for the segment.
 * @param ... Its arguments.
 */
void profile_push(const char *format, ...)
    __attribute__((format(printf, 1, 2)));

/**
 * Stops counting the module the last push opened, adds its instructions to its row and removes the segment.
 */
void profile_pop(void);

/**
 * Writes one window's rows and clears the table and the hook counters, ready for the next window.
 *
 * The first row is `window`, the instructions of the whole window with every hook's cost subtracted.
 * The second is `raw`, the same count with nothing subtracted, whose `calls` is the number of hook calls.
 * The module rows follow in the order their paths first appeared, with `depth` the number of pushes open, so a copy is depth 1.
 *
 * @param out The open CSV, its header already written.
 * @param window The window's number.
 * @param raw The instructions the window took, as read around the model run.
 * @return 0 on success, nonzero if a write failed or the hooks overflowed the path, the stack or the table.
 */
int profile_write(FILE *out, uint32_t window, uint64_t raw);

#endif
