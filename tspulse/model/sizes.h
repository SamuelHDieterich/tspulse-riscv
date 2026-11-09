/**
 * @file
 * The sizes of the model's input, shared by every file that handles a window.
 */

#ifndef SIZES_H
#define SIZES_H

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Rows the model reads as its past.
#define CONTEXT_LENGTH 512

/// Rows in a window: the past, then the row the forecast head is scored against.
#define WINDOW_LENGTH (CONTEXT_LENGTH + 1)

#endif
