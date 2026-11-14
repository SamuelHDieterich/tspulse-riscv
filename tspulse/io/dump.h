/**
 * @file
 * Hooks that write the model's inner outputs to a `.tdump` file, named as the PyTorch reference names them.
 *
 * Every hook is a macro that expands to nothing without `TSPULSE_DUMP`, so the clean build carries no dump code.
 * With `TSPULSE_PROFILE` instead, `DUMP_PUSH` and `DUMP_POP` count instructions per module, and every other hook stays empty.
 * A record's name is the current path, built with `DUMP_PUSH` and `DUMP_POP`, then a dot and the name given to `DUMP`.
 * The path starts with the copy, such as `plain/` or `hidden03/`, then the PyTorch module path, such as `backbone.encoder_block`.
 * A module function writes its own output with an empty name, and its parts with their attribute names, after its caller pushed the module's name.
 */

#ifndef DUMP_H
#define DUMP_H

// Built-in
#include <stdint.h>

// Internal
#include "io/profile.h"
#include "io/tdump.h"

#if defined(TSPULSE_DUMP) && defined(TSPULSE_PROFILE)
#error "TSPULSE_DUMP and TSPULSE_PROFILE are different builds, define one"
#endif
#if defined(TSPULSE_PROFILE) && !(defined(__riscv) && __riscv_xlen == 64)
#error                                                                         \
    "TSPULSE_PROFILE reads the instruction counter, which only rv64 has, so it needs a 64-bit RISC-V target"
#endif

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// The dimensions of a record, as an array literal for `DUMP`'s `dims`.
#define DIMS(...) ((const uint32_t[]){__VA_ARGS__})

/// Dimensions of a record inside the network: the channel, then rows, then features.
#define NETWORK_NDIM 3

/// Dimensions of a flattened hidden state: the channel, then its values.
#define FLAT_NDIM 2

/// Dimensions of a time series or a per-channel statistic: rows, then the channel.
#define SERIES_NDIM 2

/// Records inside the network put the channel first.
#define CHANNEL_FIRST 0

/// Time-series records put the channel last, as PyTorch's (batch, time, channel) input does.
#define CHANNEL_LAST (SERIES_NDIM - 1)

#ifdef TSPULSE_DUMP

/// Starts one channel's pass over a window. A NULL `target` makes every other hook do nothing.
#define DUMP_BEGIN_CHANNEL(target, channel, channels)                          \
    dump_begin_channel(target, channel, channels)

/// Appends a path segment, formatted as by printf.
#define DUMP_PUSH(...) dump_push(__VA_ARGS__)

/// Removes the last path segment.
#define DUMP_POP() dump_pop()

/// Writes this channel's share of a record, as `tdump_write_channel` does.
#define DUMP(name, data, ndim, dims, channel_axis)                             \
    dump_record(name, data, ndim, dims, channel_axis)

/// The window's channel count, for the channel dimension in `DIMS`.
#define DUMP_CHANNELS dump_channels()

#elif defined(TSPULSE_PROFILE)

// The profile build counts instructions per module, so only the path hooks do anything
#define DUMP_BEGIN_CHANNEL(target, channel, channels) ((void)0)
#define DUMP_PUSH(...) profile_push(__VA_ARGS__)
#define DUMP_POP() profile_pop()
#define DUMP(name, data, ndim, dims, channel_axis) ((void)0)

#else

#define DUMP_BEGIN_CHANNEL(target, channel, channels) ((void)0)
#define DUMP_PUSH(...) ((void)0)
#define DUMP_POP() ((void)0)
#define DUMP(name, data, ndim, dims, channel_axis) ((void)0)

#endif

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

#ifdef TSPULSE_DUMP

/**
 * Starts one channel's pass over a window: sets the file to write into, clears the path and the failure flag on channel 0, and starts the channel in the writer.
 *
 * @param target The open writer, or NULL when this window isn't dumped.
 * @param channel The channel about to run, counted from 0.
 * @param channels The window's channel count.
 */
void dump_begin_channel(
    struct tdump *target, uint32_t channel, uint32_t channels
);

/**
 * Appends a path segment, joined with a dot unless the path is empty or ends with `/`.
 *
 * @param format A printf format for the segment.
 * @param ... Its arguments.
 */
void dump_push(const char *format, ...) __attribute__((format(printf, 1, 2)));

/**
 * Removes the last path segment.
 */
void dump_pop(void);

/**
 * Writes this channel's share of the record named by the path and `name`.
 *
 * A failed write is remembered for `dump_failed`, so module code needs no error path.
 *
 * @param name The part after the path, or an empty string for the path itself.
 * @param data This channel's values, row-major with the channel dimension left out.
 * @param ndim The number of dimensions, including the channel dimension.
 * @param dims The size of each dimension, with the channel count at `channel_axis`.
 * @param channel_axis The position of the channel dimension, either 0 or `ndim - 1`.
 */
void dump_record(
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims,
    int channel_axis
);

/**
 * The channel count given to `dump_begin_channel`.
 *
 * @return The window's channel count.
 */
uint32_t dump_channels(void);

/**
 * Whether any write, or any path too long for the name buffer, failed since channel 0 began.
 *
 * @return Nonzero after a failure.
 */
int dump_failed(void);

#endif

#endif
