/**
 * @file
 * A writer for the `.tdump` format shared with the Python reference, float32 records only.
 */

#ifndef TDUMP_H
#define TDUMP_H

// Built-in
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

#ifdef TSPULSE_DUMP
/// Where one record sits in the file while its channels are being filled in.
struct tdump_slot {
    /// The record's name, to check that every channel writes the same sequence.
    char *name;
    /// Byte offset of the record's first value.
    long offset;
    /// Values each channel writes, so a call with other dimensions can't run past the reserved body.
    size_t per_channel;
};
#endif

/// An open `.tdump` file.
struct tdump {
    // clang-format off: one comment column for the whole struct, which clang-format splits at each line without a trailing comment
    FILE *file;           ///< The underlying stream.
#ifdef TSPULSE_DUMP
    /// One slot per record of the current window, in channel 0's order.
    struct tdump_slot *slots;
    size_t slot_capacity; ///< Slots allocated.
    size_t next_slot;     ///< The slot the next `tdump_write_channel` call uses.
    uint32_t channel;     ///< The channel being written.
    uint32_t channels;    ///< The channel count of the window.
#endif
    // clang-format on
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Opens a file for writing and writes the header.
 *
 * @param dump The writer to set up.
 * @param path The file to write. It is truncated if it exists.
 * @return 0 on success, nonzero on failure.
 */
int tdump_open(struct tdump *dump, const char *path);

/**
 * Appends one float32 record.
 *
 * @param dump An open writer.
 * @param name The record name, at most 65535 bytes.
 * @param data The values, row-major.
 * @param ndim The number of dimensions, at most 255.
 * @param dims The size of each dimension.
 * @return 0 on success, nonzero on failure.
 */
int tdump_write(
    struct tdump *dump,
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims
);

#ifdef TSPULSE_DUMP
/**
 * Starts one channel's pass over a window's records.
 *
 * The first `tdump_write_channel` call after this is the first record of the window, the second is the second, and so on, so every channel must write the same records in the same order.
 *
 * @param dump An open writer.
 * @param channel The channel about to be written, counted from 0.
 * @param channels The window's channel count.
 */
void tdump_begin_channel(
    struct tdump *dump, uint32_t channel, uint32_t channels
);

/**
 * Writes one channel's share of a float32 record.
 *
 * Channel 0 appends the record's header and a zero-filled body, so each record appears once in the file, in channel 0's order.
 * Every later channel overwrites its own values in that body.
 *
 * @param dump An open writer.
 * @param name The record name, at most 65535 bytes. It must match what channel 0 wrote at this position.
 * @param data This channel's values, row-major in the record's layout with the channel dimension left out.
 * @param ndim The number of dimensions, including the channel dimension, at most 255.
 * @param dims The size of each dimension, with the channel count at `channel_axis`.
 * @param channel_axis The position of the channel dimension, either 0 or `ndim - 1`.
 * @return 0 on success, nonzero on failure.
 */
int tdump_write_channel(
    struct tdump *dump,
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims,
    int channel_axis
);
#endif

/**
 * Flushes and closes the file.
 *
 * @param dump An open writer.
 * @return 0 on success, nonzero if any write failed.
 */
int tdump_close(struct tdump *dump);

#endif
