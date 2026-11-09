/**
 * @file
 * A writer for the `.tdump` format shared with the Python reference, float32 records only.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal
#include "io/tdump.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

#define VERSION 1u
#define DTYPE_FLOAT32 1u

#ifdef TSPULSE_DUMP
/// The first table has room for a typical window's records, and it doubles when full
#define INITIAL_SLOTS 1024
/// Zeros are written in blocks of this many floats
#define ZERO_BLOCK 256
#endif

// Both targets are little-endian, like the format, so float data is written as it sits in memory
_Static_assert(
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
    "tdump writes float data in native byte order"
);

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Writes an unsigned integer in little-endian byte order.
 *
 * @param file The stream.
 * @param value The integer.
 * @param size The width in bytes, at most 4.
 * @return 0 on success, nonzero on failure.
 */
static int write_uint(FILE *file, uint32_t value, size_t size) {
    unsigned char bytes[4];
    for (size_t i = 0; i < size; i++) {
        bytes[i] = (unsigned char)(value >> (8 * i));
    }
    return fwrite(bytes, 1, size, file) != size;
}

/**
 * Writes a record's name, dtype and dimensions.
 *
 * @param dump An open writer.
 * @param name The record name, at most 65535 bytes.
 * @param ndim The number of dimensions, at most 255.
 * @param dims The size of each dimension.
 * @return 0 on success, nonzero on failure.
 */
static int write_header(
    struct tdump *dump, const char *name, int ndim, const uint32_t *dims
) {
    size_t name_size = strlen(name);
    if (name_size > UINT16_MAX || ndim < 0 || ndim > UINT8_MAX) {
        return 1;
    }
    if (write_uint(dump->file, (uint32_t)name_size, 2) ||
        fwrite(name, 1, name_size, dump->file) != name_size ||
        write_uint(dump->file, DTYPE_FLOAT32, 1) ||
        write_uint(dump->file, (uint32_t)ndim, 1)) {
        return 1;
    }
    for (int i = 0; i < ndim; i++) {
        if (write_uint(dump->file, dims[i], 4)) {
            return 1;
        }
    }
    return 0;
}

/**
 * Counts the values of a record.
 *
 * @param ndim The number of dimensions.
 * @param dims The size of each dimension.
 * @return The product of the dimensions, 1 for no dimensions.
 */
static size_t element_count(int ndim, const uint32_t *dims) {
    size_t count = 1;
    for (int i = 0; i < ndim; i++) {
        count *= dims[i];
    }
    return count;
}

int tdump_open(struct tdump *dump, const char *path) {
    dump->file = fopen(path, "wb");
    if (dump->file == NULL) {
        return 1;
    }
    if (fwrite("TDMP", 1, 4, dump->file) != 4 ||
        write_uint(dump->file, VERSION, 4)) {
        fclose(dump->file);
        dump->file = NULL;
        return 1;
    }
#ifdef TSPULSE_DUMP
    dump->slots = NULL;
    dump->slot_capacity = 0;
    dump->next_slot = 0;
    dump->channel = 0;
    dump->channels = 0;
#endif
    return 0;
}

int tdump_write(
    struct tdump *dump,
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims
) {
    if (write_header(dump, name, ndim, dims)) {
        return 1;
    }
    size_t count = element_count(ndim, dims);
    return fwrite(data, sizeof(float), count, dump->file) != count;
}

#ifdef TSPULSE_DUMP

/**
 * Finds the slot for the next record, growing the table when it is full.
 *
 * @param dump An open writer.
 * @return The slot, or NULL if the memory is not available.
 */
static struct tdump_slot *next_slot(struct tdump *dump) {
    if (dump->next_slot == dump->slot_capacity) {
        size_t capacity =
            dump->slot_capacity ? 2 * dump->slot_capacity : INITIAL_SLOTS;
        struct tdump_slot *slots =
            realloc(dump->slots, capacity * sizeof(struct tdump_slot));
        if (slots == NULL) {
            return NULL;
        }
        // New slots have no name yet, so channel 0 has nothing to free
        memset(
            slots + dump->slot_capacity, 0,
            (capacity - dump->slot_capacity) * sizeof(struct tdump_slot)
        );
        dump->slots = slots;
        dump->slot_capacity = capacity;
    }
    return &dump->slots[dump->next_slot++];
}

/**
 * Appends zeros to the file.
 *
 * @param file The stream, positioned at its end.
 * @param count How many floats of zero.
 * @return 0 on success, nonzero on failure.
 */
static int write_zeros(FILE *file, size_t count) {
    static const float zeros[ZERO_BLOCK];
    while (count > 0) {
        size_t block = count < ZERO_BLOCK ? count : ZERO_BLOCK;
        if (fwrite(zeros, sizeof(float), block, file) != block) {
            return 1;
        }
        count -= block;
    }
    return 0;
}

/**
 * Appends a record's header and zero-filled body, and remembers where the body starts.
 *
 * @param dump An open writer.
 * @param slot The slot for this record, which receives its name and offset.
 * @param name The record name.
 * @param ndim The number of dimensions.
 * @param dims The size of each dimension.
 * @return 0 on success, nonzero on failure.
 */
static int reserve_record(
    struct tdump *dump,
    struct tdump_slot *slot,
    const char *name,
    int ndim,
    const uint32_t *dims
) {
    if (write_header(dump, name, ndim, dims)) {
        return 1;
    }
    long offset = ftell(dump->file);
    if (offset < 0 || write_zeros(dump->file, element_count(ndim, dims))) {
        return 1;
    }

    // A later window reuses the slot, so the name it held is released
    free(slot->name);
    size_t name_size = strlen(name) + 1;
    slot->name = malloc(name_size);
    if (slot->name == NULL) {
        return 1;
    }
    memcpy(slot->name, name, name_size);
    slot->offset = offset;
    return 0;
}

void tdump_begin_channel(
    struct tdump *dump, uint32_t channel, uint32_t channels
) {
    dump->channel = channel;
    dump->channels = channels;
    dump->next_slot = 0;
}

int tdump_write_channel(
    struct tdump *dump,
    const char *name,
    const float *data,
    int ndim,
    const uint32_t *dims,
    int channel_axis
) {
    // Step 1: the channel dimension must sit first or last, and hold every channel
    if (ndim < 1 || (channel_axis != 0 && channel_axis != ndim - 1) ||
        dims[channel_axis] != dump->channels) {
        return 1;
    }
    size_t per_channel = element_count(ndim, dims) / dump->channels;

    // Step 2: channel 0 reserves the whole record, and every other channel checks it is filling the record it expects
    struct tdump_slot *slot = next_slot(dump);
    if (slot == NULL) {
        return 1;
    }
    if (dump->channel == 0) {
        if (reserve_record(dump, slot, name, ndim, dims)) {
            return 1;
        }
        slot->per_channel = per_channel;
    } else if (
        slot->name == NULL || strcmp(slot->name, name) != 0 ||
        slot->per_channel != per_channel
    ) {
        // Another record, or the same name with other dimensions, would write outside the body channel 0 reserved
        fprintf(
            stderr, "tdump: channel %u writes %s differently from channel 0\n",
            (unsigned)dump->channel, name
        );
        return 1;
    }

    // Step 3: this channel's values. A leading channel dimension makes them one block, a trailing one makes them every `channels`-th value
    if (channel_axis == 0) {
        long start =
            slot->offset + (long)(dump->channel * per_channel * sizeof(float));
        if (fseek(dump->file, start, SEEK_SET) ||
            fwrite(data, sizeof(float), per_channel, dump->file) !=
                per_channel) {
            return 1;
        }
    } else {
        for (size_t i = 0; i < per_channel; i++) {
            long at =
                slot->offset +
                (long)((i * dump->channels + dump->channel) * sizeof(float));
            if (fseek(dump->file, at, SEEK_SET) ||
                fwrite(&data[i], sizeof(float), 1, dump->file) != 1) {
                return 1;
            }
        }
    }

    // Step 4: back to the end, where the next record is appended
    return fseek(dump->file, 0, SEEK_END) != 0;
}

#endif

int tdump_close(struct tdump *dump) {
#ifdef TSPULSE_DUMP
    for (size_t i = 0; i < dump->slot_capacity; i++) {
        free(dump->slots[i].name);
    }
    free(dump->slots);
    dump->slots = NULL;
    dump->slot_capacity = 0;
#endif
    int failed = fflush(dump->file) != 0;
    failed |= fclose(dump->file) != 0;
    dump->file = NULL;
    return failed;
}
