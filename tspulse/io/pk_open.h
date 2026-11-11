/**
 * @file
 * An `_open` for rv64 that passes Linux's flag numbers to the proxy kernel.
 */

#ifndef PK_OPEN_H
#define PK_OPEN_H

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Opens a file the way newlib's `open` and `fopen` ask for, translating newlib's flag numbers to Linux's.
 *
 * @param path The file to open.
 * @param flags The access mode and any of `O_CREAT`, `O_TRUNC`, `O_APPEND` and `O_EXCL`, in newlib's numbers.
 * @param mode The permission bits of a file that `O_CREAT` creates.
 * @return The file descriptor, or -1 with `errno` set: `EINVAL` for any other flag, else the host's error.
 */
int _open(const char *path, int flags, int mode);

#endif
