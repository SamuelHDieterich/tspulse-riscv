/**
 * @file
 * An `_open` for rv64 that passes Linux's flag numbers to the proxy kernel.
 *
 * newlib numbers the `open` flags its own way (`O_CREAT` is 0x200, `O_TRUNC` 0x400), but the proxy kernel hands them unchanged to the host, which reads them as Linux numbers (0x200 is `O_TRUNC`, 0x400 is `O_APPEND`).
 * Under libgloss's `_open`, which passes the flags untouched, that breaks every file the program writes:
 * - a missing file can't be created, since `O_CREAT` never arrives;
 * - an existing file opens in append mode, so every write lands at the end and `fseek` is ignored;
 * - the dump, which seeks back to fill in sizes, comes out double size.
 *
 * The linker takes this `_open` in place of libgloss's, because libgloss's is only pulled from its archive while `_open` is still undefined.
 * The program is built this way only for rv64.
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <errno.h>
#include <fcntl.h>

// Internal
#include "io/pk_open.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

// Linux's flag numbers, octal as in
// https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/fcntl.h
/// Linux's `O_CREAT`.
#define LINUX_O_CREAT 0100
/// Linux's `O_EXCL`.
#define LINUX_O_EXCL 0200
/// Linux's `O_TRUNC`.
#define LINUX_O_TRUNC 01000
/// Linux's `O_APPEND`.
#define LINUX_O_APPEND 02000

/// The proxy kernel's `open` system call number.
#define SYS_OPEN 1024

/// The flags that are translated, as newlib numbers them.
#define TRANSLATED_FLAGS (O_CREAT | O_TRUNC | O_APPEND | O_EXCL)

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

int _open(const char *path, int flags, int mode) {
    // Step 1: refuse a flag with no translation, so it can't reach the host as some other flag
    if (flags & ~(O_ACCMODE | TRANSLATED_FLAGS)) {
        errno = EINVAL;
        return -1;
    }

    // Step 2: the access mode has the same numbers on both sides, and each other flag gets Linux's number
    long linux_flags = flags & O_ACCMODE;
    if (flags & O_CREAT)
        linux_flags |= LINUX_O_CREAT;
    if (flags & O_TRUNC)
        linux_flags |= LINUX_O_TRUNC;
    if (flags & O_APPEND)
        linux_flags |= LINUX_O_APPEND;
    if (flags & O_EXCL)
        linux_flags |= LINUX_O_EXCL;

    // Step 3: the system call, with the number in a7 and the arguments in a0 to a2, as libgloss does
    register long a0 __asm__("a0") = (long)path;
    register long a1 __asm__("a1") = linux_flags;
    register long a2 __asm__("a2") = mode;
    register long a7 __asm__("a7") = SYS_OPEN;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");

    // Step 4: the kernel returns a negative error number on failure
    if (a0 < 0) {
        errno = (int)-a0;
        return -1;
    }
    return (int)a0;
}
