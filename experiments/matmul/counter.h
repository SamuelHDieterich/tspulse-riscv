/**
 * @file
 * The instruction counter, read with rdinstret.
 */

#ifndef COUNTER_H
#define COUNTER_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <stdint.h>

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Reads how many instructions have retired so far.
 *
 * Under Spike, the ISA must include zicntr, or the read traps as an illegal instruction.
 *
 * @return The count, or 0 on the host, which has no such counter.
 */
static inline uint64_t read_instret(void) {
#ifdef __riscv
    uint64_t count;
    // The memory clobber keeps the compiler from moving loads and stores across the read
    __asm__ volatile("rdinstret %0" : "=r"(count) : : "memory");
    return count;
#else
    return 0;
#endif
}

#endif
