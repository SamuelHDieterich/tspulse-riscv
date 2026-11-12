/**
 * @file
 * Reads the RISC-V instructions-retired counter, to count the instructions a stretch of code runs.
 */

#ifndef INSTRET_H
#define INSTRET_H

#ifdef __riscv

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

#include <stdint.h>

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * Reads `instret`, the count of instructions the core has retired.
 *
 * The counter comes from the Zicntr extension, https://riscv.github.io/riscv-isa-manual/snapshot/unprivileged/#_zicntr_standard_extension_for_base_counters_and_timers
 * Spike only has it when `--isa` lists `zicntr`, such as `--isa=rv64gc_zicntr`.
 *
 * @return The instructions retired since the core was reset, which under pk includes the kernel's boot, so only the difference of two reads is meaningful.
 */
static inline uint64_t instret_read(void) {
    uint64_t count;
    // The memory clobber keeps loads, stores and calls on their side of the read
    __asm__ volatile("csrr %0, instret" : "=r"(count) : : "memory");
    return count;
}

#endif

#endif
