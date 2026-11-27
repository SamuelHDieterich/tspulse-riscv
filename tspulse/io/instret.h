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
 * Spike only has it when `--isa` lists `zicntr`, such as `--isa=rv64gc_zicntr` or `--isa=rv32gc_zicntr`.
 * On rv32 the counter is two 32-bit halves, `instret` and `instreth`, read here as one value.
 *
 * @return The instructions retired since the core was reset, which under pk includes the kernel's boot, so only the difference of two reads is meaningful.
 */
static inline uint64_t instret_read(void) {
#if __riscv_xlen == 64
    uint64_t count;
    // The memory clobber keeps loads, stores and calls on their side of the read
    __asm__ volatile("csrr %0, instret" : "=r"(count) : : "memory");
    return count;
#else
    // The two halves cannot be read at once, so a carry from the low half into the high one between the reads would put the count off by 2^32
    // Reading the high half before and after, and retrying when they differ, rules that out
    uint32_t high_before;
    uint32_t low;
    uint32_t high_after;
    do {
        __asm__ volatile("csrr %0, instreth" : "=r"(high_before) : : "memory");
        __asm__ volatile("csrr %0, instret" : "=r"(low) : : "memory");
        __asm__ volatile("csrr %0, instreth" : "=r"(high_after) : : "memory");
    } while (high_before != high_after);
    return ((uint64_t)high_after << 32) | low;
#endif
}

#endif

#endif
