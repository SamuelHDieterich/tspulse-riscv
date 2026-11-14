"""Profiling a build: where its instructions go, its cycle range and its memory use.

Holds the readers and subtraction of instruction-count profiles (per address, function and mnemonic, for one window), the cycle range per device class from the C906 weight table, memory use (weights, code, static buffers, stack bound and heap), memory traffic, and the tables the `benchmark profile` command writes.
"""
