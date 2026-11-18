/**
 * @file
 * Picks the GELU the kernel uses from the build's `GELU` setting.
 */

#ifndef GELU_SELECT_H
#define GELU_SELECT_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#if defined(GELU_FINE)
#include "kernels/gelu/fine.h"
#elif defined(GELU_COARSE)
#include "kernels/gelu/coarse.h"
#elif defined(GELU_TANH)
#include "kernels/gelu/tanh.h"
#elif defined(GELU_TABLE)
#include "kernels/gelu/table.h"
#else
#include "kernels/gelu/libm.h"
#endif

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// Only the chosen header is included, so a build without the table never refers to its symbols
#if defined(GELU_FINE)
/// The GELU of the build's `GELU` value.
#define gelu_one(x) gelu_fine(x)
/// Start-up work the value needs, none here.
#define gelu_setup() ((void)0)
#elif defined(GELU_COARSE)
/// The GELU of the build's `GELU` value.
#define gelu_one(x) gelu_coarse(x)
/// Start-up work the value needs, none here.
#define gelu_setup() ((void)0)
#elif defined(GELU_TANH)
/// The GELU of the build's `GELU` value.
#define gelu_one(x) gelu_tanh(x)
/// Start-up work the value needs, none here.
#define gelu_setup() ((void)0)
#elif defined(GELU_TABLE)
/// The GELU of the build's `GELU` value.
#define gelu_one(x) gelu_table(x)
/// Start-up work the value needs: filling the table.
#define gelu_setup() gelu_table_fill()
#else
/// The GELU of the build's `GELU` value.
#define gelu_one(x) gelu_libm(x)
/// Start-up work the value needs, none here.
#define gelu_setup() ((void)0)
#endif

#endif
