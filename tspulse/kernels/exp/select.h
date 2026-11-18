/**
 * @file
 * Picks the exponential softmax uses from the build's `EXP` setting.
 */

#ifndef EXP_SELECT_H
#define EXP_SELECT_H

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Internal
#if defined(EXP_FINE)
#include "kernels/exp/fine.h"
#elif defined(EXP_COARSE)
#include "kernels/exp/coarse.h"
#elif defined(EXP_BITS)
#include "kernels/exp/bits.h"
#elif defined(EXP_TABLE)
#include "kernels/exp/table.h"
#else
#include "kernels/exp/libm.h"
#endif

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// Only the chosen header is included, so a build without the table never refers to its symbols
#if defined(EXP_FINE)
/// The exponential of the build's `EXP` value.
#define softmax_exp(x) exp_fine(x)
/// Start-up work the value needs, none here.
#define exp_setup() ((void)0)
#elif defined(EXP_COARSE)
/// The exponential of the build's `EXP` value.
#define softmax_exp(x) exp_coarse(x)
/// Start-up work the value needs, none here.
#define exp_setup() ((void)0)
#elif defined(EXP_BITS)
/// The exponential of the build's `EXP` value.
#define softmax_exp(x) exp_bits(x)
/// Start-up work the value needs, none here.
#define exp_setup() ((void)0)
#elif defined(EXP_TABLE)
/// The exponential of the build's `EXP` value.
#define softmax_exp(x) exp_table(x)
/// Start-up work the value needs: filling the table.
#define exp_setup() exp_table_fill()
#else
/// The exponential of the build's `EXP` value.
#define softmax_exp(x) exp_libm(x)
/// Start-up work the value needs, none here.
#define exp_setup() ((void)0)
#endif

#endif
