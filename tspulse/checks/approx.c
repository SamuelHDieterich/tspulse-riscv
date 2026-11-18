/**
 * @file
 * Measures each exp and GELU approximation against double precision and writes one CSV row per value and range.
 *
 * Every row sweeps 2^20 evenly spaced points over its range, plus the table's nodes and node midpoints and a few edge inputs, and records the largest absolute, relative and ulp errors.
 * The true value is `exp` or `erf` in `double`, taken at the same `float` input the approximation gets.
 *
 * Usage: check-approx OUTPUT
 */

// ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
// ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

// Built-in
#include <math.h>
#include <stdio.h>

// Internal
#include "kernels/exp/bits.h"
#include "kernels/exp/coarse.h"
#include "kernels/exp/fine.h"
#include "kernels/exp/libm.h"
#include "kernels/exp/table.h"
#include "kernels/gelu/coarse.h"
#include "kernels/gelu/fine.h"
#include "kernels/gelu/libm.h"
#include "kernels/gelu/table.h"
#include "kernels/gelu/tanh.h"

// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
// ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

/// Evenly spaced points per range.
#define SWEEP_POINTS (1 << 20)

/// Below this true value a relative error is not counted.
#define REL_FLOOR 1e-30

/// The CSV's header line.
#define CSV_HEADER "function,value,low,high,points,max_abs,max_rel,max_ulp\n"

/// Softmax's domain: its inputs are the row's values minus its maximum.
#define EXP_SOFTMAX_LOW -87.0f

/// Top of the range `GELU=tanh` feeds to the fine exp.
#define EXP_POSITIVE_HIGH 88.0f

/// The GELU sweep covers the activations the model sees, and a little more.
#define GELU_RANGE 10.0f

/// Number of elements in an array.
#define ARRAY_LEN(array) ((int)(sizeof(array) / sizeof((array)[0])))

/// Inputs just below 0, which a table of e^x on [-16, 0] rounds onto its last node and an even sweep never lands on.
static const float EXP_EDGES[] = {
    -0x1p-149f, ///< nextafterf(0, -1), the float just below 0.
    -1e-9f,     ///< Within the 4.8e-7 below 0 where x + 16 rounds up to 16.
};

/// The exp table's own edges: its first node and the float above it, then the inputs just below 0.
static const float EXP_TABLE_EDGES[] = {
    -16.0f,         ///< The first node.
    -0x1.fffffep3f, ///< nextafterf(-16, 0).
    -0x1p-149f,     ///< nextafterf(0, -1), the float just below 0.
    -1e-9f,         ///< Within the 4.8e-7 below 0 where x + 16 rounds up to 16.
};

/// The GELU table's ends, +-8, and the floats just inside them.
static const float GELU_EDGES[] = {
    0x1.fffffep2f,  ///< nextafterf(8, 0).
    8.0f,           ///< The top end.
    -0x1.fffffep2f, ///< nextafterf(-8, 0).
    -8.0f,          ///< The bottom end.
};

// ░▀█▀░█░█░█▀█░█▀▀░█▀▀
// ░░█░░░█░░█▀▀░█▀▀░▀▀█
// ░░▀░░░▀░░▀░░░▀▀▀░▀▀▀

/// The function being measured.
typedef float (*approx_fn)(float);

/// The double-precision function it is measured against.
typedef double (*truth_fn)(double);

/// One CSV row: a function over a range.
struct row {
    const char *function; ///< Which function: `exp` or `gelu`.
    const char *value;    ///< The setting's value: `libm`, `fine`, ...
    approx_fn approx;     ///< The approximation.
    truth_fn truth;       ///< The reference in `double`.
    float low;            ///< Start of the range.
    float high;           ///< End of the range.
    int high_included;    ///< Whether `high` itself is swept.
    float table_lowest;   ///< Input of the table's first node.
    float table_per_unit; ///< Nodes per unit of x; 0 means no table.
    int table_nodes;      ///< Number of nodes.
    const float *edges;   ///< Extra inputs to measure, those inside the range.
    int edge_count;       ///< Number of edges.
};

/// Running errors of one row.
struct errors {
    long points;    ///< Points measured.
    double max_abs; ///< Largest |approx - true|.
    double max_rel; ///< Largest relative error where |true| >= REL_FLOOR.
    double max_ulp; ///< Largest error in ulps of the true value in float.
};

// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

/**
 * GELU in double: 0.5 * x * (1 + erf(x / sqrt(2))).
 * https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
 *
 * @param x The input.
 * @return GELU(x).
 */
static double gelu_truth(double x) {
    return 0.5 * x * (1.0 + erf(x * M_SQRT1_2));
}

/**
 * Compares the approximation with the reference at one input and updates the running errors.
 *
 * @param row The function being measured.
 * @param x The input.
 * @param errors The running errors to update.
 */
static void measure(const struct row *row, float x, struct errors *errors) {
    double truth = row->truth((double)x);
    double diff = fabs((double)row->approx(x) - truth);

    // Step 1: the absolute error
    errors->points++;
    if (diff > errors->max_abs) {
        errors->max_abs = diff;
    }

    // Step 2: the relative error, where the true value is big enough to mean something
    if (fabs(truth) >= REL_FLOOR && diff / fabs(truth) > errors->max_rel) {
        errors->max_rel = diff / fabs(truth);
    }

    // Step 3: the error in ulps, where an ulp is the gap from the true value in float to the next float away from zero
    float truth_float = fabsf((float)truth);
    double ulp =
        (double)nextafterf(truth_float, INFINITY) - (double)truth_float;
    if (diff / ulp > errors->max_ulp) {
        errors->max_ulp = diff / ulp;
    }
}

/**
 * Sweeps one row: the evenly spaced points, every node and node midpoint of a table, then the row's edge inputs.
 *
 * @param row The function and range.
 * @param errors Filled with the row's errors.
 */
static void sweep(const struct row *row, struct errors *errors) {
    *errors = (struct errors){0};

    // Step 1: evenly spaced points; a range that leaves its end out divides by the point count instead of one less
    double span = (double)row->high - (double)row->low;
    double divisor = row->high_included ? SWEEP_POINTS - 1 : SWEEP_POINTS;
    for (long i = 0; i < SWEEP_POINTS; i++) {
        measure(
            row, (float)((double)row->low + span * (double)i / divisor), errors
        );
    }

    // Step 2: the nodes and the midpoints between them, which are exact in float
    for (int i = 0; i < row->table_nodes; i++) {
        float node = row->table_lowest + (float)i / row->table_per_unit;
        measure(row, node, errors);
        if (i + 1 < row->table_nodes) {
            measure(row, node + 0.5f / row->table_per_unit, errors);
        }
    }

    // Step 3: the edges inside the range; these are the inputs that round onto a table's end node, which an even sweep never lands on
    for (int i = 0; i < row->edge_count; i++) {
        float x = row->edges[i];
        int below_high =
            x < row->high || (row->high_included && x == row->high);
        if (x >= row->low && below_high) {
            measure(row, x, errors);
        }
    }
}

// ░█▄█░█▀█░▀█▀░█▀█
// ░█░█░█▀█░░█░░█░█
// ░▀░▀░▀░▀░▀▀▀░▀░▀

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 1;
    }

    FILE *csv = fopen(argv[1], "w");
    if (csv == NULL) {
        fprintf(stderr, "cannot write %s\n", argv[1]);
        return 1;
    }

    exp_table_fill();
    gelu_table_fill();

    // GELU rows are judged on `max_abs`: below about -5.9, `1 + erff(x / sqrt(2))` cancels to 0 in float, so every float GELU returns 0 where the true value is tiny but above the floor, and that tail sets their `max_rel` and `max_ulp`
    // The exp table's zero tail is its own row for the same reason: the table returns 0 there by design, a different kind of error from the interpolation's
    // The tail row leaves its `high` out, since the row above it already sweeps that point
    const struct row rows[] = {
        {
            .function = "exp",
            .value = "libm",
            .approx = exp_libm,
            .truth = exp,
            .low = EXP_SOFTMAX_LOW,
            .high = 0.0f,
            .high_included = 1,
            .edges = EXP_EDGES,
            .edge_count = ARRAY_LEN(EXP_EDGES),
        },
        {
            .function = "exp",
            .value = "fine",
            .approx = exp_fine,
            .truth = exp,
            .low = EXP_SOFTMAX_LOW,
            .high = 0.0f,
            .high_included = 1,
            .edges = EXP_EDGES,
            .edge_count = ARRAY_LEN(EXP_EDGES),
        },
        {
            .function = "exp",
            .value = "fine",
            .approx = exp_fine,
            .truth = exp,
            .low = 0.0f,
            .high = EXP_POSITIVE_HIGH,
            .high_included = 1,
            .edges = EXP_EDGES,
            .edge_count = ARRAY_LEN(EXP_EDGES),
        },
        {
            .function = "exp",
            .value = "coarse",
            .approx = exp_coarse,
            .truth = exp,
            .low = EXP_SOFTMAX_LOW,
            .high = 0.0f,
            .high_included = 1,
            .edges = EXP_EDGES,
            .edge_count = ARRAY_LEN(EXP_EDGES),
        },
        {
            .function = "exp",
            .value = "bits",
            .approx = exp_bits,
            .truth = exp,
            .low = EXP_SOFTMAX_LOW,
            .high = 0.0f,
            .high_included = 1,
            .edges = EXP_EDGES,
            .edge_count = ARRAY_LEN(EXP_EDGES),
        },
        {
            .function = "exp",
            .value = "table",
            .approx = exp_table,
            .truth = exp,
            .low = EXP_TABLE_LOWEST,
            .high = 0.0f,
            .high_included = 1,
            .table_lowest = EXP_TABLE_LOWEST,
            .table_per_unit = EXP_TABLE_PER_UNIT,
            .table_nodes = EXP_TABLE_NODES,
            .edges = EXP_TABLE_EDGES,
            .edge_count = ARRAY_LEN(EXP_TABLE_EDGES),
        },
        {
            .function = "exp",
            .value = "table",
            .approx = exp_table,
            .truth = exp,
            .low = EXP_SOFTMAX_LOW,
            .high = EXP_TABLE_LOWEST,
            .high_included = 0,
        },
        {
            .function = "gelu",
            .value = "libm",
            .approx = gelu_libm,
            .truth = gelu_truth,
            .low = -GELU_RANGE,
            .high = GELU_RANGE,
            .high_included = 1,
            .edges = GELU_EDGES,
            .edge_count = ARRAY_LEN(GELU_EDGES),
        },
        {
            .function = "gelu",
            .value = "fine",
            .approx = gelu_fine,
            .truth = gelu_truth,
            .low = -GELU_RANGE,
            .high = GELU_RANGE,
            .high_included = 1,
            .edges = GELU_EDGES,
            .edge_count = ARRAY_LEN(GELU_EDGES),
        },
        {
            .function = "gelu",
            .value = "coarse",
            .approx = gelu_coarse,
            .truth = gelu_truth,
            .low = -GELU_RANGE,
            .high = GELU_RANGE,
            .high_included = 1,
            .edges = GELU_EDGES,
            .edge_count = ARRAY_LEN(GELU_EDGES),
        },
        {
            .function = "gelu",
            .value = "tanh",
            .approx = gelu_tanh,
            .truth = gelu_truth,
            .low = -GELU_RANGE,
            .high = GELU_RANGE,
            .high_included = 1,
            .edges = GELU_EDGES,
            .edge_count = ARRAY_LEN(GELU_EDGES),
        },
        {
            .function = "gelu",
            .value = "table",
            .approx = gelu_table,
            .truth = gelu_truth,
            .low = -GELU_RANGE,
            .high = GELU_RANGE,
            .high_included = 1,
            .table_lowest = GELU_TABLE_LOWEST,
            .table_per_unit = GELU_TABLE_PER_UNIT,
            .table_nodes = GELU_TABLE_NODES,
            .edges = GELU_EDGES,
            .edge_count = ARRAY_LEN(GELU_EDGES),
        },
    };
    const int row_count = (int)(sizeof rows / sizeof rows[0]);

    fputs(CSV_HEADER, csv);
    for (int i = 0; i < row_count; i++) {
        struct errors errors;
        sweep(&rows[i], &errors);
        fprintf(
            csv, "%s,%s,%g,%g,%ld,%.6g,%.6g,%.6g\n", rows[i].function,
            rows[i].value, (double)rows[i].low, (double)rows[i].high,
            errors.points, errors.max_abs, errors.max_rel, errors.max_ulp
        );
        printf(
            "%s_%s [%g, %g]: abs %.3g rel %.3g ulp %.3g\n", rows[i].function,
            rows[i].value, (double)rows[i].low, (double)rows[i].high,
            errors.max_abs, errors.max_rel, errors.max_ulp
        );
    }

    if (fclose(csv)) {
        fprintf(stderr, "cannot finish %s\n", argv[1]);
        return 1;
    }
    return 0;
}
