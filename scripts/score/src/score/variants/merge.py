"""Merge the baseline's summary and each variant's into the one variants table, and print its short form."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
from pathlib import Path

# Internal
from score.benchmark.tables import TABLES_FOLDER
from score.variants.building import BASELINE_NAME, ResolvedVariant
from score.variants.listing import VariantError
from score.variants.summary import (
    COLUMNS,
    STATUS_NOT_RUN,
    baseline_cells,
    read_row,
    settings_text,
    summary_path,
    write_row,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The merged table's name before the suffix and its extension
TABLE_NAME = "variants"
TABLE_EXTENSION = ".csv"

# What the short form shows for an empty cell
EMPTY_CELL = "-"

# The short form's header
SHORT_HEADER = [
    "name",
    "verdict",
    "instr. ratio",
    "ensemble VUS-PR",
    "difference",
    "cycles",
]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def table_path(top: Path, suffix: str) -> Path:
    """Where the merged table sits.

    Parameters
    ----------
    top : Path
        The git top level.
    suffix : str
        The call's `--suffix`.

    Returns
    -------
    Path
        `variants<suffix>.csv` in the tracked results folder.
    """

    return top / TABLES_FOLDER / f"{TABLE_NAME}{suffix}{TABLE_EXTENSION}"


def merge_variants(
    top: Path, variants: list[ResolvedVariant], suffix: str
) -> list[dict[str, str]]:
    """Rebuild the baseline's summary and write the merged table.

    Parameters
    ----------
    top : Path
        The git top level.
    variants : list[ResolvedVariant]
        Every variant of the list, in its order.
    suffix : str
        The call's `--suffix`, which makes each variant's label.

    Returns
    -------
    list[dict[str, str]]
        The table's rows: the baseline, then each variant, which has only its name, settings and `not run` when it has no summary.

    Raises
    ------
    VariantError
        When a summary can't be read or the baseline's can't be computed or written.
    """

    # Step 1: the baseline's row from its tracked tables, rewritten each time
    try:
        baseline = baseline_cells(top)
        write_row(summary_path(top, BASELINE_NAME), baseline)
        rows = [baseline]

        # Step 2: each variant's own summary, in list order
        for variant in variants:
            found = read_row(summary_path(top, variant.name + suffix))
            rows.append(
                found
                or {
                    "name": variant.name,
                    "settings": settings_text(variant),
                    "status": STATUS_NOT_RUN,
                }
            )
    except (OSError, ValueError, KeyError) as error:
        raise VariantError(f"cannot merge the summaries: {error}") from error

    # Step 3: one header, every row
    table = table_path(top, suffix)
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as file:
        writer = csv.DictWriter(file, COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})
    return [{column: row.get(column, "") for column in COLUMNS} for row in rows]


def short_form(rows: list[dict[str, str]]) -> str:
    """Lay out the merged table's main columns for the terminal.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The merged table's rows.

    Returns
    -------
    str
        A header and one line per row: name, verdict, instructions ratio, ensemble VUS-PR and its difference, and the cycle range. A variant that wasn't run shows its status instead of a verdict.
    """

    lines = [SHORT_HEADER]
    for row in rows:
        cycles = (
            f"{row['cycles_low']} to {row['cycles_high']}" if row["cycles_low"] else ""
        )
        lines.append(
            [
                row["name"],
                row["verdict"] or row["status"],
                row["instructions_ratio"],
                row["vus_pr_ensemble"],
                row["vus_pr_ensemble_diff"],
                cycles,
            ]
        )
    shown = [[cell or EMPTY_CELL for cell in line] for line in lines]
    widths = [
        max(len(line[column]) for line in shown) for column in range(len(SHORT_HEADER))
    ]
    return "\n".join(
        "  ".join(
            cell.ljust(width) for cell, width in zip(line, widths, strict=True)
        ).rstrip()
        for line in shown
    )
