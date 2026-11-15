"""Merge the baseline's summary and each variant's into the one variants file, and print its short form."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# Internal
from score.benchmark.tables import TABLES_FOLDER
from score.build_record import BASELINE_NAME
from score.variants.building import ResolvedVariant
from score.variants.listing import VariantError
from score.variants.summary import (
    ENSEMBLE_HEAD,
    baseline_summary,
    not_run_summary,
    status_text,
)
from score.variants.summary_record import (
    Summary,
    read_summary,
    summary_path,
    write_summaries,
    write_summary,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The merged file's name before the suffix and its extension
MERGED_NAME = "variants"
MERGED_EXTENSION = ".json"

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


def _text(value: float | None) -> str:
    """Spell a number for the short form.

    Parameters
    ----------
    value : float or None
        The number.

    Returns
    -------
    str
        The number in full precision, or an empty text for None.
    """

    return "" if value is None else str(value)


def merged_path(top: Path, suffix: str) -> Path:
    """Where the merged file sits.

    Parameters
    ----------
    top : Path
        The git top level.
    suffix : str
        The call's `--suffix`.

    Returns
    -------
    Path
        `variants<suffix>.json` in the tracked results folder.
    """

    return top / TABLES_FOLDER / f"{MERGED_NAME}{suffix}{MERGED_EXTENSION}"


def merge_variants(
    top: Path, variants: list[ResolvedVariant], suffix: str
) -> list[Summary]:
    """Rebuild the baseline's summary and write the merged file.

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
    list[Summary]
        The baseline's summary, then each variant's, which has only its name, settings and `not run` when it has no summary.

    Raises
    ------
    VariantError
        When a summary can't be read or the baseline's can't be computed or written.
    """

    # Step 1: the baseline's summary from its tracked tables, rewritten each time
    try:
        baseline = baseline_summary(top)
        write_summary(summary_path(top, BASELINE_NAME), baseline)
        summaries = [baseline]

        # Step 2: each variant's own summary, in list order
        for variant in variants:
            found = read_summary(summary_path(top, variant.name + suffix))
            summaries.append(found or not_run_summary(variant))

        # Step 3: one list, the baseline first
        write_summaries(merged_path(top, suffix), summaries)
    except (OSError, ValueError, KeyError) as error:
        raise VariantError(f"cannot merge the summaries: {error}") from error
    return summaries


def short_form(summaries: list[Summary]) -> str:
    """Lay out the merged file's main fields for the terminal.

    Parameters
    ----------
    summaries : list[Summary]
        The merged file's summaries.

    Returns
    -------
    str
        A header and one line per summary: name, verdict, instructions ratio, ensemble VUS-PR and its difference, and the cycle range. A label that lacks something shows what it lacks instead of a verdict.
    """

    lines = [SHORT_HEADER]
    for summary in summaries:
        instructions = summary.instructions
        ensemble = summary.vus_pr.get(ENSEMBLE_HEAD)
        cycles = summary.cycles
        lines.append(
            [
                summary.name,
                status_text(summary)
                if summary.missing or summary.verdict is None
                else summary.verdict,
                _text(instructions.ratio if instructions else None),
                _text(ensemble.value if ensemble else None),
                _text(ensemble.difference if ensemble else None),
                f"{cycles.low} to {cycles.high}" if cycles else "",
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
