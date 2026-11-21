"""Merge the held-back picks' summaries into one file, and print its short form."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# Internal
from score.kind import Kind
from score.variants.building import ResolvedVariant
from score.variants.held_record import (
    HeadScore,
    HeldSummary,
    read_held_summary,
    write_held_summaries,
)
from score.variants.held_summary import held_status_text, not_run_held_summary
from score.variants.listing import VariantError
from score.variants.merge import _text, layout_table, merged_path
from score.variants.summary import ENSEMBLE_HEAD
from score.variants.summary_record import summary_path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The short form's header
SHORT_HEADER = [
    "name",
    "U ensemble VUS-PR",
    "U difference",
    "M ensemble VUS-PR",
    "M difference",
    "U instr. ratio",
    "M instr. ratio",
    "status",
]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def merge_held(
    top: Path, picks: list[ResolvedVariant], suffix: str
) -> list[HeldSummary]:
    """Write the merged file of the held-back picks.

    Parameters
    ----------
    top : Path
        The git top level.
    picks : list[ResolvedVariant]
        The held-back picks, in the list's order.
    suffix : str
        The labels' suffix, the held-back one first, which names the file and each pick's label.

    Returns
    -------
    list[HeldSummary]
        One summary per pick, which has only its name, settings and `not run` when it has no summary. There is no baseline row.

    Raises
    ------
    VariantError
        When a summary can't be read or the merged file can't be written.
    """

    try:
        summaries = []
        for pick in picks:
            found = read_held_summary(summary_path(top, pick.name + suffix))
            summaries.append(found or not_run_held_summary(pick))
        write_held_summaries(merged_path(top, suffix), summaries)
    except (OSError, ValueError, KeyError) as error:
        raise VariantError(f"cannot merge the summaries: {error}") from error
    return summaries


def _ensemble(summary: HeldSummary, kind: Kind) -> HeadScore | None:
    """Find a pick's ensemble score of one kind.

    Parameters
    ----------
    summary : HeldSummary
        The pick's summary.
    kind : Kind
        The archive.

    Returns
    -------
    HeadScore or None
        The score, None for a pick not run.
    """

    return summary.vus_pr.get(kind, {}).get(ENSEMBLE_HEAD)


def short_form_held(summaries: list[HeldSummary]) -> str:
    """Lay out the held-back picks' main fields for the terminal.

    Parameters
    ----------
    summaries : list[HeldSummary]
        The merged file's summaries.

    Returns
    -------
    str
        A header and one line per pick: name, the U and M ensemble VUS-PR with their mean differences from the stack, the instruction ratios to the stack, and what the pick lacks or `done`.
    """

    lines = [SHORT_HEADER]
    for summary in summaries:
        cells = [summary.name]
        for kind in (Kind.U, Kind.M):
            ensemble = _ensemble(summary, kind)
            paired = ensemble.paired if ensemble else None
            cells += [
                _text(ensemble.mean if ensemble else None),
                _text(paired.mean if paired else None),
            ]
        instructions = summary.instructions
        cells += [
            _text(instructions.u_ratio if instructions else None),
            _text(instructions.m_ratio if instructions else None),
            held_status_text(summary),
        ]
        lines.append(cells)
    return layout_table(lines)
