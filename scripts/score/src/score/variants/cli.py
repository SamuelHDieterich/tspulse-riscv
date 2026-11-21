"""`benchmark variants`: build every variant of the tracked list and run its stages, each under its own label."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import signal
import socket
from pathlib import Path
from typing import Annotated

# External
import typer

from score.benchmark.run import EXIT_INTERRUPTED, EXIT_REFUSED, _folder_name
from score.benchmark.scope import read_series
from score.benchmark.units import parse_shard

# Internal
from score.build_record import BASELINE_NAME
from score.spike_check import HELD_GROUP, WORKING_GROUP, Group, _top_level
from score.variants.building import ResolvedVariant, resolve_names
from score.variants.held_merge import merge_held, short_form_held
from score.variants.listing import VariantError, find_stack, read_listing
from score.variants.merge import merge_variants, merged_path, short_form
from score.variants.stages import RunOptions, VariantOutcome, run_variant
from score.variants.state import (
    HELD_SUFFIX,
    SELECTION_FILE,
    Selection,
    label_folder,
    read_selection,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The variant list when none is named, under the git top level
VARIANTS_LIST = Path("benchmark/variants.toml")

# The log's file extension; its name is the machine's, as `run` names it
LOG_EXTENSION = ".log"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _check_names(names: list[str] | None, known: list[str], option: str) -> list[str]:
    """Check that every variant named on the command line is in the list.

    Parameters
    ----------
    names : list[str] or None
        The names given.
    known : list[str]
        The list's variant names.
    option : str
        The option, such as `--only`, which names it in the error.

    Returns
    -------
    list[str]
        The names, empty when none was given.

    Raises
    ------
    typer.BadParameter
        When a name is not in the list.
    """

    unknown = sorted(set(names or []) - set(known))
    if unknown:
        raise typer.BadParameter(
            f"Not in the variant list: {', '.join(unknown)}. The variants are {', '.join(known)}.",
            param_hint=option,
        )
    return names or []


def _refuse_other_selection(
    variants: list[ResolvedVariant],
    redo: list[str],
    suffix: str,
    selection: Selection,
    data: Path,
) -> None:
    """Refuse the call when a label was started with other series or another window cap.

    Parameters
    ----------
    variants : list[ResolvedVariant]
        The variants about to run.
    redo : list[str]
        The variants being started again, whose selection is deleted first.
    suffix : str
        Appended to each name to make its label, the held-back suffix included.
    selection : Selection
        This call's series and window cap.
    data : Path
        The folder with the data.

    Raises
    ------
    typer.Exit
        With `EXIT_REFUSED` when a label's `selection.json` can't be read or differs from this call's.
    """

    for variant in variants:
        if variant.name in redo:
            continue
        label = variant.name + suffix
        try:
            stored = read_selection(label_folder(data, label))
        except (OSError, ValueError) as error:
            typer.echo(
                f"Cannot read label {label}'s {SELECTION_FILE}: {error}", err=True
            )
            raise typer.Exit(EXIT_REFUSED) from error
        if stored is not None and stored != selection:
            typer.echo(
                f"Label {label} was started with window cap {stored.window_cap} on {len(stored.series)} series, "
                f"and this call asks for window cap {selection.window_cap} on {len(selection.series)} series.\n"
                f"Use another --suffix, or --redo {variant.name} to delete its results and start it again.",
                err=True,
            )
            raise typer.Exit(EXIT_REFUSED)


def variants(
    list_file: Annotated[
        Path | None,
        typer.Option(
            "--list",
            help="The variant list. Defaults to benchmark/variants.toml in this worktree.",
        ),
    ] = None,
    only: Annotated[
        list[str] | None,
        typer.Option(help="Only this variant, by build name. Repeatable."),
    ] = None,
    redo: Annotated[
        list[str] | None,
        typer.Option(
            help="Delete this variant's results and start it again, by build name. Repeatable."
        ),
    ] = None,
    table_only: Annotated[
        bool,
        typer.Option(
            "--table-only",
            help="Only write the merged file from what is on disk, building and running nothing.",
        ),
    ] = False,
    jobs: Annotated[
        int, typer.Option(min=1, help="Accuracy units run at once on the host.")
    ] = 1,
    spike_jobs: Annotated[
        int,
        typer.Option(
            min=1,
            help="Runs at once under Spike, for the counted windows and the profile.",
        ),
    ] = 1,
    window_cap: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Keep only the first N windows of each series' screen span, or of each held-back series with --group held, for smoke tests. Needs --suffix.",
        ),
    ] = None,
    series: Annotated[
        list[str] | None,
        typer.Option(
            help="Only this series of the group, by CSV name without `.csv`. Repeatable. Needs --suffix."
        ),
    ] = None,
    group: Annotated[
        Group,
        typer.Option(
            help="The series to run: `working`, every variant of the list against the baseline, or `held`, the picks marked `held = true` on the held-back series, under labels ending in `-held` plus --suffix."
        ),
    ] = WORKING_GROUP,
    shard: Annotated[
        str | None,
        typer.Option(
            help="Run only share K of N of each label's units, as `K/N`, and stop after `run`: the shards are merged and collected on one machine afterwards."
        ),
    ] = None,
    suffix: Annotated[
        str,
        typer.Option(
            help="Appended to each variant's name (after `-held` for the held group) to make its label, so a cut-down run never writes into a full run's results."
        ),
    ] = "",
    machine: Annotated[
        str | None,
        typer.Option(
            help="A label for this computer, recorded with each result and naming its log. Defaults to the hostname."
        ),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
) -> None:
    """Build every variant of the list and run its reference check, `run`, `collect` against the baseline and `profile`, measure its exactness against the baseline and write its summary, skipping what is already on disk.

    Each variant's results go under its own label, its build name plus `--suffix`, in the data folder and in benchmark/results/.
    A stage that fails is logged as a problem, the variant's later stages are skipped, and the next variant starts.
    At the end every call merges the summaries into benchmark/results/variants<suffix>.json and prints its short form.
    With --group held, only the marked picks run, on the held-back series, with the same stages except the comparison with the baseline and the profile; their labels, summaries and merged file carry `-held` before the suffix.

    Parameters
    ----------
    list_file : Path or None
        The variant list, or the tracked one.
    only : list[str] or None
        The variants to run, or all of them.
    redo : list[str] or None
        The variants to start again.
    table_only : bool
        Whether to build and run nothing.
    jobs : int
        Host accuracy units at once.
    spike_jobs : int
        Spike runs at once.
    window_cap : int or None
        The most windows kept of each screen span.
    series : list[str] or None
        Series stems to keep, or every series of the group.
    group : Group
        The series group, `working` or `held`.
    shard : str or None
        `K/N`, or None for every unit.
    suffix : str
        Appended to each name to make its label.
    machine : str or None
        The label of this computer.
    data : Path or None
        The folder with the data.

    Raises
    ------
    typer.Exit
        With code 1 when any variant had a problem, with `EXIT_REFUSED` when the list, a label's selection or the data folder can't be used, before any build, and with `EXIT_INTERRUPTED` after Ctrl-C or SIGTERM.
    """

    # SIGTERM would otherwise end the process without unwinding, leaving a build or a stage's Spike runs going; raised as Ctrl-C, each stage stops them first
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    # Step 1: the options that can't go together
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(EXIT_REFUSED)
    if (window_cap is not None or series) and not suffix:
        raise typer.BadParameter(
            "A cut-down run with --window-cap or --series needs a --suffix, so its labels differ from the full run's.",
            param_hint="--suffix",
        )
    if machine is not None:
        _folder_name(machine, "--machine")
    parse_shard(shard)
    if shard is not None and (redo or table_only):
        raise typer.BadParameter(
            "A shard holds only part of a label's units, so it can't start the label again or only write the merged file.",
            param_hint="--shard",
        )
    label_suffix = (HELD_SUFFIX if group == HELD_GROUP else "") + suffix

    # Step 2: the list and each variant's name, which compiles nothing
    top = _top_level()
    try:
        entries = read_listing(list_file or top / VARIANTS_LIST, top)
        resolved = resolve_names(entries, top)
    except VariantError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(EXIT_REFUSED) from error
    in_group = [v for v in resolved if group == WORKING_GROUP or v.entry.held]
    names = [variant.name for variant in resolved]
    group_names = [variant.name for variant in in_group]
    for option, given in (("--only", only), ("--redo", redo)):
        unmarked = sorted(set(given or []) & set(names) - set(group_names))
        if unmarked:
            raise typer.BadParameter(
                f"Not marked `held = true` in the variant list: {', '.join(unmarked)}. The held variants are {', '.join(group_names)}.",
                param_hint=option,
            )
    chosen_names = _check_names(only, group_names, "--only")
    redo_names = _check_names(redo, group_names, "--redo")
    selected = [v for v in in_group if not chosen_names or v.name in chosen_names]
    stack_label = None
    if group == HELD_GROUP:
        try:
            stack_entry = find_stack([v.entry for v in resolved])
        except VariantError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(EXIT_REFUSED) from error
        stack_label = (
            next(v.name for v in in_group if v.entry is stack_entry) + label_suffix
        )
        # A pick's summary pairs with the stack's tables, so the stack runs first. The sort is stable, and the merged file keeps the list's order
        selected.sort(key=lambda v: v.entry is not stack_entry)
    outside = sorted(set(redo_names) - {v.name for v in selected})
    if outside:
        raise typer.BadParameter(
            f"Not among the selected variants: {', '.join(outside)}. A --redo name needs to be run by this call, so --only has to include it.",
            param_hint="--redo",
        )
    for name in names:
        _folder_name(name + label_suffix, "--suffix")
        # A suffixed name that spells another label would make --redo delete that label's run
        if label_suffix and name + label_suffix in {*names, BASELINE_NAME}:
            raise typer.BadParameter(
                f"{name}{label_suffix} is the name of another variant or of the baseline, so the labels would collide.",
                param_hint="--suffix",
            )

    # Step 3: the series, the way `run` checks them, and the selection every label keeps
    group_series = read_series(top, group)
    known = {one.stem for one in group_series}
    unknown = sorted(set(series or []) - known)
    if unknown:
        raise typer.BadParameter(
            f"Not a {group} series: {', '.join(unknown)}", param_hint="--series"
        )
    selection = Selection(
        series=[one.stem for one in group_series if not series or one.stem in series],
        window_cap=window_cap,
    )

    # Step 4: each variant in turn. Ctrl-C or SIGTERM ends the loop, and the rest waits for the next call
    outcomes: list[VariantOutcome] = []
    if not table_only:
        _refuse_other_selection(selected, redo_names, label_suffix, selection, data)
        options = RunOptions(
            suffix=label_suffix,
            selection=selection,
            jobs=jobs,
            spike_jobs=spike_jobs,
            machine=machine,
            log_name=f"{machine or socket.gethostname()}{LOG_EXTENSION}",
            group=group,
            shard=shard,
            stack_label=stack_label,
        )
        for variant in selected:
            # Each stage catches Ctrl-C itself; this catches it between stages, so the command still reaches its end
            try:
                outcome = run_variant(
                    variant,
                    options,
                    variant.name in redo_names,
                    top,
                    data,
                    group_series,
                )
            except KeyboardInterrupt:
                typer.echo(f"Interrupted during variant {variant.name}.", err=True)
                outcome = VariantOutcome(
                    label=variant.name + label_suffix, interrupted=True
                )
            outcomes.append(outcome)
            if outcome.interrupted:
                break

    # Step 5: the merged file, rewritten from disk at the end of every call, also after Ctrl-C
    try:
        if group == HELD_GROUP:
            shown = short_form_held(merge_held(top, in_group, label_suffix))
        else:
            shown = short_form(merge_variants(top, resolved, label_suffix))
        typer.echo(f"\nWrote {merged_path(top, label_suffix)}\n\n{shown}")
    except VariantError as error:
        typer.echo(str(error), err=True)
        # The interruption is the call's result, whatever the merge says
        if not any(outcome.interrupted for outcome in outcomes):
            raise typer.Exit(1) from error

    # Step 6: the exit code, the interruption first since it says the run is unfinished
    if any(outcome.interrupted for outcome in outcomes):
        raise typer.Exit(EXIT_INTERRUPTED)
    if any(outcome.problems for outcome in outcomes):
        raise typer.Exit(1)
