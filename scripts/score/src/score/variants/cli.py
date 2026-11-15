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

# Internal
from score.build_record import BASELINE_NAME
from score.spike_check import _top_level
from score.variants.building import ResolvedVariant, resolve_names
from score.variants.listing import VariantError, read_listing
from score.variants.merge import merge_variants, merged_path, short_form
from score.variants.stages import RunOptions, VariantOutcome, run_variant
from score.variants.state import SELECTION_FILE, Selection, label_folder, read_selection


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
        Appended to each name to make its label.
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
            help="Keep only the first N windows of each series' screen span, for smoke tests. Needs --suffix.",
        ),
    ] = None,
    series: Annotated[
        list[str] | None,
        typer.Option(
            help="Only this working series, by CSV name without `.csv`. Repeatable. Needs --suffix."
        ),
    ] = None,
    suffix: Annotated[
        str,
        typer.Option(
            help="Appended to each variant's name to make its label, so a cut-down run never writes into a full run's results."
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
        Series stems to keep, or every working series.
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

    # Step 2: the list and each variant's name, which compiles nothing
    top = _top_level()
    try:
        entries = read_listing(list_file or top / VARIANTS_LIST, top)
        resolved = resolve_names(entries, top)
    except VariantError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(EXIT_REFUSED) from error
    names = [variant.name for variant in resolved]
    chosen_names = _check_names(only, names, "--only")
    redo_names = _check_names(redo, names, "--redo")
    selected = [v for v in resolved if not chosen_names or v.name in chosen_names]
    outside = sorted(set(redo_names) - {v.name for v in selected})
    if outside:
        raise typer.BadParameter(
            f"Not among the selected variants: {', '.join(outside)}. A --redo name needs to be run by this call, so --only has to include it.",
            param_hint="--redo",
        )
    for name in names:
        _folder_name(name + suffix, "--suffix")
        # A suffixed name that spells another label would make --redo delete that label's run
        if suffix and name + suffix in {*names, BASELINE_NAME}:
            raise typer.BadParameter(
                f"{name}{suffix} is the name of another variant or of the baseline, so the labels would collide.",
                param_hint="--suffix",
            )

    # Step 3: the series, the way `run` checks them, and the selection every label keeps
    working = read_series(top)
    known = {one.stem for one in working}
    unknown = sorted(set(series or []) - known)
    if unknown:
        raise typer.BadParameter(
            f"Not a working series: {', '.join(unknown)}", param_hint="--series"
        )
    selection = Selection(
        series=[one.stem for one in working if not series or one.stem in series],
        window_cap=window_cap,
    )

    # Step 4: each variant in turn. Ctrl-C or SIGTERM ends the loop, and the rest waits for the next call
    outcomes: list[VariantOutcome] = []
    if not table_only:
        _refuse_other_selection(selected, redo_names, suffix, selection, data)
        options = RunOptions(
            suffix=suffix,
            selection=selection,
            jobs=jobs,
            spike_jobs=spike_jobs,
            machine=machine,
            log_name=f"{machine or socket.gethostname()}{LOG_EXTENSION}",
        )
        for variant in selected:
            # Each stage catches Ctrl-C itself; this catches it between stages, so the command still reaches its end
            try:
                outcome = run_variant(
                    variant, options, variant.name in redo_names, top, data, working
                )
            except KeyboardInterrupt:
                typer.echo(f"Interrupted during variant {variant.name}.", err=True)
                outcome = VariantOutcome(label=variant.name + suffix, interrupted=True)
            outcomes.append(outcome)
            if outcome.interrupted:
                break

    # Step 5: the merged file, rewritten from disk at the end of every call, also after Ctrl-C
    try:
        summaries = merge_variants(top, resolved, suffix)
        typer.echo(f"\nWrote {merged_path(top, suffix)}\n\n{short_form(summaries)}")
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
