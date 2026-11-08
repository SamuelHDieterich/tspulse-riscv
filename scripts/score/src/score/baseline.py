"""The `score-baseline` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import multiprocessing
import os
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Annotated

# External
import pandas as pd
import typer

# Internal
from score.collect import collect, compare
from score.kind import Kind
from score.unit import Settings, Unit, result_path, run_unit


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def select(
    table: pd.DataFrame, kind: Kind | None, cheapest: int | None, series: list[str]
) -> list[Unit]:
    """Pick the series to score from `benchmark/series.csv`.

    Parameters
    ----------
    table : pd.DataFrame
        The benchmark series table.
    kind : Kind | None
        The archive whose working series are picked, used when no series are named.
    cheapest : int | None
        Keep only this many working series, those with the fewest passes.
    series : list[str]
        Series named explicitly, which replace the selection by kind.

    Returns
    -------
    list[Unit]
        The series, most passes first, so the longest start first.
    """

    if series:
        missing = sorted(set(series) - set(table["file_name"]))
        if missing:
            raise typer.BadParameter(
                f"Not in the series table: {', '.join(missing)}", param_hint="--series"
            )
        chosen = table[table["file_name"].isin(series)]
    else:
        if kind is None:
            raise typer.BadParameter(
                "Pass --kind, or name series with --series.", param_hint="--kind"
            )
        chosen = table[
            (table["kind"] == kind) & (table["group"] == "working")
        ].sort_values("passes")
        if cheapest is not None:
            chosen = chosen.head(cheapest)

    chosen = chosen.sort_values("passes", ascending=False)
    return [
        Unit(
            **{field: row[field] for field in Unit.__dataclass_fields__}
            | {"kind": Kind(row["kind"])}
        )
        for row in chosen.to_dict("records")
    ]


def _score_baseline(
    kind: Annotated[
        Kind | None,
        typer.Option(help="Score the working series of this kind."),
    ] = None,
    cheapest: Annotated[
        int | None,
        typer.Option(min=1, help="Only the N working series with the fewest passes."),
    ] = None,
    series: Annotated[
        list[str] | None,
        typer.Option(help="A series by CSV name, instead of --kind. Repeatable."),
    ] = None,
    jobs: Annotated[
        int, typer.Option(min=1, help="Series scored at once, each in its own process.")
    ] = 8,
    batch_size: Annotated[
        int, typer.Option(min=1, help="Windows the model handles at once.")
    ] = 128,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help="The results folder. Defaults to results/python-baseline in the data folder."
        ),
    ] = None,
    table: Annotated[Path, typer.Option(help="The benchmark series table.")] = Path(
        "benchmark/series.csv"
    ),
    tables: Annotated[
        Path, typer.Option(help="Where --collect writes the tracked tables.")
    ] = Path("benchmark/baseline"),
    collect_only: Annotated[
        bool,
        typer.Option(
            "--collect",
            help="Skip scoring and build the tables from the saved results.",
        ),
    ] = False,
) -> None:
    """Score series in Python with every head and IBM's metric, resuming where a previous run stopped."""

    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    out = out or data / "results" / "python-baseline"

    if collect_only:
        summary = compare(collect(out, data / "lists" / "ibm", tables))
        typer.echo(summary.to_string(index=False, float_format="{:.5f}".format))
        typer.echo(str(tables))
        return

    units = select(pd.read_csv(table), kind, cheapest, series or [])
    pending = [unit for unit in units if not result_path(out, unit.file_name).exists()]
    for unit in units:
        if unit not in pending:
            typer.echo(f"{unit.file_name}: already scored, skipped")
    if not pending:
        return

    # Parallel workers each get one thread, so they don't fight over cores
    threads = 1 if jobs > 1 else os.cpu_count() or 1
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    settings = Settings(
        data=data, out=out, batch_size=batch_size, threads=threads, git_sha=git_sha
    )

    # Spawn, because forking a parent that has loaded torch can deadlock. One task per child makes each peak memory reading that series' alone.
    context = multiprocessing.get_context("spawn")
    failed = 0
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = {pool.submit(run_unit, unit, settings): unit for unit in pending}
        for future in as_completed(futures):
            try:
                done = future.result()
            # A killed worker, usually by the out-of-memory killer, takes every pending series down with it
            except BrokenProcessPool:
                typer.echo(
                    f"{futures[future].file_name}: its worker was killed, so the pool stopped. Run the same command again to resume.",
                    err=True,
                )
                raise typer.Exit(1) from None
            except Exception as error:
                failed += 1
                typer.echo(f"{futures[future].file_name}: failed:", err=True)
                typer.echo("".join(traceback.format_exception(error)), err=True)
                continue
            typer.echo(
                f"{done.file_name}: {done.seconds:.1f} s, ensemble VUS-PR {done.vus_pr:.5f}, IBM {done.ibm_vus_pr:.5f}"
            )

    if failed:
        typer.echo(f"{failed} of {len(pending)} series failed", err=True)
        raise typer.Exit(1)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `score-baseline`."""

    typer.run(_score_baseline)
