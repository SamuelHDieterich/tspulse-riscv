"""`benchmark all`: run every stage of a label, then collect it, in one process that needs no one in between."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import inspect
import socket
import time
from collections.abc import Callable
from typing import Any

# External
import typer

# Internal
from score.benchmark.collect import collect, collect_label
from score.benchmark.progress import Log, duration
from score.benchmark.run import (
    EXIT_INTERRUPTED,
    EXIT_REFUSED,
    Stage,
    _folder_name,
    run,
)
from score.layout import BENCHMARK_FOLDER


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# `all` takes every option of `run` but `--stage`, which it fixes to all, and adds `collect`'s `--against`, so each option stays defined once
_RUN_PARAMETERS = [
    parameter
    for parameter in inspect.signature(run).parameters.values()
    if parameter.name != "stage"
]
_AGAINST = inspect.signature(collect).parameters["against"]

# A refused call ran nothing new and an interrupted one was asked to stop, so neither collects
_NO_COLLECT = (EXIT_REFUSED, EXIT_INTERRUPTED)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _takes_run_options(function: Callable[..., None]) -> Callable[..., None]:
    """Give a command `run`'s options and `--against`, since typer reads the parameters from the signature, not from the function's own.

    Parameters
    ----------
    function : Callable[..., None]
        The command, which takes its options as keyword arguments.

    Returns
    -------
    Callable[..., None]
        The same function, with the signature set.
    """

    function.__signature__ = inspect.Signature([*_RUN_PARAMETERS, _AGAINST])  # type: ignore[attr-defined]
    return function


@_takes_run_options
def run_all(**options: Any) -> None:
    """Run every stage of a build and collect its tables in one go, so the machine works unattended until the data is in.

    Takes `run`'s options but `--stage`, and `--against`.
    A failed or differing unit, or a scope that can't be scored, does not stop the collection, which writes what is complete and lists the rest; the exit code is then 1.
    A call stopped by Ctrl-C or SIGTERM, or refused before any stage ran, does not collect; both still write the `finished` line, with the exit codes `run` documents.

    Raises
    ------
    typer.Exit
        With code 1 when `run` or `collect` reports a problem, and with `run`'s own code when it was refused or interrupted.
    """

    started = time.monotonic()

    # Step 1: the log `run` appends to, so both halves read as one
    label = _folder_name(options["label"], "--label")
    machine = options["machine"]
    if machine is not None:
        _folder_name(machine, "--machine")
    data = options["data"]
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(EXIT_REFUSED)
    log = Log(
        data / BENCHMARK_FOLDER / label / f"{machine or socket.gethostname()}.log"
    )
    against = options.pop("against")

    # Step 2: every stage, then the tables from whatever is complete. The `finished` line is written however the call ends
    code = 1
    tables = "none written"
    try:
        try:
            run(**options, stage=Stage.ALL)
            code = 0
        except typer.Exit as stop:
            code = stop.exit_code
        if code in _NO_COLLECT:
            log.line(f"run ended with exit code {code}, not collecting")
        else:
            log.line(f"run finished with exit code {code}, collecting")
            try:
                collected = collect_label(label, against, data, log.line)
                tables = f"{collected.accuracy_table}, {collected.instruction_table}"
                # Any problem collect reports fails the call: a scope it could not score, a window whose host and Spike errors differ, or a `result.json` it could not read or that came from other binaries. It never lowers a higher code from `run`
                if collected.problems:
                    code = max(code, 1)
            except typer.Exit:
                code = 1
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    except typer.BadParameter as error:
        code = error.exit_code
        raise
    except Exception:
        # The traceback exits 1, so the `finished` line says so
        code = 1
        raise
    finally:
        verdict = ", interrupted" if code == EXIT_INTERRUPTED else ""
        log.line(
            f"finished in {duration(time.monotonic() - started)}, exit code {code}{verdict}, tables: {tables}"
        )

    # Step 3: the exit code
    if code:
        raise typer.Exit(code)
