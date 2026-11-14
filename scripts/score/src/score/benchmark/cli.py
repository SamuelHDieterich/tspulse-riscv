"""The `benchmark` command and its subcommands."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import inspect

# External
import typer

# Internal
from score.benchmark.all import run_all
from score.benchmark.collect import collect
from score.benchmark.run import run
from score.profile.cli import profile


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

# One subcommand per step. Each one's help is its docstring's summary, since typer would otherwise print the numpydoc sections too
app = typer.Typer(no_args_is_help=True, add_completion=False)
app.command("run", help=inspect.cleandoc(run.__doc__ or "").partition("\n")[0])(run)
app.command("collect", help=inspect.cleandoc(collect.__doc__ or "").partition("\n")[0])(
    collect
)
app.command("all", help=inspect.cleandoc(run_all.__doc__ or "").partition("\n")[0])(
    run_all
)
app.command("profile", help=inspect.cleandoc(profile.__doc__ or "").partition("\n")[0])(
    profile
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@app.callback()
def _benchmark() -> None:
    """Run a labelled build over the benchmark's windows on the host and under Spike, and collect its tables."""


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `benchmark`."""

    app()
