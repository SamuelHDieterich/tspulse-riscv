"""The `compare-reference` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path
from typing import Annotated

# External
import numpy as np
import typer

# Internal
from score import noise, tdump


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _max_diff(ref: np.ndarray, dump: np.ndarray) -> float:
    """Compute the largest absolute difference between two arrays of the same shape.

    Parameters
    ----------
    ref : np.ndarray
        The reference values.
    dump : np.ndarray
        The values to check.

    Returns
    -------
    float
        The largest `|dump − ref|`, computed in float64 and ignoring positions where either side is NaN. 0 when nothing is left.
    """

    diff = np.abs(dump.astype(np.float64) - ref.astype(np.float64))
    finite = diff[~np.isnan(diff)]
    return float(finite.max()) if finite.size else 0.0


def _exact(ref: np.ndarray, dump: np.ndarray) -> bool:
    """Check that two arrays hold the same bits.

    Parameters
    ----------
    ref : np.ndarray
        The reference values.
    dump : np.ndarray
        The values to check, of the same shape.

    Returns
    -------
    bool
        True when the dtypes match and every value has the same bit pattern, so −0.0 and +0.0 differ.

    Notes
    -----
    NaNs must sit at the same positions, but their bits aren't compared: the payload may differ between the host and rv64.
    """

    if ref.dtype != dump.dtype:
        return False
    nan = np.isnan(ref)
    if not np.array_equal(nan, np.isnan(dump)):
        return False
    bits = np.uint32 if ref.dtype == np.float32 else np.uint64
    return bool(np.array_equal(ref.view(bits)[~nan], dump.view(bits)[~nan]))


def _compare_reference(
    ref: Annotated[Path, typer.Argument(help="The reference `.tdump`.")],
    dump: Annotated[Path, typer.Argument(help="The `.tdump` to check against it.")],
    exact: Annotated[
        bool, typer.Option(help="Require identical bits, for C against C.")
    ] = False,
    only: Annotated[
        str | None,
        typer.Option(help="Only compare records whose name starts with this."),
    ] = None,
    noise_path: Annotated[
        Path | None,
        typer.Option(
            "--noise",
            help="The pooled noise file. Defaults to noise.csv two folders above REF.",
        ),
    ] = None,
) -> None:
    """Compare a dump against the reference, record by record, and mark where they first differ.

    Parameters
    ----------
    ref : Path
        The reference `.tdump`.
    dump : Path
        The `.tdump` to check against it.
    exact : bool
        Require identical bits instead of the noise rule.
    only : str or None
        Only compare records whose name starts with this.
    noise_path : Path or None
        The pooled noise file. Defaults to `noise.csv` two folders above `ref`.

    Raises
    ------
    typer.Exit
        With code 1 when a record fails or none was compared, and 2 when a file can't be read or a step has no noise entry. Returning normally exits 0.
    """

    # Step 1: read both dumps and, unless exact, the pooled noise. The reference sits at reference/<series>/w<index>.tdump, so its noise.csv is two folders up
    noise_path = noise_path or ref.resolve().parent.parent / "noise.csv"
    try:
        reference = tdump.read(ref)
        candidate = tdump.read(dump)
        ratios = {} if exact else noise.read_pooled(noise_path)
    except (OSError, ValueError) as error:
        typer.echo(error, err=True)
        raise typer.Exit(2) from error

    # Step 2: the records to compare, and a check that each has a noise entry before any is compared
    names = [name for name in reference if only is None or name.startswith(only)]
    if not exact:
        unknown = sorted({noise.step(name) for name in names} - ratios.keys())
        if unknown:
            typer.echo(f"Steps missing from {noise_path}:", err=True)
            for name in unknown:
                typer.echo(f"  {name}", err=True)
            raise typer.Exit(2)

    # Step 3: the table header, wide enough for the longest name
    width = max((len(name) for name in names), default=4)
    typer.echo(
        f"{'name':<{width}}  {'shape':<14}  {'max diff':>10}  {'allowed':>10}  verdict"
    )

    # Step 4: one row per reference record
    counts = {"pass": 0, "FAIL": 0, "missing": 0}
    first_failure = True
    for name in names:
        # Step 4a: the verdict. A record the dump lacks is counted as missing, and a shape mismatch fails without looking at the values
        expected = reference[name]
        shape = "x".join(map(str, expected.shape)) or "scalar"
        diff = allowed = None

        if name not in candidate:
            verdict = "missing"
        else:
            got = candidate[name]
            if got.shape != expected.shape:
                verdict = "FAIL"
                shape = f"{shape}≠{'x'.join(map(str, got.shape)) or 'scalar'}"
            elif exact:
                diff = _max_diff(expected, got)
                verdict = "pass" if _exact(expected, got) else "FAIL"
            else:
                # A NaN or Inf in the dump fails even when `_max_diff`, which skips NaNs, is within the allowed difference
                diff = _max_diff(expected, got)
                ref_max = float(np.max(np.abs(expected))) if expected.size else 0.0
                allowed = noise.allowed(ratios[noise.step(name)], ref_max)
                finite = bool(np.isfinite(got).all())
                verdict = "pass" if finite and diff <= allowed else "FAIL"

        # Step 4b: print the row, marking the first failure, which is where the port first went wrong
        counts[verdict] += 1
        diff_text = "" if diff is None else f"{diff:.3e}"
        if exact:
            allowed_text = "exact"
        elif allowed is None:
            allowed_text = ""
        else:
            allowed_text = f"{allowed:.3e}"
        marker = ""
        if verdict == "FAIL" and first_failure:
            marker = "  <- first failure"
            first_failure = False
        typer.echo(
            f"{name:<{width}}  {shape:<14}  {diff_text:>10}  {allowed_text:>10}  {verdict}{marker}"
        )

    # Step 5: the totals, and the dump's records that the reference doesn't have
    extra = [
        name
        for name in candidate
        if name not in reference and (only is None or name.startswith(only))
    ]
    typer.echo(
        f"\n{counts['pass']} pass, {counts['FAIL']} fail, {counts['missing']} missing, {len(extra)} extra"
    )
    for name in extra:
        typer.echo(f"  extra: {name}")

    # Step 6: the exit code. A dump with every name wrong, or a mistyped --only, would otherwise pass by comparing nothing
    if counts["pass"] + counts["FAIL"] == 0:
        typer.echo("No records compared", err=True)
        raise typer.Exit(1)
    if counts["FAIL"]:
        raise typer.Exit(1)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `compare-reference`."""

    typer.run(_compare_reference)
