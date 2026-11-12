"""The programs a command's units start, kept together so Ctrl-C can kill them all.

`check-spike` and `benchmark` run their units in threads, and every Spike or host run goes through one `Processes`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Processes:
    """Every program the units start, so one call can kill them all.

    Units run in threads of the parent process, and each one starts its programs through `run`.
    Once `stop` is called no program starts, and `run` reports the ones it killed as cut short.
    """

    def __init__(self) -> None:
        """Start with no programs and not stopping."""

        self._lock = threading.Lock()
        self._running: set[subprocess.Popen[bytes]] = set()
        self._stopping = False

    def run(
        self, command: list[str], stderr_path: Path, stdout_path: Path | None = None
    ) -> tuple[int, float] | None:
        """Run one program to the end, with its stderr in a file.

        Parameters
        ----------
        command : list[str]
            The program and its arguments, never joined into a shell string, because paths may contain spaces.
        stderr_path : Path
            Where its stderr goes.
        stdout_path : Path or None
            Where its stdout goes, or None to discard it.

        Returns
        -------
        tuple[int, float] or None
            Its exit code and wall seconds, or None when `stop` was called before it started or while it ran.

        Raises
        ------
        OSError
            When the program can't be started.
        """

        # Step 1: start it, unless stopping. The check and the start share the lock, so `stop` never misses a program that is just starting
        with ExitStack() as files, self._lock:
            stderr = files.enter_context(stderr_path.open("wb"))
            stdout = (
                subprocess.DEVNULL
                if stdout_path is None
                else files.enter_context(stdout_path.open("wb"))
            )
            if self._stopping:
                return None
            began = time.perf_counter()
            # stdin is closed because Spike drops into its interactive debugger on Ctrl-C, which would otherwise wait on the terminal
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            self._running.add(process)

        # Step 2: wait outside the lock, so other units can start theirs
        code = process.wait()
        seconds = time.perf_counter() - began
        with self._lock:
            self._running.discard(process)
            if self._stopping:
                return None
        return code, seconds

    def stop(self) -> None:
        """Kill every program still running and refuse to start new ones."""

        with self._lock:
            self._stopping = True
            running = list(self._running)
        for process in running:
            process.kill()
        for process in running:
            process.wait()
