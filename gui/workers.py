"""Background workers so long-running analysis never freezes the UI."""

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List

from PySide6.QtCore import QThread, Signal

# Cap on validation subprocesses, kept small on purpose. Each is a fresh
# interpreter importing pyimpspec/numpy/scipy -- ~6 s and a few hundred MB
# before computing anything -- and an 8-worker pool exhausted the paging file
# ("DLL load failed ... _core" -> BrokenProcessPool). The batch is
# import-bound, not CPU-bound, so more workers mostly buy memory pressure.
MAX_VALIDATION_WORKERS = 4

# Only spread a batch across processes when the work left after the first sweep
# should take at least this long. Every worker pays that ~6 s import up front,
# so short batches lose outright, as does Z-HIT at any realistic size
# (~0.16 s/sweep). The margin over break-even (~8 s) is deliberate: pooling also
# costs memory and pickling, so it should only be used where it wins clearly.
PARALLEL_MIN_ESTIMATED_SECONDS = 20.0


def _is_picklable(obj) -> bool:
    """Whether obj can cross a process boundary. Checked up front so an
    unpicklable runner falls back to this thread quietly."""
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


class _BatchWorker(QThread):
    """A QThread that walks a batch of datasets and can be asked to stop.

    Cancellation is checked between datasets, not inside one: the work is a
    single pyimpspec call per sweep with no callback to interrupt it, so the
    honest guarantee is "no sweep after this one", not "stop now". That is
    enough for the case it exists to serve -- quitting the app without
    destroying a live thread, which aborts the process (0xC0000409).
    """

    def __init__(self, runner: Callable, datasets: List, parent=None):
        super().__init__(parent)
        self._runner = runner
        self._datasets = datasets
        # Plain bool rather than a lock: it is written once from the UI thread
        # and read between sweeps on this one, and a missed read costs one
        # extra sweep, not correctness.
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the batch to stop after the sweep in flight."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class ValidationWorker(_BatchWorker):
    """Runs a validation method (KK or Z-HIT) over several datasets.

    The runner decides what a "result" is: a plain ValidationResult for a
    single pass, or a core.validation.PruneOutcome when the runner is an
    iterative prune, which is many passes and reports what it removed."""

    result_ready = Signal(str, str, object)  # method name, dataset key, result
    error = Signal(str, str)                 # dataset key, message
    progress = Signal(int, int)              # completed count, total

    def __init__(self, method_name: str, runner: Callable, datasets: List, parent=None):
        super().__init__(runner, datasets, parent)
        self._method_name = method_name
        self._completed = 0
        self._total = 0

    def run(self) -> None:
        self._total = len(self._datasets)
        self._completed = 0
        if self._total == 0:
            return

        first, rest = self._datasets[0], self._datasets[1:]
        elapsed = self._run_one(first)
        if not rest or self._cancelled:
            return

        if not self._worth_pooling(elapsed, rest):
            self._run_serial(rest)
            return

        unfinished = self._run_pooled(rest)
        if unfinished:
            # The pool failed as a whole (a worker died, memory ran out).
            # Finish the rest here, so a pool problem costs speed, not the run.
            self._run_serial(unfinished)

    def _worth_pooling(self, seconds_per_sweep: float, rest: List) -> bool:
        if seconds_per_sweep * len(rest) < PARALLEL_MIN_ESTIMATED_SECONDS:
            return False
        return _is_picklable(self._runner) and _is_picklable(rest[0])

    def _run_pooled(self, datasets: List) -> List:
        """The datasets the pool never reported on; empty if it saw them all
        through. A dataset whose own run raised counts as reported."""
        outstanding = {ds.key: ds for ds in datasets}
        workers = max(1, min(MAX_VALIDATION_WORKERS, os.cpu_count() or 1, len(datasets)))
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._runner, ds): ds for ds in datasets}
                for future in as_completed(futures):
                    if self._cancelled:
                        # Drops what has not started and stops waiting on what
                        # has; the subprocesses still running are torn down by
                        # the pool's own exit.
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    ds = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self._report_error(ds.key, exc)
                    else:
                        self._report_result(ds.key, result)
                    outstanding.pop(ds.key, None)
        except Exception:
            pass
        # Nothing is outstanding once cancelled -- it was abandoned, not missed,
        # and run() must not hand it to the serial fallback to be redone.
        return [] if self._cancelled else list(outstanding.values())

    def _run_serial(self, datasets: List) -> None:
        for ds in datasets:
            if self._cancelled:
                return
            self._run_one(ds)

    def _run_one(self, ds) -> float:
        """Compute and report one dataset. Returns how long it took, which
        run() uses to size up the rest of the batch."""
        started = time.perf_counter()
        try:
            result = self._runner(ds)
        except Exception as exc:
            self._report_error(ds.key, exc)
        else:
            self._report_result(ds.key, result)
        return time.perf_counter() - started

    def _report_result(self, key: str, result) -> None:
        self.result_ready.emit(self._method_name, key, result)
        self._advance()

    def _report_error(self, key: str, exc: Exception) -> None:
        self.error.emit(key, str(exc))
        self._advance()

    def _advance(self) -> None:
        self._completed += 1
        self.progress.emit(self._completed, self._total)


class DRTWorker(_BatchWorker):
    """Runs a potentially very slow DRT calculation over several datasets off
    the UI thread, emitting one result_ready per dataset."""

    result_ready = Signal(str, object)  # dataset key, result
    error = Signal(str, str)            # dataset key, message

    def run(self) -> None:
        for ds in self._datasets:
            if self._cancelled:
                return
            try:
                result = self._runner(ds)
            except Exception as exc:
                self.error.emit(ds.key, str(exc))
            else:
                self.result_ready.emit(ds.key, result)


class ECMWorker(_BatchWorker):
    """Fits an equivalent circuit to several datasets off the UI thread."""

    result_ready = Signal(str, object)  # dataset key, FitResult
    error = Signal(str, str)            # dataset key, message
    progress = Signal(int, int)         # completed count, total

    def run(self) -> None:
        total = len(self._datasets)
        for completed, ds in enumerate(self._datasets, start=1):
            if self._cancelled:
                return
            try:
                result = self._runner(ds)
            except Exception as exc:
                self.error.emit(ds.key, str(exc))
            else:
                self.result_ready.emit(ds.key, result)
            self.progress.emit(completed, total)
