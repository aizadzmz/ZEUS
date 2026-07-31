"""Background workers so long-running analysis never freezes the UI."""

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List

from PySide6.QtCore import QThread, Signal

# Cap on validation subprocesses, kept deliberately small. Each one is a
# fresh interpreter that imports pyimpspec/numpy/scipy -- ~6 s and a few
# hundred MB before it computes anything -- and an earlier 8-worker pool
# exhausted this machine's paging file ("DLL load failed ... _core" ->
# BrokenProcessPool). Past this point extra workers mostly buy memory
# pressure, since the batch is import-bound rather than CPU-bound.
MAX_VALIDATION_WORKERS = 4

# Only spread a batch across processes when the work remaining after the
# first sweep should take at least this long. Every worker pays that ~6 s
# import up front, so short batches lose outright -- as does Z-HIT at any
# realistic size (~0.16 s/sweep; a pooled 8-sweep Z-HIT batch measured
# 8.2 s against 2.4 s serial). The margin over pure break-even (~8 s) is
# deliberate: pooling also costs memory and pickling, so it should only be
# used where it wins clearly.
PARALLEL_MIN_ESTIMATED_SECONDS = 20.0


def _is_picklable(obj) -> bool:
    """Whether obj can cross a process boundary. Worth checking up front:
    an unpicklable runner (a lambda, or a closure like the one
    MainWindow._run_drt_bayesian builds) fails per-submitted-task, which
    would otherwise surface as one bogus error per dataset instead of a
    quiet fall back to running in this thread."""
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


class ValidationWorker(QThread):
    """Runs a validation method (KK or Z-HIT) over several datasets.

    Emits one result_ready per dataset so partial results land as they
    finish, plus error for any dataset that fails, and progress after each.
    QThread.finished fires when the whole batch is done.

    The first sweep always runs here, in this thread, and is timed. That
    measurement decides whether the rest is worth spreading across
    processes: it calibrates for the method, the sweep's point count and
    the machine at once, where a hardcoded per-method cost would not. That
    self-calibration is why core._pyimpspec_patches cutting Kramers-Kronig
    from ~3.4 s to ~0.75 s per 60-point sweep needed no change here; it
    just moves the crossover from a handful of sweeps to a few dozen, and
    the per-worker import cost the threshold is sized against is the same
    either way. Z-HIT (~0.05 s/sweep) essentially never crosses it, which
    is the intended outcome.

    When the pool is used, results arrive in completion order rather than
    selection order. Callers key them by dataset key, so this only shows
    up in the order signals fire.

    runner must be resolvable by name -- a module-level function, or a
    partial over one -- to be eligible for the pool, since spawned workers
    receive it pickled by reference. Anything else still works; it just
    stays serial.
    """

    result_ready = Signal(str, str, object)  # method name, dataset key, result
    error = Signal(str, str)                 # dataset key, message
    progress = Signal(int, int)              # completed count, total

    def __init__(self, method_name: str, runner: Callable, datasets: List, parent=None):
        super().__init__(parent)
        self._method_name = method_name
        self._runner = runner
        self._datasets = datasets
        self._completed = 0
        self._total = 0

    def run(self) -> None:
        self._total = len(self._datasets)
        self._completed = 0
        if self._total == 0:
            return

        first, rest = self._datasets[0], self._datasets[1:]
        elapsed = self._run_one(first)
        if not rest:
            return

        if not self._worth_pooling(elapsed, rest):
            self._run_serial(rest)
            return

        unfinished = self._run_pooled(rest)
        if unfinished:
            # The pool failed as a whole rather than per-dataset (a worker
            # died, memory ran out). Finish what it never reported here, so
            # a pool problem costs speed rather than the run.
            self._run_serial(unfinished)

    def _worth_pooling(self, seconds_per_sweep: float, rest: List) -> bool:
        if seconds_per_sweep * len(rest) < PARALLEL_MIN_ESTIMATED_SECONDS:
            return False
        return _is_picklable(self._runner) and _is_picklable(rest[0])

    def _run_pooled(self, datasets: List) -> List:
        """Returns the datasets the pool never reported on -- empty if it
        saw them all through. A dataset whose own run raised counts as
        reported: it failed on its merits, and recomputing it in this thread
        would only fail again, more slowly."""
        outstanding = {ds.key: ds for ds in datasets}
        workers = max(1, min(MAX_VALIDATION_WORKERS, os.cpu_count() or 1, len(datasets)))
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._runner, ds): ds for ds in datasets}
                for future in as_completed(futures):
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
        return list(outstanding.values())

    def _run_serial(self, datasets: List) -> None:
        for ds in datasets:
            self._run_one(ds)

    def _run_one(self, ds) -> float:
        """Computes one dataset here and reports it. Returns how long it
        took, which run() uses to size up the rest of the batch."""
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


class DRTWorker(QThread):
    """Runs a (potentially very slow) DRT calculation over several datasets
    off the UI thread — namely the Bayesian TR-RBF credible-interval run,
    whose HMC sampler can take tens of minutes per sweep.

    Emits one result_ready per dataset, plus error for any dataset that
    fails or times out. QThread.finished fires when the whole batch is done.
    """

    result_ready = Signal(str, object)  # dataset key, result
    error = Signal(str, str)            # dataset key, message

    def __init__(self, runner: Callable, datasets: List, parent=None):
        super().__init__(parent)
        self._runner = runner
        self._datasets = datasets

    def run(self) -> None:
        for ds in self._datasets:
            try:
                result = self._runner(ds)
            except Exception as exc:
                self.error.emit(ds.key, str(exc))
            else:
                self.result_ready.emit(ds.key, result)
