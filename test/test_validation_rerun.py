"""Re-running a validation must be repeatable.

The bug this pins: a run read the dataset's mask as the last redraw left it,
which included that redraw's threshold rejections. So the result was fitted on
fewer points than `pruned_points` accounted for, _refresh could not reproduce
the point set, and every later run reported the result as stale -- permanently,
since each run then started from a different mask again.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

import gui.main_window as main_window
from core.io_utils import EISDataset
from core.validation import (
    RESIDUAL_BY_COMPONENT,
    RESIDUAL_BY_MODULUS,
    unmasked_indices,
)

f = np.logspace(4, -1, 30)
w = 2 * np.pi * f
Z_true = 10 + 50 / (1 + 1j * w * 5e-3)

# Two sweeps' worth of frequencies the stub runner will fit badly, picked by
# value so they survive the masking that goes on between passes.
BAD_FREQUENCIES = {float(f[5]), float(f[9])}


class FitResult:
    """A stand-in for a KramersKronigResult carrying the three fields the app
    reads: the fit, pyimpspec's complex residuals, and its percent view."""

    def __init__(self, freq, Z_exp, Z_fit):
        self.frequencies = np.asarray(freq, dtype=float)
        self.impedances = np.asarray(Z_fit)
        self.residuals = (np.asarray(Z_exp) - self.impedances) / np.abs(Z_exp)

    def get_residuals_data(self):
        return self.frequencies, self.residuals.real * 100, self.residuals.imag * 100


def fake_run(ds, **kwargs):
    """Stands in for run_kk_test. Misses every point by 0.4% of |Z| and the two
    marked frequencies by 8%, so an advanced prune has something to remove and
    the residual arrays always line up with the points handed in."""
    freq = ds.data.get_frequencies()  # unmasked only, in order
    Z_exp = ds.data.get_impedances()
    Z_fit = Z_exp - 0.004 * np.abs(Z_exp) * (1 + 1j)
    for i, fr in enumerate(freq):
        if float(fr) in BAD_FREQUENCIES:
            Z_fit[i] = Z_exp[i] - 0.08 * np.abs(Z_exp[i])
    return FitResult(freq, Z_exp, Z_fit)


class SyncWorker(QObject):
    """ValidationWorker's interface, run inline so the test is deterministic."""

    result_ready = Signal(str, str, object)
    error = Signal(str, str)
    progress = Signal(int, int)
    finished = Signal()

    def __init__(self, method_name, runner, datasets, parent=None):
        super().__init__(parent)
        self._method_name = method_name
        self._runner = runner
        self._datasets = datasets
        # What each run was handed, so a test can see the input drifting.
        self.observed = []

    def start(self):
        for ds in self._datasets:
            self.observed.append(len(unmasked_indices(ds)))
            self.result_ready.emit(self._method_name, ds.key, self._runner(ds))
        self.finished.emit()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, monkeypatch):
    monkeypatch.setattr(main_window, "ValidationWorker", SyncWorker)
    monkeypatch.setattr("core.validation.run_kk_test", fake_run)

    window = main_window.MainWindow()
    ds = EISDataset(
        DataSet(frequencies=f, impedances=Z_true), index=0, source_file="synthetic"
    )
    window._datasets = [ds]
    window._selection.set_datasets([ds])
    window._selection.set_all(True)
    window.validation_step.single_radio.setChecked(True)
    return window


def _dataset(win):
    return win._datasets[0]


def _observed(win, runs):
    """Run `runs` times, returning the unmasked count each run started from."""
    seen = []
    real_worker_cls = main_window.ValidationWorker

    def spy(method_name, runner, datasets, parent=None):
        worker = real_worker_cls(method_name, runner, datasets, parent)
        seen.append(worker)
        return worker

    main_window.ValidationWorker = spy
    try:
        for _ in range(runs):
            win._run_validation()
    finally:
        main_window.ValidationWorker = real_worker_cls
    return [n for worker in seen for n in worker.observed]


def test_advanced_rerun_stays_consistent_after_switching_convention(win):
    """The reported case: run, switch to the component convention, run again."""
    win.validation_step.advanced_radio.setChecked(True)

    first = _observed(win, 1)
    assert win.warning_label.text() == "", win.warning_label.text()
    pruned = win._pruned_points(win._validation_method, _dataset(win).key)
    assert len(pruned) == 2, pruned  # the two badly-fitted frequencies

    # Switching re-rejects against the hard limit and must not error.
    win.validation_step.residual_component_radio.setChecked(True)
    assert win.validation_step.residual_mode == RESIDUAL_BY_COMPONENT
    assert win.warning_label.text() == "", win.warning_label.text()
    assert len(unmasked_indices(_dataset(win))) < len(f) - 2, "convention rejected nothing"

    # ...and neither must re-running, however many times.
    later = _observed(win, 3)
    assert win.warning_label.text() == "", win.warning_label.text()

    # Every run saw the same input: the filtered data, not the last redraw's
    # leftovers. This is the assertion that fails without the fix.
    assert set(first + later) == {len(f)}, first + later


def test_basic_rerun_is_repeatable(win):
    """Same defect, no convention switch needed: the first run's own threshold
    rejections used to become the second run's input."""
    win.validation_step.basic_radio.setChecked(True)
    win.validation_step.threshold_spin.setValue(0.3)  # below the stub's 0.4% miss

    observed = _observed(win, 3)
    assert set(observed) == {len(f)}, observed
    assert win.warning_label.text() == "", win.warning_label.text()


def test_the_residual_figure_survives_a_rerun(win):
    """No stale result means the figure keeps drawing."""
    win.validation_step.advanced_radio.setChecked(True)
    assert win._selection.current() is not None  # set_all seats the cursor
    # Figures are built lazily for the visible step only, so open this one.
    win.step_stack.setCurrentIndex(1)

    _observed(win, 1)
    assert win.validation_step.residuals_pane._widget is not None

    win.validation_step.residual_component_radio.setChecked(True)
    _observed(win, 2)
    assert win.validation_step.residuals_pane._widget is not None
    assert win.warning_label.text() == ""


def test_the_eraser_and_inductive_filter_still_reach_the_run(win):
    """The reset restores the base mask -- it must not throw away the user's
    own edits along with the threshold pass."""
    ds = _dataset(win)
    win.validation_step.basic_radio.setChecked(True)
    win._manual_masked = {ds.key: {2, 3, 4}}

    observed = _observed(win, 2)
    assert set(observed) == {len(f) - 3}, observed
    assert win.warning_label.text() == ""
    for index in (2, 3, 4):
        assert index not in unmasked_indices(ds)


def test_modulus_only_reruns_are_stable_too(win):
    win.validation_step.advanced_radio.setChecked(True)
    assert win.validation_step.residual_mode == RESIDUAL_BY_MODULUS
    observed = _observed(win, 4)
    assert set(observed) == {len(f)}, observed
    assert win.warning_label.text() == ""
