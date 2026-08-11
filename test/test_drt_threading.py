"""Every DRT run goes to the worker thread, on a snapshot of its input.

Plain TR-RBF used to run inline in the click handler, on the assumption that it
is the quick method. It is quick only when pyimpspec can assemble the A matrix
by its Toeplitz shortcut, which needs the sweep's frequencies log-spaced to
within 1%; a pruned sweep or an instrument writing three-significant-figure
frequencies misses it and pays 2N**2 numerical integrations instead of 2N --
tens of seconds per sweep with the window frozen for all of them.

Running off the UI thread leaves the plots live, and the eraser with them, so
the worker is handed detached copies rather than the sweeps still on screen.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox
from pyimpspec import DataSet

import gui.main_window as main_window
from core.io_utils import EISDataset

f = np.logspace(4, -1, 30)
w = 2 * np.pi * f
# The series inductance is there to give the step's inductive-tail filter
# something to drop: without it Im(Z) < 0 everywhere and that test is vacuous.
Z = 10 + 50 / (1 + 1j * w * 5e-3) + 1j * w * 1e-4


class Result:
    """The two attributes the app reads off a DRT result."""

    def __init__(self, tau, gamma):
        self._data = (tau, gamma)
        self.lambda_value = 1e-3

    def get_drt_data(self):
        return self._data


class SyncWorker(QObject):
    """DRTWorker's interface, run inline so the test is deterministic.

    Instances land in SyncWorker.spawned, which is what lets a test see the
    datasets the worker was handed rather than the ones still on screen.
    """

    result_ready = Signal(str, object)
    error = Signal(str, str)
    finished = Signal()

    spawned = []

    def __init__(self, runner, datasets, parent=None):
        super().__init__(parent)
        self._runner = runner
        self.datasets = datasets
        SyncWorker.spawned.append(self)

    def start(self):
        for ds in self.datasets:
            try:
                result = self._runner(ds)
            except Exception as exc:
                self.error.emit(ds.key, str(exc))
            else:
                self.result_ready.emit(ds.key, result)
        self.finished.emit()


def fake_run_drt(ds, **kwargs):
    """Stands in for run_drt, recording what it was called with."""
    freq = ds.data.get_frequencies()  # unmasked only
    fake_run_drt.calls.append((ds.key, len(freq), kwargs))
    tau = 1.0 / (2 * np.pi * freq)
    return Result(tau, np.ones_like(tau))


fake_run_drt.calls = []


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, monkeypatch):
    SyncWorker.spawned = []
    fake_run_drt.calls = []
    monkeypatch.setattr(main_window, "DRTWorker", SyncWorker)
    monkeypatch.setattr("core.drt.run_drt", fake_run_drt)
    # Nothing in these tests should raise, and a modal box would hang the run.
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    window = main_window.MainWindow()
    sweeps = [
        EISDataset(DataSet(frequencies=f, impedances=Z), index=i, source_file="synthetic")
        for i in range(2)
    ]
    window._datasets = sweeps
    window._selection.set_datasets(sweeps)
    window._selection.set_all(True)
    window.drt_step.trrbf_radio.setChecked(True)
    return window


def test_the_simple_run_goes_through_the_worker(win):
    """The regression: TR-RBF used to loop in the click handler."""
    win._run_drt()

    assert len(SyncWorker.spawned) == 1, "no worker was started"
    assert [key for key, _, _ in fake_run_drt.calls] == [ds.key for ds in win._datasets]
    assert set(win._drt_results) == {ds.key for ds in win._datasets}


def test_the_worker_is_handed_copies_not_the_live_sweeps(win):
    """The eraser stays live while a run is in flight, so the sweeps it can
    remask must not be the ones being computed on."""
    win.drt_step.remove_inductive_check.setChecked(False)
    win._run_drt()

    handed = SyncWorker.spawned[0].datasets
    assert len(handed) == len(win._datasets)
    for copy, original in zip(handed, win._datasets):
        assert copy.data is not original.data
        # Same sweep, so results still file under it.
        assert copy.key == original.key

    # Masking a copy leaves the sweep on screen alone, and vice versa.
    handed[0].data.set_mask({0: True})
    assert win._datasets[0].data.get_num_points(masked=False) == len(f)


def test_the_inductive_filter_still_reaches_the_run(win):
    """The step's own filter is applied to the copies, not the originals."""
    win.drt_step.remove_inductive_check.setChecked(True)
    win._run_drt()

    inductive = int(np.sum(Z.imag > 0))
    assert inductive > 0, "the fixture has no inductive tail to drop"
    assert all(n == len(f) - inductive for _, n, _ in fake_run_drt.calls)
    # Whatever it dropped, it dropped on a copy.
    for original in win._datasets:
        assert original.data.get_num_points(masked=False) == len(f)


def test_settings_unlock_and_the_summary_counts_sweeps(win):
    win._run_drt()

    for step in win._steps():
        assert step.settings_scroll.isEnabled()
    message = win.statusBar().currentMessage()
    assert "DRT computed for 2 of 2 sweep(s)." == message, message


def test_a_failed_sweep_is_reported_without_losing_the_others(win, monkeypatch):
    def half_broken(ds, **kwargs):
        if ds.index == 0:
            raise ValueError("singular matrix")
        return fake_run_drt(ds, **kwargs)

    monkeypatch.setattr("core.drt.run_drt", half_broken)
    win._run_drt()

    assert [key for key, _ in win._drt_worker_errors] == [win._datasets[0].key]
    assert set(win._drt_results) == {win._datasets[1].key}
    message = win.statusBar().currentMessage()
    assert "DRT computed for 1 of 2 sweep(s)." == message, message


def test_the_bayesian_run_still_takes_the_same_path(win, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    win.drt_step.bayesian_radio.setChecked(True)
    win._run_drt()

    assert len(SyncWorker.spawned) == 1
    _, _, kwargs = fake_run_drt.calls[0]
    assert kwargs["credible_intervals"] is True
    assert kwargs["num_samples"] == win.drt_step.num_samples_spin.value()
    assert kwargs["timeout"] == win.drt_step.timeout_spin.value()
    assert "Bayesian DRT computed for 2 of 2 sweep(s)." == win.statusBar().currentMessage()


def test_the_simple_run_asks_for_no_sampling(win):
    win._run_drt()

    _, _, kwargs = fake_run_drt.calls[0]
    assert kwargs["credible_intervals"] is False
    # The sampling rows belong to the Bayesian branch alone.
    assert "num_samples" not in kwargs
    assert "timeout" not in kwargs
