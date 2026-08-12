"""Exporting a sweep once it has been validated.

Drives MainWindow._export_validation_results rather than the writers directly
(test_zview_export covers those), because what is worth pinning here is that
the file gets the *validated* point set.
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
from core.validation import unmasked_indices

f = np.logspace(4, -1, 30)
w = 2 * np.pi * f
Z_true = 10 + 50 / (1 + 1j * w * 5e-3)

# One point the stub fit misses badly enough to be rejected at any sane limit.
BAD_INDEX = 7


class FitResult:
    """A stand-in for a KramersKronigResult, carrying what the app reads: the
    fit, pyimpspec's complex residuals, and the two accessors the ZView writer
    asks for."""

    def __init__(self, freq, Z_exp, Z_fit):
        self.frequencies = np.asarray(freq, dtype=float)
        self.impedances = np.asarray(Z_fit)
        self.residuals = (np.asarray(Z_exp) - self.impedances) / np.abs(Z_exp)

    def get_frequencies(self):
        return self.frequencies

    def get_impedances(self):
        return self.impedances

    def get_residuals_data(self):
        return self.frequencies, self.residuals.real * 100, self.residuals.imag * 100


def fake_run(ds, **kwargs):
    """Misses every point by 0.4% of |Z|, and BAD_INDEX by 8%."""
    freq = ds.data.get_frequencies()
    Z_exp = ds.data.get_impedances()
    Z_fit = Z_exp - 0.004 * np.abs(Z_exp) * (1 + 1j)
    Z_fit[BAD_INDEX] = Z_exp[BAD_INDEX] - 0.08 * np.abs(Z_exp[BAD_INDEX])
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

    def start(self):
        for ds in self._datasets:
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


def export(win, monkeypatch, path, filter_name):
    """Click the export button with the save dialog answered for us."""
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(path), filter_name)),
    )
    win._export_validation_results()


def read_z_rows(path):
    lines = path.read_bytes().decode("ascii").split("\r\n")[:-1]
    return lines[:11], lines[11:]


def test_export_needs_a_validation_first(win, monkeypatch, tmp_path):
    """Nothing has been run, so nothing is written -- and the user is told to
    run one rather than handed an empty file."""
    seen = []
    monkeypatch.setattr(
        main_window.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: seen.append(a[-1])),
    )
    export(win, monkeypatch, tmp_path / "x.z", win._VALIDATION_ZVIEW_FILTER)

    assert seen and "validation" in seen[0]
    assert not list(tmp_path.iterdir())


def test_zview_export_drops_the_rejected_point(win, monkeypatch, tmp_path):
    win.validation_step.threshold_spin.setValue(2.0)
    win._run_validation()

    ds = win._datasets[0]
    kept = len(unmasked_indices(ds))
    assert kept == len(f) - 1, "the stub's bad point should have been rejected"

    export(win, monkeypatch, tmp_path / "s.z", win._VALIDATION_ZVIEW_FILTER)

    header, rows = read_z_rows(tmp_path / "s.z")
    assert len(rows) == kept
    assert int(header[9]) == kept
    assert f"{kept} of {len(f)} points kept" in header[5]

    written = np.array([float(row.split(",")[0]) for row in rows])
    assert not np.any(np.isclose(written, f[BAD_INDEX], rtol=1e-5))


def test_zview_export_writes_the_fit_alongside(win, monkeypatch, tmp_path):
    """The companion .z spans what the validation was fitted on, not what
    survived it: a basic run fits every point and rejects afterwards, so the fit
    is drawn through the gap the spectrum now has."""
    win._run_validation()
    export(win, monkeypatch, tmp_path / "s.z", win._VALIDATION_ZVIEW_FILTER)

    fit = tmp_path / "s_fit.z"
    assert fit.exists()
    _, fit_rows = read_z_rows(fit)
    _, spectrum_rows = read_z_rows(tmp_path / "s.z")

    assert len(fit_rows) == len(f)
    assert len(spectrum_rows) == len(f) - 1
    fitted = np.array([float(row.split(",")[0]) for row in fit_rows])
    assert np.any(np.isclose(fitted, f[BAD_INDEX], rtol=1e-5))


def test_csv_export_writes_spectrum_and_residuals(win, monkeypatch, tmp_path):
    win._run_validation()
    export(win, monkeypatch, tmp_path / "s.csv", win._VALIDATION_CSV_FILTER)

    spectrum = (tmp_path / "s.csv").read_text().strip().splitlines()
    residuals = (tmp_path / "s_residuals.csv").read_text().strip().splitlines()

    # One header row each: the spectrum is the kept points, the residuals cover everything the validation was fitted on.
    assert len(spectrum) == len(unmasked_indices(win._datasets[0])) + 1
    assert len(residuals) == len(f) + 1


def test_typed_suffix_beats_the_filter(win, monkeypatch, tmp_path):
    """The dropdown says CSV, the filename says .z -- the filename wins, and no
    stray .csv is left behind."""
    win._run_validation()
    export(win, monkeypatch, tmp_path / "s.z", win._VALIDATION_CSV_FILTER)

    assert (tmp_path / "s.z").exists()
    assert not (tmp_path / "s.csv").exists()


def test_missing_suffix_takes_it_from_the_filter(win, monkeypatch, tmp_path):
    win._run_validation()
    export(win, monkeypatch, tmp_path / "s", win._VALIDATION_ZVIEW_FILTER)

    assert (tmp_path / "s.z").exists()
