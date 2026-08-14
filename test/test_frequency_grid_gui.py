"""The DRT step's silent frequency-grid correction, wired through a real
MainWindow: it has no checkbox or user-facing control, so what needs
checking is that it actually runs, that it leaves a non-qualifying sweep
alone, that it never touches the shared dataset objects, and that it stays
scoped to the DRT step the way the inductive-tail and diffusion-tail filters
beside it do.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.frequency_grid import nonuniformity
from core.io_utils import EISDataset

# A rounded geometric sweep, the same shape as the demo files this feature
# was built for: 3 significant figures, 20 points/decade.
N = 121
_ideal = np.logspace(5, -1, N)
QUANTISED_FREQUENCIES = np.array([float(f"{v:.3g}") for v in _ideal])
_w = 2 * np.pi * QUANTISED_FREQUENCIES
IMPEDANCES = 10 + 50 / (1 + 1j * _w * 5e-3) + 30 / (1 + 1j * _w * 0.3)

# Full precision: already uniform, so nothing to restore.
PLAIN_FREQUENCIES = np.logspace(5, -1, N)

assert nonuniformity(QUANTISED_FREQUENCIES) >= 0.01
assert nonuniformity(PLAIN_FREQUENCIES) < 0.01


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _dataset(frequencies, file_id=0, source_file="sim"):
    return EISDataset(DataSet(frequencies, IMPEDANCES), 0, source_file, file_id)


def _window_with(app, datasets):
    from gui.main_window import MainWindow

    window = MainWindow()
    window._datasets = datasets
    window._selection.set_datasets(datasets)
    window._selection.set_all(True)
    # The DRT step, in Singular -- where the pager lives and _drt_inputs is
    # actually exercised on redraw.
    window.step_stack.setCurrentIndex(2)
    window.drt_step.single_radio.setChecked(True)
    window._refresh()
    return window


@pytest.fixture
def window(app):
    return _window_with(app, [_dataset(QUANTISED_FREQUENCIES)])


def test_silently_restores_a_qualifying_sweep(window):
    inputs = window._drt_inputs(window._selected_datasets())
    assert nonuniformity(inputs[0].frequencies) < 1e-9

    orig = window._selected_datasets()[0]
    assert np.array_equal(inputs[0].impedances, orig.impedances)
    assert inputs[0].data.get_mask() == orig.data.get_mask()


def test_leaves_a_non_qualifying_sweep_alone(app):
    window = _window_with(app, [_dataset(PLAIN_FREQUENCIES)])
    inputs = window._drt_inputs(window._selected_datasets())
    assert np.array_equal(inputs[0].frequencies, PLAIN_FREQUENCIES)


def test_never_mutates_the_shared_dataset(window):
    """No checkbox to turn this off, so the one guarantee that matters is
    that the correction only ever touches a copy -- Validation and ECM must
    still see the sweep exactly as loaded."""
    before = window._selected_datasets()[0].frequencies.copy()
    window._drt_inputs(window._selected_datasets())
    after = window._selected_datasets()[0].frequencies
    assert np.array_equal(before, after)
    assert nonuniformity(after) >= 0.01, "the shared dataset must stay unrestored"


def test_scoped_to_the_drt_step_alone(window):
    """_drt_inputs is reached only from the DRT batch and the DRT step's own
    redraw (see gui/main_window.py) -- nothing routes it into Validation or
    ECM, so their own reads of _selected_datasets() are unaffected by
    construction, not by coincidence."""
    window._drt_inputs(window._selected_datasets())
    still_shared = window._selected_datasets()[0]
    assert nonuniformity(still_shared.frequencies) >= 0.01


def test_composes_with_the_other_drt_filters(window):
    """Correction runs first (see _drt_inputs's docstring on ordering), so
    the inductive-tail filter still sees a valid, fully-restored grid rather
    than fighting over which one goes first."""
    window.drt_step.remove_inductive_check.setChecked(True)
    inputs = window._drt_inputs(window._selected_datasets())
    assert nonuniformity(inputs[0].frequencies) < 1e-9
    assert inputs[0].num_points <= len(QUANTISED_FREQUENCIES)


def test_drt_run_uses_the_restored_frequencies(window, monkeypatch):
    """The actual batch-launch path, not just the redraw path."""
    from gui.workers import DRTWorker

    seen = {}

    def fake_run_drt(ds, **kwargs):
        seen["nonuniformity"] = nonuniformity(ds.frequencies)
        raise RuntimeError("stop before an actual DRT computation")

    monkeypatch.setattr("core.drt.run_drt", fake_run_drt)
    monkeypatch.setattr(DRTWorker, "start", lambda self: None)

    window._start_drt_run(window._selected_datasets(), {"rbf_type": "gaussian"}, "DRT")
    window._drt_worker.run()  # started is a no-op above; run it synchronously

    assert seen["nonuniformity"] < 1e-9
