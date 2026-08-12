"""What a DRT batch records about the diffusion tail it was computed on.

That record is what "Build circuit from DRT" reads to put the fitted element
back into the model, and getting it wrong is silent in both directions: a
missing element leaves the R-CPE pairs absorbing a tail that is still in the
data, and a spurious one subtracts a tail twice.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

import core.filtering as filtering
from core.io_utils import EISDataset

f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = 10.0 + 50.0 / (1 + 1j * w * 5e-3) + (1 - 1j) * 3.0 / np.sqrt(w)

ELEMENT_CDC = "W{Y=0.05}"


class _StubFit:
    """Stands in for a FitResult; the fitting itself is covered by
    test_diffusion_subtraction, and a real CNLS fit per sweep would cost the
    suite minutes. Carries a real parsed circuit, because the readout renders it
    through core.filtering.describe_diffusion_fit on every redraw."""

    pseudo_chisqr = 0.01

    def __init__(self):
        from pyimpspec import parse_cdc

        self.circuit = parse_cdc("R(RQ)W")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    from gui.main_window import MainWindow
    from gui.workers import DRTWorker

    # Stubbed at the seam core.filtering exposes, so main_window's function-local imports pick it up.
    monkeypatch.setattr(
        filtering,
        "diffusion_impedance",
        lambda ds, cdc, **kw: (np.zeros(len(f), dtype=complex), _StubFit()),
    )
    monkeypatch.setattr(filtering, "diffusion_element_cdc", lambda fit: ELEMENT_CDC)
    # The worker is never wanted here: results are delivered by calling _on_drt_worker_result directly, which leaves the test free of thread timing.
    monkeypatch.setattr(DRTWorker, "start", lambda self: None)

    window = MainWindow()
    window._datasets = [EISDataset(DataSet(f, Z), i, "sim", 0) for i in range(3)]
    window._selection.set_datasets(window._datasets)
    window._selection.set_all(True)
    # The DRT step, in Singular -- where the pager lives and draws one sweep.
    window.step_stack.setCurrentIndex(2)
    window.drt_step.single_radio.setChecked(True)
    return window


def _keys(window):
    return [ds.key for ds in window._datasets]


def _run_batch(window):
    """Start a DRT batch the way the Run button does, without a worker."""
    window._start_drt_run(window._selected_datasets(), {"rbf_type": "gaussian"}, "DRT")


def _deliver_all(window):
    """Every sweep's result arriving, as the worker thread delivers them."""
    for key in _keys(window):
        window._on_drt_worker_result(key, object())


def _recorded(window):
    return {k: window._drt_params[k].get("diffusion_element") for k in _keys(window)}


def test_the_batch_records_the_element_for_every_sweep(window):
    window.drt_step.subtract_diffusion_check.setChecked(True)
    _run_batch(window)
    _deliver_all(window)

    assert _recorded(window) == {k: ELEMENT_CDC for k in _keys(window)}


def test_paging_mid_run_does_not_lose_the_element(window):
    """The pager sits outside settings_scroll, so _set_controls_enabled leaves
    it live through a run -- deliberately, so a long batch can be watched. One
    click on › used to shrink the record to the sweep on screen, and every
    result still to come recorded no element at all."""
    window.drt_step.subtract_diffusion_check.setChecked(True)
    _run_batch(window)

    # The user looks at another sweep while the batch runs.
    window._selection.step_cursor(1)  # -> cursor_moved -> _refresh
    _deliver_all(window)

    assert _recorded(window) == {k: ELEMENT_CDC for k in _keys(window)}


def test_the_on_screen_record_still_narrows_to_what_is_drawn(window):
    """The other half of the same fix: _diffusion_shown is meant to follow the
    screen, because that is what the readout under the checkbox describes. It
    must keep doing so -- the run's copy is what stops that mattering."""
    window.drt_step.subtract_diffusion_check.setChecked(True)
    _run_batch(window)
    assert set(window._pending_diffusion) == set(_keys(window))

    window._refresh()
    assert len(window._diffusion_shown) == 1
    assert set(window._pending_diffusion) == set(_keys(window))


def test_turning_the_filter_off_records_no_element(window):
    """The mirror-image failure. Fits left over from when the subtraction was
    on used to be read as the new batch's, appending a diffusion element to a
    circuit built from a sweep that still had its tail -- counting it twice."""
    window.drt_step.subtract_diffusion_check.setChecked(True)
    window._drt_inputs(window._selected_datasets())  # a draw, which fits them
    assert window._diffusion_shown

    window.drt_step.subtract_diffusion_check.setChecked(False)
    _run_batch(window)
    _deliver_all(window)

    assert _recorded(window) == {k: None for k in _keys(window)}
    assert window._drt_params[_keys(window)[0]]["subtract_diffusion"] is False


def test_a_failed_fit_records_no_element(window, monkeypatch):
    """A failure is cached as its message in the fit's place. That is not
    something to append to a circuit."""
    monkeypatch.setattr(
        filtering,
        "diffusion_impedance",
        lambda ds, cdc, **kw: (_ for _ in ()).throw(ValueError("no fit")),
    )
    window.drt_step.subtract_diffusion_check.setChecked(True)
    _run_batch(window)
    _deliver_all(window)

    assert _recorded(window) == {k: None for k in _keys(window)}


def test_the_drt_filters_survive_a_session_round_trip(window, tmp_path, monkeypatch):
    """Saved but never restored, they came back off, and the reloaded step drew
    an unfiltered spectrum under a DRT curve computed from a filtered one."""
    from PySide6.QtWidgets import QFileDialog

    # Save and load are reached only through their file dialogs; answering those drives the real paths rather than a seam added for the test's convenience.
    path = str(tmp_path / "s.eisz")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (path, ""))

    window.drt_step.remove_inductive_check.setChecked(True)
    window.drt_step.subtract_diffusion_check.setChecked(True)
    index = window.drt_step.diffusion_cdc_combo.findData("R(RQ)Ws")
    window.drt_step.diffusion_cdc_combo.setCurrentIndex(index)

    window._save_session()

    window.drt_step.remove_inductive_check.setChecked(False)
    window.drt_step.subtract_diffusion_check.setChecked(False)
    window.drt_step.diffusion_cdc_combo.setCurrentIndex(0)

    window._load_session()

    assert window.drt_step.remove_inductive_check.isChecked()
    assert window.drt_step.subtract_diffusion_check.isChecked()
    assert window.drt_step.diffusion_cdc_combo.currentData() == "R(RQ)Ws"
    # The combo is only meaningful with the filter on, and the signal that re-enables its row was blocked through the restore.
    assert window.drt_step.diffusion_cdc_combo.isEnabled()


def test_build_from_drt_appends_the_element_after_paging(window):
    """End to end: the circuit the ECM box actually receives."""
    from core.drt import analyze_drt_peaks

    window.drt_step.subtract_diffusion_check.setChecked(True)
    _run_batch(window)
    window._selection.step_cursor(1)
    _deliver_all(window)

    # Peaks are what _build_circuit_from_drt keys off; a real DRT is not needed to check the tail is put back on the end.
    from core.drt import run_drt

    result = run_drt(window._datasets[0])
    window._drt_peaks = {k: analyze_drt_peaks(result, num_peaks=2) for k in _keys(window)}

    window._build_circuit_from_drt()
    assert window.ecm_step.cdc_edit.text().endswith(ELEMENT_CDC)
