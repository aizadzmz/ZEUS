"""The diffusion-fit cache is keyed by ds.key, and ds.key is a position
(file_id:index) rather than an identity.

That is fine while the sweeps behind those positions stay put, and wrong the
moment they are replaced wholesale -- the first sweep of a restored session is
"0:0" just as the first sweep of whatever was open before it was. Reusing that
entry subtracts the previous data's fitted tail from the new sweep and hands the
result to the DRT, with nothing on screen to say so.
"""
import os

# Must precede any QApplication: the suite runs without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

import core.filtering as filtering
from core.io_utils import EISDataset

f = np.logspace(5, -1, 40)
w = 2 * np.pi * f


def sweep(scale=1.0, index=0, file_id=0, source="sim"):
    Z = 10.0 + 50.0 * scale / (1 + 1j * w * 5e-3) + (1 - 1j) * 3.0 / np.sqrt(w)
    return EISDataset(DataSet(f, Z), index=index, source_file=source, file_id=file_id)


class _StubFit:
    """Stands in for a FitResult, as in test_diffusion_record; the fitting
    itself is covered by test_diffusion_subtraction."""

    def __init__(self, tag):
        from pyimpspec import parse_cdc

        self.circuit = parse_cdc("R(RQ)W")
        self.pseudo_chisqr = tag


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    """A window whose diffusion fit is stubbed, and counted: one call per fit
    that was actually computed rather than served from the cache."""
    from gui.main_window import MainWindow

    calls = []

    def fake_fit(ds, cdc, **kwargs):
        calls.append(ds.key)
        # Tagged by call order, so two fits are never equal by accident.
        return np.zeros(len(f), dtype=complex), _StubFit(float(len(calls)))

    monkeypatch.setattr(filtering, "diffusion_impedance", fake_fit)

    window = MainWindow()
    window.calls = calls
    return window


def _load(window, datasets):
    window._datasets = datasets
    window._selection.set_datasets(datasets)
    window._selection.set_all(True)
    window.step_stack.setCurrentIndex(2)
    window.drt_step.single_radio.setChecked(True)
    window.drt_step.subtract_diffusion_check.setChecked(True)
    window._refresh()


def test_a_restored_session_does_not_inherit_the_previous_fits(window):
    """What File > Open session does: swap _datasets wholesale, keeping the
    window. The keys collide; the fits must not."""
    _load(window, [sweep(scale=1.0)])
    first = window._diffusion_shown["0:0"]
    assert window.calls == ["0:0"], window.calls

    # A different sweep, arriving under the very same key.
    replacement = sweep(scale=8.0, source="restored")
    assert replacement.key == "0:0"
    window._discard_diffusion_fits()   # as _load_session does
    _load(window, [replacement])

    second = window._diffusion_shown["0:0"]
    assert second is not first, "the previous data's fit was reused"
    assert window.calls == ["0:0", "0:0"], window.calls


def test_a_fresh_open_does_not_inherit_them_either(window):
    """_reset_state covers the same ground for File > Open."""
    _load(window, [sweep(scale=1.0)])
    first = window._diffusion_shown["0:0"]

    window._reset_state()
    assert window._diffusion_fits == {}
    assert window._diffusion_shown == {}
    assert window._pending_diffusion == {}

    _load(window, [sweep(scale=8.0)])
    assert window._diffusion_shown["0:0"] is not first


def test_the_cache_still_serves_an_unchanged_sweep(window):
    """The point of the cache: it runs on every redraw, not just on Run. Only
    a wholesale replacement may drop it."""
    _load(window, [sweep()])
    assert window.calls == ["0:0"]

    for _ in range(4):
        window._refresh()
    assert window.calls == ["0:0"], "a redraw refitted an unchanged sweep"


def test_an_eraser_edit_refits(window):
    """The mask is part of the key, so changing it must not serve the old fit."""
    _load(window, [sweep()])
    assert len(window.calls) == 1

    window._on_point_mask_toggled("0:0", 3)
    assert len(window.calls) == 2, window.calls
