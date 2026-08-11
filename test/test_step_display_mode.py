"""Each step's Singular/Multiple toggle, and the one step that has none.

The ECM step was fixed on Singular rather than given a toggle: its circuit
diagram carries one sweep's fitted values and the report below it one sweep's
parameters, so Multiple offered a comparison only the top plot could make.
Removing a control every step was assumed to have is what these cover -- the
wiring, the mode lookup, and the pager that used to appear only in Singular.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.io_utils import EISDataset

f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = 10.0 + 50.0 / (1 + 1j * w * 5e-3)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    from gui.main_window import MainWindow

    window = MainWindow()
    window._datasets = [EISDataset(DataSet(f, Z), i, "sim", 0) for i in range(3)]
    window._selection.set_datasets(window._datasets)
    window._selection.set_all(True)
    return window


def test_ecm_has_no_toggle_and_is_fixed_on_single(window):
    assert window.ecm_step.single_radio is None
    assert window.ecm_step.combined_radio is None
    assert window.ecm_step.display_mode == "Single"


def test_every_other_step_keeps_its_toggle(window):
    for step in (window.data_viz_step, window.validation_step, window.drt_step):
        assert step.single_radio is not None, step.__class__.__name__
        step.combined_radio.setChecked(True)
        assert step.display_mode == "Combined"
        step.single_radio.setChecked(True)
        assert step.display_mode == "Single"


def test_the_mode_lookup_still_answers_for_every_step(window):
    """_display_mode_for indexes into _steps(), so a step without the control
    has to answer rather than raise -- _displayed_datasets calls it for all
    four on every refresh."""
    modes = [window._display_mode_for(i) for i in range(4)]
    assert modes[3] == "Single"
    assert all(m in ("Single", "Combined") for m in modes)


def test_ecm_draws_one_sweep_however_many_are_selected(window):
    window._refresh()

    assert len(window._selected_datasets()) == 3
    assert len(window._pending["displayed_for"][3]) == 1


def test_fitting_still_covers_the_whole_selection(window):
    """The toggle governed the view, never the action. Removing it must not
    quietly narrow what 'Fit circuit' runs on."""
    window._refresh()
    assert len(window._selected_datasets()) == 3


def test_the_ecm_pager_is_always_visible(window, app):
    """It used to be shown only in Singular. Fixed on Singular, it is the only
    way to reach the other sweeps, so it must never be hidden."""
    window.show()
    window.step_stack.setCurrentIndex(3)
    app.processEvents()

    window._sync_display_mode_widgets()
    assert window.ecm_step.pager.isVisible()


def test_combined_mode_still_hides_the_pagers_that_have_a_toggle(window, app):
    """The rule the ECM step is now exempt from, still applied to the rest."""
    window.show()
    window.step_stack.setCurrentIndex(2)
    app.processEvents()

    window.drt_step.combined_radio.setChecked(True)
    window._sync_display_mode_widgets()
    assert not window.drt_step.pager.isVisible()

    window.drt_step.single_radio.setChecked(True)
    window._sync_display_mode_widgets()
    assert window.drt_step.pager.isVisible()
