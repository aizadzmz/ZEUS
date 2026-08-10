"""The Replot / Auto-Scale / Save Image overlay stays inside the plotted area.

The legend column takes its width from its entries, so switching a pane from a
single sweep to a multi-file overlay narrows the ViewBox after the widget is
laid out. The buttons have to follow it instead of being placed once.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.plotting import build_nyquist_plot
from gui.figure_panes import PgFigurePane

STEM_A = "LGM50LT EIS CALIBRATION_C01"
STEM_B = "LGM50LT EIS CALIBRATION_C01_part2_DRIFT_REP_BAND_01_MB_C01"

f = np.logspace(5, -1, 30)
w = 2 * np.pi * f
Z = 10.0 + 50.0 / (1 + 1j * w * 5e-3)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _datasets(counts):
    return [
        EISDataset(DataSet(frequencies=f, impedances=Z), index=i,
                   source_file=stem, file_id=file_id)
        for file_id, (stem, n) in enumerate(counts)
        for i in range(n)
    ]


def _pane(width=1000, height=600, with_eraser=False):
    pane = PgFigurePane(with_overlay_actions=True, with_eraser=with_eraser)
    pane.resize(width, height)
    pane.show()
    QApplication.processEvents()
    return pane


def _draw(pane, counts):
    pane.set_widget(build_nyquist_plot(_datasets(counts)))
    # Twice: set_widget defers the first placement to a zero-timer, and the
    # legend sizes itself on the layout pass that follows.
    QApplication.processEvents()
    QApplication.processEvents()


def _plot_rect(pane):
    """The ViewBox -- the gridded area -- in the pane's own coordinates."""
    widget = pane._widget
    scene_rect = widget.getPlotItem().getViewBox().sceneBoundingRect()
    corners = [
        widget.mapTo(pane, widget.mapFromScene(QPointF(x, y)))
        for x, y in (
            (scene_rect.left(), scene_rect.top()),
            (scene_rect.right(), scene_rect.bottom()),
        )
    ]
    return corners[0].x(), corners[0].y(), corners[1].x(), corners[1].y()


def _assert_inside_plot(pane):
    left, top, right, bottom = _plot_rect(pane)
    box = pane._overlay.geometry()
    # A pixel of slack: the scene-to-widget mapping rounds.
    assert box.right() <= right + 1, f"overlay spills into the legend: {box} vs {right}"
    assert box.bottom() <= bottom + 1
    assert box.left() >= left - 1
    assert box.top() >= top - 1


def test_single_sweep_overlay_sits_in_the_plot(app):
    pane = _pane()
    _draw(pane, [(STEM_A, 1)])
    _assert_inside_plot(pane)


def test_multi_file_overlay_sits_in_the_plot(app):
    pane = _pane()
    _draw(pane, [(STEM_A, 8), (STEM_B, 4)])
    _assert_inside_plot(pane)


def test_overlay_follows_the_plot_when_the_legend_widens(app):
    """The reported case: a pane showing one sweep is replotted with many, and
    the wider legend must not be left holding the buttons."""
    pane = _pane()
    _draw(pane, [(STEM_A, 1)])
    single_right = pane._overlay.geometry().right()

    _draw(pane, [(STEM_A, 8), (STEM_B, 4)])
    _assert_inside_plot(pane)
    # The plot really did shrink, so this is not a vacuous pass.
    assert pane._overlay.geometry().right() < single_right


def test_overlay_follows_a_pane_resize(app):
    pane = _pane()
    _draw(pane, [(STEM_A, 8), (STEM_B, 4)])
    pane.resize(700, 450)
    QApplication.processEvents()
    _assert_inside_plot(pane)


# -- folding the card away ---------------------------------------------------
def _collapse(pane, collapsed=True):
    pane._collapse_button.setChecked(collapsed)
    QApplication.processEvents()
    return pane._overlay.geometry()


def test_collapsing_takes_the_card_off_the_plot_entirely(app):
    """The complaint: on a Bode plot the card sits over the data."""
    pane = _pane()
    _draw(pane, [(STEM_A, 1)])
    assert pane._overlay.isVisible()

    _collapse(pane)
    assert not pane._overlay.isVisible(), "card still drawn over the data"
    assert pane._collapse_button.isVisible()

    _collapse(pane, False)
    assert pane._overlay.isVisible()


def test_the_chevron_sits_below_the_plot_not_on_it(app):
    """Folded, the chevron is all that is left, so it must be clear of the
    plotted area -- down in the x-axis strip."""
    pane = _pane()
    _draw(pane, [(STEM_A, 1)])
    _, _, plot_right, plot_bottom = _plot_rect(pane)

    for collapsed in (True, False):
        _collapse(pane, collapsed)
        chevron = pane._collapse_button.geometry()
        assert chevron.top() > plot_bottom, (collapsed, chevron, plot_bottom)
        assert chevron.bottom() <= pane.height(), chevron
        # Right-aligned with the plot, so it clears the legend column.
        assert chevron.right() <= plot_right + 1, (chevron, plot_right)


def test_the_chevron_does_not_move_when_the_card_folds(app):
    """It is a fixed handle, not part of the card -- it must not shift out
    from under the cursor that just clicked it."""
    pane = _pane()
    _draw(pane, [(STEM_A, 1)])

    _collapse(pane, False)
    open_at = pane._collapse_button.geometry()
    _collapse(pane, True)
    assert pane._collapse_button.geometry() == open_at


def test_the_open_card_is_still_inside_the_plot(app):
    """Unchanged by the move: opening still puts the buttons in the corner of
    the plotted area, clear of the legend."""
    pane = _pane()
    _draw(pane, [(STEM_A, 8), (STEM_B, 4)])
    _collapse(pane, False)
    _assert_inside_plot(pane)


def test_collapsed_state_survives_a_replot(app):
    """The card belongs to the pane, not the figure, so folding it once holds
    for the sweep after it."""
    pane = _pane()
    _draw(pane, [(STEM_A, 1)])
    _collapse(pane)

    _draw(pane, [(STEM_B, 3)])
    assert pane._collapse_button.isChecked()
    assert not pane._overlay.isVisible()


def test_the_eraser_holds_the_card_open(app):
    """Its button is the only way out of the window lock, so it cannot be
    folded away behind a chevron."""
    pane = _pane(with_eraser=True)
    _draw(pane, [(STEM_A, 1)])
    _collapse(pane)

    pane.set_eraser_enabled(True)
    assert not pane._collapse_button.isChecked(), "eraser armed behind a folded card"
    assert not pane._collapse_button.isEnabled()
    assert pane._overlay.isVisible()

    pane.set_eraser_enabled(False)
    assert pane._collapse_button.isEnabled()


def test_a_pane_without_the_eraser_still_folds(app):
    pane = _pane(with_eraser=False)
    _draw(pane, [(STEM_A, 1)])
    assert pane._eraser_button is None
    _collapse(pane)
    assert not pane._overlay.isVisible()


def test_a_message_hides_the_chevron_too(app):
    """Nothing to fold when there is no plot."""
    pane = _pane()
    _draw(pane, [(STEM_A, 1)])
    pane.set_message("No sweep selected")
    QApplication.processEvents()
    assert not pane._collapse_button.isVisible()
    assert not pane._overlay.isVisible()
