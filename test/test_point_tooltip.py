"""The point metadata box must never be cropped by the plot's own edges: it is
a fixed-size overlay hung off a data point, so a point near the right or top
edge used to push it under the legend column or the title.
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

# A long label, so the box is wide enough to be a real test of the fit.
STEM = "LGM50LT EIS CALIBRATION_C01_part2_DRIFT_REP_BAND_01"

f = np.logspace(5, -1, 25)
w = 2 * np.pi * f
Z = 10 + 50 / (1 + 1j * w * 5e-3)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pane(app):
    p = PgFigurePane(with_overlay_actions=True)
    p.resize(760, 460)
    p.show()
    QApplication.processEvents()
    ds = EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file=STEM)
    p.set_widget(build_nyquist_plot([ds]))
    QApplication.processEvents()
    QApplication.processEvents()
    return p


def _plot_area(pane):
    return pane._widget.getPlotItem().getViewBox().sceneBoundingRect()


def _card_rect(pane):
    """The box's own rect in scene coordinates, where the cropping happens."""
    item = pane._tooltip_item
    return item.mapRectToScene(item.boundingRect())


def _corners(pane):
    """View coordinates of the four corners of the plotted area, which is
    where a point is most awkward to hang a box from."""
    (x0, x1), (y0, y1) = pane._widget.getPlotItem().getViewBox().viewRange()
    inset_x, inset_y = (x1 - x0) * 0.02, (y1 - y0) * 0.02
    return {
        "bottom-left": (x0 + inset_x, y0 + inset_y),
        "bottom-right": (x1 - inset_x, y0 + inset_y),
        "top-left": (x0 + inset_x, y1 - inset_y),
        "top-right": (x1 - inset_x, y1 - inset_y),
        "centre": ((x0 + x1) / 2, (y0 + y1) / 2),
    }


def _show(pane, x, y, text="Set: 01\nFreq: 63.1 Hz\n|Z|: 0.07435 Ω\n-phase: -60.66°"):
    pane._show_tooltip(x, y, text, pinned=True)
    QApplication.processEvents()


def test_the_box_stays_inside_the_plot_at_every_corner(pane):
    area = _plot_area(pane)
    for where, (x, y) in _corners(pane).items():
        _show(pane, x, y)
        card = _card_rect(pane)
        assert card.right() <= area.right() + 1, (where, card, area)
        assert card.left() >= area.left() - 1, (where, card, area)
        assert card.top() >= area.top() - 1, (where, card, area)
        assert card.bottom() <= area.bottom() + 1, (where, card, area)


def test_the_box_opens_left_of_a_point_near_the_right_edge(pane):
    """Flipped rather than shoved: it should still touch its own point."""
    corners = _corners(pane)
    _show(pane, *corners["bottom-right"])
    at = pane._widget.getPlotItem().getViewBox().mapViewToScene(
        QPointF(*corners["bottom-right"])
    )
    card = _card_rect(pane)
    assert card.right() <= at.x() + 1, "box still opens rightwards off the edge"


def test_the_box_opens_right_of_a_point_near_the_left_edge(pane):
    corners = _corners(pane)
    _show(pane, *corners["bottom-left"])
    card = _card_rect(pane)
    area = _plot_area(pane)
    assert card.left() >= area.left() - 1
    assert card.width() > 0


def test_a_box_too_wide_to_flip_is_pushed_back_in(pane):
    """No room either side of the point -- the tie to the point gives way
    before the box is allowed to be cropped."""
    area = _plot_area(pane)
    huge = "\n".join("x" * 90 for _ in range(3))
    centre = _corners(pane)["centre"]
    _show(pane, *centre, text=huge)

    card = _card_rect(pane)
    assert card.width() > area.width() / 2, "not actually a hard case"
    assert card.left() >= area.left() - 1, (card, area)
    assert card.right() <= area.right() + 1, (card, area)


def test_panning_re_fits_a_pinned_box(pane):
    """The Bode and Nyquist views move now, and a pinned box has to keep up."""
    area = _plot_area(pane)
    _show(pane, *_corners(pane)["centre"])
    view = pane._widget.getPlotItem().getViewBox()

    (x0, x1), _ = view.viewRange()
    view.translateBy((-(x1 - x0) * 0.45, 0.0))  # slide the point to the edge
    QApplication.processEvents()

    card = _card_rect(pane)
    assert card.right() <= area.right() + 1, (card, area)
    assert card.left() >= area.left() - 1, (card, area)


def test_hiding_forgets_the_anchor(pane):
    _show(pane, *_corners(pane)["centre"])
    assert pane._tooltip_anchor is not None
    pane._hide_tooltip()
    assert pane._tooltip_item is None and pane._tooltip_anchor is None
    # A stray re-fit after hiding must not raise.
    pane._fit_tooltip_inside_plot()
