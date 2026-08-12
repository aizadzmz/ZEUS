"""The Bode plot pans and zooms, and its two y axes move as one.

|Z| and -phase live in separate ViewBoxes with independent scales: x is linked,
but y is carried by _follow_magnitude_y.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.plotting import build_bode_plot

f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = 10 + 50 / (1 + 1j * w * 5e-3)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def plot(app):
    ds = EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file="stub")
    widget = build_bode_plot([ds])
    widget.resize(900, 500)
    widget.show()
    QApplication.processEvents()
    return widget


def _views(widget):
    return widget.getPlotItem().getViewBox(), widget.phase_view


def _ranges(widget):
    main, phase = _views(widget)
    return main.viewRange()[0], main.viewRange()[1], phase.viewRange()[1]


def test_the_magnitude_view_takes_the_mouse(plot):
    main, phase = _views(plot)
    assert main.mouseEnabled() == [True, True]
    # The phase view is driven through the magnitude one, never directly -- it sits under it and would swallow the drags.
    assert phase.mouseEnabled() == [False, False]


def test_panning_x_carries_both_curves(plot):
    main, _ = _views(plot)
    (x0, _), _, phase_before = _ranges(plot)

    main.translateBy((0.5, 0.0))
    (nx0, _), _, phase_after = _ranges(plot)

    assert nx0 == pytest.approx(x0 + 0.5)
    # x is linked, so the phase view followed; its y must not have moved.
    assert phase_after == pytest.approx(phase_before)
    assert plot.phase_view.viewRange()[0] == pytest.approx(main.viewRange()[0])


def test_zooming_y_scales_the_phase_axis_by_the_same_factor(plot):
    main, _ = _views(plot)
    _, (m0, m1), (p0, p1) = _ranges(plot)

    main.scaleBy((1.0, 0.5))  # what the wheel drives
    _, (n0, n1), (q0, q1) = _ranges(plot)

    assert (n1 - n0) == pytest.approx((m1 - m0) * 0.5)
    assert (q1 - q0) == pytest.approx((p1 - p0) * 0.5), "phase axis did not follow"
    # Both zoomed about their own centre, so the curves keep their positions.
    assert (q0 + q1) / 2 == pytest.approx((p0 + p1) / 2)


def test_panning_y_shifts_the_phase_axis_by_the_same_fraction(plot):
    main, _ = _views(plot)
    _, (m0, m1), (p0, p1) = _ranges(plot)

    main.translateBy((0.0, (m1 - m0) * 0.25))  # a quarter of a screen up
    _, (n0, n1), (q0, q1) = _ranges(plot)

    assert n0 == pytest.approx(m0 + (m1 - m0) * 0.25)
    assert q0 == pytest.approx(p0 + (p1 - p0) * 0.25), "phase axis did not follow"
    assert (q1 - q0) == pytest.approx(p1 - p0), "phase span changed on a pan"


def test_a_point_keeps_its_screen_position_under_zoom(plot):
    """The property the follow exists for: whatever the view does, a phase
    reading stays put relative to the magnitude curve beside it."""
    main, phase = _views(plot)

    def screen_fraction():
        (m0, m1) = main.viewRange()[1]
        (p0, p1) = phase.viewRange()[1]
        # Where the midpoint of the original phase span now sits, and where a magnitude value at the same starting fraction sits.
        return ((p0 + p1) / 2 - p0) / (p1 - p0), ((m0 + m1) / 2 - m0) / (m1 - m0)

    before = screen_fraction()
    main.scaleBy((1.0, 2.5))
    main.translateBy((0.0, 0.3))
    assert screen_fraction() == pytest.approx(before)


def test_the_right_click_menu_stays_off(plot):
    """Its 'View All' reframes the magnitude box alone, which is the desync
    the follow exists to prevent. Auto-Scale on the overlay is the way back."""
    main, _ = _views(plot)
    assert not main.menuEnabled()


def test_the_opening_frame_is_not_read_as_a_gesture(plot):
    """The follow is connected after both axes are framed, so building the
    plot must not have dragged the phase axis off its own data."""
    _, _, (p0, p1) = _ranges(plot)
    phases = -np.angle(Z, deg=True)
    assert p0 < phases.min() and p1 > phases.max(), (p0, p1)
