"""Auto-Scale works on the DRT panes.

The button is on every PgFigurePane, but the framing it reads is set per
builder -- and the DRT builders did not set it, so clicking Auto-Scale on a DRT
plot raised AttributeError out of PlotWidget.__getattr__ and took the window
down. These plots are also in log mode, so their range is in decades while the
series carry raw frequencies.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QApplication

from core.plotting import build_drt_peaks_plot, build_drt_plot
from gui.figure_panes import PgFigurePane

TAU = np.logspace(-5, 2, 200)
GAMMA = np.exp(-((np.log10(TAU) + 1) ** 2))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Result:
    """Stands in for a pyimpspec DRTResult: the builder only reads the curve."""

    def get_drt_data(self):
        return TAU, GAMMA


class _Peaks:
    def __init__(self, count=2):
        self._count = count

    def get_num_peaks(self):
        return self._count

    def get_time_constants(self, num_per_decade=100):
        return TAU

    def get_gammas(self, peak_indices=None, num_per_decade=100):
        # An individual peak sits under the sum, as a real deconvolution does.
        return GAMMA if peak_indices is None else GAMMA / 2


def _autoscaled(widget):
    """Click Auto-Scale on a pane showing `widget`; return the resulting view."""
    pane = PgFigurePane(with_overlay_actions=True)
    pane.set_widget(widget)
    pane._autoscale()
    (xlo, xhi), (ylo, yhi) = widget.getPlotItem().getViewBox().viewRange()
    return pane, (xlo, xhi, ylo, yhi)


def _assert_frames_the_curve(view, x_decades):
    """The view covers the curve, in the ViewBox's own units: decades across,
    raw gamma up."""
    xlo, xhi, ylo, yhi = view
    lo, hi = x_decades
    assert xlo <= lo and xhi >= hi, view
    # Snug, not zoomed miles out: 5% padding either side, so under a decade.
    assert (lo - xlo) < 1.0 and (xhi - hi) < 1.0, view
    assert ylo <= GAMMA.min() and yhi >= GAMMA.max(), view


def test_autoscale_frames_a_drt_against_frequency(app):
    """The reported crash. f = 1/(2*pi*tau), so 1e-5..1e2 s spans about
    -2.8..4.2 decades of frequency."""
    _, view = _autoscaled(build_drt_plot([("Set 01", _Result())]))
    _assert_frames_the_curve(view, (-2.8, 4.2))


def test_autoscale_frames_the_peaks_plot(app):
    _, view = _autoscaled(build_drt_peaks_plot([("Set 01", _Peaks())]))
    _assert_frames_the_curve(view, (-2.8, 4.2))


def test_a_replot_on_the_same_axis_keeps_the_view(app):
    """The other half of the range key: rebuilding the same plot -- running the
    peak fit, say -- must not snap the user's zoom back."""
    pane = PgFigurePane(with_overlay_actions=True)
    pane.set_widget(build_drt_plot([("Set 01", _Result())]))
    view_box = pane._widget.getPlotItem().getViewBox()
    view_box.setRange(xRange=(0.0, 1.0), yRange=(0.2, 0.4), padding=0)
    assert pane._locked_range == pytest.approx((0.0, 1.0, 0.2, 0.4))

    pane.set_widget(build_drt_plot([("Set 01", _Result())]))
    (xlo, xhi), (ylo, yhi) = pane._widget.getPlotItem().getViewBox().viewRange()
    assert (xlo, xhi, ylo, yhi) == pytest.approx((0.0, 1.0, 0.2, 0.4))


def test_a_peakless_result_is_not_framed(app):
    """Nothing is drawn, so there is no range to set -- and Auto-Scale must not
    fall back on a degenerate one."""
    widget = build_drt_peaks_plot([("Set 01", _Peaks(count=0))])
    assert widget.full_range is None
    pane = PgFigurePane(with_overlay_actions=True)
    pane.set_widget(widget)
    pane._autoscale()  # no-op, not a crash


def test_autoscale_on_a_figure_without_a_range_is_a_no_op(app):
    """Belt and braces for the next builder: a missing range must not raise out
    of PlotWidget's attribute forwarding."""
    pane = PgFigurePane(with_overlay_actions=True)
    pane.set_widget(pg.PlotWidget())
    pane._autoscale()
