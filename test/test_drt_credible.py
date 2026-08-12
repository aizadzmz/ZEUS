"""The Bayesian DRT run's credible intervals reach the plot.

The band is filled between two plotted curves, which makes the log x axis the
thing to watch: a FillBetweenItem tracks its curves in plotted coordinates, so
edges that never learned the axis is logarithmic would place the fill several
decades away from the distribution it belongs to.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QApplication

from core.plotting import CREDIBLE_INTERVAL_NAME, build_drt_plot, credible_intervals

TAU = np.logspace(-5, 2, 200)
GAMMA = 40 * np.exp(-((np.log10(TAU) + 1) ** 2))
FREQ = 1.0 / (2 * np.pi * TAU)


class Result:
    """A TRRBFResult's two accessors. Empty credible-interval arrays are how
    pyimpspec reports a run that did not sample, so that is the default."""

    def __init__(self, gamma=GAMMA, mean=None, lower=None, upper=None):
        self._gamma = gamma
        empty = np.array([])
        self._mean = empty if mean is None else mean
        self._lower = empty if lower is None else lower
        self._upper = empty if upper is None else upper

    def get_drt_data(self):
        return TAU, self._gamma

    def get_drt_credible_intervals_data(self):
        if len(self._mean) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        return TAU, self._mean, self._lower, self._upper


def bayesian(spread=8.0):
    """A result shaped like a Bayesian run's: a posterior mean offset from the
    MAP curve, inside a band wider than either."""
    mean = GAMMA * 0.95
    return Result(mean=mean, lower=mean - spread, upper=mean + spread)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _fills(widget):
    return [i for i in widget.getPlotItem().items if isinstance(i, pg.FillBetweenItem)]


def test_a_simple_run_has_no_intervals_to_draw(app):
    assert credible_intervals(Result()) is None
    widget = build_drt_plot([("Set 01", Result())])
    assert _fills(widget) == []


def test_a_result_without_the_accessor_is_not_an_error(app):
    class Bare:
        def get_drt_data(self):
            return TAU, GAMMA

    assert credible_intervals(Bare()) is None
    assert _fills(build_drt_plot([("Set 01", Bare())])) == []


def test_the_bayesian_band_and_mean_are_drawn(app):
    widget = build_drt_plot([("Set 01", bayesian())])

    assert len(_fills(widget)) == 1
    names = [
        entry.text for _, entry in widget.getPlotItem().legend.items
    ]
    assert "Set 01" in names
    assert f"Set 01 ({CREDIBLE_INTERVAL_NAME})" in names


def test_the_band_lands_on_the_curve_not_decades_away(app):
    """The regression this file exists for: the fill is built from its curves'
    plotted coordinates, and this axis is in decades."""
    widget = build_drt_plot([("Set 01", bayesian())])
    box = _fills(widget)[0].path().boundingRect()

    decades = np.log10(FREQ)
    assert box.left() == pytest.approx(decades.min(), abs=0.01)
    assert box.right() == pytest.approx(decades.max(), abs=0.01)


def test_the_band_sits_behind_the_curves(app):
    widget = build_drt_plot([("Set 01", bayesian())])
    curves = widget.getPlotItem().listDataItems()

    assert _fills(widget)[0].zValue() < min(c.zValue() for c in curves)


def test_the_framing_covers_the_whole_band(app):
    """Auto-Scale reads this range; cropping the uncertainty would defeat
    drawing it. The band is deliberately wider than the distribution here."""
    _, _, y_min, y_max = build_drt_plot([("Set 01", bayesian())]).full_range

    assert y_max >= GAMMA.max() * 0.95 + 8.0
    assert y_min <= GAMMA.min() * 0.95 - 8.0


def test_each_sweep_keeps_its_own_colour(app):
    """Two Bayesian sweeps: each band is tinted like the curve it belongs to,
    which is all that distinguishes them once they overlap."""
    widget = build_drt_plot([("Set 01", bayesian()), ("Set 02", bayesian(4.0))])
    fills = _fills(widget)

    assert len(fills) == 2
    first, second = (f.brush().color() for f in fills)
    assert (first.red(), first.green(), first.blue()) != (
        second.red(), second.green(), second.blue()
    )
    # Translucent, so an overlap still shows both.
    assert first.alpha() < 255 and second.alpha() < 255
