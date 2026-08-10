"""Removed (masked-out) points: where they are drawn, and what the one
"Removed" legend entry hides.

Two reported faults. On the Bode plot the grey × markers only ever reached the
|Z| series, so the -Φ series ran unbroken through gaps the magnitude series
showed. And the legend entry stood for the first sweep's markers alone, so
clicking it left every other sweep's on the plot.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.plotting import build_bode_plot, build_nyquist_plot

STEM_A = "LGM50LT EIS CALIBRATION_C01"
STEM_B = "LGM50LT EIS CALIBRATION_C01_part2"

f = np.logspace(5, -1, 30)
w = 2 * np.pi * f
Z = 10.0 + 50.0 / (1 + 1j * w * 5e-3)

MASKED = {3: True, 11: True}


class _Click:
    """Stand-in for the scene's click event; the swatch reads only these."""

    def __init__(self):
        self.accepted = False

    def button(self):
        return Qt.MouseButton.LeftButton

    def accept(self):
        self.accepted = True


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _datasets(counts=((STEM_A, 2), (STEM_B, 1))):
    datasets = [
        EISDataset(DataSet(frequencies=f, impedances=Z), index=i,
                   source_file=stem, file_id=file_id)
        for file_id, (stem, n) in enumerate(counts)
        for i in range(n)
    ]
    # Every sweep masked, so a legend click has more than one series to reach.
    for ds in datasets:
        ds.data.set_mask(dict(MASKED))
    return datasets


def _shown(widget, **kwargs):
    widget.resize(1000, 600)
    widget.show()
    QApplication.processEvents()
    return widget


def _items(container, role):
    """Series of one role in a PlotItem or a bare ViewBox. A ViewBox keeps only
    its ranged items in addedItems, so the fits (added ignoreBounds) have to be
    read off the child group instead."""
    items = getattr(container, "items", None)
    if items is None:
        items = container.allChildren()
    return [it for it in items if getattr(it, "_eis_role", None) == role]


def _removed_series(container):
    return _items(container, "removed")


def _legend_sample(widget, name):
    for sample, label in widget.getPlotItem().legend.items:
        if label._full_text == name:
            return sample
    raise AssertionError(f"no '{name}' legend entry")


# -- issue 1: the phase series was never marked -------------------------------
def test_bode_marks_removed_points_on_both_series(app):
    datasets = _datasets()
    widget = _shown(build_bode_plot(datasets, show_removed=True))
    plot_item = widget.getPlotItem()

    magnitude = _removed_series(plot_item)
    phase = _removed_series(widget.phase_view)
    assert len(magnitude) == len(datasets)
    assert len(phase) == len(datasets), "-Φ series left unmarked"

    # Same frequencies on both, at the phase of the masked points.
    xr = np.log10(f[sorted(MASKED)])
    expected_phase = -np.angle(Z[sorted(MASKED)], deg=True)
    for scatter in phase:
        pos = np.array([(p.pos().x(), p.pos().y()) for p in scatter.points()])
        assert np.allclose(pos[:, 0], xr)
        assert np.allclose(pos[:, 1], expected_phase)


def test_bode_phase_markers_carry_the_same_tooltip_data(app):
    """Hovering either × describes the whole point -- and the eraser reads the
    key/index off it, so clicking a phase × restores that point too."""
    widget = _shown(build_bode_plot(_datasets([(STEM_A, 1)]), show_removed=True))
    magnitude = _removed_series(widget.getPlotItem())[0]
    phase = _removed_series(widget.phase_view)[0]

    for m, p in zip(magnitude.points(), phase.points()):
        assert m.data() == p.data()
        assert m.data()["removed"] is True


def test_bode_frames_the_phase_axis_around_the_removed_markers(app):
    """A × drawn outside the opening view is no better than one not drawn."""
    datasets = _datasets([(STEM_A, 1)])
    widget = _shown(build_bode_plot(datasets, show_removed=True))
    lo, hi = widget.phase_view.viewRange()[1]

    for point in _removed_series(widget.phase_view)[0].points():
        assert lo <= point.pos().y() <= hi


# -- issue 2: one entry, one sweep hidden -------------------------------------
@pytest.mark.parametrize("build", [build_nyquist_plot, build_bode_plot])
def test_the_removed_legend_entry_hides_every_sweeps_markers(app, build):
    datasets = _datasets()
    widget = _shown(build(datasets, show_removed=True))
    series = _removed_series(widget.getPlotItem())
    series += _removed_series(getattr(widget, "phase_view", None) or widget.getPlotItem())
    assert len(series) > 1, "not a multi-series case, so the test is vacuous"

    sample = _legend_sample(widget, "Removed")
    sample.mouseClickEvent(_Click())
    QApplication.processEvents()
    assert not any(s.isVisible() for s in series), "some sweeps' × markers left on"

    sample.mouseClickEvent(_Click())
    QApplication.processEvents()
    assert all(s.isVisible() for s in series)


def test_the_fit_legend_entry_hides_every_fitted_curve(app):
    """Same shared-entry fault, same fix: on Bode a fit is two curves per sweep
    (|Z| and -Φ), in separate ViewBoxes."""
    datasets = _datasets([(STEM_A, 2)])
    fits = {ds.key: (f, Z) for ds in datasets}
    widget = _shown(build_bode_plot(datasets, show_removed=True, fit_curves=fits))
    curves = [
        it
        for container in (widget.getPlotItem(), widget.phase_view)
        for it in _items(container, "fit")
    ]
    assert len(curves) == 2 * len(datasets)

    sample = _legend_sample(widget, "Fit")
    sample.mouseClickEvent(_Click())
    QApplication.processEvents()
    assert not any(c.isVisible() for c in curves)


def test_hiding_removed_points_leaves_the_sweeps_alone(app):
    """The entry covers the × markers and nothing else."""
    datasets = _datasets()
    widget = _shown(build_nyquist_plot(datasets, show_removed=True))
    kept = _items(widget.getPlotItem(), "kept")

    _legend_sample(widget, "Removed").mouseClickEvent(_Click())
    QApplication.processEvents()
    assert all(k.isVisible() for k in kept)
