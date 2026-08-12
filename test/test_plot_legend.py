"""The outside legend: file names become collapsible group headers, long ones
elide rather than squeezing the axes, and the column scrolls when it overflows.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetricsF
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.plotting import _elide_entry, _GroupHeader, build_nyquist_plot

# Two batch exports sharing a long prefix and differing near the end -- the case that makes plain right-eliding useless.
STEM_A = "LGM50LT EIS CALIBRATION_C01"
STEM_B = "LGM50LT EIS CALIBRATION_C01_part2_DRIFT_REP_BAND_01_MB_C01"

f = np.logspace(5, -1, 30)
w = 2 * np.pi * f
Z = 10.0 + 50.0 / (1 + 1j * w * 5e-3)


class _Click:
    """Stand-in for the scene's click event; the header reads only these two."""

    def __init__(self):
        self.accepted = False

    def button(self):
        return Qt.MouseButton.LeftButton

    def accept(self):
        self.accepted = True


class _Wheel:
    def __init__(self, notches=-1):
        self._delta = notches * 120
        self.accepted = None

    def delta(self):
        return self._delta

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _plot(app, counts, width=1000, height=600, **kwargs):
    datasets = [
        EISDataset(DataSet(frequencies=f, impedances=Z), index=i,
                   source_file=stem, file_id=file_id)
        for file_id, (stem, n) in enumerate(counts)
        for i in range(n)
    ]
    widget = build_nyquist_plot(datasets, **kwargs)
    widget.resize(width, height)
    widget.show()
    QApplication.processEvents()
    return widget


def _parts(widget):
    legend = widget.getPlotItem().legend
    return legend, legend.parentItem()


@pytest.fixture
def two_files(app):
    return _plot(app, [(STEM_A, 8), (STEM_B, 2)])


# -- grouping -------------------------------------------------------------
def test_each_file_gets_one_header(two_files):
    legend, _ = _parts(two_files)
    headers = [g.header for g in legend._groups if g.header is not None]
    assert [h.title for h in headers] == [STEM_A, STEM_B]


def test_entries_under_a_header_are_bare_set_numbers(two_files):
    legend, _ = _parts(two_files)
    first = legend._groups[0]
    assert [label._shown_text for _, label in first.entries] == [
        f"Set {i:02d}" for i in range(1, 9)
    ]


def test_removed_sorts_below_every_file_and_belongs_to_none(app):
    datasets = [
        EISDataset(DataSet(frequencies=f, impedances=Z), index=i,
                   source_file=stem, file_id=file_id)
        for file_id, (stem, n) in enumerate([(STEM_A, 2), (STEM_B, 1)])
        for i in range(n)
    ]
    # Nothing masked means no "Removed" entry to place, so mask a point.
    datasets[0].data.set_mask({3: True})
    widget = build_nyquist_plot(datasets, show_removed=True)
    widget.resize(1000, 600)
    widget.show()
    QApplication.processEvents()

    legend, _ = _parts(widget)
    assert [g.header.title for g in legend._groups if g.header] == [STEM_A, STEM_B]
    tail = legend._groups[-1]
    assert tail.header is None
    assert [label._shown_text for _, label in tail.entries] == ["Removed"]


def test_a_single_file_gets_no_header(app):
    legend, _ = _parts(_plot(app, [(STEM_A, 3)]))
    assert all(g.header is None for g in legend._groups)
    assert [label._shown_text for _, label in legend.items] == [
        "Set 01", "Set 02", "Set 03"
    ]


# -- collapsing -----------------------------------------------------------
def test_clicking_a_header_folds_only_the_legend_rows(two_files):
    legend, viewport = _parts(two_files)
    group = legend._groups[0]
    curves = [sample.item for sample, _ in group.entries]
    tall = legend.geometry().height()

    group.header.mouseClickEvent(_Click())
    QApplication.processEvents()

    assert group.header.collapsed
    assert all(not label.isVisible() for _, label in group.entries)
    assert legend.geometry().height() < tall
    # The curves themselves are untouched -- that stays on the swatches.
    assert all(curve.isVisible() for curve in curves)


def test_unfolding_restores_the_height_rather_than_scrolling(two_files):
    legend, viewport = _parts(two_files)
    group = legend._groups[0]
    tall = legend.geometry().height()

    group.header.mouseClickEvent(_Click())
    QApplication.processEvents()
    group.header.mouseClickEvent(_Click())
    QApplication.processEvents()

    assert not group.header.collapsed
    assert legend.geometry().height() == tall
    assert viewport._max_offset() == 0


def test_groups_start_expanded(two_files):
    legend, _ = _parts(two_files)
    assert not any(g.collapsed for g in legend._groups)


# -- eliding --------------------------------------------------------------
def test_legend_never_outgrows_its_share_of_the_width(two_files):
    legend, viewport = _parts(two_files)
    assert viewport.geometry().width() <= 1000 * legend.MAX_WIDTH_FRACTION + 1


def test_headers_elide_but_stay_distinguishable(two_files):
    legend, _ = _parts(two_files)
    shown = [g.header._shown for g in legend._groups if g.header is not None]
    assert all("…" in text for text in shown)
    # Middle-eliding keeps the tail, which is where the two names differ.
    assert len(set(shown)) == 2


def test_full_name_survives_in_the_header_tooltip(two_files):
    legend, _ = _parts(two_files)
    for group in legend._groups:
        if group.header is not None:
            assert group.header.toolTip() == group.header.title


def _headers(widget):
    legend, _ = _parts(widget)
    return [g.header._shown for g in legend._groups if g.header is not None]


def test_the_name_stops_at_the_character_cap(app):
    widget = _plot(app, [(STEM_A, 2), (STEM_B, 2)], width=2600)
    # +1 for the ellipsis Qt substitutes in.
    assert all(len(text) <= _GroupHeader.MAX_NAME_CHARS + 1 for text in _headers(widget))


# Wide enough that the character cap binds rather than the pixel budget.
ROOMY, ROOMIER = 2200, 3200


def test_a_wider_pane_does_not_lengthen_the_name(app):
    """The whole reason the cap counts characters rather than pixels: the same
    file should read the same length whatever the pane, so a laptop panel and
    an external monitor agree."""
    widget = _plot(app, [(STEM_A, 2), (STEM_B, 2)], width=ROOMY)
    capped = _headers(widget)
    widget.resize(ROOMIER, 600)
    QApplication.processEvents()
    assert _headers(widget) == capped


def test_a_narrow_pane_still_wins_over_the_cap(app):
    widget = _plot(app, [(STEM_A, 2), (STEM_B, 2)], width=ROOMY)
    capped = _headers(widget)
    widget.resize(700, 600)
    QApplication.processEvents()
    assert all(len(a) < len(b) for a, b in zip(_headers(widget), capped))


def test_widening_restores_text_it_had_elided(app):
    widget = _plot(app, [(STEM_A, 2), (STEM_B, 2)], width=ROOMY)
    capped = _headers(widget)

    widget.resize(420, 600)
    QApplication.processEvents()
    widget.resize(ROOMY, 600)
    QApplication.processEvents()

    # Re-eliding an already-elided string would have eroded it past recovery.
    assert _headers(widget) == capped


def test_separatorless_entries_elide_whole(app):
    # "Fit" and "Removed" carry no ' · Set NN' suffix to protect.
    metrics = QFontMetricsF(QFont())
    text = "Removed"
    assert _elide_entry(metrics, text, metrics.horizontalAdvance(text) + 1) == text
    assert _elide_entry(metrics, text, 8) != text


def test_a_hopeless_budget_keeps_the_set_number(app):
    metrics = QFontMetricsF(QFont())
    assert _elide_entry(metrics, f"{STEM_B} · Set 03", 1) == "Set 03"


# -- scrolling ------------------------------------------------------------
def test_overflow_scrolls_instead_of_clipping_the_plot(app):
    widget = _plot(app, [(STEM_A, 20), (STEM_B, 20)], height=400)
    legend, viewport = _parts(widget)
    assert legend.geometry().height() > viewport.geometry().height()
    assert viewport._max_offset() > 0
    # The plot itself still fits the pane -- the legend absorbs the overflow.
    assert widget.getPlotItem().geometry().height() <= widget.height() + 1


def test_scrolling_clamps_at_both_ends(app):
    widget = _plot(app, [(STEM_A, 20), (STEM_B, 20)], height=400)
    legend, viewport = _parts(widget)

    viewport.wheelEvent(_Wheel(notches=-1000))
    assert legend.pos().y() == pytest.approx(-viewport._max_offset())

    viewport.wheelEvent(_Wheel(notches=1000))
    assert legend.pos().y() == 0


def test_the_wheel_passes_through_when_there_is_nothing_to_scroll(two_files):
    _, viewport = _parts(two_files)
    event = _Wheel(notches=-1)
    viewport.wheelEvent(event)
    # Ignored, so the wheel still zooms the plot when the legend fits.
    assert event.accepted is False
