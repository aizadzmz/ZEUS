"""PyQtGraph plot builders for the desktop GUI: Nyquist, Bode, Residuals, and
DRT."""
from __future__ import annotations

import math
from html import escape
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt
from PySide6.QtGui import QFont, QFontMetricsF, QPolygonF
from PySide6.QtWidgets import QSizePolicy

# Annotation-only: core.io_utils costs ~4 s to import (pyimpspec -> scipy,
# sympy), and keeping it off the runtime path lets the GUI build figure panes
# without loading the analysis stack.
if TYPE_CHECKING:
    from core.io_utils import EISDataset

# Gold crosshair marking Z'=0 and -Z''=0 on Nyquist plots, drawn heavier than
# the grid so the origin reads as a reference rather than a gridline.
ORIGIN_COLOR = "#CC9D33"
ORIGIN_WIDTH = 2.0

# The app's color cycle (matplotlib's tab10), with the leading blue swapped for
# the accent so a single-sweep plot is on-brand. The rest stay categorical --
# distinguishing sweeps beats matching the palette. tab10's grey is dropped:
# it sat too close to _REMOVED_COLOR to tell a live sweep from a masked point.
SERIES_COLORS = (
    "#2b3f9e", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf",
)

_REMOVED_COLOR = "#708090"  # slate: reads as discarded against every series


def _with_alpha(color: str, alpha: int) -> pg.QtGui.QColor:
    """One of the palette colours, translucent. Used where a series has to sit
    behind another and still let it through."""
    shaded = pg.mkColor(color)
    shaded.setAlpha(alpha)
    return shaded


# What the Bayesian DRT run's extra output is called in a legend. pyimpspec
# takes the 0.5% and 99.5% quantiles of its HMC samples, so the band spans the
# central 99% of the posterior -- upstream's own plots call that "3σ", which is
# the Gaussian approximation of it rather than what the code computes.
CREDIBLE_INTERVAL_NAME = "mean, 99% CI"

# Residual series names, keyed by core.validation.RESIDUAL_MODES. The keys are
# spelled out rather than imported: core.validation pulls in pyimpspec, and
# this module is imported to build figure panes long before the analysis stack
# is wanted (see the TYPE_CHECKING note above). test_validation pins the two
# lists together so a new mode cannot land here unnamed.
_RESIDUAL_SERIES_NAMES = {
    "modulus": ("ΔZ' / |Z|", "ΔZ'' / |Z|"),
    "component": ("ΔZ' / Z'", "ΔZ'' / Z''"),
}

# Residual axis framing, in percentage points above the highest limit line.
#
# The figure exists to show which points cleared the limit, so the limit is
# what the axis is built around: a residual well past it only has to read as
# "well past", and letting one set the scale flattens everything else onto the
# zero line -- the uncapped component convention reports 1e17% for a part that
# rounds to zero. Anything beyond the edge is pinned there and marked.
#
# Fixed rather than fitted to the data, so the scale does not jump as the pager
# steps between sweeps.
_AXIS_HEADROOM_PERCENT = 5.0

# Fallback for a figure drawn with no limit lines at all, where there is no
# limit to take headroom from. At 100% the error is the size of the quantity it
# is measured against, so the value past that says only "not resolved".
_MAX_RESIDUAL_AXIS_PERCENT = 100.0

# Which Bode series belongs to which axis. Sits under the title rather than in
# the legend, which is per-sweep.
BODE_SUBTITLE = "|Z| ●   ,   -Φ ○"

# Title-row height in px for the two-line Bode title (PlotItem.setTitle sizes
# for one line). Raise it if the subtitle wraps or the font grows.
BODE_TITLE_HEIGHT = 48

# The marker shapes a file can be given, in menu order. All filled: the Bode
# plot spends the filled/hollow distinction on telling |Z| from -Φ (see
# BODE_SUBTITLE), so offering a hollow marker would collide with it. "x" and
# "+" are excluded too -- "x" marks removed points (see _add_removed_series)
# and "+" is too near it to tell apart.
#
# (label, pyqtgraph symbol)
MARKER_SHAPES: Tuple[Tuple[str, str], ...] = (
    ("Circle", "o"),
    ("Square", "s"),
    ("Triangle", "t1"),
    ("Triangle down", "t"),
    ("Triangle right", "t2"),
    ("Triangle left", "t3"),
    ("Diamond", "d"),
    ("Pentagon", "p"),
    ("Hexagon", "h"),
    ("Star", "star"),
)

DEFAULT_MARKER = "o"
DEFAULT_MARKER_SIZE = 6.0
DEFAULT_LINE_WIDTH = 1.5

# The default shape per loaded file, by the file's position: sweeps that share
# a color because they share an index within their own file still read apart.
# Overridden per file from the Marker & line style dialog.
PG_MARKERS: Tuple[str, ...] = tuple(symbol for _, symbol in MARKER_SHAPES)


def default_marker_for(position: int) -> str:
    """The shape a file at `position` in the load order gets when it has not
    been given one explicitly."""
    return PG_MARKERS[position % len(PG_MARKERS)]


# ds.key -> (color, pyqtgraph symbol), built by the GUI so every step draws a
# given sweep the same way. See gui/main_window.py _build_style_map.
StyleMap = Dict[str, Tuple[str, str]]


def _series_style(
    style_map: Optional[StyleMap], key: str, index: int
) -> Tuple[str, Optional[str]]:
    """One sweep's (color, symbol), falling back to the color cycle when the
    caller supplied no style map -- scripts and tests do not build one."""
    if style_map is not None and key in style_map:
        return style_map[key]
    return SERIES_COLORS[index % len(SERIES_COLORS)], None


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def equal_aspect_limits(
    xmin: float, xmax: float, ymin: float, ymax: float, *, include_origin: bool = True
) -> Tuple[float, float, float, float]:
    """Pad (xmin, xmax, ymin, ymax) so both axes span the same range.
    include_origin folds 0 in so it is never cropped."""
    if include_origin:
        xmin, xmax = min(xmin, 0), max(xmax, 0)
        ymin, ymax = min(ymin, 0), max(ymax, 0)

    # Pad the narrower axis so both spans match, keeping the scale equal.
    span = max(xmax - xmin, ymax - ymin)
    x_pad = (span - (xmax - xmin)) / 2
    y_pad = (span - (ymax - ymin)) / 2
    return xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad


def _engineering_exponent(max_abs: float) -> int:
    """Round down to the nearest multiple of 3 so the scaled mantissa lands in
    1-999. Returns 0 below 1000."""
    if max_abs < 1000 or not math.isfinite(max_abs) or max_abs <= 0:
        return 0
    return (math.floor(math.log10(max_abs)) // 3) * 3


def _close_plot_box(plot_item) -> None:
    """Draw the top and right borders, so a plot reads as a closed box rather
    than an L of two axes.

    The extra axes carry no ticks or label -- they exist only for their line.
    The margins give that line somewhere to sit: a bare AxisItem has ~0
    height, so without them it lands hard against the widget edge and clips.
    """
    plot_item.layout.setContentsMargins(1, 6, 12, 12)  # left, top, right, bottom
    for side in ("top", "right"):
        plot_item.showAxis(side)
        border_axis = plot_item.getAxis(side)
        border_axis.setStyle(showValues=False, tickLength=0)
        border_axis.setLabel(None)


def _hide_plot_options_menu(plot_item) -> None:
    """Drop the "Plot Options" submenu from a plot's right-click menu.

    The PlotItem's own menu -- transforms, downsampling, averaging, alpha --
    assumes a plain plot whose items it owns, and crashes on the plots built
    here (custom axis items, a second ViewBox, items added straight to the
    scene). The ViewBox's own entries, View All and the axis/mouse-mode
    controls, are unaffected: the scene skips a PlotItem whose menu is
    disabled when it collects parent menus.
    """
    # enableViewBoxMenu=None: leave the ViewBox menu exactly as it is, rather
    # than the 'same' default, which would take it down along with this one.
    plot_item.setMenuEnabled(False, None)


def _axis_label(base: str, exponent: int) -> str:
    if exponent == 0:
        return f"{base} [Ω]"
    return f"{base} [Ω × 10{str(exponent).translate(_SUPERSCRIPT)}]"


class _ScaledAxisItem(pg.AxisItem):
    """An AxisItem whose tick numbers are pre-divided by 10**exponent, matching
    the '× 10^n' scale in the axis label."""

    def __init__(self, *args, exponent: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._exponent = exponent
        # Required: otherwise AxisItem appends its own "(x...)" scale factor
        # under range 1, which duplicates or contradicts the label.
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        factor = 10 ** self._exponent
        return [f"{v / factor:.4g}" for v in values]


def _max_abs_extent(datasets: List[EISDataset], show_removed: bool) -> float:
    """Largest |Z'| or |-Z''| across every point that will be drawn, used to
    pick a single shared engineering-notation exponent for both axes."""
    max_abs = 0.0
    for ds in datasets:
        Z = ds.impedances
        if Z.size:
            max_abs = max(max_abs, float(abs(Z.real).max()), float(abs(Z.imag).max()))
        if show_removed:
            Zr = ds.data.get_impedances(masked=True)
            if Zr.size:
                max_abs = max(max_abs, float(abs(Zr.real).max()), float(abs(Zr.imag).max()))
    return max_abs


def point_tip(data) -> str:
    """The metadata box shown for one plotted point, built from the _point_data
    payload rather than the plotted position.

    `note` qualifies the set line: "removed" on a masked point, "off scale" on
    a residual pinned to the axis edge. Spelled out by the payload rather than
    derived here, so a plot can say what its own outliers mean."""
    note = data.get("note") or ("removed" if data.get("removed") else "")
    suffix = f" ({note})" if note else ""
    return (
        f"Set: {data['label']}{suffix}\n"
        f"Freq: {data['freq']:.4g} Hz\n"
        f"{data['values']}"
    )


def _nyquist_tip_values(z) -> str:
    return f"Z': {z.real:.4g} Ω\nZ'': {z.imag:.4g} Ω"


def _bode_tip_values(z) -> str:
    return f"|Z|: {abs(z):.4g} Ω\n-phase: {-math.degrees(math.atan2(z.imag, z.real)):.4g}°"


def _point_data(
    key: str,
    label: str,
    freq,
    indices,
    impedances,
    *,
    removed: bool = False,
    values=_nyquist_tip_values,
):
    """Per-point payload on every ScatterPlotItem: dataset key, display label,
    and the point's index in the mask-inclusive array."""
    return [
        {
            "key": key,
            "label": label,
            "freq": f,
            "index": int(i),
            "removed": removed,
            "values": values(z),
        }
        for f, i, z in zip(freq, indices, impedances)
    ]


def _split_indices(ds: EISDataset):
    """(kept, removed) point indices for a dataset, in the same order as
    get_impedances(masked=False) / get_impedances(masked=True) return them."""
    mask = ds.data.get_mask()
    kept, removed = [], []
    for i in range(ds.data.get_num_points(masked=None)):
        (removed if mask.get(i, False) else kept).append(i)
    return kept, removed


def _marker_kwargs(
    color: str,
    symbol: Optional[str],
    hollow: bool,
    size: float = DEFAULT_MARKER_SIZE,
) -> dict:
    """ScatterPlotItem styling for one series: filled in the dataset's color,
    or outlined when hollow. symbol=None leaves pyqtgraph's default ('o')."""
    if hollow:
        # The outline scales with the marker: a 1.2px ring around an 18px
        # marker reads as a thin grey circle rather than the sweep's color.
        kwargs = dict(brush=None, pen=pg.mkPen(color, width=max(1.0, size / 5.0)), size=size)
    else:
        kwargs = dict(brush=pg.mkBrush(color), pen=None, size=size)
    if symbol is not None:
        kwargs["symbol"] = symbol
    return kwargs


def _add_kept_series(
    container,
    x,
    y,
    tip_data,
    name: Optional[str],
    color: str,
    style: str,
    symbol: Optional[str] = None,
    hollow: bool = False,
    marker_size: float = DEFAULT_MARKER_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    dashed: bool = False,
):
    """Draw one dataset's kept points and return the hoverable ScatterPlotItem.

    `hollow` outlines the markers; `dashed` dashes the line. Bode passes both
    for its phase series, so it stays distinct from the magnitude series
    whichever fill Plot Options is set to.
    """
    if style == "line":
        # The line carries the legend entry, so the swatch shows a colored
        # line rather than a marker; the scatter stays unnamed.
        pen = pg.mkPen(
            color, width=line_width, style=Qt.DashLine if dashed else Qt.SolidLine
        )
        container.addItem(pg.PlotDataItem(x=x, y=y, pen=pen, name=name))
        scatter_name = None
        if symbol is not None:
            marker_kwargs = _marker_kwargs(color, symbol, hollow, marker_size)
        else:
            marker_kwargs = dict(brush=None, pen=None, size=marker_size)
    elif style == "scatter":
        scatter_name = name
        marker_kwargs = _marker_kwargs(color, symbol, hollow, marker_size)
    else:
        raise ValueError(f"Unknown style '{style}'; expected 'scatter' or 'line'.")

    scatter = pg.ScatterPlotItem(
        x=x, y=y,
        name=scatter_name,
        hoverable=True,
        data=tip_data,
        **marker_kwargs,
    )
    scatter._eis_role = "kept"
    container.addItem(scatter)
    return scatter


def _add_removed_series(container, x, y, tip_data, size: float = DEFAULT_MARKER_SIZE):
    """Draw masked-out points as muted 'x' markers in `container` (a PlotItem or
    a bare ViewBox, as the Bode phase axis is). Returns the ScatterPlotItem, or
    None if there are no removed points.

    Sized with the live markers so the two stay comparable -- a fixed 6px 'x'
    disappears under 16px sweep markers."""
    if x.size == 0:
        return None

    scatter = pg.ScatterPlotItem(
        x=x, y=y,
        symbol="x",
        pen=pg.mkPen(_REMOVED_COLOR, width=max(1.0, size / 4.6)),
        brush=None,
        size=size,
        hoverable=True,
        data=tip_data,
    )
    scatter._eis_role = "removed"
    container.addItem(scatter)
    return scatter


def _add_fit_series(container, x, y, color: str, line_width: float = DEFAULT_LINE_WIDTH):
    """Draw a dataset's fitted circuit response as a dashed line. Never hit-
    tested, and excluded from the range calculation."""
    item = pg.PlotDataItem(
        x=x, y=y,
        # A shade heavier than the data line it is drawn over, as before.
        pen=pg.mkPen(color, width=line_width * 1.07, style=Qt.DashLine),
    )
    item._eis_role = "fit"
    container.addItem(item, ignoreBounds=True)
    return item


def _bounds(
    xs: List[float],
    ys: List[float],
    *,
    include_origin: bool = True,
    equal_aspect: bool = True,
) -> Optional[Tuple[float, float, float, float]]:
    """Framing for the given points, or None when there are none (callers fall
    back rather than setting a degenerate zero-span range)."""
    if not xs:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Breathing room so a point at an extreme isn't sliced by the ViewBox edge
    # (padding=0 below draws the range exactly).
    margin = 0.05
    x_pad = (xmax - xmin) * margin or 1e-6
    y_pad = (ymax - ymin) * margin or 1e-6
    xlo, xhi = xmin - x_pad, xmax + x_pad
    ylo, yhi = ymin - y_pad, ymax + y_pad
    if not equal_aspect:
        return xlo, xhi, ylo, yhi
    return equal_aspect_limits(xlo, xhi, ylo, yhi, include_origin=include_origin)


# Separator between the file name and the sweep label in a legend entry --
# mirrors EISDataset.qualified_label, which core.plotting can't import at
# runtime (see the TYPE_CHECKING note above). Eliding splits on the *last*
# occurrence, so a file name containing the separator stays harmless.
_LABEL_SEPARATOR = " · "


class _GroupHeader(pg.GraphicsWidget):
    """A file name above its sweeps, with a chevron that folds the group's rows
    away. Painted directly rather than assembled from a LabelItem so that the
    chevron, the elided name and the click target are one item."""

    HEIGHT = 19.0
    CHEVRON_BOX = 12.0  # square the arrow is drawn inside, left of the text
    CHEVRON_SIZE = 4.0  # half-width of the arrow itself
    # Roughly how many characters of the file name to show. Measured in
    # characters rather than pixels so the header reads the same length on
    # every display: a pixel budget is a share of the pane's logical width,
    # which differs between a scaled laptop panel and an external monitor,
    # and the point size renders at a different pixel height on each. Both
    # cancel out here, because the width comes from the same font metrics
    # that draw the text.
    MAX_NAME_CHARS = 26

    def __init__(self, title: str, font: QFont, color, on_toggle):
        super().__init__()
        self.title = title
        self.collapsed = False
        self._shown = title
        self._font = QFont(font)
        self._font.setBold(True)
        self._color = pg.mkColor(color)
        self._on_toggle = on_toggle
        self.setToolTip(title)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self.HEIGHT)
        self.setMaximumHeight(self.HEIGHT)

    def sizeHint(self, which, constraint=QSizeF()):
        width = self.CHEVRON_BOX + QFontMetricsF(self._font).horizontalAdvance(self._shown)
        return QSizeF(width, self.HEIGHT)

    def elide_to(self, budget: float) -> None:
        """Fit the name into MAX_NAME_CHARS, or into `budget` px when the pane
        is too narrow even for that. The whole name is one unit here -- there
        is no set number to protect, so it middle-elides entire."""
        metrics = QFontMetricsF(self._font)
        room = min(
            budget - self.CHEVRON_BOX,
            metrics.averageCharWidth() * self.MAX_NAME_CHARS,
        )
        shown = (
            self.title if room <= 0
            else metrics.elidedText(self.title, Qt.TextElideMode.ElideMiddle, room)
        )
        if shown != self._shown:
            self._shown = shown
            self.updateGeometry()
            self.update()

    def paint(self, painter, *args):
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        rect = self.boundingRect()
        middle = rect.height() / 2.0
        size = self.CHEVRON_SIZE
        centre = self.CHEVRON_BOX / 2.0
        # Right-pointing when folded, down-pointing when open -- the same
        # convention as a file tree.
        if self.collapsed:
            points = [(centre - size / 2, middle - size), (centre - size / 2, middle + size),
                      (centre + size, middle)]
        else:
            points = [(centre - size, middle - size / 2), (centre + size, middle - size / 2),
                      (centre, middle + size)]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(pg.mkBrush(self._color))
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))

        painter.setFont(self._font)
        painter.setPen(pg.mkPen(self._color))
        painter.drawText(
            QRectF(self.CHEVRON_BOX, 0, rect.width() - self.CHEVRON_BOX, rect.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._shown,
        )

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.collapsed = not self.collapsed
            self.update()
            self._on_toggle()
        event.accept()


class _LegendGroup:
    """One file's entries, plus the header that folds them. `header` is None for
    the trailing ungrouped entries ("Fit", "Removed"), which belong to no file."""

    def __init__(self, key, header: Optional[_GroupHeader]):
        self.key = key
        self.header = header
        self.entries: List[tuple] = []

    @property
    def collapsed(self) -> bool:
        return self.header is not None and self.header.collapsed


class _LinkedSample(pg.ItemSample):
    """A legend swatch that shows and hides a whole family of plot items.

    "Removed" and "Fit" are single legend entries standing for one series per
    sweep -- and on the Bode plot, one per ViewBox on top of that. pyqtgraph's
    swatch toggles only the item it was built from, so clicking it hid one
    sweep's markers and left every other sweep's on the plot."""

    def __init__(self, items):
        # The first item is the one pyqtgraph paints the swatch from; the rest
        # follow its visibility.
        super().__init__(items[0])
        self._linked = list(items[1:])

    def mouseClickEvent(self, event):
        super().mouseClickEvent(event)
        # Read back rather than inverted a second time: the base only toggles
        # on a left click, and this way the family cannot drift out of step.
        for item in self._linked:
            item.setVisible(self.item.isVisible())


class _OutsideLegend(pg.LegendItem):
    """A single-column legend beside the plot: entries sit under a collapsible
    file header, long names elide instead of stealing width from the axes, and
    the whole column scrolls rather than being clipped."""

    # Share of the plot widget's width the legend may claim before its labels
    # start eliding. A batch of long file names would otherwise squeeze the
    # axes down to a strip.
    MAX_WIDTH_FRACTION = 0.2

    def __init__(self, **kwargs):
        super().__init__(offset=None, colCount=1, **kwargs)
        # No backing plate or border: outside the plot there is nothing to
        # mask, and a box would just add a second frame beside the axes.
        self.setBrush(pg.mkBrush(0, 0, 0, 0))
        self.setPen(pg.mkPen(0, 0, 0, 0))
        # Fixed vertically so the grid gives it its content height. Stretching
        # separates each swatch (centred in its row) from its label (drawn at
        # the top of one).
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._groups: List[_LegendGroup] = []
        self._current_key = None
        self._busy = False
        # Set by the viewport that owns us; it has to re-clamp its scroll
        # whenever the content height changes.
        self.on_contents_changed = lambda: None

    # -- grouping ---------------------------------------------------------
    def begin_group(self, key, title: str) -> None:
        """Route the entries added from here on under `title`'s header. Series
        register themselves through PlotItem.addItem(name=...), so the group is
        set as context rather than passed in at each call."""
        self._current_key = key
        self._group_for(key, title)

    def end_group(self) -> None:
        self._current_key = None

    def _group_for(self, key, title: Optional[str] = None) -> _LegendGroup:
        for group in self._groups:
            if group.key == key:
                return group
        header = None
        if key is not None:
            header = _GroupHeader(
                title, self._label_font(), self._text_color(), self._relayout
            )
            header.setParentItem(self)
        group = _LegendGroup(key, header)
        # The ungrouped tail always sorts last, so "Fit" and "Removed" stay
        # below every file no matter when they were registered.
        if key is None:
            self._groups.append(group)
        else:
            tail = len(self._groups) - (1 if self._groups and self._groups[-1].key is None else 0)
            self._groups.insert(tail, group)
        return group

    # -- entries ----------------------------------------------------------
    def addItem(self, item, name):
        """Add an entry, keeping the full name for the tooltip and for eliding.

        LabelItem drops its text into an HTML span, so the name is escaped on
        the way in; `_full_text` keeps the original to re-elide from."""
        super().addItem(item, escape(name))
        entry = self.items[-1]
        _, label = entry
        label._full_text = name
        label._shown_text = name
        label.setToolTip(name)
        self._group_for(self._current_key).entries.append(entry)
        self._relayout()

    def _addItemToLayout(self, sample, label):
        # Placement is _relayout's job: it interleaves the group headers, which
        # pyqtgraph's row-filling knows nothing about.
        pass

    def _relayout(self) -> None:
        """Rebuild the grid: header, then its entries unless it is folded."""
        if self._busy:
            return
        self._busy = True
        try:
            for i in range(self.layout.count() - 1, -1, -1):
                self.layout.removeAt(i)
            row = 0
            for group in self._groups:
                if group.header is not None:
                    group.header.setVisible(True)
                    self.layout.addItem(group.header, row, 0, 1, 2)
                    row += 1
                for sample, label in group.entries:
                    # removeAt() only unhooks from the layout; without this the
                    # rows of a folded group stay painted where they last sat.
                    sample.setVisible(not group.collapsed)
                    label.setVisible(not group.collapsed)
                    if group.collapsed:
                        continue
                    self.layout.addItem(
                        sample, row, 0, alignment=Qt.AlignmentFlag.AlignVCenter
                    )
                    self.layout.addItem(label, row, 1)
                    row += 1
            self.rowCount = row
            self.updateSize()
        finally:
            self._busy = False
        self.on_contents_changed()

    def updateSize(self):
        """Size to the grid's own hint. LegendItem's version sums cells row by
        row, which double-counts a header spanning both columns."""
        if self.size is not None:
            return
        hint = self.layout.effectiveSizeHint(Qt.SizeHint.PreferredSize)
        self.setGeometry(QRectF(self.pos(), hint))

    def mouseDragEvent(self, ev):
        # LegendItem drags itself to a new anchor; here it lives in the plot's
        # grid and the drag would fight the viewport's scrolling.
        ev.ignore()

    # -- eliding ----------------------------------------------------------
    def _text_color(self):
        return self.opts["labelTextColor"] or pg.getConfigOption("foreground")

    def _label_font(self) -> QFont:
        """The font the labels actually render in. LabelItem sets the size via
        CSS in the HTML span, so an item's own font carries the right family
        but the wrong size."""
        font = QFont(self.items[0][1].item.font()) if self.items else QFont()
        size = self.opts.get("labelTextSize")
        if isinstance(size, str) and size.endswith("pt"):
            font.setPointSizeF(float(size[:-2]))
        return font

    def _non_text_width(self) -> float:
        """Width one entry spends on things other than its label -- the color
        swatch and the gap after it."""
        if not self.items:
            return 0.0
        sample, _ = self.items[0]
        return sample.boundingRect().width() + self.layout.horizontalSpacing()

    def elide_to_width(self, available: float) -> None:
        """Shorten headers and labels so the legend needs no more than
        `available` px.

        A header is the whole file name and elides from the middle: batch
        exports share a long prefix and differ near the end, so both ends have
        to survive. Entries below it read 'Set NN' and rarely need touching --
        but the flat legends (DRT, residuals) put their whole name here, so
        they elide by the same rule."""
        if self._busy or available <= 0:
            return
        self._busy = True
        try:
            for group in self._groups:
                if group.header is not None:
                    group.header.elide_to(available)
            budget = available - self._non_text_width()
            if budget > 0:
                metrics = QFontMetricsF(self._label_font())
                for _, label in self.items:
                    # Always elide from the original: re-eliding an elided
                    # string erodes it a little further on every resize.
                    full = getattr(label, "_full_text", label.text)
                    shown = _elide_entry(metrics, full, budget)
                    if shown != getattr(label, "_shown_text", None):
                        label._shown_text = shown
                        label.setText(escape(shown))
        finally:
            self._busy = False
        self.updateSize()
        self.on_contents_changed()


class _LegendViewport(pg.GraphicsWidget):
    """Clips the legend to the height the pane can spare and scrolls it there.

    The legend is a child item rather than a layout item: a QGraphicsLayout
    would squeeze its rows to fit the cap instead of letting them overflow, and
    overflow is the whole point of scrolling."""

    SCROLLBAR_WIDTH = 5.0
    WHEEL_STEP = 40.0  # px per notch

    def __init__(self, legend: _OutsideLegend):
        super().__init__()
        self.legend = legend
        legend.setParentItem(self)
        legend.on_contents_changed = self._contents_changed
        self._offset = 0.0
        self.setFlag(self.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def sizeHint(self, which, constraint=QSizeF()):
        hint = self.legend.effectiveSizeHint(which, constraint)
        # The gutter is always reserved, not only while scrollable: letting it
        # come and go would resize the plot every time a group is folded.
        return QSizeF(hint.width() + self.SCROLLBAR_WIDTH, hint.height())

    def set_available_height(self, available: float) -> None:
        if available <= 0:
            return
        self.setMaximumHeight(available)
        self._contents_changed()

    def _max_offset(self) -> float:
        return max(0.0, self.legend.geometry().height() - self.geometry().height())

    def _contents_changed(self) -> None:
        """The legend grew or shrank -- folding a group, eliding, a new entry.

        updateGeometry() belongs here and *not* in _refit: the plot's grid only
        re-reads sizeHint when asked, so without it an unfolded group stays at
        the folded height and scrolls instead of expanding. Keeping it off the
        resize path stops the two from driving each other in a loop."""
        self.updateGeometry()
        self._refit()

    def _refit(self) -> None:
        """Re-clamp the scroll and re-apply it. LegendItem.updateSize() calls
        setGeometry, which resets pos, so the offset has to be re-applied after
        every content change rather than only when scrolling."""
        self._offset = min(max(0.0, self._offset), self._max_offset())
        self.legend.setPos(0.0, -self._offset)
        self.update()

    def resizeEvent(self, ev):
        self._refit()

    def wheelEvent(self, ev):
        if self._max_offset() <= 0:
            # Nothing to scroll: leave the event for the ViewBox to zoom with,
            # so the wheel behaves the same over the legend as over the plot.
            ev.ignore()
            return
        self._offset -= ev.delta() * self.WHEEL_STEP / 120.0
        self._refit()
        ev.accept()

    def paint(self, painter, *args):
        max_offset = self._max_offset()
        if max_offset <= 0:
            return
        rect = self.boundingRect()
        content = self.legend.geometry().height()
        thumb = max(18.0, rect.height() * rect.height() / content)
        top = (rect.height() - thumb) * (self._offset / max_offset)
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        # Neutral grey at low alpha: legible on the light and dark themes
        # without picking up either one's accent.
        painter.setBrush(pg.mkBrush(128, 128, 128, 120))
        painter.drawRoundedRect(
            QRectF(rect.width() - self.SCROLLBAR_WIDTH, top,
                   self.SCROLLBAR_WIDTH - 1.0, thumb),
            2.0, 2.0,
        )


def _elide_entry(metrics: QFontMetricsF, text: str, budget: float) -> str:
    """Fit one legend entry into `budget` px, sparing the ' · Set NN' suffix."""
    if metrics.horizontalAdvance(text) <= budget:
        return text
    head, separator, tail = text.rpartition(_LABEL_SEPARATOR)
    if not separator:
        return metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, budget)
    suffix = separator + tail
    head_budget = budget - metrics.horizontalAdvance(suffix)
    if head_budget <= 0:
        # No room for the name at all; the set number alone still identifies
        # the sweep, and the tooltip carries the rest.
        return tail
    return metrics.elidedText(head, Qt.TextElideMode.ElideMiddle, head_budget) + suffix


def _plot_chrome_height(plot_item: pg.PlotItem) -> float:
    """Height of everything in the PlotItem's grid that is not the plot row --
    title, top/bottom axes, and layout margins."""
    _, top_margin, _, bottom_margin = plot_item.layout.getContentsMargins()
    height = top_margin + bottom_margin
    if plot_item.titleLabel.isVisible():
        height += plot_item.titleLabel.height()
    for edge in ("top", "bottom"):
        axis = plot_item.getAxis(edge)
        if axis is not None and axis.isVisible():
            height += axis.height()
    return height


def _add_outside_legend(plot_item: pg.PlotItem, **kwargs) -> _OutsideLegend:
    """Attach an _OutsideLegend, inside its scrolling viewport, in a grid column
    of its own to the right of the plot, and keep it fitted as the pane is
    resized."""
    legend = _OutsideLegend(labelTextSize="9pt", **kwargs)
    viewport = _LegendViewport(legend)
    # The legend, not the viewport: PlotItem.addItem(name=...) registers series
    # against plot_item.legend, and that has to reach addItem().
    plot_item.legend = legend
    plot_item.layout.addItem(viewport, 2, 3)
    # Top-aligned so the legend keeps its content height. Otherwise it stretches
    # to the plot row and the slack separates each swatch from its label.
    plot_item.layout.setAlignment(viewport, Qt.AlignmentFlag.AlignTop)

    refitting = False

    def rewrap():
        # Budget against the host widget's height, which nothing here changes,
        # so the fit converges instead of chasing its own effect.
        nonlocal refitting
        view = plot_item.getViewWidget()
        if view is None or refitting:
            return
        budget = view.width() * legend.MAX_WIDTH_FRACTION
        legend.elide_to_width(budget - _LegendViewport.SCROLLBAR_WIDTH)
        viewport.setMaximumWidth(budget)
        viewport.set_available_height(view.height() - _plot_chrome_height(plot_item))
        if plot_item.geometry().height() <= view.height() + 1:
            return
        # GraphicsView only fits its central item on a resize event, so a
        # PlotItem laid out tall before the wrap would keep that height (bottom
        # axis clipped) until the pane is resized. Ask for the fit here.
        refitting = True
        try:
            # Called unbound, as GraphicsView.resizeEvent does: PlotWidget
            # shadows setRange with an attribute forwarding to the ViewBox,
            # whose setRange is a different method.
            pg.GraphicsView.setRange(
                view,
                QRectF(0, 0, view.width(), view.height()),
                padding=0,
                disableAutoPixel=False,
            )
        finally:
            refitting = False

    plot_item.getViewBox().geometryChanged.connect(rewrap)
    return legend


def build_nyquist_plot(
    datasets: List[EISDataset],
    title: str = "Nyquist Plot",
    style: str = "scatter",
    show_removed: bool = False,
    style_map: Optional[StyleMap] = None,
    fit_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    marker_size: float = DEFAULT_MARKER_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
) -> pg.PlotWidget:
    """Equal-aspect Nyquist overlay. Pass a single-item list for "Single" mode;
    multiple items overlay with one legend entry each."""
    if not datasets:
        raise ValueError("No datasets provided to plot.")

    multi_file = len({ds.file_id for ds in datasets}) > 1

    exponent = _engineering_exponent(_max_abs_extent(datasets, show_removed))
    bottom_axis = _ScaledAxisItem(orientation="bottom", exponent=exponent)
    left_axis = _ScaledAxisItem(orientation="left", exponent=exponent)

    widget = pg.PlotWidget(title=title, axisItems={"bottom": bottom_axis, "left": left_axis})
    plot_item = widget.getPlotItem()
    plot_item.setAspectLocked(True)
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", _axis_label("Z'", exponent))
    plot_item.setLabel("left", _axis_label("-Z''", exponent))
    _close_plot_box(plot_item)
    _hide_plot_options_menu(plot_item)

    legend = _add_outside_legend(plot_item)

    # addItem(ignoreBounds=True), not PlotItem.addLine, which drops that flag:
    # an InfiniteLine reports its position from dataBounds, so it would pin the
    # origin into every range pyqtgraph computes ("A", View All). Negative z
    # sinks it below the data so the heavy stroke stays off the markers.
    for axis_kwargs in ({"pos": 0, "angle": 90}, {"pos": 0, "angle": 0}):
        line = pg.InfiniteLine(pen=pg.mkPen(ORIGIN_COLOR, width=ORIGIN_WIDTH), **axis_kwargs)
        line.setZValue(-1)
        plot_item.addItem(line, ignoreBounds=True)

    # Not seeded with the origin: equal_aspect_limits folds 0 in for
    # full_range on its own, and kept_range must be free of it entirely.
    interactive_items = []
    kept_xs: List[float] = []
    kept_ys: List[float] = []
    all_xs: List[float] = []
    all_ys: List[float] = []

    # Every sweep's removed points share one "Removed" legend entry, registered
    # after the loop so it sorts behind every sweep and toggling it hides the
    # lot (see _LinkedSample). Fits work the same way.
    removed_items: List = []
    fit_items: List = []
    for i, ds in enumerate(datasets):
        color, symbol = _series_style(style_map, ds.key, i)
        # The tooltip stays fully qualified; the legend entry does not need to
        # be, because the file name is the group header above it.
        legend_label = ds.qualified_label if multi_file else ds.label
        if multi_file:
            legend.begin_group(ds.file_id, ds.source_file)

        kept_idx, removed_idx = _split_indices(ds)
        Z = ds.impedances  # kept (unmasked) points only
        x, y = Z.real, -Z.imag
        kept_item = _add_kept_series(
            plot_item, x, y, _point_data(ds.key, legend_label, ds.frequencies, kept_idx, Z),
            ds.label, color, style, symbol=symbol,
            marker_size=marker_size, line_width=line_width,
        )
        interactive_items.append(kept_item)
        kept_xs.extend(x); kept_ys.extend(y)
        all_xs.extend(x); all_ys.extend(y)

        if fit_curves is not None and ds.key in fit_curves:
            _, Zf = fit_curves[ds.key]
            fit_items.append(
                _add_fit_series(plot_item, Zf.real, -Zf.imag, color, line_width)
            )

        if show_removed:
            Zr = ds.data.get_impedances(masked=True)
            fr = ds.data.get_frequencies(masked=True)
            xr, yr = Zr.real, -Zr.imag
            item = _add_removed_series(
                plot_item, xr, yr,
                _point_data(ds.key, legend_label, fr, removed_idx, Zr, removed=True),
                size=marker_size,
            )
            if item is not None:
                interactive_items.append(item)
                removed_items.append(item)
                all_xs.extend(xr); all_ys.extend(yr)

    legend.end_group()
    if fit_items:
        legend.addItem(_LinkedSample(fit_items), "Fit")
    if removed_items:
        legend.addItem(_LinkedSample(removed_items), "Removed")

    widget.interactive_items = interactive_items
    widget.kept_range = _bounds(kept_xs, kept_ys, include_origin=False)
    widget.full_range = _bounds(all_xs, all_ys)
    widget.range_key = "nyquist"

    if widget.full_range is not None:
        xlo, xhi, ylo, yhi = widget.full_range
        plot_item.setXRange(xlo, xhi, padding=0)
        plot_item.setYRange(ylo, yhi, padding=0)

    return widget


def build_bode_plot(
    datasets: List[EISDataset],
    title: str = "Bode Plot",
    style: str = "scatter",
    show_removed: bool = False,
    style_map: Optional[StyleMap] = None,
    fit_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    marker_size: float = DEFAULT_MARKER_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
) -> pg.PlotWidget:
    """|Z| and -phase against frequency, with the same arguments and widget
    attributes as build_nyquist_plot. Fixed view; ranges are log10."""
    if not datasets:
        raise ValueError("No datasets provided to plot.")

    multi_file = len({ds.file_id for ds in datasets}) > 1

    widget = pg.PlotWidget()
    plot_item = widget.getPlotItem()
    # escape(): the title is appended as markup, so it must be treated as text.
    # align=center: LabelItem centres the block but left-aligns the lines
    # inside it, which would push the short subtitle against the left edge.
    plot_item.setTitle(
        f"<div align='center'>{escape(title)}<br>"
        f"<span style='font-size: 9pt'>{escape(BODE_SUBTITLE)}</span></div>"
    )
    # The align above needs a document width to centre within, and LabelItem
    # never sets one. Using the widest line centres the subtitle under the
    # title without letting either wrap.
    title_doc = plot_item.titleLabel.item.document()
    title_doc.setTextWidth(title_doc.idealWidth())
    plot_item.titleLabel.resizeEvent(None)
    # setTitle fixes the title row at one line (30px), so the subtitle would
    # spill over the grid. Overridden here rather than via content margins,
    # which move the title too and never open a gap beneath it.
    plot_item.titleLabel.setMaximumHeight(BODE_TITLE_HEIGHT)
    plot_item.layout.setRowFixedHeight(0, BODE_TITLE_HEIGHT)
    # Slack so the x-axis title and bare top border axis are not sliced at the
    # plot edge (as in build_nyquist_plot).
    plot_item.layout.setContentsMargins(1, 6, 1, 12)
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", "Frequency [Hz]")
    plot_item.setLabel("left", "|Z| [Ω]")

    # Log scaling is set on the axes, not via PlotItem.setLogMode, which only
    # transforms the PlotItem's own items -- the phase series lives in a
    # separate ViewBox. The data is log10'd on the way in, keeping both
    # ViewBoxes in one coordinate system, and the axes just label decades.
    plot_item.getAxis("bottom").setLogMode(True)
    plot_item.getAxis("left").setLogMode(True)

    # The right axis is in use here, so only the top needs a bare border axis
    # to close the box (see build_nyquist_plot).
    plot_item.showAxis("top")
    top_axis = plot_item.getAxis("top")
    top_axis.setStyle(showValues=False, tickLength=0)
    top_axis.setLabel(None)

    # Pannable and zoomable, like the Nyquist plot. The menu stays off: its
    # "View All" reframes the magnitude ViewBox alone, which is exactly the
    # desync follow_magnitude_y below exists to prevent. Auto-Scale on the plot
    # overlay is the supported way back to the framing.
    main_view = plot_item.getViewBox()
    main_view.setMouseEnabled(x=True, y=True)
    main_view.setMenuEnabled(False)
    # Redundant while the ViewBox menu above is off -- kept so re-enabling it
    # cannot bring "Plot Options" back with it.
    _hide_plot_options_menu(plot_item)

    phase_view = pg.ViewBox(enableMenu=False)
    # Driven through the magnitude view rather than by the mouse directly: x is
    # linked below, y is carried by follow_magnitude_y.
    phase_view.setMouseEnabled(x=False, y=False)
    # Below the magnitude ViewBox (at -100): pyqtgraph offers drags in
    # descending z-order and a ViewBox accepts every one, so an overlay on top
    # would consume them -- and the magnitude view is the one that must get
    # them, since it drives both.
    phase_view.setZValue(-200)
    plot_item.scene().addItem(phase_view)
    plot_item.showAxis("right")
    right_axis = plot_item.getAxis("right")
    right_axis.linkToView(phase_view)
    right_axis.setLabel("-Φ [°]")
    phase_view.setXLink(main_view)

    def sync_phase_view() -> None:
        """A ViewBox added straight to the scene is outside the PlotItem's
        layout, so it must be re-aligned by hand on every resize."""
        phase_view.setGeometry(main_view.sceneBoundingRect())
        phase_view.linkedViewChanged(main_view, phase_view.XAxis)

    main_view.sigResized.connect(sync_phase_view)
    sync_phase_view()

    legend = _add_outside_legend(plot_item)

    interactive_items = []
    kept_xs: List[float] = []
    kept_ys: List[float] = []
    all_xs: List[float] = []
    all_ys: List[float] = []
    phases: List[float] = []

    # As in build_nyquist_plot: the entries are registered after every sweep,
    # so "Fit" and "Removed" read last in the legend, and each stands for every
    # series it covers -- here both the magnitude and the phase one.
    removed_items: List = []
    fit_items: List = []
    for i, ds in enumerate(datasets):
        # Shape and color come from style_map as on the Nyquist plot; fill is
        # what separates magnitude from phase here, so it is overridden below
        # rather than taken from the map.
        color, symbol = _series_style(style_map, ds.key, i)
        # As in build_nyquist_plot: qualified for the tooltip, bare for the
        # legend entry, with the file name carried by the group header.
        legend_label = ds.qualified_label if multi_file else ds.label
        if multi_file:
            legend.begin_group(ds.file_id, ds.source_file)

        kept_idx, removed_idx = _split_indices(ds)
        Z = ds.impedances  # kept (unmasked) points only
        freq = ds.frequencies
        x = np.log10(freq)
        magnitude = np.log10(np.abs(Z))
        # -phase, matching the Nyquist plot's -Z'': both read "up is more
        # capacitive".
        phase = -np.angle(Z, deg=True)
        tip_data = _point_data(
            ds.key, legend_label, freq, kept_idx, Z, values=_bode_tip_values
        )

        interactive_items.append(_add_kept_series(
            plot_item, x, magnitude, tip_data, ds.label, color, style,
            symbol=symbol or DEFAULT_MARKER,
            marker_size=marker_size, line_width=line_width,
        ))
        # Same tip data on both series, so hovering either marker describes the
        # whole point. Hollow markers and a dashed line are what separate the
        # phase series from the magnitude one -- which is why the shapes on
        # offer (MARKER_SHAPES) are all filled.
        interactive_items.append(_add_kept_series(
            phase_view, x, phase, tip_data, None, color, style,
            symbol=symbol or DEFAULT_MARKER, hollow=True,
            marker_size=marker_size, line_width=line_width, dashed=True,
        ))
        kept_xs.extend(x); kept_ys.extend(magnitude)
        all_xs.extend(x); all_ys.extend(magnitude)
        phases.extend(phase)

        if fit_curves is not None and ds.key in fit_curves:
            freq_f, Zf = fit_curves[ds.key]
            xf = np.log10(freq_f)
            # Both halves of the fit go on the one legend entry, so hiding it
            # takes the phase curve with the magnitude one.
            fit_items.append(
                _add_fit_series(plot_item, xf, np.log10(np.abs(Zf)), color, line_width)
            )
            fit_items.append(
                _add_fit_series(
                    phase_view, xf, -np.angle(Zf, deg=True), color, line_width
                )
            )

        if show_removed:
            Zr = ds.data.get_impedances(masked=True)
            fr = ds.data.get_frequencies(masked=True)
            xr, magnitude_r = np.log10(fr), np.log10(np.abs(Zr))
            phase_r = -np.angle(Zr, deg=True)
            # One tip payload for both series, as for the kept points: hovering
            # either × describes the whole point.
            removed_tips = _point_data(
                ds.key, legend_label, fr, removed_idx, Zr,
                removed=True, values=_bode_tip_values,
            )
            # A removed point is missing from *both* halves of the sweep, so it
            # is marked on both -- the phase series would otherwise read as an
            # unbroken run through a gap the magnitude series shows.
            for container, y in ((plot_item, magnitude_r), (phase_view, phase_r)):
                item = _add_removed_series(
                    container, xr, y, removed_tips, size=marker_size
                )
                if item is not None:
                    interactive_items.append(item)
                    removed_items.append(item)
            if xr.size:
                all_xs.extend(xr); all_ys.extend(magnitude_r)
                # Framed with the kept phases so the × markers land inside the
                # opening view, matching what full_range does for magnitude.
                phases.extend(phase_r)

    legend.end_group()
    if fit_items:
        legend.addItem(_LinkedSample(fit_items), "Fit")
    if removed_items:
        legend.addItem(_LinkedSample(removed_items), "Removed")

    widget.interactive_items = interactive_items
    widget.kept_range = _bounds(kept_xs, kept_ys, equal_aspect=False)
    widget.full_range = _bounds(all_xs, all_ys, equal_aspect=False)
    widget.range_key = "bode"
    # The scene owns the extra ViewBox; hang it off the widget so it is
    # reachable and its lifetime is tied to the plot.
    widget.phase_view = phase_view

    if widget.full_range is not None:
        xlo, xhi, ylo, yhi = widget.full_range
        plot_item.setXRange(xlo, xhi, padding=0)
        plot_item.setYRange(ylo, yhi, padding=0)

    # Framed explicitly: autorange recomputes on every item or view change and
    # would drift the phase curve out of step with the magnitude one.
    if phases:
        phase_pad = (max(phases) - min(phases)) * 0.05 or 1.0
        phase_view.setYRange(min(phases) - phase_pad, max(phases) + phase_pad, padding=0)

    # Connected last, once both axes are framed, so the opening layout is not
    # read as a user gesture to be mirrored.
    _follow_magnitude_y(main_view, phase_view)
    return widget


def _follow_magnitude_y(main_view: pg.ViewBox, phase_view: pg.ViewBox) -> None:
    """Carry the phase axis along with the magnitude one under pan and zoom.

    The two axes are independent -- decades of ohms against degrees -- so x can
    simply be linked but y cannot. Left alone, dragging would slide the
    magnitude curve while the phase curve sat still, and the pair would stop
    lining up. Instead the phase range is re-mapped by whatever affine change
    the magnitude range just underwent, so every point keeps its position on
    screen and the two curves move as one.
    """
    previous = [tuple(main_view.viewRange()[1])]

    def follow(_view, new_range) -> None:
        was, now = previous[0], tuple(new_range)
        previous[0] = now
        (a0, a1), (b0, b1) = was, now
        if was == now or a1 == a0 or b1 == b0:
            return
        p0, p1 = phase_view.viewRange()[1]
        # The same scale and shift, expressed as fractions of each span.
        span = (p1 - p0) * (b1 - b0) / (a1 - a0)
        low = p0 - (a0 - b0) * (p1 - p0) / (a1 - a0)
        phase_view.setYRange(low, low + span, padding=0)

    main_view.sigYRangeChanged.connect(follow)
    # Kept alive with the view it serves; a bare local would be collected.
    main_view._phase_follow = follow


class _SymmetricYViewBox(pg.ViewBox):
    """A ViewBox whose wheel zoom is anchored on y = 0 instead of on the cursor.

    pyqtgraph scales about wherever the pointer happens to be, which walks the
    residual axis off centre -- scroll near the top and you end up with, say,
    -1% to +6%, so the two limit lines no longer sit a matched distance from
    the zero line and the figure stops reading as symmetric. Residuals are
    signed deviations about zero, so the x axis is the one point the scale
    should always be stretched about.
    """

    def wheelEvent(self, ev, axis=None):
        # axis=0 is the bottom axis' own wheel handler; x is masked out here
        # (see build_residuals_plot), so there is nothing for it to scale.
        if axis == 0 or not self.state["mouseEnabled"][1]:
            ev.ignore()
            return

        s = 1.02 ** (ev.delta() * self.state["wheelScaleFactor"])
        self._resetTarget()
        # x=None leaves the frequency axis untouched; the centre's x is unused
        # for the same reason (scaleBy takes the setYRange-only path).
        self.scaleBy(y=s, center=pg.Point(0.0, 0.0))
        ev.accept()
        self.sigRangeChangedManually.emit([False, True])


def _residual_text(value: float) -> str:
    """One residual as it reads in a metadata box. The runaways a vanishing
    denominator produces are shown for what they are rather than formatted as a
    number nobody can read (see build_residuals_plot's off-scale handling)."""
    if np.isnan(value):
        return "n/a"
    if np.isinf(value):
        return "+∞ %" if value > 0 else "−∞ %"
    return f"{value:.4g} %"


def _residual_tip_values(names: Tuple[str, str], re_value, im_value) -> str:
    """Both parts at one frequency, named for the convention in force.

    Both, not just the hovered series: as on the Bode plot, what you want off a
    point is the full state at that frequency -- a real part inside the limit
    means little without the imaginary part beside it."""
    return (
        f"{names[0]}: {_residual_text(float(re_value))}\n"
        f"{names[1]}: {_residual_text(float(im_value))}"
    )


def build_residuals_plot(
    result,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
    soft_threshold: Optional[float] = None,
    residual_mode: Optional[str] = None,
    label: Optional[str] = None,
) -> pg.PlotWidget:
    """Relative residuals, in percent, of a validation result (Kramers-Kronig
    or Z-HIT) against frequency, log-x.

    `residual_mode` picks what they are relative *to* -- |Z| or each part's own
    magnitude, see core.validation.RESIDUAL_MODES -- and names the series to
    match; None takes that module's default. The same convention decides which
    points were rejected, so the limit lines below mark exactly the points that
    went.

    `threshold` is the limit points are rejected at outright; `soft_threshold`
    the advanced mode's inner limit, drawn only when the two differ. They also
    frame the figure -- the axis runs to the higher of them plus
    _AXIS_HEADROOM_PERCENT, and residuals past that are pinned to the edge and
    marked rather than allowed to set the scale.

    `label` names the sweep in a point's metadata box; it falls back to the
    title, which carries the label plus the method that produced it."""
    # Deferred, as the module docstring's import note explains. Free here: a
    # validation result in hand means the analysis stack is already loaded.
    from core.validation import RESIDUAL_BY_MODULUS, relative_residuals

    residual_mode = residual_mode or RESIDUAL_BY_MODULUS
    freq, res_re, res_im = relative_residuals(result, residual_mode)

    widget = pg.PlotWidget(title=title, viewBox=_SymmetricYViewBox())
    plot_item = widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", "Frequency [Hz]")
    plot_item.setLabel("left", "Relative residual [%]")
    _close_plot_box(plot_item)
    legend = _add_outside_legend(plot_item)

    # Plain percent values instead of pyqtgraph's SI-prefix axis scaling, which
    # would label the axis "0.001" and show values as a multiplier of it.
    plot_item.getAxis("left").enableAutoSIPrefix(False)
    plot_item.getAxis("bottom").enableAutoSIPrefix(False)

    # Y only: the wheel stretches and shrinks the residual scale, and a drag
    # slides it, but frequency stays put -- panning x would only lose the limit
    # lines' span or slide the figure off its own data.
    #
    # The scale is the one thing here worth adjusting. Framing on the limits
    # (see _AXIS_HEADROOM_PERCENT) leaves a well-fitted sweep as a near-flat
    # line across the middle, and stretching y is how you read its shape
    # without losing the fixed framing everywhere else. _SymmetricYViewBox
    # keeps that stretch centred on the zero line.
    view_box = plot_item.getViewBox()
    view_box.setMouseEnabled(x=False, y=True)
    view_box.setMenuEnabled(False)
    _hide_plot_options_menu(plot_item)  # as in build_bode_plot

    res_re = np.asarray(res_re, dtype=float)
    res_im = np.asarray(res_im, dtype=float)

    # The Y range is settled before anything is drawn: which points are
    # drawable depends on it, and so does where the off-scale markers go.
    #
    # This is the opening frame, not a lock -- the wheel rescales y from here
    # (see setMouseEnabled above). What is pinned off scale stays pinned
    # though: those points are NaN in the series, so zooming out will not
    # uncover them. The marker is how you know they are there.
    levels = [abs(lvl) for lvl in (threshold, soft_threshold) if lvl is not None]
    if levels:
        y_max = max(levels) + _AXIS_HEADROOM_PERCENT
    else:
        finite = np.concatenate(
            [res_re[np.isfinite(res_re)], res_im[np.isfinite(res_im)]]
        )
        span = float(np.abs(finite).max()) if finite.size else 1.0
        y_max = min(span, _MAX_RESIDUAL_AXIS_PERCENT) * 1.15

    # Anything the axis cannot hold -- infinite, or merely enormous -- is drawn
    # as NaN so connect="finite" breaks the line there rather than running it
    # off to nowhere. The markers below say where it went.
    drawable = [np.isfinite(v) & (np.abs(v) <= y_max) for v in (res_re, res_im)]

    re_name, im_name = _RESIDUAL_SERIES_NAMES[residual_mode]
    re_color, im_color = SERIES_COLORS[0], SERIES_COLORS[1]

    # One payload per plotted point, in the arrays' own order -- the series are
    # drawn over the whole sweep with NaN in the off-scale slots, so the tips
    # line up index for index. Note is empty here: a point drawn at its own
    # value needs no qualifier, and the pinned ones carry theirs below.
    tips = [
        {
            "label": label or title or "Residuals",
            "index": int(i),
            "freq": float(f),
            "removed": False,
            "values": _residual_tip_values((re_name, im_name), r, m),
            "note": "",
        }
        for i, (f, r, m) in enumerate(zip(freq, res_re, res_im))
    ]

    interactive_items = []
    for values, keep, name, color, symbol in (
        (res_re, drawable[0], re_name, re_color, "o"),
        (res_im, drawable[1], im_name, im_color, "s"),
    ):
        item = plot_item.plot(
            freq, np.where(keep, values, np.nan), connect="finite",
            pen=pg.mkPen(color, width=1.5),
            symbol=symbol, symbolSize=6, symbolBrush=color, symbolPen=None,
            name=name,
            # Forwarded to the internal ScatterPlotItem (PlotDataItem maps
            # 'data' straight through); `hoverable` is not on that list, so it
            # is set below.
            data=tips,
        )
        item.scatter.opts["hoverable"] = True
        interactive_items.append(item.scatter)

    # Added before setLogMode below, which skips items lacking setLogMode --
    # InfiniteLine has none, so it needs no transform of its own.
    zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#999999", width=0.8))
    zero_line.setZValue(-1)
    plot_item.addItem(zero_line, ignoreBounds=True)

    if len(freq):
        x_bounds = [float(freq.min()), float(freq.max())]

        def _limit_lines(level: float, color: str, name: str) -> None:
            # PlotDataItem, not InfiniteLine: the legend swatch painter reads
            # item.opts unconditionally, which InfiniteLine lacks. The resulting
            # AttributeError inside a Qt paint() override aborts the process.
            pen = pg.mkPen(color, width=1.5, style=Qt.DashLine)
            top = pg.PlotDataItem(x=x_bounds, y=[level, level], pen=pen)
            bottom = pg.PlotDataItem(x=x_bounds, y=[-level, -level], pen=pen)
            # ignoreBounds: these are folded into y_max above instead, so an
            # off-scale limit cannot blow out the view.
            plot_item.addItem(top, ignoreBounds=True)
            plot_item.addItem(bottom, ignoreBounds=True)
            legend.addItem(top, f"±{level}% {name}")

        # Named for what they are only when there are two of them -- a lone
        # limit has no soft one to be the "hard" half of.
        paired = threshold is not None and soft_threshold is not None
        if paired:
            _limit_lines(soft_threshold, "#7fb069", "soft limit")
        if threshold is not None:
            _limit_lines(threshold, "orange", "hard limit" if paired else "threshold")

    off_x, off_y, off_tips = [], [], []
    for values, keep in zip((res_re, res_im), drawable):
        for i, (x, value, ok) in enumerate(zip(freq, values, keep)):
            if not ok:
                off_x.append(float(x))
                off_y.append(y_max * 0.97 if value > 0 else -y_max * 0.97)
                # The marker sits at a made-up y, so its box is the only place
                # the real number is readable -- hence the note.
                off_tips.append({**tips[i], "note": "off scale"})
    if off_x:
        # Pinned inside the edge, unfilled, so a point too large to draw is
        # visibly *there* -- it was rejected, and a gap in the line would
        # otherwise read as missing data.
        off_scale = pg.PlotDataItem(
            x=off_x, y=off_y, pen=None,
            symbol="t1", symbolSize=11, symbolBrush=None,
            symbolPen=pg.mkPen(_REMOVED_COLOR, width=1.5),
            name="off scale",
            data=off_tips,
        )
        off_scale.scatter.opts["hoverable"] = True
        interactive_items.append(off_scale.scatter)
        # A named item is added to the legend by addItem itself, unlike the
        # limit lines above, whose labels are built from their levels.
        plot_item.addItem(off_scale, ignoreBounds=True)

    # Read by the hosting pane to wire up the metadata boxes, as on the
    # Nyquist and Bode plots.
    widget.interactive_items = interactive_items

    # Must precede the fixed Y range below: updateLogMode re-triggers its own
    # autorange, which would clobber an earlier setYRange.
    plot_item.setLogMode(x=True, y=False)
    plot_item.setYRange(-y_max, y_max, padding=0)

    return widget


def _drt_x_axis(plot_item) -> None:
    """Label and orient a DRT plot's x axis: frequency, in decades, inverted so
    high frequency sits on the left and the plot matches the Nyquist plot's
    orientation."""
    plot_item.setLabel("bottom", "Frequency [Hz]")
    plot_item.setLogMode(x=True, y=False)
    plot_item.getViewBox().invertX(True)


def _drt_x_values(tau):
    """The frequencies a run of time constants is drawn against."""
    return 1.0 / (2.0 * math.pi * tau)


def _collect_drt_bounds(xs: List[float], ys: List[float], x, y) -> None:
    """Fold one drawn curve into the running range.

    The x values go in as decades: _drt_x_axis puts the plot in log mode, which
    leaves the series carrying raw frequencies while the ViewBox works in log10
    of them -- so a range built from the raw values would land the view nowhere
    near the curve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # A non-positive or non-finite x has no logarithm and would poison min/max.
    keep = np.isfinite(x) & (x > 0) & np.isfinite(y)
    if not keep.any():
        return
    xs.extend(np.log10(x[keep]))
    ys.extend(y[keep])


def _set_drt_range(widget, xs: List[float], ys: List[float]) -> None:
    """Hang the framing PgFigurePane's Auto-Scale button reads off `widget`.
    The range key is what lets a rebuild of the same plot -- running the peak
    fit, say -- keep the view the user panned to."""
    widget.kept_range = widget.full_range = _bounds(xs, ys, equal_aspect=False)
    widget.range_key = "drt"


def credible_intervals(result):
    """(freq, mean, lower, upper) for a DRT result carrying Bayesian credible
    intervals, or None for one that does not.

    Only a Bayesian run has them. pyimpspec leaves the three arrays empty
    otherwise and get_drt_credible_intervals_data() then hands back four empty
    arrays, which is the "nothing to draw" case here -- as is a result class
    that has no such method at all."""
    getter = getattr(result, "get_drt_credible_intervals_data", None)
    if getter is None:
        return None
    tau, mean, lower, upper = getter()
    if len(tau) == 0:
        return None
    return _drt_x_values(tau), mean, lower, upper


def _add_credible_band(plot_item, freq, lower, upper, color) -> None:
    """Shade between the credible interval's bounds, under the curves.

    The two edges are plotted rather than passed straight to FillBetweenItem
    because the fill tracks its curves in *plotted* coordinates: this axis is
    logarithmic, and only an item the PlotItem owns gets told so. They are
    drawn faintly rather than hidden, so the envelope still reads at a glance
    where the fill is too pale to see -- over a light peak, say.

    Added after the log mode is set (see build_drt_plot), so the fill's first
    path is built from edges already mapped into decades."""
    edges = [
        plot_item.plot(freq, bound, pen=pg.mkPen(_with_alpha(color, 110), width=1.0))
        for bound in (lower, upper)
    ]
    band = pg.FillBetweenItem(*edges, brush=pg.mkBrush(_with_alpha(color, 45)))
    # Behind the curves: FillBetweenItem sinks itself below its own edges, and
    # those sit at the default depth alongside every other series.
    plot_item.addItem(band, ignoreBounds=True)


def build_drt_plot(results: List[Tuple[str, object]], title: str = "DRT") -> pg.PlotWidget:
    """Gamma vs frequency for one or more (label, result) pairs, log-x, high
    frequency on the left.

    A Bayesian run carries credible intervals as well as the distribution
    itself; those are drawn as a shaded band around a dashed posterior mean, in
    the sweep's own colour. It is the only thing that run produces beyond what
    a plain TR-RBF run does, and it takes minutes to hours to get, so it is
    always drawn when it is there."""
    if not results:
        raise ValueError("No DRT results provided to plot.")

    widget = pg.PlotWidget(title=title)
    plot_item = widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("left", "γ [Ω]")
    _close_plot_box(plot_item)
    _hide_plot_options_menu(plot_item)
    _add_outside_legend(plot_item)

    # Before anything is plotted, unlike the other builders: PlotItem.addItem
    # puts each new item into the mode the axis is already in, which is what
    # lets the credible band be filled between two curves in plotted
    # coordinates rather than raw hertz.
    _drt_x_axis(plot_item)

    # What the Auto-Scale button frames (see PgFigurePane._autoscale). A DRT
    # curve has no removed points, so the kept and full spans are the same one.
    xs: List[float] = []
    ys: List[float] = []

    for i, (label, result) in enumerate(results):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        tau, gamma = result.get_drt_data()
        freq = _drt_x_values(tau)
        plot_item.plot(freq, gamma, pen=pg.mkPen(color, width=1.5), name=label)
        _collect_drt_bounds(xs, ys, freq, gamma)

        interval = credible_intervals(result)
        if interval is None:
            continue
        band_freq, mean, lower, upper = interval
        _add_credible_band(plot_item, band_freq, lower, upper, color)
        plot_item.plot(
            band_freq, mean,
            pen=pg.mkPen(color, width=1.2, style=Qt.DashLine),
            name=f"{label} ({CREDIBLE_INTERVAL_NAME})",
        )
        # The band frames the plot too: it is wider than the curve by
        # construction, and cropping the uncertainty would defeat drawing it.
        _collect_drt_bounds(xs, ys, band_freq, lower)
        _collect_drt_bounds(xs, ys, band_freq, upper)

    _set_drt_range(widget, xs, ys)
    return widget


def build_drt_peaks_plot(
    results: List[Tuple[str, object]],
    title: str = "Peak Extraction",
    num_per_decade: int = 100,
    show_individual_peaks: bool = True,
) -> pg.PlotWidget:
    """The skew-normal peaks fitted to a DRT, on the same axes as
    build_drt_plot so the two can be read against each other."""
    if not results:
        raise ValueError("No DRT peak results provided to plot.")

    widget = pg.PlotWidget(title=title)
    plot_item = widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("left", "γ [Ω]")
    _close_plot_box(plot_item)
    _hide_plot_options_menu(plot_item)
    _add_outside_legend(plot_item)

    xs: List[float] = []
    ys: List[float] = []

    for i, (label, peaks) in enumerate(results):
        if peaks.get_num_peaks() < 1:
            continue
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        tau = peaks.get_time_constants(num_per_decade=num_per_decade)
        freq = _drt_x_values(tau)

        if show_individual_peaks:
            # Before the sum, so the heavier total line draws over them.
            for index in range(peaks.get_num_peaks()):
                plot_item.plot(
                    freq,
                    peaks.get_gammas(peak_indices=[index], num_per_decade=num_per_decade),
                    pen=pg.mkPen(color, width=1.0, style=Qt.DashLine),
                )

        # Only the summed curve is framed: an individual peak sits under it by
        # construction, so it can add nothing to the range.
        gammas = peaks.get_gammas(num_per_decade=num_per_decade)
        plot_item.plot(
            freq,
            gammas,
            pen=pg.mkPen(color, width=1.5),
            name=f"{label} ({peaks.get_num_peaks()} peak(s))",
        )
        _collect_drt_bounds(xs, ys, freq, gammas)

    _drt_x_axis(plot_item)
    _set_drt_range(widget, xs, ys)
    return widget
