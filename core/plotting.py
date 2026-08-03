"""PyQtGraph plot builders for the desktop GUI: Nyquist, Bode, Residuals, and
DRT.

Each builder returns a ready-to-embed pg.PlotWidget; gui/figure_panes.py
hosts them (PgFigurePane / PgFigureListPane). build_nyquist_plot and
build_bode_plot take the same arguments and expose the same extra widget
attributes, so the Visual tab can swap one for the other.
"""
from __future__ import annotations

import math
from html import escape
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSizeF, Qt
from PySide6.QtWidgets import QSizePolicy

# Annotation-only, and core.io_utils costs ~4 s to import (pyimpspec ->
# scipy.signal, sympy). Keeping it out of the runtime path is what lets the
# GUI build its figure panes without loading the analysis stack; see
# core/__init__.py.
if TYPE_CHECKING:
    from core.io_utils import EISDataset

# The gold crosshair marking Z'=0 and -Z''=0 on Nyquist plots, drawn heavier
# than the grid so the origin reads as a reference the eye can find rather
# than as another gridline.
ORIGIN_COLOR = "#CC9D33"
ORIGIN_WIDTH = 2.0

# Matplotlib's old default tab10 cycle (C0, C1, ...), kept as the app's
# color cycle so existing exports/screenshots don't shift.
TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

_REMOVED_COLOR = "#999999"  # matplotlib's gray "0.6", as a hex value pyqtgraph accepts

# Says which of the Bode plot's two series belongs to which axis. It goes under
# the title rather than into the legend, which is per-sweep -- the alternative
# was either two entries per sweep or a pair of styling-key entries, and both
# crowd out the sweep names that legend is actually for.
BODE_SUBTITLE = "|Z| ●   ,   -Φ ○"

# Height in px for the two-line title above, replacing the single line's worth
# that PlotItem.setTitle fixes the title row at. Raise this if the subtitle
# wraps or either line's font grows; the extra few px past the two lines are
# the gap that keeps the subtitle off the top of the grid.
BODE_TITLE_HEIGHT = 48

# PyQtGraph symbol names cycled by loaded file. "x" is deliberately excluded
# -- it's reserved for removed points (see _add_removed_series).
PG_MARKERS = ("o", "s", "t", "d", "p", "h", "star", "t1", "t2", "t3")

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def equal_aspect_limits(
    xmin: float, xmax: float, ymin: float, ymax: float, *, include_origin: bool = True
) -> Tuple[float, float, float, float]:
    """Pad (xmin, xmax, ymin, ymax) so both axes span the same range (for an
    aspect-locked plot). When include_origin is True (the Nyquist plot
    default), the origin is folded into the range first so 0 is never
    cropped out."""
    if include_origin:
        xmin, xmax = min(xmin, 0), max(xmax, 0)
        ymin, ymax = min(ymin, 0), max(ymax, 0)

    # Pad the narrower axis so both spans match, keeping the scale equal.
    span = max(xmax - xmin, ymax - ymin)
    x_pad = (span - (xmax - xmin)) / 2
    y_pad = (span - (ymax - ymin)) / 2
    return xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad


def _engineering_exponent(max_abs: float) -> int:
    """Round down to the nearest multiple of 3 (engineering notation: 10^0,
    10^3, 10^6, 10^-3, ...), so the scaled mantissa always lands in roughly
    1-999. 0 (no scaling) for anything under 1000."""
    if max_abs < 1000 or not math.isfinite(max_abs) or max_abs <= 0:
        return 0
    return (math.floor(math.log10(max_abs)) // 3) * 3


def _axis_label(base: str, exponent: int) -> str:
    if exponent == 0:
        return f"{base} [Ω]"
    return f"{base} [Ω × 10{str(exponent).translate(_SUPERSCRIPT)}]"


class _ScaledAxisItem(pg.AxisItem):
    """An AxisItem whose tick numbers are pre-divided by 10**exponent, to
    match the "× 10^n" scale called out in the axis label (built by
    _axis_label) instead of pyqtgraph's default SI-prefix behavior."""

    def __init__(self, *args, exponent: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._exponent = exponent
        # Our own "× 10^n" scale is baked into the label text (_axis_label);
        # without this, AxisItem also appends its own auto-computed "(x...)"
        # scale factor whenever the data range falls under 1, duplicating --
        # and for exponent != 0, contradicting -- our label.
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
    """The text of the metadata box shown for one plotted point.

    Composed from the payload _point_data attached to the point rather than
    from the point's plotted position: those coordinates are whatever the axes
    happen to hold (log-scaled frequency and magnitude on the Bode plot), and
    on either plot they only cover half of the values worth showing."""
    suffix = " (removed)" if data["removed"] else ""
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
    """Per-point payload attached to every ScatterPlotItem. Carries the
    dataset's stable key (unique across every loaded file -- what the eraser
    uses to resolve the point back to its dataset) and its display label
    (what the tooltip shows), plus the point's index *in the
    unmasked-inclusive array*, which is what the eraser needs to toggle a
    mask entry -- the kept series is built from unmasked points only, so
    scatter position != index.

    values formats the impedance half of the tooltip (see point_tip), so each
    plot can name the quantities it actually draws; it is applied here, at
    plot time, because the plotted coordinates alone can't be turned back into
    an impedance.

    'removed' stays a flag rather than being folded into the label, which
    consumers key datasets by."""
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


def _marker_kwargs(color: str, symbol: Optional[str], hollow: bool) -> dict:
    """ScatterPlotItem styling for one series: filled in the dataset's color,
    or outlined in it when hollow. symbol=None leaves pyqtgraph's default
    ('o') in place rather than naming it, so callers that don't style by file
    keep the behavior they had before symbols existed."""
    if hollow:
        kwargs = dict(brush=None, pen=pg.mkPen(color, width=1.2), size=6)
    else:
        kwargs = dict(brush=pg.mkBrush(color), pen=None, size=6)
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
):
    """Draw one dataset's kept points: a filled scatter for 'scatter' style,
    or a connected line plus a hoverable scatter for 'line' style. In 'line'
    style that scatter is invisible unless a per-file symbol is given, in
    which case it draws the file's marker on top of the line so 'marker =
    file' holds in both styles; either way it stays the hover/hit target.
    Returns the hoverable ScatterPlotItem (used for hit-testing, range
    calculation, and role-tagging).

    container is the PlotItem the series belongs to, or -- for the Bode plot's
    phase series -- the second ViewBox drawn on top of it. Only a PlotItem
    turns 'name' into a legend entry, so pass None for series that shouldn't
    get one.

    hollow outlines the markers and dashes the line instead of filling them
    in, which is how the Bode plot tells its right-axis phase series apart
    from the left-axis magnitude one."""
    if style == "line":
        # The line itself carries the legend entry (so the swatch shows a
        # colored line, not a marker); the scatter stays unnamed.
        pen = pg.mkPen(color, width=1.5, style=Qt.DashLine if hollow else Qt.SolidLine)
        container.addItem(pg.PlotDataItem(x=x, y=y, pen=pen, name=name))
        scatter_name = None
        if symbol is not None:
            marker_kwargs = _marker_kwargs(color, symbol, hollow)
        else:
            marker_kwargs = dict(brush=None, pen=None, size=6)
    elif style == "scatter":
        scatter_name = name
        marker_kwargs = _marker_kwargs(color, symbol, hollow)
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


def _add_removed_series(plot_item, x, y, tip_data):
    """Draw masked-out points as muted 'x' markers, matching
    core.plotting._plot_removed. Returns the ScatterPlotItem, or None if
    there are no removed points.

    Deliberately unnamed, so pyqtgraph doesn't give it a legend entry as it is
    added: the single shared "Removed" entry is registered by the caller once
    every sweep has been drawn, which is what puts it after them all rather
    than interleaved among them (it would otherwise land right behind whichever
    sweep happened to be the first with a masked point)."""
    if x.size == 0:
        return None

    scatter = pg.ScatterPlotItem(
        x=x, y=y,
        symbol="x",
        pen=pg.mkPen(_REMOVED_COLOR, width=1.3),
        brush=None,
        size=6,
        hoverable=True,
        data=tip_data,
    )
    scatter._eis_role = "removed"
    plot_item.addItem(scatter)
    return scatter


def _add_fit_series(container, x, y, color: str):
    """Draw one dataset's fitted circuit response as a dashed line in that
    dataset's own color, so a fit is read against the measurement it belongs
    to. Returns the PlotDataItem, which the caller registers once as the
    single shared "Fit" legend entry (same reasoning as _add_removed_series).

    Two things this deliberately is not:

    Not hoverable, and never added to the widget's interactive_items. A fit
    curve is interpolated at an arbitrary number of points per decade and so
    has no measured-point indices behind it, while the tooltip and the eraser
    both resolve a click to data["key"]/data["index"]. Letting the eraser
    hit-test this would mean clicking a fit could mask an unrelated point.

    Not part of the range calculation: added with ignoreBounds=True and kept
    out of the caller's kept/all coordinate lists, so a fit that diverges
    can't blow out the framing of the data it is drawn over -- nor, on the
    Nyquist plot, shift the shared engineering-notation exponent.
    """
    item = pg.PlotDataItem(
        x=x, y=y,
        pen=pg.mkPen(color, width=1.6, style=Qt.DashLine),
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
    """Framing for the given points, or None when there are none to frame
    (callers fall back rather than setting a degenerate zero-span range).

    include_origin=False frames the data alone -- what the Auto-Scale button
    wants, so a spectrum sitting far from 0 fills the view instead of being
    squashed into a corner by an origin nobody asked to see.

    equal_aspect=False skips the padding that squares the two axes up (and
    with it the origin question entirely): that padding only makes sense where
    both axes carry the same quantity, as on the Nyquist plot, and would
    wildly over-pad the Bode plot's decades-against-degrees axes."""
    if not xs:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Breathing room around the tight data box, so a point sitting right at
    # an extreme doesn't have its marker sliced by the ViewBox edge (padding=0
    # below draws the range exactly).
    margin = 0.05
    x_pad = (xmax - xmin) * margin or 1e-6
    y_pad = (ymax - ymin) * margin or 1e-6
    xlo, xhi = xmin - x_pad, xmax + x_pad
    ylo, yhi = ymin - y_pad, ymax + y_pad
    if not equal_aspect:
        return xlo, xhi, ylo, yhi
    return equal_aspect_limits(xlo, xhi, ylo, yhi, include_origin=include_origin)


class _WrappingLegend(pg.LegendItem):
    """A legend that lives beside the plot rather than on top of it, and adds
    columns rather than height once it runs out of room.

    pyqtgraph's own legend is anchored inside the ViewBox, where it covers the
    data -- with a dozen sweeps selected the labels sit unreadably on the
    points. Moving it into a column of the PlotItem's grid fixes that but
    introduces a worse failure: the grid sizes itself to its contents, so a
    long enough legend makes the PlotItem taller than the widget hosting it and
    the x axis is simply clipped away. Wrapping into columns is what keeps that
    from happening at any sweep count.

    offset=None is what stops LegendItem anchoring itself to a parent on the
    way in; it is positioned by the layout instead.
    """

    def __init__(self, **kwargs):
        super().__init__(offset=None, **kwargs)
        # No backing plate or border: outside the plot there is nothing to
        # mask, and the box would only add a second frame beside the axes.
        self.setBrush(pg.mkBrush(0, 0, 0, 0))
        self.setPen(pg.mkPen(0, 0, 0, 0))
        # Fixed vertically so the grid gives it its content height instead of
        # stretching it down the whole plot row. Stretching doesn't just space
        # the entries out: LegendItem centres each swatch in its row but draws
        # each label at the top of one, so the slack pulls every marker away
        # from the text it labels.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._wrapping = False

    def _row_height(self) -> float:
        """Height one entry wants. The label's *preferred* size hint, not its
        bounding rect -- the latter reports whatever height the grid stretched
        the row to, which is a good deal taller and would under-count how many
        entries actually fit."""
        sample, label = self.items[0]
        return max(
            label.sizeHint(Qt.SizeHint.PreferredSize, QSizeF()).height(),
            sample.boundingRect().height(),
        )

    def wrap_to_height(self, available: float) -> None:
        """Re-column so the entries need no more than `available` px of height.

        Re-entrancy is guarded because re-columning resizes the legend, which
        relays the grid, which is what calls this in the first place."""
        if self._wrapping or not self.items or available <= 0:
            return
        row_height = self._row_height()
        if row_height <= 0:
            return
        rows = max(1, int(available // row_height))
        columns = max(1, math.ceil(len(self.items) / rows))
        self._wrapping = True
        try:
            # Cap the height as well as re-columning. A QGraphicsGridLayout
            # sizes itself to its contents, and a PlotWidget only re-fits the
            # PlotItem to the viewport on a resize event -- so without a
            # ceiling, a legend that starts out too long stretches the grid
            # past the widget and it stays that way even after the wrap.
            self.setMaximumHeight(available)
            if columns != self.columnCount:
                self.setColumnCount(columns)
        finally:
            self._wrapping = False


def _plot_chrome_height(plot_item: pg.PlotItem) -> float:
    """Height of everything in the PlotItem's grid that isn't the plot row --
    the title and the top/bottom axes, plus the layout's own margins.

    Measured from those items rather than as (PlotItem - ViewBox), which looks
    equivalent but isn't usable: the ViewBox shares its row with the legend and
    so stretches to match it, making that difference move around as the legend
    grows, exactly when it needs to be stable."""
    _, top_margin, _, bottom_margin = plot_item.layout.getContentsMargins()
    height = top_margin + bottom_margin
    if plot_item.titleLabel.isVisible():
        height += plot_item.titleLabel.height()
    for edge in ("top", "bottom"):
        axis = plot_item.getAxis(edge)
        if axis is not None and axis.isVisible():
            height += axis.height()
    return height


def _add_outside_legend(plot_item: pg.PlotItem, **kwargs) -> _WrappingLegend:
    """Attach a _WrappingLegend in a grid column of its own, to the right of
    the plot, and keep it wrapped as the pane is resized.

    PlotItem's grid puts the ViewBox at (2, 1) with the y axes either side of
    it, so column 3 is free and row 2 is the one that stretches."""
    legend = _WrappingLegend(labelTextSize="9pt", **kwargs)
    plot_item.legend = legend
    plot_item.layout.addItem(legend, 2, 3)
    # Without this the legend is stretched to the full height of the plot row,
    # and it passes that slack on to its own rows: entries drift apart, and
    # each swatch -- which LegendItem centres vertically in its row, while the
    # label sits at the top -- separates from the text it belongs to. Aligning
    # the legend to the top of the cell keeps it at its content height, so the
    # rows stay tight and swatch and label line up.
    plot_item.layout.setAlignment(legend, Qt.AlignmentFlag.AlignTop)

    refitting = False

    def rewrap():
        # Budget against the hosting widget's height, which nothing here can
        # change, so the wrap converges instead of chasing its own effect.
        nonlocal refitting
        view = plot_item.getViewWidget()
        if view is None or refitting:
            return
        legend.wrap_to_height(view.height() - _plot_chrome_height(plot_item))
        if plot_item.geometry().height() <= view.height() + 1:
            return
        # A GraphicsView only fits its central item to the viewport on a resize
        # event (GraphicsView.resizeEvent -> setRange), so a PlotItem that was
        # laid out tall before the legend wrapped keeps that height, bottom axis
        # still clipped, until something resizes the pane. Ask for the fit here
        # instead, the same way resizeEvent would.
        refitting = True
        try:
            # Called unbound, as GraphicsView.resizeEvent itself does: a
            # PlotWidget shadows setRange with an instance attribute forwarding
            # to the ViewBox, whose setRange is a different method entirely.
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
    style_map: Optional[Dict[str, Tuple[str, str]]] = None,
    fit_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pg.PlotWidget:
    """Equal-aspect Nyquist overlay -- the PyQtGraph equivalent of
    core.plotting.plot_single/plot_overlay. Pass a single-item list for the
    "Single" mode; multiple items overlay with one legend entry each.

    style_map is an optional ds.key -> (color, pg symbol) override, e.g. to
    color by sweep index and shape by source file so cross-file comparisons
    stay readable (see gui.main_window._build_style_map). Datasets missing
    from the map fall back to the plain per-position tab10 cycle and no
    symbol override, which is also what happens when style_map is None.

    fit_curves is an optional ds.key -> (frequencies, complex impedances) of
    fitted equivalent-circuit responses (see core.ecm), drawn as a dashed
    line in each dataset's own color. The frequencies travel with the
    impedances because a fit is interpolated onto its own denser grid rather
    than evaluated at the measured points -- build_bode_plot needs them for
    its x axis, and nothing may assume they line up with ds.frequencies.
    Fit curves are drawn but never hit-tested or ranged over; see
    _add_fit_series.

    Legend entries and hover/click tooltips use each dataset's
    qualified_label (source file + sweep) instead of its bare label whenever
    the datasets passed in span more than one file -- there is no need to
    qualify a single-file selection.

    The returned widget carries extra attributes consumed by
    gui.figure_panes.PgFigurePane:
      - interactive_items : every hoverable ScatterPlotItem (kept and removed),
        for wiring up the hover/click-to-show-metadata tooltip.
      - kept_range / full_range : (xlo, xhi, ylo, yhi) tuples, or None when
        there was nothing to frame. kept_range is what the Auto-Scale button
        zooms to -- the kept points alone, origin excluded. full_range is the
        default framing applied here: it covers removed points too, so
        nothing drawn is cropped out, and keeps the origin in view.
      - range_key : names the coordinate system those ranges are in, so the
        pane knows not to carry a remembered zoom across to a plot where the
        same numbers mean something else (see build_bode_plot).

    Hiding removed points is handled by pyqtgraph's own legend -- clicking a
    legend swatch toggles that item's visibility -- so there's no dedicated
    toggle here.
    """
    if not datasets:
        raise ValueError("No datasets provided to plot.")

    multi_file = len({ds.file_id for ds in datasets}) > 1

    exponent = _engineering_exponent(_max_abs_extent(datasets, show_removed))
    bottom_axis = _ScaledAxisItem(orientation="bottom", exponent=exponent)
    left_axis = _ScaledAxisItem(orientation="left", exponent=exponent)

    widget = pg.PlotWidget(title=title, axisItems={"bottom": bottom_axis, "left": left_axis})
    plot_item = widget.getPlotItem()
    # AxisItem nudges its label a few px past its own laid-out row (see its
    # resizeEvent), which the default 1px layout margin doesn't leave room
    # for -- clipping the bottom of the x-axis title. The bare top/right
    # border axes (added below) are similarly prone to being sliced right at
    # their 1px margin -- their laid-out width/height rounds to ~0 since they
    # carry no ticks or label, so their line sits right at the plot's edge
    # with no slack. Pad all three so nothing lands exactly on the boundary.
    plot_item.layout.setContentsMargins(1, 6, 12, 12) #plot margin (left, top, right, bottom)
    plot_item.setAspectLocked(True)
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", _axis_label("Z'", exponent))
    plot_item.setLabel("left", _axis_label("-Z''", exponent))

    # Bottom/left get axis lines "for free" since they carry ticks and
    # labels; top/right are hidden by default (no line at all), leaving the
    # plot area open on two sides. Show them too, bare, purely to close the
    # box -- matching the bottom/left border.
    plot_item.showAxis("top")
    plot_item.showAxis("right")
    for side in ("top", "right"):
        border_axis = plot_item.getAxis(side)
        border_axis.setStyle(showValues=False, tickLength=0)
        border_axis.setLabel(None)

    legend = _add_outside_legend(plot_item)

    # Added via addItem(ignoreBounds=True) rather than PlotItem.addLine,
    # which doesn't forward that flag: an InfiniteLine reports its position
    # from dataBounds, so a bounds-participating line at 0 would pin the
    # origin into every range pyqtgraph computes itself (the "A" button,
    # right-click > View All) -- the same thing the Auto-Scale button below
    # deliberately avoids. Sunk beneath the data with a negative z so the
    # heavier stroke can't sit on top of a marker.
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

    # The first sweep's removed points, if any: it carries the one shared
    # "Removed" legend entry, registered after the loop so it sorts behind
    # every sweep instead of among them. Fits are handled the same way.
    removed_item = None
    fit_item = None
    for i, ds in enumerate(datasets):
        if style_map is not None and ds.key in style_map:
            color, symbol = style_map[ds.key]
        else:
            color, symbol = TAB10[i % len(TAB10)], None
        legend_label = ds.qualified_label if multi_file else ds.label

        kept_idx, removed_idx = _split_indices(ds)
        Z = ds.impedances  # kept (unmasked) points only
        x, y = Z.real, -Z.imag
        kept_item = _add_kept_series(
            plot_item, x, y, _point_data(ds.key, legend_label, ds.frequencies, kept_idx, Z),
            legend_label, color, style, symbol=symbol,
        )
        interactive_items.append(kept_item)
        kept_xs.extend(x); kept_ys.extend(y)
        all_xs.extend(x); all_ys.extend(y)

        if fit_curves is not None and ds.key in fit_curves:
            _, Zf = fit_curves[ds.key]
            fit_item = _add_fit_series(plot_item, Zf.real, -Zf.imag, color)

        if show_removed:
            Zr = ds.data.get_impedances(masked=True)
            fr = ds.data.get_frequencies(masked=True)
            xr, yr = Zr.real, -Zr.imag
            item = _add_removed_series(
                plot_item, xr, yr,
                _point_data(ds.key, legend_label, fr, removed_idx, Zr, removed=True),
            )
            if item is not None:
                interactive_items.append(item)
                removed_item = removed_item or item
                all_xs.extend(xr); all_ys.extend(yr)

    if fit_item is not None:
        legend.addItem(fit_item, "Fit")
    if removed_item is not None:
        legend.addItem(removed_item, "Removed")

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
    style_map: Optional[Dict[str, Tuple[str, str]]] = None,
    fit_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pg.PlotWidget:
    """|Z| and -phase against frequency: the Bode counterpart of
    build_nyquist_plot, taking the same arguments and exposing the same extra
    widget attributes, so gui.figure_panes.PgFigurePane hosts either one
    interchangeably and the eraser works on both.

    Magnitude is read on the left axis and phase on the right, each with its
    own ViewBox so their scales stay independent. Magnitude is a filled circle
    and phase a hollow one -- circles regardless of what style_map asks for,
    since here the marker shape has to carry that distinction rather than the
    source file, leaving the legend free to name one entry per sweep. Which
    shape means which is spelled out under the title (see BODE_SUBTITLE).

    Frequency and |Z| both span decades, so both are plotted as log10 values
    with their axis labelling them back as powers of ten. The ranges reported
    via kept_range/full_range are therefore in log space -- fine for the pane,
    which only ever hands them back to a ViewBox, but not comparable with the
    Nyquist plot's, which is what range_key exists to signal.

    The view is fixed: two y scales sharing one x axis have no single sensible
    response to a drag or a scroll, so panning and zooming are off (the
    Auto-Scale and Replot buttons still reframe it). Clicking points is
    unaffected, so the tooltip and eraser work as they do on the Nyquist plot.

    Removed points are drawn on the magnitude axis only: one grey × per point
    is all the eraser needs to click it back, and repeating them on the phase
    axis only crowds the plot. Fit curves, unlike removed points, are drawn on
    both axes -- a circuit is judged on how well it reproduces the phase as
    much as the magnitude, and the two are what the fit was scored against.
    """
    if not datasets:
        raise ValueError("No datasets provided to plot.")

    multi_file = len({ds.file_id for ds in datasets}) > 1

    widget = pg.PlotWidget()
    plot_item = widget.getPlotItem()
    # escape(): the subtitle is appended as markup, so the caller's title (a
    # file name, in practice) has to be treated as text rather than trusted to
    # be tag-free.
    # align=center: the title label centres the text block as a whole but lays
    # its lines out left-aligned inside it, so the short subtitle would sit
    # against the left edge of the longer file name above it.
    plot_item.setTitle(
        f"<div align='center'>{escape(title)}<br>"
        f"<span style='font-size: 9pt'>{escape(BODE_SUBTITLE)}</span></div>"
    )
    # ...and the align above only takes effect once the document has a width to
    # centre within: LabelItem never sets one, which leaves the lines centred
    # against nothing and so still flush left. The ideal width is the widest
    # line, so this centres the subtitle under the file name without letting
    # either wrap; the label keeps centring that block within the plot itself.
    title_doc = plot_item.titleLabel.item.document()
    title_doc.setTextWidth(title_doc.idealWidth())
    plot_item.titleLabel.resizeEvent(None)
    # setTitle fixes the title row at 30px, which is one line's worth: the
    # subtitle would otherwise spill down over the top of the grid. Undo that
    # here rather than by padding the layout, since the content margins move
    # the title along with everything else and so never open a gap beneath it.
    plot_item.titleLabel.setMaximumHeight(BODE_TITLE_HEIGHT)
    plot_item.layout.setRowFixedHeight(0, BODE_TITLE_HEIGHT)
    # Same reasoning as build_nyquist_plot: leave the laid-out axis rows a few
    # px of slack so neither the x-axis title nor the bare top border axis is
    # sliced right at the plot's edge.
    plot_item.layout.setContentsMargins(1, 6, 1, 12)
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", "Frequency [Hz]")
    plot_item.setLabel("left", "|Z| [Ω]")

    # Log scaling is set on the axes, not via PlotItem.setLogMode: that method
    # log-transforms the PlotItem's own items, and the phase series below lives
    # in a ViewBox the PlotItem knows nothing about. The x/|Z| data is
    # log10'd on the way in instead (keeping both ViewBoxes in one coordinate
    # system), and the axes are simply told to label decades.
    plot_item.getAxis("bottom").setLogMode(True)
    plot_item.getAxis("left").setLogMode(True)

    # The right axis is in use here, so only the top needs a bare border axis
    # to close the box (see build_nyquist_plot).
    plot_item.showAxis("top")
    top_axis = plot_item.getAxis("top")
    top_axis.setStyle(showValues=False, tickLength=0)
    top_axis.setLabel(None)

    # Static view (see the docstring): no drag-to-pan, no scroll-to-zoom, and
    # no right-click menu -- whose "View All" would in any case only ever
    # reframe the magnitude ViewBox, silently leaving the phase one behind.
    main_view = plot_item.getViewBox()
    main_view.setMouseEnabled(x=False, y=False)
    main_view.setMenuEnabled(False)

    phase_view = pg.ViewBox(enableMenu=False)
    phase_view.setMouseEnabled(x=False, y=False)
    # Below the magnitude ViewBox (which sits at -100) rather than above it: a
    # ViewBox accepts every drag offered to it, and pyqtgraph offers them in
    # descending z-order, so an overlay on top would be the one consuming them.
    # Moot while both have their mouse disabled, but it also keeps the two in
    # the order the click dispatch expects, and would matter again the moment
    # this plot were made interactive.
    phase_view.setZValue(-200)
    plot_item.scene().addItem(phase_view)
    plot_item.showAxis("right")
    right_axis = plot_item.getAxis("right")
    right_axis.linkToView(phase_view)
    right_axis.setLabel("-Φ [°]")
    phase_view.setXLink(main_view)

    def sync_phase_view() -> None:
        """A ViewBox added straight to the scene isn't part of the PlotItem's
        layout, so it has to be kept aligned with the plotted area by hand
        every time that area is resized."""
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
    # so "Fit" and "Removed" read last in the legend.
    removed_item = None
    fit_item = None
    for i, ds in enumerate(datasets):
        # Only the color half of style_map is used -- the marker shape is spoken
        # for here (filled |Z| vs hollow phase), so sweeps from different files
        # are told apart by color alone on this plot.
        if style_map is not None and ds.key in style_map:
            color = style_map[ds.key][0]
        else:
            color = TAB10[i % len(TAB10)]
        legend_label = ds.qualified_label if multi_file else ds.label

        kept_idx, removed_idx = _split_indices(ds)
        Z = ds.impedances  # kept (unmasked) points only
        freq = ds.frequencies
        x = np.log10(freq)
        magnitude = np.log10(np.abs(Z))
        # -phase, to match the -Z'' the Nyquist plot puts on its own y axis:
        # both then read "up is more capacitive".
        phase = -np.angle(Z, deg=True)
        tip_data = _point_data(
            ds.key, legend_label, freq, kept_idx, Z, values=_bode_tip_values
        )

        interactive_items.append(_add_kept_series(
            plot_item, x, magnitude, tip_data, legend_label, color, style, symbol="o",
        ))
        # Same tip data on both series, so hovering either the magnitude or the
        # phase marker for a point describes that whole point.
        interactive_items.append(_add_kept_series(
            phase_view, x, phase, tip_data, None, color, style, symbol="o", hollow=True,
        ))
        kept_xs.extend(x); kept_ys.extend(magnitude)
        all_xs.extend(x); all_ys.extend(magnitude)
        phases.extend(phase)

        if fit_curves is not None and ds.key in fit_curves:
            freq_f, Zf = fit_curves[ds.key]
            xf = np.log10(freq_f)
            # The magnitude curve carries the shared legend entry; the phase
            # one lives in the other ViewBox, which the legend can't reach.
            fit_item = _add_fit_series(
                plot_item, xf, np.log10(np.abs(Zf)), color
            )
            _add_fit_series(phase_view, xf, -np.angle(Zf, deg=True), color)

        if show_removed:
            Zr = ds.data.get_impedances(masked=True)
            fr = ds.data.get_frequencies(masked=True)
            xr, magnitude_r = np.log10(fr), np.log10(np.abs(Zr))
            item = _add_removed_series(
                plot_item, xr, magnitude_r,
                _point_data(
                    ds.key, legend_label, fr, removed_idx, Zr,
                    removed=True, values=_bode_tip_values,
                ),
            )
            if item is not None:
                interactive_items.append(item)
                removed_item = removed_item or item
                all_xs.extend(xr); all_ys.extend(magnitude_r)

    if fit_item is not None:
        legend.addItem(fit_item, "Fit")
    if removed_item is not None:
        legend.addItem(removed_item, "Removed")

    widget.interactive_items = interactive_items
    widget.kept_range = _bounds(kept_xs, kept_ys, equal_aspect=False)
    widget.full_range = _bounds(all_xs, all_ys, equal_aspect=False)
    widget.range_key = "bode"
    # The scene owns the extra ViewBox, but nothing else here refers to it --
    # hang it off the widget so it can be reached (and so its lifetime is
    # obviously tied to the plot it belongs to).
    widget.phase_view = phase_view

    if widget.full_range is not None:
        xlo, xhi, ylo, yhi = widget.full_range
        plot_item.setXRange(xlo, xhi, padding=0)
        plot_item.setYRange(ylo, yhi, padding=0)

    # Framed explicitly rather than left to autorange, which recomputes itself
    # whenever items or the view change and would drift the phase curve out of
    # step with the magnitude one it is meant to be read against.
    if phases:
        phase_pad = (max(phases) - min(phases)) * 0.05 or 1.0
        phase_view.setYRange(min(phases) - phase_pad, max(phases) + phase_pad, padding=0)

    return widget


def build_residuals_plot(
    result,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
) -> pg.PlotWidget:
    """Relative residuals (ΔZ'/|Z| and ΔZ''/|Z|, in percent) of a validation
    result (Kramers-Kronig or Z-HIT) against frequency, log-x.

    result must expose get_residuals_data() -> (freq, res_re, res_im).
    """
    freq, res_re, res_im = result.get_residuals_data()

    widget = pg.PlotWidget(title=title)
    plot_item = widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", "Frequency (Hz)")
    plot_item.setLabel("left", "Relative residual (%)")
    legend = _add_outside_legend(plot_item)

    # Plain percent values instead of pyqtgraph's default SI-prefix axis
    # scaling (which would otherwise label the axis "0.001" or similar and
    # show the real values as a multiplier of that).
    plot_item.getAxis("left").enableAutoSIPrefix(False)
    plot_item.getAxis("bottom").enableAutoSIPrefix(False)

    # Static view: this plot is meant to be read, not navigated -- no
    # drag-to-pan, scroll-to-zoom, or right-click menu.
    view_box = plot_item.getViewBox()
    view_box.setMouseEnabled(x=False, y=False)
    view_box.setMenuEnabled(False)

    re_color, im_color = TAB10[0], TAB10[1]
    plot_item.plot(
        freq, res_re, pen=pg.mkPen(re_color, width=1.5),
        symbol="o", symbolSize=6, symbolBrush=re_color, symbolPen=None,
        name="ΔZ' / |Z|",
    )
    plot_item.plot(
        freq, res_im, pen=pg.mkPen(im_color, width=1.5),
        symbol="s", symbolSize=6, symbolBrush=im_color, symbolPen=None,
        name="ΔZ'' / |Z|",
    )

    # Added last (after setLogMode below applies it via updateLogMode) so the
    # reference lines don't need setLogMode themselves -- InfiniteLine has no
    # such method and PlotItem.updateLogMode skips items without one.
    zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#999999", width=0.8))
    zero_line.setZValue(-1)
    plot_item.addItem(zero_line, ignoreBounds=True)

    y_extent = [float(v) for v in res_re] + [float(v) for v in res_im]

    if threshold is not None and len(freq):
        # PlotDataItem, not InfiniteLine: pyqtgraph's legend swatch painter
        # reads item.opts unconditionally (LegendItem.ItemSample.paint),
        # which InfiniteLine doesn't have -- registering one with the legend
        # raises AttributeError inside a Qt paint() override, which PySide6
        # turns into a hard process abort rather than a catchable exception.
        x_bounds = [float(freq.min()), float(freq.max())]
        thr_pen = pg.mkPen("orange", width=1.5, style=Qt.DashLine)
        top_line = pg.PlotDataItem(x=x_bounds, y=[threshold, threshold], pen=thr_pen)
        bottom_line = pg.PlotDataItem(x=x_bounds, y=[-threshold, -threshold], pen=thr_pen)
        plot_item.addItem(top_line, ignoreBounds=True)
        plot_item.addItem(bottom_line, ignoreBounds=True)
        legend.addItem(top_line, f"±{threshold}% threshold")
        # ignoreBounds keeps these out of autorange (so a handful of
        # far-off-scale threshold sweeps can't blow out every other plot's
        # view) -- fold the threshold into the fixed range by hand instead,
        # so it's always visible rather than only when it happens to fall
        # inside the data's own span.
        y_extent += [threshold, -threshold]

    # Applied before the fixed Y range below: PlotItem.updateLogMode both
    # walks self.items to log-transform each one's x data and re-triggers
    # its own autorange, which would otherwise clobber an explicit
    # setYRange set any earlier.
    plot_item.setLogMode(x=True, y=False)

    # Fixed range, chosen once here rather than left to autorange: the view
    # is static (see setMouseEnabled above), so there's no drag/scroll left
    # for the user to recover a clipped plot with.
    y_max = max((abs(v) for v in y_extent), default=1.0) * 1.15
    plot_item.setYRange(-y_max, y_max, padding=0)

    return widget


def build_drt_plot(results: List[Tuple[str, object]], title: str = "DRT") -> pg.PlotWidget:
    """Gamma vs frequency (distribution of relaxation times) for one or more
    DRT results, log-x, high frequency on the left to match the Nyquist
    plot's orientation.

    results is a list of (label, result) pairs, where each result exposes
    get_drt_data() -> (tau, gamma), e.g. core.drt.run_drt's return value.
    BHTResult (core.drt.run_drt_bht) instead returns (tau, gamma_re,
    gamma_im), estimated separately from the real and imaginary parts; both
    are plotted, sharing one color per result and distinguished by a dashed
    pen for the imaginary part.
    """
    if not results:
        raise ValueError("No DRT results provided to plot.")

    widget = pg.PlotWidget(title=title)
    plot_item = widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", "Frequency (Hz)")
    plot_item.setLabel("left", "γ [Ω]")
    _add_outside_legend(plot_item)

    for i, (label, result) in enumerate(results):
        color = TAB10[i % len(TAB10)]
        data = result.get_drt_data()
        if len(data) == 3:
            tau, gamma_re, gamma_im = data
            freq = 1.0 / (2.0 * math.pi * tau)
            plot_item.plot(freq, gamma_re, pen=pg.mkPen(color, width=1.5), name=f"{label} (Re)")
            plot_item.plot(
                freq, gamma_im,
                pen=pg.mkPen(color, width=1.5, style=Qt.DashLine),
                name=f"{label} (Im)",
            )
        else:
            tau, gamma = data
            freq = 1.0 / (2.0 * math.pi * tau)
            plot_item.plot(freq, gamma, pen=pg.mkPen(color, width=1.5), name=label)

    # High frequency on the left, low frequency on the right, to match the
    # Nyquist plot's orientation.
    plot_item.setLogMode(x=True, y=False)
    plot_item.getViewBox().invertX(True)
    return widget
