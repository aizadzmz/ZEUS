"""PyQtGraph plot builders for the desktop GUI: Nyquist, Bode, Residuals, and
DRT."""
from __future__ import annotations

import math
from html import escape
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSizeF, Qt
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

# The app's color cycle (matplotlib's tab10).
TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

_REMOVED_COLOR = "#999999"

# Which Bode series belongs to which axis. Sits under the title rather than in
# the legend, which is per-sweep.
BODE_SUBTITLE = "|Z| ●   ,   -Φ ○"

# Title-row height in px for the two-line Bode title (PlotItem.setTitle sizes
# for one line). Raise it if the subtitle wraps or the font grows.
BODE_TITLE_HEIGHT = 48

# PyQtGraph symbols cycled by loaded file. "x" is excluded -- it marks removed
# points (see _add_removed_series).
PG_MARKERS = ("o", "s", "t", "d", "p", "h", "star", "t1", "t2", "t3")

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
    payload rather than the plotted position."""
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


def _marker_kwargs(color: str, symbol: Optional[str], hollow: bool) -> dict:
    """ScatterPlotItem styling for one series: filled in the dataset's color,
    or outlined when hollow. symbol=None leaves pyqtgraph's default ('o')."""
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
    """Draw one dataset's kept points and return the hoverable ScatterPlotItem.
    hollow outlines the markers for the Bode phase series."""
    if style == "line":
        # The line carries the legend entry, so the swatch shows a colored
        # line rather than a marker; the scatter stays unnamed.
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
    """Draw masked-out points as muted 'x' markers. Returns the
    ScatterPlotItem, or None if there are no removed points."""
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
    """Draw a dataset's fitted circuit response as a dashed line. Never hit-
    tested, and excluded from the range calculation."""
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


class _WrappingLegend(pg.LegendItem):
    """A legend that sits beside the plot rather than on top of it, and adds
    columns rather than height when it runs out of room."""

    def __init__(self, **kwargs):
        super().__init__(offset=None, **kwargs)
        # No backing plate or border: outside the plot there is nothing to
        # mask, and a box would just add a second frame beside the axes.
        self.setBrush(pg.mkBrush(0, 0, 0, 0))
        self.setPen(pg.mkPen(0, 0, 0, 0))
        # Fixed vertically so the grid gives it its content height. Stretching
        # separates each swatch (centred in its row) from its label (drawn at
        # the top of one).
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._wrapping = False

    def _row_height(self) -> float:
        """Height one entry wants -- the label's preferred size hint, not its
        bounding rect."""
        sample, label = self.items[0]
        return max(
            label.sizeHint(Qt.SizeHint.PreferredSize, QSizeF()).height(),
            sample.boundingRect().height(),
        )

    def wrap_to_height(self, available: float) -> None:
        """Re-column so the entries need no more than `available` px of height."""
        if self._wrapping or not self.items or available <= 0:
            return
        row_height = self._row_height()
        if row_height <= 0:
            return
        rows = max(1, int(available // row_height))
        columns = max(1, math.ceil(len(self.items) / rows))
        self._wrapping = True
        try:
            # Cap the height as well as re-columning: the grid sizes to its
            # contents and PlotWidget only re-fits on a resize event, so a
            # too-long legend would stay stretched past the widget.
            self.setMaximumHeight(available)
            if columns != self.columnCount:
                self.setColumnCount(columns)
        finally:
            self._wrapping = False


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


def _add_outside_legend(plot_item: pg.PlotItem, **kwargs) -> _WrappingLegend:
    """Attach a _WrappingLegend in a grid column of its own, to the right of
    the plot, and keep it wrapped as the pane is resized."""
    legend = _WrappingLegend(labelTextSize="9pt", **kwargs)
    plot_item.legend = legend
    plot_item.layout.addItem(legend, 2, 3)
    # Top-aligned so the legend keeps its content height. Otherwise it stretches
    # to the plot row and the slack separates each swatch from its label.
    plot_item.layout.setAlignment(legend, Qt.AlignmentFlag.AlignTop)

    refitting = False

    def rewrap():
        # Budget against the host widget's height, which nothing here changes,
        # so the wrap converges instead of chasing its own effect.
        nonlocal refitting
        view = plot_item.getViewWidget()
        if view is None or refitting:
            return
        legend.wrap_to_height(view.height() - _plot_chrome_height(plot_item))
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
    style_map: Optional[Dict[str, Tuple[str, str]]] = None,
    fit_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
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
    # Padding so nothing lands exactly on the boundary: AxisItem draws its
    # label a few px past its laid-out row, and the bare top/right border axes
    # below have ~0 height, putting their line right at the edge.
    plot_item.layout.setContentsMargins(1, 6, 12, 12)  # left, top, right, bottom
    plot_item.setAspectLocked(True)
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", _axis_label("Z'", exponent))
    plot_item.setLabel("left", _axis_label("-Z''", exponent))

    # Top/right shown bare (no ticks or labels) purely to close the plot box,
    # matching the bottom/left border.
    plot_item.showAxis("top")
    plot_item.showAxis("right")
    for side in ("top", "right"):
        border_axis = plot_item.getAxis(side)
        border_axis.setStyle(showValues=False, tickLength=0)
        border_axis.setLabel(None)

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

    # The first sweep's removed points carry the one shared "Removed" legend
    # entry, registered after the loop so it sorts behind every sweep. Fits
    # work the same way.
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

    # Static view (see the docstring): no pan, zoom, or right-click menu --
    # its "View All" would reframe only the magnitude ViewBox.
    main_view = plot_item.getViewBox()
    main_view.setMouseEnabled(x=False, y=False)
    main_view.setMenuEnabled(False)

    phase_view = pg.ViewBox(enableMenu=False)
    phase_view.setMouseEnabled(x=False, y=False)
    # Below the magnitude ViewBox (at -100): pyqtgraph offers drags in
    # descending z-order and a ViewBox accepts every one, so an overlay on top
    # would consume them. Moot while the mouse is disabled, but it keeps the
    # click dispatch order correct if this plot is ever made interactive.
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
    # so "Fit" and "Removed" read last in the legend.
    removed_item = None
    fit_item = None
    for i, ds in enumerate(datasets):
        # Only the color half of style_map is used: shape already distinguishes
        # magnitude from phase, so files are told apart by color here.
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
        # -phase, matching the Nyquist plot's -Z'': both read "up is more
        # capacitive".
        phase = -np.angle(Z, deg=True)
        tip_data = _point_data(
            ds.key, legend_label, freq, kept_idx, Z, values=_bode_tip_values
        )

        interactive_items.append(_add_kept_series(
            plot_item, x, magnitude, tip_data, legend_label, color, style, symbol="o",
        ))
        # Same tip data on both series, so hovering either marker describes the
        # whole point.
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

    return widget


def build_residuals_plot(
    result,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
) -> pg.PlotWidget:
    """Relative residuals (ΔZ'/|Z| and ΔZ''/|Z|, in percent) of a validation
    result (Kramers-Kronig or Z-HIT) against frequency, log-x."""
    freq, res_re, res_im = result.get_residuals_data()

    widget = pg.PlotWidget(title=title)
    plot_item = widget.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.3)
    plot_item.setLabel("bottom", "Frequency (Hz)")
    plot_item.setLabel("left", "Relative residual (%)")
    legend = _add_outside_legend(plot_item)

    # Plain percent values instead of pyqtgraph's SI-prefix axis scaling, which
    # would label the axis "0.001" and show values as a multiplier of it.
    plot_item.getAxis("left").enableAutoSIPrefix(False)
    plot_item.getAxis("bottom").enableAutoSIPrefix(False)

    # Static view: read, not navigated -- no pan, zoom, or right-click menu.
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

    # Added before setLogMode below, which skips items lacking setLogMode --
    # InfiniteLine has none, so it needs no transform of its own.
    zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#999999", width=0.8))
    zero_line.setZValue(-1)
    plot_item.addItem(zero_line, ignoreBounds=True)

    y_extent = [float(v) for v in res_re] + [float(v) for v in res_im]

    if threshold is not None and len(freq):
        # PlotDataItem, not InfiniteLine: the legend swatch painter reads
        # item.opts unconditionally, which InfiniteLine lacks. The resulting
        # AttributeError inside a Qt paint() override aborts the process.
        x_bounds = [float(freq.min()), float(freq.max())]
        thr_pen = pg.mkPen("orange", width=1.5, style=Qt.DashLine)
        top_line = pg.PlotDataItem(x=x_bounds, y=[threshold, threshold], pen=thr_pen)
        bottom_line = pg.PlotDataItem(x=x_bounds, y=[-threshold, -threshold], pen=thr_pen)
        plot_item.addItem(top_line, ignoreBounds=True)
        plot_item.addItem(bottom_line, ignoreBounds=True)
        legend.addItem(top_line, f"±{threshold}% threshold")
        # ignoreBounds keeps these out of autorange, so an off-scale threshold
        # cannot blow out the view; fold it into the fixed range instead so it
        # stays visible.
        y_extent += [threshold, -threshold]

    # Must precede the fixed Y range below: updateLogMode re-triggers its own
    # autorange, which would clobber an earlier setYRange.
    plot_item.setLogMode(x=True, y=False)

    # Fixed range rather than autorange: the view is static (see
    # setMouseEnabled above), so a clipped plot could not be recovered.
    y_max = max((abs(v) for v in y_extent), default=1.0) * 1.15
    plot_item.setYRange(-y_max, y_max, padding=0)

    return widget


def build_drt_plot(results: List[Tuple[str, object]], title: str = "DRT") -> pg.PlotWidget:
    """Gamma vs frequency for one or more (label, result) pairs, log-x, high
    frequency on the left."""
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


def build_drt_peaks_plot(
    results: List[Tuple[str, object]],
    title: str = "Peak Deconvolution",
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
    plot_item.setLabel("bottom", "Frequency (Hz)")
    plot_item.setLabel("left", "γ [Ω]")
    _add_outside_legend(plot_item)

    for i, (label, peaks) in enumerate(results):
        if peaks.get_num_peaks() < 1:
            continue
        color = TAB10[i % len(TAB10)]
        tau = peaks.get_time_constants(num_per_decade=num_per_decade)
        freq = 1.0 / (2.0 * math.pi * tau)

        if show_individual_peaks:
            # Before the sum, so the heavier total line draws over them.
            for index in range(peaks.get_num_peaks()):
                plot_item.plot(
                    freq,
                    peaks.get_gammas(peak_indices=[index], num_per_decade=num_per_decade),
                    pen=pg.mkPen(color, width=1.0, style=Qt.DashLine),
                )

        plot_item.plot(
            freq,
            peaks.get_gammas(num_per_decade=num_per_decade),
            pen=pg.mkPen(color, width=1.5),
            name=f"{label} ({peaks.get_num_peaks()} peak(s))",
        )

    plot_item.setLogMode(x=True, y=False)
    plot_item.getViewBox().invertX(True)
    return widget
