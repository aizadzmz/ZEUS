# PyQtGraph Nyquist plot -- replaces the matplotlib version in
# core/plotting.py (plot_single/plot_overlay) for the GUI's Nyquist tab.
# Kept separate from plotting.py since the Streamlit app and any scripted/
# saved-figure use still go through matplotlib.
import math
from typing import List, Optional, Tuple

import pyqtgraph as pg

from core.io_utils import EISDataset
from core.plotting import ORIGIN_COLOR, ORIGIN_WIDTH, equal_aspect_limits

# Matches matplotlib's default tab10 cycle (C0, C1, ...) so datasets keep the
# same colors as the existing plots.
_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_REMOVED_COLOR = "#999999"  # matplotlib's gray "0.6", as a hex value pyqtgraph accepts

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


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


def point_tip(x, y, data) -> str:
    suffix = " (removed)" if data["removed"] else ""
    return (
        f"Set: {data['label']}{suffix}\nFreq: {data['freq']:.4g} Hz\n"
        f"Z': {x:.4g} Ω\nZ'': {-y:.4g} Ω"
    )


def _point_data(label: str, freq, indices, removed: bool = False):
    """Per-point payload attached to every ScatterPlotItem. Carries the
    dataset label and the point's index *in the unmasked-inclusive array*,
    which is what the eraser needs to toggle a mask entry -- the kept series
    is built from unmasked points only, so scatter position != index.

    'removed' stays a flag rather than being folded into the label, which
    consumers key datasets by."""
    return [
        {"label": label, "freq": f, "index": int(i), "removed": removed}
        for f, i in zip(freq, indices)
    ]


def _split_indices(ds: EISDataset):
    """(kept, removed) point indices for a dataset, in the same order as
    get_impedances(masked=False) / get_impedances(masked=True) return them."""
    mask = ds.data.get_mask()
    kept, removed = [], []
    for i in range(ds.data.get_num_points(masked=None)):
        (removed if mask.get(i, False) else kept).append(i)
    return kept, removed


def _add_kept_series(plot_item, x, y, tip_data, label: str, color: str, style: str):
    """Draw one dataset's kept points: a filled scatter for 'scatter' style,
    or a connected line plus an invisible hoverable scatter (for tooltips)
    for 'line' style. Returns the hoverable ScatterPlotItem (used for
    hit-testing, range calculation, and role-tagging)."""
    if style == "line":
        # The line itself carries the legend entry (so the swatch shows a
        # colored line, not a marker); the scatter stays unnamed and
        # invisible, existing only as a hover/tooltip hit-target.
        plot_item.addItem(pg.PlotDataItem(x=x, y=y, pen=pg.mkPen(color, width=1.5), name=label))
        scatter_name = None
        marker_kwargs = dict(brush=None, pen=None, size=10)
    elif style == "scatter":
        scatter_name = label
        marker_kwargs = dict(brush=pg.mkBrush(color), pen=None, size=9)
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
    plot_item.addItem(scatter)
    return scatter


def _add_removed_series(plot_item, x, y, tip_data, legend_name: Optional[str]):
    """Draw masked-out points as muted 'x' markers, matching
    core.plotting._plot_removed. Returns the ScatterPlotItem, or None if
    there are no removed points."""
    if x.size == 0:
        return None

    scatter = pg.ScatterPlotItem(
        x=x, y=y,
        name=legend_name,
        symbol="x",
        pen=pg.mkPen(_REMOVED_COLOR, width=1.3),
        brush=None,
        size=9,
        hoverable=True,
        data=tip_data,
    )
    scatter._eis_role = "removed"
    plot_item.addItem(scatter)
    return scatter


def _bounds(
    xs: List[float], ys: List[float], *, include_origin: bool = True
) -> Optional[Tuple[float, float, float, float]]:
    """Framing for the given points, or None when there are none to frame
    (callers fall back rather than setting a degenerate zero-span range).

    include_origin=False frames the data alone -- what the Auto-Scale button
    wants, so a spectrum sitting far from 0 fills the view instead of being
    squashed into a corner by an origin nobody asked to see."""
    if not xs:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Breathing room around the tight data box, so a point sitting right at
    # an extreme doesn't have its marker sliced by the ViewBox edge (padding=0
    # below draws the range exactly) -- mirrors FigurePane._zoom_to_kept_data's
    # margin for the matplotlib Auto-Scale button.
    margin = 0.05
    x_pad = (xmax - xmin) * margin or 1e-6
    y_pad = (ymax - ymin) * margin or 1e-6
    return equal_aspect_limits(
        xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad,
        include_origin=include_origin,
    )


def build_nyquist_plot(
    datasets: List[EISDataset],
    title: str = "Nyquist Plot",
    style: str = "scatter",
    show_removed: bool = False,
) -> pg.PlotWidget:
    """Equal-aspect Nyquist overlay -- the PyQtGraph equivalent of
    core.plotting.plot_single/plot_overlay. Pass a single-item list for the
    "Single" mode; multiple items overlay with one legend entry each.

    The returned widget carries extra attributes consumed by
    gui.figure_panes.PgFigurePane:
      - interactive_items : every hoverable ScatterPlotItem (kept and removed),
        for wiring up the hover/click-to-show-metadata tooltip.
      - kept_range / full_range : (xlo, xhi, ylo, yhi) tuples, or None when
        there was nothing to frame. kept_range is what the Auto-Scale button
        zooms to -- the kept points alone, origin excluded. full_range is the
        default framing applied here: it covers removed points too, so
        nothing drawn is cropped out, and keeps the origin in view.

    Hiding removed points is handled by pyqtgraph's own legend -- clicking a
    legend swatch toggles that item's visibility -- so there's no dedicated
    toggle here.
    """
    if not datasets:
        raise ValueError("No datasets provided to plot.")

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

    plot_item.addLegend(offset=(10, 10))

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

    removed_legend_name = "Removed"
    for i, ds in enumerate(datasets):
        color = _TAB10[i % len(_TAB10)]
        kept_idx, removed_idx = _split_indices(ds)
        Z = ds.impedances  # kept (unmasked) points only
        x, y = Z.real, -Z.imag
        kept_item = _add_kept_series(
            plot_item, x, y, _point_data(ds.label, ds.frequencies, kept_idx),
            ds.label, color, style,
        )
        interactive_items.append(kept_item)
        kept_xs.extend(x); kept_ys.extend(y)
        all_xs.extend(x); all_ys.extend(y)

        if show_removed:
            Zr = ds.data.get_impedances(masked=True)
            fr = ds.data.get_frequencies(masked=True)
            xr, yr = Zr.real, -Zr.imag
            item = _add_removed_series(
                plot_item, xr, yr,
                _point_data(ds.label, fr, removed_idx, removed=True),
                removed_legend_name,
            )
            if item is not None:
                interactive_items.append(item)
                removed_legend_name = None  # only the first gets a legend entry
                all_xs.extend(xr); all_ys.extend(yr)

    widget.interactive_items = interactive_items
    widget.kept_range = _bounds(kept_xs, kept_ys, include_origin=False)
    widget.full_range = _bounds(all_xs, all_ys)

    if widget.full_range is not None:
        xlo, xhi, ylo, yhi = widget.full_range
        plot_item.setXRange(xlo, xhi, padding=0)
        plot_item.setYRange(ylo, yhi, padding=0)

    return widget
