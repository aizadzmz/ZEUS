"""Widgets that host PyQtGraph plots (and circuit schematics) inside Qt
layouts."""

from typing import List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import QByteArray, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.plotting import point_tip

_OVERLAY_STYLE = """
QWidget#figureOverlay {
    background: rgba(128, 128, 128, 40%);
    border: 1px solid rgba(128, 128, 128, 60%);
    border-radius: 6px;
}
QToolButton {
    border: none;
    padding: 3px 8px;
    border-radius: 4px;
}
QToolButton:checked {
    background: rgba(0, 0, 0, 35%);
}
"""

# How long the cursor must rest on a point before its metadata box appears.
# PyQtGraph's built-in hoverable tooltip goes through Qt's native QToolTip,
# whose ~700ms delay isn't configurable from here, so PgFigurePane drives its
# own short-delay timer instead (see _on_point_hovered).
HOVER_DELAY_MS = 150


class PgFigurePane(QWidget):
    """Hosts a single PlotWidget, with optional Replot / Auto-Scale overlay
    buttons. Hiding removed points is handled by clicking the "Removed"
    legend swatch (pyqtgraph toggles item visibility there natively), so
    there's no separate button for it here.

    No navigation toolbar is needed -- a PyQtGraph ViewBox is live by default
    (drag to pan, scroll/right-drag to zoom), and the overlay buttons cover the
    reframing that pyqtgraph doesn't do itself. Plots that opt out of mouse
    navigation (core.plotting.build_bode_plot) are hosted just the same; the
    buttons keep working, since they set ranges directly.
    """

    replot_requested = Signal()
    # (dataset key, point index) -- emitted instead of pinning a tooltip when
    # eraser mode is on. The owner decides what masked/unmasked means; the
    # pane only reports which point was hit. Keyed by ds.key rather than
    # ds.label so a click resolves to the right dataset even when several
    # loaded files share the same "Set NN" labels.
    point_mask_toggled = Signal(str, int)

    def __init__(
        self,
        with_overlay_actions: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._widget: Optional[pg.PlotWidget] = None

        # Lives on the pane, not the PlotWidget, because set_widget replaces
        # the widget on every replot -- including the replot that each erase
        # triggers, which would otherwise turn the mode off after one click.
        self._eraser = False

        self._overlay: Optional[QWidget] = None
        if with_overlay_actions:
            self._build_overlay()

        # Remembers the last view (drag/scroll zoom-pan, or the Auto-Scale
        # button) so a replot triggered by unrelated actions (running
        # validation, DRT, etc.) doesn't snap the view back to the default
        # framing.
        self._locked_range: Optional[Tuple[float, float, float, float]] = None
        # Which coordinate system that remembered view belongs to (a plot
        # builder's widget.range_key). Switching the pane between plots that
        # measure different things -- Nyquist's ohms and Bode's log-decades --
        # has to drop it: the numbers would still apply cleanly and land the
        # new plot somewhere far off-screen.
        self._range_key: Optional[str] = None

        # The metadata box shown on hover or click (see _show_tooltip);
        # cleared on click-elsewhere so at most one is ever shown.
        self._tooltip_item: Optional[pg.TextItem] = None

        # True once a click has pinned the box open; hover then leaves it
        # alone until the user clicks empty space (see _on_point_hovered).
        self._pinned = False

        # Hover shows the box only after the cursor rests on a point for
        # HOVER_DELAY_MS, so brief mouse-overs while panning don't flash it.
        # (x, y, text, container) -- see _show_tooltip for the container.
        self._pending_hover: Optional[Tuple[float, float, str, object]] = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(HOVER_DELAY_MS)
        self._hover_timer.timeout.connect(self._show_pending_hover)

    def _build_overlay(self) -> None:
        overlay = QWidget(self)
        overlay.setObjectName("figureOverlay")
        overlay.setStyleSheet(_OVERLAY_STYLE)
        col = QVBoxLayout(overlay)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(2)

        replot_button = QToolButton(overlay)
        replot_button.setText("Replot")
        replot_button.setToolTip("Replot")
        replot_button.clicked.connect(self.replot_requested)
        col.addWidget(replot_button)

        autoscale_button = QToolButton(overlay)
        autoscale_button.setText("Auto-Scale")
        autoscale_button.setToolTip("Zoom to unmasked data")
        autoscale_button.clicked.connect(self._autoscale)
        col.addWidget(autoscale_button)

        overlay.adjustSize()
        self._overlay = overlay
        self._reposition_overlay()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        """Positions the overlay inside the plotted area's bottom-right corner,
        rather than the pane's own -- the pane also contains the axis/tick
        areas around the ViewBox, and the buttons would otherwise sit on top of
        them (the x-axis label, or the Bode plot's right-hand phase ticks)."""
        if self._overlay is None:
            return
        margin = 5
        self._overlay.adjustSize()

        right, bottom = self.width(), self.height()
        if self._widget is not None:
            plot_rect = self._widget.getPlotItem().getViewBox().sceneBoundingRect()
            corner = self._widget.mapFromScene(
                QPointF(plot_rect.right(), plot_rect.bottom())
            )
            mapped = self._widget.mapTo(self, corner)
            right, bottom = mapped.x(), mapped.y()

        x = right - self._overlay.width() - margin
        y = bottom - self._overlay.height() - margin
        self._overlay.move(max(0, x), max(0, y))

    def _on_range_changed(self, view_box, view_range, changed) -> None:
        """Hooked up to the ViewBox's sigRangeChanged, so this fires for
        mouse pan/zoom and the Auto-Scale button alike."""
        (xlo, xhi), (ylo, yhi) = view_range
        self._locked_range = (xlo, xhi, ylo, yhi)

    def _autoscale(self) -> None:
        """Frames the kept (unmasked) points and nothing else -- notably not
        the origin, which the default framing does include. A one-shot action
        rather than a toggle: there's no "off" view to return to, since the
        ViewBox is always live (drag/scroll) and right-click > View All
        already reframes everything drawn.

        Falls back to the default framing when every point is masked out,
        leaving kept_range with nothing to frame."""
        if self._widget is None:
            return
        bounds = self._widget.kept_range or self._widget.full_range
        if bounds is None:
            return
        xlo, xhi, ylo, yhi = bounds
        view_box = self._widget.getPlotItem().getViewBox()
        view_box.setRange(xRange=(xlo, xhi), yRange=(ylo, yhi), padding=0)

    def set_eraser_enabled(self, enabled: bool) -> None:
        """Turn click-to-mask/unmask on or off. While on, clicking a point
        reports it via point_mask_toggled instead of pinning a metadata box;
        hover boxes keep working, so you can confirm which point is under the
        cursor before erasing it."""
        self._eraser = enabled
        if enabled:
            self._pinned = False
            self._hide_tooltip()
        self._apply_eraser_cursor()

    def _apply_eraser_cursor(self) -> None:
        if self._widget is None:
            return
        self._widget.setCursor(Qt.CrossCursor if self._eraser else Qt.ArrowCursor)

    def _on_point_clicked(self, plot, points, ev) -> None:
        """Connected to each series' ScatterPlotItem.sigClicked. Pins a
        metadata box at the clicked point (label/frequency/Z'/Z''), which
        (unlike a hover-shown box) stays up regardless of later mouse
        movement, until a click elsewhere. ev.accept() was already called by
        ScatterPlotItem.mouseClickEvent before this signal fired, which is
        what tells _on_scene_clicked not to treat this as a click-on-empty-
        space."""
        if not len(points):
            return
        self._hover_timer.stop()
        self._pending_hover = None
        point = points[0]

        if self._eraser:
            # No box here: the owner is about to rebuild the whole plot with
            # this point moved between the kept and removed series, and a
            # tooltip pinned to its old position would be stale immediately.
            self._hide_tooltip()
            data = point.data()
            self.point_mask_toggled.emit(data["key"], data["index"])
            return

        x, y = point.pos().x(), point.pos().y()
        self._show_tooltip(x, y, point_tip(point.data()), pinned=True, container=plot.getViewBox())

    def _on_scene_clicked(self, ev) -> None:
        """Connected to the scene's sigMouseClicked, which fires for every
        click. If a point's sigClicked already accepted this event, leave
        its freshly-shown box alone; otherwise the click missed every point,
        so un-pin and hide whatever box is showing."""
        if ev.isAccepted():
            return
        self._pinned = False
        self._hide_tooltip()

    def _on_point_hovered(self, plot, points, ev) -> None:
        """Connected to each series' ScatterPlotItem.sigHovered. A pinned
        (click-shown) box takes priority and isn't disturbed by hovering
        elsewhere; otherwise starts/restarts the short delay before showing
        the box, or hides it once the cursor leaves every point."""
        if self._pinned:
            return
        if len(points):
            point = points[0]
            x, y = point.pos().x(), point.pos().y()
            self._pending_hover = (x, y, point_tip(point.data()), plot.getViewBox())
            self._hover_timer.start()
        else:
            self._hover_timer.stop()
            self._pending_hover = None
            self._hide_tooltip()

    def _show_pending_hover(self) -> None:
        if self._pinned or self._pending_hover is None:
            return
        x, y, text, container = self._pending_hover
        self._show_tooltip(x, y, text, pinned=False, container=container)

    def _show_tooltip(
        self, x: float, y: float, text: str, pinned: bool, container=None
    ) -> None:
        """Show the metadata box for a point at (x, y) in container's
        coordinates -- container being the ViewBox the point was drawn in.

        That isn't always the PlotItem's own ViewBox: the Bode plot's phase
        series lives in a second one with its own y scale (see
        core.plotting.build_bode_plot). The box is still parented to the
        PlotItem either way, so it can't end up painted underneath another
        series, with the position translated through the scene to land on the
        same pixel it would have in the ViewBox it came from."""
        if self._widget is None:
            return
        self._hide_tooltip()
        item = pg.TextItem(
            text,
            color=(20, 20, 20),
            anchor=(0, 1),
            border=pg.mkPen("#808080"),
            fill=pg.mkBrush(255, 255, 255, 230),
        )
        plot_item = self._widget.getPlotItem()
        main_view = plot_item.getViewBox()
        if container is not None and container is not main_view:
            point = main_view.mapSceneToView(container.mapViewToScene(QPointF(x, y)))
            x, y = point.x(), point.y()
        item.setPos(x, y)
        plot_item.addItem(item, ignoreBounds=True)
        self._tooltip_item = item
        self._pinned = pinned

    def _hide_tooltip(self) -> None:
        if self._tooltip_item is not None and self._widget is not None:
            self._widget.getPlotItem().removeItem(self._tooltip_item)
        self._tooltip_item = None

    def set_widget(self, widget: pg.PlotWidget) -> None:
        self.clear()
        self._widget = widget
        self._layout.addWidget(widget)

        range_key = getattr(widget, "range_key", None)
        if range_key != self._range_key:
            self._locked_range = None
            self._range_key = range_key

        view_box = widget.getPlotItem().getViewBox()
        if self._locked_range is not None:
            xlo, xhi, ylo, yhi = self._locked_range
            view_box.setRange(xRange=(xlo, xhi), yRange=(ylo, yhi), padding=0)
        view_box.sigRangeChanged.connect(self._on_range_changed)

        for item in getattr(widget, "interactive_items", []):
            item.sigClicked.connect(self._on_point_clicked)
            item.sigHovered.connect(self._on_point_hovered)
        widget.getPlotItem().scene().sigMouseClicked.connect(self._on_scene_clicked)

        # The mode outlives the widget it was set on (each erase replots), so
        # the fresh widget has to be told about it again.
        self._apply_eraser_cursor()

        if self._overlay is not None:
            self._overlay.raise_()
            # The freshly-added PlotWidget hasn't processed its own resize/
            # layout event yet at this point (that's queued, not synchronous),
            # so its ViewBox can still report stale/default geometry --
            # repositioning right now would visibly snap the buttons to a
            # wrong spot (e.g. mid-plot) for one frame before the deferred
            # call below corrects it. Leave them where they are and only
            # reposition once the event loop has caught up.
            QTimer.singleShot(0, self._reposition_overlay)

    def clear(self) -> None:
        self._hover_timer.stop()
        self._pending_hover = None
        self._pinned = False
        self._tooltip_item = None
        if self._widget is not None:
            self._layout.removeWidget(self._widget)
            self._widget.deleteLater()
            self._widget = None


# How far past its natural size a schematic is allowed to be blown up. It is
# vector art, so it stays crisp at any scale, but a two-element circuit
# stretched across a wide window reads as a poster rather than as a figure.
MAX_DIAGRAM_SCALE = 1.8


class SvgFigure(QWidget):
    """One SVG drawn at the width it is told to use, keeping its aspect ratio.

    Height is set explicitly (set_display_width) rather than reported through
    heightForWidth: the diagrams live in a QScrollArea, and a scroll area
    sizes its widget from plain size hints without consulting
    heightForWidth, so a column of aspect-ratio-only figures gets squeezed
    into the viewport and painted on top of each other. A fixed height per
    figure is what makes the column add up to something the scroll area can
    scroll.
    """

    def __init__(self, svg: bytes, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._renderer = QSvgRenderer(QByteArray(svg))
        natural = self._renderer.defaultSize()
        # 1x1 rather than 0x0 so the aspect divisions below are always safe,
        # which is what an unparseable SVG would otherwise leave here.
        self._natural = natural if natural.width() > 0 and natural.height() > 0 else QSize(1, 1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.set_display_width(self._natural.width())

    def set_display_width(self, width: int) -> None:
        """Draw at `width` px (clamped to the legible range) and take exactly
        the height that implies."""
        self._drawn_width = max(
            self._natural.width() // 2,
            min(int(width), int(self._natural.width() * MAX_DIAGRAM_SCALE)),
        )
        self.setFixedHeight(
            round(self._drawn_width * self._natural.height() / self._natural.width())
        )
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Centered, so a narrow circuit in a wide pane doesn't sit off to one
        # side of its caption.
        painter.translate(max(0, (self.width() - self._drawn_width) / 2), 0)
        self._renderer.render(painter, QRectF(0, 0, self._drawn_width, self.height()))
        painter.end()


class CircuitDiagramPane(QScrollArea):
    """A scrollable column of captioned circuit schematics, used by the ECM
    Parameters tab. Also shows a plain message on its own (set_message) for
    the states that have no diagram to draw: nothing fitted yet, or a circuit
    description code that doesn't parse."""

    MARGIN = 12

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(self.MARGIN, 8, self.MARGIN, 8)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self.setWidget(self._container)
        self._items: List[QWidget] = []
        self._figures: List[SvgFigure] = []

    def set_diagrams(self, diagrams: List[Tuple[str, bytes]]) -> None:
        """Replace the contents with (caption, svg) pairs, in order."""
        self.clear()
        for caption, svg in diagrams:
            label = QLabel(caption)
            label.setStyleSheet("font-weight: 600; padding-top: 6px;")
            label.setWordWrap(True)
            self._add(label)
            figure = SvgFigure(svg)
            self._figures.append(figure)
            self._add(figure)
        self._resize_figures()

    def set_message(self, text: str) -> None:
        self.clear()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("padding: 24px;")
        self._add(label)

    def add_note(self, text: str) -> None:
        """Append a line of plain text under the diagrams already set, for
        saying what was left undrawn."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("padding-top: 12px;")
        self._add(label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_figures()

    def _resize_figures(self) -> None:
        width = self.viewport().width() - 2 * self.MARGIN
        for figure in self._figures:
            figure.set_display_width(width)

    def _add(self, widget: QWidget) -> None:
        # insert above the trailing stretch, which keeps a short column of
        # diagrams pinned to the top instead of spread down the pane
        self._layout.insertWidget(self._layout.count() - 1, widget)
        self._items.append(widget)

    def clear(self) -> None:
        for widget in self._items:
            self._layout.removeWidget(widget)
            # Unparented as well as delisted: deleteLater only schedules the
            # destruction, and until the event loop gets to it a widget that
            # is merely out of the layout keeps its parent -- and goes on
            # painting itself where it last sat, under the replacements.
            widget.setParent(None)
            widget.deleteLater()
        self._items = []
        self._figures = []


class PgFigureListPane(QScrollArea):
    """A scrollable vertical stack of PlotWidgets (one per figure), used by
    the Residuals tab. Each PlotWidget is independently interactive (drag to
    pan, scroll to zoom) with no toolbar needed, same as PgFigurePane."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.addStretch()
        self.setWidget(self._container)
        self._widgets: List[pg.PlotWidget] = []

    def set_widgets(self, widgets: List[pg.PlotWidget]) -> None:
        self.clear()
        for widget in widgets:
            widget.setMinimumHeight(340)
            # insert above the trailing stretch
            self._layout.insertWidget(self._layout.count() - 1, widget)
            self._widgets.append(widget)

    def clear(self) -> None:
        for widget in self._widgets:
            self._layout.removeWidget(widget)
            widget.deleteLater()
        self._widgets = []
