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
from gui import style

# The overlay card's look lives in gui.style's QSS block, keyed off the
# objectName set in _build_overlay.

# How long the cursor must rest on a point before its metadata box appears.
# PyQtGraph's own hover tooltip uses QToolTip, whose ~700ms delay is not
# configurable here, so this pane runs its own timer (see _on_point_hovered).
HOVER_DELAY_MS = 150


class PgFigurePane(QWidget):
    """Hosts a single PlotWidget with optional Replot / Auto-Scale overlay
    buttons. Removed points hide via the "Removed" legend swatch."""

    replot_requested = Signal()
    # "Save Image" was clicked. The owner picks the path and calls
    # save_image(); see _build_overlay.
    save_image_requested = Signal()
    # (ds.key, point index) -- emitted instead of pinning a tooltip when the
    # eraser is on. The pane only reports which point was hit; the owner
    # decides what that means. Keyed by ds.key so a click resolves correctly
    # when several files share "Set NN" labels.
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
        # Shown instead of a plot when there is nothing to draw; see
        # set_message.
        self._message: Optional[QLabel] = None

        # On the pane, not the PlotWidget: set_widget replaces the widget on
        # every replot, including the one each erase triggers, which would
        # otherwise turn the mode off after a single click.
        self._eraser = False

        self._overlay: Optional[QWidget] = None
        if with_overlay_actions:
            self._build_overlay()

        # The last view (pan/zoom or Auto-Scale), so an unrelated replot --
        # running validation, DRT, etc. -- does not snap back to the default
        # framing.
        self._locked_range: Optional[Tuple[float, float, float, float]] = None
        # Which coordinate system that view belongs to (widget.range_key).
        # Dropped when switching between plots measuring different things
        # (Nyquist ohms vs Bode log-decades), where the numbers would still
        # apply and land the new plot off-screen.
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
        # Scopes the stylesheet's card and button rules to this overlay --
        # unscoped they would restyle every QToolButton in the app.
        overlay.setObjectName("figureOverlay")
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

        save_button = QToolButton(overlay)
        save_button.setText("Save Image")
        save_button.setToolTip("Save this plot as a PNG or SVG")
        # Signalled, not handled here: this pane owns no dialogs.
        save_button.clicked.connect(self.save_image_requested)
        col.addWidget(save_button)

        overlay.adjustSize()
        self._overlay = overlay
        self._reposition_overlay()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        """Position the overlay in the plotted area's bottom-right corner, not
        the pane's, which also holds the axis and tick areas."""
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
        """Hooked up to the ViewBox's sigRangeChanged, so this fires for mouse
        pan/zoom and the Auto-Scale button alike."""
        (xlo, xhi), (ylo, yhi) = view_range
        self._locked_range = (xlo, xhi, ylo, yhi)

    def _autoscale(self) -> None:
        """Frame the kept (unmasked) points only, excluding the origin. Falls
        back to the default framing when every point is masked."""
        if self._widget is None:
            return
        bounds = self._widget.kept_range or self._widget.full_range
        if bounds is None:
            return
        xlo, xhi, ylo, yhi = bounds
        view_box = self._widget.getPlotItem().getViewBox()
        view_box.setRange(xRange=(xlo, xhi), yRange=(ylo, yhi), padding=0)

    def save_image(self, path: str) -> None:
        """Write the hosted plot to path; the extension picks SVG (vector) or
        raster. Raises RuntimeError when there is no plot to save."""
        if self._widget is None:
            raise RuntimeError("There is no plot to save yet.")

        plot_item = self._widget.getPlotItem()
        if path.lower().endswith(".svg"):
            from pyqtgraph.exporters import SVGExporter

            SVGExporter(plot_item).export(path)
        else:
            from pyqtgraph.exporters import ImageExporter

            ImageExporter(plot_item).export(path)

    def view_state(self) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
        """This pane's framing, as (coordinate system, range), or None if never
        panned or zoomed. Paired with set_view_state."""
        if self._range_key is None or self._locked_range is None:
            return None
        return (self._range_key, self._locked_range)

    def set_view_state(
        self, state: Optional[Tuple[str, Tuple[float, float, float, float]]]
    ) -> None:
        """Adopt a framing captured from another pane by view_state(); the
        coordinate system travels with it and is checked on the way in."""
        if state is None:
            return
        range_key, locked = state
        self._range_key = range_key
        self._locked_range = locked

    def set_eraser_enabled(self, enabled: bool) -> None:
        """Turn click-to-mask/unmask on or off. While on, clicking a point
        reports it via point_mask_toggled instead of pinning a box."""
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
        metadata box at the clicked point until a click elsewhere."""
        if not len(points):
            return
        self._hover_timer.stop()
        self._pending_hover = None
        point = points[0]

        if self._eraser:
            # No box: the owner is about to replot with this point moved
            # between the kept and removed series, so a pinned tooltip would be
            # stale immediately.
            self._hide_tooltip()
            data = point.data()
            self.point_mask_toggled.emit(data["key"], data["index"])
            return

        x, y = point.pos().x(), point.pos().y()
        self._show_tooltip(x, y, point_tip(point.data()), pinned=True, container=plot.getViewBox())

    def _on_scene_clicked(self, ev) -> None:
        """Connected to the scene's sigMouseClicked; un-pins and hides the box
        when a click misses every point."""
        if ev.isAccepted():
            return
        self._pinned = False
        self._hide_tooltip()

    def _on_point_hovered(self, plot, points, ev) -> None:
        """Connected to each series' ScatterPlotItem.sigHovered. A pinned box
        takes priority; otherwise this starts the hover delay."""
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
        coordinates, translated through the scene onto the PlotItem."""
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

        # The eraser mode outlives the widget it was set on, so tell the fresh
        # widget about it again.
        self._apply_eraser_cursor()

        if self._overlay is not None:
            # Shown again in case set_message hid it.
            self._overlay.show()
            self._overlay.raise_()
            # Deferred: the new PlotWidget has not processed its layout event
            # yet, so its ViewBox would report stale geometry and the buttons
            # would snap to the wrong spot for one frame.
            QTimer.singleShot(0, self._reposition_overlay)

    def set_message(self, text: str) -> None:
        """Replace the plot with a line of text saying why there isn't one (see
        MainWindow._show_empty_state)."""
        self.clear()
        self._message = QLabel(text)
        self._message.setWordWrap(True)
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setObjectName("figureMessage")
        self._layout.addWidget(self._message)
        # Replot/Auto-Scale/Save Image all act on a plot that isn't there.
        if self._overlay is not None:
            self._overlay.hide()

    def clear(self) -> None:
        self._hover_timer.stop()
        self._pending_hover = None
        self._pinned = False
        self._tooltip_item = None
        if self._message is not None:
            self._layout.removeWidget(self._message)
            self._message.deleteLater()
            self._message = None
        if self._widget is not None:
            self._layout.removeWidget(self._widget)
            self._widget.deleteLater()
            self._widget = None


# How far past its natural size a schematic may be scaled up. Vector art stays
# crisp, but a two-element circuit across a wide window reads as a poster.
MAX_DIAGRAM_SCALE = 1.8


class SvgFigure(QWidget):
    """One SVG drawn at a given width, keeping its aspect ratio."""

    def __init__(self, svg: bytes, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._renderer = QSvgRenderer(QByteArray(svg))
        natural = self._renderer.defaultSize()
        # 1x1, not 0x0, so the aspect divisions below stay safe when the SVG
        # does not parse.
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
        # Centered, so a narrow circuit does not sit off to one side of its
        # caption in a wide pane.
        painter.translate(max(0, (self.width() - self._drawn_width) / 2), 0)
        self._renderer.render(painter, QRectF(0, 0, self._drawn_width, self.height()))
        painter.end()


class CircuitDiagramPane(QScrollArea):
    """A scrollable column of captioned circuit schematics for the ECM
    Parameters tab; set_message covers the states with no diagram."""

    MARGIN = 12

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(self.MARGIN, 8, self.MARGIN, 8)
        self._layout.setSpacing(style.LIST_SPACING)
        self._layout.addStretch()
        self.setWidget(self._container)
        self._items: List[QWidget] = []
        self._figures: List[SvgFigure] = []

    def set_diagrams(self, diagrams: List[Tuple[str, bytes]]) -> None:
        """Replace the contents with (caption, svg) pairs, in order."""
        self.clear()
        for caption, svg in diagrams:
            label = QLabel(caption)
            label.setObjectName("diagramCaption")
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
        label.setObjectName("figureMessage")
        self._add(label)

    def add_note(self, text: str) -> None:
        """Append a line of plain text under the diagrams already set, for
        saying what was left undrawn."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("figureNote")
        self._add(label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_figures()

    def _resize_figures(self) -> None:
        width = self.viewport().width() - 2 * self.MARGIN
        for figure in self._figures:
            figure.set_display_width(width)

    def _add(self, widget: QWidget) -> None:
        # Above the trailing stretch, so a short column stays pinned to the top.
        self._layout.insertWidget(self._layout.count() - 1, widget)
        self._items.append(widget)

    def clear(self) -> None:
        for widget in self._items:
            self._layout.removeWidget(widget)
            # Unparent as well: deleteLater only schedules destruction, and
            # until then a widget merely out of the layout keeps its parent and
            # goes on painting itself under the replacements.
            widget.setParent(None)
            widget.deleteLater()
        self._items = []
        self._figures = []


class PgFigureListPane(QScrollArea):
    """A scrollable vertical stack of PlotWidgets, one per figure, used by the
    Residuals tab."""

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
