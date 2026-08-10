"""Widgets that host PyQtGraph plots (and circuit schematics) inside Qt
layouts."""

from typing import List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.plotting import point_tip

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
    # The overlay's Eraser button was clicked. As with point_mask_toggled the
    # pane only reports it: the owner arms every eraser-capable pane together
    # and locks the rest of the window.
    eraser_toggled = Signal(bool)

    def __init__(
        self,
        with_overlay_actions: bool = True,
        with_eraser: bool = False,
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
        self._eraser_button: Optional[QToolButton] = None

        self._overlay: Optional[QWidget] = None
        # The card's buttons and the chevron that folds them away. Collapsed
        # state lives on the pane, not the plot, so it survives every replot.
        self._action_buttons: List[QToolButton] = []
        self._collapse_button: Optional[QToolButton] = None
        if with_overlay_actions:
            self._build_overlay(with_eraser)

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
        # cleared on click-elsewhere so at most one is ever shown. Its point is
        # kept alongside it so the box can be re-fitted to the plot when the
        # view moves under it.
        self._tooltip_item: Optional[pg.TextItem] = None
        self._tooltip_anchor: Optional[Tuple[float, float]] = None

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

    def _build_overlay(self, with_eraser: bool) -> None:
        overlay = QWidget(self)
        # Scopes the stylesheet's card and button rules to this overlay --
        # unscoped they would restyle every QToolButton in the app.
        overlay.setObjectName("figureOverlay")
        col = QVBoxLayout(overlay)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(2)

        if with_eraser:
            # On the plot rather than in the settings column: the eraser locks
            # that column while it is on, so this has to be the way back out.
            self._eraser_button = QToolButton(overlay)
            self._eraser_button.setText("Eraser")
            self._eraser_button.setCheckable(True)
            self._eraser_button.setToolTip(
                "Click a point on this plot to remove it, or a removed (grey ×) "
                "point to restore it. On the Bode plot each removed point is "
                "marked on both the |Z| and the -Φ series. Manual edits override "
                "the inductive filter and the outlier threshold, and are cleared "
                "when a different file is opened.\n\n"
                "While the eraser is on, the rest of the window is locked — "
                "switch it off to carry on.\n\n"
                "Hiding the 'Removed' series via its legend entry also stops "
                "those points responding to clicks."
            )
            self._eraser_button.toggled.connect(self.eraser_toggled)
            col.addWidget(self._eraser_button)

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

        # Everything the card folds away. The eraser is in here too: on a Bode
        # plot the card covers the low-frequency end of both series, and half a
        # card is no less in the way than a whole one.
        self._action_buttons = [
            b for b in (self._eraser_button, replot_button, autoscale_button, save_button)
            if b is not None
        ]

        # Stretched to the widest label rather than each sizing to its own
        # text, which left the stack with a ragged right edge.
        for button in overlay.findChildren(QToolButton):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # A sibling of the card, not a child: the card belongs over the plot,
        # but the chevron is what stays on screen once the card is folded, so
        # it lives *outside* the plotted area -- down in the x-axis strip,
        # where it covers no data at all.
        self._collapse_button = QToolButton(self)
        self._collapse_button.setObjectName("figureOverlayCollapse")
        self._collapse_button.setCheckable(True)
        self._collapse_button.toggled.connect(self._set_overlay_collapsed)

        self._overlay = overlay
        self._set_overlay_collapsed(False)

    def _set_overlay_collapsed(self, collapsed: bool) -> None:
        """Fold the action card away, or open it again in its usual corner."""
        self._overlay.setVisible(not collapsed and self._widget is not None)
        # Chevrons rather than a label: the direction matches where the card
        # goes, since it opens upwards into the plot from here.
        self._collapse_button.setText("⌃" if collapsed else "⌄")
        self._collapse_button.setToolTip(
            "Show the plot buttons" if collapsed else "Hide the plot buttons"
        )
        self._reposition_overlay()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        """Put the card in the plotted area's bottom-right corner -- not the
        pane's, which also holds the axis, tick and legend areas -- and the
        chevron below the plot entirely, in the x-axis strip."""
        if self._overlay is None:
            return
        margin = 5
        self._overlay.adjustSize()

        left, top = 0, 0
        right, bottom = self.width(), self.height()
        if self._widget is not None:
            plot_rect = self._widget.getPlotItem().getViewBox().sceneBoundingRect()
            corners = [
                self._widget.mapTo(
                    self, self._widget.mapFromScene(QPointF(x, y))
                )
                for x, y in (
                    (plot_rect.left(), plot_rect.top()),
                    (plot_rect.right(), plot_rect.bottom()),
                )
            ]
            left, top = corners[0].x(), corners[0].y()
            right, bottom = corners[1].x(), corners[1].y()

        self._reposition_collapse_button(right, bottom)

        # Clamped to the plot area's top-left as well as its bottom-right: a
        # multi-file legend takes its width from the plot, and once the stack
        # no longer fits the corner it should overflow inwards, over the plot,
        # rather than out into the legend column.
        x = max(left, right - self._overlay.width() - margin)
        y = max(top, bottom - self._overlay.height() - margin)
        self._overlay.move(max(0, int(x)), max(0, int(y)))

    def _reposition_collapse_button(self, plot_right: int, plot_bottom: int) -> None:
        """Sit the chevron in the strip below the plot, right-aligned with it.

        Outside the plotted area on purpose -- folded, this is the only thing
        left, and the whole point was to stop it covering data. It is pinned to
        the bottom of the pane so it clears the tick labels, and right-aligned
        with the plot rather than the pane so it stays clear of the legend
        column and lands under the end of the x-axis.
        """
        if self._collapse_button is None:
            return
        self._collapse_button.setVisible(self._widget is not None)
        self._collapse_button.adjustSize()
        size = self._collapse_button.size()

        x = max(0, int(plot_right) - size.width())
        # Below the plot, and as low in the pane as it will go. The max() is
        # for a pane too short to have a strip: the chevron then sits just
        # under the plot edge rather than climbing back over the data.
        y = max(int(plot_bottom) + 1, self.height() - size.height() - 2)
        self._collapse_button.move(x, min(y, max(0, self.height() - size.height())))
        self._collapse_button.raise_()

    def _on_range_changed(self, view_box, view_range, changed) -> None:
        """Hooked up to the ViewBox's sigRangeChanged, so this fires for mouse
        pan/zoom and the Auto-Scale button alike."""
        (xlo, xhi), (ylo, yhi) = view_range
        self._locked_range = (xlo, xhi, ylo, yhi)
        # A pinned box outlives the view it was placed against, and panning can
        # carry it under an edge; re-fit it where it now sits.
        self._fit_tooltip_inside_plot()

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
        if self._eraser_button is not None:
            # Blocked: one click arms every pane, and the echo back into this
            # pane's own button would toggle the mode straight off again.
            self._eraser_button.blockSignals(True)
            self._eraser_button.setChecked(enabled)
            self._eraser_button.blockSignals(False)
        if self._collapse_button is not None:
            # The eraser locks the rest of the window, and its button is the
            # way back out -- so while it is armed the card is held open and
            # the chevron refuses. Folding it away would strand the user with
            # a locked window and nothing to click.
            if enabled and self._collapse_button.isChecked():
                self._collapse_button.setChecked(False)
            self._collapse_button.setEnabled(not enabled)
            self._collapse_button.setToolTip(
                "Switch the Eraser off before hiding these"
                if enabled
                else "Hide the plot buttons"
            )
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
        self._tooltip_anchor = (x, y)
        self._pinned = pinned
        self._fit_tooltip_inside_plot()

    def _fit_tooltip_inside_plot(self) -> None:
        """Keep the whole metadata box inside the plotted area.

        The box is a fixed-size overlay hung off a data point, so near an edge
        it used to run under the axis or the legend and get cropped. Two steps:
        open it towards whichever side has room, and if it still does not fit
        -- a box wider than the space either side of the point -- push it back
        in. That last step breaks the tie to the point, which beats being
        cropped by it.
        """
        item, anchor = self._tooltip_item, self._tooltip_anchor
        if item is None or anchor is None or self._widget is None:
            return
        view = self._widget.getPlotItem().getViewBox()
        area = view.sceneBoundingRect()
        card = item.boundingRect()
        x, y = anchor
        at = view.mapViewToScene(QPointF(x, y))
        pad = 6.0

        # A box wider than the plot cannot be fitted by moving it, so cap it
        # and let the text wrap. Only ever a long file name in the label; the
        # numbers below it are short.
        limit = area.width() - 2 * pad
        if limit > 0 and card.width() > limit:
            item.setTextWidth(limit)
            card = item.boundingRect()

        # anchor=(0, 1) puts the box's bottom-left on the point, so by default
        # it opens right and up; 1 and 0 flip those.
        open_left = at.x() + card.width() + pad > area.right()
        if open_left and at.x() - card.width() - pad < area.left():
            open_left = False  # no room either way; the clamp below handles it
        open_down = at.y() - card.height() - pad < area.top()
        if open_down and at.y() + card.height() + pad > area.bottom():
            open_down = False
        item.setAnchor((1.0 if open_left else 0.0, 0.0 if open_down else 1.0))

        left = at.x() - card.width() if open_left else at.x()
        top = at.y() if open_down else at.y() - card.height()
        dx = min(0.0, (area.right() - pad) - (left + card.width()))
        dx = max(dx, (area.left() + pad) - left) if left + dx < area.left() + pad else dx
        dy = min(0.0, (area.bottom() - pad) - (top + card.height()))
        dy = max(dy, (area.top() + pad) - top) if top + dy < area.top() + pad else dy
        if dx or dy:
            # Scene pixels back into view units; scene y runs downwards.
            px, py = view.viewPixelSize()
            item.setPos(x + dx * px, y - dy * py)

    def _hide_tooltip(self) -> None:
        if self._tooltip_item is not None and self._widget is not None:
            self._widget.getPlotItem().removeItem(self._tooltip_item)
        self._tooltip_item = None
        self._tooltip_anchor = None

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
        # The plotted area is not fixed once the widget is laid out: the legend
        # column is sized from its entries, so a multi-file plot narrows the
        # ViewBox after the fact and would leave the overlay stranded over the
        # legend. Follow the ViewBox instead of placing the buttons once.
        view_box.geometryChanged.connect(self._reposition_overlay)

        for item in getattr(widget, "interactive_items", []):
            item.sigClicked.connect(self._on_point_clicked)
            item.sigHovered.connect(self._on_point_hovered)
        widget.getPlotItem().scene().sigMouseClicked.connect(self._on_scene_clicked)

        # The eraser mode outlives the widget it was set on, so tell the fresh
        # widget about it again.
        self._apply_eraser_cursor()

        if self._overlay is not None:
            # Shown again in case set_message hid it, and folded again if that
            # is how the user left it -- the card belongs to the pane, not to
            # the figure being swapped in.
            self._set_overlay_collapsed(self._collapse_button.isChecked())
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
        # Replot/Auto-Scale/Save Image all act on a plot that isn't there, and
        # the chevron has nothing to fold.
        if self._overlay is not None:
            self._overlay.hide()
            self._collapse_button.hide()

    def clear(self) -> None:
        self._hover_timer.stop()
        self._pending_hover = None
        self._pinned = False
        self._tooltip_item = None
        self._tooltip_anchor = None
        if self._message is not None:
            self._layout.removeWidget(self._message)
            self._message.deleteLater()
            self._message = None
        if self._widget is not None:
            self._layout.removeWidget(self._widget)
            self._widget.deleteLater()
            self._widget = None


class PgSingleFigurePane(QWidget):
    """Holds at most one PlotWidget, filling whatever height it is given.

    Home of the residual figure. It replaced a scrolling stack of figures,
    which the Validation step no longer produces -- residuals are drawn in
    Singular view only, one sweep at a time -- and whose per-figure minimum
    height put a scrollbar on the one figure it did draw.

    Not a PgFigurePane: that one remembers its pan/zoom across replots, and the
    residual axis is framed from the current limits (see
    core.plotting.build_residuals_plot), so it has to re-frame when those
    change or when the pager moves on.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._widget: Optional[pg.PlotWidget] = None

    def set_widget(self, widget: Optional[pg.PlotWidget]) -> None:
        """Show `widget`, or nothing at all when passed None."""
        self.clear()
        if widget is None:
            return
        self._widget = widget
        self._layout.addWidget(widget)

    def clear(self) -> None:
        if self._widget is not None:
            self._layout.removeWidget(self._widget)
            self._widget.deleteLater()
            self._widget = None
