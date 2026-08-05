"""The ECM step's editable schematic: click a component to edit it, a gap to
insert one, a parallel's rail to act on the whole block."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QMenu,
    QSizePolicy,
    QWidget,
)

from gui import style

# Clamp on how far a small circuit is blown up to fill a wide pane, matching
# gui/figure_panes.py's MAX_DIAGRAM_SCALE.
MAX_SCALE = 1.6
# Radius of an insertion badge, in device pixels.
BADGE_RADIUS = 9.0

# Element menu, grouped so the long tail of diffusion and composite elements
# does not bury the four that account for nearly every circuit. Anything the
# registry offers that is not listed here lands under "Other".
ELEMENT_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Basic", ("R", "C", "L", "Q")),
    ("Diffusion", ("W", "Ws", "Wo", "G", "Gs")),
    ("Composite", ("Zarc", "H", "K")),
)


class CircuitCanvas(QWidget):
    """Draws a CircuitDrawing and turns clicks on it into edit requests."""

    element_activated = Signal(int)          # node_id
    insert_requested = Signal(int, int, str)  # connection_id, index, symbol
    add_branch_requested = Signal(int, str)   # connection_id, symbol
    add_series_requested = Signal(int, str)   # node_id, symbol
    wrap_requested = Signal(int, str)         # node_id, symbol
    duplicate_requested = Signal(int)         # node_id
    delete_requested = Signal(int)            # node_id
    undo_requested = Signal()
    redo_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._drawing = None
        self._renderer: Optional[QSvgRenderer] = None
        self._hovered = None
        self._selected_id: Optional[int] = None
        self._mode = "light"
        self._placeholder = "No circuit yet."
        self._drawn = QRectF()

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(160)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------- contents

    def set_drawing(self, drawing) -> None:
        self._drawing = drawing
        self._renderer = QSvgRenderer(QByteArray(drawing.svg)) if drawing else None
        self._hovered = None
        if drawing is not None and self._selected_id is not None:
            if all(r.node_id != self._selected_id for r in drawing.regions):
                self._selected_id = None
        # Immediately, so anchor_for works before the first repaint.
        self._layout()
        self.update()

    def set_message(self, text: str) -> None:
        self._drawing = None
        self._renderer = None
        self._hovered = None
        self._placeholder = text
        self.update()

    def set_theme_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def selected_id(self) -> Optional[int]:
        return self._selected_id

    def select(self, node_id: Optional[int]) -> None:
        self._selected_id = node_id
        self.update()

    def anchor_for(self, node_id: int):
        """Where an editor for `node_id` should pop up, in global coordinates."""
        region = next(
            (r for r in (self._drawing.regions if self._drawing else []) if r.node_id == node_id),
            None,
        )
        if region is None:
            return self.mapToGlobal(self.rect().center())
        rect = self._to_widget(region.rect)
        return self.mapToGlobal(rect.bottomLeft().toPoint())

    # ------------------------------------------------------------ geometry

    def _layout(self) -> None:
        """Fit the drawing into the widget, keeping its aspect ratio."""
        if self._drawing is None:
            self._drawn = QRectF()
            return
        _, _, box_width, box_height = self._drawing.viewbox
        if box_width <= 0 or box_height <= 0:
            self._drawn = QRectF()
            return

        scale = min(
            self.width() / box_width,
            self.height() / box_height,
            MAX_SCALE,
        )
        width, height = box_width * scale, box_height * scale
        self._drawn = QRectF(
            (self.width() - width) / 2.0, (self.height() - height) / 2.0, width, height
        )

    def _to_widget(self, rect: Tuple[float, float, float, float]) -> QRectF:
        box_x, box_y, box_width, box_height = self._drawing.viewbox
        scale = self._drawn.width() / box_width if box_width else 0.0
        x, y, width, height = rect
        return QRectF(
            self._drawn.left() + (x - box_x) * scale,
            self._drawn.top() + (y - box_y) * scale,
            width * scale,
            height * scale,
        )

    def _to_viewbox(self, point: QPointF) -> Optional[Tuple[float, float]]:
        if self._drawing is None or self._drawn.width() <= 0:
            return None
        box_x, box_y, box_width, _ = self._drawing.viewbox
        scale = self._drawn.width() / box_width
        return (
            box_x + (point.x() - self._drawn.left()) / scale,
            box_y + (point.y() - self._drawn.top()) / scale,
        )

    def _region_at(self, point: QPointF):
        coords = self._to_viewbox(point)
        return self._drawing.region_at(*coords) if coords else None

    # ------------------------------------------------------------- painting

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        tokens = style.tokens(self._mode)

        if self._renderer is None:
            painter.setPen(QColor(tokens["text_muted"]))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._placeholder)
            painter.end()
            return

        self._layout()
        self._renderer.render(painter, self._drawn)

        accent = QColor(tokens["accent"])
        for region in self._drawing.regions:
            if region.kind == "gap":
                self._paint_badge(painter, region, accent, tokens)

        if self._selected_id is not None:
            for region in self._drawing.regions:
                if region.node_id == self._selected_id and region.kind != "gap":
                    self._paint_highlight(painter, region, accent, fill=28)
        if self._hovered is not None and self._hovered.kind != "gap":
            self._paint_highlight(painter, self._hovered, accent, fill=16)
        painter.end()

    def _paint_highlight(self, painter: QPainter, region, accent: QColor, fill: int) -> None:
        rect = self._to_widget(region.rect).adjusted(-2, -2, 2, 2)
        painter.setPen(QPen(accent, 1.4))
        painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), fill))
        painter.drawRoundedRect(rect, style.RADIUS, style.RADIUS)

    def _paint_badge(self, painter: QPainter, region, accent: QColor, tokens: dict) -> None:
        """A '+' on the wire, brightened when the pointer is over it."""
        hovered = self._hovered is region
        center = self._to_widget(region.rect).center()
        radius = BADGE_RADIUS * (1.2 if hovered else 1.0)

        painter.setPen(QPen(accent, 1.2))
        painter.setBrush(
            QColor(tokens["surface"] if not hovered else tokens["accent_deep"])
        )
        painter.drawEllipse(center, radius, radius)

        painter.setPen(
            QPen(QColor(tokens["accent_text"]) if hovered else accent, 1.6)
        )
        arm = radius * 0.5
        painter.drawLine(
            QPointF(center.x() - arm, center.y()), QPointF(center.x() + arm, center.y())
        )
        painter.drawLine(
            QPointF(center.x(), center.y() - arm), QPointF(center.x(), center.y() + arm)
        )

    # -------------------------------------------------------------- events

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout()

    def mouseMoveEvent(self, event) -> None:
        if self._drawing is None:
            return
        region = self._region_at(event.position())
        if region is not self._hovered:
            self._hovered = region
            self.setCursor(Qt.PointingHandCursor if region else Qt.ArrowCursor)
            self.setToolTip(_region_tooltip(region))
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Undo):
            self.undo_requested.emit()
        elif event.matches(QKeySequence.Redo):
            self.redo_requested.emit()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self._selected_id:
            self.delete_requested.emit(self._selected_id)
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._drawing is None or event.button() != Qt.LeftButton:
            return
        region = self._region_at(event.position())
        if region is None:
            self.select(None)
            return

        if region.kind == "gap":
            symbol = self._pick_element(event.globalPosition().toPoint())
            if symbol:
                self.insert_requested.emit(region.node_id, region.index, symbol)
        else:
            self.select(region.node_id)
            if region.kind == "element":
                self.element_activated.emit(region.node_id)

    def contextMenuEvent(self, event) -> None:
        if self._drawing is None:
            return
        region = self._region_at(QPointF(event.pos()))
        if region is None or region.kind == "gap":
            return
        self.select(region.node_id)

        menu = QMenu(self)
        node_id = region.node_id
        if region.kind == "element":
            menu.addAction("Edit…", lambda: self.element_activated.emit(node_id))
            self._add_element_submenu(
                menu, "Add in series…", lambda s: self.add_series_requested.emit(node_id, s)
            )
            self._add_element_submenu(
                menu, "Add in parallel…", lambda s: self.wrap_requested.emit(node_id, s)
            )
        else:
            self._add_element_submenu(
                menu, "Add branch…", lambda s: self.add_branch_requested.emit(node_id, s)
            )
        menu.addSeparator()
        menu.addAction("Duplicate", lambda: self.duplicate_requested.emit(node_id))
        menu.addAction("Delete", lambda: self.delete_requested.emit(node_id))
        menu.exec(event.globalPos())

    def _add_element_submenu(self, menu: QMenu, title: str, on_chosen) -> None:
        submenu = menu.addMenu(title)
        fill_element_menu(submenu, on_chosen)

    def _pick_element(self, at) -> Optional[str]:
        """A modal element menu at the pointer, for a click on a gap."""
        chosen: List[str] = []
        menu = QMenu("Insert element", self)
        fill_element_menu(menu, chosen.append)
        menu.exec(at)
        return chosen[0] if chosen else None


def _region_tooltip(region) -> str:
    if region is None:
        return ""
    if region.kind == "gap":
        return "Insert a component here"
    if region.kind == "connection":
        return "Parallel block — right-click to add a branch, duplicate or delete"
    return "Click to edit; right-click for more"


def _element_menu_entries() -> List[Tuple[str, List[Tuple[str, str]]]]:
    """[(group, [(symbol, description)])] for every registered element."""
    from core.circuit_model import element_symbols

    classes = element_symbols()
    grouped: List[Tuple[str, List[Tuple[str, str]]]] = []
    placed = set()
    for title, symbols in ELEMENT_GROUPS:
        entries = [
            (symbol, classes[symbol].get_description())
            for symbol in symbols
            if symbol in classes
        ]
        if entries:
            grouped.append((title, entries))
            placed.update(symbol for symbol, _ in entries)

    rest = [
        (symbol, cls.get_description())
        for symbol, cls in classes.items()
        if symbol not in placed
    ]
    if rest:
        grouped.append(("Other", rest))
    return grouped


def fill_element_menu(menu: QMenu, on_chosen) -> None:
    """Populate `menu` with every registered element, grouped."""
    for index, (title, entries) in enumerate(_element_menu_entries()):
        if index:
            menu.addSeparator()
        header = menu.addAction(title.upper())
        header.setEnabled(False)
        header.setFont(style.section_font())
        for symbol, description in entries:
            action = menu.addAction(f"{symbol} — {description}")
            action.setToolTip(description)
            action.triggered.connect(lambda _=False, s=symbol: on_chosen(s))
