"""The progress bar across the top naming the four workflow stages, and
switching between them.

Drawn rather than assembled from widgets: a QToolButton row cannot produce the
line running *behind* the nodes, and one paintEvent keeps the geometry in a
single place."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from gui import style

# (step id, label). Order is the display order and the stack order.
STEPS: Tuple[Tuple[str, str], ...] = (
    ("data_viz", "Data Visualisation"),
    ("validation", "Validation"),
    ("drt", "DRT"),
    ("ecm", "ECM"),
)

# ------------------------------------------------------------- geometry
# All in device-independent pixels, measured from the widget's own rect.

NODE_RADIUS = 5.5        # outer radius of a stage node
NODE_RING = 2.0          # ring thickness on the not-current nodes
TRACK_WIDTH = 2.0        # the line the nodes sit on
LABEL_GAP = 9            # label baseline -> track
EDGE_PAD = 10            # widget edge -> first/last node, on top of the
                         # half-label the end labels need to sit centred
TOP_PAD = 8
BOTTOM_PAD = 8
HOTSPOT_PAD = 16         # widened either side of a label, for clicking
LABEL_SLACK = 4          # headroom on a label rect, so a rounded-up width can
                         # never come out under what QPainter lays out


class StepBar(QWidget):
    """A track of stage nodes, exactly one of them current."""

    step_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("stepBar")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        # Arrow keys move between stages once the bar has focus.
        self.setFocusPolicy(Qt.StrongFocus)

        self._mode = "light"
        self._current = 0
        self._hovered: Optional[int] = None

    # ---------------------------------------------------------- public API

    def set_current(self, step_id: str) -> None:
        """Reflect a step change that came from elsewhere, e.g. a session
        restore. Silent, so it cannot loop back into its caller."""
        self._current = self.index_of(step_id)
        self.update()

    def set_theme_mode(self, mode: str) -> None:
        """Repaint against the palette for ``mode`` ("light" or "dark")."""
        self._mode = mode
        self.update()

    @staticmethod
    def index_of(step_id: str) -> int:
        """Position in STEPS, which is also the page index in the step stack."""
        return next(i for i, (candidate, _) in enumerate(STEPS) if candidate == step_id)

    # ------------------------------------------------------------ metrics

    def _label_font(self) -> QFont:
        font = QFont(self.font())
        font.setBold(True)
        # Every metric below -- node spacing, track height, sizeHint -- is measured from this font.
        font.setPointSizeF(style.FONT_PT_STEPPER)
        return font

    def _metrics(self) -> QFontMetricsF:
        """Metrics bound to this widget as the paint device. One built from the
        font alone measures against the application's default DPI, which on a
        fractionally scaled screen elides text that in fact fits.
        """
        return QFontMetricsF(self._label_font(), self)

    def _label_widths(self) -> List[int]:
        fm = self._metrics()
        # Rounded up, never down: a rect a fraction of a pixel short of the text is a rect that elides.
        return [int(math.ceil(fm.horizontalAdvance(label))) + LABEL_SLACK
                for _, label in STEPS]

    def _margin(self) -> float:
        """Room for the end labels, which are centred on the end nodes and so
        overhang them by half their width."""
        widths = self._label_widths()
        return max(widths[0], widths[-1]) / 2 + EDGE_PAD

    def _node_positions(self) -> List[float]:
        """Node centres, spread evenly between the margins."""
        margin = self._margin()
        span = max(self.width() - 2 * margin, 1.0)
        gaps = len(STEPS) - 1
        return [margin + span * i / gaps for i in range(len(STEPS))]

    def _track_y(self) -> float:
        fm = self._metrics()
        return TOP_PAD + fm.height() + LABEL_GAP + NODE_RADIUS

    def sizeHint(self) -> QSize:
        fm = self._metrics()
        width = int(sum(self._label_widths()) + 2 * EDGE_PAD * len(STEPS))
        height = int(TOP_PAD + fm.height() + LABEL_GAP + 2 * NODE_RADIUS + BOTTOM_PAD)
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return QSize(240, self.sizeHint().height())

    # ------------------------------------------------------------ painting

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        t = style.tokens(self._mode)
        accent = QColor(t["accent"])
        # The track reads as a rail the nodes sit on, so it stays well below them in weight.
        track = QColor(accent)
        track.setAlpha(90)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self._label_font())

        xs = self._node_positions()
        cy = self._track_y()

        painter.setPen(QPen(track, TRACK_WIDTH, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(xs[0], cy), QPointF(xs[-1], cy))

        # Widths from the painter's own metrics, not the widget's: the two disagree across scale factors, and it is the painter's that decide whether a label fits.
        fm = painter.fontMetrics()
        widths = [int(math.ceil(fm.horizontalAdvance(label))) + LABEL_SLACK
                  for _, label in STEPS]

        for i, (_, label) in enumerate(STEPS):
            current = i == self._current
            hovered = i == self._hovered

            # Node. Current is filled solid; the rest are hollow rings that let the track show through.
            painter.setBrush(QColor(t["accent_deep"]) if current else QColor(t["surface"]))
            if current:
                painter.setPen(Qt.NoPen)
            else:
                ring = QColor(accent)
                ring.setAlpha(255 if hovered else 180)
                painter.setPen(QPen(ring, NODE_RING))
            painter.drawEllipse(QPointF(xs[i], cy), NODE_RADIUS, NODE_RADIUS)

            # Label, centred on its node and clamped inside the widget so end labels cannot be clipped.
            text_color = QColor(t["text"] if current or hovered else t["text_muted"])
            painter.setPen(QPen(text_color))
            painter.setBrush(Qt.NoBrush)
            rect = self._label_rect(xs[i], widths[i], fm.height())
            painter.drawText(rect, Qt.AlignHCenter | Qt.AlignBottom,
                             fm.elidedText(label, Qt.ElideRight, int(rect.width())))

    def _label_rect(self, x: float, width: int, height: float) -> QRectF:
        left = min(max(x - width / 2, 0.0), max(self.width() - width, 0.0))
        return QRectF(left, TOP_PAD, min(float(width), float(self.width())), height)

    # ------------------------------------------------------------- input

    def _hit(self, x: float, y: float) -> Optional[int]:
        """The step whose label/node column contains ``x``, if any."""
        if not (0 <= y <= self.height()):
            return None
        fm = self._metrics()
        widths = self._label_widths()
        for i, node_x in enumerate(self._node_positions()):
            rect = self._label_rect(node_x, widths[i], fm.height())
            if rect.left() - HOTSPOT_PAD <= x <= rect.right() + HOTSPOT_PAD:
                return i
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        hit = self._hit(event.position().x(), event.position().y())
        if hit != self._hovered:
            self._hovered = hit
            self.setCursor(Qt.PointingHandCursor if hit is not None else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = None
        self.unsetCursor()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        hit = self._hit(event.position().x(), event.position().y())
        if hit is not None and hit != self._current:
            self._select(hit)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Left, Qt.Key_Up):
            self._select(max(self._current - 1, 0))
        elif event.key() in (Qt.Key_Right, Qt.Key_Down):
            self._select(min(self._current + 1, len(STEPS) - 1))
        else:
            super().keyPressEvent(event)

    def _select(self, index: int) -> None:
        if index == self._current:
            return
        self._current = index
        self.update()
        self.step_selected.emit(STEPS[index][0])
