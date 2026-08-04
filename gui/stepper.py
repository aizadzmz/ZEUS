"""The breadcrumb across the top naming the four workflow stages, and switching
between them."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QWidget,
)

# (step id, label). Order is the display order and the stack order.
STEPS: Tuple[Tuple[str, str], ...] = (
    ("data_viz", "Data Visualisation"),
    ("validation", "Validation"),
    ("drt", "DRT"),
    ("ecm", "ECM"),
)

# The bar's look lives in gui.style's theme-agnostic QSS block, keyed off the
# objectName set below.


class StepBar(QWidget):
    """A row of step buttons, exactly one of them current."""

    step_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Scopes the stylesheet's button rules to this bar. Without the
        # objectName they would restyle every other QToolButton in the app.
        self.setObjectName("stepBar")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(2)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: List[QToolButton] = []

        for i, (step_id, label) in enumerate(STEPS):
            if i:
                separator = QLabel("›")
                separator.setEnabled(False)
                row.addWidget(separator)

            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setAutoRaise(True)
            # clicked, not toggled: toggled also fires for the button being
            # unchecked as the exclusive group moves on, which would emit two
            # step_selected in a row -- the second one naming the step being
            # left.
            button.clicked.connect(lambda _checked, s=step_id: self.step_selected.emit(s))
            self._group.addButton(button, i)
            self._buttons.append(button)
            row.addWidget(button)

        row.addStretch()
        self._buttons[0].setChecked(True)

    def set_current(self, step_id: str) -> None:
        """Reflect a step change that came from elsewhere, e.g. a session
        restore. Silent, so it cannot loop back into its caller."""
        for i, (candidate, _) in enumerate(STEPS):
            if candidate == step_id:
                self._buttons[i].blockSignals(True)
                self._buttons[i].setChecked(True)
                self._buttons[i].blockSignals(False)
                return

    @staticmethod
    def index_of(step_id: str) -> int:
        """Position in STEPS, which is also the page index in the step stack."""
        return next(i for i, (candidate, _) in enumerate(STEPS) if candidate == step_id)
