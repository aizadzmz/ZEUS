"""A row of joined toggle buttons, for choosing one of a few short options."""

from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QWidget,
)


class SegmentedControl(QWidget):
    """Exactly one of `labels` is checked at a time."""

    def __init__(self, labels: Sequence[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Scopes the stylesheet's segment rules; see gui.style.build_app_qss.
        self.setObjectName("segmented")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        # Zero spacing makes the segments read as one control; the shared borders are collapsed in QSS.
        row.setSpacing(0)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: List[QToolButton] = []

        last = len(labels) - 1
        for i, label in enumerate(labels):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            # Qt style sheets have no :first-child/:last-child, so the end a segment sits on is carried as a property for the rounded corners.
            button.setProperty("seg", "first" if i == 0 else "last" if i == last else "mid")
            self._group.addButton(button, i)
            self._buttons.append(button)
            row.addWidget(button)

        self._buttons[0].setChecked(True)

    def button(self, index: int) -> QToolButton:
        """One segment, for assigning to the attribute name a window wires to."""
        return self._buttons[index]

