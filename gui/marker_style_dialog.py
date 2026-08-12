"""The Marker & line style popup: one marker shape per loaded file, plus the
marker size and line width shared by every sweep.

Per file rather than per sweep by design -- sweeps within a file are told apart
by color, and shape is what separates one file from another.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.plotting import MARKER_SHAPES, default_marker_for
from gui import style
from gui.steps.base import add_combo_items, compact_combo, section_label

# Past this many files the per-file rows get their own scroll area rather than growing the dialog past a small screen.
_SCROLL_AFTER = 8


class MarkerStyleDialog(QDialog):
    """Edits (marker per file, marker size, line width). Read the result from
    `markers`, `marker_size` and `line_width` after exec() returns Accepted."""

    def __init__(
        self,
        files: Sequence[Tuple[int, str]],
        markers: Dict[int, str],
        marker_size: float,
        line_width: float,
        parent: Optional[QWidget] = None,
    ):
        """`files` is [(file_id, display name)] in load order; `markers` maps
        file_id -> pyqtgraph symbol for the files already given one."""
        super().__init__(parent)
        self.setWindowTitle("Marker & line style")
        self._file_ids = [file_id for file_id, _ in files]
        self._combos: List[QComboBox] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(style.GROUP_SPACING)

        form = QFormLayout()
        form.setSpacing(style.GROUP_SPACING)
        # Both spinboxes are narrow; without this the label column stretches to the longest file name below.
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self._size_spin = QDoubleSpinBox()
        self._size_spin.setRange(2.0, 24.0)
        self._size_spin.setSingleStep(1.0)
        self._size_spin.setDecimals(1)
        self._size_spin.setSuffix(" px")
        self._size_spin.setValue(marker_size)
        self._size_spin.setToolTip(
            "Marker diameter. The ✕ on removed points follows this too, so the "
            "two stay comparable."
        )
        form.addRow("Marker size", self._size_spin)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(0.5, 8.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setDecimals(1)
        self._width_spin.setSuffix(" px")
        self._width_spin.setValue(line_width)
        self._width_spin.setToolTip(
            "Stroke width for the connecting line in 'Line' style, and for the "
            "dashed fit curves on the ECM step."
        )
        form.addRow("Line width", self._width_spin)
        layout.addLayout(form)

        layout.addWidget(section_label("MARKER PER FILE"))
        if files:
            hint = QLabel(
                "Sweeps within a file are told apart by color; the shape is "
                "what tells the files apart. All shapes are filled — the Bode "
                "plot uses hollow markers for -Φ, and ✕ marks removed points."
            )
        else:
            hint = QLabel("No files loaded yet — open one to give it a shape.")
        hint.setWordWrap(True)
        hint.setProperty("state", "muted")
        layout.addWidget(hint)

        if files:
            layout.addWidget(self._build_file_rows(files, markers))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------- building

    def _build_file_rows(
        self, files: Sequence[Tuple[int, str]], markers: Dict[int, str]
    ) -> QWidget:
        body = QWidget()
        rows = QFormLayout(body)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(style.GROUP_SPACING)
        rows.setRowWrapPolicy(QFormLayout.DontWrapRows)

        for position, (file_id, name) in enumerate(files):
            combo = compact_combo(QComboBox())
            add_combo_items(combo, [(label, symbol) for label, symbol in MARKER_SHAPES])
            # Falls back to the shape this file would get anyway, so the dialog opens showing what is on screen.
            current = markers.get(file_id, default_marker_for(position))
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            self._combos.append(combo)
            rows.addRow(name, combo)

        if len(files) <= _SCROLL_AFTER:
            return body

        area = QScrollArea()
        area.setWidget(body)
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area

    # --------------------------------------------------------------- result

    @property
    def markers(self) -> Dict[int, str]:
        """file_id -> chosen pyqtgraph symbol, for every file listed."""
        return {
            file_id: combo.currentData()
            for file_id, combo in zip(self._file_ids, self._combos)
        }

    @property
    def marker_size(self) -> float:
        return self._size_spin.value()

    @property
    def line_width(self) -> float:
        return self._width_spin.value()
