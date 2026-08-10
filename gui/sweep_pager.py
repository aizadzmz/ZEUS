"""The compact `File Name | Set Number | << < > >>` strip between a step's two
plots."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from gui import style
from gui.selection import SweepSelection


class SweepPager(QWidget):
    """Names the sweep on screen and steps the cursor either side of it."""

    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selection = selection
        self._stems: Dict[int, str] = {}

        row = QHBoxLayout(self)
        row.setContentsMargins(*style.CONTENT_MARGINS)
        row.setSpacing(style.LIST_SPACING)

        # Metadata about the plot rather than part of it: muted, and a size
        # down from the body text.
        self.file_label = QLabel("—")
        self.file_label.setToolTip("The file the sweep on screen came from.")
        self.set_label = QLabel("—")
        self.set_label.setToolTip("Which sweep within that file.")
        row.addWidget(QLabel("File"))
        row.addWidget(self.file_label)
        row.addSpacing(style.PAGER_FIELD_GAP)
        row.addWidget(QLabel("Set"))
        row.addWidget(self.set_label)
        row.addStretch()

        self.position_label = QLabel("")
        self.position_label.setToolTip("Position within the selected sweeps.")
        for label in (self.file_label, self.set_label, self.position_label):
            label.setProperty("state", "muted")
            font = label.font()
            font.setPointSizeF(style.FONT_PT_SMALL)
            label.setFont(font)
        row.addWidget(self.position_label)

        # `‹‹ ‹ | › ››`: the inner pair walks the checked sweeps flat, crossing
        # into the next file at a file's end; the outer pair skips a whole file
        # at a time. The gap groups them by direction, as a media player does,
        # so the four do not read as one undifferentiated run of arrows.
        self.prev_file_button = QToolButton()
        self.prev_file_button.setText("‹‹")
        self.prev_file_button.setToolTip("First sweep of the previous file")
        self.prev_file_button.clicked.connect(lambda: self._selection.step_file(-1))
        row.addWidget(self.prev_file_button)

        self.prev_button = QToolButton()
        self.prev_button.setText("‹")
        self.prev_button.setToolTip("Previous selected sweep")
        self.prev_button.clicked.connect(lambda: self._selection.step_cursor(-1))
        row.addWidget(self.prev_button)

        row.addSpacing(style.PAGER_ARROW_GAP)

        self.next_button = QToolButton()
        self.next_button.setText("›")
        self.next_button.setToolTip("Next selected sweep")
        self.next_button.clicked.connect(lambda: self._selection.step_cursor(1))
        row.addWidget(self.next_button)

        self.next_file_button = QToolButton()
        self.next_file_button.setText("››")
        self.next_file_button.setToolTip("First sweep of the next file")
        self.next_file_button.clicked.connect(lambda: self._selection.step_file(1))
        row.addWidget(self.next_file_button)

        # Both signals matter: the cursor moves on its own, but checking or
        # unchecking sweeps also changes the "2 of 7" readout and can reseat
        # the cursor.
        self._selection.cursor_moved.connect(self.sync)
        self._selection.selection_changed.connect(self.sync)
        self.sync()

    def set_files(self, files: List) -> None:
        """Supply the LoadedFile registry so the strip can name a sweep's
        source file; the datasets themselves only carry a file_id."""
        self._stems = {lf.file_id: lf.stem for lf in files}
        self.sync()

    def sync(self) -> None:
        ds = self._selection.current()
        if ds is None:
            self.file_label.setText("—")
            self.set_label.setText("—")
            self.position_label.setText("")
        else:
            self.file_label.setText(self._stems.get(ds.file_id, "—"))
            self.set_label.setText(ds.label)
            index, count = self._selection.cursor_position()
            self.position_label.setText(f"{index} of {count}" if count else "")

        self.prev_button.setEnabled(self._selection.can_step(-1))
        self.next_button.setEnabled(self._selection.can_step(1))
        self.prev_file_button.setEnabled(self._selection.can_step_file(-1))
        self.next_file_button.setEnabled(self._selection.can_step_file(1))
