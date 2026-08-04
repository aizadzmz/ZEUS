"""The "Files and sets" panel: the Data Visualisation step's full view of the
selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.selection import SweepSelection


class FilesAndSetsPanel(QWidget):
    """A checkable list of every loaded sweep, grouped under a disabled bold
    header per file."""

    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selection = selection
        self._stems: Dict[int, str] = {}
        # Guards the model -> widget direction, so repopulating the list does
        # not echo every row back into the model as a user edit.
        self._syncing = False

        col = QVBoxLayout(self)
        col.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        header.addWidget(QLabel("Files and sets"))
        header.addStretch()
        self.select_all_button = QPushButton("Select all")
        self.select_all_button.clicked.connect(lambda: self._selection.set_all(True))
        header.addWidget(self.select_all_button)
        self.select_none_button = QPushButton("Select none")
        self.select_none_button.clicked.connect(lambda: self._selection.set_all(False))
        header.addWidget(self.select_none_button)
        col.addLayout(header)

        self.list = QListWidget()
        self.list.setToolTip(
            "Which files/sets to display and run analyses over. Checking more "
            "sweeps means validation, DRT and circuit fitting all process more."
        )
        self.list.itemChanged.connect(self._on_item_changed)
        # Clicking a row's text (not its checkbox) moves the cursor there, so
        # a sweep can be jumped to without paging with the `< >` strip.
        self.list.currentItemChanged.connect(self._on_current_item_changed)
        col.addWidget(self.list, stretch=1)

        self._selection.selection_changed.connect(self.sync_checks)
        self.rebuild()

    def rebuild(self) -> None:
        """Repopulate the rows from the model's dataset list. Only needed when
        files are loaded or removed; check/uncheck goes through sync_checks."""
        self._syncing = True
        self.list.clear()

        by_file: Dict[int, List] = defaultdict(list)
        for ds in self._selection.datasets:
            by_file[ds.file_id].append(ds)

        for file_id, datasets in by_file.items():
            header_item = QListWidgetItem(self._stems.get(file_id, f"File {file_id}"))
            header_item.setFlags(Qt.NoItemFlags)
            bold = header_item.font()
            bold.setBold(True)
            header_item.setFont(bold)
            self.list.addItem(header_item)

            for ds in datasets:
                item = QListWidgetItem(ds.label)
                item.setData(Qt.UserRole, ds.key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.Checked if self._selection.is_checked(ds.key) else Qt.Unchecked
                )
                self.list.addItem(item)

        self._syncing = False

    def set_files(self, files: List) -> None:
        """Supply the LoadedFile registry for the group headers, then rebuild."""
        self._stems = {lf.file_id: lf.stem for lf in files}
        self.rebuild()

    def sync_checks(self) -> None:
        """Push the model's checked set back onto existing rows -- for changes
        that came from somewhere else, such as Select all."""
        self._syncing = True
        for i in range(self.list.count()):
            item = self.list.item(i)
            key = item.data(Qt.UserRole)
            if key is None:
                continue
            state = Qt.Checked if self._selection.is_checked(key) else Qt.Unchecked
            if item.checkState() != state:
                item.setCheckState(state)
        self._syncing = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._syncing:
            return
        key = item.data(Qt.UserRole)
        if key is None:
            return
        self._selection.set_checked(key, item.checkState() == Qt.Checked)

    def _on_current_item_changed(self, item: Optional[QListWidgetItem]) -> None:
        if self._syncing or item is None:
            return
        key = item.data(Qt.UserRole)
        # Only sweeps being drawn can be paged to; checking one first would
        # silently widen what the next analysis run covers.
        if key is not None and self._selection.is_checked(key):
            self._selection.set_cursor_key(key)
