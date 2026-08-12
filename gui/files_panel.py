"""The "Files and sets" panel: the Data Visualisation step's full view of the
selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import style
from gui.selection import SweepSelection

# Past this many files, new loads come in collapsed; a single file has nothing to hide behind.
COLLAPSE_ABOVE_FILE_COUNT = 1


class FilesAndSetsPanel(QWidget):
    """A tree of every loaded sweep, grouped under a checkable, collapsible
    header per file."""

    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selection = selection
        self._stems: Dict[int, str] = {}
        # File headers are items, not widgets, so they take their color here rather than from the stylesheet.
        self._mode = "light"
        # Guards the model -> widget direction, so repopulating the tree does not echo rows back as user edits.
        self._syncing = False
        # Survives rebuild(), which theme changes and every file load trigger.
        self._collapsed_file_ids: set[int] = set()

        col = QVBoxLayout(self)
        col.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        header.addWidget(QLabel("Active Datasets"))
        header.addStretch()
        self.select_all_button = QPushButton("Select all")
        self.select_all_button.clicked.connect(lambda: self._selection.set_all(True))
        header.addWidget(self.select_all_button)
        self.select_none_button = QPushButton("Select none")
        self.select_none_button.clicked.connect(lambda: self._selection.set_all(False))
        header.addWidget(self.select_none_button)
        self.collapse_all_button = QPushButton("Collapse all")
        self.collapse_all_button.clicked.connect(lambda: self._set_all_expanded(False))
        header.addWidget(self.collapse_all_button)
        self.expand_all_button = QPushButton("Expand all")
        self.expand_all_button.clicked.connect(lambda: self._set_all_expanded(True))
        header.addWidget(self.expand_all_button)
        col.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setObjectName("filesList")
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        # Rows are one line of text; uniform heights let the view skip measuring every item on a batch load.
        self.tree.setUniformRowHeights(True)
        self.tree.setToolTip(
            "Which files/sets to display and run analyses over. Checking more "
            "sweeps means validation, DRT and circuit fitting all process more."
        )
        self.tree.itemChanged.connect(self._on_item_changed)
        # Clicking a sweep's text (not its checkbox) moves the cursor there, without paging with the `< >` strip.
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        # Clicking a file's name checks or unchecks that whole file; wired to the click so hitting the expand arrow does not also toggle 12 sweeps.
        self.tree.itemClicked.connect(self._on_item_clicked)
        col.addWidget(self.tree, stretch=1)

        self._selection.selection_changed.connect(self.sync_checks)
        self.rebuild()

    # ------------------------------------------------------------- building

    def rebuild(self) -> None:
        """Repopulate the rows from the model's dataset list. Only needed when
        files are loaded or removed; check/uncheck goes through sync_checks."""
        self._syncing = True
        self.tree.clear()

        by_file: Dict[int, List] = defaultdict(list)
        for ds in self._selection.datasets:
            by_file[ds.file_id].append(ds)

        # A fresh multi-file load starts collapsed; a file the user collapsed by hand stays that way.
        collapse_new = len(by_file) > COLLAPSE_ABOVE_FILE_COUNT

        for file_id, datasets in by_file.items():
            stem = self._stems.get(file_id, f"File {file_id}")
            count = len(datasets)
            file_item = QTreeWidgetItem(
                [f"{stem}  ({count} sweep{'s' if count != 1 else ''})"]
            )
            # Not user-checkable: its check state is a summary of the children, so Qt must not let a stray click set it directly.
            file_item.setFlags(Qt.ItemIsEnabled)
            file_item.setData(0, Qt.UserRole + 1, file_id)
            bold = file_item.font(0)
            bold.setBold(True)
            file_item.setFont(0, bold)
            file_item.setForeground(0, QColor(style.tokens(self._mode)["accent"]))
            self.tree.addTopLevelItem(file_item)

            for ds in datasets:
                item = QTreeWidgetItem([ds.label])
                item.setData(0, Qt.UserRole, ds.key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.Checked if self._selection.is_checked(ds.key) else Qt.Unchecked,
                )
                file_item.addChild(item)

            if file_id in self._collapsed_file_ids or collapse_new:
                file_item.setExpanded(False)
                self._collapsed_file_ids.add(file_id)
            else:
                file_item.setExpanded(True)
            self._refresh_file_check_state(file_item)

        self._syncing = False

    def _file_items(self):
        for i in range(self.tree.topLevelItemCount()):
            yield self.tree.topLevelItem(i)

    def _refresh_file_check_state(self, file_item: QTreeWidgetItem) -> None:
        """Show a file as checked, unchecked, or partial from its sweeps."""
        states = [
            file_item.child(i).checkState(0) == Qt.Checked
            for i in range(file_item.childCount())
        ]
        if states and all(states):
            state = Qt.Checked
        elif any(states):
            state = Qt.PartiallyChecked
        else:
            state = Qt.Unchecked
        file_item.setCheckState(0, state)

    # ---------------------------------------------------------- expansion

    def _set_all_expanded(self, expanded: bool) -> None:
        for file_item in self._file_items():
            file_item.setExpanded(expanded)
            file_id = file_item.data(0, Qt.UserRole + 1)
            if expanded:
                self._collapsed_file_ids.discard(file_id)
            else:
                self._collapsed_file_ids.add(file_id)

    # -------------------------------------------------------------- wiring

    def set_theme_mode(self, mode: str) -> None:
        """Repaint the file headers against the palette for ``mode``."""
        self._mode = mode
        self.rebuild()

    def set_files(self, files: List) -> None:
        """Supply the LoadedFile registry for the file headers, then rebuild."""
        self._stems = {lf.file_id: lf.stem for lf in files}
        # Files that went away should not hold a collapsed state for whatever file_id gets handed out next.
        live = {lf.file_id for lf in files}
        self._collapsed_file_ids &= live
        self.rebuild()

    def sync_checks(self) -> None:
        """Push the model's checked set back onto existing rows -- for changes
        that came from somewhere else, such as Select all."""
        self._syncing = True
        for file_item in self._file_items():
            for i in range(file_item.childCount()):
                item = file_item.child(i)
                key = item.data(0, Qt.UserRole)
                if key is None:
                    continue
                state = Qt.Checked if self._selection.is_checked(key) else Qt.Unchecked
                if item.checkState(0) != state:
                    item.setCheckState(0, state)
            self._refresh_file_check_state(file_item)
        self._syncing = False

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        if self._syncing:
            return
        key = item.data(0, Qt.UserRole)
        if key is None:
            return
        self._selection.set_checked(key, item.checkState(0) == Qt.Checked)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        """A click anywhere on a file's row toggles all of its sweeps."""
        if self._syncing or item.parent() is not None:
            return
        # Anything short of every sweep checked means the click turns the file on; only a fully checked file turns off.
        turning_on = item.checkState(0) != Qt.Checked
        keys = [
            item.child(i).data(0, Qt.UserRole)
            for i in range(item.childCount())
            if item.child(i).data(0, Qt.UserRole) is not None
        ]
        self._selection.set_checked_many(keys, turning_on)

    def _on_current_item_changed(self, item: Optional[QTreeWidgetItem]) -> None:
        if self._syncing or item is None:
            return
        key = item.data(0, Qt.UserRole)
        # Only sweeps being drawn can be paged to; checking one first would silently widen the next analysis run.
        if key is not None and self._selection.is_checked(key):
            self._selection.set_cursor_key(key)
