"""The shared answer to "which sweeps am I working on, and which one am I
looking at". Two separate, both-global questions:"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from PySide6.QtCore import QObject, Signal

# How many sweeps start checked per loaded file -- per file, not overall, so
# opening a batch does not leave everything past the first file unchecked.
# The checked set is exactly what analyses process, so a cap here trades a
# lighter first analysis run against having to go and check the rest by hand.
# None means no cap: every sweep in every file starts checked.
DEFAULT_SWEEPS_CHECKED_PER_FILE: Optional[int] = None


class SweepSelection(QObject):
    """The checked set of sweeps plus a cursor into it."""

    # Membership of the checked set changed. Analyses and plot contents both
    # depend on this, so MainWindow routes it into _refresh().
    selection_changed = Signal()
    # The cursor moved; the set is unchanged. Only Single-mode drawing
    # depends on this.
    cursor_moved = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._datasets: List = []
        self._checked: Set[str] = set()
        self._cursor_key: Optional[str] = None

    # -------------------------------------------------------------- contents

    def set_datasets(self, datasets: List, checked_keys: Optional[Iterable[str]] = None) -> None:
        """Rebuild from a fresh _datasets list."""
        self._datasets = list(datasets)

        if checked_keys is None:
            self._checked = self._default_checked()
        else:
            # Intersected, not trusted: a restored session may name sweeps
            # from a file no longer loaded.
            wanted = set(checked_keys)
            self._checked = {ds.key for ds in self._datasets if ds.key in wanted}

        self._reseat_cursor()
        self.selection_changed.emit()
        self.cursor_moved.emit()

    def _default_checked(self) -> Set[str]:
        if DEFAULT_SWEEPS_CHECKED_PER_FILE is None:
            return {ds.key for ds in self._datasets}

        seen: Dict[int, int] = {}
        checked: Set[str] = set()
        for ds in self._datasets:
            n = seen.get(ds.file_id, 0)
            if n < DEFAULT_SWEEPS_CHECKED_PER_FILE:
                checked.add(ds.key)
            seen[ds.file_id] = n + 1
        return checked

    @property
    def datasets(self) -> List:
        """Every loaded sweep, checked or not, in load order."""
        return self._datasets

    # ------------------------------------------------------------- the set

    def is_checked(self, key: str) -> bool:
        return key in self._checked

    def checked_keys(self) -> Set[str]:
        return set(self._checked)

    def selected(self) -> List:
        """The checked sweeps, in load order. This is what analyses run over
        and what every step's plots draw from."""
        return [ds for ds in self._datasets if ds.key in self._checked]

    def set_checked(self, key: str, checked: bool) -> None:
        if checked == (key in self._checked):
            return
        if checked:
            self._checked.add(key)
        else:
            self._checked.discard(key)
        moved = self._reseat_cursor()
        self.selection_changed.emit()
        if moved:
            self.cursor_moved.emit()

    def set_checked_many(self, keys: Iterable[str], checked: bool) -> None:
        """Check or uncheck a group of sweeps as one edit.

        Looping over set_checked() instead would emit selection_changed per
        sweep, and MainWindow turns each of those into a full replot -- so
        ticking one 50-sweep file would redraw the plots 50 times.
        """
        keys = set(keys)
        new = self._checked | keys if checked else self._checked - keys
        if new == self._checked:
            return
        self._checked = new
        moved = self._reseat_cursor()
        self.selection_changed.emit()
        if moved:
            self.cursor_moved.emit()

    def set_all(self, checked: bool) -> None:
        new = {ds.key for ds in self._datasets} if checked else set()
        if new == self._checked:
            return
        self._checked = new
        moved = self._reseat_cursor()
        self.selection_changed.emit()
        if moved:
            self.cursor_moved.emit()

    # ---------------------------------------------------------- the cursor

    def current(self):
        """The sweep the `< >` strip is on, or None when nothing is checked."""
        if self._cursor_key is None:
            return None
        return next((ds for ds in self._datasets if ds.key == self._cursor_key), None)

    def cursor_position(self) -> tuple:
        """(index, count) within the checked set, 1-based for display. (0, 0)
        when nothing is checked."""
        selected = self.selected()
        if not selected:
            return (0, 0)
        keys = [ds.key for ds in selected]
        if self._cursor_key not in keys:
            return (0, len(keys))
        return (keys.index(self._cursor_key) + 1, len(keys))

    def set_cursor_key(self, key: Optional[str]) -> None:
        if key == self._cursor_key:
            return
        self._cursor_key = key
        self.cursor_moved.emit()

    def step_cursor(self, delta: int) -> None:
        """Move the cursor by delta places within the checked set, clamping at
        both ends rather than wrapping."""
        selected = self.selected()
        if not selected:
            return
        keys = [ds.key for ds in selected]
        if self._cursor_key in keys:
            index = keys.index(self._cursor_key)
        else:
            index = 0
        new_index = max(0, min(len(keys) - 1, index + delta))
        if keys[new_index] != self._cursor_key:
            self._cursor_key = keys[new_index]
            self.cursor_moved.emit()

    def can_step(self, delta: int) -> bool:
        """Whether stepping by delta would actually move -- drives the `< >`
        buttons' enabled state."""
        index, count = self.cursor_position()
        if count == 0:
            return False
        return 0 < index + delta <= count

    # --------------------------------------------------------- by whole file

    def _checked_files(self) -> List[tuple]:
        """The checked sweeps grouped by source file, in load order, as
        (file_id, key of its first checked sweep).

        Files with nothing checked do not appear: paging by file lands on a
        sweep, so a file offering none is not a place the cursor can go.
        """
        files: List[tuple] = []
        seen: Set[int] = set()
        for ds in self.selected():
            if ds.file_id not in seen:
                seen.add(ds.file_id)
                files.append((ds.file_id, ds.key))
        return files

    def _current_file_index(self, files: List[tuple]) -> int:
        """Where the cursor's file sits in ``files``. Falls back to the first
        file when the cursor is unseated, matching step_cursor()."""
        current = self.current()
        if current is None:
            return 0
        return next(
            (i for i, (file_id, _) in enumerate(files) if file_id == current.file_id),
            0,
        )

    def step_file(self, delta: int) -> None:
        """Move the cursor to the first checked sweep of the file delta places
        away, clamping at both ends rather than wrapping.

        Always lands on that file's first sweep, including when stepping back
        from mid-file -- `‹‹` means "the file before this one", not "the top of
        this one", so two clicks never sit on the same file.
        """
        files = self._checked_files()
        if not files:
            return
        index = self._current_file_index(files)
        new_index = max(0, min(len(files) - 1, index + delta))
        key = files[new_index][1]
        if key != self._cursor_key:
            self._cursor_key = key
            self.cursor_moved.emit()

    def can_step_file(self, delta: int) -> bool:
        """Whether stepping by delta whole files would actually move -- drives
        the `<< >>` buttons' enabled state."""
        files = self._checked_files()
        if not files:
            return False
        return 0 <= self._current_file_index(files) + delta < len(files)

    def _reseat_cursor(self) -> bool:
        """Keep the cursor on a checked sweep. Returns whether it moved."""
        selected = self.selected()
        if not selected:
            moved = self._cursor_key is not None
            self._cursor_key = None
            return moved
        if self._cursor_key is not None and any(ds.key == self._cursor_key for ds in selected):
            return False
        self._cursor_key = selected[0].key
        return True
