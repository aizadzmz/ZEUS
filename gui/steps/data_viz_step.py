"""Step 1: load files, choose what to work on, and look at the raw spectra."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import style
from gui.figure_panes import PgFigurePane
from gui.files_panel import FilesAndSetsPanel
from gui.segmented import SegmentedControl
from gui.selection import SweepSelection
from gui.steps.base import StepPage, group_box, section_label


class DataVizStep(StepPage):
    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # ---------------------------------------------------------- settings

        load_box, load_layout = group_box("Current file input")

        open_row = QHBoxLayout()
        open_row.setSpacing(style.GROUP_SPACING)
        self.open_button = QPushButton("Open EIS exports…")
        self.open_button.setToolTip(
            "Replace everything currently loaded with the file(s) you pick."
        )
        open_row.addWidget(self.open_button)

        self.add_files_button = QPushButton("Add files…")
        self.add_files_button.setToolTip(
            "Load more file(s) alongside what's already open, keeping all "
            "existing validation/DRT results."
        )
        open_row.addWidget(self.add_files_button)
        load_layout.addLayout(open_row)

        self.file_list = QListWidget()
        self.file_list.setToolTip("Loaded files. Select one and click Remove to drop it.")
        self.file_list.setMaximumHeight(90)
        load_layout.addWidget(self.file_list)

        self.remove_file_button = QPushButton("Remove selected file")
        self.remove_file_button.setProperty("variant", "quiet")
        load_layout.addWidget(self.remove_file_button)
        self.add_settings(load_box)

        self.add_display_mode_box(
            "Plot single vs multiple",
            "How the selected sweeps are drawn here. This does not change "
            "which sweeps an analysis runs over — that is the checklist under "
            "the plot — nor how the other steps draw them.",
        )

        style_container = QWidget()
        style_col = QVBoxLayout(style_container)
        style_col.setContentsMargins(0, 0, 0, 0)
        style_col.setSpacing(style.GROUP_SPACING)
        style_col.addWidget(section_label("Line vs markers"))
        marker_style = SegmentedControl(["Markers", "Line"])
        self.markers_radio = marker_style.button(0)
        self.line_radio = marker_style.button(1)
        style_col.addWidget(marker_style)
        self.add_settings(style_container)

        details_box, details_layout = group_box("Sweep details")
        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setToolTip(
            "Point counts for the selected sweeps, and which of them have "
            "been validated or fitted."
        )
        # Capped so it cannot grow with the selection and push the panel
        # off-screen; it keeps its own scrollbar.
        self.details_text.setMaximumHeight(style.TEXT_PANE_MAX_HEIGHT)
        details_layout.addWidget(self.details_text)
        self.add_settings(details_box)

        self.end_settings()

        # ----------------------------------------------------------- content

        header = QHBoxLayout()
        header.setContentsMargins(*style.CONTENT_MARGINS)
        header.setSpacing(style.GROUP_SPACING)
        header.addWidget(QLabel("Plot"))

        view = SegmentedControl(["Nyquist", "Bode"])
        # Named *_radio though they are QToolButtons: same QAbstractButton
        # API, and MainWindow._wire_steps reaches them by these names.
        self.nyquist_view_radio = view.button(0)
        self.nyquist_view_radio.setToolTip("-Z'' against Z', on equal-aspect axes.")
        self.bode_view_radio = view.button(1)
        self.bode_view_radio.setToolTip(
            "|Z| (filled circles, left axis) and -phase (hollow circles, right "
            "axis) against frequency, with frequency and |Z| drawn as decades. "
            "A fixed view — use Auto-Scale or Replot to reframe it."
        )
        header.addWidget(view)
        header.addStretch()
        self.add_content_layout(header)

        self.spectrum_pane = PgFigurePane(with_overlay_actions=True)
        self.add_content(self.spectrum_pane, stretch=3)

        self.files_panel = FilesAndSetsPanel(selection)
        self.add_content(self.files_panel, stretch=1)
