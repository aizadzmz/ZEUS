"""Step 1: load files, choose what to work on, and look at the raw spectra."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from gui import style
from gui.figure_panes import PgFigurePane
from gui.files_panel import FilesAndSetsPanel
from gui.segmented import SegmentedControl
from gui.selection import SweepSelection
from gui.steps.base import StepPage, group_box, group_form


class DataVizStep(StepPage):
    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # ---------------------------------------------------------- settings

        load_box, load_layout = group_box("Insert File(s)")

        open_row = QHBoxLayout()
        open_row.setSpacing(style.GROUP_SPACING)
        self.open_button = QPushButton("Clear All and Add Files…")
        # The one action this step exists to perform.
        self.open_button.setProperty("variant", "primary")
        self.open_button.setToolTip(
            "Replace everything currently loaded with the file(s) you pick."
        )
        open_row.addWidget(self.open_button)

        self.add_files_button = QPushButton("Add files…")
        self.add_files_button.setProperty("variant", "secondary")
        self.add_files_button.setToolTip(
            "Load more file(s) alongside what's already open."
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

        plot_box, plot_form = group_form(
            "Plot Options",
            "How the selected sweeps are drawn here. This does not change "
            "which sweeps an analysis runs over — that is the checklist under "
            "the plot — nor how the other steps draw them.",
        )
        # Both rows hold a segmented control, and the wider of the two would
        # otherwise trip group_form's WrapLongRows and drop its control onto a
        # line of its own, leaving the two rows looking unrelated.
        plot_form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        plot_form.addRow("Display", self.display_mode_control())

        marker_style = SegmentedControl(["Markers", "Line"])
        self.markers_radio = marker_style.button(0)
        self.line_radio = marker_style.button(1)
        plot_form.addRow("Style", marker_style)

        # A popup rather than rows here: the marker list is per loaded file, so
        # it cannot be laid out until the files are known.
        self.marker_style_button = QPushButton("Marker & line style…")
        self.marker_style_button.setProperty("variant", "secondary")
        self.marker_style_button.setToolTip(
            "Choose the marker shape, marker size and line width."
        )
        plot_form.addRow(self.marker_style_button)
        self.add_settings(plot_box)

        details_box, details_layout = group_box("Sweep Details")
        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setProperty("role", "output")
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
            "|Z| (solid circles, left axis) and -Φ (hollow circles, right "
            "axis) against frequency."
        )
        header.addWidget(view)
        header.addStretch()
        self.add_content_layout(header)

        self.spectrum_pane = PgFigurePane(with_overlay_actions=True, with_eraser=True)
        self.add_content(self.spectrum_pane, stretch=3)

        self.files_panel = FilesAndSetsPanel(selection)
        self.add_content(self.files_panel, stretch=1)

        # Everything on this page except the plot itself, for the eraser lock.
        self.lock_with_settings(view)
        self.lock_with_settings(self.files_panel)
