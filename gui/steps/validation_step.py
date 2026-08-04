"""Step 2: check the spectra are trustworthy, and throw away the points that
aren't."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.figure_panes import PgFigureListPane, PgFigurePane
from gui.segmented import SegmentedControl
from gui.selection import SweepSelection
from gui.steps.base import StepPage, group_box, group_form
from gui.sweep_pager import SweepPager

# How many residuals figures Combined view offers to draw at once. Each is a
# full canvas render; Single view never exceeds one and needs no cap.
DEFAULT_RESIDUALS_LIMIT = 5


class ValidationStep(StepPage):
    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # ---------------------------------------------------------- settings

        self.add_display_mode_box(
            "Plot view",
            "One at a time shows a single sweep with its own residual plot "
            "below, stepped through with the ‹ › controls. Combined overlays "
            "every selected sweep and collapses the residuals.",
        )

        filter_box, filter_layout = group_box("Filtering")
        self.inductive_check = QCheckBox("Remove inductive tail (Im(Z) > 0)")
        filter_layout.addWidget(self.inductive_check)

        self.eraser_check = QCheckBox("Eraser (click points to mask/unmask)")
        self.eraser_check.setToolTip(
            "Click a point on the spectrum above to remove it, or a removed "
            "(grey ×) point to restore it. Works on this step's plot and on "
            "the Data Visualisation one. On the Bode plot the grey × markers "
            "are on the |Z| series. Manual edits override the filter above "
            "and the outlier threshold below, and are cleared when a "
            "different file is opened.\n\n"
            "Hiding the 'Removed' series via its legend entry also stops "
            "those points responding to clicks."
        )
        filter_layout.addWidget(self.eraser_check)
        self.add_settings(filter_box)

        valid_box, valid_form = group_form(
            "Validation",
            "Kramers-Kronig checks linearity/causality via a lin-KK fit, on "
            "the impedance representation only — fitting the admittance "
            "representation as well costs roughly 3x the time and mainly "
            "helps spectra with negative differential resistance. "
            "Z-HIT reconstructs the modulus from the phase data and is good "
            "at catching non-steady-state artifacts such as low-frequency "
            "drift; it is also far quicker, since it does no model fitting.",
        )
        method = SegmentedControl(["Kramers-Kronig", "Z-HIT"])
        # Named *_radio though they are QToolButtons: same QAbstractButton
        # API, and MainWindow._wire_steps reaches them by these names.
        self.kk_radio = method.button(0)
        self.zhit_radio = method.button(1)
        valid_form.addRow("Method", method)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(2.0)
        self.threshold_spin.setToolTip(
            "Outlier threshold. Points whose relative residual (real or "
            "imaginary) exceeds this percentage are removed."
        )
        valid_form.addRow("Threshold (%)", self.threshold_spin)

        self.run_validation_button = QPushButton()
        # The one action this step exists to perform.
        self.run_validation_button.setProperty("variant", "primary")
        valid_form.addRow(self.run_validation_button)
        self.add_settings(valid_box)

        residuals_box, residuals_form = group_form("Residual plot")
        self.residuals_limit_spin = QSpinBox()
        self.residuals_limit_spin.setMinimum(1)
        self.residuals_limit_spin.setMaximum(50)
        self.residuals_limit_spin.setValue(DEFAULT_RESIDUALS_LIMIT)
        self.residuals_limit_spin.setToolTip(
            "How many residual figures to draw at once. Only applies in "
            "Combined view — drawing one sweep at a time never exceeds one, "
            "so it draws itself with no trigger."
        )
        residuals_form.addRow("Show at most", self.residuals_limit_spin)

        self.residuals_plot_button = QPushButton("Plot residuals")
        residuals_form.addRow(self.residuals_plot_button)

        self.residuals_status_label = QLabel()
        self.residuals_status_label.setWordWrap(True)
        self.residuals_status_label.setProperty("state", "muted")
        residuals_form.addRow(self.residuals_status_label)
        self.add_settings(residuals_box)

        export_box, export_layout = group_box("Export")
        self.export_image_button = QPushButton("Save spectrum as image…")
        self.export_image_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_image_button)
        self.add_settings(export_box)

        self.end_settings()

        # ----------------------------------------------------------- content
        #
        # A splitter, not fixed halves: Combined view collapses the residuals
        # pane outright and gives the spectrum the whole height.
        self.splitter = QSplitter(Qt.Vertical)
        self.spectrum_pane = PgFigurePane(with_overlay_actions=True)
        self.splitter.addWidget(self.spectrum_pane)

        lower = QWidget()
        lower_col = QVBoxLayout(lower)
        lower_col.setContentsMargins(0, 0, 0, 0)
        lower_col.setSpacing(0)
        self.pager = SweepPager(selection)
        lower_col.addWidget(self.pager)
        self.residuals_pane = PgFigureListPane()
        lower_col.addWidget(self.residuals_pane, stretch=1)
        self.splitter.addWidget(lower)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        # Pixels, not ratios -- setSizes takes real heights. The residuals
        # list asks for a tall minimum (one 340px figure per sweep), so
        # anything smaller opens with the spectrum crushed to a sliver.
        self.splitter.setSizes([560, 380])
        self.splitter.setChildrenCollapsible(False)
        self.add_content(self.splitter, stretch=1)

    def set_residuals_visible(self, visible: bool) -> None:
        """Collapse the residuals half so the spectrum fills the step."""
        self.splitter.widget(1).setVisible(visible)
