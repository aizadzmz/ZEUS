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
from gui.steps.base import (
    StepPage,
    group_box,
    group_form,
    quiet_group,
    set_row_visible,
)
from gui.sweep_pager import SweepPager

# How many residuals figures Multiple view offers to draw at once. Each is a
# full canvas render; Singular view never exceeds one and needs no cap.
DEFAULT_RESIDUALS_LIMIT = 5


def _percent_spin(value: float) -> QDoubleSpinBox:
    """A residual-limit spinbox: percent, two decimals, half-percent steps."""
    spin = QDoubleSpinBox()
    spin.setMinimum(0.0)
    spin.setMaximum(100.0)
    spin.setSingleStep(0.5)
    spin.setValue(value)
    return spin


class ValidationStep(StepPage):
    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # ---------------------------------------------------------- settings

        self.add_display_mode_box(
            "Plot Options",
            "Singular shows a single sweep with its own residual plot "
            "below, stepped through with the ‹ › controls. Multiple overlays "
            "every selected sweep and collapses the residuals.",
        )

        filter_box, filter_layout = group_box("Filtering")
        self.inductive_check = QCheckBox("Remove inductive tail (Im(Z) > 0)")
        filter_layout.addWidget(self.inductive_check)

        # The eraser lives on the plot overlay, not here: it locks this column
        # while it is on, so its own switch cannot sit inside it.
        eraser_hint = QLabel(
            "Point-by-point edits: use the Eraser button on the spectrum."
        )
        eraser_hint.setWordWrap(True)
        eraser_hint.setProperty("state", "muted")
        filter_layout.addWidget(eraser_hint)
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
        self._form = valid_form

        mode = SegmentedControl(["Basic", "Advanced"])
        self.basic_radio = mode.button(0)
        self.basic_radio.setToolTip(
            "Hard Limit Only"
        )
        self.advanced_radio = mode.button(1)
        self.advanced_radio.setToolTip(
            "Soft and Hard Limits"
        )
        valid_form.addRow("Mode", mode)

        self.threshold_spin = _percent_spin(2.0)
        self.threshold_spin.setToolTip(
            "Outlier threshold. Points whose relative residual (real or "
            "imaginary) exceeds this percentage are removed."
        )
        valid_form.addRow("Threshold [%]", self.threshold_spin)

        self.soft_limit_spin = _percent_spin(2.0)
        self.soft_limit_spin.setToolTip(
            "Where the iteration stops. Points above this but below the hard "
            "limit are removed one per pass, worst first, until none are left "
            "above it.\n\n"
            "Only read when a run starts — changing it does not re-prune "
            "what is already on screen."
        )
        valid_form.addRow("Soft limit [%]", self.soft_limit_spin)

        self.hard_limit_spin = _percent_spin(5.0)
        self.hard_limit_spin.setToolTip(
            "Points above this are removed immediately, however many there "
            "are, and re-checked on every pass. Must be at or above the soft "
            "limit."
        )
        valid_form.addRow("Hard limit [%]", self.hard_limit_spin)

        self.max_removed_spin = QSpinBox()
        self.max_removed_spin.setMinimum(1)
        self.max_removed_spin.setMaximum(200)
        self.max_removed_spin.setValue(10)
        self.max_removed_spin.setToolTip(
            "How many points the loop may remove before giving up, so a "
            "spectrum that never settles cannot cost an unbounded number of "
            "re-validations. It stops just short of exceeding this and says "
            "so; a sweep is also never pruned below 8 points.\n\n"
            "Not a cap on what you end up losing: whatever is still over the "
            "hard limit when the loop stops is rejected anyway, as it would "
            "be in Basic mode."
        )
        valid_form.addRow("Max removed", self.max_removed_spin)

        self.run_validation_button = QPushButton()
        # The one action this step exists to perform.
        self.run_validation_button.setProperty("variant", "primary")
        valid_form.addRow(self.run_validation_button)

        # An output, so it sits with the action that produces it. Empty until
        # an advanced run reports back.
        self.prune_status_label = QLabel()
        self.prune_status_label.setWordWrap(True)
        self.prune_status_label.setProperty("state", "muted")
        valid_form.addRow(self.prune_status_label)
        self.add_settings(valid_box)

        self.basic_radio.toggled.connect(self._sync_mode_rows)
        self._sync_mode_rows()

        # Kept ordered at the widget, so prune_iteratively can treat soft > hard
        # as the programming error it would be rather than a user typo.
        self.soft_limit_spin.valueChanged.connect(self.hard_limit_spin.setMinimum)
        self.hard_limit_spin.valueChanged.connect(self.soft_limit_spin.setMaximum)
        self._clamp_limits()

        residuals_box, residuals_form = group_form("Residual Plot")
        self.residuals_limit_spin = QSpinBox()
        self.residuals_limit_spin.setMinimum(1)
        self.residuals_limit_spin.setMaximum(50)
        self.residuals_limit_spin.setValue(DEFAULT_RESIDUALS_LIMIT)
        self.residuals_limit_spin.setToolTip(
            "How many residual figures to draw at once. Only applies in "
            "Multiple view — Singular never exceeds one, so it draws itself "
            "with no trigger."
        )
        residuals_form.addRow("Show at most", self.residuals_limit_spin)

        self.residuals_plot_button = QPushButton("Plot residuals")
        self.residuals_plot_button.setProperty("variant", "secondary")
        residuals_form.addRow(self.residuals_plot_button)

        self.residuals_status_label = QLabel()
        self.residuals_status_label.setWordWrap(True)
        self.residuals_status_label.setProperty("state", "muted")
        residuals_form.addRow(self.residuals_status_label)
        self.add_settings(residuals_box)

        export_box, export_layout = quiet_group()
        self.export_image_button = QPushButton("Save spectrum as image…")
        self.export_image_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_image_button)
        self.add_settings(export_box)

        self.end_settings()

        # ----------------------------------------------------------- content
        #
        # A splitter, not fixed halves: Multiple view collapses the residuals
        # pane outright and gives the spectrum the whole height.
        self.splitter = QSplitter(Qt.Vertical)
        self.spectrum_pane = PgFigurePane(with_overlay_actions=True, with_eraser=True)
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

        # Everything on this page except the spectrum, for the eraser lock:
        # paging to another sweep or reading residuals is "the rest of the
        # work", and waits until the eraser is switched off.
        self.lock_with_settings(lower)

    @property
    def mode(self) -> str:
        """"Basic" or "Advanced" -- one threshold pass, or the iterative prune
        between two limits."""
        return "Basic" if self.basic_radio.isChecked() else "Advanced"

    @property
    def reject_threshold(self) -> float:
        """The limit a stored result's points are rejected against on every
        redraw. The hard limit is the advanced mode's equivalent: the soft one
        is spent during the run and cannot be re-applied without re-running."""
        return (
            self.threshold_spin.value()
            if self.mode == "Basic"
            else self.hard_limit_spin.value()
        )

    def _clamp_limits(self) -> None:
        """Pen each of the two limits in on the other's current value."""
        self.hard_limit_spin.setMinimum(self.soft_limit_spin.value())
        self.soft_limit_spin.setMaximum(self.hard_limit_spin.value())

    def set_limits(self, threshold=None, soft=None, hard=None, max_removed=None) -> None:
        """Restore the limit spinboxes without emitting -- for a session load,
        whose one _refresh() at the end covers every widget it touched."""
        spins = (
            self.threshold_spin,
            self.soft_limit_spin,
            self.hard_limit_spin,
            self.max_removed_spin,
        )
        for spin in spins:
            spin.blockSignals(True)
        # Widened before any of them is set: soft and hard pen each other in, so
        # a saved pair below the current one would otherwise be clipped up
        # against the values it is replacing.
        self.soft_limit_spin.setMaximum(100.0)
        self.hard_limit_spin.setMinimum(0.0)
        for value, spin in zip((threshold, soft, hard, max_removed), spins):
            if value is not None:
                spin.setValue(value)
        self._clamp_limits()
        for spin in spins:
            spin.blockSignals(False)

    def set_mode(self, mode: str) -> None:
        """Restore the mode without emitting -- for a session load, whose one
        _refresh() at the end covers every widget it touched."""
        advanced = mode == "Advanced"
        for button in (self.basic_radio, self.advanced_radio):
            button.blockSignals(True)
        (self.advanced_radio if advanced else self.basic_radio).setChecked(True)
        for button in (self.basic_radio, self.advanced_radio):
            button.blockSignals(False)
        self._sync_mode_rows()

    def _sync_mode_rows(self) -> None:
        """Show only the limits the current mode actually reads."""
        advanced = self.mode == "Advanced"
        set_row_visible(self._form, self.threshold_spin, not advanced)
        for field in (self.soft_limit_spin, self.hard_limit_spin, self.max_removed_spin):
            set_row_visible(self._form, field, advanced)
        if not advanced:
            self.prune_status_label.clear()

    def set_residuals_visible(self, visible: bool) -> None:
        """Collapse the residuals half so the spectrum fills the step."""
        self.splitter.widget(1).setVisible(visible)
