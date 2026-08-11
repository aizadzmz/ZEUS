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

from gui.figure_panes import PgFigurePane, PgSingleFigurePane
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
            "Display Option",
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
            "Kramers-Kronig or Z-HIT.",
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
            "Outlier threshold. Points with relative residuals exceeding this percentage are removed."
        )
        valid_form.addRow("Threshold [%]", self.threshold_spin)

        self.soft_limit_spin = _percent_spin(2.0)
        self.soft_limit_spin.setToolTip(
            "The worst point between the soft and hard limit is removed. \n"
            "Validation is repeated again until no points fall between the soft \n"
            "and hard limits or until a maximum number of run set below."
        )
        valid_form.addRow("Soft limit [%]", self.soft_limit_spin)

        self.hard_limit_spin = _percent_spin(5.0)
        self.hard_limit_spin.setToolTip(
            "All points above the hard limit are removed immediately."
        )
        valid_form.addRow("Hard limit [%]", self.hard_limit_spin)

        self.max_removed_spin = QSpinBox()
        self.max_removed_spin.setMinimum(1)
        self.max_removed_spin.setMaximum(200)
        self.max_removed_spin.setValue(10)
        self.max_removed_spin.setToolTip(
            "How many points the loop may remove before stopping to \n"
            "to prevent crashing."
        )
        valid_form.addRow("Max removed", self.max_removed_spin)

        # Last of the settings the run reads, and directly under the limits it
        # governs: whichever convention is picked here is the one they reject
        # on, whether that is the Basic threshold or the Advanced pair above.
        definition = SegmentedControl(["ΔZ / |Z|", "ΔZ′ / Z′"])
        self.residual_modulus_radio = definition.button(0)
        self.residual_modulus_radio.setToolTip(
            "Residuals defined as: ΔZ′/|Z| and ΔZ″/|Z|."
        )
        self.residual_component_radio = definition.button(1)
        self.residual_component_radio.setToolTip(
            "Residuals defined as: ΔZ′/Z′ and ΔZ″/Z″."
        )
        valid_form.addRow("Residuals", definition)

        self.run_validation_button = QPushButton()
        # The one action this step exists to perform.
        self.run_validation_button.setProperty("variant", "primary")
        valid_form.addRow(self.run_validation_button)

        # Outputs, so they sit below the action that produces them rather than
        # between the settings and the button. Empty until a run reports back.
        self.prune_status_label = QLabel()
        self.prune_status_label.setWordWrap(True)
        self.prune_status_label.setProperty("state", "muted")
        valid_form.addRow(self.prune_status_label)

        self.residuals_status_label = QLabel()
        self.residuals_status_label.setWordWrap(True)
        self.residuals_status_label.setProperty("state", "muted")
        valid_form.addRow(self.residuals_status_label)
        self.add_settings(valid_box)

        self.basic_radio.toggled.connect(self._sync_mode_rows)
        self._sync_mode_rows()

        # Kept ordered at the widget, so prune_iteratively can treat soft > hard
        # as the programming error it would be rather than a user typo.
        self.soft_limit_spin.valueChanged.connect(self.hard_limit_spin.setMinimum)
        self.hard_limit_spin.valueChanged.connect(self.soft_limit_spin.setMaximum)
        self._clamp_limits()

        export_box, export_layout = quiet_group()
        self.export_results_button = QPushButton("Export validated data…")
        self.export_results_button.setProperty("variant", "quiet")
        self.export_results_button.setToolTip(
            "Export as .csv or .z"
        )
        export_layout.addWidget(self.export_results_button)

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
        self.residuals_pane = PgSingleFigurePane()
        lower_col.addWidget(self.residuals_pane, stretch=1)
        self.splitter.addWidget(lower)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        # Pixels, not ratios -- setSizes takes real heights. Only the opening
        # split: the residual figure claims no minimum of its own, so dragging
        # the handle resizes it rather than pushing it under a scrollbar.
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

    @property
    def residual_mode(self) -> str:
        """One of core.validation.RESIDUAL_MODES: what the limits above and the
        residual plot below both measure a point against."""
        # Deferred: core.validation drags in pyimpspec, and this step is built
        # during startup, before any analysis has been asked for.
        from core.validation import RESIDUAL_BY_COMPONENT, RESIDUAL_BY_MODULUS

        return (
            RESIDUAL_BY_MODULUS
            if self.residual_modulus_radio.isChecked()
            else RESIDUAL_BY_COMPONENT
        )

    def set_residual_mode(self, mode: Optional[str]) -> None:
        """Restore the convention without emitting -- for a session load, whose
        one _refresh() at the end covers every widget it touched."""
        from core.validation import RESIDUAL_BY_COMPONENT

        if mode is None:
            return
        buttons = (self.residual_modulus_radio, self.residual_component_radio)
        for button in buttons:
            button.blockSignals(True)
        buttons[1 if mode == RESIDUAL_BY_COMPONENT else 0].setChecked(True)
        for button in buttons:
            button.blockSignals(False)

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
