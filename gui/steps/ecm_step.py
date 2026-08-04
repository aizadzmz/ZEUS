"""Step 3b: fit an equivalent circuit and read off its parameters."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import style
from gui.figure_panes import CircuitDiagramPane, PgFigurePane
from gui.selection import SweepSelection
from gui.steps.base import (
    StepPage,
    add_combo_items,
    compact_combo,
    group_box,
    group_form,
    section_label,
)
from gui.sweep_pager import SweepPager


class ECMStep(StepPage):
    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        from core.ecm import CIRCUIT_PRESETS, FIT_METHODS, WEIGHT_FORMS

        # ---------------------------------------------------------- settings

        self.add_display_mode_box(
            "Plot view",
            "One at a time shows a single sweep's fit and its circuit, "
            "stepped through with the ‹ › controls. Combined overlays every "
            "selected sweep's fit.",
        )

        ecm_box, form = group_form(
            "ECM settings",
            "Fits an equivalent circuit to each selected sweep by complex "
            "non-linear least squares.",
        )

        form.addRow(section_label("Circuit"))

        self.preset_combo = compact_combo(QComboBox())
        add_combo_items(
            self.preset_combo, [(f"{name} — {cdc}", cdc) for name, cdc in CIRCUIT_PRESETS]
        )
        self.preset_combo.setToolTip(
            "Fills the circuit description code below. The code stays "
            "editable — the preset is only a starting point."
        )
        form.addRow("Preset", self.preset_combo)

        self.build_from_drt_button = QPushButton("Build circuit from DRT peaks")
        self.build_from_drt_button.setToolTip(
            "Derive the circuit from the selected sweep's DRT peak analysis: "
            "one parallel R-CPE pair per resolved peak, in series with the "
            "high-frequency resistance, with every value pre-filled from the "
            "peak's resistance and time constant. Fills the code below — it "
            "does not fit. Run peak deconvolution on the DRT step first."
        )
        form.addRow(self.build_from_drt_button)

        self.cdc_edit = QLineEdit()
        self.cdc_edit.setToolTip(
            "Boukamp notation: parentheses are parallel connections, square "
            "brackets series ones, so R(RC) is a resistor in series with a "
            "parallel RC pair. Elements: R C L Q W Ws Wo G H Zarc and the "
            "transmission-line family. Initial values and bounds can be "
            "pinned inline, e.g. R{R=5}(R{R=100/1/1e4}C)."
        )
        form.addRow("Code", self.cdc_edit)

        self.cdc_status_label = QLabel()
        self.cdc_status_label.setWordWrap(True)
        form.addRow(self.cdc_status_label)

        form.addRow(section_label("Fit"))

        self.method_combo = compact_combo(QComboBox())
        add_combo_items(self.method_combo, [(m, m) for m in FIT_METHODS])
        self.method_combo.setToolTip(
            "'least_squares' is fast and always reports parameter error "
            "bars. 'auto' tries every method and keeps the best-scoring one "
            "— several times slower, and the winner is often a gradient-free "
            "method that reports no error bars at all."
        )
        form.addRow("Method", self.method_combo)

        self.weight_combo = compact_combo(QComboBox())
        add_combo_items(self.weight_combo, [(w, w) for w in WEIGHT_FORMS])
        self.weight_combo.setToolTip(
            "How each residual is weighted. 'boukamp' (1/|Z|²) is the EIS "
            "standard and stops the high-impedance end of the spectrum from "
            "dominating the fit."
        )
        form.addRow("Weight", self.weight_combo)

        self.seed_check = QCheckBox("Seed fit from previous sweep")
        self.seed_check.setToolTip(
            "Start each fit from the previous sweep's fitted values instead "
            "of the generic defaults. Helps convergence across a cycling or "
            "degradation series, where consecutive spectra are similar — but "
            "one bad fit then seeds the next."
        )
        form.addRow(self.seed_check)

        self.run_button = QPushButton("Fit circuit")
        # The one action this step exists to perform.
        self.run_button.setProperty("variant", "primary")
        form.addRow(self.run_button)

        self.shown_label = QLabel("Show fit")
        self.shown_combo = compact_combo(QComboBox())
        self.shown_combo.setToolTip(
            "Which fitted circuit the plot above overlays. Every circuit you "
            "fit is kept, so you can switch between them to compare."
        )
        form.addRow(self.shown_label, self.shown_combo)
        self.shown_status_label = QLabel()
        self.shown_status_label.setWordWrap(True)
        self.shown_status_label.setProperty("state", "muted")
        form.addRow(self.shown_status_label)
        self.add_settings(ecm_box)

        export_box, export_layout = group_box("Export")
        self.export_params_button = QPushButton("Export ECM parameters…")
        self.export_params_button.setToolTip(
            "Write the fitted parameters, and the fit curve, for the sweep on "
            "screen. Use File ▸ Save session to keep everything instead."
        )
        self.export_params_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_params_button)
        self.export_image_button = QPushButton("Save fit plot as image…")
        self.export_image_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_image_button)
        self.add_settings(export_box)

        self.end_settings()

        # ----------------------------------------------------------- content

        splitter = QSplitter(Qt.Vertical)
        self.spectrum_pane = PgFigurePane(with_overlay_actions=True)
        splitter.addWidget(self.spectrum_pane)

        lower = QWidget()
        lower_col = QVBoxLayout(lower)
        lower_col.setContentsMargins(0, 0, 0, 0)
        lower_col.setSpacing(0)
        self.pager = SweepPager(selection)
        lower_col.addWidget(self.pager)

        circuit_splitter = QSplitter(Qt.Vertical)
        self.circuit_pane = CircuitDiagramPane()
        self.params_text = QPlainTextEdit()
        self.params_text.setReadOnly(True)
        # Fixed-pitch: the fit report is pre-aligned text.
        self.params_text.setFont(style.mono_font())
        circuit_splitter.addWidget(self.circuit_pane)
        circuit_splitter.addWidget(self.params_text)
        circuit_splitter.setStretchFactor(0, 3)
        circuit_splitter.setStretchFactor(1, 2)
        # Pixels, not ratios -- setSizes takes real heights.
        circuit_splitter.setSizes([280, 180])
        # Non-collapsible: a pane dragged shut looks like lost content.
        circuit_splitter.setChildrenCollapsible(False)
        lower_col.addWidget(circuit_splitter, stretch=1)
        splitter.addWidget(lower)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 470])
        splitter.setChildrenCollapsible(False)
        self.add_content(splitter, stretch=1)

        # Signals blocked: letting textChanged fire here would call
        # validate_cdc -> parse_cdc and pay the ~4 s pyimpspec import while the
        # window is still being built. Validation runs on the first edit
        # instead, by which point gui/app.py's warm-up has made it free.
        self.cdc_edit.blockSignals(True)
        self.cdc_edit.setText(CIRCUIT_PRESETS[0][1])
        self.cdc_edit.blockSignals(False)
