"""Step 3b: fit an equivalent circuit and read off its parameters."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import style
from gui.circuit_canvas import CircuitCanvas
from gui.figure_panes import PgFigurePane
from gui.segmented import SegmentedControl
from gui.selection import SweepSelection
from gui.steps.base import (
    StepPage,
    add_combo_items,
    compact_combo,
    group_box,
    group_form,
    quiet_group,
    section_label,
)
from gui.sweep_pager import SweepPager


def _label_for(value: str) -> str:
    """'least_squares' -> 'Least squares'. The stored value is unchanged; only
    the machine identifier is kept out of the UI."""
    return value.replace("_", " ").capitalize()


class ECMStep(StepPage):
    # No Singular/Multiple toggle: this step pages one sweep at a time, since the diagram and report below carry one sweep's values. "Fit circuit" still runs on the whole selection.
    fixed_display_mode = "Single"

    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        from core.ecm import CIRCUIT_PRESETS, FIT_METHODS, WEIGHT_FORMS

        # ---------------------------------------------------------- settings

        ecm_box, form = group_form(
            "ECM Settings",
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

        self.build_hint_label = QLabel(
            "Click the schematic below to edit it: a component to change its "
            "type, values and bounds, a ⊕ on the wire to insert one, or "
            "right-click for more."
        )
        self.build_hint_label.setWordWrap(True)
        self.build_hint_label.setProperty("state", "muted")
        form.addRow(self.build_hint_label)

        self.build_from_drt_button = QPushButton("Build circuit from DRT peaks")
        self.build_from_drt_button.setProperty("variant", "secondary")
        self.build_from_drt_button.setToolTip(
            "Derive the circuit from the selected sweep's DRT peak analysis: "
            "one parallel R-CPE pair per resolved peak, in series with the "
            "high-frequency resistance, with every value pre-filled from the "
            "peak's resistance and time constant. Fills the code below — it "
            "does not fit. Run peak extraction on the DRT step first."
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
        add_combo_items(self.method_combo, [(_label_for(m), m) for m in FIT_METHODS])
        self.method_combo.setToolTip(
            "'Least squares' is fast and always reports parameter error "
            "bars. 'Auto' tries every method and keeps the best-scoring one "
            "— several times slower, and the winner is often a gradient-free "
            "method that reports no error bars at all."
        )
        form.addRow("Method", self.method_combo)

        self.weight_combo = compact_combo(QComboBox())
        add_combo_items(self.weight_combo, [(_label_for(w), w) for w in WEIGHT_FORMS])
        self.weight_combo.setToolTip(
            "How each residual is weighted. 'Boukamp' (1/|Z|²) is the EIS "
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

        # No circuit picker: the step shows the circuit in the code box and nothing else, so a dozen guesses do not leave a dozen on screen. Earlier fits stay cached.
        self.shown_status_label = QLabel()
        self.shown_status_label.setWordWrap(True)
        self.shown_status_label.setProperty("state", "muted")
        form.addRow(self.shown_status_label)
        self.add_settings(ecm_box)

        # Between the settings and the exports, and given the column's spare
        # height: the fit report is the answer this step produces, so it reads
        # beside the settings that produced it rather than under the schematic.
        stats_box, stats_layout = group_box(
            "Fit Statistics",
            "Goodness-of-fit figures and the fitted parameter values for the "
            "sweep on screen.",
        )
        # Overriding _section's Maximum policy, the one card here meant to grow.
        stats_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.params_text = QPlainTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setProperty("role", "output")
        # Fixed-pitch: the fit report is pre-aligned text.
        self.params_text.setFont(style.mono_font())
        # A floor, not a cap: the report takes what the column has left and
        # scrolls itself, so a long one never pushes the exports off-panel.
        # Ignored, so the column is sized from the settings above it and this
        # takes the remainder -- a QPlainTextEdit's own size hint is several
        # hundred pixels and would force the scrollbar this move is undoing.
        self.params_text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.params_text.setMinimumHeight(120)
        # stretch, because QSizePolicy.Ignored carries no ExpandFlag: without it
        # the card's spare height is split with its title label.
        stats_layout.addWidget(self.params_text, stretch=1)
        self.add_settings(stats_box, stretch=1)

        export_box, export_layout = quiet_group()
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

        # No end_settings(): the statistics card above already holds the slack,
        # which keeps the exports pinned to the bottom of the panel.

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

        # The schematic has the whole lower half now that the report has moved
        # into the settings column, so no splitter divides it.
        self.circuit_pane = self._build_circuit_pane()
        lower_col.addWidget(self.circuit_pane, stretch=1)
        splitter.addWidget(lower)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 470])
        splitter.setChildrenCollapsible(False)
        self.add_content(splitter, stretch=1)

        # Signals blocked: letting textChanged fire here would call parse_cdc and pay the ~4 s pyimpspec import while the window is still being built.
        self.cdc_edit.blockSignals(True)
        self.cdc_edit.setText(CIRCUIT_PRESETS[0][1])
        self.cdc_edit.blockSignals(False)

    def _build_circuit_pane(self) -> QWidget:
        """The editable schematic, with its caption above and toolbar below."""
        pane = QWidget()
        column = QVBoxLayout(pane)
        column.setContentsMargins(*style.CONTENT_MARGINS)
        column.setSpacing(style.GROUP_SPACING)

        self.circuit_caption = QLabel()
        self.circuit_caption.setObjectName("diagramCaption")
        self.circuit_caption.setWordWrap(True)
        column.addWidget(self.circuit_caption)

        self.canvas = CircuitCanvas()
        column.addWidget(self.canvas, stretch=1)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(style.GROUP_SPACING)

        self.add_element_button = QToolButton()
        self.add_element_button.setText("Add element")
        self.add_element_button.setPopupMode(QToolButton.InstantPopup)
        self.add_element_button.setToolTip("Append a component to the end of the circuit.")
        self.add_element_menu = QMenu(self.add_element_button)
        self.add_element_button.setMenu(self.add_element_menu)
        toolbar.addWidget(self.add_element_button)

        self.undo_button = QToolButton()
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("Undo the last circuit edit (Ctrl+Z).")
        toolbar.addWidget(self.undo_button)

        self.redo_button = QToolButton()
        self.redo_button.setText("Redo")
        self.redo_button.setToolTip("Redo the last undone circuit edit (Ctrl+Shift+Z).")
        toolbar.addWidget(self.redo_button)

        self.clear_circuit_button = QToolButton()
        self.clear_circuit_button.setText("Clear")
        self.clear_circuit_button.setToolTip("Start from an empty circuit.")
        toolbar.addWidget(self.clear_circuit_button)

        toolbar.addStretch()

        self.values_segmented = SegmentedControl(["Initial", "Fitted"])
        self.values_segmented.setToolTip(
            "Which values the schematic is labelled with: the starting values "
            "that will be fitted, or the fitted result for the sweep on screen."
        )
        self.initial_values_radio = self.values_segmented.button(0)
        self.fitted_values_radio = self.values_segmented.button(1)
        toolbar.addWidget(self.values_segmented)

        column.addLayout(toolbar)
        return pane

    @property
    def values_mode(self) -> str:
        return "Fitted" if self.fitted_values_radio.isChecked() else "Initial"
