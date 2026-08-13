"""Step 3a: the distribution of relaxation times, and the peaks it resolves."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from gui import style
from gui.figure_panes import PgFigurePane
from gui.segmented import SegmentedControl
from gui.selection import SweepSelection
from gui.steps.base import (
    StepPage,
    add_combo_items,
    compact_combo,
    group_box,
    group_form,
    mirror_row_tooltips,
    quiet_group,
    section_label,
    set_row_enabled,
    set_row_label,
)
from gui.sweep_pager import SweepPager


class _TwoLineLabel(QLabel):
    """A label of exactly two lines: one per line of the text it is given,
    each elided to the width available rather than wrapped.

    A wrapped QLabel in a form layout keeps the height it was first laid out
    at, so a value spilling onto a third rendered line lost it. The size hints
    reserve the two lines from the live font metrics, so the reservation holds
    after the stylesheet has changed the font.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full_text = ""
        self.setWordWrap(False)
        self.setText(text)

    @property
    def full_text(self) -> str:
        """The text as given, before eliding -- what the label means, as
        opposed to what currently fits."""
        return self._full_text

    def setText(self, text: str) -> None:
        self._full_text = text
        self._apply_elision()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self) -> None:
        metrics = self.fontMetrics()
        # A hard floor, not a size hint a cramped form layout may compress below; guarded, since setMinimumHeight can trigger a resize back through here.
        two_lines = metrics.lineSpacing() * 2
        if self.minimumHeight() != two_lines:
            self.setMinimumHeight(two_lines)

        # A couple of pixels back for the frame, so the last glyph is not shaved by a rounding error.
        available = max(self.width() - 2, 10)
        super().setText(
            "\n".join(
                metrics.elidedText(line, Qt.ElideRight, available)
                for line in self._full_text.split("\n")
            )
        )

    def _two_lines(self, height: int) -> int:
        return max(height, self.fontMetrics().lineSpacing() * 2)

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setHeight(self._two_lines(hint.height()))
        return hint

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(self._two_lines(hint.height()))
        return hint


def _titleize(rbf_type: str) -> str:
    """'c2-matern' -> 'C2 Matern', 'piecewise-linear' -> 'Piecewise Linear'."""
    return " ".join(word.capitalize() for word in rbf_type.split("-"))


class DRTStep(StepPage):
    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Plain tuples of option strings; core.drt keeps its pyimpspec imports inside its functions, so reading these stays free.
        from core.drt import MAX_PEAKS, RBF_TYPES
        from core.ecm import DIFFUSION_PRESETS

        # ---------------------------------------------------------- settings

        self.add_display_mode_box(
            "Disply Option",
            "Singular shows a single sweep's DRT, stepped through with "
            "the ‹ › controls. Multiple draws every selected sweep's DRT on "
            "one figure.",
        )

        settings_box, form = group_form(
            "DRT Settings",
            "Applied to each selected sweep's currently unmasked points.\n "
            "Settings the chosen method does not use are greyed out.",
        )
        self._form = form

        # First, because it decides which of the rows below matter; see _sync_relevance.
        method = SegmentedControl(["TR-RBF", "Bayesian"])
        # Named *_radio though they are QToolButtons: same QAbstractButton API, and MainWindow._wire_steps reaches them by these names.
        self.trrbf_radio = method.button(0)
        self.trrbf_radio.setToolTip(
            "Tikhonov Regularization-Radial Basis Function"
        )
        self.bayesian_radio = method.button(1)
        self.bayesian_radio.setToolTip(
            "TR-RBF with Bayesian credible intervals by Hamiltonian Monte Carlo "
            "sampling. Computationally expensive.\n\n"
            "The interval is what this method adds: the DRT plot gains a shaded "
            "99% band and a dashed posterior mean in the sweep's own colour, and "
            "a CSV export gains three columns for them."
        )
        # On the control, so the "Method" label has something to mirror; the two buttons keep their own, which win on hover.
        method.setToolTip(
            "Which solver computes the distribution. Both fit the same "
            "TR-RBF model; Bayesian adds an uncertainty band at a large "
            "cost in run time."
        )
        form.addRow("Method", method)

        form.addRow(
            section_label(
                "Discretisation",
                "How the continuous distribution is represented numerically "
                "before it is fitted: the basis function each collocation "
                "point contributes, and which part of the impedance is used.",
            )
        )

        self.rbf_combo = compact_combo(QComboBox())
        add_combo_items(self.rbf_combo, [(_titleize(v), v) for v in RBF_TYPES])
        # Maps to rbf_type upstream.
        self.rbf_combo.setToolTip(
            "Shape of the basis function the distribution is discretised onto. "
            "Piecewise Linear uses no basis function, so it ignores the RBF "
            "shape settings below."
        )
        form.addRow("Basis function", self.rbf_combo)

        self.mode_combo = compact_combo(QComboBox())
        add_combo_items(
            self.mode_combo,
            [
                ("Re + Im", "complex"),
                ("Re only", "real"),
                ("Im only", "imaginary"),
            ],
        )
        self.mode_combo.setToolTip(
            "Which part of the impedance is fitted. Re + Im uses the whole "
            "spectrum and is the default; the single-part fits are mainly for "
            "checking one against the other."
        )
        form.addRow("Fit to", self.mode_combo)

        self.remove_inductive_check = QCheckBox("Remove inductive tail (Im(Z) > 0)")
        self.remove_inductive_check.setToolTip(
            "Drop the points with a positive imaginary impedance before "
            "computing the DRT, whether or not the Validation step's filter of "
            "the same name is on.\n\n"
            "This one applies to the DRT alone: it leaves the sweep's own mask "
            "where it is, so validation results stay valid and the Validation "
            "plot is unchanged. The measured plot above shows the dropped "
            "points as removed, so you can see what the DRT ran on."
        )
        form.addRow(self.remove_inductive_check)

        self.subtract_diffusion_check = QCheckBox(
            "[BETA] Subtract diffusion tail (fitted)"
        )
        self.subtract_diffusion_check.setToolTip(
            "Fit the model below to each sweep and subtract the fitted "
            "diffusion element's own impedance, so the low-frequency tail "
            "stops dominating the longest time constants of the DRT.\n\n"
            "Unlike the filter above, this keeps every point: the tail is "
            "flattened rather than dropped. It is also a subtraction of a "
            "*model*, so the fit quality shown below is part of the result — "
            "a poor fit subtracts a tail that was never measured.\n\n"
            "Pairs with the filter above on a cell with cabling inductance. "
            "None of these models has an inductor, so the inductive points are "
            "always held out of the fit; but they are not removed by this "
            "setting, and on a cell whose inductive tail is larger than its "
            "arc they will still dominate the plot until the filter above is "
            "on too.\n\n"
            "Scoped to the DRT alone, like the filter above: the sweep's own "
            "data is untouched, so Validation and ECM are unaffected."
        )
        form.addRow(self.subtract_diffusion_check)

        self.diffusion_cdc_combo = compact_combo(QComboBox())
        add_combo_items(self.diffusion_cdc_combo, [(n, c) for n, c in DIFFUSION_PRESETS])
        self.diffusion_cdc_combo.setToolTip(
            "The circuit fitted before subtracting. Each one ends in a "
            "diffusion element in series with an ohmic resistance and one "
            "R-CPE arc, and it is the series element alone that is removed.\n\n"
            "Series is the point: only an element in series with the rest "
            "contributes its own impedance to the total, so only then does "
            "subtracting it leave the rest of the spectrum behind."
        )
        form.addRow("Diffusion model", self.diffusion_cdc_combo)

        # Two lines by construction: the element on the first, the fit quality on the second.
        self.diffusion_status_label = _TwoLineLabel("—")
        self.diffusion_status_label.setProperty("state", "muted")
        # Top-aligned so the first line does not drift down the reserved space on one-line messages.
        self.diffusion_status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.diffusion_status_label.setToolTip(
            "The fitted diffusion parameters actually subtracted from the "
            "sweep on screen, and the fit's pseudo chi-squared. Shown because "
            "a subtraction rewrites measured points: this is what says whether "
            "the flattened tail is a cleaned spectrum or an invented one."
        )
        # Kept so MainWindow can append a failed fit's full message without losing the explanation the label has no room for.
        self.diffusion_status_tooltip = self.diffusion_status_label.toolTip()
        # Spans both columns, so it carries its own "Subtracted:" prefix: in the value column "pseudo chi-squared 0.0114" alone measured 208 px against the 206 available, and was clipped.
        form.addRow(self.diffusion_status_label)

        self.inductance_check = QCheckBox("Include series inductance (L)")
        self.inductance_check.setToolTip(
            "Add a series inductance to the model, so cabling and cell "
            "inductance are fitted rather than smeared into the shortest time "
            "constants. Distinct from the filter above, which discards the "
            "inductive points instead of modelling them."
        )
        # Spans both columns: a checkbox is a complete statement, not a label/value pair.
        form.addRow(self.inductance_check)

        form.addRow(
            section_label(
                "Regularisation",
                "The smoothness penalty that keeps the inversion stable. "
                "Which derivative of the distribution is penalised, and how "
                "strongly (λ) — the main trade-off between resolving peaks "
                "and amplifying noise.",
            )
        )

        self.derivative_combo = compact_combo(QComboBox())
        add_combo_items(self.derivative_combo, [("1st order", 1), ("2nd order", 2)])
        # Only 1st/2nd order Tikhonov are implemented upstream; no 0th order.
        self.derivative_combo.setToolTip(
            "Order of the derivative the regularisation penalises. 2nd order "
            "smooths harder, which merges closely-spaced peaks."
        )
        self.derivative_combo.setCurrentIndex(0)
        form.addRow("Derivative order", self.derivative_combo)

        self.cv_combo = compact_combo(QComboBox())
        add_combo_items(
            self.cv_combo,
            [
                ("Custom", ""),
                ("GCV", "gcv"),
                ("mGCV", "mgcv"),
                ("rGCV", "rgcv"),
                ("Re-Im", "re-im"),
                ("L-curve", "lc"),
            ],
        )
        self.cv_combo.setToolTip(
            "How λ is chosen. 'Custom' uses the value below directly; every "
            "other option optimises λ and treats that value as the starting "
            "point. The λ actually used is reported under Run DRT."
        )
        form.addRow("λ selection", self.cv_combo)

        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setMinimum(1e-10)
        self.lambda_spin.setMaximum(10.0)
        self.lambda_spin.setSingleStep(0.001)
        self.lambda_spin.setValue(0.001)
        self.lambda_spin.setToolTip(
            "Regularisation strength. Too low and noise shows up as spurious "
            "peaks; too high and real peaks merge. Used directly when λ "
            "selection is 'Custom', otherwise the optimiser's starting value — "
            "which is why the label changes rather than the row greying out."
        )
        form.addRow("λ", self.lambda_spin)

        self._shape_header = section_label(
            "RBF Shape",
            "How wide each radial basis function is, which sets the finest "
            "detail the distribution can carry. Ignored when the basis "
            "function is Piecewise Linear, which uses none.",
        )
        form.addRow(self._shape_header)

        # Paired on one row: the combo decides whether the spinbox is an FWHM coefficient or a shape factor.
        shape_row = QWidget()
        shape_layout = QHBoxLayout(shape_row)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.setSpacing(style.GROUP_SPACING)
        self.shape_control_combo = compact_combo(QComboBox())
        add_combo_items(
            self.shape_control_combo,
            [("FWHM coefficient", "fwhm"), ("Shape factor", "factor")],
        )
        shape_layout.addWidget(self.shape_control_combo, stretch=1)
        self.shape_coeff_spin = QDoubleSpinBox()
        self.shape_coeff_spin.setDecimals(4)
        self.shape_coeff_spin.setMinimum(0.0001)
        self.shape_coeff_spin.setMaximum(10.0)
        self.shape_coeff_spin.setSingleStep(0.05)
        self.shape_coeff_spin.setValue(0.5)
        shape_layout.addWidget(self.shape_coeff_spin)
        self._shape_row = shape_row
        # On the container, not the two children: it is the row's field widget, so mirror_row_tooltips copies it onto the "Shape control" label.
        shape_row.setToolTip(
            "How the width beside it is specified. 'FWHM coefficient' sets the "
            "full width at half maximum as a fraction of the spacing between "
            "adjacent time constants, \n so the shape follows the frequency grid; "
            "'Shape factor' sets the basis function's own width parameter "
            "directly. Smaller widths resolve more peaks and admit more noise."
        )
        form.addRow("Shape control", shape_row)

        self._sampling_header = section_label(
            "Sampling",
            "Monte Carlo draws behind the Bayesian credible intervals. Plain "
            "TR-RBF is a point estimate and does no sampling, so these rows "
            "grey out for it.",
        )
        form.addRow(self._sampling_header)

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setMinimum(1000)
        self.num_samples_spin.setMaximum(100000)
        self.num_samples_spin.setSingleStep(500)
        self.num_samples_spin.setValue(1000)
        self.num_samples_spin.setToolTip(
            "Draws used for the credible intervals (Bayesian). Must be >= 1000; "
            "more is more accurate and slower. Plain TR-RBF does no sampling."
        )
        form.addRow("Samples", self.num_samples_spin)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setMinimum(0)
        self.timeout_spin.setMaximum(36000)
        self.timeout_spin.setSingleStep(60)
        self.timeout_spin.setValue(300)
        self.timeout_spin.setToolTip(
            "The Bayesian credible-interval sampler can be extremely slow "
            "(tens of minutes for even modest sweeps). The run is stopped once "
            "this many seconds pass, from outside if need be — the sampler can "
            "reach a state it never leaves.\n\n"
            "0 disables the limit, and then a stalled run has to be waited out."
        )
        form.addRow("Timeout [s]", self.timeout_spin)
        # After the last row: the labels addRow() builds from a string have no tooltip, so hovering one would show the card's instead.
        mirror_row_tooltips(form)
        self.add_settings(settings_box)

        run_box, run_layout = group_box("Run DRT")

        # One button, since the method is picked in the settings above -- which is what lets the irrelevant rows grey out.
        self.run_drt_button = QPushButton("Run DRT")
        self.run_drt_button.setProperty("variant", "primary")
        run_layout.addWidget(self.run_drt_button)

        # An output, so it sits with the action that produces it rather than among the inputs above.
        self._lambda_readout = QWidget()
        lambda_row = QHBoxLayout(self._lambda_readout)
        lambda_row.setContentsMargins(0, 0, 0, 0)
        lambda_row.setSpacing(style.FORM_H_SPACING)
        lambda_row.addWidget(QLabel("λ used"))
        self.optimal_lambda_label = QLabel("—")
        self.optimal_lambda_label.setToolTip(
            "The regularisation strength the most recent run actually used — "
            "the optimised value when λ selection is not 'Custom'. Shown when "
            "exactly one sweep is selected."
        )
        self.optimal_lambda_label.setProperty("state", "muted")
        lambda_row.addWidget(self.optimal_lambda_label, stretch=1)
        run_layout.addWidget(self._lambda_readout)
        self.add_settings(run_box)

        peak_box, peak_layout = group_box(
            "Peak Extraction",
            "Fits individual (skew) normal peaks to the most recently "
            "computed DRT result for each selected sweep.",
        )
        peak_form = QFormLayout()
        peak_form.setContentsMargins(0, 0, 0, 0)
        peak_form.setHorizontalSpacing(style.FORM_H_SPACING)
        peak_form.setVerticalSpacing(style.FORM_V_SPACING)
        self.num_peaks_spin = QSpinBox()
        self.num_peaks_spin.setMinimum(0)
        # Capped at what the simultaneous fit can actually solve; see core.drt.
        self.num_peaks_spin.setMaximum(MAX_PEAKS)
        self.num_peaks_spin.setValue(0)
        self.num_peaks_spin.setToolTip(
            f"Number of peaks to fit, largest first. 0 analyses every detected "
            f"peak. All of them are fitted in one go, so the limit is "
            f"{MAX_PEAKS}; a distribution resolving more than that is usually "
            f"fitting noise, and wants a larger λ."
        )
        peak_form.addRow("Peaks", self.num_peaks_spin)
        mirror_row_tooltips(peak_form)
        peak_layout.addLayout(peak_form)

        self.run_peak_analysis_button = QPushButton("Run peak extraction")
        self.run_peak_analysis_button.setProperty("variant", "secondary")
        peak_layout.addWidget(self.run_peak_analysis_button)

        self.peaks_table = QTableWidget(0, 0)
        self.peaks_table.setObjectName("peaksTable")
        self.peaks_table.setToolTip(
            "Time constant, height and resistance of each fitted peak."
        )
        self.peaks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peaks_table.setSelectionMode(QTableWidget.NoSelection)
        self.peaks_table.verticalHeader().setVisible(False)
        self.peaks_table.horizontalHeader().setStretchLastSection(True)
        self.peaks_table.setFont(style.mono_font())
        # Capped so it cannot push the panel off-screen; it keeps its scrollbar.
        self.peaks_table.setMaximumHeight(style.TEXT_PANE_MAX_HEIGHT)
        peak_layout.addWidget(self.peaks_table)
        self.add_settings(peak_box)

        export_box, export_layout = quiet_group()
        self.export_results_button = QPushButton("Export DRT results…")
        self.export_results_button.setToolTip(
            "Write the DRT curve, and the fitted peaks if any, for the sweep "
            "on screen. Use File ▸ Save session to keep everything instead.\n\n"
            "Pick the format in the save dialog:\n"
            "• DRT table (.csv) — τ and γ as written, plus a companion "
            "'_peaks.csv'.\n"
            "• ZView data file (.z) — the curve with τ mapped to a frequency "
            "so ZView can plot it, plus the peaks as an Rs(RC)(RC)… model "
            "(.mdl) ZView can fit from."
        )
        self.export_results_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_results_button)
        self.export_image_button = QPushButton("Save DRT plot as image…")
        self.export_image_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_image_button)
        self.add_settings(export_box)

        self.end_settings()

        # ----------------------------------------------------------- content

        splitter = QSplitter(Qt.Vertical)
        # Always the measured spectrum the DRT below was computed from.
        self.top_pane = PgFigurePane(with_overlay_actions=True)
        splitter.addWidget(self.top_pane)

        lower = QWidget()
        lower_col = QVBoxLayout(lower)
        lower_col.setContentsMargins(0, 0, 0, 0)
        lower_col.setSpacing(0)
        self.pager = SweepPager(selection)
        lower_col.addWidget(self.pager)
        self.drt_pane = PgFigurePane(with_overlay_actions=True)
        lower_col.addWidget(self.drt_pane, stretch=1)
        splitter.addWidget(lower)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        # Pixels, not ratios -- setSizes takes real heights. An even split, so both halves get the same room.
        splitter.setSizes([470, 470])
        splitter.setChildrenCollapsible(False)
        self.add_content(splitter, stretch=1)

        # The three settings that decide whether other rows apply; kept inside the step, being presentation only.
        for control in (self.trrbf_radio, self.bayesian_radio):
            control.toggled.connect(self._sync_relevance)
        self.rbf_combo.currentIndexChanged.connect(self._sync_relevance)
        self.cv_combo.currentIndexChanged.connect(self._sync_relevance)
        # Decides whether the inductance row applies; see _sync_relevance.
        self.mode_combo.currentIndexChanged.connect(self._sync_relevance)
        self.subtract_diffusion_check.toggled.connect(self._sync_relevance)
        self._sync_relevance()

    @property
    def method(self) -> str:
        """"trrbf" or "bayesian" -- which calculation Run DRT performs. Also
        decides which settings apply; see _sync_relevance."""
        return "trrbf" if self.trrbf_radio.isChecked() else "bayesian"

    def _sync_relevance(self) -> None:
        """Grey out the settings the current choices make inert, so the panel
        only ever offers controls that will reach the calculation.

        Every rule here is a property of pyimpspec's API, not a house style:

        - Plain TR-RBF passes credible_intervals=False, and num_samples/timeout
          are read only inside the credible-interval branch.
        - Piecewise-linear discretisation short-circuits _compute_epsilon to
          0.0, so rbf_shape and shape_coeff are discarded whatever they say.
        - _prepare_real_matrices takes no inductance argument, so "Re only"
          fits the same model whether or not the box is ticked.
        """
        form = self._form

        # λ is not inert once a cross-validation method is chosen -- it becomes that optimiser's starting value -- so it is relabelled, not greyed.
        set_row_label(
            form,
            self.lambda_spin,
            "λ" if self.cv_combo.currentData() == "" else "λ (initial)",
        )

        shaped = self.rbf_combo.currentData() != "piecewise-linear"
        self._shape_header.setEnabled(shaped)
        set_row_enabled(form, self._shape_row, shaped)

        # Left ticked but inert, rather than silently cleared: switching Fit to
        # back restores the setting the user chose.
        set_row_enabled(
            form, self.inductance_check, self.mode_combo.currentData() != "real"
        )

        # Not a pyimpspec rule but the same principle: neither the model nor the readout means anything with the filter off.
        subtracting = self.subtract_diffusion_check.isChecked()
        set_row_enabled(form, self.diffusion_cdc_combo, subtracting)
        set_row_enabled(form, self.diffusion_status_label, subtracting)

        # The header follows its rows: TR-RBF leaves both of them inert.
        samples = self.method == "bayesian"
        self._sampling_header.setEnabled(samples)
        set_row_enabled(form, self.num_samples_spin, samples)
        set_row_enabled(form, self.timeout_spin, samples)
