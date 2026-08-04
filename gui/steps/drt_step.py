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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
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
    section_label,
)
from gui.sweep_pager import SweepPager


def _titleize(rbf_type: str) -> str:
    """'c2-matern' -> 'C2 Matern', 'piecewise-linear' -> 'Piecewise Linear'."""
    return " ".join(word.capitalize() for word in rbf_type.split("-"))


class DRTStep(StepPage):
    def __init__(self, selection: SweepSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Plain tuples of option strings; core.drt keeps its pyimpspec imports
        # inside its functions so reading these here stays free.
        from core.drt import RBF_TYPES

        # ---------------------------------------------------------- settings

        self.add_display_mode_box(
            "Plot view",
            "One at a time shows a single sweep's DRT, stepped through with "
            "the ‹ › controls. Combined draws every selected sweep's DRT on "
            "one figure.",
        )

        settings_box, form = group_form(
            "DRT settings",
            "Settings for Tikhonov regularization + radial basis function "
            "discretization (TR-RBF) and the Bayesian Hilbert Transform "
            "(BHT), applied to each selected sweep's currently unmasked "
            "points.",
        )

        form.addRow(section_label("Discretisation"))

        self.rbf_combo = compact_combo(QComboBox())
        add_combo_items(self.rbf_combo, [(_titleize(v), v) for v in RBF_TYPES])
        self.rbf_combo.setToolTip("Method of Discretization (pyDRTtools' rbf_type).")
        form.addRow("Method", self.rbf_combo)

        self.mode_combo = compact_combo(QComboBox())
        add_combo_items(
            self.mode_combo,
            [
                ("Combined Re-Im Data", "complex"),
                ("Re Data", "real"),
                ("Im Data", "imaginary"),
            ],
        )
        self.mode_combo.setToolTip("Data Used: which part of the impedance is fitted.")
        form.addRow("Data used", self.mode_combo)

        self.inductance_check = QCheckBox("Fit with inductance")
        self.inductance_check.setToolTip(
            "Include an inductive element in the fit. To discard inductive "
            "points entirely, use the 'Remove inductive tail' filter on the "
            "Validation step instead."
        )
        # Spans both columns: a checkbox is a complete statement, not a
        # label/value pair.
        form.addRow(self.inductance_check)

        form.addRow(section_label("Regularisation"))

        self.derivative_combo = compact_combo(QComboBox())
        add_combo_items(self.derivative_combo, [("1st order", 1), ("2nd order", 2)])
        self.derivative_combo.setToolTip(
            "Regularization Derivative. pyimpspec's TR-RBF/BHT only implement "
            "1st- and 2nd-order Tikhonov regularization (0th order is not "
            "available)."
        )
        self.derivative_combo.setCurrentIndex(0)
        form.addRow("Derivative", self.derivative_combo)

        self.cv_combo = compact_combo(QComboBox())
        add_combo_items(
            self.cv_combo,
            [
                ("custom", ""),
                ("GCV", "gcv"),
                ("mGCV", "mgcv"),
                ("rGCV", "rgcv"),
                ("re-im", "re-im"),
                ("L-curve", "lc"),
            ],
        )
        self.cv_combo.setToolTip(
            "Parameter Selection Method: how the regularization parameter "
            "is chosen. 'custom' uses the value below directly."
        )
        form.addRow("Selection method", self.cv_combo)

        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setMinimum(1e-10)
        self.lambda_spin.setMaximum(10.0)
        self.lambda_spin.setSingleStep(0.001)
        self.lambda_spin.setValue(0.001)
        self.lambda_spin.setToolTip(
            "Regularization parameter (λ). Used directly when the selection "
            "method is 'custom'; otherwise the initial value for the chosen "
            "cross-validation method."
        )
        form.addRow("Lambda (λ)", self.lambda_spin)

        form.addRow(section_label("RBF shape"))

        # Paired on one row: the combo decides whether the spinbox is an FWHM
        # coefficient or a shape factor.
        shape_row = QWidget()
        shape_layout = QHBoxLayout(shape_row)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.setSpacing(style.GROUP_SPACING)
        self.shape_control_combo = compact_combo(QComboBox())
        add_combo_items(
            self.shape_control_combo,
            [("FWHM Coefficient", "fwhm"), ("Shape Factor", "factor")],
        )
        shape_layout.addWidget(self.shape_control_combo, stretch=1)
        self.shape_coeff_spin = QDoubleSpinBox()
        self.shape_coeff_spin.setDecimals(4)
        self.shape_coeff_spin.setMinimum(0.0001)
        self.shape_coeff_spin.setMaximum(10.0)
        self.shape_coeff_spin.setSingleStep(0.05)
        self.shape_coeff_spin.setValue(0.5)
        shape_layout.addWidget(self.shape_coeff_spin)
        form.addRow("Shape control", shape_row)

        # A named section carries the scope context, so the two controls below
        # can keep short labels.
        form.addRow(section_label("Bayesian & Hilbert only"))

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setMinimum(1000)
        self.num_samples_spin.setMaximum(100000)
        self.num_samples_spin.setSingleStep(500)
        self.num_samples_spin.setValue(1000)
        self.num_samples_spin.setToolTip(
            "Only used by Bayesian Run and Hilbert Transform. Must be >= "
            "1000; larger values are more accurate but slower."
        )
        form.addRow("Samples", self.num_samples_spin)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setMinimum(0)
        self.timeout_spin.setMaximum(36000)
        self.timeout_spin.setSingleStep(60)
        self.timeout_spin.setValue(300)
        self.timeout_spin.setToolTip(
            "Bayesian Run's credible-interval sampler can be extremely slow "
            "(tens of minutes for even modest sweeps). It aborts once this "
            "many seconds pass; 0 disables the limit entirely."
        )
        form.addRow("Timeout (s)", self.timeout_spin)
        self.add_settings(settings_box)

        run_box, run_layout = group_box("Run DRT")

        self.run_simple_button = QPushButton("Simple Run")
        self.run_simple_button.setToolTip(
            "Fast, deterministic TR-RBF point estimate (no credible intervals)."
        )
        # The one action this step exists to perform.
        self.run_simple_button.setProperty("variant", "primary")
        run_layout.addWidget(self.run_simple_button)

        self.run_bayesian_button = QPushButton("Bayesian Run")
        self.run_bayesian_button.setToolTip(
            "TR-RBF with Bayesian credible intervals via HMC sampling. Can "
            "be very slow — runs in the background so the UI stays "
            "responsive; see the timeout setting above."
        )
        run_layout.addWidget(self.run_bayesian_button)

        self.run_bht_button = QPushButton("Hilbert Transform")
        self.run_bht_button.setToolTip(
            "Bayesian Hilbert Transform (BHT): estimates the DRT separately "
            "from the real and imaginary parts, and scores how well they "
            "agree (a Kramers-Kronig-style consistency check)."
        )
        run_layout.addWidget(self.run_bht_button)

        # An output, so it sits with the action that produces it rather than
        # among the inputs above.
        lambda_row = QHBoxLayout()
        lambda_row.setContentsMargins(0, 0, 0, 0)
        lambda_row.setSpacing(style.FORM_H_SPACING)
        lambda_row.addWidget(QLabel("Optimal λ used"))
        self.optimal_lambda_label = QLabel("—")
        self.optimal_lambda_label.setToolTip(
            "The regularization parameter actually used by the most recent "
            "run, shown when exactly one sweep is selected."
        )
        self.optimal_lambda_label.setProperty("state", "muted")
        lambda_row.addWidget(self.optimal_lambda_label, stretch=1)
        run_layout.addLayout(lambda_row)
        self.add_settings(run_box)

        peak_box, peak_layout = group_box(
            "Peak deconvolution",
            "Fits individual (skew) normal peaks to the most recently "
            "computed DRT result for each selected sweep.",
        )
        peak_form = QFormLayout()
        peak_form.setContentsMargins(0, 0, 0, 0)
        peak_form.setHorizontalSpacing(style.FORM_H_SPACING)
        peak_form.setVerticalSpacing(style.FORM_V_SPACING)
        self.num_peaks_spin = QSpinBox()
        self.num_peaks_spin.setMinimum(0)
        self.num_peaks_spin.setMaximum(50)
        self.num_peaks_spin.setValue(0)
        self.num_peaks_spin.setToolTip(
            "Number of peaks to fit. 0 analyses every detected peak."
        )
        peak_form.addRow("Peaks", self.num_peaks_spin)
        peak_layout.addLayout(peak_form)

        self.run_peak_analysis_button = QPushButton("Peak deconvolution")
        peak_layout.addWidget(self.run_peak_analysis_button)

        self.peaks_text = QPlainTextEdit()
        self.peaks_text.setReadOnly(True)
        self.peaks_text.setToolTip(
            "Time constant, height and resistance of each fitted peak. Plot "
            "them with the 'Peak deconvolution' view above."
        )
        # Fixed-pitch: the table is pre-aligned text a proportional face
        # throws out of column. Capped so it cannot push the panel off-screen;
        # it keeps its own scrollbar.
        self.peaks_text.setFont(style.mono_font())
        self.peaks_text.setMaximumHeight(style.TEXT_PANE_MAX_HEIGHT)
        peak_layout.addWidget(self.peaks_text)
        self.add_settings(peak_box)

        export_box, export_layout = group_box("Export")
        self.export_results_button = QPushButton("Export DRT results…")
        self.export_results_button.setToolTip(
            "Write the DRT curve, and the fitted peaks if any, for the sweep "
            "on screen. Use File ▸ Save session to keep everything instead."
        )
        self.export_results_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_results_button)
        self.export_image_button = QPushButton("Save DRT plot as image…")
        self.export_image_button.setProperty("variant", "quiet")
        export_layout.addWidget(self.export_image_button)
        self.add_settings(export_box)

        self.end_settings()

        # ----------------------------------------------------------- content

        header = QHBoxLayout()
        header.setContentsMargins(*style.CONTENT_MARGINS)
        header.setSpacing(style.GROUP_SPACING)
        header.addWidget(QLabel("Top plot"))
        top_view = SegmentedControl(["Measured", "Peak deconvolution"])
        # Named *_radio though they are QToolButtons: same QAbstractButton
        # API, and MainWindow._wire_steps reaches them by these names.
        self.measured_radio = top_view.button(0)
        self.measured_radio.setToolTip(
            "The spectrum the DRT below was computed from, in whichever view "
            "the Data Visualisation step is set to."
        )
        self.peaks_radio = top_view.button(1)
        self.peaks_radio.setToolTip(
            "The individual peaks fitted to the DRT (dashed) and their sum "
            "(solid), on the same frequency axis as the DRT below."
        )
        header.addWidget(top_view)
        header.addStretch()
        self.add_content_layout(header)

        splitter = QSplitter(Qt.Vertical)
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
        # Pixels, not ratios -- setSizes takes real heights. An even split, so
        # the DRT curve and whatever is read against it get the same room.
        splitter.setSizes([470, 470])
        splitter.setChildrenCollapsible(False)
        self.add_content(splitter, stretch=1)

    @property
    def top_view(self) -> str:
        """"Measured" or "Peaks" -- what the upper pane is showing."""
        return "Measured" if self.measured_radio.isChecked() else "Peaks"
