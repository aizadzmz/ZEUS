"""Main window: sidebar controls on the left, plot tabs on the right.

Mirrors the Streamlit app's workflow (load -> select -> style -> filter ->
validate) but with explicit event handling instead of rerun-everything.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core import EISParseError, parse_eis_file
from core.drt import (
    CROSS_VALIDATION_METHODS,
    DATA_MODES,
    RBF_SHAPE_CONTROLS,
    RBF_TYPES,
    analyze_drt_peaks,
    run_drt,
    run_drt_bht,
)
from core.filtering import apply_manual_overrides, clear_mask, mask_inductive_points
from core.generic_parser import parse_generic_file, sniff_columns
from core.mb_parser import parse_modulobat_file
from core.plotting import plot_drt, plot_residuals
from core.plotting_pg import build_nyquist_plot
from core.validation import mask_residual_outliers, run_kk_test, run_zhit
from gui.figure_panes import FigureListPane, FigurePane, PgFigurePane
from gui.generic_import_dialog import GenericImportDialog
from gui.theme import THEMES, apply_theme
from gui.workers import DRTWorker, ValidationWorker

SIDEBAR_WIDTH = 320
VALIDATION_METHODS = ("Kramers-Kronig", "Z-HIT")

# How many residuals figures the Residuals tab offers to draw by default.
# Each one costs a full matplotlib rasterization (~0.3 s per canvas, which
# dwarfs the ~0.04 s to build the figure itself), so a 20-sweep selection
# meant ~6 s of drawing on every visit to the tab. Rendering is capped to
# this many and only happens when the user asks for it -- see
# _build_residuals_tab and MainWindow._residuals_armed.
DEFAULT_RESIDUALS_LIMIT = 5
ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


def _titleize(rbf_type: str) -> str:
    """'c2-matern' -> 'C2 Matern', 'piecewise-linear' -> 'Piecewise Linear'."""
    return " ".join(word.capitalize() for word in rbf_type.split("-"))


def _add_combo_items(combo: QComboBox, pairs) -> None:
    """Populate a QComboBox from (display_text, value) pairs, retrievable
    via combo.currentData()."""
    for display, value in pairs:
        combo.addItem(display, value)


def _parse_file(path: str):
    """Try the Modulo Bat cycling-sequence parser first (content-detected
    via its 'Nb header lines' signature, so this is safe to attempt
    regardless of extension), then the standard BioLogic/pyimpspec parser
    for genuine .mpt exports. Returns (datasets, parser_name). Raises
    EISParseError otherwise - including for every .txt/.csv, since
    pyimpspec's own column-guessing heuristics can silently misread an
    unfamiliar plaintext layout instead of raising, which would skip the
    user-confirmed column mapping entirely. The caller should fall back to
    the generic parser on this error (see MainWindow._open_generic_file)."""
    mb_error: Optional[Exception] = None
    try:
        return parse_modulobat_file(path), "Modulo Bat (cycling sequence)"
    except Exception as exc:
        mb_error = exc

    if Path(path).suffix.lower() == ".mpt":
        try:
            return parse_eis_file(path), "Standard EIS export"
        except Exception as std_exc:
            raise EISParseError(
                f"- Modulo Bat parser: {mb_error}\n"
                f"- Standard EIS parser: {std_exc}"
            ) from std_exc

    raise EISParseError(f"- Modulo Bat parser: {mb_error}") from mb_error


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EIS Batch Analysis — Parser & Plotter")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1200, 800)

        self._datasets: List = []
        self._parser_used = ""
        self._source_name = ""
        # {(method, dataset label): KramersKronigResult | ZHITResult}
        self._validation_results = {}
        # {dataset label: set of point indices} — the eraser's per-point
        # overrides. Held here rather than in the DataSet's own mask because
        # _refresh() rebuilds that mask from the automatic filters on every
        # call and would wipe them; they are re-applied as a layer instead.
        self._manual_masked: Dict[str, set] = {}
        self._manual_kept: Dict[str, set] = {}
        self._worker: Optional[ValidationWorker] = None
        self._worker_errors: List[Tuple[str, str]] = []
        # Method name of the run in flight, for the progress message — the
        # sidebar radio it came from is disabled mid-run, but reading the
        # captured value keeps the message correct regardless.
        self._running_method = ""
        # {dataset label: TRRBFResult | BHTResult} — last DRT run wins
        self._drt_results = {}
        # {dataset label: DRTPeaks}
        self._drt_peaks = {}
        self._drt_worker: Optional[DRTWorker] = None
        self._drt_worker_errors: List[Tuple[str, str]] = []
        # Rebuilding every tab's figures on every checkbox click is O(tabs *
        # selected sweeps) and dominates when overlaying many curves, so tab
        # content is rebuilt lazily: _refresh() does the cheap masking/
        # validity bookkeeping and marks every tab dirty, then only the
        # currently visible tab is actually (re)plotted. Switching tabs
        # renders whichever tab was still dirty.
        self._pending: Optional[dict] = None
        self._tab_dirty: set = set()
        # The Residuals tab goes a step further than the dirty-tab laziness
        # above: it stays blank until the user clicks "Plot residuals",
        # because rendering it is by far the most expensive thing the window
        # does (see DEFAULT_RESIDUALS_LIMIT).
        #
        # Once armed it stays armed until a different file is opened, so the
        # plot survives switching tabs and is rebuilt from current state on
        # the way back. It used to disarm on every _refresh(), which meant any
        # sidebar click -- a threshold nudge, a checkbox -- silently emptied
        # the tab the next time the user looked at it.
        self._residuals_armed = False

        self._settings = QSettings()
        saved = self._settings.value("theme", "light")
        self._theme_mode = saved if saved in THEMES else "light"

        self._build_menu()
        self._build_ui()

        # Apply the theme once the widgets exist, then reflect the current
        # mode in the menu without re-triggering the toggle handler.
        apply_theme(self._theme_mode)
        self.dark_action.blockSignals(True)
        self.dark_action.setChecked(self._theme_mode == "dark")
        self.dark_action.blockSignals(False)

    # ------------------------------------------------------------------ UI

    def _build_menu(self) -> None:
        # One QAction backs the menu item, the status-bar button, and the
        # shortcut, so their checked states stay in sync automatically.
        self.dark_action = QAction("🌙", self, checkable=True)
        self.dark_action.setShortcut("Ctrl+D")
        self.dark_action.setStatusTip("Toggle between light and dark themes (Ctrl+D)")
        self.dark_action.toggled.connect(self._on_theme_toggled)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.dark_action)

        # A flat button in the status bar shares the same action, so it stays
        # in sync with the menu item and Ctrl+D. addPermanentWidget docks it
        # in the bottom-right corner; swap to addWidget for the bottom-left.
        theme_button = QToolButton()
        theme_button.setDefaultAction(self.dark_action)
        theme_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        theme_button.setAutoRaise(True)
        self.statusBar().addPermanentWidget(theme_button)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_main_area(), stretch=1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Open a .mpt, .txt, or .csv file to begin.")

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. Load file
        load_box = QGroupBox("1. Load file")
        load_layout = QVBoxLayout(load_box)
        self.open_button = QPushButton("Open EIS export…")
        self.open_button.clicked.connect(self._open_file)
        self.file_label = QLabel("No file loaded.")
        self.file_label.setWordWrap(True)
        load_layout.addWidget(self.open_button)
        load_layout.addWidget(self.file_label)
        layout.addWidget(load_box)

        # 2. Plot selection
        select_box = QGroupBox("2. Plot selection")
        select_layout = QVBoxLayout(select_box)
        self.single_radio = QRadioButton("Single")
        self.overlay_radio = QRadioButton("Overlay")
        self.single_radio.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.single_radio)
        mode_group.addButton(self.overlay_radio)
        self.single_radio.toggled.connect(self._on_mode_changed)
        select_layout.addWidget(self.single_radio)
        select_layout.addWidget(self.overlay_radio)

        self.sweep_combo = QComboBox()
        self.sweep_combo.currentIndexChanged.connect(self._refresh)
        select_layout.addWidget(self.sweep_combo)

        self.sweep_list = QListWidget()
        self.sweep_list.itemChanged.connect(self._refresh)
        self.sweep_list.setVisible(False)
        self.sweep_list.setMaximumHeight(160)
        select_layout.addWidget(self.sweep_list)
        layout.addWidget(select_box)

        # 3. Style
        style_box = QGroupBox("3. Style")
        style_layout = QVBoxLayout(style_box)
        self.markers_radio = QRadioButton("Markers")
        self.line_radio = QRadioButton("Line")
        self.markers_radio.setChecked(True)
        style_group = QButtonGroup(self)
        style_group.addButton(self.markers_radio)
        style_group.addButton(self.line_radio)
        self.markers_radio.toggled.connect(self._refresh)
        style_layout.addWidget(self.markers_radio)
        style_layout.addWidget(self.line_radio)
        layout.addWidget(style_box)

        # 4. Filtering
        filter_box = QGroupBox("4. Filtering")
        filter_layout = QVBoxLayout(filter_box)
        self.inductive_check = QCheckBox("Remove inductive tail (Im(Z) > 0)")
        self.inductive_check.toggled.connect(self._refresh)
        filter_layout.addWidget(self.inductive_check)

        self.eraser_check = QCheckBox("Eraser (click points to mask/unmask)")
        self.eraser_check.setToolTip(
            "On the Nyquist plot, click a point to remove it, or a removed "
            "(grey ×) point to restore it. Manual edits override the filter "
            "above and the validation outlier threshold, and are cleared when "
            "a different file is opened.\n\n"
            "Hiding the 'Removed' series via its legend entry also stops "
            "those points responding to clicks."
        )
        self.eraser_check.toggled.connect(self._on_eraser_toggled)
        filter_layout.addWidget(self.eraser_check)
        layout.addWidget(filter_box)

        # 5. Validation
        valid_box = QGroupBox("5. Validation")
        valid_layout = QVBoxLayout(valid_box)
        self.kk_radio = QRadioButton("Kramers-Kronig")
        self.zhit_radio = QRadioButton("Z-HIT")
        self.kk_radio.setChecked(True)
        method_group = QButtonGroup(self)
        method_group.addButton(self.kk_radio)
        method_group.addButton(self.zhit_radio)
        valid_box.setToolTip(
            "Kramers-Kronig checks linearity/causality via a lin-KK fit, on "
            "the impedance representation only — fitting the admittance "
            "representation as well costs roughly 3x the time and mainly "
            "helps spectra with negative differential resistance. "
            "Z-HIT reconstructs the modulus from the phase data and is good "
            "at catching non-steady-state artifacts such as low-frequency "
            "drift; it is also far quicker, since it does no model fitting."
        )
        self.kk_radio.toggled.connect(self._on_method_changed)
        valid_layout.addWidget(self.kk_radio)
        valid_layout.addWidget(self.zhit_radio)

        valid_layout.addWidget(QLabel("Outlier threshold (%)"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(2.0)
        self.threshold_spin.setToolTip(
            "Points whose relative residual (real or imaginary) exceeds "
            "this percentage are removed."
        )
        self.threshold_spin.valueChanged.connect(self._refresh)
        valid_layout.addWidget(self.threshold_spin)

        self.run_validation_button = QPushButton()
        self.run_validation_button.clicked.connect(self._run_validation)
        valid_layout.addWidget(self.run_validation_button)
        layout.addWidget(valid_box)

        # 6. DRT settings
        drt_settings_box = QGroupBox("6. DRT settings")
        drt_settings_layout = QVBoxLayout(drt_settings_box)
        drt_settings_box.setToolTip(
            "Settings for Tikhonov regularization + radial basis function "
            "discretization (TR-RBF) and the Bayesian Hilbert Transform "
            "(BHT), applied to each selected sweep's currently unmasked "
            "points."
        )

        drt_settings_layout.addWidget(QLabel("Method of Discretization"))
        self.drt_rbf_combo = QComboBox()
        _add_combo_items(self.drt_rbf_combo, [(_titleize(v), v) for v in RBF_TYPES])
        drt_settings_layout.addWidget(self.drt_rbf_combo)

        drt_settings_layout.addWidget(QLabel("Data Used"))
        self.drt_mode_combo = QComboBox()
        _add_combo_items(
            self.drt_mode_combo,
            [
                ("Combined Re-Im Data", "complex"),
                ("Re Data", "real"),
                ("Im Data", "imaginary"),
            ],
        )
        drt_settings_layout.addWidget(self.drt_mode_combo)

        self.drt_inductance_check = QCheckBox("Fit with inductance")
        self.drt_inductance_check.setToolTip(
            "Include an inductive element in the fit. To discard inductive "
            "points entirely, use the 'Remove inductive tail' filter above "
            "instead."
        )
        drt_settings_layout.addWidget(self.drt_inductance_check)

        drt_settings_layout.addWidget(QLabel("Regularization Derivative"))
        self.drt_derivative_combo = QComboBox()
        _add_combo_items(
            self.drt_derivative_combo,
            [("1st order", 1), ("2nd order", 2)],
        )
        self.drt_derivative_combo.setToolTip(
            "pyimpspec's TR-RBF/BHT only implement 1st- and 2nd-order "
            "Tikhonov regularization (0th order is not available)."
        )
        self.drt_derivative_combo.setCurrentIndex(0)
        drt_settings_layout.addWidget(self.drt_derivative_combo)

        drt_settings_layout.addWidget(QLabel("Parameter Selection Method"))
        self.drt_cv_combo = QComboBox()
        _add_combo_items(
            self.drt_cv_combo,
            [
                ("custom", ""),
                ("GCV", "gcv"),
                ("mGCV", "mgcv"),
                ("rGCV", "rgcv"),
                ("re-im", "re-im"),
                ("L-curve", "lc"),
            ],
        )
        drt_settings_layout.addWidget(self.drt_cv_combo)

        drt_settings_layout.addWidget(QLabel("Regularization parameter"))
        self.drt_lambda_spin = QDoubleSpinBox()
        self.drt_lambda_spin.setDecimals(6)
        self.drt_lambda_spin.setMinimum(1e-10)
        self.drt_lambda_spin.setMaximum(10.0)
        self.drt_lambda_spin.setSingleStep(0.001)
        self.drt_lambda_spin.setValue(0.001)
        self.drt_lambda_spin.setToolTip(
            "Used directly when Parameter Selection Method is 'custom'; "
            "otherwise used as the initial value for the chosen "
            "cross-validation method."
        )
        drt_settings_layout.addWidget(self.drt_lambda_spin)

        drt_settings_layout.addWidget(QLabel("Optimal Regularization parameter"))
        self.drt_optimal_lambda_label = QLabel("—")
        self.drt_optimal_lambda_label.setToolTip(
            "The regularization parameter actually used by the most recent "
            "run, shown when exactly one sweep is selected."
        )
        drt_settings_layout.addWidget(self.drt_optimal_lambda_label)

        drt_settings_layout.addWidget(QLabel("RBF Shape Control"))
        self.drt_shape_control_combo = QComboBox()
        _add_combo_items(
            self.drt_shape_control_combo,
            [("FWHM Coefficient", "fwhm"), ("Shape Factor", "factor")],
        )
        drt_settings_layout.addWidget(self.drt_shape_control_combo)

        drt_settings_layout.addWidget(QLabel("FWHM / Shape Factor Control"))
        self.drt_shape_coeff_spin = QDoubleSpinBox()
        self.drt_shape_coeff_spin.setDecimals(4)
        self.drt_shape_coeff_spin.setMinimum(0.0001)
        self.drt_shape_coeff_spin.setMaximum(10.0)
        self.drt_shape_coeff_spin.setSingleStep(0.05)
        self.drt_shape_coeff_spin.setValue(0.5)
        drt_settings_layout.addWidget(self.drt_shape_coeff_spin)

        drt_settings_layout.addWidget(QLabel("Number of Samples"))
        self.drt_num_samples_spin = QSpinBox()
        self.drt_num_samples_spin.setMinimum(1000)
        self.drt_num_samples_spin.setMaximum(100000)
        self.drt_num_samples_spin.setSingleStep(500)
        self.drt_num_samples_spin.setValue(1000)
        self.drt_num_samples_spin.setToolTip(
            "Only used by Bayesian Run and Hilbert Transform. Must be >= "
            "1000; larger values are more accurate but slower."
        )
        drt_settings_layout.addWidget(self.drt_num_samples_spin)

        drt_settings_layout.addWidget(QLabel("Bayesian Run timeout (s)"))
        self.drt_timeout_spin = QSpinBox()
        self.drt_timeout_spin.setMinimum(0)
        self.drt_timeout_spin.setMaximum(36000)
        self.drt_timeout_spin.setSingleStep(60)
        self.drt_timeout_spin.setValue(300)
        self.drt_timeout_spin.setToolTip(
            "Bayesian Run's credible-interval sampler can be extremely slow "
            "(tens of minutes for even modest sweeps). It aborts once this "
            "many seconds pass; 0 disables the limit entirely."
        )
        drt_settings_layout.addWidget(self.drt_timeout_spin)

        layout.addWidget(drt_settings_box)

        # 7. Run DRT
        run_drt_box = QGroupBox("7. Run DRT")
        run_drt_layout = QVBoxLayout(run_drt_box)

        self.run_drt_simple_button = QPushButton("Simple Run")
        self.run_drt_simple_button.setToolTip(
            "Fast, deterministic TR-RBF point estimate (no credible intervals)."
        )
        self.run_drt_simple_button.clicked.connect(self._run_drt_simple)
        run_drt_layout.addWidget(self.run_drt_simple_button)

        self.run_drt_bayesian_button = QPushButton("Bayesian Run")
        self.run_drt_bayesian_button.setToolTip(
            "TR-RBF with Bayesian credible intervals via HMC sampling. Can "
            "be very slow — runs in the background so the UI stays "
            "responsive; see the timeout setting above."
        )
        self.run_drt_bayesian_button.clicked.connect(self._run_drt_bayesian)
        run_drt_layout.addWidget(self.run_drt_bayesian_button)

        self.run_drt_bht_button = QPushButton("Hilbert Transform")
        self.run_drt_bht_button.setToolTip(
            "Bayesian Hilbert Transform (BHT): estimates the DRT separately "
            "from the real and imaginary parts, and scores how well they "
            "agree (a Kramers-Kronig-style consistency check)."
        )
        self.run_drt_bht_button.clicked.connect(self._run_drt_bht)
        run_drt_layout.addWidget(self.run_drt_bht_button)

        layout.addWidget(run_drt_box)

        # 8. DRT peak analysis
        peak_box = QGroupBox("8. DRT peak analysis")
        peak_layout = QVBoxLayout(peak_box)
        peak_box.setToolTip(
            "Fits individual (skew) normal peaks to the most recently "
            "computed DRT result for each selected sweep."
        )

        peak_layout.addWidget(QLabel("Number of peaks (0 = all detected)"))
        self.drt_num_peaks_spin = QSpinBox()
        self.drt_num_peaks_spin.setMinimum(0)
        self.drt_num_peaks_spin.setMaximum(50)
        self.drt_num_peaks_spin.setValue(0)
        peak_layout.addWidget(self.drt_num_peaks_spin)

        self.run_peak_analysis_button = QPushButton("Peak deconvolution")
        self.run_peak_analysis_button.clicked.connect(self._run_peak_analysis)
        peak_layout.addWidget(self.run_peak_analysis_button)

        layout.addWidget(peak_box)

        layout.addStretch()
        self._update_validation_button_text()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(SIDEBAR_WIDTH)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._sidebar = scroll
        return scroll

    def _build_main_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "color: #7a5900; background: #fff3cd; border: 1px solid #ffe08a;"
            "border-radius: 4px; padding: 6px;"
        )
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        self.stack = QStackedWidget()

        placeholder = QLabel("Open a .mpt, .txt, or .csv file to begin.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(placeholder)

        self.tabs = QTabWidget()
        self.nyquist_pane = PgFigurePane(with_overlay_actions=True)
        self.nyquist_pane.replot_requested.connect(self._force_replot_nyquist)
        self.nyquist_pane.point_mask_toggled.connect(self._on_point_mask_toggled)
        self.residuals_pane = FigureListPane()
        self.drt_pane = FigurePane(with_toolbar=True)
        self.drt_peaks_text = QPlainTextEdit()
        self.drt_peaks_text.setReadOnly(True)
        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.tabs.addTab(self.nyquist_pane, "Nyquist")
        self.tabs.addTab(self._build_residuals_tab(), "Residuals")
        self.tabs.addTab(self.drt_pane, "DRT")
        self.tabs.addTab(self.drt_peaks_text, "DRT Peaks")
        self.tabs.addTab(self.details_text, "Sweep details")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.stack.addWidget(self.tabs)

        layout.addWidget(self.stack, stretch=1)
        return container

    def _build_residuals_tab(self) -> QWidget:
        """The residuals list, above it a header that decides how much of the
        selection actually gets drawn: a count cap and an explicit trigger.
        Every other tab plots itself as soon as you look at it; this one
        doesn't, because it's the only one whose cost scales with the number
        of selected sweeps (see DEFAULT_RESIDUALS_LIMIT)."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(6, 4, 6, 4)
        header.setSpacing(6)
        header.addWidget(QLabel("Plot first"))

        self.residuals_limit_spin = QSpinBox()
        self.residuals_limit_spin.setRange(1, 99)
        self.residuals_limit_spin.setValue(DEFAULT_RESIDUALS_LIMIT)
        self.residuals_limit_spin.setToolTip(
            "How many of the validated sweeps to draw, in selection order."
        )
        header.addWidget(self.residuals_limit_spin)

        self.residuals_plot_button = QPushButton("Plot residuals")
        self.residuals_plot_button.setToolTip(
            "Draw the residuals figures for the current selection."
        )
        self.residuals_plot_button.clicked.connect(self._on_plot_residuals_clicked)
        header.addWidget(self.residuals_plot_button)

        self.residuals_status_label = QLabel()
        header.addWidget(self.residuals_status_label)
        header.addStretch()

        col.addLayout(header)
        col.addWidget(self.residuals_pane, stretch=1)
        self._update_residuals_header(0, 0)
        return container

    # ------------------------------------------------------------- helpers

    @property
    def _mode(self) -> str:
        return "Single" if self.single_radio.isChecked() else "Overlay"

    @property
    def _style(self) -> str:
        return "scatter" if self.markers_radio.isChecked() else "line"

    @property
    def _validation_method(self) -> str:
        return VALIDATION_METHODS[0] if self.kk_radio.isChecked() else VALIDATION_METHODS[1]

    def _selected_datasets(self) -> List:
        if not self._datasets:
            return []
        if self._mode == "Single":
            idx = self.sweep_combo.currentIndex()
            return [self._datasets[idx]] if 0 <= idx < len(self._datasets) else []
        return [
            ds
            for i, ds in enumerate(self._datasets)
            if self.sweep_list.item(i).checkState() == Qt.Checked
        ]

    def _apply_manual_overrides(self, ds) -> None:
        """Re-assert this sweep's eraser edits over whatever the automatic
        filters have just decided. A no-op for sweeps that have none."""
        masked = self._manual_masked.get(ds.label)
        kept = self._manual_kept.get(ds.label)
        if masked or kept:
            apply_manual_overrides(ds, masked or (), kept or ())

    def _update_validation_button_text(self) -> None:
        # Keep it short — the selected method is shown by the radios above.
        if self._worker is not None:
            self.run_validation_button.setText("Running…")
        else:
            self.run_validation_button.setText("Run validation")

    # ------------------------------------------------------------ handlers

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open EIS export", "", "EIS exports (*.mpt *.txt *.csv)"
        )
        if not path:
            return

        try:
            datasets, parser_used = _parse_file(path)
        except EISParseError as exc:
            result = self._open_generic_file(path, exc)
            if result is None:
                return
            datasets, parser_used = result

        self._datasets = datasets
        self._parser_used = parser_used
        self._source_name = Path(path).name
        self._validation_results = {}
        self._drt_results = {}
        self._drt_peaks = {}
        # Labels ("Set 01", ...) are only unique within a file, so the
        # previous file's eraser edits would otherwise land on this one's
        # sweeps.
        self._manual_masked = {}
        self._manual_kept = {}
        # A new file has nothing validated yet, so start the Residuals tab
        # blank again rather than inheriting the previous file's arming.
        self._residuals_armed = False

        # Elide instead of wrapping: filenames are one unbreakable token, and
        # a word-wrapped QLabel's minimum width would force the sidebar to
        # scroll sideways. Full name stays available as a tooltip.
        metrics = self.file_label.fontMetrics()
        self.file_label.setText(
            metrics.elidedText(self._source_name, Qt.ElideMiddle, SIDEBAR_WIDTH - 60)
        )
        self.file_label.setToolTip(self._source_name)
        self.statusBar().showMessage(
            f"Parsed {len(datasets)} EIS sweep(s) from '{self._source_name}' "
            f"using the {parser_used} parser."
        )

        # Repopulate the sweep selectors without triggering refreshes.
        self.sweep_combo.blockSignals(True)
        self.sweep_combo.clear()
        self.sweep_combo.addItems([ds.label for ds in datasets])
        self.sweep_combo.setCurrentIndex(0)
        self.sweep_combo.blockSignals(False)

        self.sweep_list.blockSignals(True)
        self.sweep_list.clear()
        for i, ds in enumerate(datasets):
            item = QListWidgetItem(ds.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < 5 else Qt.Unchecked)
            self.sweep_list.addItem(item)
        self.sweep_list.blockSignals(False)

        self.stack.setCurrentWidget(self.tabs)
        self._refresh()

    def _open_generic_file(self, path: str, prior_error: EISParseError):
        """Fall back to the generic txt/csv parser: sniff headers/roles,
        let the user confirm/correct the mapping in a dialog, then parse.
        Returns (datasets, parser_name), or None if the user cancelled or
        the file couldn't be parsed at all."""
        try:
            headers, sample_rows, guessed_roles = sniff_columns(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Parse error",
                f"Could not parse '{Path(path).name}'.\n"
                f"{prior_error}\n"
                f"- Generic parser: {exc}",
            )
            return None

        dialog = GenericImportDialog(
            Path(path).name, headers, sample_rows, guessed_roles, parent=self
        )
        if dialog.exec() != QDialog.Accepted:
            return None

        try:
            datasets = parse_generic_file(path, column_roles=dialog.column_roles())
        except EISParseError as exc:
            QMessageBox.critical(self, "Parse error", str(exc))
            return None

        return datasets, "Generic txt/csv export (user-confirmed)"

    def _on_mode_changed(self) -> None:
        single = self._mode == "Single"
        self.sweep_combo.setVisible(single)
        self.sweep_list.setVisible(not single)
        self._refresh()

    def _on_method_changed(self, checked: bool) -> None:
        if not checked:
            return  # fires once per radio; only act on the newly-checked one
        self._update_validation_button_text()
        self._refresh()

    def _on_eraser_toggled(self, checked: bool) -> None:
        """Purely a mode switch — no mask changes, so no _refresh()."""
        self.nyquist_pane.set_eraser_enabled(checked)
        if checked:
            self.statusBar().showMessage(
                "Eraser on — click a point on the Nyquist plot to remove it, "
                "or a grey × to restore it."
            )
        else:
            self.statusBar().showMessage("Eraser off.")

    def _on_point_mask_toggled(self, label: str, index: int) -> None:
        """Flip one point's manual override, then let _refresh() recompose the
        mask and redraw.

        Which way it flips is read from the sweep's *current* mask rather than
        from the override sets, so a point the filters removed toggles to
        'keep' on the first click regardless of how it got removed."""
        ds = next((d for d in self._datasets if d.label == label), None)
        if ds is None:
            return

        masked = self._manual_masked.setdefault(label, set())
        kept = self._manual_kept.setdefault(label, set())
        if ds.data.get_mask().get(index, False):
            masked.discard(index)
            kept.add(index)
            action = "restored"
        else:
            kept.discard(index)
            masked.add(index)
            action = "removed"

        self._refresh()
        self.statusBar().showMessage(f"{label}: point {index + 1} {action}.")

    def _on_theme_toggled(self, checked: bool) -> None:
        self._theme_mode = "dark" if checked else "light"
        apply_theme(self._theme_mode)
        self._settings.setValue("theme", self._theme_mode)
        # Existing figures were drawn with the old rcParams; regenerate them
        # so plot colors follow the new theme.
        if self._datasets:
            self._refresh()

    def _run_validation(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return
        method = self._validation_method
        runner = run_kk_test if method == VALIDATION_METHODS[0] else run_zhit

        # Masks must be stable while the worker reads them, so lock the
        # sidebar for the duration of the run.
        self._worker_errors = []
        self._worker = ValidationWorker(method, runner, selected, parent=self)
        self._worker.result_ready.connect(self._on_validation_result)
        self._worker.error.connect(self._on_validation_error)
        self._worker.progress.connect(self._on_validation_progress)
        self._worker.finished.connect(self._on_validation_finished)
        self._running_method = method
        self._sidebar.setEnabled(False)
        self._update_validation_button_text()
        self.statusBar().showMessage(f"Running {method} analysis… 0 of {len(selected)}")
        self._worker.start()

    def _on_validation_result(self, method: str, label: str, result) -> None:
        self._validation_results[(method, label)] = result

    def _on_validation_progress(self, done: int, total: int) -> None:
        """Sweeps finish out of order (they run in a process pool), so this
        is a count, not a name — a batch can otherwise sit on one unchanging
        status message for minutes."""
        self.statusBar().showMessage(
            f"Running {self._running_method} analysis… {done} of {total}"
        )

    def _on_validation_error(self, label: str, message: str) -> None:
        self._worker_errors.append((label, message))

    def _on_validation_finished(self) -> None:
        self._worker = None
        self._sidebar.setEnabled(True)
        self._update_validation_button_text()
        self.statusBar().showMessage("Validation finished.")
        if self._worker_errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in self._worker_errors)
            QMessageBox.warning(
                self, "Validation errors", f"Some sweeps failed:\n{details}"
            )
        self._refresh()

    def _drt_settings(self) -> dict:
        """Shared TR-RBF/BHT settings read from the '6. DRT settings' panel."""
        return dict(
            rbf_type=self.drt_rbf_combo.currentData(),
            derivative_order=self.drt_derivative_combo.currentData(),
            rbf_shape=self.drt_shape_control_combo.currentData(),
            shape_coeff=self.drt_shape_coeff_spin.value(),
        )

    def _update_optimal_lambda_label(self, selected: List) -> None:
        lambda_value = None
        if len(selected) == 1:
            result = self._drt_results.get(selected[0].label)
            lambda_value = getattr(result, "lambda_value", None)
        self.drt_optimal_lambda_label.setText(
            f"{lambda_value:.4g}" if lambda_value is not None else "—"
        )

    def _run_drt_simple(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        settings = self._drt_settings()
        errors = []
        for ds in selected:
            try:
                self._drt_results[ds.label] = run_drt(
                    ds,
                    mode=self.drt_mode_combo.currentData(),
                    inductance=self.drt_inductance_check.isChecked(),
                    cross_validation=self.drt_cv_combo.currentData(),
                    lambda_value=self.drt_lambda_spin.value(),
                    credible_intervals=False,
                    **settings,
                )
            except Exception as exc:
                errors.append((ds.label, str(exc)))

        if errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in errors)
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")

        self._update_optimal_lambda_label(selected)
        self.statusBar().showMessage(f"Simple Run DRT computed for {len(selected) - len(errors)} sweep(s).")
        self._refresh()

    def _run_drt_bayesian(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        confirm = QMessageBox.question(
            self,
            "Bayesian DRT",
            "Computing Bayesian credible intervals can take a very long "
            "time (tens of minutes per sweep is common). It will run in "
            "the background, but may still be slow to finish. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        settings = self._drt_settings()
        mode = self.drt_mode_combo.currentData()
        inductance = self.drt_inductance_check.isChecked()
        cross_validation = self.drt_cv_combo.currentData()
        lambda_value = self.drt_lambda_spin.value()
        num_samples = self.drt_num_samples_spin.value()
        timeout = self.drt_timeout_spin.value()

        def runner(ds):
            return run_drt(
                ds,
                mode=mode,
                inductance=inductance,
                cross_validation=cross_validation,
                lambda_value=lambda_value,
                credible_intervals=True,
                num_samples=num_samples,
                timeout=timeout,
                **settings,
            )

        self._drt_worker_errors = []
        self._drt_worker = DRTWorker(runner, selected, parent=self)
        self._drt_worker.result_ready.connect(self._on_drt_worker_result)
        self._drt_worker.error.connect(self._on_drt_worker_error)
        self._drt_worker.finished.connect(self._on_drt_worker_finished)
        self._sidebar.setEnabled(False)
        self.statusBar().showMessage("Running Bayesian DRT… this may take a while.")
        self._drt_worker.start()

    def _on_drt_worker_result(self, label: str, result) -> None:
        self._drt_results[label] = result

    def _on_drt_worker_error(self, label: str, message: str) -> None:
        self._drt_worker_errors.append((label, message))

    def _on_drt_worker_finished(self) -> None:
        self._drt_worker = None
        self._sidebar.setEnabled(True)
        self.statusBar().showMessage("Bayesian DRT finished.")
        if self._drt_worker_errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in self._drt_worker_errors)
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")
        self._update_optimal_lambda_label(self._selected_datasets())
        self._refresh()

    def _run_drt_bht(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        settings = self._drt_settings()
        errors = []
        for ds in selected:
            try:
                self._drt_results[ds.label] = run_drt_bht(
                    ds,
                    num_samples=self.drt_num_samples_spin.value(),
                    **settings,
                )
            except Exception as exc:
                errors.append((ds.label, str(exc)))

        if errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in errors)
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")

        self._update_optimal_lambda_label(selected)
        self.statusBar().showMessage(
            f"Hilbert Transform DRT computed for {len(selected) - len(errors)} sweep(s)."
        )
        self._refresh()

    def _run_peak_analysis(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        num_peaks = self.drt_num_peaks_spin.value()
        errors = []
        for ds in selected:
            result = self._drt_results.get(ds.label)
            if result is None:
                errors.append((ds.label, "Run a DRT calculation first."))
                continue
            try:
                self._drt_peaks[ds.label] = analyze_drt_peaks(result, num_peaks=num_peaks)
            except Exception as exc:
                errors.append((ds.label, str(exc)))

        if errors:
            details = "\n".join(f"- {label}: {msg}" for label, msg in errors)
            QMessageBox.warning(self, "Peak analysis errors", f"Some sweeps failed:\n{details}")

        self.statusBar().showMessage(
            f"Peak analysis computed for {len(selected) - len(errors)} sweep(s)."
        )
        self._refresh()

    # ------------------------------------------------------------- refresh

    def _refresh(self) -> None:
        """Cheap, always-run bookkeeping: apply masks, validate against the
        current selection, and update tab labels/counts. The actual figures
        are (re)built lazily by _render_active_tab(), since replotting every
        tab (in particular one residuals figure per selected sweep) on every
        single checkbox click is wasted work for tabs the user isn't even
        looking at — this is what made overlaying many curves feel slow.
        """
        if not self._datasets:
            return

        selected = self._selected_datasets()
        if not selected:
            self.warning_label.setText("Select at least one sweep to plot.")
            self.warning_label.show()
            self.nyquist_pane.clear()
            self.residuals_pane.clear()
            self.drt_pane.clear()
            self.drt_peaks_text.clear()
            self.details_text.clear()
            self._pending = None
            self._tab_dirty.clear()
            # Stays armed: unchecking the last sweep and rechecking one is a
            # transient state on the way to a new selection, not a request to
            # go back to a blank tab.
            self._update_residuals_header(0, 0)
            return

        method = self._validation_method
        threshold = self.threshold_spin.value()

        for ds in selected:
            if self.inductive_check.isChecked():
                mask_inductive_points(ds)
            else:
                clear_mask(ds)
            # Before the outlier pass, not just after: this reproduces the
            # mask a validation run actually observed, so erasing a point and
            # then running validation doesn't trip the length check below and
            # report a spurious "stale" result.
            self._apply_manual_overrides(ds)

        stale_labels = []
        for ds in selected:
            result = self._validation_results.get((method, ds.label))
            if result is not None:
                try:
                    mask_residual_outliers(ds, result, threshold)
                except ValueError:
                    stale_labels.append(ds.label)
            # Again, on top of the outlier pass -- which only ever adds masks,
            # so this is what lets a manually restored point survive a
            # threshold that would otherwise drop it.
            self._apply_manual_overrides(ds)

        if stale_labels:
            self.warning_label.setText(
                f"{method} results for {', '.join(stale_labels)} no longer match "
                f"the current mask (e.g. the inductive-tail filter or the "
                f"eraser changed it) — "
                f"click 'Run {method} validation' again."
            )
            self.warning_label.show()
        else:
            self.warning_label.hide()

        validated_selected = [
            ds
            for ds in selected
            if (method, ds.label) in self._validation_results and ds.label not in stale_labels
        ]
        drt_selected = [
            (ds.label, self._drt_results[ds.label])
            for ds in selected
            if ds.label in self._drt_results
        ]
        self.tabs.setTabText(1, f"Residuals ({len(validated_selected)})")
        self.tabs.setTabText(2, f"DRT ({len(drt_selected)})")
        self.tabs.setTabText(
            3, f"DRT Peaks ({sum(1 for ds in selected if ds.label in self._drt_peaks)})"
        )

        self._pending = dict(
            selected=selected,
            method=method,
            threshold=threshold,
            validated_selected=validated_selected,
            drt_selected=drt_selected,
        )
        self._tab_dirty = {0, 1, 2, 3, 4}
        # _residuals_armed deliberately survives this. Marking the tab dirty
        # is enough: the redraw is deferred until the Residuals tab is the
        # visible one, so a checkbox click from another tab still costs
        # nothing. The one case that does redraw immediately is a sidebar
        # change made while already looking at the residuals, which is the
        # point -- they would otherwise be showing the wrong threshold.
        self._render_active_tab()

    def _on_tab_changed(self, _index: int) -> None:
        self._render_active_tab()

    def _on_plot_residuals_clicked(self) -> None:
        """Arms the Residuals tab and draws it. Stays armed for the rest of
        the session on this file, so the plot persists across tab switches
        and sidebar changes and only needs this click once."""
        self._residuals_armed = True
        self._tab_dirty.add(1)
        self._render_active_tab()

    def _update_residuals_header(self, shown: int, total: int) -> None:
        """Says how much of the selection made it onto the screen.

        Deliberately does not clamp the spin box to the available count:
        QSpinBox truncates its value to fit a lowered maximum, so tracking
        the selection size would quietly rewrite the user's chosen limit
        every time they narrowed the selection (and, at startup, would pin
        it to 1 before any sweep exists). The limit is a standing preference;
        the cap is applied when slicing instead."""
        self.residuals_plot_button.setEnabled(total > 0)
        if total == 0:
            text = "No validated sweeps — run a validation first."
        elif shown == 0:
            text = f"{total} validated sweep(s) ready."
        elif shown < total:
            text = f"Showing {shown} of {total}."
        else:
            text = f"Showing all {total}."
        self.residuals_status_label.setText(text)

    def _force_replot_nyquist(self) -> None:
        """Rebuild the Nyquist tab from current state, bypassing the dirty
        check — used by the plot-area 'Replot' button to guarantee a fresh
        figure even if nothing else changed."""
        if self._pending is None:
            return
        self._tab_dirty.add(0)
        self._render_active_tab()

    def _render_active_tab(self) -> None:
        """Build the figures/text for whichever tab is currently visible, if
        it's still dirty. Other tabs stay dirty and get built the moment the
        user actually clicks over to them."""
        if self._pending is None:
            return
        index = self.tabs.currentIndex()
        if index not in self._tab_dirty:
            return
        p = self._pending

        if index == 0:
            # Removed points are always drawn; the Nyquist pane's "Hide
            # Removed Points" overlay button toggles their visibility.
            title = (
                p["selected"][0].full_label
                if self._mode == "Single"
                else p["selected"][0].source_file
            )
            widget = build_nyquist_plot(
                p["selected"], title=title, style=self._style, show_removed=True
            )
            self.nyquist_pane.set_widget(widget)

        elif index == 1:
            # Unlike every other branch here, this one can decline to draw:
            # it stays empty until "Plot residuals" arms it, and even then
            # only covers the first N of the selection.
            validated = p["validated_selected"]
            shown = validated[: self.residuals_limit_spin.value()] if self._residuals_armed else []
            residual_figs = []
            for ds in shown:
                result = self._validation_results[(p["method"], ds.label)]
                fig_r, _ = plot_residuals(
                    result,
                    title=f"{p['method']} residuals — {ds.label}",
                    threshold=p["threshold"],
                    show=False,
                )
                residual_figs.append(fig_r)
            self.residuals_pane.set_figures(residual_figs)
            self._update_residuals_header(len(shown), len(validated))

        elif index == 2:
            if p["drt_selected"]:
                fig_drt, _ = plot_drt(p["drt_selected"], show=False)
                self.drt_pane.set_figure(fig_drt)
            else:
                self.drt_pane.clear()

        elif index == 3:
            peak_lines = []
            for ds in p["selected"]:
                peaks = self._drt_peaks.get(ds.label)
                if peaks is None:
                    continue
                peak_lines.append(f"=== {ds.label} ({peaks.get_num_peaks()} peak(s)) ===")
                peak_lines.append(peaks.to_peaks_dataframe().to_string(index=False))
                peak_lines.append("")
            self.drt_peaks_text.setPlainText("\n".join(peak_lines))

        elif index == 4:
            lines = []
            for ds in p["selected"]:
                validated_with = [
                    m for m in VALIDATION_METHODS if (m, ds.label) in self._validation_results
                ]
                note = f" (validated: {', '.join(validated_with)})" if validated_with else ""
                lines.append(f"{ds.label} — {ds.num_points} points{note}")
            self.details_text.setPlainText("\n".join(lines))

        self._tab_dirty.discard(index)
