"""Main window: sidebar controls on the left, plot tabs on the right.

Mirrors the Streamlit app's workflow (load -> select -> style -> filter ->
validate) but with explicit event handling instead of rerun-everything.
"""

# Annotations are strings, so a signature may name a core.* type without that
# module being imported at class-definition time (see the import note below).
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

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

# The core.* analysis modules are deliberately NOT imported here -- each use
# site imports what it needs. Together they pull in pyimpspec (and through it
# scipy.signal and sympy) for ~4 s, none of which is needed to put a window on
# screen, and all of which used to be paid before QApplication even existed.
# gui/app.py warms them on a background thread once the window is up, so by
# the time any of these functions is called the import is already a no-op.
# core.drt is the one exception in spirit: its option tuples are read while
# building the sidebar, which is why that module keeps its own pyimpspec
# imports inside its functions.
if TYPE_CHECKING:  # names used only in annotations
    from core import EISParseError

from gui.figure_panes import PgFigureListPane, PgFigurePane
from gui.generic_import_dialog import GenericImportDialog
from gui.theme import THEMES, apply_theme
from gui.workers import DRTWorker, ValidationWorker

SIDEBAR_WIDTH = 320
VALIDATION_METHODS = ("Kramers-Kronig", "Z-HIT")

# How many residuals figures the Residuals tab offers to draw by default.
# Each one costs a full canvas render, so a large selection can still add up
# across a tab visit. Rendering is capped to this many and only happens when
# the user asks for it -- see _build_residuals_tab and
# MainWindow._residuals_armed.
DEFAULT_RESIDUALS_LIMIT = 5
# How many sweeps start checked per loaded file in Overlay mode -- matches
# the single-file default this replaces (the first 5 sweeps), just applied
# per file instead of once overall, so a batch of several files doesn't
# start with everything but the first file's sweeps unchecked.
DEFAULT_SWEEPS_CHECKED_PER_FILE = 5
ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


class LoadedFile(NamedTuple):
    """One entry in MainWindow._files -- the registry of currently loaded
    files, independent of the flat MainWindow._datasets list of every sweep
    across all of them. file_id is what ties a sweep back to its file (see
    core.io_utils.EISDataset.file_id/key); it's a monotonic counter assigned
    by _load_files, not derived from the path, so two files that happen to
    share a name never collide."""
    file_id: int
    path: str
    stem: str
    parser_used: str
    n_sweeps: int


def _titleize(rbf_type: str) -> str:
    """'c2-matern' -> 'C2 Matern', 'piecewise-linear' -> 'Piecewise Linear'."""
    return " ".join(word.capitalize() for word in rbf_type.split("-"))


def _add_combo_items(combo: QComboBox, pairs) -> None:
    """Populate a QComboBox from (display_text, value) pairs, retrievable
    via combo.currentData()."""
    for display, value in pairs:
        combo.addItem(display, value)


def _parse_file(path: str, file_id: int):
    """Try the Modulo Bat cycling-sequence parser first (content-detected
    via its 'Nb header lines' signature, so this is safe to attempt
    regardless of extension), then the standard BioLogic/pyimpspec parser
    for genuine .mpt exports. Returns (datasets, parser_name). Raises
    EISParseError otherwise - including for every .txt/.csv, since
    pyimpspec's own column-guessing heuristics can silently misread an
    unfamiliar plaintext layout instead of raising, which would skip the
    user-confirmed column mapping entirely. The caller should fall back to
    the generic parser on this error (see MainWindow._open_generic_file)."""
    from core import EISParseError, parse_eis_file
    from core.mb_parser import parse_modulobat_file

    mb_error: Optional[Exception] = None
    try:
        return parse_modulobat_file(path, file_id=file_id), "Modulo Bat (cycling sequence)"
    except Exception as exc:
        mb_error = exc

    if Path(path).suffix.lower() == ".mpt":
        try:
            return parse_eis_file(path, file_id=file_id), "Standard EIS export"
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

        # Every loaded file's registry entry (see LoadedFile), and the flat
        # list of every sweep across all of them -- order matches _files
        # (each _load_files call extends both together), but code that needs
        # "this sweep's file" should go through ds.file_id, not position.
        self._files: List[LoadedFile] = []
        self._datasets: List = []
        # Monotonic counter, not reused after a file is removed -- so a
        # removed file's old keys can never collide with a newly added one.
        self._next_file_id = 0
        # {(method, ds.key): KramersKronigResult | ZHITResult}. Keyed by
        # ds.key rather than ds.label -- "Set 01" is only unique within a
        # single file, so a label-keyed dict would collide as soon as a
        # second file's "Set 01" was validated. Same reasoning applies to
        # every dict below that used to be label-keyed.
        self._validation_results = {}
        # {(method, ds.key): effective kwargs used for that run} --
        # saved alongside the result so a reloaded session can explain (or
        # reproduce) how it was computed; see core.session for the schema.
        self._validation_params: Dict[Tuple[str, str], dict] = {}
        # {ds.key: set of point indices} — the eraser's per-point
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
        # {ds.key: TRRBFResult | BHTResult} — last DRT run wins
        self._drt_results = {}
        # {ds.key: effective kwargs used for that run} -- see
        # _validation_params above.
        self._drt_params: Dict[str, dict] = {}
        # {ds.key: DRTPeaks}
        self._drt_peaks = {}
        self._drt_worker: Optional[DRTWorker] = None
        self._drt_worker_errors: List[Tuple[str, str]] = []
        # Params for the Bayesian batch currently in flight -- the worker's
        # result_ready signal only carries back (label, result), so this is
        # how _on_drt_worker_result learns what settings produced it.
        self._pending_drt_params: dict = {}
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
        self.save_session_action = QAction("&Save session…", self)
        self.save_session_action.setShortcut("Ctrl+S")
        self.save_session_action.setStatusTip(
            "Save the loaded sweeps, validation/DRT results, and filter state to a file."
        )
        self.save_session_action.triggered.connect(self._save_session)

        self.open_session_action = QAction("&Open session…", self)
        self.open_session_action.setShortcut("Ctrl+O")
        self.open_session_action.setStatusTip("Restore a previously saved session.")
        self.open_session_action.triggered.connect(self._load_session)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_session_action)
        file_menu.addAction(self.save_session_action)

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
        # Plain tuples of option strings; core.drt keeps its pyimpspec imports
        # inside its functions so reading them here stays free.
        from core.drt import RBF_TYPES

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. Load files
        load_box = QGroupBox("1. Load files")
        load_layout = QVBoxLayout(load_box)

        open_row = QHBoxLayout()
        self.open_button = QPushButton("Open EIS exports…")
        self.open_button.setToolTip(
            "Replace everything currently loaded with the file(s) you pick."
        )
        self.open_button.clicked.connect(self._open_file)
        open_row.addWidget(self.open_button)

        self.add_files_button = QPushButton("Add files…")
        self.add_files_button.setToolTip(
            "Load more file(s) alongside what's already open, keeping all "
            "existing validation/DRT results."
        )
        self.add_files_button.clicked.connect(self._add_files_dialog)
        open_row.addWidget(self.add_files_button)
        load_layout.addLayout(open_row)

        self.file_list = QListWidget()
        self.file_list.setToolTip("Loaded files. Select one and click Remove to drop it.")
        self.file_list.setMaximumHeight(90)
        load_layout.addWidget(self.file_list)

        self.remove_file_button = QPushButton("Remove selected file")
        self.remove_file_button.clicked.connect(self._on_remove_file_clicked)
        load_layout.addWidget(self.remove_file_button)
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

        self.select_buttons_row = QWidget()
        select_buttons_layout = QHBoxLayout(self.select_buttons_row)
        select_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.select_all_button = QPushButton("Select all")
        self.select_all_button.clicked.connect(lambda: self._set_all_sweeps_checked(True))
        select_buttons_layout.addWidget(self.select_all_button)
        self.select_none_button = QPushButton("Select none")
        self.select_none_button.clicked.connect(lambda: self._set_all_sweeps_checked(False))
        select_buttons_layout.addWidget(self.select_none_button)
        self.select_buttons_row.setVisible(False)
        select_layout.addWidget(self.select_buttons_row)
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
            "On the Visual tab (either view), click a point to remove it, or a "
            "removed (grey ×) point to restore it. On the Bode plot the grey × "
            "markers are on the |Z| series. Manual edits override the filter "
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
        self.visual_pane = PgFigurePane(with_overlay_actions=True)
        self.visual_pane.replot_requested.connect(self._force_replot_visual)
        self.visual_pane.point_mask_toggled.connect(self._on_point_mask_toggled)
        self.residuals_pane = PgFigureListPane()
        self.drt_pane = PgFigurePane(with_overlay_actions=False)
        self.drt_peaks_text = QPlainTextEdit()
        self.drt_peaks_text.setReadOnly(True)
        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.tabs.addTab(self._build_visual_tab(), "Visual")
        self.tabs.addTab(self._build_residuals_tab(), "Residuals")
        self.tabs.addTab(self.drt_pane, "DRT")
        self.tabs.addTab(self.drt_peaks_text, "DRT Peaks")
        self.tabs.addTab(self.details_text, "Sweep details")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.stack.addWidget(self.tabs)

        layout.addWidget(self.stack, stretch=1)
        return container

    def _build_visual_tab(self) -> QWidget:
        """The spectrum plot plus the header that picks how to draw it. Nyquist
        and Bode are two views of the same thing -- same selection, same style,
        same eraser -- so the choice only decides which builder
        _render_active_tab calls, and nothing about it needs remembering
        elsewhere (it isn't part of the saved session, same as the active tab)."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(6, 4, 6, 4)
        header.setSpacing(6)
        header.addWidget(QLabel("Plot"))

        self.nyquist_view_radio = QRadioButton("Nyquist")
        self.nyquist_view_radio.setToolTip("-Z'' against Z', on equal-aspect axes.")
        self.bode_view_radio = QRadioButton("Bode")
        self.bode_view_radio.setToolTip(
            "|Z| (filled circles, left axis) and -phase (hollow circles, right "
            "axis) against frequency, with frequency and |Z| drawn as decades. "
            "A fixed view — use Auto-Scale or Replot to reframe it."
        )
        self.nyquist_view_radio.setChecked(True)
        view_group = QButtonGroup(self)
        view_group.addButton(self.nyquist_view_radio)
        view_group.addButton(self.bode_view_radio)
        # One connection, not one per radio: this fires after the group has
        # settled, so reading _visual_view below already gives the new choice.
        self.nyquist_view_radio.toggled.connect(self._on_visual_view_changed)
        header.addWidget(self.nyquist_view_radio)
        header.addWidget(self.bode_view_radio)
        header.addStretch()

        col.addLayout(header)
        col.addWidget(self.visual_pane, stretch=1)
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
    def _visual_view(self) -> str:
        return "Nyquist" if self.nyquist_view_radio.isChecked() else "Bode"

    @property
    def _validation_method(self) -> str:
        return VALIDATION_METHODS[0] if self.kk_radio.isChecked() else VALIDATION_METHODS[1]

    @property
    def _multi_file(self) -> bool:
        """Whether more than one file is currently loaded -- when true,
        legends/titles/messages qualify sweep names with their source file
        (ds.qualified_label) instead of the bare ds.label, which is only
        unique within a single file."""
        return len(self._files) > 1

    def _display_label(self, ds_or_key) -> str:
        """A sweep's display name for legends/titles/messages: qualified
        with its source file when several are loaded, plain otherwise.
        Accepts either an EISDataset or a ds.key (e.g. from a worker signal
        or an error list, where only the key survived the round trip)."""
        ds = ds_or_key
        if isinstance(ds_or_key, str):
            ds = next((d for d in self._datasets if d.key == ds_or_key), None)
            if ds is None:
                return ds_or_key
        return ds.qualified_label if self._multi_file else ds.label

    def _build_style_map(self) -> Dict[str, Tuple[str, str]]:
        """ds.key -> (color, pg symbol) for every currently loaded sweep:
        color by sweep index (so the same sweep number matches across files),
        symbol by the sweep's file's position in _files (so files stay
        tellable apart regardless of which sweeps are selected). Passed to
        build_nyquist_plot; see core.plotting.TAB10/PG_MARKERS."""
        from core.plotting import PG_MARKERS, TAB10

        file_position = {lf.file_id: i for i, lf in enumerate(self._files)}
        return {
            ds.key: (
                TAB10[ds.index % len(TAB10)],
                PG_MARKERS[file_position.get(ds.file_id, 0) % len(PG_MARKERS)],
            )
            for ds in self._datasets
        }

    def _selected_datasets(self) -> List:
        if not self._datasets:
            return []
        if self._mode == "Single":
            key = self.sweep_combo.currentData()
            ds = next((d for d in self._datasets if d.key == key), None)
            return [ds] if ds is not None else []
        checked_keys = {
            self.sweep_list.item(i).data(Qt.UserRole)
            for i in range(self.sweep_list.count())
            if (self.sweep_list.item(i).flags() & Qt.ItemIsUserCheckable)
            and self.sweep_list.item(i).checkState() == Qt.Checked
        }
        return [ds for ds in self._datasets if ds.key in checked_keys]

    def _apply_manual_overrides(self, ds) -> None:
        """Re-assert this sweep's eraser edits over whatever the automatic
        filters have just decided. A no-op for sweeps that have none."""
        masked = self._manual_masked.get(ds.key)
        kept = self._manual_kept.get(ds.key)
        if masked or kept:
            from core.filtering import apply_manual_overrides

            apply_manual_overrides(ds, masked or (), kept or ())

    def _update_validation_button_text(self) -> None:
        # Keep it short — the selected method is shown by the radios above.
        if self._worker is not None:
            self.run_validation_button.setText("Running…")
        else:
            self.run_validation_button.setText("Run validation")

    def _refresh_file_list_widget(self) -> None:
        self.file_list.clear()
        for lf in self._files:
            item = QListWidgetItem(f"{lf.stem}  ({lf.n_sweeps} sweep(s))")
            item.setData(Qt.UserRole, lf.file_id)
            item.setToolTip(f"{lf.path}\nParsed with: {lf.parser_used}")
            self.file_list.addItem(item)

    def _populate_sweep_selectors(self, checked_keys=None) -> None:
        """(Re)fill the Single/Overlay selectors from the current _files/
        _datasets, without the itemChanged/currentIndexChanged handlers
        firing partway through. The Overlay list is grouped by file, with a
        disabled header row per file, since sweep labels ("Set 01") repeat
        across files and only the header makes clear which is which.

        checked_keys controls which sweeps start checked in Overlay mode --
        defaults to the first DEFAULT_SWEEPS_CHECKED_PER_FILE of each file,
        same as a fresh load; a session restore passes back whichever sweeps
        were checked when it was saved."""
        multi_file = self._multi_file
        self.sweep_combo.blockSignals(True)
        self.sweep_combo.clear()
        for ds in self._datasets:
            self.sweep_combo.addItem(ds.qualified_label if multi_file else ds.label, ds.key)
        if self.sweep_combo.count():
            self.sweep_combo.setCurrentIndex(0)
        self.sweep_combo.blockSignals(False)

        by_file: Dict[int, List] = defaultdict(list)
        for ds in self._datasets:
            by_file[ds.file_id].append(ds)

        self.sweep_list.blockSignals(True)
        self.sweep_list.clear()
        for lf in self._files:
            header_item = QListWidgetItem(lf.stem)
            header_item.setFlags(Qt.NoItemFlags)
            bold_font = header_item.font()
            bold_font.setBold(True)
            header_item.setFont(bold_font)
            self.sweep_list.addItem(header_item)

            for i, ds in enumerate(by_file[lf.file_id]):
                item = QListWidgetItem(ds.label)
                item.setData(Qt.UserRole, ds.key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                if checked_keys is None:
                    checked = i < DEFAULT_SWEEPS_CHECKED_PER_FILE
                else:
                    checked = ds.key in checked_keys
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                self.sweep_list.addItem(item)
        self.sweep_list.blockSignals(False)

    def _set_all_sweeps_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self.sweep_list.blockSignals(True)
        for i in range(self.sweep_list.count()):
            item = self.sweep_list.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(state)
        self.sweep_list.blockSignals(False)
        self._refresh()

    # ------------------------------------------------------------ handlers

    def _open_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open EIS exports", "", "EIS exports (*.mpt *.txt *.csv)"
        )
        if not paths:
            return
        self._load_files(paths, clear_first=True)

    def _add_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add EIS exports", "", "EIS exports (*.mpt *.txt *.csv)"
        )
        if not paths:
            return
        self._load_files(paths, clear_first=False)

    def _reset_state(self) -> None:
        """Clears everything a fresh 'Open' replaces. Not called for 'Add
        files' -- keeping existing results is the whole point of that button."""
        self._files = []
        self._datasets = []
        self._validation_results = {}
        self._validation_params = {}
        self._drt_results = {}
        self._drt_params = {}
        self._drt_peaks = {}
        self._manual_masked = {}
        self._manual_kept = {}
        # Nothing validated yet, so start the Residuals tab blank again
        # rather than inheriting whatever was armed before.
        self._residuals_armed = False

    def _load_files(self, paths: List[str], clear_first: bool) -> None:
        """Parses each path and, if at least one succeeds, commits them to
        _files/_datasets -- replacing everything first if clear_first (the
        'Open' button), or appending (the 'Add files' button). Parsing is
        staged before anything is committed, so a batch that fails outright
        (e.g. every file cancelled out of its generic-import dialog) leaves
        existing state untouched, matching the old single-file behavior."""
        from core import EISParseError

        generic_cache: dict = {}
        failures: List[Tuple[str, str]] = []
        staged: List[Tuple[LoadedFile, List]] = []
        next_id = 0 if clear_first else self._next_file_id

        for path in paths:
            file_id = next_id
            try:
                datasets, parser_used = _parse_file(path, file_id)
            except EISParseError as exc:
                result = self._open_generic_file(path, exc, file_id, generic_cache)
                if result is None:
                    failures.append((Path(path).name, str(exc)))
                    continue
                datasets, parser_used = result

            next_id += 1
            staged.append((
                LoadedFile(file_id, path, Path(path).stem, parser_used, len(datasets)),
                datasets,
            ))

        if failures:
            details = "\n".join(f"- {name}: {msg}" for name, msg in failures)
            QMessageBox.warning(
                self, "Some files failed to load", f"Could not load:\n{details}"
            )

        if not staged:
            return

        if clear_first:
            self._reset_state()
        self._next_file_id = next_id

        for lf, datasets in staged:
            self._files.append(lf)
            self._datasets.extend(datasets)

        self._refresh_file_list_widget()
        self._populate_sweep_selectors()
        self.stack.setCurrentWidget(self.tabs)
        self.statusBar().showMessage(
            f"Loaded {len(self._datasets)} sweep(s) from {len(self._files)} file(s) total."
        )
        self._refresh()

    def _open_generic_file(
        self, path: str, prior_error: EISParseError, file_id: int, generic_cache: dict
    ):
        """Fall back to the generic txt/csv parser: sniff headers/roles,
        let the user confirm/correct the mapping in a dialog, then parse.
        Returns (datasets, parser_name), or None if the user cancelled or
        the file couldn't be parsed at all.

        generic_cache carries the last accepted mapping across calls within
        one _load_files batch: if the user checked "apply to all remaining
        files" and this file's headers match exactly, the dialog is skipped
        and that mapping is reused directly."""
        from core.generic_parser import parse_generic_file, sniff_columns

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

        if generic_cache.get("headers") == headers:
            column_roles = generic_cache["roles"]
        else:
            dialog = GenericImportDialog(
                Path(path).name, headers, sample_rows, guessed_roles, parent=self
            )
            if dialog.exec() != QDialog.Accepted:
                return None
            column_roles = dialog.column_roles()
            if dialog.apply_to_all():
                generic_cache["headers"] = headers
                generic_cache["roles"] = column_roles

        # Deliberately broad: anything the mapped columns turn out not to
        # support (a frequency column that isn't one, a value DataSet
        # rejects) belongs in a dialog. Letting it propagate out of this
        # slot aborts the whole process, losing every already-loaded file.
        try:
            datasets = parse_generic_file(path, column_roles=column_roles, file_id=file_id)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Parse error",
                f"Could not parse '{Path(path).name}' with the selected "
                f"column mapping.\n{exc}",
            )
            return None

        return datasets, "Generic txt/csv export (user-confirmed)"

    def _on_remove_file_clicked(self) -> None:
        item = self.file_list.currentItem()
        if item is None:
            return
        self._remove_file(item.data(Qt.UserRole))

    def _remove_file(self, file_id: int) -> None:
        """Drops one loaded file and every cache entry keyed to one of its
        sweeps (validation/DRT/peaks/eraser overrides), identified via
        ds.key so a same-labeled sweep in a different file is untouched."""
        removed_keys = {ds.key for ds in self._datasets if ds.file_id == file_id}
        self._files = [lf for lf in self._files if lf.file_id != file_id]
        self._datasets = [ds for ds in self._datasets if ds.file_id != file_id]

        self._validation_results = {
            k: v for k, v in self._validation_results.items() if k[1] not in removed_keys
        }
        self._validation_params = {
            k: v for k, v in self._validation_params.items() if k[1] not in removed_keys
        }
        self._drt_results = {k: v for k, v in self._drt_results.items() if k not in removed_keys}
        self._drt_params = {k: v for k, v in self._drt_params.items() if k not in removed_keys}
        self._drt_peaks = {k: v for k, v in self._drt_peaks.items() if k not in removed_keys}
        self._manual_masked = {
            k: v for k, v in self._manual_masked.items() if k not in removed_keys
        }
        self._manual_kept = {
            k: v for k, v in self._manual_kept.items() if k not in removed_keys
        }

        self._refresh_file_list_widget()
        self._populate_sweep_selectors()

        if not self._datasets:
            self.stack.setCurrentWidget(self.stack.widget(0))
            self.statusBar().showMessage("Open a .mpt, .txt, or .csv file to begin.")
            self.visual_pane.clear()
            self.residuals_pane.clear()
            self.drt_pane.clear()
            self.drt_peaks_text.clear()
            self.details_text.clear()
            self._pending = None
            self._tab_dirty.clear()
            return

        self.statusBar().showMessage(
            f"Removed file. {len(self._datasets)} sweep(s) from {len(self._files)} file(s) remain."
        )
        self._refresh()

    @staticmethod
    def _files_from_datasets(datasets: List) -> List[LoadedFile]:
        """Rebuilds the file registry after a session restore. A saved
        session has no separate file-list record (see core.session) -- just
        datasets that each already remember their own file_id/source_file,
        which is everything grouping needs. parser_used/path aren't
        recoverable, so both get a placeholder; they're informational only
        (tooltip/status text), never read for identity."""
        by_file: Dict[int, List] = defaultdict(list)
        for ds in datasets:
            by_file[ds.file_id].append(ds)
        return [
            LoadedFile(
                file_id=file_id,
                path=group[0].source_file,
                stem=group[0].source_file,
                parser_used="restored session",
                n_sweeps=len(group),
            )
            for file_id, group in sorted(by_file.items())
        ]

    def _save_session(self) -> None:
        if not self._datasets:
            QMessageBox.information(self, "Save session", "Open a file first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "", "EIS sessions (*.eisz)"
        )
        if not path:
            return
        if not path.lower().endswith(".eisz"):
            path += ".eisz"

        from core.session import save_session, ui_state_to_dict

        ui_state = ui_state_to_dict(
            self._manual_masked,
            self._manual_kept,
            self._validation_method,
            self.inductive_check.isChecked(),
            self.threshold_spin.value(),
        )
        try:
            save_session(
                path,
                self._datasets,
                self._validation_results,
                self._drt_results,
                self._drt_peaks,
                self._validation_params,
                self._drt_params,
                ui_state,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save session", f"Could not save session:\n{exc}")
            return

        self.statusBar().showMessage(f"Session saved to '{Path(path).name}'.")

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session", "", "EIS sessions (*.eisz *.json)"
        )
        if not path:
            return

        from core.session import load_session

        try:
            (
                datasets,
                validation_results,
                drt_results,
                drt_peaks,
                validation_params,
                drt_params,
                ui_state,
            ) = load_session(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open session", f"Could not open session:\n{exc}")
            return

        self._datasets = datasets
        self._files = self._files_from_datasets(datasets)
        self._next_file_id = max((lf.file_id for lf in self._files), default=-1) + 1
        self._validation_results = validation_results
        self._validation_params = validation_params
        self._drt_results = drt_results
        self._drt_params = drt_params
        self._drt_peaks = drt_peaks
        self._manual_masked = ui_state["manual_masked"]
        self._manual_kept = ui_state["manual_kept"]
        # Same reasoning as a fresh file load: nothing has been plotted yet
        # in this window instance, so the Residuals tab starts blank rather
        # than inheriting whatever was armed before the load.
        self._residuals_armed = False

        self._refresh_file_list_widget()
        self._populate_sweep_selectors()

        # Restore the filter widgets without each one triggering its own
        # _refresh() -- a single _refresh() below covers all of them once
        # the datasets/results/overrides above are already in place.
        method = ui_state.get("validation_method")
        self.kk_radio.blockSignals(True)
        self.zhit_radio.blockSignals(True)
        if method == VALIDATION_METHODS[1]:
            self.zhit_radio.setChecked(True)
        else:
            self.kk_radio.setChecked(True)
        self.kk_radio.blockSignals(False)
        self.zhit_radio.blockSignals(False)
        self._update_validation_button_text()

        self.inductive_check.blockSignals(True)
        self.inductive_check.setChecked(bool(ui_state.get("inductive_filter", False)))
        self.inductive_check.blockSignals(False)

        threshold = ui_state.get("residual_threshold")
        if threshold is not None:
            self.threshold_spin.blockSignals(True)
            self.threshold_spin.setValue(threshold)
            self.threshold_spin.blockSignals(False)

        self.stack.setCurrentWidget(self.tabs)
        self.statusBar().showMessage(
            f"Restored session from '{Path(path).name}' "
            f"({len(datasets)} sweep(s))."
        )
        self._refresh()

    def _on_mode_changed(self) -> None:
        single = self._mode == "Single"
        self.sweep_combo.setVisible(single)
        self.sweep_list.setVisible(not single)
        self.select_buttons_row.setVisible(not single)
        self._refresh()

    def _on_method_changed(self, checked: bool) -> None:
        if not checked:
            return  # fires once per radio; only act on the newly-checked one
        self._update_validation_button_text()
        self._refresh()

    def _on_eraser_toggled(self, checked: bool) -> None:
        """Purely a mode switch — no mask changes, so no _refresh()."""
        self.visual_pane.set_eraser_enabled(checked)
        if checked:
            self.statusBar().showMessage(
                "Eraser on — click a point on the Visual tab's plot to remove "
                "it, or a grey × to restore it."
            )
        else:
            self.statusBar().showMessage("Eraser off.")

    def _on_point_mask_toggled(self, key: str, index: int) -> None:
        """Flip one point's manual override, then let _refresh() recompose the
        mask and redraw.

        Which way it flips is read from the sweep's *current* mask rather than
        from the override sets, so a point the filters removed toggles to
        'keep' on the first click regardless of how it got removed."""
        ds = next((d for d in self._datasets if d.key == key), None)
        if ds is None:
            return

        masked = self._manual_masked.setdefault(key, set())
        kept = self._manual_kept.setdefault(key, set())
        if ds.data.get_mask().get(index, False):
            masked.discard(index)
            kept.add(index)
            action = "restored"
        else:
            kept.discard(index)
            masked.add(index)
            action = "removed"

        self._refresh()
        self.statusBar().showMessage(f"{self._display_label(ds)}: point {index + 1} {action}.")

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
        # Module-level functions either way, so ValidationWorker can still
        # pickle the runner by reference when it spreads a batch over
        # processes -- a local import binds the same object.
        from core.validation import run_kk_test, run_zhit

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

    def _on_validation_result(self, method: str, key: str, result) -> None:
        self._validation_results[(method, key)] = result
        # Z-HIT is run with no extra kwargs here; KK's non-default arguments
        # live in core.validation.run_kk_test's own signature (admittance,
        # num_F_ext_evaluations), so mirror them rather than hardcoding a
        # second copy of pyimpspec's defaults.
        if method == VALIDATION_METHODS[0]:
            self._validation_params[(method, key)] = {
                "admittance": False,
                "num_F_ext_evaluations": 10,
            }
        else:
            self._validation_params[(method, key)] = {}

    def _on_validation_progress(self, done: int, total: int) -> None:
        """Sweeps finish out of order (they run in a process pool), so this
        is a count, not a name — a batch can otherwise sit on one unchanging
        status message for minutes."""
        self.statusBar().showMessage(
            f"Running {self._running_method} analysis… {done} of {total}"
        )

    def _on_validation_error(self, key: str, message: str) -> None:
        self._worker_errors.append((key, message))

    def _on_validation_finished(self) -> None:
        self._worker = None
        self._sidebar.setEnabled(True)
        self._update_validation_button_text()
        self.statusBar().showMessage("Validation finished.")
        if self._worker_errors:
            details = "\n".join(
                f"- {self._display_label(key)}: {msg}" for key, msg in self._worker_errors
            )
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
            result = self._drt_results.get(selected[0].key)
            lambda_value = getattr(result, "lambda_value", None)
        self.drt_optimal_lambda_label.setText(
            f"{lambda_value:.4g}" if lambda_value is not None else "—"
        )

    def _run_drt_simple(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        from core.drt import run_drt

        settings = self._drt_settings()
        params = dict(
            settings,
            mode=self.drt_mode_combo.currentData(),
            inductance=self.drt_inductance_check.isChecked(),
            cross_validation=self.drt_cv_combo.currentData(),
            lambda_value=self.drt_lambda_spin.value(),
            credible_intervals=False,
        )
        errors = []
        for ds in selected:
            try:
                self._drt_results[ds.key] = run_drt(ds, **params)
            except Exception as exc:
                errors.append((ds.key, str(exc)))
            else:
                self._drt_params[ds.key] = params

        if errors:
            details = "\n".join(f"- {self._display_label(key)}: {msg}" for key, msg in errors)
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

        from core.drt import run_drt

        settings = self._drt_settings()
        params = dict(
            settings,
            mode=self.drt_mode_combo.currentData(),
            inductance=self.drt_inductance_check.isChecked(),
            cross_validation=self.drt_cv_combo.currentData(),
            lambda_value=self.drt_lambda_spin.value(),
            credible_intervals=True,
            num_samples=self.drt_num_samples_spin.value(),
            timeout=self.drt_timeout_spin.value(),
        )

        def runner(ds):
            return run_drt(ds, **params)

        self._pending_drt_params = params
        self._drt_worker_errors = []
        self._drt_worker = DRTWorker(runner, selected, parent=self)
        self._drt_worker.result_ready.connect(self._on_drt_worker_result)
        self._drt_worker.error.connect(self._on_drt_worker_error)
        self._drt_worker.finished.connect(self._on_drt_worker_finished)
        self._sidebar.setEnabled(False)
        self.statusBar().showMessage("Running Bayesian DRT… this may take a while.")
        self._drt_worker.start()

    def _on_drt_worker_result(self, key: str, result) -> None:
        self._drt_results[key] = result
        self._drt_params[key] = self._pending_drt_params

    def _on_drt_worker_error(self, key: str, message: str) -> None:
        self._drt_worker_errors.append((key, message))

    def _on_drt_worker_finished(self) -> None:
        self._drt_worker = None
        self._sidebar.setEnabled(True)
        self.statusBar().showMessage("Bayesian DRT finished.")
        if self._drt_worker_errors:
            details = "\n".join(
                f"- {self._display_label(key)}: {msg}" for key, msg in self._drt_worker_errors
            )
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")
        self._update_optimal_lambda_label(self._selected_datasets())
        self._refresh()

    def _run_drt_bht(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        from core.drt import run_drt_bht

        settings = self._drt_settings()
        params = dict(settings, num_samples=self.drt_num_samples_spin.value())
        errors = []
        for ds in selected:
            try:
                self._drt_results[ds.key] = run_drt_bht(ds, **params)
            except Exception as exc:
                errors.append((ds.key, str(exc)))
            else:
                self._drt_params[ds.key] = params

        if errors:
            details = "\n".join(f"- {self._display_label(key)}: {msg}" for key, msg in errors)
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

        from core.drt import analyze_drt_peaks

        num_peaks = self.drt_num_peaks_spin.value()
        errors = []
        for ds in selected:
            result = self._drt_results.get(ds.key)
            if result is None:
                errors.append((ds.key, "Run a DRT calculation first."))
                continue
            try:
                self._drt_peaks[ds.key] = analyze_drt_peaks(result, num_peaks=num_peaks)
            except Exception as exc:
                errors.append((ds.key, str(exc)))

        if errors:
            details = "\n".join(f"- {self._display_label(key)}: {msg}" for key, msg in errors)
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

        # Only reachable once a file has been loaded, so these are warm.
        from core.filtering import clear_mask, mask_inductive_points
        from core.validation import mask_residual_outliers

        selected = self._selected_datasets()
        if not selected:
            self.warning_label.setText("Select at least one sweep to plot.")
            self.warning_label.show()
            self.visual_pane.clear()
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

        stale_keys = []
        for ds in selected:
            result = self._validation_results.get((method, ds.key))
            if result is not None:
                try:
                    mask_residual_outliers(ds, result, threshold)
                except ValueError:
                    stale_keys.append(ds.key)
            # Again, on top of the outlier pass -- which only ever adds masks,
            # so this is what lets a manually restored point survive a
            # threshold that would otherwise drop it.
            self._apply_manual_overrides(ds)

        if stale_keys:
            names = ", ".join(self._display_label(k) for k in stale_keys)
            self.warning_label.setText(
                f"{method} results for {names} no longer match "
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
            if (method, ds.key) in self._validation_results and ds.key not in stale_keys
        ]
        drt_selected = [
            (self._display_label(ds), self._drt_results[ds.key])
            for ds in selected
            if ds.key in self._drt_results
        ]
        self.tabs.setTabText(1, f"Residuals ({len(validated_selected)})")
        self.tabs.setTabText(2, f"DRT ({len(drt_selected)})")
        self.tabs.setTabText(
            3, f"DRT Peaks ({sum(1 for ds in selected if ds.key in self._drt_peaks)})"
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

    def _force_replot_visual(self) -> None:
        """Rebuild the Visual tab from current state, bypassing the dirty
        check — used by the plot-area 'Replot' button to guarantee a fresh
        figure even if nothing else changed."""
        if self._pending is None:
            return
        self._tab_dirty.add(0)
        self._render_active_tab()

    def _on_visual_view_changed(self, _checked: bool) -> None:
        """Redraw the Visual tab in the newly picked view. Nothing about the
        masking/validation bookkeeping depends on it, so this goes straight to
        the plot rather than through _refresh()."""
        self._force_replot_visual()

    def _render_active_tab(self) -> None:
        """Build the figures/text for whichever tab is currently visible, if
        it's still dirty. Other tabs stay dirty and get built the moment the
        user actually clicks over to them."""
        if self._pending is None:
            return
        index = self.tabs.currentIndex()
        if index not in self._tab_dirty:
            return

        from core.plotting import (
            build_bode_plot,
            build_drt_plot,
            build_nyquist_plot,
            build_residuals_plot,
        )

        p = self._pending

        if index == 0:
            # Removed points are always drawn; hiding them is done by clicking
            # the "Removed" legend entry.
            if self._mode == "Single":
                title = p["selected"][0].full_label
            else:
                files_in_selection = {ds.file_id for ds in p["selected"]}
                title = (
                    f"{len(files_in_selection)} files · {len(p['selected'])} sweeps"
                    if len(files_in_selection) > 1
                    else p["selected"][0].source_file
                )
            # Both builders take the same arguments; the header radios pick one.
            build = (
                build_nyquist_plot if self._visual_view == "Nyquist" else build_bode_plot
            )
            widget = build(
                p["selected"], title=title, style=self._style, show_removed=True,
                style_map=self._build_style_map(),
            )
            self.visual_pane.set_widget(widget)

        elif index == 1:
            # Unlike every other branch here, this one can decline to draw:
            # it stays empty until "Plot residuals" arms it, and even then
            # only covers the first N of the selection.
            validated = p["validated_selected"]
            shown = validated[: self.residuals_limit_spin.value()] if self._residuals_armed else []
            residual_widgets = []
            for ds in shown:
                result = self._validation_results[(p["method"], ds.key)]
                widget_r = build_residuals_plot(
                    result,
                    title=f"{p['method']} residuals — {self._display_label(ds)}",
                    threshold=p["threshold"],
                )
                residual_widgets.append(widget_r)
            self.residuals_pane.set_widgets(residual_widgets)
            self._update_residuals_header(len(shown), len(validated))

        elif index == 2:
            if p["drt_selected"]:
                widget_drt = build_drt_plot(p["drt_selected"])
                self.drt_pane.set_widget(widget_drt)
            else:
                self.drt_pane.clear()

        elif index == 3:
            peak_lines = []
            for ds in p["selected"]:
                peaks = self._drt_peaks.get(ds.key)
                if peaks is None:
                    continue
                display = self._display_label(ds)
                peak_lines.append(f"=== {display} ({peaks.get_num_peaks()} peak(s)) ===")
                peak_lines.append(peaks.to_peaks_dataframe().to_string(index=False))
                peak_lines.append("")
            self.drt_peaks_text.setPlainText("\n".join(peak_lines))

        elif index == 4:
            lines = []
            for ds in p["selected"]:
                validated_with = [
                    m for m in VALIDATION_METHODS if (m, ds.key) in self._validation_results
                ]
                note = f" (validated: {', '.join(validated_with)})" if validated_with else ""
                lines.append(f"{self._display_label(ds)} — {ds.num_points} points{note}")
            self.details_text.setPlainText("\n".join(lines))

        self._tab_dirty.discard(index)
