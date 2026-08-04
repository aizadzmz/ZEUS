"""Main window: a step bar across the top, one page per stage below it."""

# Annotations are strings, so signatures can name core.* types without
# importing those modules here.
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Do not import core.* at module level: they pull in pyimpspec (~4 s) which is
# not needed to show a window. Import inside each use site instead; gui/app.py
# warms them on a background thread so those imports are no-ops by then.
if TYPE_CHECKING:  # names used only in annotations
    from core import EISParseError

from gui import style
from gui.generic_import_dialog import GenericImportDialog
from gui.selection import SweepSelection
from gui.steps.base import (
    DEFAULT_SETTINGS_WIDTH,
    MAX_SETTINGS_WIDTH,
    MIN_SETTINGS_WIDTH,
    add_combo_items,
)
from gui.steps.data_viz_step import DataVizStep
from gui.steps.drt_step import DRTStep
from gui.steps.ecm_step import ECMStep
from gui.steps.validation_step import ValidationStep
from gui.stepper import STEPS, StepBar
from gui.theme import THEMES, apply_theme, diagram_colors
from gui.workers import DRTWorker, ECMWorker, ValidationWorker

VALIDATION_METHODS = ("Kramers-Kronig", "Z-HIT")

# Circuit schematics the ECM step draws before falling back to the text
# report. One per (sweep, fitted circuit).
MAX_CIRCUIT_DIAGRAMS = 12
ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


class LoadedFile(NamedTuple):
    """One loaded file in MainWindow._files. file_id ties a sweep back to its
    file and is a counter, so same-named files never collide."""
    file_id: int
    path: str
    stem: str
    parser_used: str
    n_sweeps: int


def _parse_file(path: str, file_id: int):
    """Try the Modulo Bat parser, then the BioLogic/pyimpspec one for .mpt.
    Returns (datasets, parser_name) or raises EISParseError."""
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

        # Loaded files, and the flat list of every sweep across them. To find
        # a sweep's file use ds.file_id, not list position.
        self._files: List[LoadedFile] = []
        self._datasets: List = []
        # Active/on-screen sweeps (see gui.selection). Built before the
        # widgets, which bind to it in _build_ui.
        self._selection = SweepSelection(self)
        # Never reused, so a removed file's keys cannot collide with a new one.
        self._next_file_id = 0
        # {(method, ds.key): KramersKronigResult | ZHITResult}. All caches here
        # key on ds.key, not ds.label -- labels like "Set 01" repeat across files.
        self._validation_results = {}
        # {(method, ds.key): effective kwargs} -- saved with the result so a
        # reloaded session can reproduce it; see core.session for the schema.
        self._validation_params: Dict[Tuple[str, str], dict] = {}
        # {ds.key: point indices} -- eraser overrides. Kept out of the DataSet
        # mask, which _refresh() rebuilds from the filters; re-applied as a layer.
        self._manual_masked: Dict[str, set] = {}
        self._manual_kept: Dict[str, set] = {}
        self._worker: Optional[ValidationWorker] = None
        self._worker_errors: List[Tuple[str, str]] = []
        # Method of the run in flight, for the progress message.
        self._running_method = ""
        # {ds.key: TRRBFResult | BHTResult} -- last DRT run wins
        self._drt_results = {}
        # {ds.key: effective kwargs}
        self._drt_params: Dict[str, dict] = {}
        # {ds.key: DRTPeaks}
        self._drt_peaks = {}
        self._drt_worker: Optional[DRTWorker] = None
        self._drt_worker_errors: List[Tuple[str, str]] = []
        # Settings for the batch in flight: result_ready carries only
        # (label, result), so _on_drt_worker_result reads them from here.
        self._pending_drt_params: dict = {}
        # {(canonical cdc, ds.key): FitResult}. Keyed by circuit as well as
        # sweep, so fitting a second circuit keeps the first one's result.
        self._ecm_results: Dict[Tuple[str, str], object] = {}
        # {(canonical cdc, ds.key): effective kwargs}
        self._ecm_params: Dict[Tuple[str, str], dict] = {}
        # Which fitted circuit the ECM step draws; one overlay at a time.
        self._ecm_shown_cdc: Optional[str] = None
        self._ecm_worker: Optional[ECMWorker] = None
        self._ecm_worker_errors: List[Tuple[str, str]] = []
        # As _pending_drt_params, plus the canonical CDC the cache key needs.
        self._pending_ecm: dict = {}
        # Steps replot lazily: _refresh() does the cheap masking/validity
        # bookkeeping and marks every step dirty, but only the visible step
        # redraws. Switching steps renders whichever was left dirty.
        self._pending: Optional[dict] = None
        self._step_dirty: set = set()
        # The residual plot is the most expensive draw in the window, so it
        # stays blank until "Plot residuals" is clicked (see
        # gui.steps.validation_step.DEFAULT_RESIDUALS_LIMIT). Once armed it
        # stays armed until a different file is opened, so it survives step
        # switches and settings changes.
        self._residuals_armed = False
        # Spectrum framing handed between steps, so Nyquist panning survives
        # moving to the next step. See _on_step_changed.
        self._spectrum_view_state = None
        # Guards the settings-width mirror from re-entering itself while it
        # pushes the new width onto the other three steps.
        self._syncing_panel_width = False
        # splitterMoved fires continuously through a drag; coalesce the writes.
        self._panel_width_save_timer = QTimer(self)
        self._panel_width_save_timer.setSingleShot(True)
        self._panel_width_save_timer.setInterval(300)
        self._panel_width_save_timer.timeout.connect(self._save_settings_width)
        # Width for every step's settings column; restored from QSettings
        # once the steps exist (see _restore_settings_width).
        self._panel_width = DEFAULT_SETTINGS_WIDTH

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

        self._panel_width = self._restore_settings_width()
        self._apply_settings_width(self._panel_width)

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

        self.export_bdf_action = QAction("&Export to BDF…", self)
        self.export_bdf_action.setShortcut("Ctrl+E")
        self.export_bdf_action.setStatusTip(
            "Write every sweep and its analysis results as Battery Data Format "
            "CSV plus a JSON-LD sidecar."
        )
        self.export_bdf_action.triggered.connect(self._export_bdf)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_session_action)
        file_menu.addAction(self.save_session_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_bdf_action)

        # One QAction backs the menu item, the status-bar button, and the
        # shortcut, so their checked states stay in sync automatically.
        self.dark_action = QAction("🌙", self, checkable=True)
        self.dark_action.setShortcut("Ctrl+D")
        self.dark_action.setStatusTip("Toggle between light and dark themes (Ctrl+D)")
        self.dark_action.toggled.connect(self._on_theme_toggled)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.dark_action)

        # Shares the menu action, so it stays in sync with it and Ctrl+D.
        # addPermanentWidget docks bottom-right; addWidget for bottom-left.
        theme_button = QToolButton()
        theme_button.setDefaultAction(self.dark_action)
        theme_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        theme_button.setAutoRaise(True)
        self.statusBar().addPermanentWidget(theme_button)

    def _build_ui(self) -> None:
        """The step bar across the top, over one page per step."""
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self.step_bar = StepBar()
        self.step_bar.step_selected.connect(self._on_step_selected)
        root.addWidget(self.step_bar)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        # Colored by the app stylesheet, not inline, so it follows the theme.
        self.warning_label.setObjectName("warningBanner")
        self.warning_label.hide()
        root.addWidget(self.warning_label)

        self.data_viz_step = DataVizStep(self._selection)
        self.validation_step = ValidationStep(self._selection)
        self.drt_step = DRTStep(self._selection)
        self.ecm_step = ECMStep(self._selection)

        self.step_stack = QStackedWidget()
        # Order must match gui.stepper.STEPS -- StepBar.index_of maps between
        # them, and _render_active_step branches on the page index.
        for page in (self.data_viz_step, self.validation_step, self.drt_step, self.ecm_step):
            self.step_stack.addWidget(page)
        self.step_stack.currentChanged.connect(self._on_step_changed)

        root.addWidget(self.step_stack, stretch=1)
        self.setCentralWidget(central)

        self._wire_steps()
        self._show_empty_state()
        self.statusBar().showMessage("Open a .mpt, .txt, or .csv file to begin.")

    def _show_empty_state(self) -> None:
        """Say what to do next, in the plot areas themselves."""
        self.data_viz_step.spectrum_pane.set_message(
            "Open a .mpt, .txt, or .csv file on the left to begin."
        )
        self.validation_step.spectrum_pane.set_message("No sweep selected.")
        self.validation_step.residuals_pane.clear()
        self.drt_step.top_pane.set_message("No sweep selected.")
        self.drt_step.drt_pane.set_message(
            "Run a DRT on the left to see the distribution of relaxation times."
        )
        self.drt_step.peaks_text.clear()
        self.ecm_step.spectrum_pane.set_message("No sweep selected.")
        self.ecm_step.params_text.clear()
        self.data_viz_step.details_text.clear()

    def _wire_steps(self) -> None:
        """Connect every step's controls to the handlers here."""
        # Both run the full bookkeeping pass: the per-sweep lists _refresh()
        # derives are keyed off the displayed subset, so a cursor move has to
        # recompute them or the plots go stale.
        self._selection.selection_changed.connect(self._refresh)
        self._selection.cursor_moved.connect(self._refresh)

        viz = self.data_viz_step
        viz.open_button.clicked.connect(self._open_file)
        viz.add_files_button.clicked.connect(self._add_files_dialog)
        viz.remove_file_button.clicked.connect(self._on_remove_file_clicked)
        for step in self._steps():
            step.single_radio.toggled.connect(self._on_display_mode_changed)
            step.settings_width_changed.connect(self._on_settings_width_changed)
        viz.markers_radio.toggled.connect(self._refresh)
        viz.nyquist_view_radio.toggled.connect(self._on_visual_view_changed)

        val = self.validation_step
        val.inductive_check.toggled.connect(self._refresh)
        val.eraser_check.toggled.connect(self._on_eraser_toggled)
        val.kk_radio.toggled.connect(self._on_method_changed)
        val.threshold_spin.valueChanged.connect(self._refresh)
        val.run_validation_button.clicked.connect(self._run_validation)
        val.residuals_plot_button.clicked.connect(self._on_plot_residuals_clicked)

        drt = self.drt_step
        drt.run_simple_button.clicked.connect(self._run_drt_simple)
        drt.run_bayesian_button.clicked.connect(self._run_drt_bayesian)
        drt.run_bht_button.clicked.connect(self._run_drt_bht)
        drt.run_peak_analysis_button.clicked.connect(self._run_peak_analysis)
        drt.measured_radio.toggled.connect(self._on_drt_top_view_changed)
        drt.export_results_button.clicked.connect(self._export_drt_results)

        ecm = self.ecm_step
        ecm.preset_combo.currentIndexChanged.connect(self._on_ecm_preset_changed)
        ecm.build_from_drt_button.clicked.connect(self._build_circuit_from_drt)
        ecm.cdc_edit.textChanged.connect(self._on_ecm_cdc_changed)
        ecm.run_button.clicked.connect(self._run_ecm_fit)
        ecm.shown_combo.currentIndexChanged.connect(self._on_ecm_shown_changed)
        ecm.export_params_button.clicked.connect(self._export_ecm_parameters)

        # The eraser works on both the Data Visualisation and Validation
        # spectra; one checkbox arms them together.
        for pane in (viz.spectrum_pane, val.spectrum_pane):
            pane.point_mask_toggled.connect(self._on_point_mask_toggled)

        viz.spectrum_pane.replot_requested.connect(self._force_replot_spectrum)
        val.spectrum_pane.replot_requested.connect(self._force_replot_spectrum)
        drt.top_pane.replot_requested.connect(self._force_replot_spectrum)
        ecm.spectrum_pane.replot_requested.connect(self._force_replot_ecm)
        drt.drt_pane.replot_requested.connect(self._force_replot_drt)

        # Scoped image export: the pane reports the click, this picks a path.
        for pane in (
            viz.spectrum_pane, val.spectrum_pane, drt.top_pane,
            drt.drt_pane, ecm.spectrum_pane,
        ):
            pane.save_image_requested.connect(
                lambda p=pane: self._save_pane_image(p)
            )
        val.export_image_button.clicked.connect(
            lambda: self._save_pane_image(val.spectrum_pane)
        )
        drt.export_image_button.clicked.connect(
            lambda: self._save_pane_image(drt.drt_pane)
        )
        ecm.export_image_button.clicked.connect(
            lambda: self._save_pane_image(ecm.spectrum_pane)
        )

        self._update_validation_button_text()
        self._update_ecm_shown_combo()
        self._update_residuals_header(0, 0)
        self._sync_display_mode_widgets()

    def _on_step_selected(self, step_id: str) -> None:
        """A click on the step bar. No gating: every step is reachable at any
        time (see gui.stepper)."""
        self.step_stack.setCurrentIndex(StepBar.index_of(step_id))

    def _on_step_changed(self, index: int) -> None:
        """Carry the spectrum framing onto the step being opened, then draw it
        if stale, so panning survives stepping across."""
        self.step_bar.set_current(STEPS[index][0])
        # Hidden QStackedWidget pages are not laid out, so re-assert the width
        # here -- the visible step is then always correct.
        self._steps()[index].set_settings_width(self._panel_width)
        incoming = self._spectrum_pane_for(index)
        if incoming is not None and self._spectrum_view_state is not None:
            incoming.set_view_state(self._spectrum_view_state)
        self._render_active_step()

    def _spectrum_pane_for(self, index: int):
        """The pane showing the measured spectrum on a given step, if any. The
        DRT upper pane counts only while showing the spectrum."""
        if index == 0:
            return self.data_viz_step.spectrum_pane
        if index == 1:
            return self.validation_step.spectrum_pane
        if index == 2:
            return self.drt_step.top_pane if self.drt_step.top_view == "Measured" else None
        if index == 3:
            return self.ecm_step.spectrum_pane
        return None

    # ------------------------------------------------------------- helpers

    def _display_mode_for(self, index: int) -> str:
        """How a step draws the selection: "Single" pages one sweep at a time,
        "Combined" puts them on one figure. Per step, not global."""
        return self._steps()[index].display_mode

    @property
    def _style(self) -> str:
        return "scatter" if self.data_viz_step.markers_radio.isChecked() else "line"

    @property
    def _visual_view(self) -> str:
        return "Nyquist" if self.data_viz_step.nyquist_view_radio.isChecked() else "Bode"

    @property
    def _validation_method(self) -> str:
        return VALIDATION_METHODS[0] if self.validation_step.kk_radio.isChecked() else VALIDATION_METHODS[1]

    @property
    def _multi_file(self) -> bool:
        """More than one file loaded; when true, labels are qualified with
        their source file (ds.qualified_label)."""
        return len(self._files) > 1

    def _display_label(self, ds_or_key) -> str:
        """A sweep's display name, qualified with its file when several are
        loaded. Accepts an EISDataset or a ds.key."""
        ds = ds_or_key
        if isinstance(ds_or_key, str):
            ds = next((d for d in self._datasets if d.key == ds_or_key), None)
            if ds is None:
                return ds_or_key
        return ds.qualified_label if self._multi_file else ds.label

    def _build_style_map(self) -> Dict[str, Tuple[str, str]]:
        """ds.key -> (color, pg symbol) for every loaded sweep: color by sweep
        index, symbol by the file's position in _files."""
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
        """Every checked sweep -- the working set that gets masked, processed
        by the Run/Fit buttons, and cached, regardless of how many are drawn."""
        return self._selection.selected()

    def _displayed_datasets(self, index: int) -> List:
        """The subset a step draws: the whole working set in Combined mode, or
        the sweep under the pager cursor in Single mode."""
        if self._display_mode_for(index) == "Combined":
            return self._selection.selected()
        ds = self._selection.current()
        return [ds] if ds is not None else []

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
            self.validation_step.run_validation_button.setText("Running…")
        else:
            self.validation_step.run_validation_button.setText("Run validation")

    def _refresh_file_list_widget(self) -> None:
        self.data_viz_step.file_list.clear()
        for lf in self._files:
            item = QListWidgetItem(f"{lf.stem}  ({lf.n_sweeps} sweep(s))")
            item.setData(Qt.UserRole, lf.file_id)
            item.setToolTip(f"{lf.path}\nParsed with: {lf.parser_used}")
            self.data_viz_step.file_list.addItem(item)

    def _populate_sweep_selectors(self, checked_keys=None) -> None:
        """Point the selection model and its views at the current _files/
        _datasets."""
        self._selection.set_datasets(self._datasets, checked_keys)
        for view in self._selection_views():
            view.set_files(self._files)

    def _selection_views(self) -> List:
        """Every widget showing the selection; this exists only to hand them
        the LoadedFile registry, which datasets do not carry."""
        return [
            self.data_viz_step.files_panel,
            self.validation_step.pager,
            self.drt_step.pager,
            self.ecm_step.pager,
        ]

    # ------------------------------------------------------- panel width

    def _restore_settings_width(self) -> int:
        """The saved settings-column width, clamped to the allowed range."""
        width = self._settings.value(
            "settings_panel_width", DEFAULT_SETTINGS_WIDTH, type=int
        )
        return max(MIN_SETTINGS_WIDTH, min(MAX_SETTINGS_WIDTH, width))

    def _apply_settings_width(self, width: int) -> None:
        """Put every step's settings column at the same width, silently."""
        self._panel_width = width
        self._syncing_panel_width = True
        try:
            for step in self._steps():
                step.set_settings_width(width)
        finally:
            self._syncing_panel_width = False

    def _on_settings_width_changed(self, width: int) -> None:
        """Mirror a drag onto the other steps, then persist it, so the panel
        edge does not jump when moving between steps."""
        if self._syncing_panel_width:
            return
        self._apply_settings_width(width)
        # Debounced: splitterMoved fires continuously through a drag and each
        # write is a registry hit.
        self._panel_width_save_timer.start()

    def _save_settings_width(self) -> None:
        self._settings.setValue("settings_panel_width", self._panel_width)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Lock the settings panels while a worker runs. Plots and the step bar
        stay live, so a long batch can still be watched and panned."""
        for step in self._steps():
            step.settings_scroll.setEnabled(enabled)

    def _steps(self) -> List:
        return [self.data_viz_step, self.validation_step, self.drt_step, self.ecm_step]

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
        self._ecm_results = {}
        self._ecm_params = {}
        self._ecm_shown_cdc = None
        self._manual_masked = {}
        self._manual_kept = {}
        # Nothing validated yet, so the residual plot starts blank.
        self._residuals_armed = False

    def _load_files(self, paths: List[str], clear_first: bool) -> None:
        """Parse each path and commit to _files/_datasets, replacing if
        clear_first else appending. A wholly failed batch changes nothing."""
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
        self.statusBar().showMessage(
            f"Loaded {len(self._datasets)} sweep(s) from {len(self._files)} file(s) total."
        )
        self._refresh()

    def _open_generic_file(
        self, path: str, prior_error: EISParseError, file_id: int, generic_cache: dict
    ):
        """Fall back to the generic txt/csv parser, confirming columns in a
        dialog. Returns (datasets, parser_name), or None if cancelled."""
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

        # Broad on purpose: a bad column mapping must surface as a dialog.
        # Escaping this slot would abort the process and lose loaded files.
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
        item = self.data_viz_step.file_list.currentItem()
        if item is None:
            return
        self._remove_file(item.data(Qt.UserRole))

    def _remove_file(self, file_id: int) -> None:
        """Drop one loaded file and every cache entry keyed to one of its
        sweeps, identified via ds.key."""
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
        # Keyed by (cdc, ds.key), so the sweep is the second half.
        self._ecm_results = {
            k: v for k, v in self._ecm_results.items() if k[1] not in removed_keys
        }
        self._ecm_params = {
            k: v for k, v in self._ecm_params.items() if k[1] not in removed_keys
        }
        self._manual_masked = {
            k: v for k, v in self._manual_masked.items() if k not in removed_keys
        }
        self._manual_kept = {
            k: v for k, v in self._manual_kept.items() if k not in removed_keys
        }

        self._refresh_file_list_widget()
        self._populate_sweep_selectors()

        if not self._datasets:
            self.statusBar().showMessage("Open a .mpt, .txt, or .csv file to begin.")
            self.ecm_step.circuit_pane.clear()
            self._show_empty_state()
            # This branch returns before _refresh(), which normally
            # repopulates the circuit picker; do it here so the picker stops
            # offering circuits from the removed file.
            self._update_ecm_shown_combo()
            self._pending = None
            self._step_dirty.clear()
            return

        self.statusBar().showMessage(
            f"Removed file. {len(self._datasets)} sweep(s) from {len(self._files)} file(s) remain."
        )
        self._refresh()

    @staticmethod
    def _files_from_datasets(datasets: List) -> List[LoadedFile]:
        """Rebuild the file registry after a session restore from the datasets'
        own file_id/source_file; parser_used and path get placeholders."""
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
            self.validation_step.inductive_check.isChecked(),
            self.validation_step.threshold_spin.value(),
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
                self._ecm_results,
                self._ecm_params,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save session", f"Could not save session:\n{exc}")
            return

        self.statusBar().showMessage(f"Session saved to '{Path(path).name}'.")

    def _export_bdf(self) -> None:
        """Write every loaded sweep, and whatever analyses have been run, as
        Battery Data Format files."""
        if not self._datasets:
            QMessageBox.information(self, "Export to BDF", "Open a file first.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Export to BDF")
        if not directory:
            return

        from core.bdf_export import export_batch

        try:
            written = export_batch(
                directory,
                self._datasets,
                files=self._files,
                validation_results=self._validation_results,
                validation_params=self._validation_params,
                drt_results=self._drt_results,
                drt_params=self._drt_params,
                drt_peaks=self._drt_peaks,
                ecm_results=self._ecm_results,
                ecm_params=self._ecm_params,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export to BDF", f"Could not export:\n{exc}")
            return

        self.statusBar().showMessage(
            f"Exported {len(written)} file(s) for {len(self._datasets)} sweep(s) "
            f"to '{Path(directory).name}'."
        )

    # --------------------------------------------------------- step exports
    #
    # Scoped to the sweep on screen, unlike _export_bdf and _save_session
    # above, which cover everything loaded.

    def _cursor_dataset(self, title: str):
        """The sweep the pagers are on, or None (with a message) when there is
        none. Scoped exports need one specific sweep, not the selection."""
        ds = self._selection.current()
        if ds is None:
            QMessageBox.information(self, title, "Select a sweep first.")
        return ds

    def _save_pane_image(self, pane) -> None:
        title = "Save plot as image"
        path, _ = QFileDialog.getSaveFileName(
            self, title, "", "PNG image (*.png);;SVG image (*.svg)"
        )
        if not path:
            return
        try:
            pane.save_image(path)
        except Exception as exc:
            QMessageBox.critical(self, title, f"Could not save the image:\n{exc}")
            return
        self.statusBar().showMessage(f"Saved plot to '{Path(path).name}'.")

    def _export_drt_results(self) -> None:
        """The DRT curve for the sweep on screen, plus its peaks if they have
        been fitted."""
        title = "Export DRT results"
        ds = self._cursor_dataset(title)
        if ds is None:
            return
        result = self._drt_results.get(ds.key)
        if result is None:
            QMessageBox.information(
                self, title, "Run a DRT on this sweep first."
            )
            return

        directory = QFileDialog.getExistingDirectory(self, title)
        if not directory:
            return

        from core.bdf_export import file_stem, write_drt, write_drt_peaks

        stem = Path(directory) / file_stem(ds)
        try:
            written = [write_drt(stem.with_suffix(".drt.csv"), result)]
            peaks = self._drt_peaks.get(ds.key)
            if peaks is not None:
                written.append(write_drt_peaks(stem.with_suffix(".drt_peaks.csv"), peaks))
        except Exception as exc:
            QMessageBox.critical(self, title, f"Could not export:\n{exc}")
            return

        self.statusBar().showMessage(
            f"Exported {len(written)} file(s) for {self._display_label(ds)}."
        )

    def _export_ecm_parameters(self) -> None:
        """Every circuit fitted to the sweep on screen, with its fit curve."""
        title = "Export ECM parameters"
        ds = self._cursor_dataset(title)
        if ds is None:
            return
        fits = {
            cdc: result
            for (cdc, key), result in self._ecm_results.items()
            if key == ds.key
        }
        if not fits:
            QMessageBox.information(self, title, "Fit a circuit to this sweep first.")
            return

        directory = QFileDialog.getExistingDirectory(self, title)
        if not directory:
            return

        from core.bdf_export import file_stem, write_ecm_fit_curve, write_ecm_parameters

        stem = Path(directory) / file_stem(ds)
        try:
            written = [write_ecm_parameters(stem.with_suffix(".ecm.csv"), fits)]
            # One curve per circuit, so candidates can be compared.
            for cdc, result in fits.items():
                safe = "".join(c if c.isalnum() else "_" for c in cdc)
                written.append(
                    write_ecm_fit_curve(stem.with_suffix(f".ecm_fit.{safe}.csv"), result)
                )
        except Exception as exc:
            QMessageBox.critical(self, title, f"Could not export:\n{exc}")
            return

        self.statusBar().showMessage(
            f"Exported {len(written)} file(s) for {self._display_label(ds)}."
        )

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
                ecm_results,
                ecm_params,
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
        self._ecm_results = ecm_results
        self._ecm_params = ecm_params
        # Not saved in a session; _update_ecm_shown_combo settles on the first
        # fitted circuit, as with the active step and the Nyquist/Bode choice.
        self._ecm_shown_cdc = None
        self._manual_masked = ui_state["manual_masked"]
        self._manual_kept = ui_state["manual_kept"]
        # Nothing plotted yet, so the residual plot starts blank.
        self._residuals_armed = False

        self._refresh_file_list_widget()
        self._populate_sweep_selectors()

        # Restore the filter widgets with signals blocked; the single
        # _refresh() below covers them all.
        method = ui_state.get("validation_method")
        self.validation_step.kk_radio.blockSignals(True)
        self.validation_step.zhit_radio.blockSignals(True)
        if method == VALIDATION_METHODS[1]:
            self.validation_step.zhit_radio.setChecked(True)
        else:
            self.validation_step.kk_radio.setChecked(True)
        self.validation_step.kk_radio.blockSignals(False)
        self.validation_step.zhit_radio.blockSignals(False)
        self._update_validation_button_text()

        self.validation_step.inductive_check.blockSignals(True)
        self.validation_step.inductive_check.setChecked(bool(ui_state.get("inductive_filter", False)))
        self.validation_step.inductive_check.blockSignals(False)

        threshold = ui_state.get("residual_threshold")
        if threshold is not None:
            self.validation_step.threshold_spin.blockSignals(True)
            self.validation_step.threshold_spin.setValue(threshold)
            self.validation_step.threshold_spin.blockSignals(False)

        self.statusBar().showMessage(
            f"Restored session from '{Path(path).name}' "
            f"({len(datasets)} sweep(s))."
        )
        self._refresh()

    def _on_display_mode_changed(self) -> None:
        """Each step has its own toggle, so this just re-reads all of them."""
        self._sync_display_mode_widgets()
        self._refresh()

    def _sync_display_mode_widgets(self) -> None:
        for index, step in enumerate(self._steps()):
            single = self._display_mode_for(index) == "Single"
            # Paging only means anything when one sweep is drawn at a time.
            pager = getattr(step, "pager", None)
            if pager is not None:
                pager.setVisible(single)
        # Residuals are Single-mode only: one figure per selected sweep would
        # crowd out the combined spectra comparison.
        self.validation_step.set_residuals_visible(
            self._display_mode_for(1) == "Single"
        )

    def _on_method_changed(self, checked: bool) -> None:
        if not checked:
            return  # fires once per radio; only act on the newly-checked one
        self._update_validation_button_text()
        self._refresh()

    def _on_eraser_toggled(self, checked: bool) -> None:
        """A mode switch only -- no mask changes, so no _refresh(). Arms the
        Data Visualisation and Validation spectrum panes together."""
        for step in (self.data_viz_step, self.validation_step):
            step.spectrum_pane.set_eraser_enabled(checked)
        if checked:
            self.statusBar().showMessage(
                "Eraser on — click a point on the Data Visualisation or "
                "Validation plot to remove it, or a grey × to restore it."
            )
        else:
            self.statusBar().showMessage("Eraser off.")

    def _on_point_mask_toggled(self, key: str, index: int) -> None:
        """Flip one point's manual override; _refresh() recomposes the mask and
        redraws. Direction is read from the sweep's current mask."""
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
        # Regenerate existing figures so plot colors follow the new theme.
        if self._datasets:
            self._refresh()

    def _run_validation(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return
        # Module-level functions, so ValidationWorker can pickle the runner by
        # reference when it spreads a batch over processes.
        from core.validation import run_kk_test, run_zhit

        method = self._validation_method
        runner = run_kk_test if method == VALIDATION_METHODS[0] else run_zhit

        # Masks must stay stable while the worker reads them, so the settings
        # panels are locked for the run.
        self._worker_errors = []
        self._worker = ValidationWorker(method, runner, selected, parent=self)
        self._worker.result_ready.connect(self._on_validation_result)
        self._worker.error.connect(self._on_validation_error)
        self._worker.progress.connect(self._on_validation_progress)
        self._worker.finished.connect(self._on_validation_finished)
        self._running_method = method
        self._set_controls_enabled(False)
        self._update_validation_button_text()
        self.statusBar().showMessage(f"Running {method} analysis… 0 of {len(selected)}")
        self._worker.start()

    def _on_validation_result(self, method: str, key: str, result) -> None:
        self._validation_results[(method, key)] = result
        # Z-HIT takes no extra kwargs; these mirror the non-default arguments
        # in core.validation.run_kk_test's signature.
        if method == VALIDATION_METHODS[0]:
            self._validation_params[(method, key)] = {
                "admittance": False,
                "num_F_ext_evaluations": 10,
            }
        else:
            self._validation_params[(method, key)] = {}

    def _on_validation_progress(self, done: int, total: int) -> None:
        """A count, not a name: sweeps run in a process pool and finish out of
        order."""
        self.statusBar().showMessage(
            f"Running {self._running_method} analysis… {done} of {total}"
        )

    def _on_validation_error(self, key: str, message: str) -> None:
        self._worker_errors.append((key, message))

    def _on_validation_finished(self) -> None:
        self._worker = None
        self._set_controls_enabled(True)
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
            rbf_type=self.drt_step.rbf_combo.currentData(),
            derivative_order=self.drt_step.derivative_combo.currentData(),
            rbf_shape=self.drt_step.shape_control_combo.currentData(),
            shape_coeff=self.drt_step.shape_coeff_spin.value(),
        )

    def _update_optimal_lambda_label(self, selected: List) -> None:
        lambda_value = None
        if len(selected) == 1:
            result = self._drt_results.get(selected[0].key)
            lambda_value = getattr(result, "lambda_value", None)
        self.drt_step.optimal_lambda_label.setText(
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
            mode=self.drt_step.mode_combo.currentData(),
            inductance=self.drt_step.inductance_check.isChecked(),
            cross_validation=self.drt_step.cv_combo.currentData(),
            lambda_value=self.drt_step.lambda_spin.value(),
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
            mode=self.drt_step.mode_combo.currentData(),
            inductance=self.drt_step.inductance_check.isChecked(),
            cross_validation=self.drt_step.cv_combo.currentData(),
            lambda_value=self.drt_step.lambda_spin.value(),
            credible_intervals=True,
            num_samples=self.drt_step.num_samples_spin.value(),
            timeout=self.drt_step.timeout_spin.value(),
        )

        def runner(ds):
            return run_drt(ds, **params)

        self._pending_drt_params = params
        self._drt_worker_errors = []
        self._drt_worker = DRTWorker(runner, selected, parent=self)
        self._drt_worker.result_ready.connect(self._on_drt_worker_result)
        self._drt_worker.error.connect(self._on_drt_worker_error)
        self._drt_worker.finished.connect(self._on_drt_worker_finished)
        self._set_controls_enabled(False)
        self.statusBar().showMessage("Running Bayesian DRT… this may take a while.")
        self._drt_worker.start()

    def _on_drt_worker_result(self, key: str, result) -> None:
        self._drt_results[key] = result
        self._drt_params[key] = self._pending_drt_params

    def _on_drt_worker_error(self, key: str, message: str) -> None:
        self._drt_worker_errors.append((key, message))

    def _on_drt_worker_finished(self) -> None:
        self._drt_worker = None
        self._set_controls_enabled(True)
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
        params = dict(settings, num_samples=self.drt_step.num_samples_spin.value())
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

        num_peaks = self.drt_step.num_peaks_spin.value()
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

    # ----------------------------------------------------------------- ECM

    def _on_ecm_preset_changed(self, _index: int) -> None:
        """Preset -> CDC box. One-way: the text box is the source of truth and
        the combo only fills it, so later edits leave the combo as-is."""
        self.ecm_step.cdc_edit.setText(self.ecm_step.preset_combo.currentData())

    def _build_circuit_from_drt(self) -> None:
        """Fill the CDC box from a DRT peak analysis and stop there, using the
        first selected sweep with peaks."""
        from core.ecm import circuit_from_drt_peaks, series_resistance

        selected = self._selected_datasets()
        source = next((ds for ds in selected if ds.key in self._drt_peaks), None)
        if source is None:
            QMessageBox.information(
                self,
                "Build circuit from DRT",
                "No DRT peak analysis for the selected sweep(s).\n\n"
                "Run a DRT calculation (7) and then 'Peak deconvolution' (8) "
                "first — the peaks are what decide how many RC elements the "
                "circuit needs.",
            )
            return

        try:
            cdc = circuit_from_drt_peaks(
                self._drt_peaks[source.key], series_resistance(source)
            )
        except Exception as exc:
            QMessageBox.warning(self, "Build circuit from DRT", str(exc))
            return

        self.ecm_step.cdc_edit.setText(cdc)
        num_pairs = cdc.count("(")
        self.statusBar().showMessage(
            f"Circuit built from {self._display_label(source)}: "
            f"{num_pairs} R-CPE pair(s) + series R."
        )

    def _on_ecm_cdc_changed(self, text: str) -> None:
        """Live-validate the circuit code, so a typo is caught before a batch
        is spent on it rather than as one identical error per sweep."""
        from core.ecm import validate_cdc

        ok, message = validate_cdc(text)
        self.ecm_step.cdc_status_label.setText(message)
        style.set_state(self.ecm_step.cdc_status_label, "ok" if ok else "error")
        self.ecm_step.run_button.setEnabled(ok)

        # The ECM step previews this code as a schematic while nothing is
        # fitted, so redraw immediately for the preview. Once there are fits,
        # only mark dirty: rendering also rebuilds the text report, which would
        # make typing lag.
        self._step_dirty.add(3)
        fitted_keys = {key for _, key in self._ecm_results}
        previewing = self._pending is None or not any(
            ds.key in fitted_keys for ds in self._pending["selected"]
        )
        if previewing:
            self._render_active_step()

    def _on_ecm_shown_changed(self, _index: int) -> None:
        """Switch which fitted circuit the ECM step draws. Needs a full
        _refresh(), since _pending's fit curves are per-circuit."""
        data = self.ecm_step.shown_combo.currentData()
        if data is None or data == self._ecm_shown_cdc:
            return
        self._ecm_shown_cdc = data
        self._refresh()

    def _update_ecm_shown_combo(self) -> None:
        """Repopulate the circuit picker from what has been fitted, and report
        how much of the selection it covers. Signals stay blocked."""
        fitted = sorted({cdc for cdc, _ in self._ecm_results})
        if self._ecm_shown_cdc not in fitted:
            self._ecm_shown_cdc = fitted[0] if fitted else None

        self.ecm_step.shown_combo.blockSignals(True)
        self.ecm_step.shown_combo.clear()
        add_combo_items(self.ecm_step.shown_combo, [(cdc, cdc) for cdc in fitted])
        if self._ecm_shown_cdc is not None:
            self.ecm_step.shown_combo.setCurrentIndex(fitted.index(self._ecm_shown_cdc))
        self.ecm_step.shown_combo.blockSignals(False)

        has_fits = bool(fitted)
        self.ecm_step.shown_label.setVisible(has_fits)
        self.ecm_step.shown_combo.setVisible(has_fits)
        self.ecm_step.shown_status_label.setVisible(has_fits)
        if has_fits:
            selected = self._selected_datasets()
            covered = sum(
                1 for ds in selected if (self._ecm_shown_cdc, ds.key) in self._ecm_results
            )
            self.ecm_step.shown_status_label.setText(
                f"{covered} of {len(selected)} selected sweep(s) fitted"
            )

    def _ecm_settings(self) -> dict:
        """Fit settings read from the '9. ECM fitting' panel, minus the circuit
        itself, which is tracked separately."""
        return dict(
            method=self.ecm_step.method_combo.currentData(),
            weight=self.ecm_step.weight_combo.currentData(),
        )

    def _run_ecm_fit(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        from core.ecm import canonical_cdc, run_ecm_fit, run_ecm_fit_seeded

        cdc = self.ecm_step.cdc_edit.text().strip()
        try:
            canonical = canonical_cdc(cdc)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid circuit", str(exc))
            return

        settings = self._ecm_settings()
        seed = self.ecm_step.seed_check.isChecked()

        if seed:
            # A one-element list so the closure can rebind it: each fit seeds
            # the next, which is why ECMWorker runs the batch serially.
            previous = [None]

            def runner(ds):
                result = run_ecm_fit_seeded(ds, cdc, previous[0], **settings)
                previous[0] = result
                return result
        else:
            def runner(ds):
                return run_ecm_fit(ds, cdc, **settings)

        self._pending_ecm = dict(cdc=canonical, params=dict(settings, cdc=cdc, seeded=seed))
        self._ecm_worker_errors = []
        self._ecm_worker = ECMWorker(runner, selected, parent=self)
        self._ecm_worker.result_ready.connect(self._on_ecm_result)
        self._ecm_worker.error.connect(self._on_ecm_error)
        self._ecm_worker.progress.connect(self._on_ecm_progress)
        self._ecm_worker.finished.connect(self._on_ecm_finished)
        # Masks must stay stable while the worker reads them.
        self._set_controls_enabled(False)
        self.statusBar().showMessage(
            f"Fitting {canonical}… 0 of {len(selected)}"
        )
        self._ecm_worker.start()

    def _on_ecm_result(self, key: str, result) -> None:
        cdc = self._pending_ecm["cdc"]
        self._ecm_results[(cdc, key)] = result
        self._ecm_params[(cdc, key)] = self._pending_ecm["params"]

    def _on_ecm_error(self, key: str, message: str) -> None:
        self._ecm_worker_errors.append((key, message))

    def _on_ecm_progress(self, completed: int, total: int) -> None:
        self.statusBar().showMessage(
            f"Fitting {self._pending_ecm['cdc']}… {completed} of {total}"
        )

    def _on_ecm_finished(self) -> None:
        self._ecm_worker = None
        self._set_controls_enabled(True)
        # Show what was just fitted.
        self._ecm_shown_cdc = self._pending_ecm["cdc"]
        failed = len(self._ecm_worker_errors)
        self.statusBar().showMessage(
            f"Circuit fitting finished ({failed} failed)." if failed
            else "Circuit fitting finished."
        )
        if self._ecm_worker_errors:
            details = "\n".join(
                f"- {self._display_label(key)}: {msg}"
                for key, msg in self._ecm_worker_errors
            )
            QMessageBox.warning(self, "ECM errors", f"Some sweeps failed:\n{details}")
        self._refresh()

    def _render_ecm_circuits(self, fits: List[Tuple[str, object]]) -> None:
        """Draw the schematics in the ECM step's lower pane: one per (sweep,
        fitted circuit) in `fits`, annotated with that fit's values."""
        from core.circuit_diagram import build_fit_diagram, build_preview_diagram

        colors = diagram_colors(self._theme_mode)
        if not fits:
            cdc = self.ecm_step.cdc_edit.text().strip()
            try:
                svg = build_preview_diagram(
                    cdc, foreground=colors["wire"], accent=colors["muted"]
                )
            except Exception:
                # A half-typed code is normal, not an error worth a dialog --
                # the panel's status label already reports the problem.
                self.ecm_step.circuit_pane.set_message(
                    "Enter a valid circuit description code on the left to "
                    "preview the circuit, then click 'Fit circuit'."
                )
                return
            self.ecm_step.circuit_pane.set_diagrams(
                [(f"{cdc} — not fitted yet (initial values)", svg)]
            )
            return

        # Capped, not paged: every fit is listed in full in the report below.
        shown = fits[:MAX_CIRCUIT_DIAGRAMS]
        self.ecm_step.circuit_pane.set_diagrams(
            [
                (
                    f"{label} — {result.circuit.to_string()}"
                    f"   χ² = {result.pseudo_chisqr:.4g}",
                    build_fit_diagram(
                        result, foreground=colors["wire"], accent=colors["value"]
                    ),
                )
                for label, result in shown
            ]
        )
        if len(fits) > len(shown):
            self.ecm_step.circuit_pane.add_note(
                f"… and {len(fits) - len(shown)} more fit(s), listed in the "
                f"report below. Narrow the sweep selection to draw them."
            )

    def _force_replot_ecm(self) -> None:
        """Rebuild the ECM step from current state, bypassing the dirty check
        -- the plot-area 'Replot' button, mirroring _force_replot_spectrum."""
        if self._pending is None:
            return
        self._step_dirty.add(3)
        self._render_active_step()

    # ------------------------------------------------------------- refresh

    def _refresh(self) -> None:
        """Cheap, always-run bookkeeping: apply masks and validate against the
        current selection. Figures rebuild lazily in _render_active_step()."""
        if not self._datasets:
            return

        # Only reachable once a file has been loaded, so these are warm.
        from core.filtering import clear_mask, mask_inductive_points
        from core.validation import mask_residual_outliers

        selected = self._selected_datasets()
        if not selected:
            self.warning_label.setText("Select at least one sweep to plot.")
            self.warning_label.show()
            self.ecm_step.circuit_pane.clear()
            self._show_empty_state()
            self.data_viz_step.spectrum_pane.set_message(
                "No sweep selected — tick one in Files and sets below."
            )
            # This branch returns before the repopulation below, so refresh
            # the picker's "N of M fitted" count here.
            self._update_ecm_shown_combo()
            self._pending = None
            self._step_dirty.clear()
            # _residuals_armed survives: an empty selection is a transient
            # state on the way to a new one, not a request to blank the plot.
            self._update_residuals_header(0, 0)
            return

        method = self._validation_method
        threshold = self.validation_step.threshold_spin.value()

        for ds in selected:
            if self.validation_step.inductive_check.isChecked():
                mask_inductive_points(ds)
            else:
                clear_mask(ds)
            # Before the outlier pass as well as after, so the mask matches
            # what a validation run observed and does not read as stale.
            self._apply_manual_overrides(ds)

        stale_keys = []
        for ds in selected:
            result = self._validation_results.get((method, ds.key))
            if result is not None:
                try:
                    mask_residual_outliers(ds, result, threshold)
                except ValueError:
                    stale_keys.append(ds.key)
            # Re-applied on top of the outlier pass, which only adds masks, so
            # a manually restored point survives a threshold that would drop it.
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

        # Masking above covers the whole working set. From here on, figures
        # concern only the on-screen subset, which differs per step.
        displayed_for = {i: self._displayed_datasets(i) for i in range(len(STEPS))}

        # Each list below is read by exactly one step, built from that step's
        # own displayed subset.
        validated_selected = [
            ds
            for ds in displayed_for[1]
            if (method, ds.key) in self._validation_results and ds.key not in stale_keys
        ]
        drt_selected = [
            (self._display_label(ds), self._drt_results[ds.key])
            for ds in displayed_for[2]
            if ds.key in self._drt_results
        ]
        # Must run before _ecm_shown_cdc is read below: it settles which
        # circuit is displayed if the previous choice was removed.
        self._update_ecm_shown_combo()
        shown_cdc = self._ecm_shown_cdc
        ecm_selected = [
            ds for ds in displayed_for[3] if (shown_cdc, ds.key) in self._ecm_results
        ]
        # (frequencies, impedances) per sweep -- a fit is interpolated onto its
        # own denser grid, so plot builders need both.
        ecm_curves = {
            ds.key: (
                self._ecm_results[(shown_cdc, ds.key)].get_frequencies(num_per_decade=100),
                self._ecm_results[(shown_cdc, ds.key)].get_impedances(num_per_decade=100),
            )
            for ds in ecm_selected
        }
        drt_peaks_selected = [
            (self._display_label(ds), self._drt_peaks[ds.key])
            for ds in displayed_for[2]
            if ds.key in self._drt_peaks
        ]

        self._pending = dict(
            selected=selected,
            displayed_for=displayed_for,
            method=method,
            threshold=threshold,
            validated_selected=validated_selected,
            drt_selected=drt_selected,
            drt_peaks_selected=drt_peaks_selected,
            ecm_curves=ecm_curves,
            ecm_shown_cdc=shown_cdc,
        )
        self._step_dirty = {0, 1, 2, 3}
        # _residuals_armed survives: the redraw waits until the Validation step
        # is visible, so only a settings change made while looking at the
        # residuals redraws them immediately.
        self._render_active_step()

    def _on_plot_residuals_clicked(self) -> None:
        """Arm the residual plot and draw it. Stays armed for this file, so the
        plot persists across step switches and settings changes."""
        self._residuals_armed = True
        self._step_dirty.add(1)
        self._render_active_step()

    def _update_residuals_header(self, shown: int, total: int) -> None:
        """Report how much of the selection made it onto the screen."""
        val = self.validation_step
        # The cap applies to Combined view only; Single draws itself.
        combined = self._display_mode_for(1) == "Combined"
        val.residuals_limit_spin.setEnabled(combined)
        val.residuals_plot_button.setEnabled(combined and total > 0)

        if total == 0:
            text = "No validated sweeps — run a validation first."
        elif not combined:
            text = "Showing the sweep on screen; page with ‹ › above."
        elif shown == 0:
            text = f"{total} validated sweep(s) ready."
        elif shown < total:
            text = f"Showing {shown} of {total}."
        else:
            text = f"Showing all {total}."
        val.residuals_status_label.setText(text)

    def _force_replot_spectrum(self) -> None:
        """Rebuild the step being looked at, bypassing the dirty check — the
        plot-area 'Replot' button, which every spectrum pane has."""
        if self._pending is None:
            return
        self._step_dirty.add(self.step_stack.currentIndex())
        self._render_active_step()

    def _on_visual_view_changed(self, _checked: bool) -> None:
        """Redraw in the newly picked view. All steps go stale but only the
        visible one rebuilds; this skips _refresh()."""
        if self._pending is None:
            return
        self._step_dirty |= {0, 1, 2, 3}
        self._render_active_step()

    def _draw_spectrum(self, pane, drawn: List, show_removed: bool) -> None:
        """Draw the measured spectrum into one of the steps' panes, in the view
        the Data Visualisation step is set to."""
        from core.plotting import build_bode_plot, build_nyquist_plot

        if not drawn:
            pane.clear()
            return

        if len(drawn) == 1:
            title = drawn[0].full_label
        else:
            files_in_selection = {ds.file_id for ds in drawn}
            title = (
                f"{len(files_in_selection)} files · {len(drawn)} sweeps"
                if len(files_in_selection) > 1
                else drawn[0].source_file
            )
        build = build_nyquist_plot if self._visual_view == "Nyquist" else build_bode_plot
        pane.set_widget(
            build(
                drawn,
                title=title,
                style=self._style,
                show_removed=show_removed,
                style_map=self._build_style_map(),
            )
        )

    def _render_peak_report(self, drawn: List) -> None:
        """The per-peak table for whatever is on screen, shown beside the DRT
        controls rather than in a pane of its own."""
        peak_lines = []
        for ds in drawn:
            peaks = self._drt_peaks.get(ds.key)
            if peaks is None:
                continue
            peak_lines.append(
                f"=== {self._display_label(ds)} ({peaks.get_num_peaks()} peak(s)) ==="
            )
            peak_lines.append(peaks.to_peaks_dataframe().to_string(index=False))
            peak_lines.append("")
        self.drt_step.peaks_text.setPlainText("\n".join(peak_lines))

    def _render_sweep_details(self, selected: List) -> None:
        """An inventory of the working set -- everything selected, not just
        what is drawn, so it reports what an analysis would cover."""
        lines = []
        for ds in selected:
            validated_with = [
                m for m in VALIDATION_METHODS if (m, ds.key) in self._validation_results
            ]
            note = f" (validated: {', '.join(validated_with)})" if validated_with else ""
            fitted_with = sorted(
                f"{cdc} χ²={result.pseudo_chisqr:.3g}"
                for (cdc, key), result in self._ecm_results.items()
                if key == ds.key
            )
            if fitted_with:
                note += f" (fitted: {'; '.join(fitted_with)})"
            lines.append(f"{self._display_label(ds)} — {ds.num_points} points{note}")
        self.data_viz_step.details_text.setPlainText("\n".join(lines))

    def _on_drt_top_view_changed(self, _checked: bool) -> None:
        """Swap the DRT step's upper pane between the measured spectrum and the
        deconvolved peaks; skips _refresh()."""
        if self._pending is None:
            return
        self._step_dirty.add(2)
        self._render_active_step()

    def _force_replot_drt(self) -> None:
        """Rebuild the DRT step, bypassing the dirty check -- the plot-area
        'Replot' button."""
        if self._pending is None:
            return
        self._step_dirty.add(2)
        self._render_active_step()

    def _render_active_step(self) -> None:
        """Build the figures/text for the visible step, if it is still dirty.
        Other steps stay dirty and are built when moved to."""
        if self._pending is None:
            return
        index = self.step_stack.currentIndex()
        if index not in self._step_dirty:
            return

        from core.plotting import (
            build_bode_plot,
            build_drt_peaks_plot,
            build_drt_plot,
            build_nyquist_plot,
            build_residuals_plot,
        )

        p = self._pending
        drawn = p["displayed_for"][index]

        if index == 0:
            self._draw_spectrum(self.data_viz_step.spectrum_pane, drawn, show_removed=True)
            self._render_sweep_details(p["selected"])

        elif index == 1:
            self._draw_spectrum(self.validation_step.spectrum_pane, drawn, show_removed=True)
            # Single mode draws its one residual figure straight away. The
            # button and cap are for Combined view, which costs one figure per
            # selected sweep.
            validated = p["validated_selected"]
            if self._display_mode_for(1) == "Single":
                shown = validated[:1]
            else:
                limit = self.validation_step.residuals_limit_spin.value()
                shown = validated[:limit] if self._residuals_armed else []
            residual_widgets = []
            for ds in shown:
                result = self._validation_results[(p["method"], ds.key)]
                residual_widgets.append(
                    build_residuals_plot(
                        result,
                        title=f"{p['method']} residuals — {self._display_label(ds)}",
                        threshold=p["threshold"],
                    )
                )
            self.validation_step.residuals_pane.set_widgets(residual_widgets)
            self._update_residuals_header(len(shown), len(validated))

        elif index == 2:
            if self.drt_step.top_view == "Peaks":
                if p["drt_peaks_selected"]:
                    self.drt_step.top_pane.set_widget(
                        build_drt_peaks_plot(p["drt_peaks_selected"])
                    )
                else:
                    self.drt_step.top_pane.clear()
            else:
                self._draw_spectrum(self.drt_step.top_pane, drawn, show_removed=True)

            if p["drt_selected"]:
                self.drt_step.drt_pane.set_widget(build_drt_plot(p["drt_selected"]))
            else:
                self.drt_step.drt_pane.clear()

            self._render_peak_report(drawn)

        elif index == 3:
            if p["ecm_curves"]:
                build = (
                    build_nyquist_plot if self._visual_view == "Nyquist" else build_bode_plot
                )
                self.ecm_step.spectrum_pane.set_widget(
                    build(
                        drawn,
                        title=f"Circuit fit — {p['ecm_shown_cdc']}",
                        style=self._style,
                        show_removed=False,
                        style_map=self._build_style_map(),
                        fit_curves=p["ecm_curves"],
                    )
                )
            else:
                # No fit for what is on screen: show the measured spectrum, so
                # the step still says which sweep "Fit circuit" would act on.
                self._draw_spectrum(self.ecm_step.spectrum_pane, drawn, show_removed=False)

            # Every circuit fitted to each sweep, best chi-squared first.
            from core.ecm import format_fit_report

            report_lines = []
            fits_by_sweep = []
            for ds in drawn:
                fits = sorted(
                    (
                        (result.pseudo_chisqr, cdc, result)
                        for (cdc, key), result in self._ecm_results.items()
                        if key == ds.key
                    ),
                    key=lambda item: item[0],
                )
                for _, _, result in fits:
                    report_lines.append(format_fit_report(result, self._display_label(ds)))
                    fits_by_sweep.append((self._display_label(ds), result))
            self.ecm_step.params_text.setPlainText("\n".join(report_lines))
            self._render_ecm_circuits(fits_by_sweep)

        # Remember the framing so the next step opens on the same view.
        pane = self._spectrum_pane_for(index)
        if pane is not None:
            state = pane.view_state()
            if state is not None:
                self._spectrum_view_state = state

        self._step_dirty.discard(index)

