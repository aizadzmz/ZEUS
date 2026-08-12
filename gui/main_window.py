"""Main window: a step bar across the top, one page per stage below it."""

# Annotations are strings, so signatures can name core.* types without importing those modules.
from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

from PySide6.QtCore import QEventLoop, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Do not import core.* at module level: they pull in pyimpspec (~4 s), which is not needed to show a window. Import inside each use site instead.
if TYPE_CHECKING:  # names used only in annotations
    from core import EISParseError

from gui import style
from gui.generic_import_dialog import GenericImportDialog
# Safe at module scope: core.plotting needs only numpy and pyqtgraph, both warmed by gui/app.py.
from core.plotting import DEFAULT_LINE_WIDTH, DEFAULT_MARKER_SIZE
from gui.selection import SweepSelection
from gui.steps.base import (
    DEFAULT_SETTINGS_WIDTH,
    MAX_SETTINGS_WIDTH,
    MIN_SETTINGS_WIDTH,
)
from gui.steps.data_viz_step import DataVizStep
from gui.steps.drt_step import DRTStep
from gui.steps.ecm_step import ECMStep
from gui.steps.validation_step import ValidationStep
from gui.stepper import STEPS, StepBar
from gui.theme import THEMES, apply_theme, diagram_colors
from gui.workers import DRTWorker, ECMWorker, ValidationWorker

VALIDATION_METHODS = ("Kramers-Kronig", "Z-HIT")

# How many diffusion fits to keep. Sized to hold a large batch several times over, since a redraw must not evict what the same redraw is about to read.
MAX_DIFFUSION_FITS = 512

# Circuit edits kept for undo, as CDC snapshots.
MAX_CIRCUIT_UNDO = 50
# Significant figures written into the CDC field by a canvas edit: enough for a starting guess, short enough to stay readable.
CIRCUIT_CDC_DECIMALS = 6
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
        self.setWindowTitle("ZEUS")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1200, 800)

        # Loaded files, and the flat list of every sweep across them. To find a sweep's file use ds.file_id, not list position.
        self._files: List[LoadedFile] = []
        self._datasets: List = []
        # Active/on-screen sweeps (see gui.selection); built before the widgets, which bind to it in _build_ui.
        self._selection = SweepSelection(self)
        # Never reused, so a removed file's keys cannot collide with a new one.
        self._next_file_id = 0
        # {(method, ds.key): KramersKronigResult | ZHITResult}. Every cache here keys on ds.key, not ds.label, which repeats across files.
        self._validation_results = {}
        # {(method, ds.key): effective kwargs} -- saved with the result so a reloaded session can reproduce it.
        self._validation_params: Dict[Tuple[str, str], dict] = {}
        # {ds.key: point indices} -- eraser overrides, kept out of the DataSet mask and re-applied as a layer.
        self._manual_masked: Dict[str, set] = {}
        self._manual_kept: Dict[str, set] = {}
        self._worker: Optional[ValidationWorker] = None
        self._worker_errors: List[Tuple[str, str]] = []
        # Method of the run in flight, for the progress message.
        self._running_method = ""
        # The sweeps it covers, so the prune summary can be built once it ends.
        self._running_keys: List[str] = []
        # {ds.key: TRRBFResult} -- last DRT run wins
        self._drt_results = {}
        # {ds.key: effective kwargs}
        self._drt_params: Dict[str, dict] = {}
        # {ds.key: DRTPeaks}
        self._drt_peaks = {}
        # {(ds.key, cdc, masked indices): (impedances, FitResult) or (None, msg)}; keyed by the mask too, so an eraser edit refits rather than reusing a stale tail.
        self._diffusion_fits: Dict[tuple, tuple] = {}
        # {ds.key: FitResult or a failed fit's message}, rebuilt by _diffusion_applied and scoped to the screen: it answers what the readout describes.
        self._diffusion_shown: Dict[str, object] = {}
        self._drt_worker: Optional[DRTWorker] = None
        self._drt_worker_errors: List[Tuple[str, str]] = []
        # Settings for the batch in flight: result_ready carries only (label, result), so _on_drt_worker_result reads them from here.
        self._pending_drt_params: dict = {}
        # The same, per sweep: {ds.key: FitResult or message} frozen at _start_drt_run, and so scoped to the run rather than to the screen.
        self._pending_diffusion: Dict[str, object] = {}
        # What the batch in flight is called and how many sweeps it covers, for the summary shown once it ends.
        self._drt_run_name = "DRT"
        self._drt_run_total = 0
        # {(canonical cdc, ds.key): FitResult}, keyed by circuit as well as sweep so a second circuit keeps the first one's result.
        self._ecm_results: Dict[Tuple[str, str], object] = {}
        # {(canonical cdc, ds.key): effective kwargs}
        self._ecm_params: Dict[Tuple[str, str], dict] = {}
        # Which fitted circuit the ECM step draws; one overlay at a time.
        self._ecm_shown_cdc: Optional[str] = None
        self._ecm_worker: Optional[ECMWorker] = None
        self._ecm_worker_errors: List[Tuple[str, str]] = []
        # Marker & line style, edited through gui/marker_style_dialog.py. Per-file shapes are keyed by file_id and outlive a file being closed; a file with no entry falls back to the default cycle.
        self._file_markers: Dict[int, str] = {}
        self._marker_size = DEFAULT_MARKER_SIZE
        self._line_width = DEFAULT_LINE_WIDTH
        # As _pending_drt_params, plus the canonical CDC the cache key needs.
        self._pending_ecm: dict = {}
        # The editable circuit behind the canvas, derived from the CDC field, which stays the source of truth; _ecm_syncing marks writes that came from the canvas.
        self._ecm_tree = None
        self._ecm_syncing = False
        self._ecm_undo: List[str] = []
        self._ecm_redo: List[str] = []
        self._ecm_editor = None
        # What the canvas currently shows, so a redundant redraw can be skipped.
        self._ecm_drawn = None
        # textChanged fires per keystroke and the ECM overlay/report follow the code box, so coalesce until the text settles.
        self._ecm_display_timer = QTimer(self)
        self._ecm_display_timer.setSingleShot(True)
        self._ecm_display_timer.setInterval(250)
        self._ecm_display_timer.timeout.connect(self._refresh_ecm_display)
        # Steps replot lazily: _refresh() does the cheap bookkeeping and marks every step dirty, but only the visible step redraws.
        self._pending: Optional[dict] = None
        self._step_dirty: set = set()
        # Spectrum framing handed between steps, so Nyquist panning survives moving to the next step.
        self._spectrum_view_state = None
        # Which step that framing is read off when the next one opens; the stack reports the step being *entered*, not the one being left.
        self._step_index = 0
        # Guards the settings-width mirror from re-entering itself while it pushes the new width onto the other three steps.
        self._syncing_panel_width = False
        # splitterMoved fires continuously through a drag; coalesce the writes.
        self._panel_width_save_timer = QTimer(self)
        self._panel_width_save_timer.setSingleShot(True)
        self._panel_width_save_timer.setInterval(300)
        self._panel_width_save_timer.timeout.connect(self._save_settings_width)
        # Width for every step's settings column; restored from QSettings once the steps exist.
        self._panel_width = DEFAULT_SETTINGS_WIDTH

        self._settings = QSettings()
        saved = self._settings.value("theme", "light")
        self._theme_mode = saved if saved in THEMES else "light"

        self._build_menu()
        self._build_ui()

        # Apply the theme once the widgets exist, then reflect the mode in the menu without re-triggering the toggle handler.
        apply_theme(self._theme_mode)
        self.step_bar.set_theme_mode(self._theme_mode)
        self.data_viz_step.files_panel.set_theme_mode(self._theme_mode)
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

        # One QAction backs the menu item, the status-bar button and the shortcut, so their checked states stay in sync.
        self.dark_action = QAction("Dark mode", self, checkable=True)
        self.dark_action.setShortcut("Ctrl+D")
        self.dark_action.setStatusTip("Toggle between light and dark themes (Ctrl+D)")
        self.dark_action.toggled.connect(self._on_theme_toggled)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.dark_action)

        # Shares the menu action, so it stays in sync with it and Ctrl+D; addPermanentWidget docks bottom-right.
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

        # Says why the window has gone dead, and how to get it back; its own label rather than the warning banner _refresh() owns.
        self.eraser_banner = QLabel(
            "Eraser on — the rest of the window is locked. Click points on the "
            "spectrum to remove or restore them, then switch the Eraser button "
            "on the plot off to carry on."
        )
        self.eraser_banner.setWordWrap(True)
        self.eraser_banner.setObjectName("warningBanner")
        self.eraser_banner.hide()
        root.addWidget(self.eraser_banner)

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
        # Order must match gui.stepper.STEPS -- StepBar.index_of maps between them, and _render_active_step branches on the page index.
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
        self.drt_step.peaks_table.setRowCount(0)
        self.ecm_step.spectrum_pane.set_message("No sweep selected.")
        self.ecm_step.params_text.clear()
        self.data_viz_step.details_text.clear()

    def _wire_steps(self) -> None:
        """Connect every step's controls to the handlers here."""
        # Both run the full bookkeeping pass: the per-sweep lists _refresh() derives are keyed off the displayed subset.
        self._selection.selection_changed.connect(self._refresh)
        self._selection.cursor_moved.connect(self._refresh)

        viz = self.data_viz_step
        viz.open_button.clicked.connect(self._open_file)
        viz.add_files_button.clicked.connect(self._add_files_dialog)
        viz.remove_file_button.clicked.connect(self._on_remove_file_clicked)
        for step in self._steps():
            # The ECM step has no toggle to wire -- it is fixed on Single.
            if step.single_radio is not None:
                step.single_radio.toggled.connect(self._on_display_mode_changed)
            step.settings_width_changed.connect(self._on_settings_width_changed)
        viz.markers_radio.toggled.connect(self._refresh)
        viz.marker_style_button.clicked.connect(self._on_marker_style_clicked)
        viz.nyquist_view_radio.toggled.connect(self._on_visual_view_changed)

        val = self.validation_step
        val.inductive_check.toggled.connect(self._refresh)
        val.kk_radio.toggled.connect(self._on_method_changed)
        val.threshold_spin.valueChanged.connect(self._refresh)
        # The mode decides which limit the stored results are rejected against, so switching it re-masks without re-running.
        val.basic_radio.toggled.connect(self._refresh)
        val.hard_limit_spin.valueChanged.connect(self._refresh)
        # Not applied on redraw -- it is spent during a run -- but it moves the second line on the residual plot.
        val.soft_limit_spin.valueChanged.connect(self._refresh)
        val.run_validation_button.clicked.connect(self._run_validation)
        # Rejection reads the residual definition, so switching it re-masks the stored results; only a re-run can change what an *advanced* prune already removed.
        val.residual_modulus_radio.toggled.connect(self._refresh)
        val.export_results_button.clicked.connect(self._export_validation_results)

        drt = self.drt_step
        # Redraws the measured plot with the dropped points greyed out; the masks, and so every other step, are untouched.
        drt.remove_inductive_check.toggled.connect(self._refresh)
        # Both redraw the measured plot with the tail flattened, touching neither the masks nor any other step.
        drt.subtract_diffusion_check.toggled.connect(self._refresh)
        drt.diffusion_cdc_combo.currentIndexChanged.connect(self._on_diffusion_model_changed)
        drt.run_drt_button.clicked.connect(self._run_drt)
        drt.run_peak_analysis_button.clicked.connect(self._run_peak_analysis)
        drt.export_results_button.clicked.connect(self._export_drt_results)

        ecm = self.ecm_step
        ecm.preset_combo.currentIndexChanged.connect(self._on_ecm_preset_changed)
        ecm.build_from_drt_button.clicked.connect(self._build_circuit_from_drt)
        ecm.cdc_edit.textChanged.connect(self._on_ecm_cdc_changed)
        ecm.run_button.clicked.connect(self._run_ecm_fit)
        ecm.export_params_button.clicked.connect(self._export_ecm_parameters)

        from core.circuit_model import (
            add_branch,
            add_in_series,
            delete,
            duplicate,
            insert_at,
            wrap_in_parallel,
        )
        from gui.circuit_canvas import fill_element_menu

        canvas = ecm.canvas
        canvas.element_activated.connect(self._on_ecm_element_activated)
        canvas.insert_requested.connect(
            lambda cid, index, symbol: self._edit_ecm_circuit(insert_at, cid, index, symbol)
        )
        canvas.add_branch_requested.connect(
            lambda cid, symbol: self._edit_ecm_circuit(add_branch, cid, symbol)
        )
        canvas.add_series_requested.connect(
            lambda nid, symbol: self._edit_ecm_circuit(add_in_series, nid, symbol)
        )
        canvas.wrap_requested.connect(
            lambda nid, symbol: self._edit_ecm_circuit(wrap_in_parallel, nid, symbol)
        )
        canvas.duplicate_requested.connect(lambda nid: self._edit_ecm_circuit(duplicate, nid))
        canvas.delete_requested.connect(lambda nid: self._edit_ecm_circuit(delete, nid))
        canvas.undo_requested.connect(self._undo_ecm_edit)
        canvas.redo_requested.connect(self._redo_ecm_edit)

        # Filled on first open, not now: listing the elements imports pyimpspec, ~4 s while the window is still being built.
        ecm.add_element_menu.aboutToShow.connect(
            lambda: fill_element_menu(ecm.add_element_menu, self._append_ecm_element)
            if ecm.add_element_menu.isEmpty()
            else None
        )
        ecm.undo_button.clicked.connect(self._undo_ecm_edit)
        ecm.redo_button.clicked.connect(self._redo_ecm_edit)
        ecm.clear_circuit_button.clicked.connect(self._clear_ecm_circuit)
        ecm.fitted_values_radio.toggled.connect(self._on_ecm_values_mode_changed)
        self._update_ecm_undo_buttons()

        # The eraser works on both the Data Visualisation and Validation spectra; either pane's button arms them together.
        for pane in (viz.spectrum_pane, val.spectrum_pane):
            pane.point_mask_toggled.connect(self._on_point_mask_toggled)
            pane.eraser_toggled.connect(self._on_eraser_toggled)

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
        self._update_ecm_coverage()
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
        # Read off the step being left, here rather than when it was last drawn: an Auto-Scale or a pan comes long after that render.
        outgoing = self._spectrum_pane_for(self._step_index)
        if outgoing is not None:
            state = outgoing.view_state()
            if state is not None:
                self._spectrum_view_state = state
        self._step_index = index
        # Hidden QStackedWidget pages are not laid out, so re-assert the width here.
        self._steps()[index].set_settings_width(self._panel_width)
        incoming = self._spectrum_pane_for(index)
        if incoming is not None and self._spectrum_view_state is not None:
            incoming.set_view_state(self._spectrum_view_state)
        # The circuit is not tied to any sweep, so it draws even with no data loaded.
        if index == 3:
            self._render_ecm_circuits()
        self._render_active_step()

    def _spectrum_pane_for(self, index: int):
        """The pane showing the measured spectrum on a given step, if any."""
        if index == 0:
            return self.data_viz_step.spectrum_pane
        if index == 1:
            return self.validation_step.spectrum_pane
        if index == 2:
            return self.drt_step.top_pane
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
        index, shape by its file -- from the Marker & line style dialog, falling
        back to the file's position in the default cycle."""
        from core.plotting import SERIES_COLORS, default_marker_for

        file_position = {lf.file_id: i for i, lf in enumerate(self._files)}
        return {
            ds.key: (
                SERIES_COLORS[ds.index % len(SERIES_COLORS)],
                self._marker_for(ds.file_id, file_position.get(ds.file_id, 0)),
            )
            for ds in self._datasets
        }

    def _marker_for(self, file_id: int, position: int) -> str:
        """A file's marker: whatever it was given in the style dialog, else the
        shape its load position would hand it."""
        from core.plotting import default_marker_for

        return self._file_markers.get(file_id) or default_marker_for(position)

    def _on_marker_style_clicked(self) -> None:
        """The Marker & line style popup. Applies on OK only, so a cancelled
        dialog cannot leave the plots half-restyled."""
        from gui.marker_style_dialog import MarkerStyleDialog

        file_position = {lf.file_id: i for i, lf in enumerate(self._files)}
        dialog = MarkerStyleDialog(
            files=[(lf.file_id, lf.stem) for lf in self._files],
            markers={
                lf.file_id: self._marker_for(lf.file_id, file_position[lf.file_id])
                for lf in self._files
            },
            marker_size=self._marker_size,
            line_width=self._line_width,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        # Choices for files that have since been closed are kept, so reopening one in the same session brings its shape back.
        self._file_markers.update(dialog.markers)
        self._marker_size = dialog.marker_size
        self._line_width = dialog.line_width
        self._refresh()

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

    def _apply_base_mask(self, ds) -> None:
        """Reset ds to the mask a validation run is meant to see: the inductive
        filter (or none at all) plus the user's own eraser edits, pointedly
        *not* the threshold pass, whose rejections are derived from a result.
        """
        from core.filtering import clear_mask, mask_inductive_points

        if self.validation_step.inductive_check.isChecked():
            mask_inductive_points(ds)
        else:
            clear_mask(ds)
        self._apply_manual_overrides(ds)

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
        # Debounced: splitterMoved fires continuously through a drag and each write is a registry hit.
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

    # -------------------------------------------------------------- closing

    def _running_workers(self) -> List:
        """Every worker thread still going, in the order they are named to the
        user."""
        candidates = (
            ("validation", self._worker),
            ("DRT", self._drt_worker),
            ("circuit fit", self._ecm_worker),
        )
        return [(name, w) for name, w in candidates if w is not None and w.isRunning()]

    def closeEvent(self, event) -> None:
        """Never let the window take a live worker thread down with it: Qt aborts
        the process outright when a running QThread is destroyed (0xC0000409).
        Cancellation lands between sweeps rather than inside one, so the wait
        below is bounded by the sweep in flight and not by the rest of the batch.
        """
        running = self._running_workers()
        if not running:
            event.accept()
            return

        names = ", ".join(name for name, _ in running)
        confirm = QMessageBox.question(
            self,
            "Quit ZEUS",
            f"A {names} run is still going.\n\n"
            "Quitting stops it after the sweep it is on, which may take a "
            "moment to finish. Results already computed are kept, but "
            "unsaved ones are lost — use File ▸ Save session to keep them.\n\n"
            "Quit anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            event.ignore()
            return

        self.statusBar().showMessage("Finishing the current sweep before quitting…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for _, worker in running:
                worker.cancel()
            for _, worker in running:
                # Sliced rather than one open-ended wait() so the window keeps repainting; user input is excluded, a click landing during the wait being able to start another run.
                while not worker.wait(100):
                    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        finally:
            QApplication.restoreOverrideCursor()
        event.accept()

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
        self._discard_diffusion_fits()
        self._ecm_results = {}
        self._ecm_params = {}
        self._ecm_shown_cdc = None
        self._manual_masked = {}
        self._manual_kept = {}
        self._running_keys = []
        self.validation_step.prune_status_label.clear()

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

        # Broad on purpose: a bad column mapping must surface as a dialog rather than abort the process and lose loaded files.
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
        # Keyed by (ds.key, cdc, mask), so the sweep is the first third.
        self._diffusion_fits = {
            k: v for k, v in self._diffusion_fits.items() if k[0] not in removed_keys
        }
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
            self._show_empty_state()
            # This branch returns before _refresh(), which normally refreshes the ECM coverage line, so do it here.
            self._update_ecm_coverage()
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
            self.validation_step.mode,
            self.validation_step.soft_limit_spin.value(),
            self.validation_step.hard_limit_spin.value(),
            self.validation_step.max_removed_spin.value(),
            self.validation_step.residual_mode,
            self.drt_step.remove_inductive_check.isChecked(),
            self.drt_step.subtract_diffusion_check.isChecked(),
            self.drt_step.diffusion_cdc_combo.currentData(),
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
    # Scoped to the sweep on screen, unlike _export_bdf and _save_session above, which cover everything loaded.

    def _cursor_dataset(self, title: str):
        """The sweep the pagers are on, or None (with a message) when there is
        none. Scoped exports need one specific sweep, not the selection."""
        ds = self._selection.current()
        if ds is None:
            QMessageBox.information(self, title, "Select a sweep first.")
        return ds

    def _save_pane_image(self, pane) -> None:
        title = "Save plot as image"
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, title, "", f"{self._PNG_FILTER};;{self._SVG_FILTER}"
        )
        if not path:
            return
        path = self._chosen_image_path(path, chosen_filter)
        try:
            pane.save_image(str(path))
        except Exception as exc:
            QMessageBox.critical(self, title, f"Could not save the image:\n{exc}")
            return
        self.statusBar().showMessage(f"Saved plot to '{path.name}'.")

    # The two image formats every plot pane offers.
    _PNG_FILTER = "PNG image (*.png)"
    _SVG_FILTER = "SVG image (*.svg)"
    # Suffixes the two exporters between them recognise; anything else is given
    # one, since a format neither knows writes no file at all.
    _IMAGE_SUFFIXES = (".png", ".svg", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def _chosen_image_path(self, path: str, chosen_filter: str) -> Path:
        """A save path that names an image format, mirroring
        _chosen_export_path. A recognised suffix the user typed wins over the
        filter dropdown, which they may never have touched; anything else takes
        the dropdown's, since the exporter reads the format off the suffix and
        silently writes nothing when there is none it knows.

        Appended rather than substituted as _chosen_export_path does: that one
        works on a stem this app suggested, while an image name is typed from
        scratch, and with_suffix would turn "cell A v1.2" into "cell A v1.png".
        """
        path = Path(path)
        if path.suffix.lower() in self._IMAGE_SUFFIXES:
            return path
        suffix = ".svg" if chosen_filter == self._SVG_FILTER else ".png"
        return path.with_name(path.name + suffix)

    def _chosen_export_path(self, path: str, chosen_filter: str, zview_filter: str):
        """(path, is_zview) for a save dialog offering CSV or ZView. An explicit
        .z or .csv the user typed wins over the filter dropdown, which they may
        never have touched."""
        path = Path(path)
        if path.suffix.lower() in (".z", ".csv"):
            return path, path.suffix.lower() == ".z"
        zview = chosen_filter == zview_filter
        return path.with_suffix(".z" if zview else ".csv"), zview

    # How the validated spectrum leaves the app; both formats carry the kept points only.
    _VALIDATION_CSV_FILTER = "Spectrum table (*.csv)"
    _VALIDATION_ZVIEW_FILTER = "ZView data file (*.z)"

    def _export_validation_results(self) -> None:
        """The validated sweep on screen: the surviving points, plus whatever
        the chosen format can carry of the validation itself."""
        title = "Export validated data"
        ds = self._cursor_dataset(title)
        if ds is None:
            return
        method = self._validation_method
        result = self._validation_results.get((method, ds.key))
        if result is None:
            QMessageBox.information(
                self, title, f"Run a {method} validation on this sweep first."
            )
            return

        from core.bdf_export import file_stem

        suggested = f"{file_stem(ds)}.validated.csv"
        path, chosen_filter = QFileDialog.getSaveFileName(
            self,
            title,
            suggested,
            f"{self._VALIDATION_CSV_FILTER};;{self._VALIDATION_ZVIEW_FILTER}",
        )
        if not path:
            return
        path, zview = self._chosen_export_path(
            path, chosen_filter, self._VALIDATION_ZVIEW_FILTER
        )

        try:
            if zview:
                written = self._write_validation_zview(path, ds, result, method)
            else:
                written = self._write_validation_csv(path, ds, result)
        except Exception as exc:
            QMessageBox.critical(self, title, f"Could not export:\n{exc}")
            return

        kept = ds.data.get_num_points(masked=False)
        total = ds.data.get_num_points(masked=None)
        self.statusBar().showMessage(
            f"Exported {len(written)} file(s) for {self._display_label(ds)} — "
            f"{kept} of {total} point(s) kept."
        )

    def _write_validation_csv(self, path: Path, ds, result) -> list:
        from core.bdf_export import write_residuals, write_spectrum

        written = [write_spectrum(path, ds, kept_only=True)]
        # path.stem is "<sweep>.validated", so this lands beside the spectrum. Always in the ΔZ/|Z| convention, whatever the Residuals setting displays.
        written.append(
            write_residuals(path.with_name(f"{path.stem}_residuals.csv"), result)
        )
        return written

    def _write_validation_zview(self, path: Path, ds, result, method: str) -> list:
        """The kept points as a .z, and the validation's own reconstruction as a
        second .z beside it, which ZView opens as a separate data set to
        overlay."""
        from core.zview_export import write_spectrum_z, write_validation_fit_z

        kept = ds.data.get_num_points(masked=False)
        total = ds.data.get_num_points(masked=None)
        written = [
            write_spectrum_z(
                path,
                ds,
                kept_only=True,
                comment=f"{kept} of {total} points kept after {method} validation",
            )
        ]
        written.append(
            write_validation_fit_z(
                path.with_name(f"{path.stem}_fit.z"),
                result,
                comment=f"{method} fit to the {kept} kept points",
            )
        )
        return written

    # The two ways the DRT can leave the app; fitted peaks always follow the curve into a companion file.
    _DRT_CSV_FILTER = "DRT table (*.csv)"
    _DRT_ZVIEW_FILTER = "ZView data file (*.z)"

    def _export_drt_results(self) -> None:
        """The DRT curve for the sweep on screen, plus its peaks if they have
        been fitted, as either CSV or the formats ZView reads."""
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

        from core.bdf_export import file_stem

        suggested = f"{file_stem(ds)}.drt.csv"
        path, chosen_filter = QFileDialog.getSaveFileName(
            self,
            title,
            suggested,
            f"{self._DRT_CSV_FILTER};;{self._DRT_ZVIEW_FILTER}",
        )
        if not path:
            return

        path, zview = self._chosen_export_path(
            path, chosen_filter, self._DRT_ZVIEW_FILTER
        )

        peaks = self._drt_peaks.get(ds.key)
        try:
            if zview:
                written = self._write_drt_zview(path, ds, result, peaks)
            else:
                written = self._write_drt_csv(path, result, peaks)
        except Exception as exc:
            QMessageBox.critical(self, title, f"Could not export:\n{exc}")
            return

        self.statusBar().showMessage(
            f"Exported {len(written)} file(s) for {self._display_label(ds)}."
        )

    def _write_drt_csv(self, path: Path, result, peaks) -> list:
        from core.bdf_export import write_drt, write_drt_peaks

        written = [write_drt(path, result)]
        if peaks is not None:
            # path.stem is "<sweep>.drt", so this lands beside the curve as "<sweep>.drt_peaks.csv".
            written.append(
                write_drt_peaks(path.with_name(f"{path.stem}_peaks.csv"), peaks)
            )
        return written

    def _write_drt_zview(self, path: Path, ds, result, peaks) -> list:
        """The curve as a .z data file, and the peaks as the equivalent circuit
        they imply, which ZView opens as a model to fit from."""
        from core.zview_export import (
            estimate_series_resistance,
            write_drt_model,
            write_drt_z,
        )

        written = [write_drt_z(path, result)]
        if peaks is not None:
            # A DRT says nothing about the ohmic resistance, so Rs is seeded from the sweep itself.
            written.append(
                write_drt_model(
                    path.with_suffix(".mdl"),
                    peaks,
                    series_resistance=estimate_series_resistance(ds),
                )
            )
        return written

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
        # Not saved in a session; _update_ecm_coverage re-derives it from the restored circuit code.
        self._ecm_shown_cdc = None
        self._manual_masked = ui_state["manual_masked"]
        self._manual_kept = ui_state["manual_kept"]
        # Not restored but dropped: these are keyed by ds.key, which is a
        # position (file_id:index) rather than an identity, so the first sweep
        # of the restored session collides with the first sweep of whatever was
        # loaded before it. Keeping them would subtract the previous data's
        # fitted tail from this one, and the DRT would run on the result
        # without a word about it.
        self._discard_diffusion_fits()

        self._refresh_file_list_widget()
        self._populate_sweep_selectors()

        # Restore the filter widgets with signals blocked; the single _refresh() below covers them all.
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

        self.validation_step.set_limits(
            threshold=ui_state.get("residual_threshold"),
            soft=ui_state.get("soft_limit"),
            hard=ui_state.get("hard_limit"),
            max_removed=ui_state.get("max_removed"),
        )
        self.validation_step.set_mode(ui_state.get("validation_mode", "Basic"))
        # None for a session saved before the convention was a setting, leaving the widget on its default.
        self.validation_step.set_residual_mode(ui_state.get("residual_mode"))

        # The DRT step's own filters, which rewrite what that step plots. Left off, the restored spectrum would be drawn unfiltered under a curve computed from a filtered sweep.
        self.drt_step.remove_inductive_check.blockSignals(True)
        self.drt_step.remove_inductive_check.setChecked(
            bool(ui_state.get("drt_inductive_filter", False))
        )
        self.drt_step.remove_inductive_check.blockSignals(False)

        self.drt_step.subtract_diffusion_check.blockSignals(True)
        self.drt_step.subtract_diffusion_check.setChecked(
            bool(ui_state.get("drt_subtract_diffusion", False))
        )
        self.drt_step.subtract_diffusion_check.blockSignals(False)

        cdc = ui_state.get("drt_diffusion_cdc")
        if cdc is not None:
            index = self.drt_step.diffusion_cdc_combo.findData(cdc)
            if index >= 0:
                self.drt_step.diffusion_cdc_combo.blockSignals(True)
                self.drt_step.diffusion_cdc_combo.setCurrentIndex(index)
                self.drt_step.diffusion_cdc_combo.blockSignals(False)
        # The rows these two govern are enabled by the checkbox, whose signal was blocked above.
        self.drt_step._sync_relevance()

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
        # Residuals are Single-mode only: one figure per selected sweep would crowd out the combined spectra.
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
        Data Visualisation and Validation spectrum panes together, and takes
        the rest of the window out of reach until it is switched off."""
        for step in (self.data_viz_step, self.validation_step):
            step.spectrum_pane.set_eraser_enabled(checked)
        self._set_eraser_lock(checked)
        if checked:
            self.statusBar().showMessage(
                "Eraser on — click a point to remove it, or a grey × to "
                "restore it."
            )
        else:
            self.statusBar().showMessage("Eraser off.")

    def _set_eraser_lock(self, locked: bool) -> None:
        """Make the eraser modal: while it is on, only the spectra respond, the
        menus, step bar and every step's controls going dead. The one live
        control is the Eraser button on the plot overlay, which is why it lives
        there rather than in a settings column."""
        for action in (
            self.open_session_action,
            self.save_session_action,
            self.export_bdf_action,
            self.dark_action,
        ):
            # Disabled individually as well as via the menu bar: their shortcuts are owned by this window and would still fire.
            action.setEnabled(not locked)
        self.menuBar().setEnabled(not locked)
        self.step_bar.setEnabled(not locked)
        for step in self._steps():
            step.set_controls_locked(locked)

        self.eraser_banner.setVisible(locked)

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
        # Self-painted, so they take no colors from the stylesheet.
        self.step_bar.set_theme_mode(self._theme_mode)
        self.data_viz_step.files_panel.set_theme_mode(self._theme_mode)
        self._settings.setValue("theme", self._theme_mode)
        # The schematic's colors are baked into its SVG, so a theme change has to redraw it -- here rather than in _refresh(), which returns early with no data.
        if self._ecm_tree is not None:
            self._render_ecm_circuits()
        # Regenerate existing figures so plot colors follow the new theme.
        if self._datasets:
            self._refresh()

    def _run_validation(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return

        # The run reads these masks, so put them back to the filtered data first; otherwise each run starts from the last redraw's threshold pass and refits on fewer and fewer points.
        for ds in selected:
            self._apply_base_mask(ds)

        # Module-level functions, so ValidationWorker can pickle the runner by reference across processes.
        from core.validation import prune_iteratively, run_kk_test, run_zhit

        method = self._validation_method
        runner = run_kk_test if method == VALIDATION_METHODS[0] else run_zhit
        if self.validation_step.mode == "Advanced":
            # partial of module-level callables, so this still pickles into the process pool -- a closure or bound method would not.
            runner = partial(prune_iteratively, runner=runner, **self._prune_settings())

        # Masks must stay stable while the worker reads them, so the settings panels are locked for the run.
        self._worker_errors = []
        self._running_keys = [ds.key for ds in selected]
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

    def _prune_settings(self) -> dict:
        """The iterative prune's limits, as core.validation.prune_iteratively
        takes them."""
        return dict(
            hard_percent=self.validation_step.hard_limit_spin.value(),
            soft_percent=self.validation_step.soft_limit_spin.value(),
            max_removed=self.validation_step.max_removed_spin.value(),
            residual_mode=self.validation_step.residual_mode,
        )

    def _pruned_points(self, method: str, key: str) -> List[int]:
        """The points a stored result had already been pruned of. Empty for a
        basic run -- and inseparable from the result for an advanced one, which
        was fitted with exactly these points gone."""
        return self._validation_params.get((method, key), {}).get("pruned_points", [])

    def _on_validation_result(self, method: str, key: str, result) -> None:
        from core.validation import PruneOutcome

        # Z-HIT takes no extra kwargs; these mirror the non-default arguments in core.validation.run_kk_test.
        params = (
            {"test": "complex", "admittance": False, "num_F_ext_evaluations": 10}
            if method == VALIDATION_METHODS[0]
            else {}
        )
        if isinstance(result, PruneOutcome):
            # The removals are part of how this result was produced, not a separate cache: without them it describes no point set.
            params.update(
                self._prune_settings(),
                pruned_points=list(result.removed),
                passes=result.passes,
                stop_reason=result.stop_reason,
            )
            result = result.result
        self._validation_results[(method, key)] = result
        self._validation_params[(method, key)] = params

    def _on_validation_progress(self, done: int, total: int) -> None:
        """A count, not a name: sweeps run in a process pool and finish out of
        order."""
        self.statusBar().showMessage(
            f"Running {self._running_method} analysis… {done} of {total}"
        )

    def _on_validation_error(self, key: str, message: str) -> None:
        self._worker_errors.append((key, message))

    def _report_prune(self) -> None:
        """Summarise what the run that just ended pruned, per sweep. Silent
        after a basic run, which removes nothing during the run itself."""
        label = self.validation_step.prune_status_label
        lines = []
        for key in self._running_keys:
            params = self._validation_params.get((self._running_method, key), {})
            if "pruned_points" not in params:
                continue
            removed = len(params["pruned_points"])
            note = "" if params["stop_reason"] == "converged" else f", {params['stop_reason']}"
            lines.append(
                f"{self._display_label(key)}: {removed} point(s) over "
                f"{params['passes']} pass(es){note}"
            )
        label.setText("\n".join(lines))

    def _on_validation_finished(self) -> None:
        self._worker = None
        self._set_controls_enabled(True)
        self._update_validation_button_text()
        self._report_prune()
        self.statusBar().showMessage("Validation finished.")
        if self._worker_errors:
            details = "\n".join(
                f"- {self._display_label(key)}: {msg}" for key, msg in self._worker_errors
            )
            QMessageBox.warning(
                self, "Validation errors", f"Some sweeps failed:\n{details}"
            )
        self._refresh()

    def _drt_inputs(self, datasets: List, detached: bool = False) -> List:
        """The sweeps the DRT sees, which are not necessarily the sweeps
        themselves: the step's own inductive-tail mask and diffusion-tail
        subtraction are applied to copies, so the shared mask stays put.

        `detached` guarantees copies whether or not that filter is on, for the
        worker thread. The two filters compose in the order the spectrum is
        read, the inductive points going first.
        """
        from core.filtering import detached_copy, inductive_tail_removed

        if self.drt_step.remove_inductive_check.isChecked():
            datasets = [inductive_tail_removed(ds) for ds in datasets]
            detached = False  # already copies
        # Always returns copies, which is why it settles `detached` too.
        if self.drt_step.subtract_diffusion_check.isChecked():
            return self._diffusion_applied(datasets)
        # Nothing was subtracted, so nothing is on record. Cleared rather than left alone: stale fits would be read as this batch's and double-count the tail.
        self._diffusion_shown = {}
        return [detached_copy(ds) for ds in datasets] if detached else datasets

    def _diffusion_applied(self, datasets: List) -> List:
        """The sweeps with a fitted diffusion tail subtracted, always as copies.
        Fits are cached because this runs on every redraw, not just on Run. A
        sweep whose fit fails passes through unsubtracted rather than dropping
        out -- the readout says which."""
        from core.filtering import detached_copy, diffusion_impedance, impedance_subtracted

        cdc = self.drt_step.diffusion_cdc_combo.currentData()
        # What the readout describes, recorded here rather than looked up later: the sweep reaching this point may already be the inductive filter's copy, whose cache key differs.
        self._diffusion_shown = {}
        applied = []
        for ds in datasets:
            masked = frozenset(i for i, is_masked in ds.data.get_mask().items() if is_masked)
            cache_key = (ds.key, cdc, masked)
            entry = self._diffusion_fits.get(cache_key)
            if entry is None:
                try:
                    entry = diffusion_impedance(ds, cdc)
                except Exception as exc:
                    entry = (None, str(exc))
                self._diffusion_fits[cache_key] = entry
                self._evict_diffusion_fits()

            impedances, fit = entry
            self._diffusion_shown[ds.key] = fit
            applied.append(
                detached_copy(ds)
                if impedances is None
                else impedance_subtracted(ds, impedances)
            )
        return applied

    def _discard_diffusion_fits(self) -> None:
        """Throw away every diffusion fit on record. All three go together: the
        cache, what the readout is describing, and the snapshot a run in flight
        is working from. Called wherever the sweeps behind those keys are
        replaced wholesale, since ds.key names a position and not a sweep."""
        self._diffusion_fits = {}
        self._diffusion_shown = {}
        self._pending_diffusion = {}

    def _evict_diffusion_fits(self) -> None:
        """Keep the fit cache bounded, oldest first. The mask is part of the key,
        so every eraser click on a subtracted sweep mints an entry that can never
        be hit again; the cap sits well above a batch, so one redraw over the
        whole selection still lands entirely in cache.
        """
        while len(self._diffusion_fits) > MAX_DIFFUSION_FITS:
            self._diffusion_fits.pop(next(iter(self._diffusion_fits)))

    def _update_diffusion_label(self, displayed: List) -> None:
        """Say what was subtracted from the sweep on screen. Like the λ
        readout, only meaningful for a single sweep -- with several on screen
        each has its own fit."""
        from core.filtering import describe_diffusion_fit

        label = self.drt_step.diffusion_status_label
        # The row holds two lines; anything longer belongs in the tooltip, which is restored on every branch.
        label.setToolTip(self.drt_step.diffusion_status_tooltip)

        # The row spans both form columns and so has no label of its own; each message names itself.
        if not self.drt_step.subtract_diffusion_check.isChecked():
            label.setText("Subtracted: —")
            style.set_state(label, "muted")
            return

        if len(displayed) != 1:
            label.setText(
                f"Subtracted from {len(displayed)} sweeps\neach fitted separately"
            )
            style.set_state(label, "muted")
            return

        fit = self._diffusion_shown.get(displayed[0].key)
        if fit is None:
            label.setText("Subtracted: —")
            style.set_state(label, "muted")
        elif isinstance(fit, str):
            # The cache stores a failed fit's message in the result's place; the label elides what will not fit, and the tooltip keeps all of it, because a fitter's message can run to a paragraph.
            label.setText(f"Not subtracted\n{fit}")
            label.setToolTip(
                f"{self.drt_step.diffusion_status_tooltip}\n\nThis sweep: {fit}"
            )
            style.set_state(label, "error")
        else:
            # No "Subtracted:" prefix here: the element's own name opens the line, and that width is what the fitted values need.
            label.setText(describe_diffusion_fit(fit))
            style.set_state(label, "ok")

    def _drt_record(self, params: dict) -> dict:
        """What gets saved/exported for a run. The filter changes the input
        data rather than the algorithm, so run_drt never sees it as a kwarg,
        but a reader of the session needs to know it was on."""
        subtracting = self.drt_step.subtract_diffusion_check.isChecked()
        return dict(
            params,
            remove_inductive_tail=self.drt_step.remove_inductive_check.isChecked(),
            # The circuit as well as the flag: a subtraction cannot be reconstructed from saved data that no longer holds the tail it removed.
            subtract_diffusion=subtracting,
            diffusion_cdc=(
                self.drt_step.diffusion_cdc_combo.currentData() if subtracting else None
            ),
        )

    def _on_diffusion_model_changed(self) -> None:
        """Only redraws when the model is actually in use -- the combo is
        greyed out with the filter off, but a session restore can still move
        it, and refitting every sweep for a control nothing reads would be a
        long stall for no visible change."""
        if self.drt_step.subtract_diffusion_check.isChecked():
            self._refresh()

    def _update_optimal_lambda_label(self, selected: List) -> None:
        lambda_value = None
        if len(selected) == 1:
            result = self._drt_results.get(selected[0].key)
            lambda_value = getattr(result, "lambda_value", None)
        self.drt_step.optimal_lambda_label.setText(
            f"{lambda_value:.4g}" if lambda_value is not None else "—"
        )

    def _run_drt(self) -> None:
        """Run DRT, dispatching on the method chosen in the settings panel."""
        if self.drt_step.method == "trrbf":
            self._run_drt_simple()
        else:
            self._run_drt_bayesian()

    def _drt_run_settings(self) -> dict:
        """The '6. DRT settings' panel as run_drt kwargs. Both methods pass all
        of these; what separates them is credible_intervals and the sampling
        rows that only it reads."""
        return dict(
            rbf_type=self.drt_step.rbf_combo.currentData(),
            derivative_order=self.drt_step.derivative_combo.currentData(),
            rbf_shape=self.drt_step.shape_control_combo.currentData(),
            shape_coeff=self.drt_step.shape_coeff_spin.value(),
            mode=self.drt_step.mode_combo.currentData(),
            inductance=self.drt_step.inductance_check.isChecked(),
            cross_validation=self.drt_step.cv_combo.currentData(),
            lambda_value=self.drt_step.lambda_spin.value(),
        )

    def _run_drt_simple(self) -> None:
        selected = self._selected_datasets()
        if not selected:
            return
        self._start_drt_run(
            selected,
            dict(self._drt_run_settings(), credible_intervals=False),
            "DRT",
        )

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

        self._start_drt_run(
            selected,
            dict(
                self._drt_run_settings(),
                credible_intervals=True,
                num_samples=self.drt_step.num_samples_spin.value(),
                timeout=self.drt_step.timeout_spin.value(),
            ),
            "Bayesian DRT",
        )

    def _start_drt_run(self, selected: List, params: dict, name: str) -> None:
        """Hand a batch of sweeps to the DRT worker thread -- plain TR-RBF too,
        not just the Bayesian method. TR-RBF is fast only on a sweep pyimpspec
        can take its Toeplitz shortcut on, which needs the frequencies log-spaced
        to within 1%; the fallback takes tens of seconds a sweep.
        """
        from core.drt import run_drt

        def runner(ds):
            return run_drt(ds, **params)

        self._drt_run_name = name
        self._drt_run_total = len(selected)
        self._pending_drt_params = self._drt_record(params)
        self._drt_worker_errors = []
        inputs = self._drt_inputs(selected, detached=True)
        # A copy, not a reference: _diffusion_shown is rebuilt by every redraw while the pager stays live through a run, so one click on › would otherwise shrink it to that one sweep.
        self._pending_diffusion = dict(self._diffusion_shown)
        self._drt_worker = DRTWorker(runner, inputs, parent=self)
        self._drt_worker.result_ready.connect(self._on_drt_worker_result)
        self._drt_worker.error.connect(self._on_drt_worker_error)
        self._drt_worker.finished.connect(self._on_drt_worker_finished)
        self._set_controls_enabled(False)
        self.statusBar().showMessage(
            f"Running {name} on {len(selected)} sweep(s)… this may take a while."
        )
        self._drt_worker.start()

    def _on_drt_worker_result(self, key: str, result) -> None:
        from core.filtering import diffusion_element_cdc

        self._drt_results[key] = result
        # The diffusion fit is recorded per sweep, being what "Build circuit from DRT" puts back into the model. Read from _start_drt_run's snapshot, not the screen's.
        fit = self._pending_diffusion.get(key)
        self._drt_params[key] = dict(
            self._pending_drt_params,
            diffusion_element=(
                diffusion_element_cdc(fit) if fit is not None and not isinstance(fit, str) else None
            ),
        )

    def _on_drt_worker_error(self, key: str, message: str) -> None:
        self._drt_worker_errors.append((key, message))

    def _on_drt_worker_finished(self) -> None:
        self._drt_worker = None
        self._set_controls_enabled(True)
        computed = self._drt_run_total - len(self._drt_worker_errors)
        self.statusBar().showMessage(
            f"{self._drt_run_name} computed for {computed} of "
            f"{self._drt_run_total} sweep(s)."
        )
        if self._drt_worker_errors:
            details = "\n".join(
                f"- {self._display_label(key)}: {msg}" for key, msg in self._drt_worker_errors
            )
            QMessageBox.warning(self, "DRT errors", f"Some sweeps failed:\n{details}")
        self._update_optimal_lambda_label(self._selected_datasets())
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
        self._push_ecm_undo()
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
                "Run a DRT calculation (7) and then 'Run peak extraction' (8) "
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

        # A DRT of a subtracted sweep gives a circuit with no tail, while ECM fits the sweep as measured -- so the R-CPE pairs absorb the tail and the resistances come out badly.
        subtracted = (self._drt_params.get(source.key) or {}).get("diffusion_element")
        if subtracted:
            cdc += subtracted

        self._push_ecm_undo()
        self.ecm_step.cdc_edit.setText(cdc)
        num_pairs = cdc.count("(")
        tail = " + fitted diffusion element" if subtracted else ""
        self.statusBar().showMessage(
            f"Circuit built from {self._display_label(source)}: "
            f"{num_pairs} R-CPE pair(s) + series R{tail}."
        )

    def _on_ecm_cdc_changed(self, text: str) -> None:
        """Live-validate the circuit code, so a typo is caught before a batch
        is spent on it rather than as one identical error per sweep."""
        from core.ecm import validate_cdc

        ok, message = validate_cdc(text)
        self.ecm_step.cdc_status_label.setText(message)
        style.set_state(self.ecm_step.cdc_status_label, "ok" if ok else "error")
        self.ecm_step.run_button.setEnabled(ok)

        # Typed edits rebuild the canvas's tree; canvas edits wrote this text and must not have it parsed back over them.
        if not self._ecm_syncing:
            self._rebuild_ecm_tree(text, ok)

        # The ECM step previews this code as a schematic while nothing is fitted, so redraw immediately; once there are fits, only mark dirty, since rendering also rebuilds the text report.
        self._step_dirty.add(3)
        # The canvas is the editing surface, so it always follows the code; _render_ecm_circuits skips the redraw when nothing changed.
        self._render_ecm_circuits()

        # The plots and text report follow this code box, but rebuilding them per keystroke would lag, so they wait for the typing to settle.
        self._ecm_display_timer.start()

    def _refresh_ecm_display(self) -> None:
        """Re-point the overlay and the report at the circuit now in the code
        box. Cheaper than _refresh(): the masks and validity bookkeeping cannot
        have changed, only which cached fit is on show."""
        if self._pending is None:
            return
        self._update_ecm_coverage()
        self._pending["ecm_shown_cdc"] = self._ecm_shown_cdc
        self._pending["ecm_curves"] = self._ecm_fit_curves(
            self._pending["displayed_for"][3], self._ecm_shown_cdc
        )
        self._step_dirty.add(3)
        self._render_active_step()

    # --- the editable circuit --------------------------------------------

    def _rebuild_ecm_tree(self, text: str, ok: bool) -> None:
        """Re-derive the canvas's tree from the code. An unparseable code leaves
        the tree alone; _render_ecm_circuits reports it instead of drawing."""
        from core.circuit_model import empty_root, from_cdc

        if ok:
            self._ecm_tree = from_cdc(text)
        elif not text.strip():
            self._ecm_tree = empty_root()

    def _set_ecm_tree(self, tree) -> None:
        """Push a tree out to the CDC field, which redraws everything else."""
        from core.circuit_model import to_cdc

        self._ecm_tree = tree
        self._ecm_syncing = True
        try:
            self.ecm_step.cdc_edit.setText(to_cdc(tree, CIRCUIT_CDC_DECIMALS))
        finally:
            self._ecm_syncing = False

    def _push_ecm_undo(self) -> None:
        """Remember the current code before something replaces it."""
        self._ecm_undo.append(self.ecm_step.cdc_edit.text())
        del self._ecm_undo[:-MAX_CIRCUIT_UNDO]
        self._ecm_redo.clear()
        self._update_ecm_undo_buttons()

    def _apply_ecm_edit(self, tree) -> None:
        """Apply a canvas edit, remembering the code it replaced for undo."""
        self._push_ecm_undo()
        self._set_ecm_tree(tree)

    def _edit_ecm_circuit(self, action, *args) -> None:
        """Run one core.circuit_model action against the current tree."""
        if self._ecm_tree is None:
            return
        try:
            self._apply_ecm_edit(action(self._ecm_tree, *args))
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))

    def _undo_ecm_edit(self) -> None:
        if not self._ecm_undo:
            return
        self._ecm_redo.append(self.ecm_step.cdc_edit.text())
        self.ecm_step.cdc_edit.setText(self._ecm_undo.pop())
        self._update_ecm_undo_buttons()

    def _redo_ecm_edit(self) -> None:
        if not self._ecm_redo:
            return
        self._ecm_undo.append(self.ecm_step.cdc_edit.text())
        self.ecm_step.cdc_edit.setText(self._ecm_redo.pop())
        self._update_ecm_undo_buttons()

    def _update_ecm_undo_buttons(self) -> None:
        self.ecm_step.undo_button.setEnabled(bool(self._ecm_undo))
        self.ecm_step.redo_button.setEnabled(bool(self._ecm_redo))

    def _clear_ecm_circuit(self) -> None:
        from core.circuit_model import empty_root

        self._apply_ecm_edit(empty_root())

    def _append_ecm_element(self, symbol: str) -> None:
        from core.circuit_model import append_element

        self._edit_ecm_circuit(append_element, symbol)

    def _on_ecm_element_activated(self, node_id: int) -> None:
        """Open the parameter editor over the component that was clicked."""
        from core.circuit_model import ElementNode, find

        node = find(self._ecm_tree, node_id) if self._ecm_tree else None
        if not isinstance(node, ElementNode):
            return

        if self._ecm_editor is None:
            from gui.element_editor import ElementEditor

            self._ecm_editor = ElementEditor(self)
            self._ecm_editor.type_changed.connect(self._on_ecm_type_changed)
            self._ecm_editor.parameters_changed.connect(self._on_ecm_parameters_changed)

        self._ecm_editor.show_node(
            node,
            self._ecm_element_name(node_id),
            self._ecm_fitted_values(node_id),
            self.ecm_step.canvas.anchor_for(node_id),
        )

    def _on_ecm_type_changed(self, node_id: int, symbol: str) -> None:
        from core.circuit_model import replace_element

        self._edit_ecm_circuit(replace_element, node_id, symbol)
        # The new type has different parameters, so the open editor is stale.
        self._on_ecm_element_activated(node_id)

    def _on_ecm_parameters_changed(self, node_id: int, payload: dict) -> None:
        from core.circuit_model import set_element

        if self._ecm_tree is None:
            return
        try:
            self._apply_ecm_edit(set_element(self._ecm_tree, node_id, **payload))
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))

    def _ecm_element_name(self, node_id: int) -> str:
        """A component's display name (R_1, Q_ct), taken from the same circuit
        the schematic is drawn from so the two agree."""
        from core.circuit_model import ElementNode, to_circuit, walk

        if self._ecm_tree is None:
            return ""
        order = [n for n in walk(self._ecm_tree) if isinstance(n, ElementNode)]
        circuit = to_circuit(self._ecm_tree)
        for node, element in zip(order, circuit.get_elements()):
            if node.node_id == node_id:
                return circuit.get_element_name(element)
        return ""

    def _ecm_fitted_values(self, node_id: int) -> Optional[dict]:
        """This component's fitted values in the annotation on screen, if any."""
        result = self._ecm_annotation()[0]
        if result is None:
            return None
        fitted = result.parameters.get(self._ecm_element_name(node_id))
        return {symbol: p.get_value() for symbol, p in fitted.items()} if fitted else None

    def _ecm_annotation(self) -> Tuple[Optional[object], Optional[object]]:
        """The (fit, sweep) whose values label the schematic, or (None, None)
        when the canvas is showing starting values."""
        from core.ecm import canonical_cdc

        if self.ecm_step.values_mode != "Fitted" or self._ecm_tree is None:
            return None, None
        try:
            canonical = canonical_cdc(self.ecm_step.cdc_edit.text())
        except Exception:
            return None, None

        drawn = self._pending["displayed_for"][3] if self._pending else []
        for ds in drawn:
            result = self._ecm_results.get((canonical, ds.key))
            if result is not None:
                return result, ds
        return None, None

    def _render_ecm_circuits(self) -> None:
        """Draw the working circuit on the canvas, labelled with either its own
        starting values or the fit for the sweep on screen."""
        from core.circuit_diagram import build_editor_drawing

        canvas = self.ecm_step.canvas
        canvas.set_theme_mode(self._theme_mode)

        text = self.ecm_step.cdc_edit.text().strip()
        if self._ecm_tree is None:
            self._ecm_drawn = None
            # First draw: ECMStep seeds the code with signals blocked, so the tree has not been built from it yet.
            from core.ecm import validate_cdc

            self._rebuild_ecm_tree(text, validate_cdc(text)[0])

        if self._ecm_tree is None or not self._ecm_tree.children:
            self._ecm_drawn = None
            canvas.set_message(
                self.ecm_step.cdc_status_label.text()
                if text
                else "Empty circuit — use 'Add element' below, or pick a preset."
            )
            self.ecm_step.circuit_caption.setText("")
            return

        colors = diagram_colors(self._theme_mode)
        result, ds = self._ecm_annotation()
        # Every caller redraws unconditionally and a keystroke can reach here twice, so skip the schemdraw pass when nothing changed.
        signature = (text, id(result), self._theme_mode)
        if signature == self._ecm_drawn:
            return
        self._ecm_drawn = signature

        try:
            canvas.set_drawing(
                build_editor_drawing(
                    self._ecm_tree,
                    parameters=result.parameters if result is not None else None,
                    foreground=colors["wire"],
                    accent=colors["value"] if result is not None else colors["muted"],
                    show_errors=result is not None,
                )
            )
        except Exception as exc:
            canvas.set_message(f"Could not draw this circuit: {exc}")
            return

        self.ecm_step.circuit_caption.setText(self._ecm_caption(result, ds))

    def _ecm_caption(self, result, ds) -> str:
        from core.circuit_model import to_cdc

        topology = to_cdc(self._ecm_tree, -1)
        if result is not None:
            return (
                f"{topology} — fitted to {self._display_label(ds)}"
                f"   χ² = {result.pseudo_chisqr:.4g}"
            )
        if self.ecm_step.values_mode == "Fitted":
            return f"{topology} — no fit for this circuit on the sweep(s) shown"
        return f"{topology} — starting values"

    def _on_ecm_values_mode_changed(self, _checked: bool) -> None:
        """Initial/Fitted only changes the labels on the canvas."""
        self._render_ecm_circuits()

    def _ecm_current_cdc(self) -> Optional[str]:
        """The canonical form of the circuit in the code box, or None when it
        does not parse. The ECM step displays this circuit and no other."""
        from core.ecm import canonical_cdc

        try:
            return canonical_cdc(self.ecm_step.cdc_edit.text().strip())
        except Exception:
            return None

    def _ecm_fit_curves(self, drawn: List, cdc: Optional[str]) -> dict:
        """{key: (frequencies, impedances)} for `cdc`, per drawn sweep that has
        a fit of it. A fit is interpolated onto its own denser grid, so the
        plot builders need both."""
        curves = {}
        for ds in drawn:
            result = self._ecm_results.get((cdc, ds.key))
            if result is not None:
                curves[ds.key] = (
                    result.get_frequencies(num_per_decade=100),
                    result.get_impedances(num_per_decade=100),
                )
        return curves

    def _update_ecm_coverage(self) -> None:
        """Point the step at the circuit being edited -- the only one it draws
        -- and say how much of the selection has been fitted with it."""
        self._ecm_shown_cdc = self._ecm_current_cdc()

        label = self.ecm_step.shown_status_label
        label.setVisible(bool(self._ecm_results))
        if not self._ecm_results:
            return

        selected = self._selected_datasets()
        covered = sum(
            1 for ds in selected if (self._ecm_shown_cdc, ds.key) in self._ecm_results
        )
        if covered:
            label.setText(
                f"{covered} of {len(selected)} selected sweep(s) fitted with "
                f"this circuit."
            )
        else:
            # Says so explicitly, because the fits for other circuits are still cached and should not look thrown away.
            label.setText(
                "Not fitted with this circuit yet. Earlier circuits' fits are "
                "kept — put one back in the code box to see it again."
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
            # A one-element list so the closure can rebind it: each fit seeds the next, which is why ECMWorker runs the batch serially.
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
        # No need to point the step at what was just fitted: _refresh() below re-derives that from the code box.
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

    def _force_replot_ecm(self) -> None:
        """Rebuild the ECM step from current state, bypassing the dirty check
        -- the plot-area 'Replot' button, mirroring _force_replot_spectrum."""
        if self._pending is None:
            return
        self._step_dirty.add(3)
        self._ecm_drawn = None
        self._render_active_step()

    # ------------------------------------------------------------- refresh

    def _refresh(self) -> None:
        """Cheap, always-run bookkeeping: apply masks and validate against the
        current selection. Figures rebuild lazily in _render_active_step()."""
        if not self._datasets:
            return

        # Only reachable once a file has been loaded, so these are warm.
        from core.filtering import clear_mask, mask_inductive_points, mask_points
        from core.validation import mask_residual_outliers

        selected = self._selected_datasets()
        if not selected:
            self.warning_label.setText("Select at least one sweep to plot.")
            self.warning_label.show()
            self._show_empty_state()
            self.data_viz_step.spectrum_pane.set_message(
                "No sweep selected — tick one in Files and sets below."
            )
            # This branch returns before the recompute below, so refresh the "N of M fitted" count here.
            self._update_ecm_coverage()
            self._pending = None
            self._step_dirty.clear()
            self._update_residuals_header(0, 0)
            return

        method = self._validation_method
        threshold = self.validation_step.reject_threshold
        residual_mode = self.validation_step.residual_mode

        # Exactly what a run starts from, so the replay below lands on the same point set the stored result was fitted on.
        for ds in selected:
            self._apply_base_mask(ds)

        stale_keys = []
        for ds in selected:
            result = self._validation_results.get((method, ds.key))
            if result is not None:
                # Replayed before the threshold pass, not derived by it: an iterative prune's removals cannot be re-derived without re-running it.
                mask_points(ds, self._pruned_points(method, ds.key))
                try:
                    mask_residual_outliers(ds, result, threshold, residual_mode)
                except ValueError:
                    stale_keys.append(ds.key)
            # Re-applied on top of the outlier pass, which only adds masks, so a manually restored point survives a threshold that would drop it.
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

        # Masking above covers the whole working set; from here on, figures concern only the on-screen subset, which differs per step.
        displayed_for = {i: self._displayed_datasets(i) for i in range(len(STEPS))}

        # Each list below is read by exactly one step, built from that step's own displayed subset.
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
        # Must run before _ecm_shown_cdc is read below: it settles which circuit is displayed, from the code box.
        self._update_ecm_coverage()
        shown_cdc = self._ecm_shown_cdc
        ecm_curves = self._ecm_fit_curves(displayed_for[3], shown_cdc)
        self._pending = dict(
            selected=selected,
            displayed_for=displayed_for,
            method=method,
            threshold=threshold,
            residual_mode=residual_mode,
            # Second residual line, drawn only where it means something: in basic mode there is one limit, `threshold`.
            soft_threshold=(
                self.validation_step.soft_limit_spin.value()
                if self.validation_step.mode == "Advanced"
                else None
            ),
            validated_selected=validated_selected,
            drt_selected=drt_selected,
            ecm_curves=ecm_curves,
            ecm_shown_cdc=shown_cdc,
        )
        self._step_dirty = {0, 1, 2, 3}
        self._render_active_step()

    def _update_residuals_header(self, shown: int, total: int) -> None:
        """Say what the residual plot is showing, or why it is showing nothing.
        The plot is Singular-only (see _sync_display_mode_widgets), so in Multiple
        view there is no figure to describe and only the convention applies."""
        val = self.validation_step
        single = self._display_mode_for(1) == "Single"

        if total == 0:
            text = "No validated sweeps — run a validation first."
        elif not single:
            # The convention still governs rejection here, so it is worth saying the setting has not gone dead with the plot.
            text = (
                f"{total} validated sweep(s). Switch to Singular for the "
                "residual plot; the Residuals setting above still sets what "
                "is rejected."
            )
        elif shown == 0:
            text = "Nothing validated for the sweep on screen."
        else:
            text = "Showing the sweep on screen; page with ‹ › above."

        if self.validation_step.mode == "Advanced":
            # An advanced prune's removals are baked into the stored result, so re-reading them under a new convention is not possible.
            text += (
                "\n\nChanging the residual definition re-rejects against the "
                "hard limit, but what the iterative prune already removed "
                "stands until you re-run it."
            )
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
                marker_size=self._marker_size,
                line_width=self._line_width,
            )
        )

    def _render_peak_report(self, drawn: List) -> None:
        """The per-peak table for whatever is on screen, shown beside the DRT
        controls rather than in a pane of its own."""
        table = self.drt_step.peaks_table
        # Columns come from the upstream dataframe, so they follow the peak fitter; a "Set" column is prepended only when it earns its width.
        rows: List[Tuple[str, List[str]]] = []
        columns: List[str] = []
        for ds in drawn:
            peaks = self._drt_peaks.get(ds.key)
            if peaks is None:
                continue
            frame = peaks.to_peaks_dataframe()
            if not columns:
                columns = [str(c) for c in frame.columns]
            label = self._display_label(ds)
            for values in frame.itertuples(index=False):
                rows.append((label, [f"{v:.4g}" if isinstance(v, float) else str(v)
                                     for v in values]))

        multi = len({label for label, _ in rows}) > 1
        headers = (["Set"] if multi else []) + columns
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, (label, values) in enumerate(rows):
            for c, text in enumerate((([label] if multi else []) + values)):
                table.setItem(r, c, QTableWidgetItem(text))
        table.resizeColumnsToContents()

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
            # Singular draws its one residual figure straight away; Multiple collapses the pane outright, so there is nothing to draw.
            validated = p["validated_selected"]
            shown = validated[:1] if self._display_mode_for(1) == "Single" else []
            figure = None
            for ds in shown:
                result = self._validation_results[(p["method"], ds.key)]
                figure = build_residuals_plot(
                    result,
                    title=f"{p['method']} residuals — {self._display_label(ds)}",
                    threshold=p["threshold"],
                    soft_threshold=p["soft_threshold"],
                    residual_mode=p["residual_mode"],
                    # Bare, without the method the title carries: a point's metadata box names the sweep, as on every other plot.
                    label=self._display_label(ds),
                )
            self.validation_step.residuals_pane.set_widget(figure)
            self._update_residuals_header(len(shown), len(validated))

        elif index == 2:
            # Through _drt_inputs, so this step's own inductive-tail filter shows here as removed points without moving the other steps' plots or masks.
            self._draw_spectrum(
                self.drt_step.top_pane, self._drt_inputs(drawn), show_removed=True
            )
            # After the draw, not before: _drt_inputs is what fills in the fits the readout describes.
            self._update_diffusion_label(drawn)

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
                        marker_size=self._marker_size,
                        line_width=self._line_width,
                    )
                )
            else:
                # No fit for what is on screen: show the measured spectrum, so the step still says which sweep "Fit circuit" would act on.
                self._draw_spectrum(self.ecm_step.spectrum_pane, drawn, show_removed=False)

            # The circuit in the code box only, one entry per drawn sweep; the other circuits' fits stay cached but listing them all made this unreadable.
            from core.ecm import format_fit_report

            report_lines = []
            for ds in drawn:
                result = self._ecm_results.get((p["ecm_shown_cdc"], ds.key))
                if result is not None:
                    report_lines.append(format_fit_report(result, self._display_label(ds)))
            self.ecm_step.params_text.setPlainText("\n".join(report_lines))
            self._render_ecm_circuits()

        # The framing is not read off here: _on_step_changed takes it from the
        # step being left, which is the only moment it is final.
        self._step_dirty.discard(index)

