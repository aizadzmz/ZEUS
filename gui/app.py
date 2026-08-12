"""Entry point for the desktop GUI: python -m gui.app"""

import multiprocessing
import sys
from pathlib import Path

def _assets_dir() -> Path:
    """gui/assets, in a checkout and inside a PyInstaller bundle alike. This
    file is the frozen app's entry script, so the plain __file__-relative path
    resolves one directory too high and every asset below goes missing.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "gui" / "assets"
    return Path(__file__).resolve().parent / "assets"


ASSETS_DIR = _assets_dir()
ICON_PATH = ASSETS_DIR / "icon.ico"
# The splash uses the full logo rather than the icon, which is the wordmark alone, padded square for the taskbar.
SPLASH_LOGO_PATH = ASSETS_DIR / "splash.png"

# gui.style is Qt-light (no pyqtgraph, no qdarktheme), so the splash can share the real accent color.
from gui.style import ACCENT  # noqa: E402

# Splash card geometry, read by both _splash_pixmap and main() so the logo stays centred on the progress bar.
SPLASH_SIZE = (420, 260)
# installer/make_wizard_images.py mirrors this so the wizard and the app share a background; change both together.
SPLASH_BG = "#15181d"
_SPLASH_MARGIN = 24
# Caps the logo's width well inside the card, so a wide asset stays a mark rather than filling it.
SPLASH_LOGO_WIDTH = 260
_PROGRESS_MARGIN = 20  # left/right inset, and the gap below the bar
_PROGRESS_HEIGHT = 16
# Top edge of the progress bar, measured up from the bottom of the card.
_PROGRESS_TOP = _PROGRESS_MARGIN + _PROGRESS_HEIGHT

# Shown one at a time on the splash while each module loads; blocking here, with visible progress, puts the several-second wait up front.
_SPLASH_STEPS = (
    ("Loading numerical libraries...", "numpy"),
    ("Loading plotting libraries (PyQtGraph)...", "pyqtgraph"),
    ("Loading theme engine...", "qdarktheme"),
    ("Loading EIS parsers...", "core.io_utils"),
    ("Loading validation tools (pyimpspec)...", "core.validation"),
    ("Loading DRT tools...", "core.drt"),
    ("Loading circuit fitting tools...", "core.ecm"),
    # ~0.5 s, deferred by core.circuit_diagram, so without this the first visit to ECM Parameters pays it mid-click.
    ("Loading circuit diagrams...", "schemdraw.elements"),
    ("Loading filters...", "core.filtering"),
    ("Loading generic parser...", "core.generic_parser"),
    ("Loading multi-block parser...", "core.mb_parser"),
)


def _crash_log_path():
    """Where the crash log goes: the per-user app data directory, so it needs
    no write access to the install location."""
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    directory = Path(base) if base else Path.home()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "crash.log"


def _install_crash_reporter():
    """Make an unhandled failure visible instead of silent."""
    import faulthandler
    import traceback
    from datetime import datetime

    log_path = _crash_log_path()
    # Held open for the process lifetime: faulthandler writes to this descriptor from inside a fault handler, and line buffering means a hard crash loses nothing.
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=log_file)

    seen = set()
    reporting = False

    def report(exc_type, exc, tb):
        nonlocal reporting
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log_file.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} ===\n{text}")
        sys.stderr.write(text)

        # One dialog per distinct traceback, and never a second while the first is open: an exception from a paint or timer handler repeats on every event.
        if reporting or text in seen:
            return
        seen.add(text)
        reporting = True
        try:
            from PySide6.QtWidgets import QMessageBox

            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Unexpected error")
            box.setText(
                f"{exc_type.__name__}: {exc}\n\n"
                "The action was cancelled. Details were written to:\n"
                f"{log_path}"
            )
            box.setDetailedText(text)
            box.exec()
        except Exception:
            pass  # A failing error dialog must not replace the real error.
        finally:
            reporting = False

    sys.excepthook = report
    return log_path, log_file


def _install_hang_watchdog(log_file, seconds=12):
    """Dump every thread's stack to the log if the GUI stops responding."""
    import faulthandler

    from PySide6.QtCore import QTimer

    def rearm():
        faulthandler.dump_traceback_later(
            seconds, repeat=True, file=log_file, exit=False
        )

    timer = QTimer()
    # Well under `seconds`, so an ordinary slow operation never trips it.
    timer.setInterval(max(1000, int(seconds * 1000 / 3)))
    timer.timeout.connect(rearm)
    timer.start()
    rearm()
    return timer


def _splash_pixmap():
    """Build the splash background: the logo on a dark card. Composed in code
    rather than shipped pre-rendered, so the card and progress bar stay in step
    with the logo asset and the accent color."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter, QPixmap

    width, height = SPLASH_SIZE

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(SPLASH_BG))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    logo = QPixmap(str(SPLASH_LOGO_PATH))
    if logo.isNull():  # missing asset should not cost us the splash
        logo = QPixmap(str(ICON_PATH))
    if not logo.isNull():
        # Centred in the whole band above the progress bar, which holds nothing else.
        band = height - _PROGRESS_TOP
        # Bounded on both axes so a logo of any aspect ratio stays inside the band.
        logo = logo.scaled(
            SPLASH_LOGO_WIDTH,
            band - 2 * _SPLASH_MARGIN,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        painter.drawPixmap(
            (width - logo.width()) // 2,
            (band - logo.height()) // 2,
            logo,
        )

    painter.end()

    return pixmap


def main() -> None:
    # Must run before anything else: in a bundled executable this stops a spawned worker re-running the whole GUI.
    multiprocessing.freeze_support()

    # Keep these imports inside main(): Windows spawns each worker by re-importing this file, and at module scope they cost every worker ~8 s and exhausted the paging file.
    from importlib import import_module

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QIcon
    from PySide6.QtWidgets import QApplication, QProgressBar, QSplashScreen

    app = QApplication(sys.argv)
    # Gives QSettings a stable place to persist the theme choice.
    app.setOrganizationName("EIS Batch Analysis")
    app.setApplicationName("EIS Batch Analysis")
    app.setWindowIcon(QIcon(str(ICON_PATH)))

    # Before the splash: an import failure below should land in the log too.
    log_path, log_file = _install_crash_reporter()
    # Bound to `app` so the timer outlives main()'s locals.
    app._hang_watchdog = _install_hang_watchdog(log_file)
    print(f"Diagnostics log: {log_path}", file=sys.stderr)

    splash = QSplashScreen(_splash_pixmap())
    progress = QProgressBar(splash)
    progress.setGeometry(
        _PROGRESS_MARGIN,
        splash.height() - _PROGRESS_TOP,
        splash.width() - 2 * _PROGRESS_MARGIN,
        _PROGRESS_HEIGHT,
    )
    progress.setRange(0, len(_SPLASH_STEPS))
    progress.setTextVisible(False)
    progress.setStyleSheet(
        f"QProgressBar {{ background: #0b0d10; border: none; border-radius: 4px; }}"
        f"QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}"
    )
    splash.show()
    app.processEvents()

    for step, (message, module_name) in enumerate(_SPLASH_STEPS, start=1):
        splash.showMessage(
            message, Qt.AlignBottom | Qt.AlignHCenter, QColor("#e3e6ea")
        )
        app.processEvents()
        try:
            import_module(module_name)
        except Exception:
            # Let a broken import surface at its real use site, with a real traceback.
            pass
        progress.setValue(step)
        app.processEvents()

    from gui.main_window import MainWindow

    window = MainWindow()  # applies the saved (or default) theme itself
    window.show()
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
