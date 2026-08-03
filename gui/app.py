"""Entry point for the desktop GUI: python -m gui.app"""

import multiprocessing
import sys
from pathlib import Path

ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"
ACCENT = "#3b6fd4"  # matches gui.theme.ACCENT; duplicated so the splash can
                    # render before gui.theme (and the pyqtgraph import it
                    # drags in) is loaded.

# Shown one at a time on the splash screen while each module loads. Together
# these cost several seconds -- more on a cold boot, before Windows has these
# DLLs/pyd files cached -- and every one of them used to be paid silently
# after the window was already shown, so the app *looked* fast but the first
# click on Plot/Validate ate the real cost. Blocking on them here instead,
# with visible progress, moves the wait to where the user expects it.
_SPLASH_STEPS = (
    ("Loading numerical libraries...", "numpy"),
    ("Loading plotting libraries (PyQtGraph)...", "pyqtgraph"),
    ("Loading theme engine...", "qdarktheme"),
    ("Loading EIS parsers...", "core.io_utils"),
    ("Loading validation tools (pyimpspec)...", "core.validation"),
    ("Loading DRT tools...", "core.drt"),
    ("Loading circuit fitting tools...", "core.ecm"),
    # ~0.5 s, and core.circuit_diagram defers it like every other heavy
    # import, so without this the first visit to the ECM Parameters tab pays
    # it mid-click.
    ("Loading circuit diagrams...", "schemdraw.elements"),
    ("Loading filters...", "core.filtering"),
    ("Loading generic parser...", "core.generic_parser"),
    ("Loading multi-block parser...", "core.mb_parser"),
)


def _crash_log_path():
    """Where the crash log goes: the per-user app data directory, so it
    needs no write access to the install location."""
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    directory = Path(base) if base else Path.home()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "crash.log"


def _install_crash_reporter():
    """Make an unhandled failure visible instead of silent.

    PySide6 does not abort on an exception raised inside a slot -- it hands
    it to sys.excepthook and carries on with a half-updated window. In a
    windowed launch stderr goes nowhere, so the default outcome is a button
    that appears to do nothing, with no clue why. Route Python exceptions
    to a log file *and* a dialog naming that file, and point faulthandler
    at the same log so a native crash (the one failure mode that really
    does kill the process) leaves a C-level traceback behind too.

    Returns the log path.
    """
    import faulthandler
    import traceback
    from datetime import datetime

    log_path = _crash_log_path()
    # Held open for the process lifetime: faulthandler writes to this
    # descriptor from inside a fault handler, where opening a file is not
    # safe. Line-buffered so a hard crash can't lose what preceded it.
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=log_file)

    seen = set()
    reporting = False

    def report(exc_type, exc, tb):
        nonlocal reporting
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log_file.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} ===\n{text}")
        sys.stderr.write(text)

        # One dialog per distinct traceback, and never a second dialog while
        # the first is open: an exception thrown from a paint or timer
        # handler repeats on every event, and a modal box per repeat would
        # bury the app in unclosable windows.
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
    """Dump every thread's stack to the log if the GUI stops responding.

    A freeze is the one failure that leaves nothing behind to report:
    there's no exception, and Windows just offers to close the window.
    faulthandler's timer runs on its own thread and fires unless something
    keeps pushing it back, so a QTimer on the event loop pushes it back on
    every tick. While the GUI is healthy the dump never happens; once the
    event loop is stuck for `seconds`, the log gets the stack it is stuck
    inside -- repeatedly, so a slow loop is distinguishable from a deadlock.

    Returns the timer, which the caller must keep a reference to.
    """
    import faulthandler

    from PySide6.QtCore import QTimer

    def rearm():
        faulthandler.dump_traceback_later(
            seconds, repeat=True, file=log_file, exit=False
        )

    timer = QTimer()
    # Comfortably more often than `seconds` so an ordinary slow-but-working
    # operation never trips it.
    timer.setInterval(max(1000, int(seconds * 1000 / 3)))
    timer.timeout.connect(rearm)
    timer.start()
    rearm()
    return timer


def _splash_pixmap():
    """Build the splash background: app icon + name on a dark card.

    Drawn in code (rather than shipping a PNG) so it always matches
    ICON_PATH and needs no extra asset to keep in sync.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap

    width, height = 420, 260
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#1e2228"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    icon = QPixmap(str(ICON_PATH))
    if not icon.isNull():
        icon = icon.scaled(
            96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap((width - icon.width()) // 2, 36, icon)

    painter.setPen(QColor("#e3e6ea"))
    painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
    painter.drawText(
        pixmap.rect().adjusted(0, 150, 0, 0), Qt.AlignHCenter, "EIS Batch Analysis"
    )
    painter.end()

    return pixmap


def main() -> None:
    # freeze_support() first: in a bundled executable it is what stops a
    # spawned worker from re-running the whole GUI, and it must run before
    # anything else in the process.
    multiprocessing.freeze_support()

    # These imports are inside main() deliberately -- do not hoist them to
    # module scope. ValidationWorker spreads large batches across processes,
    # and Windows starts each one by spawning a fresh interpreter that
    # re-imports this file (as __mp_main__) to rebuild its state. At module
    # scope, PySide6 + pyqtgraph + pyimpspec made that re-import
    # cost every worker ~8 s and enough memory that a pool of 8 exhausted the
    # paging file ("DLL load failed ... _core") and died. Down here, a worker
    # re-importing this module pays for `sys` and `pathlib` and nothing else.
    from importlib import import_module

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QIcon
    from PySide6.QtWidgets import QApplication, QProgressBar, QSplashScreen

    app = QApplication(sys.argv)
    # Organization/application names give QSettings a stable place to persist
    # the user's theme choice across launches.
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
    progress.setGeometry(20, splash.height() - 36, splash.width() - 40, 16)
    progress.setRange(0, len(_SPLASH_STEPS))
    progress.setTextVisible(False)
    progress.setStyleSheet(
        f"QProgressBar {{ background: #12151a; border: none; border-radius: 4px; }}"
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
            # A broken import must surface at the real use site with a real
            # traceback, not swallowed here where nobody can act on it.
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
