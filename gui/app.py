"""Entry point for the desktop GUI: python -m gui.app"""

import multiprocessing
import sys
from pathlib import Path

ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.ico"


def main() -> None:
    # freeze_support() first: in a bundled executable it is what stops a
    # spawned worker from re-running the whole GUI, and it must run before
    # anything else in the process.
    multiprocessing.freeze_support()

    # These imports are inside main() deliberately -- do not hoist them to
    # module scope. ValidationWorker spreads large batches across processes,
    # and Windows starts each one by spawning a fresh interpreter that
    # re-imports this file (as __mp_main__) to rebuild its state. At module
    # scope, PySide6 + matplotlib + pyqtgraph + pyimpspec made that re-import
    # cost every worker ~8 s and enough memory that a pool of 8 exhausted the
    # paging file ("DLL load failed ... _core") and died. Down here, a worker
    # re-importing this module pays for `sys` and `pathlib` and nothing else.
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    # Organization/application names give QSettings a stable place to persist
    # the user's theme choice across launches.
    app.setOrganizationName("EIS Batch Analysis")
    app.setApplicationName("EIS Batch Analysis")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()  # applies the saved (or default) theme itself
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
