"""Light/dark theming for the app."""

import pyqtgraph as pg
import qdarktheme

from gui import style
from gui.style import ACCENT

THEMES = ("light", "dark")


def diagram_colors(mode: str) -> dict:
    """Colors for a circuit schematic under ``mode``."""
    t = style.tokens(mode)
    return {"wire": t["text"], "value": t["diagram_value"], "muted": t["text_muted"]}


def apply_theme(mode: str) -> None:
    """Apply ``mode`` ("light" or "dark") to Qt and PyQtGraph."""
    if mode not in THEMES:
        mode = "light"
    qdarktheme.setup_theme(
        mode,
        custom_colors={"primary": ACCENT},
        additional_qss=style.build_app_qss(mode),
    )
    style.apply_app_font()

    # PyQtGraph has no figure/axes split: "background" covers both, and
    # "foreground" covers axis lines, ticks, and labels together.
    t = style.tokens(mode)
    pg.setConfigOption("background", t["surface"])
    pg.setConfigOption("foreground", t["text"])
