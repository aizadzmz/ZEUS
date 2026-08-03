"""Light/dark theming for the app.

Qt chrome is themed with qdarktheme (the pyqtdarktheme-fork package, imported
as ``qdarktheme``). PyQtGraph plots are themed via pg.setConfigOption, which
only affects widgets created *after* the call, so plots must be rebuilt after
a theme switch to pick up the new colors.
"""

import pyqtgraph as pg
import qdarktheme

THEMES = ("light", "dark")
ACCENT = "#3b6fd4"  # muted blue, applied as qdarktheme's "primary" color

# PyQtGraph doesn't distinguish figure vs. axes background, so "background"
# covers both, and "foreground" covers axis lines/ticks/labels together.
_LIGHT_PG = {"background": "white", "foreground": "#252931"}
_DARK_PG = {"background": "#1e2228", "foreground": "#e3e6ea"}

# Circuit schematics (core.circuit_diagram) are SVG rather than PyQtGraph, so
# they can't read pg's config and are handed their colors instead. "wire" is
# the symbols and element names, "value" the fitted values called out under
# each component, and "muted" the initial values of a circuit that has not
# been fitted yet. ACCENT itself is too dark to read against the dark
# background, hence the lighter value color there.
_LIGHT_DIAGRAM = {"wire": "#252931", "value": "#2f5fbf", "muted": "#7b8290"}
_DARK_DIAGRAM = {"wire": "#e3e6ea", "value": "#7aa5f0", "muted": "#98a0ac"}


def diagram_colors(mode: str) -> dict:
    """Colors for a circuit schematic under ``mode`` -- see _LIGHT_DIAGRAM."""
    return _DARK_DIAGRAM if mode == "dark" else _LIGHT_DIAGRAM


def apply_theme(mode: str) -> None:
    """Apply ``mode`` ("light" or "dark") to Qt and PyQtGraph.

    qdarktheme.setup_theme finds the running QApplication itself, so no app
    handle is needed here. Call before (re)rendering figures.

    pg.setConfigOption only affects widgets created *after* the call, so
    PyQtGraph plots must be rebuilt after a theme switch to pick it up.
    """
    if mode not in THEMES:
        mode = "light"
    qdarktheme.setup_theme(mode, custom_colors={"primary": ACCENT})
    pg_colors = _DARK_PG if mode == "dark" else _LIGHT_PG
    pg.setConfigOption("background", pg_colors["background"])
    pg.setConfigOption("foreground", pg_colors["foreground"])
