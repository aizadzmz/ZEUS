"""Design tokens and the application stylesheet -- the one place the app's
font, palette, spacing, and widget styling are decided."""

from __future__ import annotations

from typing import Dict

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget

ACCENT = "#3b6fd4"  # muted blue, applied as qdarktheme's "primary" color

# --------------------------------------------------------------- spacing
#
# A 4px base step, named for the use site so call sites read as intent.

PANEL_MARGINS = (12, 12, 12, 12)   # around a step's settings column
PANEL_SPACING = 10                 # between group boxes in that column
GROUP_MARGINS = (12, 10, 12, 12)   # inside a group box
GROUP_SPACING = 6                  # between rows inside a group box
FORM_H_SPACING = 10                # QFormLayout label -> field
FORM_V_SPACING = 6                 # QFormLayout row -> row
CONTENT_MARGINS = (8, 6, 8, 6)     # header rows above a plot
LIST_SPACING = 8                   # between stacked figures

RADIUS = 6
RADIUS_PILL = 13

# Height cap for read-only text panes, which would otherwise grow unbounded in
# a scroll area and push everything below off-screen. They keep their own
# scrollbars, so nothing becomes unreachable.
TEXT_PANE_MAX_HEIGHT = 130

# ------------------------------------------------------------------ type
#
# qdarktheme's stylesheet sets no font-family or font-size, so setting the
# application font here is safe and will not be overridden.

FONT_PT_BASE = 10.0
FONT_PT_SMALL = 9.0      # status lines, pager metadata, figure notes
FONT_PT_SECTION = 8.5    # uppercase sub-section headers


def apply_app_font() -> None:
    """Scale the application font without touching its family."""
    app = QApplication.instance()
    if app is None:
        return
    font = app.font()
    font.setPointSizeF(FONT_PT_BASE)
    app.setFont(font)


def section_font() -> QFont:
    """Small uppercase label that divides a long form into sections."""
    font = QFont(QApplication.instance().font())
    font.setPointSizeF(FONT_PT_SECTION)
    font.setBold(True)
    font.setCapitalization(QFont.AllUppercase)
    font.setLetterSpacing(QFont.PercentageSpacing, 108)
    return font


def mono_font() -> QFont:
    """Fixed-pitch font for the DRT peak table and circuit fit report, which
    are pre-aligned text."""
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPointSizeF(FONT_PT_SMALL)
    return font


# ---------------------------------------------------------------- colors

_LIGHT: Dict[str, str] = {
    "text": "#252931",
    "text_muted": "#7b8290",
    "surface": "#ffffff",
    "surface_alt": "#f4f6f8",
    "border": "#dfe3e8",
    "accent": ACCENT,
    "accent_text": "#ffffff",
    "diagram_value": "#2f5fbf",
    "warning_fg": "#7a5900",
    "warning_bg": "#fff3cd",
    "warning_border": "#ffe08a",
    "success_fg": "#1a7f37",
    "danger_fg": "#b42318",
}

_DARK: Dict[str, str] = {
    "text": "#e3e6ea",
    "text_muted": "#98a0ac",
    "surface": "#1e2228",
    "surface_alt": "#23272e",
    "border": "#333a43",
    # ACCENT is too dark to read against the dark background, so the lighter
    # value used for fitted circuit values does double duty here.
    "accent": "#7aa5f0",
    "accent_text": "#12151a",
    "diagram_value": "#7aa5f0",
    "warning_fg": "#e5b95c",
    "warning_bg": "#3a2f14",
    "warning_border": "#5e4c1d",
    "success_fg": "#56d364",
    "danger_fg": "#ff7b72",
}


def tokens(mode: str) -> Dict[str, str]:
    return _DARK if mode == "dark" else _LIGHT


def set_state(widget: QWidget, state: str) -> None:
    """Tag a widget as ok / error / muted so the stylesheet can color it."""
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ------------------------------------------------------------------- QSS

# Emitted identically for both themes: semi-transparent greys read correctly
# over light and dark alike, so the step bar and figure overlay need only one
# definition.
#
# The #stepBar / #figureOverlay ancestor scopes are load-bearing -- unscoped,
# these bare `QToolButton` rules would also restyle the sweep pager arrows and
# the status-bar theme toggle.
_AGNOSTIC_QSS = f"""
QWidget#stepBar QToolButton {{
    border: none;
    padding: 6px 14px;
    border-radius: {RADIUS_PILL}px;
    font-weight: normal;
}}
QWidget#stepBar QToolButton:hover {{
    background: rgba(128, 128, 128, 25%);
}}
QWidget#stepBar QToolButton:checked {{
    background: rgba(128, 128, 128, 45%);
    font-weight: bold;
}}

QWidget#figureOverlay {{
    background: rgba(128, 128, 128, 40%);
    border: 1px solid rgba(128, 128, 128, 60%);
    border-radius: {RADIUS}px;
}}
QWidget#figureOverlay QToolButton {{
    border: none;
    padding: 3px 8px;
    border-radius: 4px;
}}
QWidget#figureOverlay QToolButton:checked {{
    background: rgba(0, 0, 0, 35%);
}}
"""


def build_app_qss(mode: str) -> str:
    """The application stylesheet, layered on top of qdarktheme's."""
    t = tokens(mode)
    return _AGNOSTIC_QSS + f"""
/* Settings column: a faint tint separates controls from the plot area
   without drawing a border down the middle. */
QScrollArea#settingsPanel {{
    border: none;
}}
QWidget#settingsPanelBody {{
    background: {t["surface_alt"]};
}}

QLabel#sectionHeader {{
    color: {t["text_muted"]};
    margin-top: 10px;
}}

QLabel#warningBanner {{
    color: {t["warning_fg"]};
    background: {t["warning_bg"]};
    border: 1px solid {t["warning_border"]};
    border-radius: 4px;
    padding: 6px 10px;
}}

QLabel[state="ok"] {{ color: {t["success_fg"]}; }}
QLabel[state="error"] {{ color: {t["danger_fg"]}; }}
QLabel[state="muted"] {{ color: {t["text_muted"]}; }}

/* One filled button per step -- the action that step exists to perform.
   Everything else keeps qdarktheme's outlined default. */
QPushButton[variant="primary"] {{
    color: {t["accent_text"]};
    background: {t["accent"]};
    border: 1px solid {t["accent"]};
}}
QPushButton[variant="primary"]:hover {{
    background: {t["diagram_value"]};
    border-color: {t["diagram_value"]};
}}
QPushButton[variant="primary"]:disabled {{
    color: {t["text_muted"]};
    background: transparent;
    border: 1px solid {t["border"]};
}}
/* Exports: muted text but the border stays. Infrequent is not obscure, and
   stripping the affordance entirely would hurt discoverability. */
QPushButton[variant="quiet"] {{
    color: {t["text_muted"]};
}}

QWidget#segmented QToolButton {{
    border: 1px solid {t["border"]};
    border-left-width: 0;
    padding: 4px 10px;
    border-radius: 0;
}}
QWidget#segmented QToolButton[seg="first"] {{
    border-left-width: 1px;
    border-top-left-radius: {RADIUS}px;
    border-bottom-left-radius: {RADIUS}px;
}}
QWidget#segmented QToolButton[seg="last"] {{
    border-top-right-radius: {RADIUS}px;
    border-bottom-right-radius: {RADIUS}px;
}}
QWidget#segmented QToolButton:checked {{
    color: {t["accent_text"]};
    background: {t["accent"]};
    border-color: {t["accent"]};
}}
QWidget#segmented QToolButton:!checked:hover {{
    background: rgba(128, 128, 128, 20%);
}}

QLabel#figureMessage {{
    color: {t["text_muted"]};
    padding: 24px;
}}
QLabel#diagramCaption {{
    font-weight: 600;
    padding-top: 6px;
}}
QLabel#figureNote {{
    color: {t["text_muted"]};
    padding-top: 12px;
}}
"""
