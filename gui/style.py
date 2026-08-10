"""Design tokens and the application stylesheet -- the one place the app's
font, palette, spacing, and widget styling are decided."""

from __future__ import annotations

from typing import Dict

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget

# Navy: the deep end of the accent ramp, behind anything the accent fills.
# Text takes the lighter `accent` instead -- navy reads as black at body size.
ACCENT = "#000080"

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
PAGER_FIELD_GAP = 24               # sweep pager: File ... | Set ... , so the
                                   # two readouts do not read as one phrase
PAGER_ARROW_GAP = 10               # sweep pager: `‹‹ ‹ | › ››`, splitting the
                                   # four arrows into a back and a forward group

RADIUS = 6

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
FONT_PT_STEPPER = 11.5   # the step bar's stage names, the top-level navigation


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


def section_title_font() -> QFont:
    """The title on a settings card -- heavier than its sub-headers, not upper."""
    font = QFont(QApplication.instance().font())
    font.setBold(True)
    return font


def mono_font() -> QFont:
    """Fixed-pitch font for the DRT peak table and circuit fit report, which
    are pre-aligned text."""
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPointSizeF(FONT_PT_SMALL)
    return font


# ---------------------------------------------------------------- colors
#
# The accent is a ramp: subtle (washes) -> soft (hovers) -> border (hairlines)
# -> accent (text and strokes) -> deep (anything filled). Dark mode inverts the
# deep end. The split at the top of the ramp is the rule: navy fills, medium
# blue writes.

_LIGHT: Dict[str, str] = {
    "text": "#252931",
    "text_muted": "#708090",   # slate
    "surface": "#ffffff",
    # Blue-tinted rather than neutral grey, so the settings column reads as the
    # accent's side of the window before a single control is styled.
    "surface_alt": "#f4f7fc",
    # Neutral and slightly sunken, for read-only output panes. Deliberately not
    # blue: output should sit back from the tinted inputs around it.
    "surface_sunken": "#f5f5f5",
    "border": "#dfe3e8",
    "accent_subtle": "#eceff9",
    "accent_soft": "#d2daf2",
    "accent_border": "#8f9fdc",
    "accent": "#0000cd",  # mediumblue: navy's readable weight for text
    "accent_deep": ACCENT,
    "accent_hover": "#1a1a99",  # navy lifted a step, for hover on filled navy
    "accent_text": "#ffffff",
    "diagram_value": "#1a2f8c",
    "warning_fg": "#8b4513",    # saddlebrown: warmer than the blues, still dark
    "warning_bg": "#fdf6dd",
    "warning_border": "#f0e68c",  # khaki
    "success_fg": "#1a7f37",
    "danger_fg": "#b42318",
}

_DARK: Dict[str, str] = {
    "text": "#e3e6ea",
    "text_muted": "#98a0ac",
    "surface": "#1e2228",
    # Carries a blue cast of its own, mirroring the light panel tint.
    "surface_alt": "#232333",
    "surface_sunken": "#191c22",
    "border": "#333a43",
    "accent_subtle": "#1b2033",
    "accent_soft": "#26304d",
    "accent_border": "#3d4a72",
    # ACCENT is too dark to read against the dark background, so dark mode uses
    # a lightened navy of the same hue family instead.
    "accent": "#8fa2e6",
    "accent_deep": "#a8b6ee",
    "accent_hover": "#bcc7f2",
    "accent_text": "#12151a",
    "diagram_value": "#8fa2e6",
    "warning_fg": "#f0e68c",    # khaki reads as the warm tone on dark
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

def build_app_qss(mode: str) -> str:
    """The application stylesheet, layered on top of qdarktheme's."""
    t = tokens(mode)
    return f"""
/* Settings column: a faint tint separates controls from the plot area
   without drawing a border down the middle. */
QScrollArea#settingsPanel {{
    border: none;
}}
QWidget#settingsPanelBody {{
    background: {t["surface_alt"]};
}}
/* The panel's own scrollbar picks up the accent; the plot-side ones do not. */
QScrollArea#settingsPanel QScrollBar::handle:vertical,
QScrollArea#settingsPanel QScrollBar::handle:horizontal {{
    background: {t["accent_soft"]};
    border-radius: 3px;
}}
QScrollArea#settingsPanel QScrollBar::handle:vertical:hover,
QScrollArea#settingsPanel QScrollBar::handle:horizontal:hover {{
    background: {t["accent_border"]};
}}

/* Each settings group is a card on the tinted column rather than a bordered
   box with a title cut into its edge -- see gui.steps.base.section. */
QWidget#settingsSection {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: {RADIUS}px;
}}
QLabel#sectionTitle {{
    color: {t["accent"]};
    border-bottom: 1px solid {t["accent_border"]};
    padding-bottom: 5px;
}}
/* Sub-divider inside a section: lighter than the card's own title. */
QLabel#sectionHeader {{
    color: {t["text_muted"]};
    margin-top: 10px;
}}

/* Read-only output sits back from the inputs above it. */
QPlainTextEdit[role="output"], QTextEdit[role="output"] {{
    background: {t["surface_sunken"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
}}

/* The panel edge is draggable; the accent says so on approach. */
QSplitter::handle:hover {{
    background: {t["accent_border"]};
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

/* Three tiers: one filled primary per step, accent-outlined secondaries for
   the other real actions, muted quiet ones for exports. */
QPushButton[variant="primary"] {{
    color: {t["accent_text"]};
    background: {t["accent_deep"]};
    border: 1px solid {t["accent_deep"]};
}}
QPushButton[variant="primary"]:hover {{
    background: {t["accent_hover"]};
    border-color: {t["accent_hover"]};
}}
QPushButton[variant="primary"]:disabled {{
    color: {t["text_muted"]};
    background: transparent;
    border: 1px solid {t["border"]};
}}
QPushButton[variant="secondary"] {{
    color: {t["accent"]};
    background: transparent;
    border: 1px solid {t["accent_border"]};
}}
QPushButton[variant="secondary"]:hover {{
    background: {t["accent_subtle"]};
    border-color: {t["accent"]};
}}
QPushButton[variant="secondary"]:disabled {{
    color: {t["text_muted"]};
    border-color: {t["border"]};
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
    background: {t["accent_deep"]};
    border-color: {t["accent_deep"]};
}}
QWidget#segmented QToolButton:!checked:hover {{
    background: rgba(128, 128, 128, 20%);
}}

/* Files and sets: the cursor row is marked with an accent bar rather than a
   full highlight, so it cannot be mistaken for a checked state. */
QTreeWidget#filesList::item {{
    border-left: 2px solid transparent;
    padding: 3px 4px;
}}
QTreeWidget#filesList::item:selected {{
    background: {t["accent_subtle"]};
    border-left-color: {t["accent"]};
    color: {t["text"]};
}}
QTreeWidget#filesList::item:hover:!selected {{
    background: {t["accent_subtle"]};
}}
QLabel#fileGroupHeader {{
    color: {t["accent"]};
}}

QTableWidget#peaksTable {{
    background: {t["surface_sunken"]};
    gridline-color: {t["border"]};
}}
QTableWidget#peaksTable QHeaderView::section {{
    background: {t["accent_subtle"]};
    color: {t["accent"]};
    border: none;
    border-bottom: 1px solid {t["accent_border"]};
    padding: 3px 6px;
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

/* The on-plot action card. Themed rather than a grey wash: a translucent grey
   dimmed the data under it and read as a smudge on both palettes. The
   #figureOverlay scope is load-bearing -- unscoped, these bare QToolButton
   rules would also restyle the sweep pager arrows and the status-bar theme
   toggle. The step bar draws itself; see gui.stepper. */
QWidget#figureOverlay {{
    background: {t["surface_alt"]};
    border: 1px solid {t["border"]};
    border-radius: {RADIUS}px;
}}
QWidget#figureOverlay QToolButton {{
    background: transparent;
    color: {t["text"]};
    border: none;
    border-radius: 4px;
    padding: 4px 10px;
    text-align: center;
}}
QWidget#figureOverlay QToolButton:hover {{
    background: {t["accent_subtle"]};
    color: {t["accent"]};
}}
QWidget#figureOverlay QToolButton:pressed {{
    background: {t["accent_soft"]};
}}
/* Only the eraser is checkable, and it locks the rest of the window while on,
   so it is filled rather than tinted -- the state has to be unmissable. */
QWidget#figureOverlay QToolButton:checked {{
    background: {t["accent_deep"]};
    color: {t["accent_text"]};
}}
QWidget#figureOverlay QToolButton:disabled {{
    color: {t["text_muted"]};
}}
/* The fold-away chevron. Not inside #figureOverlay: it sits in the axis strip
   below the plot, so it needs the card's own surface to read against the pane
   rather than inheriting the buttons' transparent one. Slim and quiet -- it is
   a handle, not a fifth action. */
QToolButton#figureOverlayCollapse {{
    background: {t["surface_alt"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 0px 8px;
    min-height: 13px;
    color: {t["text_muted"]};
    font-size: 10px;
}}
QToolButton#figureOverlayCollapse:hover {{
    background: {t["accent_subtle"]};
    color: {t["accent"]};
}}
/* Checked means "folded", which is a resting state, not an armed one -- so it
   must not pick up the filled treatment the eraser uses. */
QToolButton#figureOverlayCollapse:checked {{
    background: {t["surface_alt"]};
    color: {t["text_muted"]};
}}
QToolButton#figureOverlayCollapse:checked:hover {{
    background: {t["accent_subtle"]};
    color: {t["accent"]};
}}
QToolButton#figureOverlayCollapse:disabled {{
    color: {t["text_muted"]};
}}
"""
