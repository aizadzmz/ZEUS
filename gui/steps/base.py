"""The layout every step shares: settings down the left, plots on the right."""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import style
from gui.segmented import SegmentedControl

# Sized to fit a two-column form: at 10pt the widest label is ~223px, so a row
# needs ~425px including the form gap, a usable spinbox, margins, and a
# scrollbar. Labels are kept short to stay under this -- pyDRTtools' full names
# would push it past 550 and eat into the plots. Section headers carry the
# context, and every full name lives in its control's tooltip.
DEFAULT_SETTINGS_WIDTH = 430
MIN_SETTINGS_WIDTH = 380   # below this, the longer labels start wrapping
MAX_SETTINGS_WIDTH = 620   # beyond this the panel starts eating the plots


class StepPage(QWidget):
    """A resizable scrollable settings column beside a content area."""

    # This step's splitter was dragged. MainWindow mirrors the width onto the
    # other steps and persists it, so the panel edge does not jump between them.
    settings_width_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        # Named so the stylesheet can tint the settings column, separating it
        # from the plot area without a hard border down the middle.
        panel.setObjectName("settingsPanelBody")
        self._settings = QVBoxLayout(panel)
        self._settings.setContentsMargins(*style.PANEL_MARGINS)
        self._settings.setSpacing(style.PANEL_SPACING)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setObjectName("settingsPanel")
        self.settings_scroll.setWidget(panel)
        self.settings_scroll.setWidgetResizable(True)
        # No sunken frame: the stylesheet's tint already separates the column.
        self.settings_scroll.setFrameShape(QFrame.NoFrame)
        self.settings_scroll.setMinimumWidth(MIN_SETTINGS_WIDTH)
        self.settings_scroll.setMaximumWidth(MAX_SETTINGS_WIDTH)
        # The form layout wraps long rows, so a horizontal bar is never useful.
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        self._content = QVBoxLayout(content)
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(0)

        # Underscored: ValidationStep, DRTStep and ECMStep each define their own
        # vertical `self.splitter`, which would shadow this one and point the
        # settings-width logic at a splitter measuring heights.
        self._page_splitter = QSplitter(Qt.Horizontal)
        self._page_splitter.addWidget(self.settings_scroll)
        self._page_splitter.addWidget(content)
        # Window resizes grow the plots, not the settings.
        self._page_splitter.setStretchFactor(0, 0)
        self._page_splitter.setStretchFactor(1, 1)
        # Non-collapsible: a pane dragged shut looks like lost content.
        self._page_splitter.setCollapsible(0, False)
        self._page_splitter.setCollapsible(1, False)
        self._page_splitter.splitterMoved.connect(self._on_splitter_moved)
        root.addWidget(self._page_splitter)

        self._wanted_width = DEFAULT_SETTINGS_WIDTH

    # Panel width

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        self._wanted_width = self.settings_width()
        self.settings_width_changed.emit(self._wanted_width)

    def settings_width(self) -> int:
        return self._page_splitter.sizes()[0]

    def set_settings_width(self, width: int) -> None:
        """Set the settings column's width, without emitting."""
        self._wanted_width = width
        self._apply_wanted_width()

    def _apply_wanted_width(self) -> None:
        total = self._page_splitter.width()
        if total <= 0:
            return
        self._page_splitter.setSizes([self._wanted_width, max(1, total - self._wanted_width)])

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_wanted_width()
        # Again once the event loop settles: a page is shown at its *previous*
        # geometry and resized right after, so the call above can be sizing
        # against a stale width and get clamped by the content minimum.
        QTimer.singleShot(0, self._apply_wanted_width)

    def resizeEvent(self, event) -> None:
        """Re-assert the settings width whenever this page is given geometry."""
        super().resizeEvent(event)
        self._apply_wanted_width()

    # Display mode

    def add_display_mode_box(
        self, title: str = "Plot view", tooltip: str = ""
    ) -> QWidget:
        """Add this step's own one-at-a-time/combined toggle. Each step keeps
        an independent one."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(style.GROUP_SPACING)
        col.addWidget(section_label(title))

        segmented = SegmentedControl(["One at a time", "Combined"])
        if tooltip:
            segmented.setToolTip(tooltip)
        # Named *_radio though they are QToolButtons: same QAbstractButton API,
        # and MainWindow._wire_steps reaches them by these names.
        self.single_radio = segmented.button(0)
        self.single_radio.setToolTip(
            "Show one sweep, and step between them with the ‹ › controls."
        )
        self.combined_radio = segmented.button(1)
        self.combined_radio.setToolTip("Draw every selected sweep on one figure.")
        col.addWidget(segmented)

        self.add_settings(container)
        return container

    @property
    def display_mode(self) -> str:
        """"Single" or "Combined" for this step alone."""
        return "Single" if self.single_radio.isChecked() else "Combined"

    # Settings column

    def add_settings(self, widget: QWidget) -> None:
        self._settings.addWidget(widget)

    def end_settings(self) -> None:
        """Push the settings to the top. Call once, after the last one --
        without it they spread down the full panel height."""
        self._settings.addStretch()

    # Content column

    def add_content(self, widget: QWidget, stretch: int = 0) -> None:
        self._content.addWidget(widget, stretch)

    def add_content_layout(self, layout: QLayout) -> None:
        self._content.addLayout(layout)


# ------------------------------------------------------------ layout helpers


def group_box(title: str, tooltip: str = "") -> Tuple[QGroupBox, QVBoxLayout]:
    """A group box whose inner layout already carries the spacing tokens."""
    box = QGroupBox(title)
    if tooltip:
        box.setToolTip(tooltip)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(*style.GROUP_MARGINS)
    layout.setSpacing(style.GROUP_SPACING)
    return box, layout


def group_form(title: str, tooltip: str = "") -> Tuple[QGroupBox, QFormLayout]:
    """A group box laid out as a two-column form: label beside control, which
    halves the height of a settings panel."""
    box = QGroupBox(title)
    if tooltip:
        box.setToolTip(tooltip)
    form = QFormLayout(box)
    form.setContentsMargins(*style.GROUP_MARGINS)
    form.setHorizontalSpacing(style.FORM_H_SPACING)
    form.setVerticalSpacing(style.FORM_V_SPACING)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    # Combos and line edits fill the field column rather than sitting at their
    # size hint, so a wider panel gives the controls the room.
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    return box, form


def add_combo_items(combo: QComboBox, pairs) -> None:
    """Populate a QComboBox from (display_text, value) pairs, retrievable via
    combo.currentData()."""
    for display, value in pairs:
        combo.addItem(display, value)


def compact_combo(combo: QComboBox, chars: int = 12) -> QComboBox:
    """Stop a combo's longest item from dictating its width."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)
    return combo


def section_label(text: str) -> QLabel:
    """A small uppercase divider inside a long form, used instead of adding
    group boxes."""
    label = QLabel(text)
    label.setObjectName("sectionHeader")
    label.setFont(style.section_font())
    return label
