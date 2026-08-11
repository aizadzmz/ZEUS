"""The layout every step shares: settings down the left, plots on the right."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
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

    # The mode a step is fixed in when it never adds the Singular/Multiple
    # toggle. Only ECMStep does that: its circuit diagram and parameter report
    # describe one sweep, so offering to overlay the selection promised a
    # comparison the rest of the step could not show.
    fixed_display_mode = "Combined"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Set here rather than only in display_mode_control, so a step that
        # never calls it still answers display_mode without an AttributeError.
        self.single_radio = None
        self.combined_radio = None

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
        # Content-side widgets that grey out with the settings column; see
        # lock_with_settings.
        self._locked_extras: List[QWidget] = []

    # Interaction lock

    def lock_with_settings(self, widget: QWidget) -> None:
        """Register a content-column widget that greys out alongside the
        settings, i.e. everything on the page that is not a plot."""
        self._locked_extras.append(widget)

    def set_controls_locked(self, locked: bool) -> None:
        """Leave the plots live and disable everything else. Disabling the
        containers, not their children, so unlocking restores whatever each
        control's own state was (a Run button greyed out mid-run stays so)."""
        self.settings_scroll.setEnabled(not locked)
        for widget in self._locked_extras:
            widget.setEnabled(not locked)

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

    def display_mode_control(self, tooltip: str = "") -> SegmentedControl:
        """This step's singular/multiple toggle, uncarded -- for a step that
        wants it as one row of a card it shares with other plot settings.
        add_display_mode_box() is the same control in a card of its own."""
        segmented = SegmentedControl(["Singular", "Multiple"])
        if tooltip:
            segmented.setToolTip(tooltip)
        # Named *_radio though they are QToolButtons: same QAbstractButton API,
        # and MainWindow._wire_steps reaches them by these names.
        self.single_radio = segmented.button(0)
        self.single_radio.setToolTip(
            "Display one sweep at a time controlled by ‹ ›."
        )
        self.combined_radio = segmented.button(1)
        self.combined_radio.setToolTip("Display all active sweeps.")
        return segmented

    def add_display_mode_box(
        self, title: str = "Plot view", tooltip: str = ""
    ) -> QWidget:
        """Add this step's own singular/multiple toggle. Each step keeps an
        independent one."""
        container, col = group_box(title)
        col.addWidget(self.display_mode_control(tooltip))
        self.add_settings(container)
        return container

    @property
    def display_mode(self) -> str:
        """"Single" or "Combined" for this step alone. A step without the
        toggle reports its fixed_display_mode instead."""
        if self.single_radio is None:
            return self.fixed_display_mode
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


def _section(title: str, tooltip: str) -> Tuple[QWidget, QVBoxLayout]:
    """A settings card: accent title over a rule, then the caller's content.
    Replaces QGroupBox, whose title-cut-into-the-border look is the strongest
    remaining pyDRTtools tell in this column."""
    card = QWidget()
    card.setObjectName("settingsSection")
    # Never taller than its content. A card holding an Expanding child that is
    # also height-capped (a list, a read-only text pane) otherwise claims the
    # column's spare height, and since the capped child cannot use it the space
    # lands on the title label, which grows and centres its text.
    # end_settings()'s stretch is what the surplus is for.
    card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    outer = QVBoxLayout(card)
    outer.setContentsMargins(*style.GROUP_MARGINS)
    outer.setSpacing(style.GROUP_SPACING)

    if title:
        label = QLabel(title)
        label.setObjectName("sectionTitle")
        label.setFont(style.section_title_font())
        outer.addWidget(label)
    if tooltip:
        card.setToolTip(tooltip)
    return card, outer


def quiet_group() -> Tuple[QWidget, QVBoxLayout]:
    """Un-carded container for exports, which sit outside the settings cards
    rather than claiming a titled section of their own."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(style.GROUP_SPACING)
    return box, layout


def group_box(title: str, tooltip: str = "") -> Tuple[QWidget, QVBoxLayout]:
    """A settings card whose inner layout already carries the spacing tokens."""
    return _section(title, tooltip)


def group_form(title: str, tooltip: str = "") -> Tuple[QWidget, QFormLayout]:
    """A settings card laid out as a two-column form: label beside control,
    which halves the height of a settings panel."""
    box, outer = _section(title, tooltip)
    form = QFormLayout()
    outer.addLayout(form)
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(style.FORM_H_SPACING)
    form.setVerticalSpacing(style.FORM_V_SPACING)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    # Combos and line edits fill the field column rather than sitting at their
    # size hint, so a wider panel gives the controls the room.
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    return box, form


def set_row_enabled(form: QFormLayout, field: QWidget, enabled: bool) -> None:
    """Grey out a form row, label included -- disabling the field alone leaves
    its label at full contrast, which reads as an enabled control."""
    field.setEnabled(enabled)
    label = form.labelForField(field)
    if label is not None:
        label.setEnabled(enabled)


def set_row_visible(form: QFormLayout, field: QWidget, visible: bool) -> None:
    """Show or hide a form row, label included. For settings a mode makes
    meaningless rather than merely inapplicable -- greying out (set_row_enabled)
    says 'not right now', hiding says 'not in this mode'."""
    # setRowVisible, not setVisible on the widgets: hiding them individually
    # leaves the row's slot in the grid and so a gap where it used to be.
    form.setRowVisible(field, visible)


def set_row_label(form: QFormLayout, field: QWidget, text: str) -> None:
    """Retitle a form row whose meaning depends on another setting."""
    label = form.labelForField(field)
    if label is not None:
        label.setText(text)


def set_row_tooltip(form: QFormLayout, field: QWidget, text: str) -> None:
    """Put one tooltip on a row's field and on its label. A label built from an
    addRow() string carries no tooltip of its own, so hovering the row's name
    otherwise falls through to the settings card's, which explains the card
    rather than the setting under the cursor."""
    field.setToolTip(text)
    label = form.labelForField(field)
    if label is not None:
        label.setToolTip(text)


def mirror_row_tooltips(form: QFormLayout) -> None:
    """Copy each row's field tooltip onto its label, for a form whose fields
    were given tooltips as they were built. Call once, after the last row.
    Rows whose label already has its own tooltip are left alone."""
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.FieldRole)
        field = item.widget() if item is not None else None
        if field is None or not field.toolTip():
            continue
        label = form.labelForField(field)
        if label is not None and not label.toolTip():
            label.setToolTip(field.toolTip())


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


def section_label(text: str, tooltip: str = "") -> QLabel:
    """A small uppercase divider inside a long form, used instead of adding
    group boxes. Without a tooltip of its own it inherits the card's, so give
    one to any header whose name is not self-explanatory."""
    label = QLabel(text)
    label.setObjectName("sectionHeader")
    label.setFont(style.section_font())
    if tooltip:
        label.setToolTip(tooltip)
    return label
