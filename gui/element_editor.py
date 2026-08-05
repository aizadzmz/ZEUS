"""The popup behind a click on a component: its type, parameters and label."""

from __future__ import annotations

import math
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import style
from gui.steps.base import add_combo_items, section_label

# Wide enough for "value / min / max / fixed" on one row without wrapping.
POPUP_WIDTH = 380


def _format(value: float) -> str:
    """A number as something editable, with 'inf' left spelled out."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def _parse(text: str) -> Optional[float]:
    text = text.strip().lower()
    if text in ("inf", "+inf", "infinity"):
        return math.inf
    if text in ("-inf", "-infinity"):
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return None


class _NumberEdit(QLineEdit):
    """A line edit for an EIS-scale quantity. A spin box cannot span 1e-24 to
    1e6, so this validates scientific notation and commits on focus-out."""

    committed = Signal(float)

    def __init__(self, value: float, parent: Optional[QWidget] = None):
        super().__init__(_format(value), parent)
        self._value = value
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.ScientificNotation)
        self.setValidator(validator)
        self.editingFinished.connect(self._commit)

    def _commit(self) -> None:
        parsed = _parse(self.text())
        if parsed is None or parsed == self._value:
            self.setText(_format(self._value))
            return
        self._value = parsed
        self.setText(_format(parsed))
        self.committed.emit(parsed)


class ElementEditor(QFrame):
    """Edits one ElementNode. Emits on every committed change, so the canvas and
    the CDC field follow along as the user types."""

    type_changed = Signal(int, str)        # node_id, symbol
    parameters_changed = Signal(int, dict)  # node_id, set_element kwargs

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.Popup)
        self.setObjectName("elementEditor")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(POPUP_WIDTH)

        self._node_id: Optional[int] = None
        self._symbol = ""
        self._fitted: Dict[str, float] = {}

        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(*style.GROUP_MARGINS)
        self._column.setSpacing(style.GROUP_SPACING)

        self._title = QLabel()
        self._title.setObjectName("sectionHeader")
        self._title.setFont(style.section_font())
        self._column.addWidget(self._title)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type"))
        self._type_combo = QComboBox()
        self._type_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        type_row.addWidget(self._type_combo, stretch=1)
        self._column.addLayout(type_row)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(style.FORM_H_SPACING)
        self._grid.setVerticalSpacing(style.FORM_V_SPACING)
        self._column.addWidget(self._grid_host)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label"))
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("optional, e.g. ct")
        self._label_edit.setToolTip("Names the component on the schematic: R_ct.")
        self._label_edit.editingFinished.connect(self._commit_label)
        label_row.addWidget(self._label_edit, stretch=1)
        self._column.addLayout(label_row)

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        self._column.addWidget(self._status)

        buttons = QHBoxLayout()
        self._adopt_button = QPushButton("Use fitted values")
        self._adopt_button.setProperty("variant", "quiet")
        self._adopt_button.setToolTip(
            "Take this component's fitted values as its starting values."
        )
        self._adopt_button.clicked.connect(self._adopt_fitted)
        buttons.addWidget(self._adopt_button)
        reset_button = QPushButton("Reset")
        reset_button.setProperty("variant", "quiet")
        reset_button.setToolTip("Back to pyimpspec's default values and bounds.")
        reset_button.clicked.connect(self._reset)
        buttons.addWidget(reset_button)
        self._column.addLayout(buttons)

        self._type_combo.currentIndexChanged.connect(self._on_type_changed)

    # ---------------------------------------------------------------- state

    def show_node(self, node, name: str, fitted: Optional[Dict[str, float]], anchor) -> None:
        """Populate for `node` and pop up under the point `anchor` (global)."""
        self._node_id = node.node_id
        self._symbol = node.symbol
        self._fitted = dict(fitted or {})
        self._title.setText(name)
        self._fill_types(node.symbol)
        self._fill_parameters(node)

        self._label_edit.blockSignals(True)
        self._label_edit.setText(node.label)
        self._label_edit.blockSignals(False)
        self._set_status("")

        self._adopt_button.setVisible(bool(self._fitted))
        self.adjustSize()
        self.move(anchor)
        self.show()

    def _fill_types(self, symbol: str) -> None:
        from core.circuit_model import element_symbols

        self._type_combo.blockSignals(True)
        self._type_combo.clear()
        classes = element_symbols()
        add_combo_items(
            self._type_combo,
            [(f"{key} — {cls.get_description()}", key) for key, cls in classes.items()],
        )
        self._type_combo.setCurrentIndex(list(classes).index(symbol))
        self._type_combo.blockSignals(False)

    def _fill_parameters(self, node) -> None:
        from core.circuit_diagram import unit_text

        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for column, heading in enumerate(("", "Value", "Min", "Max", "")):
            if heading:
                self._grid.addWidget(section_label(heading), 0, column)

        units = _element_units(node.symbol)
        for row, symbol in enumerate(node.values, start=1):
            unit = unit_text(units.get(symbol, ""))
            self._grid.addWidget(QLabel(f"{symbol} {unit}".strip()), row, 0)
            self._grid.addWidget(self._number(node, symbol, "values"), row, 1)
            self._grid.addWidget(self._number(node, symbol, "lower"), row, 2)
            self._grid.addWidget(self._number(node, symbol, "upper"), row, 3)

            fixed = QCheckBox("fix")
            fixed.setChecked(bool(node.fixed.get(symbol)))
            fixed.setToolTip("Hold this parameter at its value instead of fitting it.")
            fixed.toggled.connect(
                lambda on, s=symbol: self._emit({"fixed": {s: bool(on)}})
            )
            self._grid.addWidget(fixed, row, 4)

    def _number(self, node, symbol: str, field: str) -> _NumberEdit:
        edit = _NumberEdit(getattr(node, field)[symbol])
        edit.committed.connect(lambda value, s=symbol, f=field: self._emit({f: {s: value}}))
        return edit

    # --------------------------------------------------------------- edits

    def _emit(self, payload: dict) -> None:
        if self._node_id is not None:
            self.parameters_changed.emit(self._node_id, payload)

    def _on_type_changed(self, _index: int) -> None:
        symbol = self._type_combo.currentData()
        if self._node_id is not None and symbol and symbol != self._symbol:
            self._symbol = symbol
            self.type_changed.emit(self._node_id, symbol)

    def _commit_label(self) -> None:
        from core.circuit_model import validate_label

        try:
            label = validate_label(self._label_edit.text())
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return
        self._set_status("")
        self._emit({"label": label})

    def _reset(self) -> None:
        from core.circuit_model import new_element

        fresh = new_element(self._symbol)
        self._emit(
            {
                "values": fresh.values,
                "lower": fresh.lower,
                "upper": fresh.upper,
                "fixed": fresh.fixed,
            }
        )

    def _adopt_fitted(self) -> None:
        if self._fitted:
            self._emit({"values": dict(self._fitted)})

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setVisible(bool(text))
        style.set_state(self._status, "error" if error else "muted")


def _element_units(symbol: str) -> Dict[str, str]:
    from core.circuit_model import element_symbols

    return dict(element_symbols()[symbol].get_units())
