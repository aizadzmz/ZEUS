"""Confirmation dialog for the generic txt/csv importer.

Shown when neither the Modulo Bat nor the standard BioLogic parser
recognizes a file. Displays the sniffed headers and a few sample rows,
pre-selects columns using the automatic header-based guess from
core.generic_parser, and lets the user correct the mapping (including
the Im/Phase sign) before the file is actually parsed.
"""
from typing import Dict, Optional, Sequence

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class GenericImportDialog(QDialog):
    """Lets the user confirm/correct the column-role guess for a generic
    txt/csv EIS export before it's parsed."""

    def __init__(
        self,
        filename: str,
        headers: Sequence[str],
        sample_rows: Sequence[Sequence[str]],
        guessed_roles: Dict[str, int],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Confirm column mapping")
        self.resize(640, 460)
        self._headers = list(headers)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"'{filename}' doesn't match a known export format.\n"
            "Please manually map the column."
        ))
        layout.addWidget(self._build_preview_table(sample_rows))

        self._freq_combo = self._make_combo(guessed_roles.get("frequency"))

        # Prefer Re/Im mode unless the guess only found mag/phase columns.
        has_re_im = "re" in guessed_roles or "im" in guessed_roles or "neg_im" in guessed_roles
        has_mag_phase = "mag" in guessed_roles or "phase" in guessed_roles or "neg_phase" in guessed_roles
        use_mag_phase = has_mag_phase and not has_re_im

        self._re_im_radio = QRadioButton("Real / Imaginary  (Z', Z'')")
        self._mag_phase_radio = QRadioButton("Magnitude / Phase  (|Z|, θ)")
        self._re_im_radio.setChecked(not use_mag_phase)
        self._mag_phase_radio.setChecked(use_mag_phase)
        self._re_im_radio.toggled.connect(self._update_mode)

        self._re_combo = self._make_combo(guessed_roles.get("re"))
        self._im_combo = self._make_combo(guessed_roles.get("im", guessed_roles.get("neg_im")))
        self._im_negate = QCheckBox("Column already stores -Im(Z) (negated)")
        self._im_negate.setChecked("neg_im" in guessed_roles)

        self._mag_combo = self._make_combo(guessed_roles.get("mag"))
        self._phase_combo = self._make_combo(guessed_roles.get("phase", guessed_roles.get("neg_phase")))
        self._phase_negate = QCheckBox("Column already stores -Phase (negated)")
        self._phase_negate.setChecked("neg_phase" in guessed_roles)

        form = QFormLayout()
        form.addRow("Frequency:", self._freq_combo)
        form.addRow(self._re_im_radio)
        form.addRow("Z' [Re(Z)]:", self._re_combo)
        form.addRow("Z'' [Im(Z)]:", self._wrap(self._im_combo, self._im_negate))
        form.addRow(self._mag_phase_radio)
        form.addRow("|Z| (Mag):", self._mag_combo)
        form.addRow("Phase:", self._wrap(self._phase_combo, self._phase_negate))
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_mode()

    def _make_combo(self, selected_index: Optional[int]) -> QComboBox:
        combo = QComboBox()
        combo.addItem("(none)", None)
        for i, header in enumerate(self._headers):
            combo.addItem(header, i)
        if selected_index is not None:
            combo.setCurrentIndex(selected_index + 1)
        return combo

    @staticmethod
    def _wrap(combo: QComboBox, checkbox: QCheckBox) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(combo, stretch=1)
        row.addWidget(checkbox)
        return w

    def _build_preview_table(self, sample_rows: Sequence[Sequence[str]]) -> QTableWidget:
        table = QTableWidget(len(sample_rows), len(self._headers))
        table.setHorizontalHeaderLabels(self._headers)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setMaximumHeight(140)
        for r, row in enumerate(sample_rows):
            for c, value in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        return table

    def _update_mode(self) -> None:
        re_im = self._re_im_radio.isChecked()
        self._re_combo.setEnabled(re_im)
        self._im_combo.setEnabled(re_im)
        self._im_negate.setEnabled(re_im)
        self._mag_combo.setEnabled(not re_im)
        self._phase_combo.setEnabled(not re_im)
        self._phase_negate.setEnabled(not re_im)

    def _on_accept(self) -> None:
        if self._freq_combo.currentData() is None:
            QMessageBox.warning(self, "Missing column", "Select a Frequency column.")
            return
        if self._re_im_radio.isChecked():
            if self._re_combo.currentData() is None or self._im_combo.currentData() is None:
                QMessageBox.warning(
                    self, "Missing column", "Select both Z' (Re) and Z'' (Im) columns."
                )
                return
        else:
            if self._mag_combo.currentData() is None or self._phase_combo.currentData() is None:
                QMessageBox.warning(
                    self, "Missing column", "Select both |Z| (Mag) and Phase columns."
                )
                return
        self.accept()

    def column_roles(self) -> Dict[str, int]:
        """Build the column_roles mapping expected by parse_generic_file."""
        roles: Dict[str, int] = {"frequency": self._freq_combo.currentData()}
        if self._re_im_radio.isChecked():
            roles["re"] = self._re_combo.currentData()
            im_key = "neg_im" if self._im_negate.isChecked() else "im"
            roles[im_key] = self._im_combo.currentData()
        else:
            roles["mag"] = self._mag_combo.currentData()
            phase_key = "neg_phase" if self._phase_negate.isChecked() else "phase"
            roles[phase_key] = self._phase_combo.currentData()
        return roles
