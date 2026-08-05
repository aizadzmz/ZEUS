"""Manual visual check of the Modulo Bat parser -- not a test.

Point SAMPLE_FILE at a local .mpt and run it directly; each plot blocks until
you close its window. Lived in test/ until it was moved here, where pytest
does not try to collect it.
"""

from PySide6.QtWidgets import QApplication

from core.mb_parser import parse_modulobat_file
from core.plotting import build_nyquist_plot

SAMPLE_FILE = r"C:\Users\aizad\Downloads\L07A-MSM_Lifun_FC25_EIS12_SL_pouch_4,20V_2,5V_43mAh_Gr_Ni92_02_MB_CA7.mpt"

datasets = parse_modulobat_file(SAMPLE_FILE)
print(f"Found {len(datasets)} EIS sweep(s)\n")

app = QApplication.instance() or QApplication([])


def show(widget, title: str) -> None:
    """Show a plot widget and block until it's closed, mirroring matplotlib's
    plt.show() for these manual visual checks."""
    widget.setWindowTitle(title)
    widget.resize(800, 600)
    widget.show()
    app.exec()


# --- Show the first extracted EIS sweep ---
show(build_nyquist_plot([datasets[13]], style="line"), datasets[13].full_label)

# --- Overlay all extracted sweeps ---
# show(build_nyquist_plot(datasets, title="Modulo Bat EIS sweeps", style="line"), "Modulo Bat EIS sweeps")
