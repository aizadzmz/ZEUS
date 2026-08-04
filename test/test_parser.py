from pathlib import Path

from PySide6.QtWidgets import QApplication

from core import parse_eis_file
from core.filtering import mask_inductive_points
from core.plotting import build_nyquist_plot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "C:/Users/aizad/Downloads/EIS_for_Aizad/363/cyclage 363 NMC811 Coated 220622 37 vs Gr Cid  LP57+2%VC ICA DVA + IMPEDANCE_05_PEIS_07.mpt"

datasets = parse_eis_file(SAMPLE_FILE)
print(f"Found {len(datasets)} dataset(s)\n")

app = QApplication.instance() or QApplication([])


def show(widget, title: str) -> None:
    """Show a plot widget and block until it's closed, mirroring matplotlib's
    plt.show() for these manual visual checks."""
    widget.setWindowTitle(title)
    widget.resize(800, 600)
    widget.show()
    app.exec()


# --- Default: show first dataset ---
title = datasets[1].full_label
show(build_nyquist_plot([datasets[1]], title=title), title)

# --- Test overlay: show a range e.g. sets 3 to 7 ---
title = "Sets 3 to 7 Overlay"
show(build_nyquist_plot(datasets[2:7], title=title), title)

# --- Test masking: mask inductive points (Im(Z) > 0) on the first dataset ---
mask_inductive_points(datasets[0])
num_masked = datasets[0].data.get_num_points(masked=True)
print(f"Masked {num_masked} inductive point(s) in {datasets[0].label}\n")
title = f"{datasets[0].full_label} (inductive points masked)"
show(build_nyquist_plot([datasets[0]], title=title), title)
