"""Saving a plot as an image: that both offered formats actually write a file,
and that the name the dialog returns names one of them.
"""
import os
import xml.dom.minidom

# Must precede any Qt import; the suite runs without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.plotting import build_bode_plot, build_nyquist_plot
from gui.figure_panes import PgFigurePane

app = QApplication.instance() or QApplication([])

f = np.logspace(4, -1, 25)
Z = 10 + 50 / (1 + 1j * 2 * np.pi * f * 50 * 1e-4)


def dataset(index=0, file_id=0, source="stub"):
    return EISDataset(DataSet(f, Z), index=index, source_file=source, file_id=file_id)


def pane_showing(widget):
    pane = PgFigurePane()
    pane.set_widget(widget)
    return pane


PLOTS = {
    "nyquist": lambda: build_nyquist_plot([dataset()], title="t"),
    "bode": lambda: build_bode_plot([dataset()], title="t"),
    # Two files, so the grouped legend headers are in the scene as well.
    "multi-file": lambda: build_nyquist_plot(
        [dataset(0, 0, "A"), dataset(1, 1, "B")], title="t"
    ),
}


@pytest.mark.parametrize("name", sorted(PLOTS))
def test_svg_export_writes_a_readable_drawing(tmp_path, name):
    """pyqtgraph 0.14.0's exporter splits every path token on a comma and
    unpacks two numbers, so the closepath Qt ends a filled rectangle with used
    to abort the export -- and every plot here has one, around the ViewBox.
    See core._pyqtgraph_patches."""
    path = tmp_path / f"{name}.svg"
    pane_showing(PLOTS[name]()).save_image(str(path))

    assert path.stat().st_size > 2000, path.stat().st_size
    assert xml.dom.minidom.parse(str(path)).documentElement.tagName == "svg"

    text = path.read_text(encoding="utf-8")
    paths = [
        node.getAttribute("d")
        for node in xml.dom.minidom.parseString(text).getElementsByTagName("path")
    ]
    assert paths, "no path data in the exported drawing"
    # The closepath is put back after the coordinates are rewritten, rather than
    # dropped to get past the exporter.
    assert any(d.rstrip().endswith(("Z", "z")) for d in paths)
    # And every remaining token is still the coordinate pair it was.
    for d in paths:
        for token in d.split(" "):
            if token and token not in ("Z", "z"):
                assert len(token.split(",")) == 2, (d, token)


@pytest.mark.parametrize("name", sorted(PLOTS))
def test_raster_export_still_writes(tmp_path, name):
    path = tmp_path / f"{name}.png"
    pane_showing(PLOTS[name]()).save_image(str(path))
    assert path.stat().st_size > 2000, path.stat().st_size


# --- what the save dialog hands back ---------------------------------------
# The exporter reads the format off the suffix and writes nothing at all when
# there is none it knows -- silently, so the status bar reported a saved file
# that was never there.

@pytest.mark.parametrize(
    "typed, chosen, expected",
    [
        ("plot", "PNG image (*.png)", "plot.png"),
        ("plot", "SVG image (*.svg)", "plot.svg"),
        # A suffix the user typed wins over a dropdown they may never have opened.
        ("plot.svg", "PNG image (*.png)", "plot.svg"),
        ("plot.PNG", "SVG image (*.svg)", "plot.PNG"),
        ("plot.jpg", "PNG image (*.png)", "plot.jpg"),
        # Appended, not substituted: with_suffix would make this "cell A v1.png".
        ("cell A v1.2", "PNG image (*.png)", "cell A v1.2.png"),
        ("plot.dat", "PNG image (*.png)", "plot.dat.png"),
    ],
)
def test_chosen_image_path(qt_window, typed, chosen, expected):
    assert qt_window._chosen_image_path(typed, chosen).name == expected


def test_an_extensionless_name_really_writes_a_file(tmp_path, qt_window):
    path = qt_window._chosen_image_path(str(tmp_path / "noext"), "PNG image (*.png)")
    pane_showing(PLOTS["nyquist"]()).save_image(str(path))
    assert (tmp_path / "noext.png").stat().st_size > 2000


@pytest.fixture(scope="module")
def qt_window():
    from gui.main_window import MainWindow

    window = MainWindow()
    yield window
    window.close()
