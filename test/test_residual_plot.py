"""The residual figure under both conventions, and what it does with the
infinite residual an uncapped component ratio can produce."""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.plotting import build_residuals_plot
from core.validation import RESIDUAL_BY_COMPONENT, RESIDUAL_BY_MODULUS
from gui.figure_panes import PgSingleFigurePane

# Wide enough to run past the arc's apex, so Z'' gets small at the bottom end
# -- which is where the component convention goes off scale on real data.
f = np.logspace(4, -1, 20)

# One decade either side of the apex (~32 Hz for tau = 5 ms), where both parts
# stay well clear of zero and neither convention runs away.
f_benign = np.logspace(2.5, 0.5, 20)


class FitResult:
    """As in test_validation: the three fields relative_residuals reads."""

    def __init__(self, freq, Z_exp, Z_fit):
        self.frequencies = np.asarray(freq, dtype=float)
        self.impedances = np.asarray(Z_fit)
        self.residuals = (np.asarray(Z_exp) - self.impedances) / np.abs(Z_exp)

    def get_residuals_data(self):
        return self.frequencies, self.residuals.real * 100, self.residuals.imag * 100


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _result(zero_imag_at=None):
    """A clean sweep, optionally with one point whose Z'' rounds to zero.

    Note this is the *realistic* case rather than a true division by zero:
    relative_residuals reconstructs Z_exp arithmetically, so a measured 0j
    comes back as ~1e-16 and the residual is enormous but finite. The plot has
    to survive both, which is why the assertions below check magnitude rather
    than isinf."""
    w = 2 * np.pi * f
    Z_exp = 10 + 50 / (1 + 1j * w * 5e-3)
    if zero_imag_at is not None:
        Z_exp[zero_imag_at] = Z_exp[zero_imag_at].real + 0j
    Z_fit = Z_exp - 0.005 * np.abs(Z_exp) * (1 + 1j)  # 0.5% off on both parts
    return FitResult(f, Z_exp, Z_fit)


def _benign():
    """A sweep whose Z' and Z'' both stay well clear of zero, so nothing is off
    scale under either convention."""
    w = 2 * np.pi * f_benign
    Z_exp = 10 + 50 / (1 + 1j * w * 5e-3)
    return FitResult(f_benign, Z_exp, Z_exp - 0.005 * np.abs(Z_exp) * (1 + 1j))


def _series(widget):
    return {item.name(): item for item in widget.getPlotItem().listDataItems() if item.name()}


def _y_range(widget):
    return widget.getPlotItem().getViewBox().viewRange()[1]


def test_series_are_named_for_the_convention(app):
    modulus = _series(build_residuals_plot(_result(), residual_mode=RESIDUAL_BY_MODULUS))
    component = _series(build_residuals_plot(_result(), residual_mode=RESIDUAL_BY_COMPONENT))
    assert "ΔZ' / |Z|" in modulus and "ΔZ'' / |Z|" in modulus
    assert "ΔZ' / |Z'|" in component and "ΔZ'' / |Z''|" in component


def test_the_axis_is_framed_on_the_highest_limit(app):
    """Not on the data: the limit plus 5 points, whatever the residuals do."""
    for kwargs, expected in (
        ({"threshold": 2.0}, 7.0),
        ({"threshold": 5.0, "soft_threshold": 2.0}, 10.0),
        # Order does not matter -- the higher of the two wins.
        ({"threshold": 2.0, "soft_threshold": 8.0}, 13.0),
    ):
        widget = build_residuals_plot(
            _result(zero_imag_at=7), residual_mode=RESIDUAL_BY_COMPONENT, **kwargs
        )
        low, high = _y_range(widget)
        assert high == pytest.approx(expected), (kwargs, high)
        assert low == pytest.approx(-expected), (kwargs, low)


def test_a_runaway_residual_cannot_set_the_scale(app):
    """One point at a vanishing denominator reports ~1e17%; the axis must not
    follow it and flatten everything else onto the zero line."""
    widget = build_residuals_plot(
        _result(zero_imag_at=7), threshold=2.0, residual_mode=RESIDUAL_BY_COMPONENT
    )
    low, high = _y_range(widget)
    assert np.isfinite(low) and np.isfinite(high), (low, high)
    assert high == pytest.approx(7.0), high


def test_a_small_residual_does_not_shrink_the_axis(app):
    """Fixed framing, so paging between a clean sweep and a bad one does not
    move the limit lines."""
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    _, y = _series(widget)["ΔZ' / |Z|"].getData()
    assert np.abs(y).max() < 1.0  # residuals well inside the limit
    assert _y_range(widget)[1] == pytest.approx(7.0)


def test_a_true_infinity_is_handled_the_same_way(app):
    """core.validation maps an exact 0/0 numerator to inf; the plot must cope
    with it as well as with the merely enormous."""
    result = _result()
    result.residuals = result.residuals.copy()
    Z_exp = 10 + 50 / (1 + 1j * 2 * np.pi * f * 5e-3)
    result.impedances = Z_exp.copy()
    result.impedances[7] = Z_exp[7].real + 0j  # fit's Z'' is zero, data's is not

    widget = build_residuals_plot(result, threshold=2.0,
                                  residual_mode=RESIDUAL_BY_COMPONENT)
    assert np.isfinite(_y_range(widget)).all()


def test_runaway_residual_is_marked_off_scale(app):
    """The point was rejected; a gap in the line would read as missing data."""
    widget = build_residuals_plot(
        _result(zero_imag_at=7), threshold=2.0, residual_mode=RESIDUAL_BY_COMPONENT
    )
    marker = _series(widget).get("off scale")
    assert marker is not None, sorted(_series(widget))
    x, y = marker.getData()
    high = _y_range(widget)[1]
    # Pinned inside an edge, not drawn at its own value. More than one is
    # expected: the arc's low-frequency end sends Z'' small there too.
    assert len(y) >= 1
    assert all(0 < abs(v) < high for v in y), y
    # setLogMode has run, so the marker's x values are decades.
    assert np.log10(f[7]) in x


def _with_outlier(percent):
    """The benign sweep with one point knocked `percent` off on the real part."""
    w = 2 * np.pi * f_benign
    Z_exp = 10 + 50 / (1 + 1j * w * 5e-3)
    Z_fit = Z_exp - 0.005 * np.abs(Z_exp) * (1 + 1j)
    Z_fit[4] = Z_exp[4] - (percent / 100.0) * np.abs(Z_exp[4])
    return FitResult(f_benign, Z_exp, Z_fit)


def test_an_outlier_inside_the_headroom_is_drawn_at_its_value(app):
    """A 6% residual against a 2% limit fits in the 7% axis, so it is drawn
    rather than pinned -- the headroom is there to be used."""
    widget = build_residuals_plot(_with_outlier(6.0), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    _, y = _series(widget)["ΔZ' / |Z|"].getData()
    assert abs(y[4]) == pytest.approx(6.0, abs=0.5)
    assert "off scale" not in _series(widget)


def test_an_outlier_past_the_headroom_is_pinned(app):
    """A 20% residual against a 2% limit is 'well over' and nothing more; it
    must not drag the axis out to 20 and flatten the rest."""
    widget = build_residuals_plot(_with_outlier(20.0), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    assert _y_range(widget)[1] == pytest.approx(7.0)

    _, y = _series(widget)["ΔZ' / |Z|"].getData()
    assert np.isnan(y[4]), y[4]
    marker = _series(widget).get("off scale")
    assert marker is not None, sorted(_series(widget))
    assert len(marker.getData()[0]) == 1


def test_no_off_scale_entry_on_a_well_behaved_sweep(app):
    """Neither convention marks anything when both parts stay clear of zero --
    the ceiling must not fire on ordinary data."""
    for mode in (RESIDUAL_BY_MODULUS, RESIDUAL_BY_COMPONENT):
        widget = build_residuals_plot(_benign(), threshold=2.0, residual_mode=mode)
        assert "off scale" not in _series(widget), mode
        for _, item in _series(widget).items():
            assert np.isfinite(item.getData()[1]).all(), mode


def test_the_drawn_curve_breaks_at_the_off_scale_point(app):
    """NaN in its place, so the line stops rather than running off the top."""
    widget = build_residuals_plot(
        _result(zero_imag_at=7), threshold=2.0, residual_mode=RESIDUAL_BY_COMPONENT
    )
    _, y = _series(widget)["ΔZ'' / |Z''|"].getData()
    high = _y_range(widget)[1]
    assert np.isnan(y[7]), y[7]
    # Whatever survived is on scale; NaN elsewhere is another pinned point.
    drawn = y[~np.isnan(y)]
    assert drawn.size and np.isfinite(drawn).all()
    assert (np.abs(drawn) <= high).all()


def test_modulus_plot_is_unchanged_by_a_zero_component(app):
    """Dividing by |Z| never hits the zero denominator, so this figure draws
    exactly as it always did."""
    widget = build_residuals_plot(
        _result(zero_imag_at=7), threshold=2.0, residual_mode=RESIDUAL_BY_MODULUS
    )
    _, y = _series(widget)["ΔZ'' / |Z|"].getData()
    assert np.isfinite(y).all()
    assert "off scale" not in _series(widget)


# -- navigation ------------------------------------------------------------
def test_the_wheel_rescales_y_and_leaves_frequency_alone(app):
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    view_box = widget.getPlotItem().getViewBox()
    assert view_box.state["mouseEnabled"] == [False, True]

    (x0, x1), (y0, y1) = view_box.viewRange()
    view_box.scaleBy((1.0, 0.5))  # what the wheel drives, x masked out
    (nx0, nx1), (ny0, ny1) = view_box.viewRange()

    assert (nx1 - nx0) == pytest.approx(x1 - x0), "frequency axis moved"
    assert (ny1 - ny0) == pytest.approx((y1 - y0) * 0.5), "y did not rescale"


def test_the_right_click_menu_stays_off(app):
    """The wheel is the only navigation offered; the menu would put x back."""
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    assert not widget.getPlotItem().getViewBox().menuEnabled()


# -- the pane the figure lives in -------------------------------------------
def test_the_pane_never_forces_a_scrollbar(app):
    """The old list pane gave each figure a 340px minimum, which put a
    scrollbar on the single figure this step draws."""
    pane = PgSingleFigurePane()
    pane.resize(700, 200)
    pane.show()
    figure = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    pane.set_widget(figure)
    QApplication.processEvents()

    assert figure.minimumHeight() == 0
    assert figure.height() <= 200

    # Squeezed hard, the figure shrinks rather than overflowing.
    pane.resize(700, 90)
    QApplication.processEvents()
    assert pane.minimumSizeHint().height() <= 90
    assert figure.height() <= 90


def test_the_pane_holds_one_figure_and_clears(app):
    pane = PgSingleFigurePane()
    first = build_residuals_plot(_benign(), threshold=2.0)
    pane.set_widget(first)
    assert pane._widget is first

    second = build_residuals_plot(_benign(), threshold=5.0)
    pane.set_widget(second)
    assert pane._widget is second

    # None is how the step blanks it when nothing is validated.
    pane.set_widget(None)
    assert pane._widget is None
