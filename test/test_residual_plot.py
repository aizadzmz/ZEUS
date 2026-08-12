"""The residual figure under both conventions, and what it does with the
infinite residual an uncapped component ratio can produce."""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.plotting import _residual_text, build_residuals_plot, point_tip
from core.validation import RESIDUAL_BY_COMPONENT, RESIDUAL_BY_MODULUS
from gui.figure_panes import PgSingleFigurePane

# Wide enough to run past the arc's apex, so Z'' gets small at the bottom end -- where the component convention goes off scale on real data.
f = np.logspace(4, -1, 20)

# One decade either side of the apex (~32 Hz for tau = 5 ms), where both parts stay well clear of zero.
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
    """A clean sweep, optionally with one point whose Z'' rounds to zero -- the
    *realistic* case rather than a true division by zero, since Z_exp is
    reconstructed arithmetically and the residual is enormous but finite.
    """
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
    assert "ΔZ' / Z'" in component and "ΔZ'' / Z''" in component


def test_the_axis_is_framed_on_the_highest_limit(app):
    """Not on the data: the limit plus 1 point, whatever the residuals do."""
    for kwargs, expected in (
        ({"threshold": 2.0}, 3.0),
        ({"threshold": 5.0, "soft_threshold": 2.0}, 6.0),
        # Order does not matter -- the higher of the two wins.
        ({"threshold": 2.0, "soft_threshold": 8.0}, 9.0),
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
    assert high == pytest.approx(3.0), high


def test_a_small_residual_does_not_shrink_the_axis(app):
    """Fixed framing, so paging between a clean sweep and a bad one does not
    move the limit lines."""
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    _, y = _series(widget)["ΔZ' / |Z|"].getData()
    assert np.abs(y).max() < 1.0  # residuals well inside the limit
    assert _y_range(widget)[1] == pytest.approx(3.0)


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
    # Pinned inside an edge, not drawn at its own value; more than one is expected, the arc's low-frequency end sending Z'' small too.
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
    """A 2.5% residual against a 2% limit fits in the 3% axis, so it is drawn
    rather than pinned -- the headroom is there to be used."""
    widget = build_residuals_plot(_with_outlier(2.5), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    _, y = _series(widget)["ΔZ' / |Z|"].getData()
    assert abs(y[4]) == pytest.approx(2.5, abs=0.4)
    assert "off scale" not in _series(widget)


def test_an_outlier_past_the_headroom_is_pinned(app):
    """A 20% residual against a 2% limit is 'well over' and nothing more; it
    must not drag the axis out to 20 and flatten the rest."""
    widget = build_residuals_plot(_with_outlier(20.0), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    assert _y_range(widget)[1] == pytest.approx(3.0)

    _, y = _series(widget)["ΔZ' / |Z|"].getData()
    assert np.isnan(y[4]), y[4]
    marker = _series(widget).get("off scale")
    assert marker is not None, sorted(_series(widget))
    assert len(marker.getData()[0]) == 1


def test_no_off_scale_entry_on_a_well_behaved_sweep(app):
    """Neither convention marks anything when both parts stay clear of zero --
    the ceiling must not fire on ordinary data. The 6% limit is what puts the
    benign sweep's worst component residual (~6.07%) inside the axis; against
    the 2% limit the 1-point headroom stops well short of it, and pinning is
    then the correct answer."""
    for mode in (RESIDUAL_BY_MODULUS, RESIDUAL_BY_COMPONENT):
        widget = build_residuals_plot(_benign(), threshold=6.0, residual_mode=mode)
        assert "off scale" not in _series(widget), mode
        for _, item in _series(widget).items():
            assert np.isfinite(item.getData()[1]).all(), mode


def test_the_drawn_curve_breaks_at_the_off_scale_point(app):
    """NaN in its place, so the line stops rather than running off the top."""
    widget = build_residuals_plot(
        _result(zero_imag_at=7), threshold=2.0, residual_mode=RESIDUAL_BY_COMPONENT
    )
    _, y = _series(widget)["ΔZ'' / Z''"].getData()
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


class _Wheel:
    """The three members _SymmetricYViewBox.wheelEvent reads off a real
    QGraphicsSceneWheelEvent, which cannot be constructed outside a scene."""

    def __init__(self, delta):
        self._delta = delta
        self.accepted = False
        self.ignored = False

    def delta(self):
        return self._delta

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def test_the_wheel_stretches_about_the_zero_line(app):
    """Anchored on y = 0, not on the cursor: the limit lines have to stay a
    matched distance above and below however far the scale is stretched."""
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    view_box = widget.getPlotItem().getViewBox()

    for delta in (120, -120, 360):
        view_box.wheelEvent(_Wheel(delta))
        low, high = _y_range(widget)
        assert low == pytest.approx(-high), (delta, low, high)

    # ...and it is a real zoom, not a no-op that happens to stay centred.
    before = _y_range(widget)[1]
    view_box.wheelEvent(_Wheel(240))
    assert _y_range(widget)[1] < before


def test_the_wheel_still_leaves_frequency_alone(app):
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    view_box = widget.getPlotItem().getViewBox()
    x_before = view_box.viewRange()[0]

    event = _Wheel(120)
    view_box.wheelEvent(event)

    assert view_box.viewRange()[0] == pytest.approx(x_before)
    assert event.accepted and not event.ignored


def test_the_bottom_axis_wheel_is_ignored(app):
    """axis=0 is the frequency axis' own handler; x is masked out, so there is
    nothing for it to scale and the event must pass on."""
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    view_box = widget.getPlotItem().getViewBox()
    x_before, y_before = view_box.viewRange()

    event = _Wheel(120)
    view_box.wheelEvent(event, axis=0)

    assert event.ignored and not event.accepted
    x_after, y_after = view_box.viewRange()
    assert x_after == pytest.approx(x_before)
    assert y_after == pytest.approx(y_before)


def test_the_right_click_menu_stays_off(app):
    """The wheel is the only navigation offered; the menu would put x back."""
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                  residual_mode=RESIDUAL_BY_MODULUS)
    assert not widget.getPlotItem().getViewBox().menuEnabled()


# -- point metadata ---------------------------------------------------------
def _tips(widget):
    """Every hoverable series' per-point payloads, keyed by series name. Read
    off the scatter's parent PlotDataItem: the name is set on the item plot()
    returns, and the scatter inside it carries the payloads."""
    return {
        item.parentItem().name(): list(item.data["data"])
        for item in widget.interactive_items
    }


def test_every_series_is_hoverable_and_carries_a_payload(app):
    widget = build_residuals_plot(_benign(), threshold=2.0, label="Set 01",
                                 residual_mode=RESIDUAL_BY_MODULUS)
    items = widget.interactive_items
    assert len(items) == 2  # nothing off scale on a benign sweep
    for item in items:
        assert item.opts["hoverable"], item.name()
        # One payload per plotted point, or the tips would be off by one.
        assert len(item.data["data"]) == len(f_benign)


def test_a_point_reports_its_sweep_frequency_and_both_parts(app):
    widget = build_residuals_plot(_benign(), threshold=2.0, label="Set 01",
                                 residual_mode=RESIDUAL_BY_MODULUS)
    data = _tips(widget)["ΔZ' / |Z|"][3]
    text = point_tip(data)

    assert text.startswith("Set: Set 01\n")
    assert f"Freq: {f_benign[3]:.4g} Hz" in text
    # Both parts, as on the Bode plot: a real residual means little without the imaginary one beside it.
    assert "ΔZ' / |Z|: " in text and "ΔZ'' / |Z|: " in text
    assert text.count("%") == 2


def test_the_tip_is_named_for_the_convention(app):
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                 residual_mode=RESIDUAL_BY_COMPONENT)
    text = point_tip(_tips(widget)["ΔZ' / Z'"][3])
    assert "ΔZ' / Z': " in text and "ΔZ'' / Z'': " in text


def test_the_label_falls_back_to_the_title(app):
    widget = build_residuals_plot(_benign(), threshold=2.0, title="K-K — Set 07")
    assert point_tip(_tips(widget)["ΔZ' / |Z|"][0]).startswith("Set: K-K — Set 07")


def test_an_off_scale_point_says_so_and_still_reports_its_value(app):
    """Its marker sits at a made-up y, so the box is the only place the real
    number can be read."""
    widget = build_residuals_plot(
        _with_outlier(20.0), threshold=2.0, label="Set 01",
        residual_mode=RESIDUAL_BY_MODULUS,
    )
    off = _tips(widget)["off scale"]
    assert len(off) == 1
    text = point_tip(off[0])
    assert text.startswith("Set: Set 01 (off scale)")
    assert "20" in text.split("\n")[2]  # the real residual, not the pinned y


def test_an_unusable_residual_reads_as_a_word_not_a_number(app):
    """core.validation._relative_to hands back a signed infinity for a point
    whose component measured a hard zero; '1e+308 %' would say nothing."""
    assert _residual_text(np.inf) == "+∞ %"
    assert _residual_text(-np.inf) == "−∞ %"
    assert _residual_text(np.nan) == "n/a"
    assert _residual_text(-0.51234) == "-0.5123 %"


def test_a_drawn_point_is_not_marked_off_scale(app):
    widget = build_residuals_plot(_benign(), threshold=2.0,
                                 residual_mode=RESIDUAL_BY_MODULUS)
    for data in _tips(widget)["ΔZ' / |Z|"]:
        assert "(off scale)" not in point_tip(data)


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


def _shown_pane(width=760, height=300, **kwargs):
    pane = PgSingleFigurePane()
    pane.resize(width, height)
    pane.show()
    QApplication.processEvents()
    pane.set_widget(build_residuals_plot(
        _benign(), threshold=2.0, label="Set 01",
        residual_mode=RESIDUAL_BY_MODULUS, **kwargs,
    ))
    QApplication.processEvents()
    QApplication.processEvents()
    return pane


def _plot_area(pane):
    return pane._widget.getPlotItem().getViewBox().sceneBoundingRect()


def _card_rect(pane):
    item = pane._tooltip_item
    return item.mapRectToScene(item.boundingRect())


def _assert_inside(pane, where=""):
    card, area = _card_rect(pane), _plot_area(pane)
    assert card.left() >= area.left() - 1, (where, card, area)
    assert card.right() <= area.right() + 1, (where, card, area)
    assert card.top() >= area.top() - 1, (where, card, area)
    assert card.bottom() <= area.bottom() + 1, (where, card, area)


class _Hover:
    """A pyqtgraph hover event, which cannot be constructed outside a scene.
    Dispatched straight to the item, so this goes through ScatterPlotItem's own
    hit test -- the part that does nothing unless `hoverable` was set."""

    def __init__(self, pos=None):
        self.exit = pos is None
        self._pos = pos

    def pos(self):
        return self._pos


def test_hovering_a_residual_point_shows_its_box(app):
    """End to end: the series' sigHovered has to reach the pane, which is what
    interactive_items and the hoverable flag are between them for."""
    pane = _shown_pane()
    scatter = pane._widget.interactive_items[0]

    scatter.hoverEvent(_Hover(scatter.points()[3].pos()))
    QApplication.processEvents()
    assert pane._pending_hover is not None
    assert pane._tooltip_item is None, "shown without waiting out the delay"

    pane._show_pending_hover()  # what the hover timer fires
    QApplication.processEvents()
    assert pane._tooltip_item is not None
    assert "Set: Set 01" in pane._tooltip_item.toPlainText()

    # Hovering off the series takes it away again.
    scatter.hoverEvent(_Hover())
    QApplication.processEvents()
    assert pane._tooltip_item is None


def test_clicking_a_residual_point_pins_its_box(app):
    pane = _shown_pane()
    scatter = pane._widget.interactive_items[0]
    pane._on_point_clicked(scatter, [scatter.points()[3]], None)
    QApplication.processEvents()
    assert pane._pinned and pane._tooltip_item is not None

    # Pinned, so a hover elsewhere leaves it alone...
    pane._on_point_hovered(scatter, [], None)
    assert pane._tooltip_item is not None

    # ...until a click misses every point.
    class _Miss:
        def isAccepted(self):
            return False

    pane._on_scene_clicked(_Miss())
    assert not pane._pinned and pane._tooltip_item is None


def test_the_box_is_not_clipped_at_any_corner(app):
    """The pane is short and has a legend column, so every edge is close."""
    pane = _shown_pane()
    (x0, x1), (y0, y1) = pane._widget.getPlotItem().getViewBox().viewRange()
    ix, iy = (x1 - x0) * 0.02, (y1 - y0) * 0.02
    corners = {
        "bottom-left": (x0 + ix, y0 + iy),
        "bottom-right": (x1 - ix, y0 + iy),
        "top-left": (x0 + ix, y1 - iy),
        "top-right": (x1 - ix, y1 - iy),
    }
    for where, (x, y) in corners.items():
        pane._show_tooltip(x, y, "Set: Set 01\nFreq: 63.1 Hz\nΔZ' / |Z|: 0.5 %"
                                 "\nΔZ'' / |Z|: -0.3 %", pinned=True)
        QApplication.processEvents()
        _assert_inside(pane, where)


def test_the_wheel_re_fits_a_pinned_box(app):
    """Zooming y moves the point the box hangs off; it must not be left under
    an edge."""
    pane = _shown_pane()
    view = pane._widget.getPlotItem().getViewBox()
    (x0, x1), (y0, y1) = view.viewRange()
    pane._show_tooltip((x0 + x1) / 2, y1 * 0.02, "Set: Set 01\nFreq: 63.1 Hz\n"
                                                 "ΔZ' / |Z|: 0.5 %\nΔZ'' / |Z|: -0.3 %",
                       pinned=True)
    QApplication.processEvents()

    view.wheelEvent(_Wheel(-600))  # zoom out hard, so the point runs to zero
    QApplication.processEvents()
    _assert_inside(pane, "after the wheel")


def test_swapping_the_figure_drops_the_box(app):
    """The old box lived on the widget being deleted; a stale reference to it
    would crash the next re-fit."""
    pane = _shown_pane()
    scatter = pane._widget.interactive_items[0]
    pane._on_point_clicked(scatter, [scatter.points()[3]], None)
    assert pane._tooltip_item is not None

    pane.set_widget(build_residuals_plot(_benign(), threshold=5.0))
    QApplication.processEvents()
    assert pane._tooltip_item is None and not pane._pinned
    pane._fit_tooltip_inside_plot()  # must not raise


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
