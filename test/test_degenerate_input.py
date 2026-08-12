"""What the plots and exports do with numbers that have no logarithm, no
percentage, or no JSON spelling.

Every case here is one a real file can carry: instruments pad a sweep out with
zeroed rows, a fit that never converged reports a NaN, and a parameter can
settle on exactly the zero its lower limit allows. None of them is interesting
in itself -- what matters is that none of them takes down a plot, an export or
a saved session.
"""
import os

# Must precede any Qt import; the suite runs without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.plotting import build_bode_plot, build_nyquist_plot

app = QApplication.instance() or QApplication([])


def sweep(zero_at=None, frequencies=None):
    f = np.logspace(4, -1, 25)
    Z = 10 + 50 / (1 + 1j * 2 * np.pi * f * 50 * 1e-4)
    if zero_at is not None:
        Z[zero_at] = 0 + 0j
    if frequencies is not None:
        f = np.asarray(frequencies, dtype=float)
    return EISDataset(DataSet(f, Z), index=0, source_file="stub", file_id=0)


# --- log10 of zero must not reach the ViewBox -------------------------------
# pyqtgraph raises outright on a non-finite range, and both builders frame
# themselves explicitly rather than autoranging, so one such point used to take
# the whole step down -- from a Qt slot, so it surfaced as a crash dialog and
# came back on every redraw until the view was switched away.

@pytest.mark.parametrize("build", [build_bode_plot, build_nyquist_plot])
@pytest.mark.parametrize(
    "name, dataset",
    [
        ("|Z| = 0 at one point", sweep(zero_at=7)),
        ("|Z| = 0 at the first point", sweep(zero_at=0)),
        ("|Z| = 0 throughout", EISDataset(
            DataSet(np.logspace(3, 1, 8), np.zeros(8, dtype=complex)),
            index=0, source_file="stub")),
        ("f = 0 at one point", sweep(
            frequencies=[1e4, 1e3, 1e2, 0.0] + list(np.logspace(0, -1, 21)))),
    ],
)
def test_a_point_with_no_logarithm_still_draws(build, name, dataset):
    widget = build([dataset], title="t")
    for attribute in ("kept_range", "full_range"):
        framing = getattr(widget, attribute)
        # None is a legitimate answer (nothing finite to frame); a range with an
        # infinity in it is not, and is what pyqtgraph refuses.
        assert framing is None or all(np.isfinite(v) for v in framing), (
            name, attribute, framing
        )


def test_a_normal_sweep_is_framed_as_before():
    """The screen above must not have cost ordinary data its framing."""
    widget = build_bode_plot([sweep()], title="t")
    xlo, xhi, ylo, yhi = widget.full_range
    # 25 points from 1e4 down to 1e-1 Hz, drawn in decades.
    assert -1.5 < xlo < -0.9 and 3.9 < xhi < 4.5, (xlo, xhi)
    # |Z| runs from about 10 to about 60 ohm, so log10 spans roughly 1 to 1.8.
    assert 0.9 < ylo < 1.05 and 1.7 < yhi < 1.9, (ylo, yhi)


# --- a percentage of nothing ------------------------------------------------

def test_a_zero_valued_parameter_does_not_abort_the_export(tmp_path):
    """pyimpspec divides by the value to report a relative error, and a
    resistor's lower limit is exactly 0. One such parameter used to take out the
    whole batch, every other sweep included."""
    from core.bdf_export import write_ecm_parameters

    class Parameter:
        value, stderr, unit, fixed = 0.0, float("nan"), "ohm", False

        def get_relative_error(self):
            return self.stderr / self.value  # ZeroDivisionError, as upstream

    class Fit:
        pseudo_chisqr = 0.1

        def get_parameters(self):
            return {"R_1": {"R": Parameter()}}

    path = write_ecm_parameters(tmp_path / "p.csv", {"R": Fit()})
    row = path.read_text(encoding="utf-8").splitlines()[1].split(",")
    # The value is still reported; only the percentage of it is blank.
    assert row[3] == "0.0", row
    assert row[5] == "", row


# --- numbers JSON cannot hold ----------------------------------------------

class _Result:
    """The shape core.session reads off a TRRBFResult."""

    def __init__(self, pseudo_chisqr):
        k = 8
        self.time_constants = np.linspace(1.0, 2.0, k)
        self.frequencies = np.linspace(1.0, 2.0, k)
        self.impedances = np.ones(k, dtype=complex)
        self.residuals = np.zeros(k, dtype=complex)
        self.pseudo_chisqr = pseudo_chisqr
        self.gammas = np.ones(k)
        self.mean_gammas = np.array([])
        self.lower_bounds = np.array([])
        self.upper_bounds = np.array([])
        self.lambda_value = 1e-3


@pytest.fixture
def registered_stub():
    """core.session dispatches on the result's type, so the stub needs a kind."""
    import core.session as session

    session._DRT_KIND[_Result] = "tr_rbf"
    yield
    del session._DRT_KIND[_Result]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_result_can_still_be_saved(tmp_path, registered_stub, bad):
    """A run that failed to converge is the one most worth keeping, so it must
    not be the thing that makes the whole session unsaveable."""
    from core.session import load_session, save_session

    dataset = sweep()
    path = tmp_path / "s.eisz"
    save_session(path, [dataset], {}, {"0:0": _Result(bad)}, {})
    restored = load_session(path)[2]["0:0"]
    # JSON has no infinity either, so both come back as the NaN they stood for.
    assert np.isnan(restored.pseudo_chisqr), restored.pseudo_chisqr


def test_non_finite_measurements_survive_a_round_trip(tmp_path):
    from core.session import load_session, save_session

    f = np.logspace(4, -1, 10)
    Z = (10 + 50 / (1 + 1j * 2 * np.pi * f * 50 * 1e-4)).astype(complex)
    Z[3] = complex(np.nan, np.nan)
    dataset = EISDataset(DataSet(f, Z), index=0, source_file="stub")

    path = tmp_path / "n.eisz"
    save_session(path, [dataset], {}, {}, {})
    restored = load_session(path)[0][0].data.get_impedances(masked=None)
    assert np.isnan(restored[3]), restored[3]
    assert np.allclose(restored[[0, 1, 2, 4]], Z[[0, 1, 2, 4]])


def test_numpy_scalars_in_stored_parameters(tmp_path, registered_stub):
    """The params dicts are recorded verbatim from the widgets; np.bool_ is not
    a bool and json refuses it outright."""
    from core.session import load_session, save_session

    path = tmp_path / "p.eisz"
    save_session(
        path, [sweep()], {}, {"0:0": _Result(0.01)}, {},
        drt_params={"0:0": {
            "inductance": np.bool_(True),
            "shape_coeff": np.float64(0.5),
            "lambda_value": float("nan"),
            "pruned_points": np.array([1, 2, 3]),
        }},
    )
    params = load_session(path)[5]["0:0"]
    assert params["inductance"] is True, params
    assert params["shape_coeff"] == 0.5, params
    assert params["lambda_value"] is None, params
    assert params["pruned_points"] == [1, 2, 3], params
