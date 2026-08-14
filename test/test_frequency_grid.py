"""core.frequency_grid: recovering a rounded-off geometric frequency grid.

The Toeplitz shortcut pyimpspec's TR-RBF assembly takes needs the frequencies
ln-spaced to within 1%. A sweep exported at a few significant figures often
misses that by a hair even though it was measured on an exact geometric
progression -- this module recovers the progression, verifying its own
correctness by requiring the recovered grid's rounding to reproduce every
stored value exactly.
"""
import numpy as np
import pytest
from pyimpspec import DataSet
from pyimpspec.analysis.drt import tr_rbf

from core.frequency_grid import (
    TOEPLITZ_LIMIT,
    geometric_grid,
    inspect_grid,
    nonuniformity,
    regridded,
    significant_figures,
)
from core.io_utils import EISDataset


def _quantised_sweep(decades=1, points_per_decade=20, digits=3):
    """Same shape as the demo files this module was built for -- 20
    points/decade, rounded to a few significant figures, which is what pushes
    them over the 1% threshold. A short span keeps the DRT calls below fast
    without changing what is being demonstrated: the point is the rounding,
    not the sweep length."""
    n = decades * points_per_decade + 1
    f = np.logspace(decades, 0, n)
    return np.array([float(f"{v:.{digits}g}") for v in f])


FREQUENCIES = _quantised_sweep()


def _dataset(frequencies, mask=None, source_file="synthetic"):
    w = 2 * np.pi * frequencies
    Z = 10 + 50 / (1 + 1j * w * 5e-3) + 30 / (1 + 1j * w * 0.3)
    return EISDataset(DataSet(frequencies, Z, mask=mask), 0, source_file)


# --------------------------------------------------------------- detection


def test_detection_is_exact_on_a_rounded_geometric_sweep():
    report = inspect_grid(FREQUENCIES)
    assert report.applicable
    assert report.digits == 3
    assert report.reproduced == report.total == len(FREQUENCIES)
    assert report.points_per_decade == pytest.approx(20.0, abs=0.1)
    # A fraction of a percent, not a percent: 3 s.f. gives ~0.5% granularity
    # near a mantissa of 1.0, so this is the width of one rounding bin.
    assert 0 < report.max_shift < 0.01
    assert report.ideal_nonuniformity < 1e-9
    assert report.nonuniformity >= TOEPLITZ_LIMIT  # this is *why* it applies


@pytest.mark.parametrize("digits", [1, 2, 3])
def test_detection_works_at_other_precisions(digits):
    """Not hardcoded to 3 s.f. -- whatever precision a file was exported at.
    Capped at 3 here: on this sweep 4+ significant figures already round-trip
    under the threshold on their own (see test_significant_figures_finds_the_
    smallest_that_fits for precision detection independent of applicability)."""
    report = inspect_grid(_quantised_sweep(digits=digits))
    assert report.applicable
    assert report.digits == digits
    assert report.reproduced == report.total


@pytest.mark.parametrize("digits", [4, 5, 6])
def test_finer_precision_already_clears_the_threshold(digits):
    """The other side of the same coin: once the rounding is fine enough that
    it does not matter, there is nothing to restore -- declined for having
    nothing to gain, not because the pattern was not found."""
    report = inspect_grid(_quantised_sweep(digits=digits))
    assert not report.applicable
    assert "already uniform" in report.reason.lower()


def test_significant_figures_finds_the_smallest_that_fits():
    assert significant_figures(_quantised_sweep(digits=3)) == 3
    # A value that only round-trips at 4 s.f., not 3.
    assert significant_figures(np.array([1.234, 5.678])) == 4
    # Full precision: nothing up to the search limit reproduces it exactly.
    assert significant_figures(np.array([np.pi, np.e])) is None


def test_geometric_grid_is_anchored_on_the_endpoints():
    """Not a least-squares fit: the two endpoints (themselves stored, so
    exact under the same rounding) must reappear identically -- a fitted
    line is tilted by the rounding error on every point and would miss
    them, which is how the first version of this got it wrong."""
    grid = geometric_grid(FREQUENCIES)
    assert grid[0] == FREQUENCIES[0]
    assert grid[-1] == FREQUENCIES[-1]


# ----------------------------------------------------------------- declines


def test_declines_a_non_geometric_sweep():
    rng = np.random.default_rng(0)
    f = np.sort(rng.uniform(0.1, 1e5, 60))[::-1]
    report = inspect_grid(f)
    assert not report.applicable
    assert "not rounded" in report.reason.lower()


def test_declines_when_already_uniform():
    """Nothing to gain -- a full-precision logspace grid already passes
    pyimpspec's own test."""
    report = inspect_grid(np.logspace(5, -1, 121))
    assert not report.applicable
    assert "already uniform" in report.reason.lower()


def test_declines_when_an_interior_point_is_missing():
    """Removing an array element merges two spacings into one double-width
    gap, so the retained set can no longer round-trip against any geometric
    progression -- most stored values still land on the pattern, but not the
    ones straddling the gap."""
    f = np.delete(FREQUENCIES, len(FREQUENCIES) // 2)
    report = inspect_grid(f)
    assert not report.applicable
    assert report.digits == 3  # precision is still detected
    assert report.reproduced < report.total


def test_declines_too_few_points():
    report = inspect_grid(np.logspace(2, 0, 5))
    assert not report.applicable
    assert "point" in report.reason.lower()


def test_decline_reason_is_never_empty():
    for f in (np.logspace(5, -1, 121), np.delete(FREQUENCIES, 3)):
        report = inspect_grid(f)
        assert not report.applicable
        assert report.reason


# ------------------------------------------------------- masking awareness


def test_masked_interior_points_reopen_the_gap_and_decline():
    """pyimpspec's own uniformity test sees only the *unmasked* points, so a
    sweep validation has already pruned the middle of can restore correctly
    and still gain nothing -- the gap the mask leaves behind is exactly the
    irregularity this module exists to fix, and masking cannot un-fix it."""
    report = inspect_grid(FREQUENCIES, mask={9: True, 10: True, 11: True})
    assert not report.applicable
    assert "already masked" in report.reason.lower()


def test_masked_edge_points_are_still_fine():
    """A contiguous truncation from the edges is still a (shorter) geometric
    progression -- unlike an interior gap, there is nothing here for
    pyimpspec's own test to trip on."""
    n = len(FREQUENCIES)
    report = inspect_grid(FREQUENCIES, mask={0: True, n - 1: True})
    assert report.applicable


def test_no_mask_and_an_empty_mask_agree():
    assert inspect_grid(FREQUENCIES, mask=None) == inspect_grid(FREQUENCIES, mask={})
    assert inspect_grid(FREQUENCIES, mask={0: False}) == inspect_grid(FREQUENCIES)


# --------------------------------------------- regridded(): identity, mask


def test_regridded_preserves_mask_label_key_and_impedances():
    # A single edge point: masking stays applicable (see
    # test_masked_edge_points_are_still_fine), which is what this test needs
    # -- its point is what regridded() carries over, not whether it applies.
    ds = _dataset(FREQUENCIES, mask={0: True})
    new_ds, report = regridded(ds)

    assert report.applicable
    assert new_ds.data.get_mask() == ds.data.get_mask()
    assert new_ds.data.get_label() == ds.data.get_label()
    assert new_ds.key == ds.key
    assert new_ds.num_points == ds.num_points
    assert np.array_equal(
        new_ds.data.get_impedances(masked=None), ds.data.get_impedances(masked=None)
    )
    assert nonuniformity(new_ds.data.get_frequencies(masked=None)) < 1e-9


def test_regridded_returns_a_detached_copy_even_when_it_declines():
    """Callers read the report, not identity -- but a decline still must not
    hand back the caller's own live object, or a later mask edit on the
    'unchanged' copy would corrupt the original."""
    ds = _dataset(np.logspace(5, -1, 121))  # already uniform: declines
    new_ds, report = regridded(ds)

    assert not report.applicable
    assert new_ds.data is not ds.data
    assert np.array_equal(new_ds.frequencies, ds.frequencies)

    new_ds.data.set_mask({0: True})
    # pyimpspec's own get_mask() is dense (every index, mostly False) rather
    # than sparse, so "untouched" means no True survives, not an empty dict.
    assert not any(ds.data.get_mask().values()), "mutating the copy must not touch the original"


def test_regridded_changes_nothing_when_declined():
    ds = _dataset(np.logspace(5, -1, 121))
    new_ds, report = regridded(ds)
    assert not report.applicable
    assert np.array_equal(new_ds.data.get_frequencies(masked=None), ds.frequencies)


# --------------------------------------------------- the Toeplitz shortcut


@pytest.fixture
def quadrature_counter(monkeypatch):
    """Counts calls to the quadrature pyimpspec falls back to when the
    Toeplitz shortcut is unavailable -- the thing whose call count actually
    proves the shortcut engaged, rather than asserting on wall-clock."""
    calls = {"n": 0}
    original = tr_rbf._A_matrix_element

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tr_rbf, "_A_matrix_element", counting)
    return calls


def test_the_shortcut_actually_engages(quadrature_counter):
    """The decisive check: not just that the grid measures as uniform, but
    that pyimpspec's own assembly takes the fast path because of it."""
    ds = _dataset(FREQUENCIES)
    n = len(FREQUENCIES)

    quadrature_counter["n"] = 0
    tr_rbf.calculate_drt_tr_rbf(
        ds.data, rbf_type="gaussian", mode="complex", cross_validation="",
        lambda_value=1e-3, num_procs=1,
    )
    brute_force_calls = quadrature_counter["n"]
    # O(n^2): every entry of both matrices gets its own integration.
    assert brute_force_calls == 2 * n * n

    new_ds, report = regridded(ds)
    assert report.applicable

    quadrature_counter["n"] = 0
    tr_rbf.calculate_drt_tr_rbf(
        new_ds.data, rbf_type="gaussian", mode="complex", cross_validation="",
        lambda_value=1e-3, num_procs=1,
    )
    toeplitz_calls = quadrature_counter["n"]
    # 4n: the Toeplitz trick still calls the quadrature once per row (n) and
    # once per column (n) rather than deduplicating their one shared point,
    # for each of the real and imaginary matrices -- 2*(n+n) = 4n.
    assert toeplitz_calls == 4 * n
    assert toeplitz_calls < brute_force_calls


def test_the_answer_barely_moves():
    """What actually matters to a user: the DRT computed from the restored
    grid is close to the one computed from the sweep as measured, and finds
    the same peaks in the same places."""
    ds = _dataset(FREQUENCIES)
    new_ds, report = regridded(ds)
    assert report.applicable

    kwargs = dict(rbf_type="gaussian", mode="complex", cross_validation="",
                  lambda_value=1e-3, num_procs=1)
    result_as_stored = tr_rbf.calculate_drt_tr_rbf(ds.data, **kwargs)
    result_regridded = tr_rbf.calculate_drt_tr_rbf(new_ds.data, **kwargs)

    tau_a, gamma_a = result_as_stored.get_drt_data()
    tau_b, gamma_b = result_regridded.get_drt_data()
    peaks_a, _ = result_as_stored.get_peaks()
    peaks_b, _ = result_regridded.get_peaks()
    assert len(peaks_a) == len(peaks_b)

    lo = max(tau_a.min(), tau_b.min())
    hi = min(tau_a.max(), tau_b.max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 500)
    a = np.interp(grid, tau_a, gamma_a)
    b = np.interp(grid, tau_b, gamma_b)
    max_relative_error = np.abs(a - b).max() / np.abs(a).max()
    assert max_relative_error < 5e-3
