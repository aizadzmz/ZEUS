"""The iterative prune's control flow, and the two residual conventions.

No real Kramers-Kronig fit for the prune on purpose: a stub makes every one of
the loop's decisions observable and deterministic. The one real fit at the
bottom pins what pyimpspec stores on a result, which COMPONENT reads.
"""
import os

# Must precede any Qt import; core.plotting is read for its series names below.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.validation import (
    MIN_POINTS_AFTER_PRUNE,
    RESIDUAL_BY_COMPONENT,
    RESIDUAL_BY_MODULUS,
    RESIDUAL_MODES,
    _measured_impedances,
    _relative_to,
    mask_residual_outliers,
    prune_iteratively,
    relative_residuals,
    residual_deviations,
    unmasked_indices,
)


class FakeResult:
    """Stands in for a KramersKronigResult; the prune only ever asks it for
    residuals."""

    def __init__(self, freq, deviations):
        self._freq = np.asarray(freq)
        self._dev = np.asarray(deviations, dtype=float)

    def get_residuals_data(self):
        # Deviation is max(|ΔZ'|, |ΔZ''|), so putting it all in the real part exercises the same comparison.
        return self._freq, self._dev, np.zeros_like(self._dev)


def make_dataset(n=20):
    f = np.logspace(5, -1, n)
    w = 2 * np.pi * f
    Z = 10 + 50 / (1 + 1j * w * 5e-3)
    return EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file="stub")


def make_runner(dataset, deviations):
    """A runner reporting each surviving point's fixed deviation, keyed by
    frequency so it survives the masking that goes on between passes."""
    badness = {
        float(f): float(d)
        for f, d in zip(dataset.data.get_frequencies(masked=None), deviations)
    }
    seen = []

    def run(ds):
        freq = ds.data.get_frequencies()  # unmasked only, in order
        seen.append(len(freq))
        return FakeResult(freq, [badness[float(f)] for f in freq])

    return run, seen


# --- The worst point first, one per pass, until nothing is over the soft limit ---
ds = make_dataset(20)
devs = [0.1] * 17 + [6.0, 3.0, 2.5]
runner, seen = make_runner(ds, devs)
outcome = prune_iteratively(ds, runner, hard_percent=5.0, soft_percent=2.0, max_removed=10)

assert outcome.stop_reason == "converged", outcome.stop_reason
assert outcome.removed == [17, 18, 19], outcome.removed
# One pass to find them, then one per removal.
assert outcome.passes == 4, outcome.passes
# 6.0 goes first (over the hard limit), then 3.0 and 2.5 worst-first.
assert seen == [20, 19, 18, 17], seen
print("worst-first pruning OK:", outcome.removed, f"in {outcome.passes} passes")

# The caller's own dataset must come back untouched -- the loop works on a copy, and main_window replays `removed` itself.
assert len(unmasked_indices(ds)) == 20

# --- Everything over the hard limit goes in one pass, however many there are ---
ds = make_dataset(20)
devs = [0.1] * 15 + [9.0, 8.0, 7.0, 6.0, 2.5]
runner, seen = make_runner(ds, devs)
outcome = prune_iteratively(ds, runner, hard_percent=5.0, soft_percent=2.0, max_removed=10)

assert outcome.removed == [15, 16, 17, 18, 19], outcome.removed
# Four hard rejects together, then the lone soft one: three passes, not six.
assert outcome.passes == 3, outcome.passes
print("hard-limit batching OK:", outcome.removed, f"in {outcome.passes} passes")

# --- max_removed stops the loop before it exceeds the cap, and says so ---
ds = make_dataset(20)
devs = [0.1] * 14 + [3.0] * 6
runner, seen = make_runner(ds, devs)
outcome = prune_iteratively(ds, runner, hard_percent=5.0, soft_percent=2.0, max_removed=2)

assert outcome.stop_reason == "limit reached", outcome.stop_reason
assert len(outcome.removed) == 2, outcome.removed
print("max_removed cap OK:", outcome.removed, outcome.stop_reason)

# --- A sweep is never pruned below the floor, whatever the cap allows ---
n = MIN_POINTS_AFTER_PRUNE + 3
ds = make_dataset(n)
runner, seen = make_runner(ds, [3.0] * n)
outcome = prune_iteratively(ds, runner, hard_percent=5.0, soft_percent=2.0, max_removed=99)

assert outcome.stop_reason == "too few points", outcome.stop_reason
assert n - len(outcome.removed) == MIN_POINTS_AFTER_PRUNE
print("point floor OK: left", n - len(outcome.removed), "of", n)

# --- A clean sweep is one pass and no removals ---
ds = make_dataset(20)
runner, seen = make_runner(ds, [0.5] * 20)
outcome = prune_iteratively(ds, runner, hard_percent=5.0, soft_percent=2.0, max_removed=10)

assert outcome.removed == [] and outcome.passes == 1 and seen == [20]
print("clean sweep OK: no removals, one pass")

# --- soft above hard is a caller bug, not a user typo (the widgets clamp it) ---
ds = make_dataset(20)
runner, _ = make_runner(ds, [0.5] * 20)
try:
    prune_iteratively(ds, runner, hard_percent=1.0, soft_percent=2.0, max_removed=10)
except ValueError as exc:
    print("soft > hard rejected OK:", exc)
else:
    raise AssertionError("expected a ValueError for soft > hard")

# --- The single-pass path still rejects on either residual part ---
ds = make_dataset(20)
freq = ds.data.get_frequencies()
result = FakeResult(freq, [0.1] * 18 + [4.0, 0.2])
assert residual_deviations(result)[18] == 4.0
mask_residual_outliers(ds, result, 2.0)
assert unmasked_indices(ds) == [i for i in range(20) if i != 18]
print("mask_residual_outliers OK: dropped index 18")


# =========================== residual conventions ===========================

class FitResult:
    """Carries what relative_residuals reads off a real validation result: the
    fit, pyimpspec's complex residuals, and its percent view of them."""

    def __init__(self, freq, Z_exp, Z_fit):
        self.frequencies = np.asarray(freq, dtype=float)
        self.impedances = np.asarray(Z_fit)
        # Schoenleber et al. 2014, eqs. 15-16 -- what pyimpspec stores.
        self.residuals = (np.asarray(Z_exp) - self.impedances) / np.abs(Z_exp)

    def get_residuals_data(self):
        return self.frequencies, self.residuals.real * 100, self.residuals.imag * 100


def offset_fit(Z_exp, fraction=0.01):
    """A fit sitting `fraction` of |Z| away from the data, on both parts."""
    Z_exp = np.asarray(Z_exp)
    return Z_exp - fraction * np.abs(Z_exp) * (1 + 1j)


# --- Z_exp is recoverable from the fit and the residuals alone ---
freq = np.logspace(5, -1, 12)
Z_exp = 10 + 50 / (1 + 1j * 2 * np.pi * freq * 5e-3) + 1 / (1j * 2 * np.pi * freq * 0.5)
result = FitResult(freq, Z_exp, offset_fit(Z_exp))
assert np.allclose(_measured_impedances(result), Z_exp), "modulus recovery drifted"
print("Z_exp recovery OK: exact to", float(np.abs(_measured_impedances(result) - Z_exp).max()))

# --- COMPONENT is MODULUS rescaled by |Z| / Z_component ---
# A point well clear of both axes, where the two conventions stay comparable.
Z_exp = np.array([100 + 60j])
result = FitResult([1.0], Z_exp, offset_fit(Z_exp))
_, mod_re, mod_im = relative_residuals(result, RESIDUAL_BY_MODULUS)
_, comp_re, comp_im = relative_residuals(result, RESIDUAL_BY_COMPONENT)
modulus = float(np.abs(Z_exp[0]))
assert np.allclose(comp_re, mod_re * modulus / 100.0)
assert np.allclose(comp_im, mod_im * modulus / 60.0)
# The imaginary part is the smaller one, so it is the one that reads worse.
assert abs(comp_im[0]) > abs(comp_re[0]) > abs(mod_re[0])
print(f"component rescaling OK: {mod_im[0]:.3f}% by |Z| -> {comp_im[0]:.3f}% by Z''")

# --- ...and the denominator is signed, so a capacitive Z'' mirrors ---
# The case measured data actually presents, Z'' < 0: dividing by |Z''| would flip only the sign, which is what makes such a bug survive a spread- or threshold-based check.
Z_exp = np.array([100 - 60j])
result = FitResult([1.0], Z_exp, offset_fit(Z_exp))
_, mod_re, mod_im = relative_residuals(result, RESIDUAL_BY_MODULUS)
_, comp_re, comp_im = relative_residuals(result, RESIDUAL_BY_COMPONENT)
modulus = float(np.abs(Z_exp[0]))
assert np.allclose(comp_re, mod_re * modulus / 100.0), comp_re
assert np.allclose(comp_im, mod_im * modulus / -60.0), comp_im
# Not merely "the right size": the sign is opposite MODULUS, and dividing by the magnitude would have given the other one.
assert comp_im[0] * mod_im[0] < 0, (comp_im[0], mod_im[0])
assert not np.allclose(comp_im, mod_im * modulus / 60.0), "denominator lost its sign"
# Rejection reads magnitudes, so the mirroring must not reach it.
assert np.allclose(
    residual_deviations(result, RESIDUAL_BY_COMPONENT),
    [max(abs(comp_re[0]), abs(comp_im[0]))],
)
print(f"signed denominator OK: Z''<0 gives {mod_im[0]:+.3f}% by |Z| -> {comp_im[0]:+.3f}% by Z''")

# --- A small component is amplified without limit: no cap ---
# Z'' at 0.001% of |Z| reads as a residual in the tens of thousands of percent, which is the point -- that component is not resolved, so the point goes.
Z_exp = np.array([100 + 0.001j])
result = FitResult([1.0], Z_exp, offset_fit(Z_exp))
_, mod_re, mod_im = relative_residuals(result, RESIDUAL_BY_MODULUS)
_, comp_re, comp_im = relative_residuals(result, RESIDUAL_BY_COMPONENT)
assert np.allclose(comp_im, mod_im * 100.0 / 0.001), comp_im
assert abs(comp_im[0]) > 1e5, comp_im
print(f"uncapped OK: {mod_im[0]:.3f}% by |Z| -> {comp_im[0]:,.0f}% by |Z''|")

# --- A measured part of zero, with an error: rejected, whatever it reports ---
# Z_exp is reconstructed arithmetically, so an exact 0j comes back as a ~1e-16 residue and the ratio is astronomic-but-finite; the point goes either way.
Z_exp = np.array([100 + 0j])
result = FitResult([1.0], Z_exp, offset_fit(Z_exp))  # fit misses on both parts
_, comp_re, comp_im = relative_residuals(result, RESIDUAL_BY_COMPONENT)
assert abs(comp_im[0]) > 1e12, comp_im
assert abs(comp_re[0]) < 10, comp_re  # Z' is 100, unaffected
ds = EISDataset(DataSet(frequencies=[1.0], impedances=Z_exp), index=0, source_file="stub")
mask_residual_outliers(ds, result, 2.0, RESIDUAL_BY_COMPONENT)
assert unmasked_indices(ds) == [], "an unresolved component must be rejected"
print(f"zero-ish denominator OK: {comp_im[0]:.1e}% -> rejected")

# --- _relative_to's own contract, at a denominator that really is 0.0 ---
# Reached when the arithmetic lands on a hard zero rather than a residue; both cases are decided rather than left to float semantics.
signed = _relative_to(np.array([2.0, -2.0]), np.array([0.0, 0.0]))
assert np.isinf(signed).all() and signed[0] > 0 > signed[1], signed
# 0/0 is NaN by default, and NaN fails every > comparison, so the point would be silently *kept*; a perfectly fitted point is not an outlier, so: 0.0.
exact = _relative_to(np.array([0.0]), np.array([0.0]))
assert exact[0] == 0.0 and not np.isnan(exact[0]), exact
print("_relative_to OK: x/0 -> +-inf, 0/0 -> 0.0")

# --- A residual at or above 100% of |Z|: not recoverable, and not kept ---
# _measured_impedances inverts r = (Z_exp - Z_fit)/|Z_exp| through a quadratic
# whose leading coefficient is 1 - |r|^2. While that is positive the two roots
# straddle zero and the positive one is the modulus; at |r| >= 1 it turns
# negative, both roots come out positive, and both reproduce the same (Z_fit, r)
# pair -- the measurement is genuinely gone. The point is off by at least the
# whole of |Z| either way, so it must be rejected rather than guessed at, and it
# must not arrive as the NaN that _relative_to reads as a perfect fit.
class RawResult(FitResult):
    """A result built from the fit and the residual directly. FitResult derives
    the residual from a Z_exp, which cannot express |r| >= 1 -- that is the
    region where no Z_exp implies it uniquely."""

    def __init__(self, Z_fit, r):
        self.frequencies = np.array([1.0])
        self.impedances = np.asarray(Z_fit)
        self.residuals = np.asarray(r)


for name, Z_fit, r in (
    # |r| exactly 1: the leading coefficient is 0 outright.
    ("|r| = 1", [0.0 + 0j], [1.0 + 0j]),
    # A residual that is large and mostly imaginary: negative discriminant.
    ("|r| > 1, no real root", [1.0 + 0j], [0.0 + 2.0j]),
    # Two positive roots, both self-consistent: Z_exp = 10 and Z_exp = -6 give
    # this very same pair, so there is nothing to choose between them.
    ("|r| > 1, ambiguous", [-30.0 + 0j], [4.0 + 0j]),
):
    result = RawResult(Z_fit, r)
    recovered = _measured_impedances(result)
    assert np.isnan(recovered).all(), (name, recovered)

    deviations = residual_deviations(result, RESIDUAL_BY_COMPONENT)
    assert deviations == [float("inf")], (name, deviations)

    ds = EISDataset(
        DataSet(frequencies=[1.0], impedances=np.array([10.0 + 0j])),
        index=0, source_file="stub",
    )
    mask_residual_outliers(ds, result, 5.0, RESIDUAL_BY_COMPONENT)
    assert unmasked_indices(ds) == [], f"{name} must be rejected, not kept"
    print(f"unrecoverable residual OK ({name}): -> inf, rejected")

# And the far side of that boundary is untouched: |r| just under 1 still inverts
# exactly, so the fix costs ordinary residuals nothing.
Z_exp = np.array([10.0 - 4.0j, 25.0 - 3.0j])
Z_fit_close = np.array([10.4 - 4.2j, 24.5 - 3.3j])
result = FitResult([1.0, 2.0], Z_exp, Z_fit_close)
assert np.allclose(_measured_impedances(result), Z_exp), _measured_impedances(result)
_, comp_re, comp_im = relative_residuals(result, RESIDUAL_BY_COMPONENT)
assert np.isfinite(comp_re).all() and np.isfinite(comp_im).all()
print("|r| < 1 OK: recovery still exact, residuals still finite")

# --- An infinite deviation is rejected; a zero one is kept ---
class _Inf(FitResult):
    def get_residuals_data(self):
        return self.frequencies, np.array([0.0, 0.0]), np.array([np.inf, 0.0])


result = _Inf([1.0, 2.0], np.array([1 + 1j, 1 + 1j]), np.array([1 + 1j, 1 + 1j]))
devs = residual_deviations(result, RESIDUAL_BY_MODULUS)
assert devs[0] == float("inf") and devs[1] == 0.0, devs
ds = EISDataset(
    DataSet(frequencies=[1.0, 2.0], impedances=np.array([1 + 1j, 1 + 1j])),
    index=0, source_file="stub",
)
mask_residual_outliers(ds, result, 2.0, RESIDUAL_BY_MODULUS)
assert unmasked_indices(ds) == [1], unmasked_indices(ds)
print("infinite deviation OK: rejected; a zero one kept")

# --- Rejection follows the convention, not just the plot ---
# 1% of |Z| off on both parts; |Z''| is a fifth of |Z|, so the imaginary residual sits inside a 2% threshold under MODULUS and outside it under COMPONENT.
freq = np.logspace(3, 0, 20)
Z_exp = np.full(20, 100 + 20j, dtype=complex)
Z_fit = Z_exp.copy()
Z_fit[7] = offset_fit(Z_exp[7:8])[0]  # one bad point; the rest fit perfectly
result = FitResult(freq, Z_exp, Z_fit)

assert residual_deviations(result, RESIDUAL_BY_MODULUS)[7] < 2.0
assert residual_deviations(result, RESIDUAL_BY_COMPONENT)[7] > 2.0

for mode, expected in ((RESIDUAL_BY_MODULUS, list(range(20))),
                       (RESIDUAL_BY_COMPONENT, [i for i in range(20) if i != 7])):
    ds = EISDataset(DataSet(frequencies=freq, impedances=Z_exp), index=0, source_file="stub")
    mask_residual_outliers(ds, result, 2.0, mode)
    assert unmasked_indices(ds) == expected, (mode, unmasked_indices(ds))
print("convention drives rejection OK: point 7 survives |Z|, fails |Z''|")

# --- ...including inside the prune loop ---
ds = make_dataset(20)
runner, seen = make_runner(ds, [0.1] * 20)  # deviations the stub reports as-is
outcome = prune_iteratively(
    ds, runner, hard_percent=5.0, soft_percent=2.0, max_removed=10,
    residual_mode=RESIDUAL_BY_MODULUS,
)
assert outcome.removed == [], outcome.removed
print("prune_iteratively takes a residual_mode OK")

# --- An unknown convention is rejected rather than silently defaulted ---
try:
    relative_residuals(result, "by-eye")
except ValueError as exc:
    print("unknown mode rejected OK:", exc)
else:
    raise AssertionError("expected a ValueError for an unknown residual mode")

# --- Every mode has a name to plot under ---
# core.plotting spells the keys out rather than importing them, so this is the guard that keeps the two lists together.
from core.plotting import _RESIDUAL_SERIES_NAMES

assert set(_RESIDUAL_SERIES_NAMES) == set(RESIDUAL_MODES), _RESIDUAL_SERIES_NAMES
print("plot series names cover every mode OK:", sorted(_RESIDUAL_SERIES_NAMES))

# --- What a real pyimpspec result stores is what the recovery assumes ---
# The one place this suite touches a genuine fit, so an upgrade that changes `.impedances` or `.residuals` fails here and not in the GUI.
from core.validation import run_kk_test, run_zhit

freq = np.logspace(4, -1, 25)
w = 2 * np.pi * freq
Z_true = 10 + 50 / (1 + 1j * w * 5e-3)
rng = np.random.default_rng(0)
measured = Z_true * (1 + rng.normal(0, 0.004, Z_true.shape))
ds = EISDataset(DataSet(frequencies=freq, impedances=measured), index=0, source_file="stub")

# Both methods the step offers: Z-HIT reconstructs the modulus rather than fitting a circuit, but stores the same three fields.
for name, run in (("KK", run_kk_test), ("Z-HIT", run_zhit)):
    result = run(ds)
    recovered = _measured_impedances(result)
    error = float(
        np.abs(recovered - ds.data.get_impedances()).max() / np.abs(measured).max()
    )
    assert error < 1e-12, (name, error)
    # And the modulus convention is still pyimpspec's own numbers, untouched.
    _, mod_re, mod_im = relative_residuals(result, RESIDUAL_BY_MODULUS)
    assert np.array_equal(mod_re, result.get_residuals_data()[1])
    assert np.array_equal(mod_im, result.get_residuals_data()[2])
    # Measured data never lands a part on exactly 0.0, so the component view is large near the axes but finite.
    _, comp_re, comp_im = relative_residuals(result, RESIDUAL_BY_COMPONENT)
    assert np.isfinite(comp_re).all() and np.isfinite(comp_im).all(), name
    print(f"real {name} result OK: Z_exp recovered to {error:.1e} relative")

print("\nAll validation tests passed.")
