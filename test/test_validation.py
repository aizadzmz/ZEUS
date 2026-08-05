"""The iterative prune's control flow, driven by a stub runner.

No real Kramers-Kronig fit here on purpose: the loop's job is deciding which
point to drop and when to stop, and a stub makes every one of those decisions
observable and deterministic.
"""

import numpy as np
from pyimpspec import DataSet

from core.io_utils import EISDataset
from core.validation import (
    MIN_POINTS_AFTER_PRUNE,
    mask_residual_outliers,
    prune_iteratively,
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
        # Deviation is max(|ΔZ'|, |ΔZ''|), so putting it all in the real part
        # exercises the same comparison a real lopsided residual would.
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

# The caller's own dataset must come back untouched -- the loop works on a copy,
# and main_window replays `removed` onto the real mask itself.
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

print("\nAll validation tests passed.")
