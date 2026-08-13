"""Recovering a rounded-off geometric frequency grid, for the Toeplitz shortcut.

pyimpspec's TR-RBF assembly takes a ~30x faster path when the frequencies are
ln-spaced to within 1% (see tr_rbf._assemble_A_matrix). A sweep commanded as a
geometric progression but exported to a few significant figures routinely
misses that threshold -- not because the instrument drifted, but because the
export rounded it there. Worse, the rounding is fixed in absolute terms while
the true spacing shrinks as more points are added, so a denser sweep is
*more* likely to miss the threshold, not less.

The check here is self-verifying: it only ever regenerates a grid that proves
itself to be a rounding of a geometric progression, by reproducing every
stored value exactly. A stitched measurement, a genuinely non-uniform sweep,
or a spectrum with points already removed will fail that round-trip and is
left untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

# core.io_utils is imported inside regridded(), not at module scope, so this
# stays free to import from core.drt/core.ecm the way those do.
if TYPE_CHECKING:
    from core.io_utils import EISDataset

# Mirrors the threshold in pyimpspec.analysis.drt.tr_rbf._assemble_A_matrix.
TOEPLITZ_LIMIT = 0.01

# Significant figures tried, smallest first, before giving up on quantisation.
QUANTISED_MAX_DIGITS = 6

# Below this many points a "grid" is not worth reasoning about, and the
# Toeplitz shortcut needs at least a handful of points to matter anyway.
MIN_POINTS = 8


@dataclass(frozen=True)
class GridReport:
    """What inspect_grid() found, and what a caller should tell the user."""

    applicable: bool
    digits: Optional[int]          # significant figures the column carries
    points_per_decade: float
    reproduced: int                # stored values the ideal grid round-trips
    total: int
    max_shift: float               # largest relative correction, e.g. 0.0031
    nonuniformity: float           # as stored
    ideal_nonuniformity: float
    reason: str                    # empty when applicable, else why not


def nonuniformity(frequencies) -> float:
    """pyimpspec's own uniformity measure: relative spread of the ln-spacing.
    Below TOEPLITZ_LIMIT, the fast assembly path is taken."""
    f = np.asarray(frequencies, dtype=float)
    spacing = np.diff(np.log(1.0 / f))
    mean = np.mean(spacing)
    if mean == 0:
        return float("inf")
    return float(np.std(spacing) / mean)


def significant_figures(values, limit: int = QUANTISED_MAX_DIGITS) -> Optional[int]:
    """The smallest k <= limit for which every value equals its own
    k-significant-figure rounding, or None if none does (already full
    precision, or not decimal-quantised at all)."""
    values = np.asarray(values, dtype=float)
    for k in range(1, limit + 1):
        rounded = np.array([float(f"{v:.{k}g}") for v in values])
        if np.array_equal(rounded, values):
            return k
    return None


def geometric_grid(frequencies) -> np.ndarray:
    """The geometric progression `frequencies` would be an exact instance of,
    anchored on its own first and last value.

    Anchored on the endpoints rather than least-squares fit through ln f:
    a fitted line is tilted by the rounding error on every point, including
    the once-per-decade round numbers (20000, 2000, 200, ...), and misses
    them. Anchoring on the endpoints (themselves stored values, so exact
    under the same rounding) does not have that problem.
    """
    f = np.asarray(frequencies, dtype=float)
    n = len(f)
    idx = np.arange(n)
    return f[0] * (f[-1] / f[0]) ** (idx / (n - 1))


def inspect_grid(frequencies, mask: Optional[Dict[int, bool]] = None) -> GridReport:
    """Whether `frequencies` is a rounded geometric progression, and what
    regenerating it would gain.

    `mask`, if given, is a pyimpspec-style {index: bool} mask (True = masked
    out) over the same array. It cannot change whether the column round-trips
    -- masking does not alter which frequencies were actually measured -- but
    it can still cost the whole result: pyimpspec's own uniformity test sees
    only the *unmasked* points, so a sweep with interior points already
    masked out (typically by validation, upstream of the DRT step) can
    restore correctly and still gain nothing, because the gap the mask
    leaves behind is exactly the kind of irregularity this exists to fix.
    """
    f = np.asarray(frequencies, dtype=float)
    n = len(f)

    def _decline(reason: str, digits: Optional[int] = None) -> GridReport:
        return GridReport(
            applicable=False, digits=digits, points_per_decade=0.0,
            reproduced=0, total=n, max_shift=0.0,
            nonuniformity=nonuniformity(f) if n > 1 else 0.0,
            ideal_nonuniformity=0.0, reason=reason,
        )

    if n < MIN_POINTS:
        return _decline(f"Only {n} point(s); need at least {MIN_POINTS}.")

    stored_nu = nonuniformity(f)
    if stored_nu < TOEPLITZ_LIMIT:
        return _decline(
            "Frequencies are already uniform enough to take the fast path; "
            "nothing to gain."
        )

    digits = significant_figures(f)
    if digits is None:
        return _decline(
            "Frequencies are not rounded to a fixed number of significant "
            "figures, so there is nothing to recover them from."
        )

    ideal = geometric_grid(f)
    rounded_back = np.array([float(f"{v:.{digits}g}") for v in ideal])
    reproduced = int(np.sum(rounded_back == f))

    if reproduced != n:
        return _decline(
            f"Only {reproduced} of {n} stored values match a geometric "
            f"progression rounded to {digits} significant figures -- this "
            f"does not look like a rounded geometric sweep (a point may "
            f"already be removed, or the sweep may be stitched from "
            f"several ranges).",
            digits=digits,
        )

    # What actually reaches pyimpspec's own test: every point when nothing is
    # masked, otherwise only the unmasked ones -- geometric_grid is exact by
    # construction, so this is the only branch where ideal_nu is not ~1e-15.
    unmasked = [i for i in range(n) if not (mask or {}).get(i, False)]
    if len(unmasked) < n:
        if len(unmasked) < 2:
            return _decline(
                f"Only {len(unmasked)} unmasked point(s) remain; too few to "
                f"reason about.",
                digits=digits,
            )
        ideal_nu = nonuniformity(ideal[unmasked])
        if ideal_nu >= TOEPLITZ_LIMIT:
            return _decline(
                f"{n - len(unmasked)} point(s) are already masked. The "
                f"frequency grid verifiably restores, but the points "
                f"actually used are not uniform enough on their own for "
                f"the fast path.",
                digits=digits,
            )
    else:
        ideal_nu = nonuniformity(ideal)
        if ideal_nu >= TOEPLITZ_LIMIT:
            return _decline(
                "Recovering the grid would not bring it under the fast-path "
                "threshold.",
                digits=digits,
            )

    shift = np.abs(ideal - f) / f
    decades = abs(np.log10(f[-1] / f[0]))
    return GridReport(
        applicable=True, digits=digits,
        points_per_decade=(n - 1) / decades if decades else 0.0,
        reproduced=reproduced, total=n, max_shift=float(shift.max()),
        nonuniformity=stored_nu, ideal_nonuniformity=ideal_nu, reason="",
    )


def regridded(dataset) -> Tuple["EISDataset", GridReport]:
    """A detached copy of an EISDataset with its frequency column replaced by
    the geometric progression it verifiably rounds, and the report describing
    what changed. The sweep is returned unchanged (as a detached copy) when
    inspect_grid() declines -- callers should read the report rather than
    branch on identity.

    Detection runs on every point (masked=None) for the round-trip -- the
    true pattern is recoverable from the sweep as acquired regardless of
    what is currently masked -- but the mask is also handed to inspect_grid()
    so a sweep with points already masked out declines rather than promising
    a speed-up the retained points cannot actually reach. The existing mask
    is carried over unchanged either way; only the frequency column moves,
    and every impedance is passed through exactly as measured, at the same
    index.
    """
    from pyimpspec import DataSet

    from core.filtering import detached_copy
    from core.io_utils import EISDataset

    data = dataset.data
    mask = data.get_mask()
    frequencies = data.get_frequencies(masked=None)
    report = inspect_grid(frequencies, mask=mask)
    if not report.applicable:
        return detached_copy(dataset), report

    ideal = geometric_grid(frequencies)
    new_data = DataSet(
        frequencies=ideal,
        impedances=data.get_impedances(masked=None),
        mask=mask,
        label=data.get_label(),
        path=data.get_path(),
    )
    return (
        EISDataset(new_data, dataset.index, dataset.source_file, dataset.file_id),
        report,
    )
