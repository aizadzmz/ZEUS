"""Monkey patches for pyimpspec hot spots.

Importing this module applies the patches. core.validation does so before it
touches pyimpspec's analysis functions.

How much they buy depends on the configuration, because they target the
num_RC suggestion phase rather than the fitting. At run_kk_test's defaults
(admittance=False, num_F_ext_evaluations=20), interleaved A/B over 3 rounds:

    points   unpatched   patched
       40      3.18 s     2.00 s    1.59x
       60      5.29 s     2.98 s    1.78x
      120     21.04 s    11.51 s    1.83x

With the log_F_ext search disabled (num_F_ext_evaluations=0) the same patches
give 2.5-3.7x, because the suggestion phase is then a larger share of the
total. The share that is left either way is real numerics
(numpy.linalg.lstsq) plus pyimpspec's per-element circuit evaluation.

Measure any revision of these interleaved, alternating patched and unpatched
runs. Timing all of one and then all of the other let a background load spike
land on one half and produced a cell claiming the patched code was slower.

Neither patch changes what the test reports. Checked against unpatched
pyimpspec over 30 synthetic spectra (20-200 points, 0/5/20% drift, with and
without noise) and the example file: identical num_RC, identical
pseudo_chisqr, and bit-identical residuals in every case.

Verified against pyimpspec 5.1.3.
"""

import functools
import threading

import numpy as np
from numpy.typing import NDArray

_PATCHED_VERSION = "5.1.3"


def _calculate_curvatures(Z: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Vectorised drop-in for pyimpspec's calculate_curvatures.

    The upstream implementation is already written against numpy arrays but
    drives them from two Python-level loops over the points, calling
    numpy.linalg.norm three times and numpy.linalg.det once per point. Each of
    those is a µs-scale dispatch on a 2- or 3-element array, so the loops cost
    ~100x what the arithmetic does.

    That made it the single biggest cost in a Kramers-Kronig run. Curvatures
    are recomputed for every candidate number of RC elements on a subdivided
    frequency grid (~200 calls over ~300 points for a 60-point sweep), which
    put ~70% of the test's time in this one function. Vectorised it is ~85x
    faster and effectively free.

    The arithmetic below is deliberately expressed the same way upstream
    expresses it -- sqrt(re*re + im*im) rather than the more accurate
    numpy.abs/hypot, because abs_kappa goes through sin(arccos(cos_alpha)),
    which amplifies a last-bit difference in cos_alpha into a ~1e-4 relative
    difference in the curvature. Checked over 300 random spectra of 5 to 400
    points: all 61230 curvatures bit-identical to upstream, no sign flips, and
    NaNs land in the same places for degenerate (duplicate-point) input.

    The 3x3 determinant is expanded by cofactors instead of calling
    numpy.linalg.det. Only its sign is used, and the sign can only disagree
    with an LU-based determinant when the three points are collinear, where
    the curvature is zero and the sign is meaningless either way.
    """
    Z = np.asarray(Z)

    a = Z[1:-1] - Z[:-2]
    b = Z[2:] - Z[1:-1]
    c = Z[2:] - Z[:-2]

    a_re, a_im = a.real, a.imag
    b_re, b_im = b.real, b.imag

    a_dot_b = a_re * b_re + a_im * b_im
    a_norm = np.sqrt(a_re * a_re + a_im * a_im)
    b_norm = np.sqrt(b_re * b_re + b_im * b_im)
    c_norm = np.sqrt(c.real * c.real + c.imag * c.imag)

    # Coincident points give 0/0 here. Upstream lets that through as NaN too.
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_alpha = a_dot_b / (a_norm * b_norm)
        np.clip(cos_alpha, -1.0, 1.0, out=cos_alpha)
        abs_kappa = 2.0 * np.sin(np.arccos(cos_alpha)) / c_norm

    # Sign of det([[Re, -Im, 1], ...]) over each consecutive triple: negative
    # for clockwise motion in a Nyquist plot when sorted by decreasing
    # frequency.
    u = Z.real
    v = -Z.imag
    u1, u2, u3 = u[:-2], u[1:-1], u[2:]
    v1, v2, v3 = v[:-2], v[1:-1], v[2:]
    determinant = u1 * (v2 - v3) - v1 * (u2 - u3) + (u2 * v3 - v2 * u3)

    return abs_kappa * np.sign(determinant)


def _patch_calculate_curvatures() -> None:
    """Rebind calculate_curvatures everywhere pyimpspec imported it.

    Each importer did `from ... import calculate_curvatures`, so it holds its
    own module-level reference; patching the defining module alone would miss
    all the call sites.
    """
    from pyimpspec.analysis.kramers_kronig import algorithms
    from pyimpspec.analysis.kramers_kronig.algorithms import (
        method_3,
        method_4,
        method_5,
        utility,
    )
    from pyimpspec.analysis.kramers_kronig.algorithms.utility import osculating_circle

    modules = (osculating_circle, utility, algorithms, method_3, method_4, method_5)
    for module in modules:
        if not hasattr(module, "calculate_curvatures"):
            raise AttributeError(
                f"{module.__name__} has no calculate_curvatures to patch; "
                f"pyimpspec has changed since {_PATCHED_VERSION}"
            )
        module.calculate_curvatures = _calculate_curvatures


# --------------------------------------------------------------------------
# Impedance memoisation during a Kramers-Kronig run
# --------------------------------------------------------------------------

# Active cache for the calling thread, or None when no Kramers-Kronig test is
# running here. Thread-local because ValidationWorker runs tests off the UI
# thread while the UI thread may be evaluating circuits of its own for plots,
# and those must not share a cache lifetime with the run.
_impedance_cache = threading.local()


def _patch_circuit_get_impedances() -> None:
    """Memoise Circuit.get_impedances while num_RC is being suggested.

    Having made the curvature calculation cheap, evaluating the candidate
    circuits became the next cost: 547 calls for a 60-point sweep, of which
    319 (58%) recompute a (circuit, frequencies) pair that was already
    computed. The duplication is structural -- suggest_num_RC_limits derives
    the num_RC range using method 5, then _suggest_using_default builds its
    own curvature dictionary for methods 3, 4 and 5 over the same circuits and
    the same subdivided frequency grid, and neither is given the other's work.

    The cache is scoped to suggest_num_RC rather than to the whole test, and
    that boundary is load-bearing. During fitting, _real_test builds a circuit
    and then mutates its element values in place via _update_circuit, so the
    same object at the same frequencies legitimately has different impedances
    at different moments; caching across that made the test select num_RC=7
    and report a 60% residual on a clean spectrum whose true residual is 0.17%.
    By the time suggest_num_RC runs, every circuit is a finished fit that the
    suggestion algorithms only read from.

    The cache holds a reference to each circuit it keys on. That is what makes
    id() safe as part of the key: a circuit that stayed alive cannot have had
    its address reused by another one.
    """
    from pyimpspec.circuit.circuit import Circuit
    from pyimpspec.analysis.kramers_kronig import single
    from pyimpspec.analysis.kramers_kronig import algorithms

    original_get_impedances = Circuit.get_impedances

    @functools.wraps(original_get_impedances)
    def get_impedances(self, frequencies):
        cache = getattr(_impedance_cache, "store", None)
        if cache is None:
            return original_get_impedances(self, frequencies)

        key = (id(self), np.asarray(frequencies).tobytes())
        hit = cache.get(key)
        if hit is None:
            Z = original_get_impedances(self, frequencies)
            # Hand out a fresh array every time and keep the cached one
            # private. Callers in pyimpspec do modify the array they get back
            # in place -- returning the cached array directly makes the second
            # caller inherit the first one's edits, which silently corrupts
            # the fit and the suggested num_RC. The copy is a few kB against a
            # recomputation that is three orders of magnitude dearer.
            cache[key] = (self, Z.copy())  # circuit ref pins id() in the key
            return Z

        return hit[1].copy()

    Circuit.get_impedances = get_impedances

    original_suggest = algorithms.suggest_num_RC

    @functools.wraps(original_suggest)
    def suggest_num_RC(*args, **kwargs):
        # Nested calls share the outermost cache and let it own the teardown.
        if getattr(_impedance_cache, "store", None) is not None:
            return original_suggest(*args, **kwargs)

        _impedance_cache.store = {}
        try:
            return original_suggest(*args, **kwargs)
        finally:
            _impedance_cache.store = None

    # single.py imported suggest_num_RC by name, so it holds its own reference.
    for module in (algorithms, single):
        if not hasattr(module, "suggest_num_RC"):
            raise AttributeError(
                f"{module.__name__} has no suggest_num_RC to patch; "
                f"pyimpspec has changed since {_PATCHED_VERSION}"
            )
        module.suggest_num_RC = suggest_num_RC


_patch_calculate_curvatures()
_patch_circuit_get_impedances()
