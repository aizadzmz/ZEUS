"""Monkey patches for pyimpspec hot spots. Importing this module applies them;
core.validation does so before touching pyimpspec's analysis functions."""

import functools
import threading

import numpy as np
from numpy.typing import NDArray

_PATCHED_VERSION = "5.1.3"


def _calculate_curvatures(Z: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Vectorised drop-in for pyimpspec's calculate_curvatures, ~85x faster."""
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

    # Sign of det([[Re, -Im, 1], ...]) per triple: negative for clockwise Nyquist motion.
    u = Z.real
    v = -Z.imag
    u1, u2, u3 = u[:-2], u[1:-1], u[2:]
    v1, v2, v3 = v[:-2], v[1:-1], v[2:]
    determinant = u1 * (v2 - v3) - v1 * (u2 - u3) + (u2 * v3 - v2 * u3)

    return abs_kappa * np.sign(determinant)


def _patch_calculate_curvatures() -> None:
    """Rebind calculate_curvatures everywhere pyimpspec imported it."""
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


# --- Impedance memoisation during a Kramers-Kronig run ---

# Cache for the calling thread; thread-local since validation runs off the UI thread.
_impedance_cache = threading.local()


def _patch_circuit_get_impedances() -> None:
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
            # Hand out a fresh array: pyimpspec callers modify it in place.
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
