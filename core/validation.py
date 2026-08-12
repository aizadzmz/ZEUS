#KK and Z-HT
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

import numpy as np

# Import for side effects: replaces pyimpspec's loop-driven curvature calculation, which dominates the Kramers-Kronig run time.
import core._pyimpspec_patches  # noqa: F401
from pyimpspec import (
    KramersKronigResult,
    ZHITResult,
    perform_kramers_kronig_test,
    perform_zhit,
)

ValidationResult = Union[KramersKronigResult, ZHITResult]

# Floor on what an iterative prune may leave behind; below this the loop chases its own residuals rather than the data's.
MIN_POINTS_AFTER_PRUNE = 8

# How a residual is made relative: MODULUS divides both parts by |Z| (pyimpspec's), COMPONENT each by its own signed value (RelaxIS's). Governs rejection, not just the plot.
RESIDUAL_BY_MODULUS = "modulus"
RESIDUAL_BY_COMPONENT = "component"
RESIDUAL_MODES = (RESIDUAL_BY_MODULUS, RESIDUAL_BY_COMPONENT)

# COMPONENT is deliberately uncapped: a component not resolved above the noise is a point to drop, and rejection reads these numbers directly. See _relative_to for a component of exactly 0.0.


def _measured_impedances(result: ValidationResult) -> np.ndarray:
    """Z_exp, recovered from what a validation result carries: the fit Z_fit and
    the residuals r = (Z_exp - Z_fit)/|Z_exp|, solved for |Z_exp| through
    (1 - |r|^2) m^2 - 2 Re(Z_fit * conj(r)) m - |Z_fit|^2 = 0.

    Recoverable only while |r| < 1. There the leading coefficient is positive
    and the constant term negative, so the two roots straddle zero and the
    single positive one -- the branch taken below -- is the modulus. At |r| >= 1
    the leading coefficient turns negative, both roots come out positive, and
    both reproduce the very same (Z_fit, r) pair: the measurement is genuinely
    not recoverable, not merely awkward to compute. Those points come back as
    NaN, and relative_residuals rejects them outright rather than guessing.
    """
    r = np.asarray(result.residuals)
    Z_fit = np.asarray(result.impedances)
    a = 1.0 - np.abs(r) ** 2
    b = -2.0 * np.real(Z_fit * np.conj(r))
    c = -np.abs(Z_fit) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        modulus = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        Z_exp = Z_fit + r * modulus
    return np.where(a > 0.0, Z_exp, complex(np.nan, np.nan))


def _relative_to(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, with the two zero-denominator cases decided
    rather than left to float arithmetic: x/0 -> signed infinity, so the point
    exceeds every threshold and is rejected, and 0/0 -> 0.0, a perfectly fitted
    component being no outlier (left as NaN it would fail every `>` and so be
    silently kept).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(numerator / denominator, dtype=float)
    return np.where(np.isnan(out), 0.0, out)


def relative_residuals(
    result: ValidationResult, mode: str = RESIDUAL_BY_MODULUS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frequencies, real residuals, imaginary residuals) in percent, under the
    chosen convention. The MODULUS pair is passed through from pyimpspec
    untouched; see RESIDUAL_MODES."""
    if mode not in RESIDUAL_MODES:
        raise ValueError(f"Unknown residual mode {mode!r}; expected one of {RESIDUAL_MODES}.")

    freq, res_re, res_im = result.get_residuals_data()
    if mode == RESIDUAL_BY_MODULUS:
        return freq, res_re, res_im

    # Both conventions share a numerator, so |Z| / Z_component is all that separates them; the denominator is signed, as RelaxIS divides, and sign never reaches rejection either way.
    Z_exp = _measured_impedances(result)
    modulus = np.abs(Z_exp)
    # Where the measurement could not be recovered (|r| >= 1, see
    # _measured_impedances) the point is already off by at least the whole of
    # |Z|, so it is past every threshold under any convention -- reported as
    # infinite rather than left to arrive as a NaN, which _relative_to would
    # read as a perfect fit and quietly keep.
    unrecoverable = ~np.isfinite(modulus)
    return (
        freq,
        np.where(unrecoverable, np.inf, _relative_to(res_re * modulus, Z_exp.real)),
        np.where(unrecoverable, np.inf, _relative_to(res_im * modulus, Z_exp.imag)),
    )


def run_kk_test(
    dataset,
    *,
    test: str = "complex",
    admittance: bool = False,
    num_F_ext_evaluations: int = 10,
    **kwargs,
) -> KramersKronigResult:
    """Run pyimpspec's linear Kramers-Kronig test on the unmasked points.
    Narrows three defaults: test="complex", admittance=False,
    num_F_ext_evaluations=10 -- pyimpspec's own test="real" fits Z' alone and
    leaves Z'' a byproduct, so the two residual series are not comparable.
    """
    return perform_kramers_kronig_test(
        dataset.data,
        test=test,
        admittance=admittance,
        num_F_ext_evaluations=num_F_ext_evaluations,
        **kwargs,
    )


def run_zhit(dataset, **kwargs) -> ZHITResult:
    """Run pyimpspec's Z-HIT analysis on the dataset's unmasked points; the
    modulus is reconstructed from the phase, which exposes drift."""
    return perform_zhit(dataset.data, **kwargs)


def unmasked_indices(dataset) -> List[int]:
    """The dataset's own indices for the points a validation result covers, in
    the order its residual arrays report them."""
    mask = dataset.data.get_mask()
    return [
        i
        for i in range(dataset.data.get_num_points(masked=None))
        if not mask.get(i, False)
    ]


def residual_deviations(
    result: ValidationResult, mode: str = RESIDUAL_BY_MODULUS
) -> List[float]:
    """How far each point strays, as max(|ΔZ'|, |ΔZ''|) in percent -- the same
    "either part is enough" rule mask_residual_outliers rejects on."""
    _, res_re, res_im = relative_residuals(result, mode)
    return [max(abs(float(re)), abs(float(im))) for re, im in zip(res_re, res_im)]


def mask_residual_outliers(
    dataset,
    result: ValidationResult,
    threshold_percent: float,
    mode: str = RESIDUAL_BY_MODULUS,
) -> None:
    """Mask points whose relative residual exceeds threshold_percent, in place.
    Already-masked points stay masked."""
    indices = unmasked_indices(dataset)
    deviations = residual_deviations(result, mode)
    if len(indices) != len(deviations):
        raise ValueError(
            "Validation result does not match the dataset's current mask; "
            "re-run the validation first."
        )

    over = {idx: True for idx, dev in zip(indices, deviations) if dev > threshold_percent}
    if over:
        dataset.data.set_mask(over)


# Backwards-compatible alias (residual masking is method-agnostic).
mask_kk_outliers = mask_residual_outliers


@dataclass
class PruneOutcome:
    """What one sweep's iterative prune did. `result` is the validation of the
    *final* pass, so it describes the sweep with `removed` already gone -- the
    two only make sense applied together."""

    result: ValidationResult
    removed: List[int] = field(default_factory=list)  # dataset indices, sorted
    passes: int = 1                                   # validations run, incl. the first
    stop_reason: str = "converged"  # converged | limit reached | too few points


def prune_iteratively(
    dataset,
    runner: Callable,
    *,
    hard_percent: float,
    soft_percent: float,
    max_removed: int,
    residual_mode: str = RESIDUAL_BY_MODULUS,
) -> PruneOutcome:
    """Validate, drop the worst offenders, and validate again until nothing is
    left above soft_percent.

    Each pass drops every point above hard_percent at once and otherwise the
    single worst point above soft_percent, one per pass. Both limits are read
    under `residual_mode` (see RESIDUAL_MODES). Runs on a detached copy, so the
    caller's mask is untouched -- apply `removed` yourself.
    """
    if soft_percent > hard_percent:
        raise ValueError("The soft limit must not exceed the hard limit.")

    from copy import deepcopy

    from core.io_utils import EISDataset

    working = EISDataset(
        deepcopy(dataset.data), dataset.index, dataset.source_file, dataset.file_id
    )

    outcome = PruneOutcome(result=runner(working))
    while True:
        indices = unmasked_indices(working)
        deviations = residual_deviations(outcome.result, residual_mode)
        if len(indices) != len(deviations):
            raise ValueError(
                "Validation result does not match the dataset's current mask; "
                "re-run the validation first."
            )

        doomed = [i for i, dev in zip(indices, deviations) if dev > hard_percent]
        if not doomed:
            over_soft = [
                (dev, i) for i, dev in zip(indices, deviations) if dev > soft_percent
            ]
            if not over_soft:
                break
            doomed = [max(over_soft)[1]]

        # Both caps stop *before* the removal, so the reported result describes exactly the points kept.
        if len(outcome.removed) + len(doomed) > max_removed:
            outcome.stop_reason = "limit reached"
            break
        if len(indices) - len(doomed) < MIN_POINTS_AFTER_PRUNE:
            outcome.stop_reason = "too few points"
            break

        working.data.set_mask({i: True for i in doomed})
        outcome.removed.extend(doomed)
        outcome.result = runner(working)
        outcome.passes += 1

    outcome.removed.sort()
    return outcome
