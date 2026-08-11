#KK and Z-HT
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

import numpy as np

# Import for side effects: replaces pyimpspec's loop-driven curvature
# calculation, which dominates the Kramers-Kronig run time. Must come before
# the pyimpspec analysis functions are used.
import core._pyimpspec_patches  # noqa: F401
from pyimpspec import (
    KramersKronigResult,
    ZHITResult,
    perform_kramers_kronig_test,
    perform_zhit,
)

ValidationResult = Union[KramersKronigResult, ZHITResult]

# Floor on what an iterative prune may leave behind. Below this a validation
# fit has too little left to be consistent *with*, and the loop starts chasing
# its own residuals rather than the data's.
MIN_POINTS_AFTER_PRUNE = 8

# How a residual is made relative. Both conventions are in use, and they
# disagree by the factor |Z| / Z_component:
#
#   MODULUS   -- DZ'/|Z| and DZ''/|Z|, what pyimpspec reports (Schoenleber et
#                al. 2014, eqs. 15-16). One common scale for both parts, so a
#                point's two residuals are directly comparable.
#   COMPONENT -- DZ'/Z' and DZ''/Z'', each part against its own signed value:
#                the error on that part as a fraction of that part. Because the
#                denominator carries Z'''s sign, the imaginary series comes out
#                mirrored relative to MODULUS on a capacitive sweep. That is
#                the convention RelaxIS reports and the ordinary definition of
#                a relative error; see relative_residuals.
#
# This is not display-only. The threshold, soft and hard limits reject on
# whichever convention is chosen, so the limit lines drawn across the residual
# plot always mark the points that were actually removed.
RESIDUAL_BY_MODULUS = "modulus"
RESIDUAL_BY_COMPONENT = "component"
RESIDUAL_MODES = (RESIDUAL_BY_MODULUS, RESIDUAL_BY_COMPONENT)

# COMPONENT is deliberately uncapped. Z'' passes through zero entering an
# inductive tail and again as an arc closes back onto the real axis, and Z' can
# be small on a strongly capacitive sweep; the ratio grows without bound there.
# That is the intent -- a point whose component is not resolved above the noise
# is a point to drop, and rejection reads these numbers directly.
#
# The one case arithmetic cannot express is a component of exactly 0.0; see
# _relative_to for what happens there.


def _measured_impedances(result: ValidationResult) -> np.ndarray:
    """Z_exp, recovered from what a validation result carries.

    A result stores the fit Z_fit and the residuals r = (Z_exp - Z_fit)/|Z_exp|
    but not Z_exp itself. Writing m for |Z_exp|, Z_exp = Z_fit + r*m, and
    taking the modulus of both sides squares to

        (1 - |r|^2) m^2 - 2 Re(Z_fit * conj(r)) m - |Z_fit|^2 = 0,

    whose positive root is m. Exact to rounding while |r| < 1, which a residual
    of a few percent is by three orders of magnitude.
    """
    r = np.asarray(result.residuals)
    Z_fit = np.asarray(result.impedances)
    a = 1.0 - np.abs(r) ** 2
    b = -2.0 * np.real(Z_fit * np.conj(r))
    c = -np.abs(Z_fit) ** 2
    modulus = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return Z_fit + r * modulus


def _relative_to(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, with the two zero-denominator cases decided
    rather than left to float arithmetic.

    A component of exactly 0.0 is rare -- it needs the measured Z' or Z'' to be
    a hard zero, not merely small -- but it has to land somewhere:

    * error present, component zero (x/0) -> signed infinity. The error is
      infinitely large as a fraction of the component, so the point exceeds
      every threshold and is rejected. That is the limit of the surrounding
      points' behaviour, not a special case bolted on.
    * no error, component zero (0/0) -> 0.0. The fit reproduced a component
      that is itself zero; there is no discrepancy to express as a fraction of
      anything, and calling that an outlier would be wrong. Left as NaN it
      would be worse than wrong: NaN fails every `>` comparison, so the point
      would be silently *kept* whatever the threshold.
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

    # Both conventions share a numerator, so rescaling by |Z| / Z_component is
    # all that separates them.
    #
    # The denominator is signed, not |Z_component|. Z'' is negative across a
    # capacitive sweep, so taking its magnitude here would mirror the whole
    # imaginary series about zero -- same numbers, opposite sign, and no longer
    # comparable point-for-point with RelaxIS, which divides by the signed
    # component. Sign never reaches rejection either way (residual_deviations
    # takes absolute values), so this is a plot- and export-facing choice.
    Z_exp = _measured_impedances(result)
    modulus = np.abs(Z_exp)
    return (
        freq,
        _relative_to(res_re * modulus, Z_exp.real),
        _relative_to(res_im * modulus, Z_exp.imag),
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
    num_F_ext_evaluations=10.

    pyimpspec defaults to test="real", which fits the RC amplitudes to Z' alone
    and leaves only the series L and C to absorb any imaginary-part misfit. Z'
    is then the fitted quantity and Z'' a byproduct, so the two residual series
    are not comparably trustworthy: against a RelaxIS reference on the same
    sweep, the real fit tracks Z' to r = 0.96 but never gets Z'' past r = 0.84
    at any number of RC elements, while the complex fit reaches r = 1.00 and
    0.999. Both parts are fitted here, and it costs nothing -- the complex
    solve timed marginally faster than the real one.
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

    Each pass drops every point above hard_percent at once -- those are bad
    beyond argument -- and otherwise the single worst point above soft_percent,
    one per pass, because removing a point moves every other point's residual
    and the next-worst may well come back inside the limit on its own.

    Both limits are read under `residual_mode` (see RESIDUAL_MODES), so which
    points count as offenders depends on it.

    Runs on a detached copy, so the caller's mask is untouched and this is safe
    to hand to a worker thread or subprocess; apply `removed` yourself.
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

        # Both caps stop *before* the removal rather than trimming it, so the
        # reported result always describes exactly the points that were kept.
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
