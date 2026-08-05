#KK and Z-HT
from dataclasses import dataclass, field
from typing import Callable, List, Union

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


def run_kk_test(
    dataset,
    *,
    admittance: bool = False,
    num_F_ext_evaluations: int = 10,
    **kwargs,
) -> KramersKronigResult:
    """Run pyimpspec's linear Kramers-Kronig test on the unmasked points.
    Narrows two defaults: admittance=False, num_F_ext_evaluations=10."""
    return perform_kramers_kronig_test(
        dataset.data,
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


def residual_deviations(result: ValidationResult) -> List[float]:
    """How far each point strays, as max(|ΔZ'|, |ΔZ''|) in percent -- the same
    "either part is enough" rule mask_residual_outliers rejects on."""
    _, res_re, res_im = result.get_residuals_data()
    return [max(abs(float(re)), abs(float(im))) for re, im in zip(res_re, res_im)]


def mask_residual_outliers(
    dataset, result: ValidationResult, threshold_percent: float
) -> None:
    """Mask points whose relative residual exceeds threshold_percent, in place.
    Already-masked points stay masked."""
    indices = unmasked_indices(dataset)
    deviations = residual_deviations(result)
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
) -> PruneOutcome:
    """Validate, drop the worst offenders, and validate again until nothing is
    left above soft_percent.

    Each pass drops every point above hard_percent at once -- those are bad
    beyond argument -- and otherwise the single worst point above soft_percent,
    one per pass, because removing a point moves every other point's residual
    and the next-worst may well come back inside the limit on its own.

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
        deviations = residual_deviations(outcome.result)
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
