#KK and Z-HT
from typing import Union

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


def mask_residual_outliers(
    dataset, result: ValidationResult, threshold_percent: float
) -> None:
    """Mask points whose relative residual exceeds threshold_percent, in place.
    Already-masked points stay masked."""
    mask = dataset.data.get_mask()
    unmasked_indices = [
        i
        for i in range(dataset.data.get_num_points(masked=None))
        if not mask.get(i, False)
    ]

    _, res_re, res_im = result.get_residuals_data()
    if len(unmasked_indices) != len(res_re):
        raise ValueError(
            "Validation result does not match the dataset's current mask; "
            "re-run the validation first."
        )

    for idx, re, im in zip(unmasked_indices, res_re, res_im):
        if abs(re) > threshold_percent or abs(im) > threshold_percent:
            mask[idx] = True

    dataset.data.set_mask(mask)


# Backwards-compatible alias (residual masking is method-agnostic).
mask_kk_outliers = mask_residual_outliers
