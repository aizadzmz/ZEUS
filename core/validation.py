#KK and Z-HT
from typing import Tuple, Union

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


def run_kk_test(
    dataset,
    *,
    admittance: bool = False,
    num_F_ext_evaluations: int = 10,
    **kwargs,
) -> KramersKronigResult:
    """
    Run pyimpspec's linear Kramers-Kronig test on the dataset's currently
    unmasked points. Extra keyword arguments are passed through to
    pyimpspec.perform_kramers_kronig_test (e.g. test, num_RC).

    Two defaults are narrowed from pyimpspec's: admittance=False and
    num_F_ext_evaluations=10.

    admittance=False: pyimpspec's None fits both the impedance and the
    admittance representation and keeps whichever scores better, for
    several times the cost. The admittance representation mainly earns its
    keep on spectra with negative differential resistance; pass
    admittance=None for those.

    num_F_ext_evaluations=10 searches for an optimal extension or
    contraction of the time-constant range (pyimpspec defaults to 20; 10 is
    the lowest magnitude pyimpspec accepts for a non-zero search -- below
    that it raises). Setting it to 0 instead pins the range to the measured
    frequency span, which is the classical setup -- eq. 18 in Boukamp (1995)
    and eq. 12 in Schoenleber et al. (2014); see _generate_time_constants,
    where F_ext=1 gives tau_min=1/w_max and tau_max=1/w_min. The search is
    pyimpspec's own later refinement (Yrjaenae and Bobacka, 2024).

    Historical numbers (superseded below): an earlier benchmark at
    num_F_ext_evaluations=20, on synthetic Randles spectra whose generation
    script was not preserved, found a fabricated ~59% worst-case residual
    at F_ext=0 on clean data -- the RC order selection occasionally
    collapsed (7 elements where it normally picks 16) with the range
    pinned, and the search kept it in a 14-23 band and avoided that. That
    finding is the reason this function runs the search by default at all.

    Re-measured for the 10 default, interleaved A/B, pyimpspec's own
    CIRCUIT_1 mock spectrum (Boukamp 1995 TC-1), 8 noise seeds per cell,
    max |relative residual| on data that is KK-consistent *by construction*
    (so any large value is a fabricated violation):

        noise    F_ext=0 median / worst   F_ext=10 median / worst
        0.5%          1.28% /  1.44%            1.30% /  2.74%
        1%            2.64% /  4.32%            2.59% /  2.85%
        2%            5.46% /  7.40%            5.01% /  5.95%

    and on CIRCUIT_1_INVALID (built-in drift injection, drift multiplier
    1x/4x against the definition's own 5.0 base):

        drift    F_ext=0 median / worst   F_ext=10 median / worst
        1x             2.73% /  4.32%            2.83% /  3.67%
        4x             4.48% /  5.54%            5.13% /  6.32%

    This run did NOT reproduce the old tail-collapse: F_ext=0's worst case
    tops out at 7.4%, comparable to F_ext=10, not 59%. That is most likely
    because CIRCUIT_1 is a different spectrum/seed set than whatever
    produced the historical numbers (its generation script was never
    committed), not proof the failure mode is gone at 10 evaluations --
    the RC-order collapse could easily be specific to circuit/noise
    combinations this run didn't hit. Treat the search-vs-pinned question
    as still open pending a benchmark that deliberately hunts for the
    collapse (more circuits, more seeds, higher noise) before leaning on
    F_ext=0 for speed.

    Timing, same interleaved setup, median of 3 reps:

        points   F_ext=0   F_ext=10   search costs
           40     0.38 s     0.78 s        2.1x
           60     0.85 s     1.77 s        2.1x
          120     4.68 s     8.46 s        1.8x

    Roughly half the overhead of the old 20-evaluation numbers (2.0-3.4x),
    as expected from halving the evaluation budget. Measure timings
    interleaved if you revisit this -- running all of one config then all
    of the other let a background load spike land on one half and briefly
    made the search look *cheaper*, which it never is.

    - num_RC=0 leaves the number of RC elements to pyimpspec's own search,
      which fits every candidate order from 2 upwards and is most of the
      run time. Two shortcuts were tried and both fabricate violations.
      Pinning num_RC returns in milliseconds but the residual-vs-order
      curve is erratic from ill-conditioning as the RC count approaches the
      point count: at orders 15, 60 and 80 the test reports 5.4%, 65.7% and
      11.5% on *clean* data while returning the same figures for drifted
      data, i.e. it can no longer tell good from bad. Bounding the search
      via suggest_num_RC's lower_limit/upper_limit made it select the floor
      and report a 20% maximum residual on a spectrum whose true residual
      is 0.4%. The order search is what makes the residual mean anything.

    log_F_ext is inert unless num_F_ext_evaluations is set to 0: pyimpspec
    uses it as the fixed value only in that case, and otherwise derives it
    from the search. A non-zero num_F_ext_evaluations must also be at least
    10 in magnitude or pyimpspec raises.
    """
    return perform_kramers_kronig_test(
        dataset.data,
        admittance=admittance,
        num_F_ext_evaluations=num_F_ext_evaluations,
        **kwargs,
    )


def run_zhit(dataset, **kwargs) -> ZHITResult:
    """
    Run pyimpspec's Z-HIT analysis on the dataset's currently unmasked
    points. The modulus is reconstructed from the phase data, which helps
    detect non-steady-state artifacts such as low-frequency drift. Extra
    keyword arguments are passed through to pyimpspec.perform_zhit
    (e.g. smoothing, interpolation, window).
    """
    return perform_zhit(dataset.data, **kwargs)


def mask_residual_outliers(
    dataset, result: ValidationResult, threshold_percent: float
) -> None:
    """
    Mask points whose relative residual (real or imaginary, in percent)
    exceeds threshold_percent, in place. Points already masked stay masked.
    Works with both Kramers-Kronig and Z-HIT results.

    result must have been produced by run_kk_test/run_zhit on this dataset
    without changing the dataset's mask in between, since residuals are only
    reported for the points that were unmasked at test time.
    """
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
