#DRT (distribution of relaxation times)
from __future__ import annotations

from typing import TYPE_CHECKING

# pyimpspec is imported inside the run_* functions rather than here, and the
# annotations below are strings thanks to the __future__ import. That is what
# lets the GUI read the option tuples while building its sidebar without
# paying ~4 s for pyimpspec (scipy.signal, sympy); see core/__init__.py.
if TYPE_CHECKING:
    from pyimpspec.analysis.drt import TRRBFResult
    from pyimpspec.analysis.drt.peak_analysis import DRTPeaks

RBF_TYPES = (
    "gaussian",
    "c0-matern",
    "c2-matern",
    "c4-matern",
    "c6-matern",
    "inverse-quadratic",
    "inverse-quadric",
    "cauchy",
    "piecewise-linear",
)
DATA_MODES = ("complex", "real", "imaginary")
CROSS_VALIDATION_METHODS = ("", "gcv", "mgcv", "rgcv", "re-im", "lc")
RBF_SHAPE_CONTROLS = ("fwhm", "factor")


def run_drt(
    dataset,
    rbf_type: str = "gaussian",
    mode: str = "complex",
    inductance: bool = False,
    derivative_order: int = 1,
    cross_validation: str = "",
    lambda_value: float = 1e-3,
    rbf_shape: str = "fwhm",
    shape_coeff: float = 0.5,
    credible_intervals: bool = False,
    num_samples: int = 1000,
    timeout: int = 60,
    num_procs: int = -1,
) -> TRRBFResult:
    """Compute the DRT of the dataset's unmasked points using Tikhonov
    regularization with RBF or piecewise-linear discretization (TR-RBF).
    Settings map onto pyDRTtools' GUI panel as follows:
      rbf_type           -> Method of Discretization
      mode                -> Data Used ("complex" = Combined Re-Im Data)
      inductance          -> Inductance (fit with/without an inductive term)
      derivative_order    -> Regularization Derivative
      cross_validation    -> Parameter Selection Method ("" = custom, i.e.
                             lambda_value is used directly instead of being
                             optimized)
      lambda_value        -> Regularization parameter
      rbf_shape           -> RBF Shape Control ("fwhm" or "factor")
      shape_coeff         -> FWHM Control / Shape Factor value
      credible_intervals  -> False = Simple Run, True = Bayesian Run (slow;
                             see timeout)
      num_samples         -> Number of Samples (Bayesian run only; must be
                             >= 1000)
      timeout             -> Seconds to allow the Bayesian sampler to run
                             before giving up (Bayesian run only)

    The result exposes:
      - get_drt_data() -> (tau, gamma)
      - get_drt_credible_intervals_data() -> (tau, mean, lower, upper),
        only meaningful when credible_intervals=True
      - lambda_value: the regularization parameter actually used (the
        "Optimal Regularization parameter" once cross-validated)
    Pass the result to analyze_drt_peaks() for peak positions."""
    from pyimpspec.analysis.drt import calculate_drt_tr_rbf

    return calculate_drt_tr_rbf(
        dataset.data,
        mode=mode,
        lambda_value=lambda_value,
        cross_validation=cross_validation,
        rbf_type=rbf_type,
        derivative_order=derivative_order,
        rbf_shape=rbf_shape,
        shape_coeff=shape_coeff,
        inductance=inductance,
        credible_intervals=credible_intervals,
        num_samples=num_samples,
        timeout=timeout,
        num_procs=num_procs,
    )


def analyze_drt_peaks(
    result,
    num_peaks: int = 0,
    disallow_skew: bool = False,
) -> DRTPeaks:
    """Fit individual peaks in a DRT result using skew-normal distributions.
    num_peaks=0 analyzes every detected peak."""
    return result.analyze_peaks(num_peaks=num_peaks, disallow_skew=disallow_skew)
