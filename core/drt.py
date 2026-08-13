#DRT (distribution of relaxation times)
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

# pyimpspec is imported inside the run_* functions, so the GUI can read the option tuples below without paying its ~4 s import; see core/__init__.py.
if TYPE_CHECKING:
    from pyimpspec.analysis.drt import TRRBFResult
    from pyimpspec.analysis.drt.peak_analysis import DRTPeaks

# "c0-matern" is deliberately absent: its exp(-|eps*x|) kink has no derivative at the origin, so the A-matrix quadrature crawls -- up to 2h49m for one sweep here, against ~11 s for the rest.
RBF_TYPES = (
    "gaussian",
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


# Seconds past pyimpspec's own limit before the process is killed outright.
# Its limit is the polite path and gives the better message, so it is given a
# chance to fire first -- but it is not a guarantee; see _calculate_with_deadline.
DEADLINE_GRACE_SECONDS = 30


def _calculate_tr_rbf(data, settings: dict):
    """Top level so a child process can pickle it; see _calculate_with_deadline."""
    from pyimpspec.analysis.drt import calculate_drt_tr_rbf

    return calculate_drt_tr_rbf(data, **settings)


def _calculate_with_deadline(data, settings: dict, timeout: int):
    """Run the calculation in a child process, and kill it if it overruns.

    pyimpspec enforces `timeout` from a callback that only fires once the
    sampler *accepts* a draw. A run whose acceptance rate collapses -- which
    a series inductance, the 'factor' shape control, and larger sample counts
    all provoked here -- never reaches the check and ignores the limit
    entirely: observed still running at 5400 s with timeout=600. Killing a
    child process is the only reliable way to bound it, since the sampler
    spins inside numpy and cannot be interrupted in-process.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    from concurrent.futures.process import BrokenProcessPool

    pool = ProcessPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_calculate_tr_rbf, data, settings)
        result = future.result(timeout=timeout + DEADLINE_GRACE_SECONDS)
    except BrokenProcessPool:
        # The child never started. On Windows that usually means the caller's
        # __main__ has no main guard, which the spawn start method requires --
        # a script's problem, not a reason to refuse the calculation.
        pool.shutdown(wait=False, cancel_futures=True)
        warn(
            "Could not start a worker process, so the Bayesian run cannot be "
            "stopped from outside; falling back to pyimpspec's own timeout, "
            "which a stalled sampler can overrun. Guard the entry point with "
            "`if __name__ == \"__main__\":` to get the hard limit back.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _calculate_tr_rbf(data, settings)
    except FuturesTimeoutError:
        # Killed rather than shut down: shutdown() joins a worker that is still spinning, which is the hang being prevented.
        for process in list(getattr(pool, "_processes", {}).values()):
            process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(
            f"The Bayesian run passed its {timeout} s limit and was stopped.\n\n"
            f"The sampler can reach a state it draws no accepted samples from, "
            f"where its own limit never gets checked, so the calculation is "
            f"ended from outside. Try fewer samples, a different basis "
            f"function, or plain TR-RBF."
        ) from None
    pool.shutdown(wait=True)
    return result


def series_terms(mode: str, inductance: bool) -> int:
    """How many extra unknowns (series R, series L) pyimpspec puts in front of
    the distribution itself. Mirrors _prepare_{complex,real,imaginary}_matrices
    in pyimpspec 5.1.3 -- note the real branch takes no inductance argument at
    all, so "Re only" ignores that setting entirely."""
    if mode == "complex":
        return 2 if inductance else 1
    if mode == "imaginary":
        return 1 if inductance else 0
    return 1


def unsupported_reason(mode: str, inductance: bool, cross_validation: str):
    """Why this combination of settings cannot run, or None if it can.

    Checked up front because the failure would otherwise land minutes into the
    calculation, as a raw shape-mismatch from inside cvxopt.
    """
    if cross_validation == "re-im" and series_terms(mode, inductance) != 1:
        # _compute_re_im_cross_validation hardcodes G = -eye(num_freqs + 1),
        # so it only lines up when the model carries exactly one extra term.
        remedy = (
            "Turn off 'Include series inductance'"
            if mode == "complex"
            else "Turn on 'Include series inductance'"
        )
        return (
            f"'Re-Im' λ selection does not support this combination: it "
            f"assumes exactly one series term besides the distribution, and "
            f"these settings give {series_terms(mode, inductance)}.\n\n"
            f"{remedy}, or choose a different λ selection."
        )
    return None


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
                             before giving up (Bayesian run only). Enforced by
                             running the calculation in a child process and
                             killing it, since pyimpspec's own check can be
                             skipped entirely; a caller therefore needs the
                             usual `if __name__ == "__main__":` guard, and 0
                             disables the limit and can hang indefinitely.

    The result exposes:
      - get_drt_data() -> (tau, gamma)
      - get_drt_credible_intervals_data() -> (tau, mean, lower, upper),
        only meaningful when credible_intervals=True
      - lambda_value: the regularization parameter actually used (the
        "Optimal Regularization parameter" once cross-validated)
    Pass the result to analyze_drt_peaks() for peak positions."""
    from numpy.linalg import LinAlgError
    from pyimpspec.analysis.drt import calculate_drt_tr_rbf

    reason = unsupported_reason(mode, inductance, cross_validation)
    if reason is not None:
        raise ValueError(reason)

    settings = dict(
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

    try:
        # Only the Bayesian branch reads `timeout`, and only it can ignore one.
        if credible_intervals and timeout > 0:
            return _calculate_with_deadline(dataset.data, settings, timeout)
        return calculate_drt_tr_rbf(dataset.data, **settings)
    except LinAlgError as exc:
        # The L-curve search factorises a matrix the other criteria never form; with a series inductance in the model it is routinely singular on real data.
        if cross_validation != "lc":
            raise
        raise ValueError(
            f"The 'L-curve' λ search could not factorise its matrix ({exc}). "
            f"This combination is numerically unstable"
            f"{' with a series inductance in the model' if inductance else ''}"
            f".\n\nTry another λ selection, or turn off 'Include series "
            f"inductance'."
        ) from exc


# Every requested peak is fitted in one least-squares call carrying three
# parameters, so the cost climbs steeply: on a spiky DRT here, 10 peaks took
# 0.4 s, 15 took 13 s, 20 took 89 s, 30 took 285 s, and 114 never returned.
MAX_PEAKS = 15


def analyze_drt_peaks(
    result,
    num_peaks: int = 0,
    disallow_skew: bool = False,
) -> DRTPeaks:
    """Fit individual peaks in a DRT result using skew-normal distributions.
    num_peaks=0 analyzes every detected peak, up to MAX_PEAKS.

    Refuses rather than runs when the distribution resolves more peaks than
    that: the fit is one simultaneous least-squares problem, so an over-fitted
    DRT asks for hundreds of parameters and never comes back."""
    detected = len(result.get_peaks()[0])
    fitting = detected if num_peaks == 0 else min(num_peaks, detected)
    if fitting > MAX_PEAKS:
        raise ValueError(
            f"This DRT resolves {detected} peaks, and fitting {fitting} of them "
            f"at once needs {3 * fitting} free parameters — that does not "
            f"finish in reasonable time.\n\n"
            f"Set Peaks to {MAX_PEAKS} or fewer to fit only the largest, or "
            f"smooth the distribution first: a larger λ, or a smaller FWHM "
            f"coefficient, resolves fewer peaks."
        )
    return result.analyze_peaks(num_peaks=num_peaks, disallow_skew=disallow_skew)
