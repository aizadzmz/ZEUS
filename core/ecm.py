#ECM (equivalent circuit modelling)
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

# pyimpspec is imported inside the functions rather than here, and the
# annotations below are strings thanks to the __future__ import. That is what
# lets the GUI read the option tuples while building its sidebar without
# paying ~4 s for pyimpspec (scipy.signal, sympy); see core/__init__.py.
if TYPE_CHECKING:
    from pyimpspec.analysis.fitting import FitResult

# (menu label, circuit description code). CDC syntax is Boukamp's: square
# brackets are series connections, parentheses are parallel ones, so R(RC) is
# a resistor in series with a parallel RC pair. The outermost series brackets
# are implicit -- pyimpspec's canonical form adds them back, which is why
# canonical_cdc() below is what gets stored rather than these strings.
CIRCUIT_PRESETS = (
    ("Randles", "R(RW)"),
    ("Randles + double layer", "R(C[RW])"),
    ("Simplified Randles (CPE)", "R(QR)"),
    ("Single RC", "R(RC)"),
    ("Two time constants (CPE)", "R(RQ)(RQ)"),
    ("Two RC", "R(RC)(RC)"),
    ("Series inductance + two CPE", "LR(RQ)(RQ)"),
    ("Finite-length Warburg", "R(Q[RWs])"),
)

# "auto" is offered but deliberately not the default -- see run_ecm_fit.
FIT_METHODS = ("least_squares", "auto", "leastsq", "powell", "nelder", "cg")
WEIGHT_FORMS = ("boukamp", "auto", "modulus", "proportional", "unity")

DEFAULT_METHOD = FIT_METHODS[0]
DEFAULT_WEIGHT = WEIGHT_FORMS[0]


def canonical_cdc(cdc: str) -> str:
    """
    Return pyimpspec's canonical spelling of a circuit description code.

    Results are cached under a (circuit, sweep) key, so the circuit half of
    that key has to be stable against harmless typing differences: "R(RC)",
    "R( RC )" and "[R(RC)]" are one circuit and must not produce three cache
    entries. Parsing and re-emitting normalizes all three to "[R(RC)]".

    Note this drops any initial values, bounds, or labels carried by the
    extended CDC syntax -- two fits that differ only in their starting guess
    share a key, and the later one wins. That is intentional: they are the
    same model, and keeping both would defeat the point of the comparison.
    """
    from pyimpspec import parse_cdc

    return parse_cdc(cdc).to_string()


def validate_cdc(cdc: str) -> Tuple[bool, str]:
    """
    Check whether a CDC parses, for the sidebar's live-validation label.

    Returns (ok, message). On success the message summarizes the circuit so
    the user can sanity-check what they typed before spending a batch on it;
    on failure it is pyimpspec's own parser error, which points at the
    offending character and is more useful than anything paraphrased here.
    """
    cdc = cdc.strip()
    if not cdc:
        return False, "Enter a circuit description code."

    from pyimpspec import parse_cdc

    try:
        circuit = parse_cdc(cdc)
    except Exception as exc:
        return False, str(exc)

    num_elements = len(circuit.get_elements())
    num_params = sum(len(e.get_values()) for e in circuit.get_elements())
    if num_params == 0:
        return False, "Circuit has no fittable parameters."
    return True, f"valid - {num_elements} element(s), {num_params} parameter(s)"


def run_ecm_fit(
    dataset,
    cdc: str,
    method: str = DEFAULT_METHOD,
    weight: str = DEFAULT_WEIGHT,
    max_nfev: int = -1,
    timeout: int = 0,
) -> FitResult:
    """
    Fit an equivalent circuit to the dataset's currently unmasked points by
    complex non-linear least squares (CNLS), via pyimpspec.fit_circuit.

    The circuit is passed as a CDC string rather than a Circuit object so
    this function stays picklable by reference -- the same constraint
    core.validation.run_kk_test documents, which keeps it eligible for
    ValidationWorker's process pool should batch fitting ever need it.
    The extended CDC syntax is accepted here too, so a caller can pin
    initial values or bounds by passing e.g.
    "R{R=5}(R{R=100/1/1e4}C)" -- that is how run_ecm_fit_seeded works.

    Two defaults are narrowed from pyimpspec's, both away from "auto":

    method="least_squares" (pyimpspec defaults to "auto"): "auto" fits with
    every available method and keeps whichever scores the best pseudo
    chi-squared. It costs several times as much -- measured at ~0.4 s vs
    ~0.02 s per sweep for R(RC)(RC) on 40 points, which matters across a
    50-sweep batch -- and, more importantly, it regularly *wins* with a
    gradient-free method (Powell, Nelder-Mead). Those produce no covariance
    matrix, so every fitted parameter comes back with a NaN standard error,
    for a chi-squared indistinguishable from the least-squares fit's. Error
    bars are most of the point of reporting circuit parameters, so the
    default keeps them. Switch to "auto" for a spectrum least_squares
    cannot converge on, and expect to lose the error bars if it picks a
    gradient-free winner.

    weight="boukamp" (pyimpspec defaults to "auto"): weighting each residual
    by 1/|Z|^2 is the standard choice in the EIS literature (Boukamp, 1995)
    and stops the high-impedance end of the spectrum from dominating the fit
    -- impedances routinely span decades, so an unweighted fit is effectively
    a fit to the largest few points. "auto" tries every weighting as well,
    compounding the cost above.

    max_nfev=-1 leaves the iteration count uncapped; timeout=0 disables the
    per-fit time limit. num_procs is pinned to 1 rather than passed through:
    pyimpspec would otherwise spawn its own process pool inside whatever
    worker thread or process is already running this, which oversubscribes
    the machine on a batch and has no benefit for a single fast fit.

    The result exposes:
      - get_impedances(num_per_decade=N) -> smooth interpolated fit curve
      - get_nyquist_data() / get_bode_data() -> the fit at the measured points
      - get_residuals_data() -> (freq, res_re, res_im), the same shape
        core.plotting.build_residuals_plot consumes
      - to_parameters_dataframe() / to_statistics_dataframe()
      - pseudo_chisqr: the fit-quality scalar used to rank circuits
    """
    from pyimpspec import fit_circuit, parse_cdc

    return fit_circuit(
        parse_cdc(cdc),
        dataset.data,
        method=method,
        weight=weight,
        max_nfev=max_nfev,
        timeout=timeout,
        num_procs=1,
    )


def seed_cdc(cdc: str, previous: Optional[FitResult]) -> str:
    """
    Rewrite a CDC so its initial values are a previous fit's fitted values,
    for chaining a fit along a series of sweeps.

    Returns the plain cdc unchanged when there is nothing to seed from, or
    when the previous fit was of a different circuit -- element-by-element
    seeding is only meaningful between identical topologies, and a mismatch
    here means the user changed the CDC mid-batch rather than that anything
    went wrong. Values transfer positionally: parse_cdc builds elements in a
    fixed order for a given string, so the same string twice gives two lists
    that line up.

    The seeded circuit is handed back as an extended CDC string rather than
    a Circuit object to keep run_ecm_fit's picklable-string interface. The
    round trip through to_string(12) carries values *and* bounds at twelve
    decimals, which is lossless enough that re-parsing reproduces the fit.
    """
    if previous is None:
        return cdc

    from pyimpspec import parse_cdc

    circuit = parse_cdc(cdc)
    if circuit.to_string() != previous.circuit.to_string():
        return cdc

    for element, fitted in zip(circuit.get_elements(), previous.circuit.get_elements()):
        element.set_values(**fitted.get_values())
    return circuit.to_string(12)


def run_ecm_fit_seeded(
    dataset,
    cdc: str,
    previous: Optional[FitResult] = None,
    **kwargs,
) -> FitResult:
    """
    run_ecm_fit with the previous sweep's result as the starting guess.

    Worth it on a cycling or degradation series, where consecutive spectra
    are near-identical and pyimpspec's generic defaults (R=1 kOhm, C=1 uF)
    may be decades away from the truth for every sweep. The cost is that one
    bad fit seeds the next, so this is opt-in rather than the default.
    """
    return run_ecm_fit(dataset, seed_cdc(cdc, previous), **kwargs)


def series_resistance(dataset) -> float:
    """
    Estimate the series (ohmic) resistance as Re(Z) at the highest measured
    frequency -- the classic high-frequency intercept, and the starting guess
    for the R that sits in front of the RC/RQ pairs.

    Reads the highest frequency by argmax rather than taking the first point:
    EISDataset yields its points high-to-low, but that ordering is a property
    of the parser rather than something this needs to depend on.
    """
    import numpy as np

    frequencies = dataset.frequencies
    if len(frequencies) == 0:
        raise ValueError("Dataset has no unmasked points.")
    return float(dataset.impedances[np.argmax(frequencies)].real)


def circuit_from_drt_peaks(
    peaks,
    r_series: float,
    use_cpe: bool = True,
    cpe_exponent: float = 0.9,
    min_area_fraction: float = 0.005,
) -> str:
    """
    Build an extended CDC from a DRT peak analysis: one parallel R-C or R-CPE
    pair per resolved peak, in series with the ohmic resistance.

    This is the step that turns a DRT into a circuit hypothesis. The DRT
    already answers the question an ECM cannot -- *how many* distinct
    processes there are and where they sit -- so its peaks supply both the
    topology and the initial values, in place of guessing a preset and
    letting pyimpspec start every parameter from its generic default (1 kOhm,
    1 uF) regardless of the data.

    The mapping, per peak, using the quantities analyze_drt_peaks already
    reports (see DRTPeaks.to_peaks_dataframe: "tau (s)" and "R_peak (ohm)")::

        R = the peak's area, which is its polarization resistance
        C = tau / R                       (use_cpe=False)
        Y = tau**n / R,  n = cpe_exponent (use_cpe=True)

    Y follows from a parallel RQ's characteristic time constant,
    tau = (R * Y) ** (1 / n), and reduces to C = tau / R at n = 1.

    use_cpe=True by default because a DRT peak has finite width precisely
    when the process is distributed, which is the same thing that depresses
    the semicircle an ideal capacitor then cannot fit. n is only an initial
    value and is left free, so it relaxes toward 1 by itself where the
    process really is ideal.

    min_area_fraction drops peaks whose resistance is below that fraction of
    the total polarization resistance. Peak fitting routinely returns a
    couple of numerically-negligible peaks at the edges of the time-constant
    range -- on a clean synthetic spectrum, two at ~1e-7 Ohm against real
    peaks of 51 and 31 Ohm -- and each one kept would add two free parameters
    that fit nothing. Pass 0.0 to keep every peak.

    Raises ValueError when no peak clears the threshold, since an ECM with no
    RC elements is not a circuit worth fitting.
    """
    num_peaks = peaks.get_num_peaks()
    areas = [peaks.get_peak_area(i) for i in range(num_peaks)]
    taus = peaks.to_peaks_dataframe()["tau (s)"].tolist()

    total = sum(a for a in areas if a > 0)
    cutoff = total * min_area_fraction
    kept = [
        (tau, area)
        for tau, area in zip(taus, areas)
        if area > 0 and area >= cutoff
    ]
    if not kept:
        raise ValueError(
            f"No DRT peak carries a meaningful resistance "
            f"(largest was {max(areas, default=0.0):.3g} ohm). "
            f"Try a peak analysis with a different number of peaks."
        )

    terms = [f"R{{R={r_series:.6g}}}"]
    for tau, area in kept:
        if use_cpe:
            admittance = tau**cpe_exponent / area
            element = f"Q{{Y={admittance:.6g},n={cpe_exponent:.6g}}}"
        else:
            element = f"C{{C={tau / area:.6g}}}"
        terms.append(f"(R{{R={area:.6g}}}{element})")
    return "".join(terms)


def format_fit_report(result: FitResult, label: str) -> str:
    """
    Render one fit as the plain text shown in the ECM Parameters tab,
    mirroring how the DRT Peaks tab renders DRTPeaks.to_peaks_dataframe().

    A NaN in the "Std. err. (%)" column is not a failure -- it means the
    fitting method produced no covariance matrix (see run_ecm_fit).
    """
    circuit = result.circuit.to_string()
    lines = [
        f"=== {label} - {circuit} ===",
        f"pseudo chi-squared: {result.pseudo_chisqr:.6g}"
        f"   (method: {result.method}, weight: {result.weight})",
        "",
        result.to_parameters_dataframe().to_string(index=False),
        "",
        result.to_statistics_dataframe().to_string(index=False),
        "",
    ]
    return "\n".join(lines)
