#ECM (equivalent circuit modelling)
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

# pyimpspec is imported per function, so the GUI can read the option tuples below free.
if TYPE_CHECKING:
    from pyimpspec.analysis.fitting import FitResult

# (menu label, CDC) in Boukamp's syntax: brackets series, parentheses parallel, outermost brackets implicit -- hence canonical_cdc() output is what gets stored.
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

# Mass-transport (diffusion) elements, by pyimpspec's own symbols; read by core.filtering.
DIFFUSION_SYMBOLS = frozenset(
    {
        "W", "Ws", "Wo",           # Warburg: semi-infinite, transmissive, reflective
        "G", "Ga",                 # Gerischer
        "Ls",                      # de Levie
        "Tlm", "Tlmbo", "Tlmbq", "Tlmbs", "Tlmno", "Tlmnq", "Tlmns",
    }
)

# (menu label, CDC) for the DRT step's tail subtraction; each ends in a diffusion element in *series*, which is what makes subtracting it exact.
DIFFUSION_PRESETS = (
    ("Warburg (semi-infinite)", "R(RQ)W"),
    ("Warburg, transmissive", "R(RQ)Ws"),
    ("Warburg, reflective", "R(RQ)Wo"),
    ("Gerischer", "R(RQ)G"),
)

DEFAULT_DIFFUSION_CDC = DIFFUSION_PRESETS[0][1]

# "auto" is offered but deliberately not the default -- see run_ecm_fit.
FIT_METHODS = ("least_squares", "auto", "leastsq", "powell", "nelder", "cg")
WEIGHT_FORMS = ("boukamp", "auto", "modulus", "proportional", "unity")

DEFAULT_METHOD = FIT_METHODS[0]
DEFAULT_WEIGHT = WEIGHT_FORMS[0]


def canonical_cdc(cdc: str) -> str:
    """Return pyimpspec's canonical spelling of a circuit description code."""
    from pyimpspec import parse_cdc

    return parse_cdc(cdc).to_string()


def validate_cdc(cdc: str) -> Tuple[bool, str]:
    """Check whether a CDC parses, for the sidebar's live-validation label."""
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
    """Fit an equivalent circuit to the dataset's currently unmasked points by
    complex non-linear least squares (CNLS), via pyimpspec.fit_circuit."""
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
    """Rewrite a CDC so its initial values are a previous fit's fitted values,
    for chaining a fit along a series of sweeps."""
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
    """run_ecm_fit with the previous sweep's result as the starting guess."""
    return run_ecm_fit(dataset, seed_cdc(cdc, previous), **kwargs)


def ohmic_resistance(frequencies, impedances) -> float:
    """Re(Z) where the spectrum crosses the real axis, scanning down from the
    highest frequency. Falls back to the first point when the sweep never
    crosses, which a cell with cabling inductance always does.
    """
    import numpy as np

    order = np.argsort(frequencies)[::-1]  # high -> low
    real = np.asarray(impedances).real[order]
    imag = np.asarray(impedances).imag[order]

    crossings = np.flatnonzero(np.sign(imag[:-1]) * np.sign(imag[1:]) < 0)
    if len(crossings) == 0:
        return float(real[0])

    # Interpolated between the straddling points: adjacent points sit far apart relative to the polarisation resistance being estimated.
    i = crossings[0]
    span = imag[i] - imag[i + 1]
    if span == 0:
        return float(real[i])
    return float(real[i] + (imag[i] / span) * (real[i + 1] - real[i]))


def series_resistance(dataset) -> float:
    """Estimate a sweep's series (ohmic) resistance. See ohmic_resistance for
    why this is the axis crossing rather than the first point."""
    frequencies = dataset.frequencies
    if len(frequencies) == 0:
        raise ValueError("Dataset has no unmasked points.")
    return ohmic_resistance(frequencies, dataset.impedances)


def _element_term(symbol: str, **values) -> str:
    """One element as CDC text, with a default limit widened wherever a value
    read off the DRT falls outside it.

    fit_circuit rejects an initial value outside its element's limits, so
    without this the built circuit parses but cannot be fitted -- and the
    limits are generic defaults, not physics: a slow peak with a small
    resistance puts C = tau/R above the 1000 F ceiling on ordinary data.
    """
    from pyimpspec import parse_cdc

    element = parse_cdc(symbol).get_elements()[0]
    parts = []
    for key, value in values.items():
        lower = element.get_lower_limit(key)
        upper = element.get_upper_limit(key)
        if lower <= value <= upper:
            parts.append(f"{key}={value:.6g}")
            continue
        # Only the side actually breached moves, by a decade, so the fit has
        # room to leave the value it starts on. Scaling rather than offsetting
        # keeps the sign: a negative floor under a capacitance means nothing.
        if value > upper:
            upper = value * 10 if value > 0 else value / 10
        else:
            lower = value / 10 if value > 0 else value * 10
        parts.append(f"{key}={value:.6g}/{lower:.6g}/{upper:.6g}")
    return f"{symbol}{{{','.join(parts)}}}"


def circuit_from_drt_peaks(
    peaks,
    r_series: float,
    use_cpe: bool = True,
    cpe_exponent: float = 0.9,
    min_area_fraction: float = 0.005,
) -> str:
    """Build an extended CDC from a DRT peak analysis: one parallel R-C or
    R-CPE pair per resolved peak, in series with the ohmic resistance."""
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

    terms = [_element_term("R", R=r_series)]
    for tau, area in kept:
        if use_cpe:
            element = _element_term("Q", Y=tau**cpe_exponent / area, n=cpe_exponent)
        else:
            element = _element_term("C", C=tau / area)
        terms.append(f"({_element_term('R', R=area)}{element})")
    return "".join(terms)


def format_fit_report(result: FitResult, label: str) -> str:
    """Render one fit as the plain text shown in the ECM Parameters tab,
    mirroring how the DRT Peaks tab renders DRTPeaks.to_peaks_dataframe()."""
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
