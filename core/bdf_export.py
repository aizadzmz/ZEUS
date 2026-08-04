#Export analysis output in the Battery Data Alliance's Battery Data Format (BDF)
"""Write sweeps and their analysis results as BDF-aligned CSV plus a JSON-LD
sidecar. https://github.com/battery-data-alliance/battery-data-format"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from core.io_utils import EISDataset

# Terms from the BDF application ontology, verified 2026-08-02 against
# https://battery-data-alliance.github.io/battery-data-format-ontology/battery-data-format.html
# Mapping is machine-readable name -> preferred label, and the preferred label
# is what BDF puts in the header row.
BDF_NAMESPACE = "https://w3id.org/battery-data-alliance/ontology/battery-data-format#"
BDF_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("frequency_hertz", "Frequency / Hz"),
    ("real_impedance_ohm", "Real Impedance / ohm"),
    ("imaginary_impedance_ohm", "Imaginary Impedance / ohm"),
    ("absolute_impedance_ohm", "Absolute Impedance / ohm"),
    ("phase_degree", "Phase / deg"),
)

# Quantities BDF has no term for. Under a URN so they read as app-local and
# can never collide with a real BDF term added later.
LOCAL_NAMESPACE = "urn:eis-batch-analysis:term:"

SPECTRUM_SUFFIX = ".bdf.csv"
SIDECAR_SUFFIX = ".jsonld"

_APP_NAME = "eis-batch-analysis"


# ---- low-level writing helpers ----

def _app_version() -> str:
    """The installed package version, or "unknown" when running from a source
    checkout that was never installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(_APP_NAME)
    except PackageNotFoundError:
        return "unknown"


def _num(value) -> str:
    """Format one number for CSV."""
    number = float(value)
    if not np.isfinite(number):
        return ""
    return repr(number)


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable]) -> Path:
    # newline="" per the csv module's own instruction; without it every row is
    # followed by a blank line on Windows.
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
    return path


def _safe_stem(text: str) -> str:
    r"""Make a filename component safe on Windows, which rejects <>:"/\|?*."""
    cleaned = "".join("_" if c in '<>:"/\\|?*' else c for c in text)
    return cleaned.strip(" .") or "sweep"


def file_stem(dataset: EISDataset, metadata: Optional[dict] = None, sequence: int = 1) -> str:
    """Build the filename stem for one sweep."""
    metadata = metadata or {}
    institution = str(metadata.get("institution_code") or "").strip()
    cell = str(metadata.get("cell_name") or "").strip()
    if institution and cell:
        date = str(metadata.get("date") or datetime.now().strftime("%Y%m%d")).strip()
        return _safe_stem(f"{institution}__{cell}__{date}_{sequence:03d}")
    return _safe_stem(dataset.full_label)


# ---- spectrum ----

def write_spectrum(path: Path, dataset: EISDataset, kept_only: bool = True) -> Path:
    """Write one sweep's impedance spectrum as BDF-aligned CSV."""
    masked = False if kept_only else None
    frequencies = dataset.data.get_frequencies(masked=masked)
    impedances = dataset.data.get_impedances(masked=masked)
    phases = np.angle(impedances, deg=True)

    rows = (
        (_num(f), _num(z.real), _num(z.imag), _num(abs(z)), _num(p))
        for f, z, p in zip(frequencies, impedances, phases)
    )
    return _write_csv(path, [label for _, label in BDF_COLUMNS], rows)


# ---- companion tables (no BDF terms exist for any of these) ----

def write_residuals(path: Path, result) -> Path:
    """Relative residuals from a Kramers-Kronig or Z-HIT result. Both expose
    get_residuals_data() with the same shape, as does an ECM FitResult."""
    frequencies, res_re, res_im = result.get_residuals_data()
    rows = (
        (_num(f), _num(re), _num(im))
        for f, re, im in zip(frequencies, res_re, res_im)
    )
    return _write_csv(
        path,
        [
            "Frequency / Hz",
            "eis:residual_real_percent",
            "eis:residual_imaginary_percent",
        ],
        rows,
    )


def write_ecm_parameters(path: Path, fits: Dict[str, object]) -> Path:
    """Fitted circuit parameters for one sweep, one row per parameter."""
    rows = []
    for cdc in sorted(fits):
        result = fits[cdc]
        for element, parameters in result.get_parameters().items():
            for symbol, parameter in parameters.items():
                rows.append((
                    cdc,
                    element,
                    symbol,
                    _num(parameter.value),
                    _num(parameter.stderr),
                    _num(parameter.get_relative_error() * 100.0),
                    parameter.unit,
                    "true" if parameter.fixed else "false",
                    _num(result.pseudo_chisqr),
                ))
    return _write_csv(
        path,
        [
            "eis:circuit_cdc",
            "eis:element",
            "eis:parameter",
            "eis:value",
            "eis:standard_error",
            "eis:relative_standard_error_percent",
            "eis:unit",
            "eis:fixed",
            "eis:pseudo_chi_squared",
        ],
        rows,
    )


def write_ecm_fit_curve(path: Path, result, num_per_decade: int = 20) -> Path:
    """The fitted circuit's interpolated impedance response, written alongside
    the parameters so the fit can be re-plotted."""
    frequencies = result.get_frequencies(num_per_decade=num_per_decade)
    impedances = result.get_impedances(num_per_decade=num_per_decade)
    rows = (
        (_num(f), _num(z.real), _num(z.imag), _num(abs(z)), _num(p))
        for f, z, p in zip(frequencies, impedances, np.angle(impedances, deg=True))
    )
    return _write_csv(path, [label for _, label in BDF_COLUMNS], rows)


def write_drt(path: Path, result) -> Path:
    """A DRT result as tau vs gamma."""
    data = result.get_drt_data()
    taus, gammas = data[0], data[1]

    header = ["eis:tau_second", "eis:gamma_ohm"]
    columns = [taus, gammas]

    if len(data) == 3:  # BHT
        header = ["eis:tau_second", "eis:gamma_real_ohm", "eis:gamma_imaginary_ohm"]
        columns = [taus, data[1], data[2]]
    else:
        try:
            _, mean, lower, upper = result.get_drt_credible_intervals_data()
        except Exception:
            mean = lower = upper = None
        # An empty array is how pyimpspec reports "no credible intervals" for a
        # non-Bayesian run, so length-check rather than trusting the call alone.
        if mean is not None and len(mean) == len(taus):
            header += ["eis:gamma_mean_ohm", "eis:gamma_lower_ohm", "eis:gamma_upper_ohm"]
            columns += [mean, lower, upper]

    rows = (tuple(_num(v) for v in row) for row in zip(*columns))
    return _write_csv(path, header, rows)


def write_drt_peaks(path: Path, peaks) -> Path:
    """Fitted DRT peaks, one row each."""
    rows = []
    for index, peak in enumerate(peaks.peaks):
        tau = 10 ** ((peak.position * peak.x_scale) + peak.x_offset)
        gamma = (peak.height * peak.y_scale) + peak.y_offset
        rows.append((
            index,
            _num(tau),
            _num(gamma),
            _num(peaks.get_peak_area(index)),
            _num(peak.alpha),
            _num(peak.sigma),
        ))
    return _write_csv(
        path,
        [
            "eis:peak_index",
            "eis:tau_second",
            "eis:gamma_ohm",
            "eis:peak_resistance_ohm",
            "eis:skew_alpha",
            "eis:standard_deviation",
        ],
        rows,
    )


# ---- sidecar ----

def _variable_measured() -> List[dict]:
    """The spectrum's columns as schema.org PropertyValues, each pointing at
    its BDF ontology IRI."""
    return [
        {
            "@type": "PropertyValue",
            "name": label,
            "propertyID": f"{BDF_NAMESPACE}{mr_name}",
            "unitText": label.split("/")[-1].strip(),
        }
        for mr_name, label in BDF_COLUMNS
    ]


def build_sidecar(
    dataset: EISDataset,
    distributions: List[Tuple[str, str]],
    *,
    source_file: Optional[str] = None,
    parser_used: Optional[str] = None,
    analysis: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Build the JSON-LD sidecar for one sweep."""
    total_points = dataset.data.get_num_points(masked=None)
    # masked=False is pyimpspec for "the points that were not omitted".
    kept_points = dataset.data.get_num_points(masked=False)

    document = {
        "@context": {
            "@vocab": "https://schema.org/",
            "bdf": BDF_NAMESPACE,
            "eis": LOCAL_NAMESPACE,
        },
        "@type": "Dataset",
        "name": dataset.qualified_label,
        "description": (
            "Electrochemical impedance spectrum"
            + (f" from {source_file}" if source_file else "")
            + f" ({kept_points} of {total_points} points retained)."
        ),
        "measurementTechnique": "Electrochemical impedance spectroscopy",
        "dateCreated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "creator": {
            "@type": "SoftwareApplication",
            "name": _APP_NAME,
            "softwareVersion": _app_version(),
        },
        "variableMeasured": _variable_measured(),
        "distribution": [
            {
                "@type": "DataDownload",
                "name": name,
                "contentUrl": name,
                "encodingFormat": "text/csv",
                "description": description,
            }
            for name, description in distributions
        ],
        # Provenance, under the local prefix: none of it is a BDF term.
        "eis:provenance": {
            "sourceFile": source_file,
            "parser": parser_used,
            "sweepIndex": dataset.index,
            "sweepLabel": dataset.label,
            "pointsMeasured": total_points,
            "pointsRetained": kept_points,
        },
    }
    if analysis:
        document["eis:analysis"] = analysis

    if metadata:
        # Merged last so a caller can override anything above, and so future
        # cell/instrument fields need no change here.
        document.update(metadata)
    return document


def write_sidecar(path: Path, document: dict) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


# ---- batch entry point ----

def export_batch(
    directory,
    datasets: List[EISDataset],
    *,
    files: Optional[List] = None,
    validation_results: Optional[Dict[Tuple[str, str], object]] = None,
    validation_params: Optional[Dict[Tuple[str, str], dict]] = None,
    drt_results: Optional[Dict[str, object]] = None,
    drt_params: Optional[Dict[str, dict]] = None,
    drt_peaks: Optional[Dict[str, object]] = None,
    ecm_results: Optional[Dict[Tuple[str, str], object]] = None,
    ecm_params: Optional[Dict[Tuple[str, str], dict]] = None,
    metadata: Optional[dict] = None,
    kept_only: bool = True,
) -> List[Path]:
    """Export every sweep, plus whichever analyses have been run, into
    directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    validation_results = validation_results or {}
    validation_params = validation_params or {}
    drt_results = drt_results or {}
    drt_params = drt_params or {}
    drt_peaks = drt_peaks or {}
    ecm_results = ecm_results or {}
    ecm_params = ecm_params or {}

    parser_by_file_id = {lf.file_id: lf.parser_used for lf in (files or [])}
    path_by_file_id = {lf.file_id: lf.path for lf in (files or [])}

    written: List[Path] = []
    used_stems: Dict[str, int] = {}

    for sequence, dataset in enumerate(datasets, start=1):
        stem = file_stem(dataset, metadata, sequence)
        # Two files with the same name can contribute same-labelled sweeps, and
        # full_label is only unique within one file. Disambiguate rather than
        # silently overwrite.
        if stem in used_stems:
            used_stems[stem] += 1
            stem = f"{stem}_{used_stems[stem]}"
        else:
            used_stems[stem] = 1

        distributions: List[Tuple[str, str]] = []
        analysis: dict = {}

        spectrum_name = stem + SPECTRUM_SUFFIX
        written.append(write_spectrum(directory / spectrum_name, dataset, kept_only=kept_only))
        distributions.append((spectrum_name, "Impedance spectrum (BDF-aligned)."))

        # -- validation (Kramers-Kronig / Z-HIT), one file per method run --
        for (method, key), result in sorted(validation_results.items()):
            if key != dataset.key:
                continue
            name = f"{stem}.{method.lower()}.residuals.csv"
            written.append(write_residuals(directory / name, result))
            distributions.append((name, f"{method} relative residuals."))
            entry = {
                "method": method,
                "parameters": _jsonable(validation_params.get((method, key), {})),
            }
            if hasattr(result, "pseudo_chisqr"):
                entry["pseudoChiSquared"] = float(result.pseudo_chisqr)
            analysis.setdefault("validation", []).append(entry)

        # -- ECM: every candidate circuit fitted to this sweep --
        fits = {cdc: r for (cdc, key), r in ecm_results.items() if key == dataset.key}
        if fits:
            name = f"{stem}.ecm.csv"
            written.append(write_ecm_parameters(directory / name, fits))
            distributions.append((name, "Fitted equivalent-circuit parameters."))

            for cdc in sorted(fits):
                result = fits[cdc]
                curve_name = f"{stem}.ecm_fit.{_safe_stem(cdc)}.csv"
                written.append(write_ecm_fit_curve(directory / curve_name, result))
                distributions.append((curve_name, f"Fitted response of {cdc}."))
                analysis.setdefault("equivalentCircuit", []).append({
                    "circuit": cdc,
                    "pseudoChiSquared": float(result.pseudo_chisqr),
                    "method": result.method,
                    "weight": result.weight,
                    "parameters": _jsonable(ecm_params.get((cdc, dataset.key), {})),
                })

        # -- DRT --
        drt_result = drt_results.get(dataset.key)
        if drt_result is not None:
            name = f"{stem}.drt.csv"
            written.append(write_drt(directory / name, drt_result))
            distributions.append((name, "Distribution of relaxation times."))
            analysis["drt"] = {"parameters": _jsonable(drt_params.get(dataset.key, {}))}

        peaks = drt_peaks.get(dataset.key)
        if peaks is not None:
            name = f"{stem}.drt_peaks.csv"
            written.append(write_drt_peaks(directory / name, peaks))
            distributions.append((name, "Fitted DRT peaks."))

        sidecar = build_sidecar(
            dataset,
            distributions,
            source_file=path_by_file_id.get(dataset.file_id, dataset.source_file),
            parser_used=parser_by_file_id.get(dataset.file_id),
            analysis=analysis or None,
            metadata=metadata,
        )
        written.append(write_sidecar(directory / (stem + SIDECAR_SUFFIX), sidecar))

    return written


def _jsonable(value):
    """Coerce stored analysis parameters into something json.dump accepts with
    allow_nan=False."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return str(value)
