# Save/load an analysis session (datasets + validation + DRT results) as gzipped JSON (.eisz); bump SCHEMA_VERSION with a migration on any format change.
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyimpspec import DataSet, KramersKronigResult, ZHITResult, parse_cdc
from pyimpspec.analysis.drt import TRRBFResult
from pyimpspec.analysis.drt.peak_analysis import DRTPeak, DRTPeaks
from pyimpspec.analysis.fitting import FitResult, FittedParameter

from core.io_utils import EISDataset

SCHEMA_VERSION = 5
_GZIP_MAGIC = b"\x1f\x8b"


# ---- shared numeric helpers ----
# The writer runs with allow_nan=False, so nothing here may hand json a NaN or
# an infinity: JSON has no spelling for either and json.dumps raises rather than
# emitting one. They travel as null and are read back as NaN, the same round
# trip fit_result_to_dict already makes for an unavailable standard error. A
# result that failed to converge is exactly the one worth saving, so a NaN in it
# must not be what makes the whole session unsaveable.


def _number(value) -> Optional[float]:
    """One float on its way into the session file, or None where JSON cannot
    hold it."""
    value = float(value)
    return value if np.isfinite(value) else None


def _from_number(value) -> float:
    """The inverse of _number: null comes back as the NaN it stood for."""
    return float("nan") if value is None else float(value)


def _real_array(arr) -> list:
    return [_number(v) for v in np.asarray(arr, dtype=float).tolist()]


def _from_real_array(values) -> np.ndarray:
    # dtype=float, so the nulls _real_array wrote come back as NaN rather than
    # leaving an object array that no downstream arithmetic can use.
    return np.asarray([_from_number(v) for v in values], dtype=float)


def _complex_array(arr) -> dict:
    arr = np.asarray(arr)
    return {"re": _real_array(arr.real), "im": _real_array(arr.imag)}


def _from_complex_array(d: dict) -> np.ndarray:
    return _from_real_array(d["re"]) + 1j * _from_real_array(d["im"])


def _jsonable(value):
    """Coerce a stored parameter dict into something json.dump accepts. These
    are recorded verbatim from the widgets and are only ever displayed or
    exported, so a lossy conversion costs nothing -- unlike the result arrays
    above, which have to survive the round trip intact."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        # Before the sequence check: an ndarray is not a list, and the str()
        # fallback would quietly turn one into "[1. 2. 3.]".
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # Before the bool/int check below: np.bool_ is not a bool and np.float64 is
    # not a float, so an unconverted scalar reaches json as an unknown type.
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _number(value)
    return str(value)


# ---- datasets ----

# The three numeric lists in DataSet.to_dict(); nulled and restored like every
# other array here, so one NaN point cannot make a session unsaveable.
_DATASET_ARRAYS = ("frequencies", "real_impedances", "imaginary_impedances")


def dataset_to_dict(ds: EISDataset) -> dict:
    data_dict = dict(ds.data.to_dict())
    for key in _DATASET_ARRAYS:
        if key in data_dict:
            data_dict[key] = _real_array(data_dict[key])
    return {
        "index": ds.index,
        "source_file": ds.source_file,
        "file_id": ds.file_id,
        "data": data_dict,
    }


def dataset_from_dict(d: dict) -> EISDataset:
    data_dict = dict(d["data"])
    for key in _DATASET_ARRAYS:
        if key in data_dict:
            data_dict[key] = _from_real_array(data_dict[key]).tolist()
    mask = data_dict.get("mask")
    if mask:
        # JSON round-trips int dict keys as strings; DataSet expects int keys.
        data_dict["mask"] = {int(k): v for k, v in mask.items()}
    return EISDataset(
        DataSet.from_dict(data_dict),
        index=d["index"],
        source_file=d["source_file"],
        # Absent in schema v1/v2, which predate multi-file support.
        file_id=d.get("file_id", 0),
    )


# ---- Kramers-Kronig ----

def kk_result_to_dict(result: KramersKronigResult) -> dict:
    return {
        "circuit_cdc": result.circuit.serialize(),
        "pseudo_chisqr": _number(result.pseudo_chisqr),
        "frequencies": _real_array(result.frequencies),
        "impedances": _complex_array(result.impedances),
        "residuals": _complex_array(result.residuals),
        "test": result.test,
    }


def kk_result_from_dict(d: dict) -> KramersKronigResult:
    return KramersKronigResult(
        circuit=parse_cdc(d["circuit_cdc"]),
        pseudo_chisqr=_from_number(d["pseudo_chisqr"]),
        frequencies=_from_real_array(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        test=d["test"],
    )


# ---- Z-HIT ----

def zhit_result_to_dict(result: ZHITResult) -> dict:
    return {
        "frequencies": _real_array(result.frequencies),
        "impedances": _complex_array(result.impedances),
        "residuals": _complex_array(result.residuals),
        "pseudo_chisqr": _number(result.pseudo_chisqr),
        "smoothing": result.smoothing,
        "interpolation": result.interpolation,
        "window": result.window,
    }


def zhit_result_from_dict(d: dict) -> ZHITResult:
    return ZHITResult(
        frequencies=_from_real_array(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        pseudo_chisqr=_from_number(d["pseudo_chisqr"]),
        smoothing=d["smoothing"],
        interpolation=d["interpolation"],
        window=d["window"],
    )


_VALIDATION_KIND = {KramersKronigResult: "kk", ZHITResult: "zhit"}
_VALIDATION_TO_DICT = {"kk": kk_result_to_dict, "zhit": zhit_result_to_dict}
_VALIDATION_FROM_DICT = {"kk": kk_result_from_dict, "zhit": zhit_result_from_dict}


# ---- DRT: TR-RBF ----

def trrbf_result_to_dict(result: TRRBFResult) -> dict:
    return {
        "time_constants": _real_array(result.time_constants),
        "frequencies": _real_array(result.frequencies),
        "impedances": _complex_array(result.impedances),
        "residuals": _complex_array(result.residuals),
        "pseudo_chisqr": _number(result.pseudo_chisqr),
        "gammas": _real_array(result.gammas),
        "mean_gammas": _real_array(result.mean_gammas),
        "lower_bounds": _real_array(result.lower_bounds),
        "upper_bounds": _real_array(result.upper_bounds),
        "lambda_value": _number(result.lambda_value),
    }


def trrbf_result_from_dict(d: dict) -> TRRBFResult:
    return TRRBFResult(
        time_constants=_from_real_array(d["time_constants"]),
        frequencies=_from_real_array(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        pseudo_chisqr=_from_number(d["pseudo_chisqr"]),
        gammas=_from_real_array(d["gammas"]),
        mean_gammas=_from_real_array(d["mean_gammas"]),
        lower_bounds=_from_real_array(d["lower_bounds"]),
        upper_bounds=_from_real_array(d["upper_bounds"]),
        lambda_value=_from_number(d["lambda_value"]),
    )


_DRT_KIND = {TRRBFResult: "tr_rbf"}
_DRT_TO_DICT = {"tr_rbf": trrbf_result_to_dict}
# Sessions written before the BHT method was dropped may hold a "bht" result; those kinds are skipped on load rather than migrated.
_DRT_FROM_DICT = {"tr_rbf": trrbf_result_from_dict}


# ---- DRT peaks ----

_DRT_PEAK_FIELDS = (
    "position", "height", "alpha", "sigma",
    "x_offset", "x_scale", "y_offset", "y_scale",
)


def drt_peak_to_dict(peak: DRTPeak) -> dict:
    return {field: _number(getattr(peak, field)) for field in _DRT_PEAK_FIELDS}


def drt_peak_from_dict(d: dict) -> DRTPeak:
    return DRTPeak(**{field: _from_number(d[field]) for field in _DRT_PEAK_FIELDS})


def drt_peaks_to_dict(result: DRTPeaks) -> dict:
    return {
        "time_constants": _real_array(result.time_constants),
        "peaks": [drt_peak_to_dict(p) for p in result.peaks],
        "suffix": result.suffix,
    }


def drt_peaks_from_dict(d: dict) -> DRTPeaks:
    return DRTPeaks(
        time_constants=_from_real_array(d["time_constants"]),
        peaks=[drt_peak_from_dict(p) for p in d["peaks"]],
        suffix=d["suffix"],
    )


# ---- ECM (equivalent circuit fits) ----
# Only the seven scalars to_statistics_dataframe() reads are stored; the covariance matrix and the ability to resume a fit do not survive.


class _RestoredMinimizerResult:
    """The scalars FitResult reads off lmfit's MinimizerResult, restored from a
    saved session. Not an lmfit object or a substitute for one."""

    def __init__(self, chisqr, redchi, aic, bic, nfree, ndata, nfev):
        self.chisqr = chisqr
        self.redchi = redchi
        self.aic = aic
        self.bic = bic
        self.nfree = nfree
        self.ndata = ndata
        self.nfev = nfev


def fit_result_to_dict(result: FitResult) -> dict:
    m = result.minimizer_result
    return {
        # serialize(), not to_string(), so the fitted values travel with the topology.
        "circuit_cdc": result.circuit.serialize(),
        "parameters": {
            element: {
                symbol: {
                    "value": _number(p.value),
                    # stderr is NaN when the fit produced no covariance matrix, which is what makes it the commonest null in the file.
                    "stderr": _number(p.stderr),
                    "fixed": bool(p.fixed),
                    "unit": p.unit,
                }
                for symbol, p in params.items()
            }
            for element, params in result.parameters.items()
        },
        "frequencies": _real_array(result.frequencies),
        "impedances": _complex_array(result.impedances),
        "residuals": _complex_array(result.residuals),
        "pseudo_chisqr": _number(result.pseudo_chisqr),
        "method": result.method,
        "weight": result.weight,
        # aic/bic go infinite on a perfect fit and redchi is NaN without free
        # parameters, so these are nulled like every other float here; the three
        # counts are ints and cannot be.
        "statistics": {
            "chisqr": _number(m.chisqr),
            "redchi": _number(m.redchi),
            "aic": _number(m.aic),
            "bic": _number(m.bic),
            "nfree": int(m.nfree),
            "ndata": int(m.ndata),
            "nfev": int(m.nfev),
        },
    }


def fit_result_from_dict(d: dict) -> FitResult:
    return FitResult(
        circuit=parse_cdc(d["circuit_cdc"]),
        parameters={
            element: {
                symbol: FittedParameter(
                    value=_from_number(p["value"]),
                    stderr=_from_number(p["stderr"]),
                    fixed=p["fixed"],
                    unit=p["unit"],
                )
                for symbol, p in params.items()
            }
            for element, params in d["parameters"].items()
        },
        minimizer_result=_RestoredMinimizerResult(
            **{
                key: value if key in ("nfree", "ndata", "nfev") else _from_number(value)
                for key, value in d["statistics"].items()
            }
        ),
        frequencies=_from_real_array(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        pseudo_chisqr=_from_number(d["pseudo_chisqr"]),
        method=d["method"],
        weight=d["weight"],
    )


# ---- UI state ----
# What _refresh() cannot reconstruct: the eraser's per-point overrides plus the filter widget values, keyed by ds.key since schema v3.

def ui_state_to_dict(
    manual_masked: Dict[str, set],
    manual_kept: Dict[str, set],
    validation_method: str,
    inductive_filter: bool,
    residual_threshold: float,
    validation_mode: str = "Basic",
    soft_limit: Optional[float] = None,
    hard_limit: Optional[float] = None,
    max_removed: Optional[int] = None,
    residual_mode: Optional[str] = None,
    drt_inductive_filter: bool = False,
    drt_subtract_diffusion: bool = False,
    drt_diffusion_cdc: Optional[str] = None,
) -> dict:
    # The advanced-mode fields and residual_mode are optional, falling back to the widgets' own defaults; the three drt_* fields are the DRT step's filters, not the validation step's.
    return {
        "manual_masked": {key: sorted(idxs) for key, idxs in manual_masked.items()},
        "manual_kept": {key: sorted(idxs) for key, idxs in manual_kept.items()},
        "validation_method": validation_method,
        "inductive_filter": inductive_filter,
        "residual_threshold": residual_threshold,
        "validation_mode": validation_mode,
        "soft_limit": soft_limit,
        "hard_limit": hard_limit,
        "max_removed": max_removed,
        "residual_mode": residual_mode,
        "drt_inductive_filter": drt_inductive_filter,
        "drt_subtract_diffusion": drt_subtract_diffusion,
        "drt_diffusion_cdc": drt_diffusion_cdc,
    }


def ui_state_from_dict(d: dict, key_by_label: Optional[Dict[str, str]] = None) -> dict:
    """Rebuild ui_state from a saved session; key_by_label remaps a v1/v2
    session's label-keyed overrides to ds.key."""
    key_by_label = key_by_label or {}

    def _remap(overrides: dict) -> Dict[str, set]:
        return {
            key_by_label.get(label, label): set(idxs)
            for label, idxs in overrides.items()
        }

    return {
        "manual_masked": _remap(d.get("manual_masked", {})),
        "manual_kept": _remap(d.get("manual_kept", {})),
        "validation_method": d.get("validation_method"),
        "inductive_filter": d.get("inductive_filter", False),
        "residual_threshold": d.get("residual_threshold"),
        "validation_mode": d.get("validation_mode", "Basic"),
        "soft_limit": d.get("soft_limit"),
        "hard_limit": d.get("hard_limit"),
        "max_removed": d.get("max_removed"),
        "residual_mode": d.get("residual_mode"),
        # Absent in schema v1-v4, where the DRT filters were not saved at all; False/None is how those sessions behaved.
        "drt_inductive_filter": d.get("drt_inductive_filter", False),
        "drt_subtract_diffusion": d.get("drt_subtract_diffusion", False),
        "drt_diffusion_cdc": d.get("drt_diffusion_cdc"),
    }


# ---- whole session ----
# Result dicts are keyed as main_window keys its caches, by ds.key (file_id:index); the *_params dicts hold the effective keyword arguments behind each result.

def save_session(
    path,
    datasets: List[EISDataset],
    validation_results: Dict[Tuple[str, str], Any],
    drt_results: Dict[str, Any],
    drt_peaks: Dict[str, DRTPeaks],
    validation_params: Dict[Tuple[str, str], dict] = None,
    drt_params: Dict[str, dict] = None,
    ui_state: dict = None,
    ecm_results: Dict[Tuple[str, str], FitResult] = None,
    ecm_params: Dict[Tuple[str, str], dict] = None,
) -> None:
    validation_params = validation_params or {}
    drt_params = drt_params or {}
    ecm_results = ecm_results or {}
    ecm_params = ecm_params or {}
    label_by_key = {ds.key: ds.label for ds in datasets}

    session = {
        "schema_version": SCHEMA_VERSION,
        "datasets": [dataset_to_dict(ds) for ds in datasets],
        "validation_results": [
            {
                "method": method,
                "dataset_key": key,
                "dataset_label": label_by_key.get(key, key),
                "kind": _VALIDATION_KIND[type(result)],
                "result": _VALIDATION_TO_DICT[_VALIDATION_KIND[type(result)]](result),
                "params": _jsonable(validation_params.get((method, key), {})),
            }
            for (method, key), result in validation_results.items()
        ],
        "drt_results": [
            {
                "dataset_key": key,
                "dataset_label": label_by_key.get(key, key),
                "kind": _DRT_KIND[type(result)],
                "result": _DRT_TO_DICT[_DRT_KIND[type(result)]](result),
                "params": _jsonable(drt_params.get(key, {})),
            }
            for key, result in drt_results.items()
        ],
        "drt_peaks": [
            {
                "dataset_key": key,
                "dataset_label": label_by_key.get(key, key),
                "peaks": drt_peaks_to_dict(peaks),
            }
            for key, peaks in drt_peaks.items()
        ],
        # Keyed by (circuit, sweep) so several candidate circuits fitted to one sweep all survive.
        "ecm_results": [
            {
                "cdc": cdc,
                "dataset_key": key,
                "dataset_label": label_by_key.get(key, key),
                "result": fit_result_to_dict(result),
                "params": _jsonable(ecm_params.get((cdc, key), {})),
            }
            for (cdc, key), result in ecm_results.items()
        ],
        # Widget values, so finite in practice -- sanitized anyway, since one
        # unwritable number here would cost the user the whole session.
        "ui_state": _jsonable(ui_state or {}),
    }
    payload = json.dumps(session, separators=(",", ":"), allow_nan=False)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(payload)


def load_session(
    path,
) -> Tuple[
    List[EISDataset],
    Dict[Tuple[str, str], Any],
    Dict[str, Any],
    Dict[str, DRTPeaks],
    Dict[Tuple[str, str], dict],
    Dict[str, dict],
    dict,
    Dict[Tuple[str, str], FitResult],
    Dict[Tuple[str, str], dict],
]:
    raw = Path(path).read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if raw[:2] == _GZIP_MAGIC else raw.decode("utf-8")
    session = json.loads(text)

    version = session.get("schema_version")
    if version not in (1, 2, 3, 4, SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported session schema version: {version!r} "
            f"(this build supports {SCHEMA_VERSION})"
        )

    datasets = [dataset_from_dict(d) for d in session["datasets"]]

    # v1/v2 sessions keyed results by "dataset_label" alone; map those back to ds.key, which v3+ carries directly.
    key_by_label = {ds.label: ds.key for ds in datasets}

    def _resolve_key(entry: dict) -> str:
        if "dataset_key" in entry:
            return entry["dataset_key"]
        label = entry["dataset_label"]
        return key_by_label.get(label, label)

    validation_results = {
        (v["method"], _resolve_key(v)): _VALIDATION_FROM_DICT[v["kind"]](v["result"])
        for v in session["validation_results"]
    }
    validation_params = {
        (v["method"], _resolve_key(v)): v.get("params", {})
        for v in session["validation_results"]
    }

    # Unknown kinds are dropped rather than raising, so an old "bht" result does not stop the rest loading.
    drt_entries = [d for d in session["drt_results"] if d["kind"] in _DRT_FROM_DICT]
    drt_results = {
        _resolve_key(d): _DRT_FROM_DICT[d["kind"]](d["result"])
        for d in drt_entries
    }
    drt_params = {
        _resolve_key(d): d.get("params", {})
        for d in drt_entries
    }

    drt_peaks = {
        _resolve_key(p): drt_peaks_from_dict(p["peaks"])
        for p in session["drt_peaks"]
    }

    # .get(): ECM fits arrived in schema v4, so a v1-v3 session has none.
    ecm_entries = session.get("ecm_results", [])
    ecm_results = {
        (e["cdc"], _resolve_key(e)): fit_result_from_dict(e["result"])
        for e in ecm_entries
    }
    ecm_params = {
        (e["cdc"], _resolve_key(e)): e.get("params", {})
        for e in ecm_entries
    }

    ui_state = ui_state_from_dict(session.get("ui_state", {}), key_by_label)

    return (
        datasets,
        validation_results,
        drt_results,
        drt_peaks,
        validation_params,
        drt_params,
        ui_state,
        ecm_results,
        ecm_params,
    )
