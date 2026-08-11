# Save/load an analysis session (datasets + validation + DRT results) as
# gzipped JSON (.eisz).
#
# pyimpspec's result classes (KramersKronigResult, ZHITResult, TRRBFResult,
# DRTPeaks) have no to_dict()/from_dict(), so this module owns that conversion
# rather than pickling, which would tie a session to one pyimpspec class
# layout. Bump SCHEMA_VERSION (with a migration) on any format change.
#
# Gzipped rather than plain JSON: numeric arrays compress ~4-5x, and the gzip
# magic bytes (1f 8b) let load_session still open old plain-JSON sessions.
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

SCHEMA_VERSION = 4
_GZIP_MAGIC = b"\x1f\x8b"


# ---- shared numeric helpers ----

def _real_array(arr) -> list:
    return np.asarray(arr).tolist()


def _complex_array(arr) -> dict:
    arr = np.asarray(arr)
    return {"re": arr.real.tolist(), "im": arr.imag.tolist()}


def _from_complex_array(d: dict) -> np.ndarray:
    return np.asarray(d["re"]) + 1j * np.asarray(d["im"])


# ---- datasets ----

def dataset_to_dict(ds: EISDataset) -> dict:
    return {
        "index": ds.index,
        "source_file": ds.source_file,
        "file_id": ds.file_id,
        "data": ds.data.to_dict(),
    }


def dataset_from_dict(d: dict) -> EISDataset:
    data_dict = dict(d["data"])
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
        "pseudo_chisqr": result.pseudo_chisqr,
        "frequencies": _real_array(result.frequencies),
        "impedances": _complex_array(result.impedances),
        "residuals": _complex_array(result.residuals),
        "test": result.test,
    }


def kk_result_from_dict(d: dict) -> KramersKronigResult:
    return KramersKronigResult(
        circuit=parse_cdc(d["circuit_cdc"]),
        pseudo_chisqr=d["pseudo_chisqr"],
        frequencies=np.asarray(d["frequencies"]),
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
        "pseudo_chisqr": result.pseudo_chisqr,
        "smoothing": result.smoothing,
        "interpolation": result.interpolation,
        "window": result.window,
    }


def zhit_result_from_dict(d: dict) -> ZHITResult:
    return ZHITResult(
        frequencies=np.asarray(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        pseudo_chisqr=d["pseudo_chisqr"],
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
        "pseudo_chisqr": result.pseudo_chisqr,
        "gammas": _real_array(result.gammas),
        "mean_gammas": _real_array(result.mean_gammas),
        "lower_bounds": _real_array(result.lower_bounds),
        "upper_bounds": _real_array(result.upper_bounds),
        "lambda_value": result.lambda_value,
    }


def trrbf_result_from_dict(d: dict) -> TRRBFResult:
    return TRRBFResult(
        time_constants=np.asarray(d["time_constants"]),
        frequencies=np.asarray(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        pseudo_chisqr=d["pseudo_chisqr"],
        gammas=np.asarray(d["gammas"]),
        mean_gammas=np.asarray(d["mean_gammas"]),
        lower_bounds=np.asarray(d["lower_bounds"]),
        upper_bounds=np.asarray(d["upper_bounds"]),
        lambda_value=d["lambda_value"],
    )


_DRT_KIND = {TRRBFResult: "tr_rbf"}
_DRT_TO_DICT = {"tr_rbf": trrbf_result_to_dict}
# Sessions written before the BHT method was dropped may hold a "bht" result.
# Those kinds are skipped on load (see load_session) rather than migrated:
# nothing left in the app can plot or export that shape of result.
_DRT_FROM_DICT = {"tr_rbf": trrbf_result_from_dict}


# ---- DRT peaks ----

def drt_peak_to_dict(peak: DRTPeak) -> dict:
    return {
        "position": float(peak.position),
        "height": float(peak.height),
        "alpha": float(peak.alpha),
        "sigma": float(peak.sigma),
        "x_offset": float(peak.x_offset),
        "x_scale": float(peak.x_scale),
        "y_offset": float(peak.y_offset),
        "y_scale": float(peak.y_scale),
    }


def drt_peak_from_dict(d: dict) -> DRTPeak:
    return DRTPeak(**d)


def drt_peaks_to_dict(result: DRTPeaks) -> dict:
    return {
        "time_constants": _real_array(result.time_constants),
        "peaks": [drt_peak_to_dict(p) for p in result.peaks],
        "suffix": result.suffix,
    }


def drt_peaks_from_dict(d: dict) -> DRTPeaks:
    return DRTPeaks(
        time_constants=np.asarray(d["time_constants"]),
        peaks=[drt_peak_from_dict(p) for p in d["peaks"]],
        suffix=d["suffix"],
    )


# ---- ECM (equivalent circuit fits) ----
# FitResult holds an lmfit MinimizerResult, which is not JSON-serializable.
# Only the seven scalars to_statistics_dataframe() reads are stored, handed
# back via the stand-in class below so that method keeps working on a reloaded
# session. The covariance matrix, parameter correlations, and the ability to
# resume or refine the fit do not survive -- refit if you need them.


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
        # serialize(), not to_string(), so the fitted values travel with the
        # topology and the reloaded circuit reproduces the curve.
        "circuit_cdc": result.circuit.serialize(),
        "parameters": {
            element: {
                symbol: {
                    "value": float(p.value),
                    # stderr is NaN when the fit produced no covariance matrix
                    # (see core.ecm.run_ecm_fit); allow_nan=False has no NaN,
                    # so it travels as None.
                    "stderr": None if p.stderr != p.stderr else float(p.stderr),
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
        "pseudo_chisqr": result.pseudo_chisqr,
        "method": result.method,
        "weight": result.weight,
        "statistics": {
            "chisqr": m.chisqr,
            "redchi": m.redchi,
            "aic": m.aic,
            "bic": m.bic,
            "nfree": m.nfree,
            "ndata": m.ndata,
            "nfev": m.nfev,
        },
    }


def fit_result_from_dict(d: dict) -> FitResult:
    return FitResult(
        circuit=parse_cdc(d["circuit_cdc"]),
        parameters={
            element: {
                symbol: FittedParameter(
                    value=p["value"],
                    stderr=float("nan") if p["stderr"] is None else p["stderr"],
                    fixed=p["fixed"],
                    unit=p["unit"],
                )
                for symbol, p in params.items()
            }
            for element, params in d["parameters"].items()
        },
        minimizer_result=_RestoredMinimizerResult(**d["statistics"]),
        frequencies=np.asarray(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        pseudo_chisqr=d["pseudo_chisqr"],
        method=d["method"],
        weight=d["weight"],
    )


# ---- UI state ----
# What _refresh() needs to rebuild a sweep's mask and the automatic filters
# cannot reconstruct: the eraser's per-point overrides
# (main_window._manual_masked/_manual_kept) plus the filter widget values.
# Without these, touching any sidebar control after a reload would derive a
# different mask than the one saved.
#
# Keyed by ds.key since schema v3, so overrides land on the right sweep in a
# multi-file session (see ui_state_from_dict's v1/v2 fallback). No
# "source_name" field: main_window rebuilds its file list from the datasets'
# file_id/source_file.

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
) -> dict:
    # The advanced mode's fields are optional: a session saved before it
    # existed has none, and load falls back to the widgets' own defaults. So is
    # residual_mode, for the same reason -- and a session predating it was
    # rejected under the modulus convention, which is that default.
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
    }


# ---- whole session ----
# Result dicts are keyed as main_window.py keys its in-memory caches:
# validation_results by (method, ds.key); drt_results and drt_peaks by ds.key.
# ds.key (file_id:index) is unique across files, unlike ds.label ("Set 01").
# dataset_label is stored alongside dataset_key only for humans reading the
# raw JSON; reconstruction uses the key alone.
#
# validation_params/drt_params hold the effective keyword arguments behind each
# result (rbf_type, shape_coeff, lambda_value, admittance, ...), without which
# a reloaded curve cannot be explained or reproduced. Free-form dicts of JSON
# scalars, keyed like their result.

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
                "params": validation_params.get((method, key), {}),
            }
            for (method, key), result in validation_results.items()
        ],
        "drt_results": [
            {
                "dataset_key": key,
                "dataset_label": label_by_key.get(key, key),
                "kind": _DRT_KIND[type(result)],
                "result": _DRT_TO_DICT[_DRT_KIND[type(result)]](result),
                "params": drt_params.get(key, {}),
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
        # Keyed by (circuit, sweep) so several candidate circuits fitted to one
        # sweep all survive (see gui.main_window._ecm_results).
        "ecm_results": [
            {
                "cdc": cdc,
                "dataset_key": key,
                "dataset_label": label_by_key.get(key, key),
                "result": fit_result_to_dict(result),
                "params": ecm_params.get((cdc, key), {}),
            }
            for (cdc, key), result in ecm_results.items()
        ],
        "ui_state": ui_state or {},
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
    if version not in (1, 2, 3, SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported session schema version: {version!r} "
            f"(this build supports {SCHEMA_VERSION})"
        )

    datasets = [dataset_from_dict(d) for d in session["datasets"]]

    # v1/v2 sessions keyed results by "dataset_label" alone, unambiguous
    # because they held one file's sweeps. Map those back to ds.key; v3+
    # sessions carry "dataset_key" and skip this.
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

    # Unknown kinds are dropped rather than raising: an old session may carry a
    # "bht" result from when that method existed, and the rest of it still loads.
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
