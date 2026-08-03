# Save/load an analysis session (datasets + validation + DRT results) as
# gzipped JSON (.eisz).
#
# pyimpspec's analysis result classes (KramersKronigResult, ZHITResult,
# TRRBFResult, BHTResult, DRTPeaks) have no to_dict()/from_dict() of their
# own, so this module owns that conversion instead of pickling the objects.
# Pickling would tie a saved session to the exact pyimpspec class layout it
# was created with; this schema is versioned and survives library upgrades
# as long as SCHEMA_VERSION is bumped (with a migration) when the format
# changes.
#
# The file itself is gzipped JSON rather than plain JSON: numeric arrays
# compress ~4-5x, and gzip's magic bytes (1f 8b) let load_session tell a
# compressed file from a plain one, so old plain-JSON sessions still open.
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyimpspec import DataSet, KramersKronigResult, ZHITResult, parse_cdc
from pyimpspec.analysis.drt import BHTResult, TRRBFResult
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
        # Absent in schema v1/v2 sessions, which predate multi-file support
        # -- every dataset they ever saved had the implicit file_id=0.
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


# ---- DRT: BHT ----

def bht_result_to_dict(result: BHTResult) -> dict:
    return {
        "time_constants": _real_array(result.time_constants),
        "frequencies": _real_array(result.frequencies),
        "impedances": _complex_array(result.impedances),
        "residuals": _complex_array(result.residuals),
        "pseudo_chisqr": result.pseudo_chisqr,
        "real_gammas": _real_array(result.real_gammas),
        "imaginary_gammas": _real_array(result.imaginary_gammas),
        "scores": {k: {"re": v.real, "im": v.imag} for k, v in result.scores.items()},
    }


def bht_result_from_dict(d: dict) -> BHTResult:
    return BHTResult(
        time_constants=np.asarray(d["time_constants"]),
        frequencies=np.asarray(d["frequencies"]),
        impedances=_from_complex_array(d["impedances"]),
        residuals=_from_complex_array(d["residuals"]),
        pseudo_chisqr=d["pseudo_chisqr"],
        real_gammas=np.asarray(d["real_gammas"]),
        imaginary_gammas=np.asarray(d["imaginary_gammas"]),
        scores={k: complex(v["re"], v["im"]) for k, v in d["scores"].items()},
    )


_DRT_KIND = {TRRBFResult: "tr_rbf", BHTResult: "bht"}
_DRT_TO_DICT = {"tr_rbf": trrbf_result_to_dict, "bht": bht_result_to_dict}
_DRT_FROM_DICT = {"tr_rbf": trrbf_result_from_dict, "bht": bht_result_from_dict}


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
# FitResult is the one result type here that holds a live third-party object:
# an lmfit MinimizerResult, which is not JSON-serializable. Everything the app
# actually reads off it is a handful of scalars (FitResult.to_statistics_
# dataframe uses exactly the seven below), so those are stored and handed back
# via the stand-in class -- which keeps to_statistics_dataframe() working on a
# reloaded session instead of raising on a None. What does not survive is the
# rest of lmfit's machinery: the covariance matrix, the parameter correlations,
# and the ability to resume or refine the fit. Reloading gives you the fitted
# curve, its parameters with their error bars, and its statistics; refitting is
# what you do if you need more than that.


class _RestoredMinimizerResult:
    """The scalars FitResult reads off lmfit's MinimizerResult, restored from
    a saved session. Not an lmfit object and not a substitute for one -- see
    the note above."""

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
        # serialize() (not to_string()) so the fitted values travel with the
        # topology -- that is what makes the reloaded circuit reproduce the
        # curve rather than just its shape.
        "circuit_cdc": result.circuit.serialize(),
        "parameters": {
            element: {
                symbol: {
                    "value": float(p.value),
                    # stderr is NaN whenever the fitting method produced no
                    # covariance matrix (see core.ecm.run_ecm_fit). JSON has no
                    # NaN under allow_nan=False, so it travels as None.
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
# Everything _refresh() reads to decide what a sweep's mask looks like, and
# that the automatic filters alone can't reconstruct: the eraser's per-point
# overrides (main_window._manual_masked/_manual_kept -- see the comment at
# their definition for why those live outside DataSet.mask) plus the filter
# widget values. Without this, reloading a session and touching any sidebar
# control would silently re-derive a different mask than the one saved.
#
# manual_masked/manual_kept are keyed by ds.key (not ds.label) as of schema
# v3, so overrides land on the right sweep when a session has more than one
# source file -- see ui_state_from_dict's v1/v2 fallback below. There is no
# "source_name" field as of v3: with several files loaded there's no single
# name to store, and it's not needed anyway -- gui.main_window rebuilds its
# file list straight from the loaded datasets' file_id/source_file.

def ui_state_to_dict(
    manual_masked: Dict[str, set],
    manual_kept: Dict[str, set],
    validation_method: str,
    inductive_filter: bool,
    residual_threshold: float,
) -> dict:
    return {
        "manual_masked": {key: sorted(idxs) for key, idxs in manual_masked.items()},
        "manual_kept": {key: sorted(idxs) for key, idxs in manual_kept.items()},
        "validation_method": validation_method,
        "inductive_filter": inductive_filter,
        "residual_threshold": residual_threshold,
    }


def ui_state_from_dict(d: dict, key_by_label: Optional[Dict[str, str]] = None) -> dict:
    """key_by_label resolves a v1/v2 session's label-keyed manual_masked/
    manual_kept back to ds.key -- see load_session's own _resolve_key for why
    that mapping is safe (every pre-v3 session had exactly one file). Omit it
    (or leave a label unresolved) and the raw label passes through instead,
    which is only reachable from a hand-edited or corrupt session file."""
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
    }


# ---- whole session ----
# Result dicts are keyed the same way main_window.py keys its in-memory
# caches: validation_results by (method, ds.key), drt_results and drt_peaks
# by ds.key. ds.key (file_id:index) is unique across every loaded file,
# unlike ds.label ("Set 01") which only a single-file session could get away
# with -- see core.io_utils.EISDataset. Each entry also carries dataset_label
# alongside dataset_key purely for a human reading the raw JSON; only the key
# is used to reconstruct the in-memory dicts.
#
# validation_params/drt_params carry the effective keyword arguments behind
# each result (e.g. rbf_type, shape_coeff, lambda_value, admittance) --
# without them a reloaded curve can be viewed but not explained or
# reproduced. They're free-form dicts of JSON scalars, keyed the same way as
# their corresponding result.

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
        # Keyed by (circuit, sweep) rather than sweep alone, so several
        # candidate circuits fitted to one sweep all survive -- see
        # gui.main_window._ecm_results.
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

    # v1/v2 sessions predate multi-file support and keyed every result by
    # "dataset_label" alone (e.g. "Set 01"), which was safe back then because
    # a session could only ever hold one file's worth of sweeps. Resolve
    # those old labels back to the now-canonical ds.key via the datasets just
    # loaded -- unambiguous for the same reason. v3+ sessions carry
    # "dataset_key" directly and skip this entirely.
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

    drt_results = {
        _resolve_key(d): _DRT_FROM_DICT[d["kind"]](d["result"])
        for d in session["drt_results"]
    }
    drt_params = {
        _resolve_key(d): d.get("params", {})
        for d in session["drt_results"]
    }

    drt_peaks = {
        _resolve_key(p): drt_peaks_from_dict(p["peaks"])
        for p in session["drt_peaks"]
    }

    # .get(), unlike the blocks above: ECM fits arrived in schema v4, so a
    # v1-v3 session simply has none rather than a malformed entry.
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
