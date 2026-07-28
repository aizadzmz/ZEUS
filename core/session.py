# Save/load an analysis session (datasets + validation + DRT results) as JSON.
#
# pyimpspec's analysis result classes (KramersKronigResult, ZHITResult,
# TRRBFResult, BHTResult, DRTPeaks) have no to_dict()/from_dict() of their
# own, so this module owns that conversion instead of pickling the objects.
# Pickling would tie a saved session to the exact pyimpspec class layout it
# was created with; this schema is versioned and survives library upgrades
# as long as SCHEMA_VERSION is bumped (with a migration) when the format
# changes.
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pyimpspec import DataSet, KramersKronigResult, ZHITResult, parse_cdc
from pyimpspec.analysis.drt import BHTResult, TRRBFResult
from pyimpspec.analysis.drt.peak_analysis import DRTPeak, DRTPeaks

from core.io_utils import EISDataset

SCHEMA_VERSION = 1


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


# ---- whole session ----
# Result dicts are keyed the same way main_window.py keys its in-memory
# caches: validation_results by (method, dataset_label), drt_results and
# drt_peaks by dataset_label. Labels are stable (derived from dataset index),
# so they round-trip correctly as long as dataset order/index is preserved.

def save_session(
    path,
    datasets: List[EISDataset],
    validation_results: Dict[Tuple[str, str], Any],
    drt_results: Dict[str, Any],
    drt_peaks: Dict[str, DRTPeaks],
) -> None:
    session = {
        "schema_version": SCHEMA_VERSION,
        "datasets": [dataset_to_dict(ds) for ds in datasets],
        "validation_results": [
            {
                "method": method,
                "dataset_label": label,
                "kind": _VALIDATION_KIND[type(result)],
                "result": _VALIDATION_TO_DICT[_VALIDATION_KIND[type(result)]](result),
            }
            for (method, label), result in validation_results.items()
        ],
        "drt_results": [
            {
                "dataset_label": label,
                "kind": _DRT_KIND[type(result)],
                "result": _DRT_TO_DICT[_DRT_KIND[type(result)]](result),
            }
            for label, result in drt_results.items()
        ],
        "drt_peaks": [
            {"dataset_label": label, "peaks": drt_peaks_to_dict(peaks)}
            for label, peaks in drt_peaks.items()
        ],
    }
    Path(path).write_text(json.dumps(session, indent=2))


def load_session(
    path,
) -> Tuple[List[EISDataset], Dict[Tuple[str, str], Any], Dict[str, Any], Dict[str, DRTPeaks]]:
    session = json.loads(Path(path).read_text())

    version = session.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported session schema version: {version!r} "
            f"(this build supports {SCHEMA_VERSION})"
        )

    datasets = [dataset_from_dict(d) for d in session["datasets"]]

    validation_results = {
        (v["method"], v["dataset_label"]): _VALIDATION_FROM_DICT[v["kind"]](v["result"])
        for v in session["validation_results"]
    }

    drt_results = {
        d["dataset_label"]: _DRT_FROM_DICT[d["kind"]](d["result"])
        for d in session["drt_results"]
    }

    drt_peaks = {
        p["dataset_label"]: drt_peaks_from_dict(p["peaks"])
        for p in session["drt_peaks"]
    }

    return datasets, validation_results, drt_results, drt_peaks
