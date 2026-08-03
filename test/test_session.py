import numpy as np
from pyimpspec import DataSet

from core.drt import analyze_drt_peaks, run_drt, run_drt_bht
from core.io_utils import EISDataset
from core.session import _GZIP_MAGIC, load_session, save_session, ui_state_to_dict
from core.validation import run_kk_test, run_zhit

# --- Synthetic spectrum: R0 + two parallel RC elements (tau = 5 ms, 300 ms) ---
rng = np.random.default_rng(42)
f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = 10 + 50 / (1 + 1j * w * 5e-3) + 30 / (1 + 1j * w * 0.3)
Z = Z * (1 + 0.001 * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))))

dataset = EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file="synthetic")
dataset.data.set_mask({0: True})  # exercise mask round-trip too

kk_result = run_kk_test(dataset)
zhit_result = run_zhit(dataset)
drt_result = run_drt(dataset)
bht_result = run_drt_bht(dataset, num_samples=200, num_attempts=2)
peaks_result = analyze_drt_peaks(drt_result)

validation_results = {("kk", dataset.key): kk_result, ("zhit", dataset.key): zhit_result}
validation_params = {
    ("kk", dataset.key): {"admittance": False, "num_F_ext_evaluations": 10},
    ("zhit", dataset.key): {},
}
drt_results = {dataset.key: drt_result}
drt_params = {dataset.key: {"rbf_type": "gaussian", "mode": "complex", "shape_coeff": 0.5}}
drt_peaks = {dataset.key: peaks_result}

manual_masked = {dataset.key: {3, 7}}
manual_kept = {dataset.key: {0}}
ui_state = ui_state_to_dict(manual_masked, manual_kept, "kk", True, 2.5)

path = "test/data/_session_roundtrip_tmp.eisz"
save_session(
    path,
    [dataset],
    validation_results,
    drt_results,
    drt_peaks,
    validation_params,
    drt_params,
    ui_state,
)

# --- container: gzip magic bytes ---
with open(path, "rb") as fh:
    magic = fh.read(2)
assert magic == _GZIP_MAGIC, f"expected gzip magic, got {magic!r}"
print("gzip container OK, magic =", magic)

(
    loaded_datasets,
    loaded_validation,
    loaded_drt,
    loaded_peaks,
    loaded_validation_params,
    loaded_drt_params,
    loaded_ui_state,
    loaded_ecm,
    loaded_ecm_params,
) = load_session(path)
# This session saved no circuit fits; test_ecm.py covers the ECM round-trip.
assert loaded_ecm == {} and loaded_ecm_params == {}

# --- dataset ---
loaded_ds = loaded_datasets[0]
assert loaded_ds.source_file == "synthetic"
assert loaded_ds.index == 0
assert np.allclose(loaded_ds.frequencies, dataset.frequencies)
assert np.allclose(loaded_ds.impedances, dataset.impedances)
assert loaded_ds.data.get_mask() == dataset.data.get_mask()
print("dataset round-trip OK, mask =", loaded_ds.data.get_mask())

# --- KK result (including circuit CDC round-trip) ---
loaded_kk = loaded_validation[("kk", dataset.key)]
assert np.allclose(loaded_kk.impedances, kk_result.impedances)
assert loaded_kk.pseudo_chisqr == kk_result.pseudo_chisqr
assert loaded_kk.circuit.to_string() == kk_result.circuit.to_string()
assert loaded_kk.num_RC == kk_result.num_RC  # exercises cached_property post-reconstruction
print("KK round-trip OK, circuit =", loaded_kk.circuit.to_string(), "num_RC =", loaded_kk.num_RC)

# --- ZHIT result ---
loaded_zhit = loaded_validation[("zhit", dataset.key)]
assert np.allclose(loaded_zhit.impedances, zhit_result.impedances)
print("ZHIT round-trip OK")

# --- DRT (TR-RBF) result ---
loaded_drt = loaded_drt[dataset.key]
assert np.allclose(loaded_drt.gammas, drt_result.gammas)
assert loaded_drt.lambda_value == drt_result.lambda_value
print("DRT (TR-RBF) round-trip OK, lambda =", loaded_drt.lambda_value)

# --- DRT peaks ---
loaded_peaks_result = loaded_peaks[dataset.key]
assert len(loaded_peaks_result.peaks) == len(peaks_result.peaks)
assert loaded_peaks_result.peaks[0].position == peaks_result.peaks[0].position
print("DRT peaks round-trip OK, num_peaks =", len(loaded_peaks_result.peaks))

# --- analysis params ---
assert loaded_validation_params[("kk", dataset.key)] == validation_params[("kk", dataset.key)]
assert loaded_validation_params[("zhit", dataset.key)] == {}
assert loaded_drt_params[dataset.key] == drt_params[dataset.key]
print("params round-trip OK:", loaded_drt_params[dataset.key])

# --- ui_state: eraser overrides, filter settings ---
assert loaded_ui_state["manual_masked"][dataset.key] == manual_masked[dataset.key]
assert loaded_ui_state["manual_kept"][dataset.key] == manual_kept[dataset.key]
assert loaded_ui_state["validation_method"] == "kk"
assert loaded_ui_state["inductive_filter"] is True
assert loaded_ui_state["residual_threshold"] == 2.5
print("ui_state round-trip OK, manual_masked =", loaded_ui_state["manual_masked"][dataset.key])

# --- backward compatibility: a v1 (plain-JSON, no params/ui_state) session still loads ---
import json
from pathlib import Path

from core.session import (
    dataset_to_dict,
    drt_peaks_to_dict,
    kk_result_to_dict,
    trrbf_result_to_dict,
)

v1_session = {
    "schema_version": 1,
    "datasets": [dataset_to_dict(dataset)],
    "validation_results": [
        {
            "method": "kk",
            "dataset_label": dataset.label,
            "kind": "kk",
            "result": kk_result_to_dict(kk_result),
        }
    ],
    "drt_results": [
        {
            "dataset_label": dataset.label,
            "kind": "tr_rbf",
            "result": trrbf_result_to_dict(drt_result),
        }
    ],
    "drt_peaks": [
        {"dataset_label": dataset.label, "peaks": drt_peaks_to_dict(peaks_result)}
    ],
}
v1_path = "test/data/_session_v1_tmp.json"
Path(v1_path).write_text(json.dumps(v1_session))

(
    v1_datasets,
    v1_validation,
    v1_drt,
    v1_peaks,
    v1_validation_params,
    v1_drt_params,
    v1_ui_state,
    v1_ecm,
    v1_ecm_params,
) = load_session(v1_path)
# ECM fits arrived in schema v4; a v1 session simply has none.
assert v1_ecm == {} and v1_ecm_params == {}
assert v1_datasets[0].source_file == "synthetic"
assert v1_datasets[0].file_id == 0  # absent in v1 -> defaults to the single-file assumption
# v1 sessions stored "dataset_label" ("Set 01") rather than a real key; on
# load that gets resolved back to the now-canonical ds.key (see
# load_session's _resolve_key), which for this single-dataset session is
# "0:0" -- not dataset.label itself.
v1_key = v1_datasets[0].key
assert v1_validation_params == {("kk", v1_key): {}}
assert v1_drt_params == {v1_key: {}}
assert v1_ui_state["manual_masked"] == {}
assert v1_ui_state["inductive_filter"] is False
print("v1 backward-compatibility OK, resolved label -> key:", dataset.label, "->", v1_key)

# --- multi-file: two files whose sweeps share the same label ("Set 01") ---
# must not collide -- this is the whole reason results are keyed by
# ds.key (file_id:index) rather than ds.label. See core.io_utils.EISDataset.
dataset_b = EISDataset(DataSet(frequencies=f, impedances=Z * 1.1), index=0, source_file="synthetic_b", file_id=1)
assert dataset_b.label == dataset.label == "Set 01"
assert dataset_b.key != dataset.key

kk_result_b = run_kk_test(dataset_b)
mf_validation_results = {
    ("kk", dataset.key): kk_result,
    ("kk", dataset_b.key): kk_result_b,
}
mf_path = "test/data/_session_multifile_tmp.eisz"
save_session(mf_path, [dataset, dataset_b], mf_validation_results, {}, {})

(mf_datasets, mf_validation, _, _, _, _, _, _, _) = load_session(mf_path)
mf_by_key = {d.key: d for d in mf_datasets}
assert mf_by_key[dataset.key].source_file == "synthetic"
assert mf_by_key[dataset_b.key].source_file == "synthetic_b"
assert np.allclose(mf_validation[("kk", dataset.key)].impedances, kk_result.impedances)
assert np.allclose(mf_validation[("kk", dataset_b.key)].impedances, kk_result_b.impedances)
Path(mf_path).unlink()
print("multi-file round-trip OK: two 'Set 01' sweeps from different files kept separate results")

print("ALL SESSION ROUND-TRIP CHECKS PASSED")
