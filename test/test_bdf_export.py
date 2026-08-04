"""Export tests for core.bdf_export."""
import csv
import json

import numpy as np
import pytest
from pyimpspec import DataSet, parse_cdc

from core.bdf_export import BDF_COLUMNS, export_batch, write_spectrum
from core.ecm import run_ecm_fit
from core.io_utils import EISDataset
from core.validation import run_kk_test


@pytest.fixture
def dataset():
    """R0 + 2xRC, lightly noised, as a single EISDataset."""
    f = np.logspace(5, -2, 40)
    circuit = parse_cdc("R{R=10}(R{R=100}C{C=1e-5})(R{R=250}C{C=1e-2})")
    Z = circuit.get_impedances(f)
    rng = np.random.default_rng(0)
    Z = Z * (1 + rng.normal(0, 0.001, Z.shape) + 1j * rng.normal(0, 0.001, Z.shape))
    return EISDataset(DataSet(f, Z), index=0, source_file="synthetic", file_id=0)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


# ---- spectrum ----

def test_header_uses_bdf_preferred_labels(tmp_path, dataset):
    rows = read_csv(write_spectrum(tmp_path / "s.bdf.csv", dataset))
    assert rows[0] == [
        "Frequency / Hz",
        "Real Impedance / ohm",
        "Imaginary Impedance / ohm",
        "Absolute Impedance / ohm",
        "Phase / deg",
    ]
    assert rows[0] == [label for _, label in BDF_COLUMNS]


def test_values_round_trip(tmp_path, dataset):
    rows = read_csv(write_spectrum(tmp_path / "s.bdf.csv", dataset))[1:]
    assert len(rows) == 40

    frequencies = np.array([float(r[0]) for r in rows])
    real = np.array([float(r[1]) for r in rows])
    imag = np.array([float(r[2]) for r in rows])
    magnitude = np.array([float(r[3]) for r in rows])
    phase = np.array([float(r[4]) for r in rows])

    assert np.allclose(frequencies, dataset.frequencies)
    assert np.allclose(real + 1j * imag, dataset.impedances)
    # Magnitude and phase are derived, so check them against the definition.
    assert np.allclose(magnitude, np.abs(dataset.impedances))
    assert np.allclose(phase, np.angle(dataset.impedances, deg=True))


def test_export_honors_the_mask(tmp_path, dataset):
    """The default must write the points still on screen, not the erased ones;
    pyimpspec's `masked` flag is inverted relative to how it reads."""
    mask = dataset.data.get_mask()
    for i in range(5):
        mask[i] = True
    dataset.data.set_mask(mask)

    kept = read_csv(write_spectrum(tmp_path / "kept.bdf.csv", dataset))[1:]
    assert len(kept) == 35

    exported = np.array([float(r[0]) for r in kept])
    erased = dataset.data.get_frequencies(masked=True)
    assert not np.any(np.isin(exported, erased)), "erased points leaked into the export"

    every = read_csv(write_spectrum(tmp_path / "all.bdf.csv", dataset, kept_only=False))[1:]
    assert len(every) == 40


# ---- batch + sidecar ----

def test_batch_writes_spectrum_and_sidecar(tmp_path, dataset):
    written = export_batch(tmp_path, [dataset])
    names = sorted(p.name for p in written)
    assert names == ["synthetic_set01.bdf.csv", "synthetic_set01.jsonld"]


def test_sidecar_omits_unknown_metadata(tmp_path, dataset):
    export_batch(tmp_path, [dataset])
    doc = json.loads((tmp_path / "synthetic_set01.jsonld").read_text(encoding="utf-8"))

    assert doc["@type"] == "Dataset"
    # Absent, not blank: a blank field invites a reader to trust it.
    for absent in ("chemistry", "ambientTemperature", "instrument"):
        assert absent not in doc

    ids = {v["propertyID"] for v in doc["variableMeasured"]}
    assert "https://w3id.org/battery-data-alliance/ontology/battery-data-format#frequency_hertz" in ids
    assert doc["eis:provenance"]["pointsRetained"] == 40


def test_caller_metadata_is_merged_and_names_the_file(tmp_path, dataset):
    export_batch(
        tmp_path,
        [dataset],
        metadata={
            "institution_code": "UCam",
            "cell_name": "A0001",
            "date": "20241031",
            "chemistry": "NMC811/Gr",
        },
    )
    spectrum = tmp_path / "UCam__A0001__20241031_001.bdf.csv"
    assert spectrum.exists(), "BDF naming convention not applied"

    doc = json.loads((tmp_path / "UCam__A0001__20241031_001.jsonld").read_text(encoding="utf-8"))
    assert doc["chemistry"] == "NMC811/Gr"


def test_duplicate_labels_do_not_overwrite(tmp_path, dataset):
    """Two files can contribute same-labelled sweeps; full_label is only unique
    within one file."""
    other = EISDataset(dataset.data, index=0, source_file="synthetic", file_id=1)
    written = export_batch(tmp_path, [dataset, other])
    spectra = sorted(p.name for p in written if p.name.endswith(".bdf.csv"))
    assert len(spectra) == 2 and len(set(spectra)) == 2


# ---- companion tables ----

def test_companions_for_each_analysis(tmp_path, dataset):
    kk = run_kk_test(dataset)
    fit = run_ecm_fit(dataset, "R(RC)(RC)")

    written = export_batch(
        tmp_path,
        [dataset],
        validation_results={("Kramers-Kronig", dataset.key): kk},
        validation_params={("Kramers-Kronig", dataset.key): {"admittance": False}},
        ecm_results={("[R(RC)(RC)]", dataset.key): fit},
        ecm_params={("[R(RC)(RC)]", dataset.key): {"method": "least_squares"}},
    )
    names = {p.name for p in written}
    assert "synthetic_set01.kramers-kronig.residuals.csv" in names
    assert "synthetic_set01.ecm.csv" in names

    ecm = read_csv(tmp_path / "synthetic_set01.ecm.csv")
    assert ecm[0][0] == "eis:circuit_cdc"
    # Five free parameters across R + 2 RC pairs.
    assert len(ecm) - 1 == 5

    doc = json.loads((tmp_path / "synthetic_set01.jsonld").read_text(encoding="utf-8"))
    circuits = doc["eis:analysis"]["equivalentCircuit"]
    assert circuits[0]["circuit"] == "[R(RC)(RC)]"
    assert circuits[0]["pseudoChiSquared"] < 1e-2
    # Every companion is discoverable from the sidecar.
    listed = {d["name"] for d in doc["distribution"]}
    assert "synthetic_set01.ecm.csv" in listed


def test_drt_and_peaks_companions(tmp_path, dataset):
    """Exercises the DRT writers, including the pandas-free reconstruction of
    each peak's tau/gamma from its relative position and height."""
    from core.drt import analyze_drt_peaks, run_drt

    drt = run_drt(dataset)
    peaks = analyze_drt_peaks(drt)

    export_batch(
        tmp_path,
        [dataset],
        drt_results={dataset.key: drt},
        drt_params={dataset.key: {"rbf_type": "gaussian"}},
        drt_peaks={dataset.key: peaks},
    )

    curve = read_csv(tmp_path / "synthetic_set01.drt.csv")
    assert curve[0][:2] == ["eis:tau_second", "eis:gamma_ohm"]
    taus = np.array([float(r[0]) for r in curve[1:]])
    assert len(taus) > 1 and np.all(taus > 0)

    rows = read_csv(tmp_path / "synthetic_set01.drt_peaks.csv")
    assert rows[0][1:3] == ["eis:tau_second", "eis:gamma_ohm"]
    assert len(rows) - 1 == peaks.get_num_peaks()

    # tau/gamma must match what pyimpspec itself reports for those peaks.
    reference = peaks.to_peaks_dataframe()
    assert np.allclose(
        [float(r[1]) for r in rows[1:]], reference["tau (s)"].to_numpy()
    )
    assert np.allclose(
        [float(r[2]) for r in rows[1:]], reference["gamma (ohm)"].to_numpy()
    )


def test_nan_standard_error_becomes_an_empty_cell(tmp_path, dataset):
    """Gradient-free methods produce no covariance matrix, so stderr is NaN.
    JSON/CSV should say "no value", not the string "nan"."""
    fit = run_ecm_fit(dataset, "R(RC)(RC)", method="powell")
    export_batch(tmp_path, [dataset], ecm_results={("[R(RC)(RC)]", dataset.key): fit})

    rows = read_csv(tmp_path / "synthetic_set01.ecm.csv")[1:]
    stderrs = [r[4] for r in rows]
    assert all(s == "" or np.isfinite(float(s)) for s in stderrs)
    assert "nan" not in [s.lower() for s in stderrs]


# ---- conformance against the reference implementation ----

def test_every_column_is_a_recognized_bdf_term(tmp_path, dataset):
    """Read the export back with the reference batterydf package."""
    bdf = pytest.importorskip("bdf", reason="batterydf not installed (dev extra)")

    path = write_spectrum(tmp_path / "conformance.bdf.csv", dataset)
    # validate=False so we can inspect the report rather than take the raise.
    frame = bdf.read(str(path), validate=False)

    # The file is well-formed BDF CSV: the reference reader recovers every row
    # and column, with the values intact.
    assert len(frame) == 40
    assert list(frame.columns) == [label for _, label in BDF_COLUMNS]
    assert np.allclose(frame["Frequency / Hz"].to_numpy(), dataset.frequencies)
    assert np.allclose(frame["Real Impedance / ohm"].to_numpy(), dataset.impedances.real)

    report = bdf.validate_df(frame, raise_on_error=False)
    assert report["legacy_labels"] == [], "using deprecated BDF labels"
    assert sorted(report["missing"]) == sorted(
        ["Test Time / s", "Voltage / V", "Current / A"]
    ), "the only missing-column complaint should be BDF v1's time-series trio"


def test_known_gap_between_batterydf_and_the_ontology(tmp_path, dataset):
    """Characterization test, pinning a discrepancy in the upstream project."""
    bdf = pytest.importorskip("bdf", reason="batterydf not installed (dev extra)")

    path = write_spectrum(tmp_path / "gap.bdf.csv", dataset)
    report = bdf.validate_df(bdf.read(str(path), validate=False), raise_on_error=False)

    assert sorted(report["extras"]) == sorted(
        [label for _, label in BDF_COLUMNS]
    ), "batterydf's column table changed -- re-check it against the ontology"
