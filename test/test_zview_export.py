"""Export tests for core.zview_export.

The formats are undocumented, so the structural assertions here are pinned
against the files ZView ships (C:\\SAI). Those are only present where ZView is
installed, hence the skips in the conformance section at the bottom -- the rest
of the file runs everywhere.
"""
import re
from pathlib import Path

import numpy as np
import pytest
from pyimpspec import DataSet, parse_cdc

from core.io_utils import EISDataset
from core.zview_export import (
    estimate_series_resistance,
    peak_rc_pairs,
    write_drt_model,
    write_drt_z,
)

ZVIEW_SAMPLES = Path("C:/SAI")


@pytest.fixture(scope="module")
def dataset():
    """R0 + 2xRC, so the DRT has two peaks to deconvolve."""
    f = np.logspace(5, -2, 40)
    circuit = parse_cdc("R{R=10}(R{R=100}C{C=1e-5})(R{R=250}C{C=1e-2})")
    return EISDataset(
        DataSet(f, circuit.get_impedances(f)),
        index=0,
        source_file="synthetic",
        file_id=0,
    )


@pytest.fixture(scope="module")
def drt(dataset):
    from core.drt import run_drt

    return run_drt(dataset)


@pytest.fixture(scope="module")
def peaks(drt):
    from core.drt import analyze_drt_peaks

    return analyze_drt_peaks(drt)


def read_z(path):
    """(header lines, data lines) of a .z file."""
    text = path.read_bytes().decode("ascii")
    lines = text.split("\r\n")[:-1]  # trailing CRLF, not a line
    return lines[:11], lines[11:]


# ---- data file ----

def test_z_header_is_exactly_eleven_lines(tmp_path, drt):
    """ZView reads the point count from line 10 and the first row from line 12,
    so the header length is part of the format."""
    header, rows = read_z(write_drt_z(tmp_path / "a.z", drt))

    assert header[0] == '"ZView Calculated Data File: Version 1.1"'
    assert header[7] == '"Frequency"'
    assert int(header[9]) == len(rows), "line 10 must be the point count"
    assert header[10].startswith('"  Freq (Hz)')


def test_z_uses_crlf(tmp_path, drt):
    raw = (write_drt_z(tmp_path / "a.z", drt)).read_bytes()
    assert b"\r\n" in raw
    assert b"\r\r\n" not in raw, "newline translation doubled the CR"
    assert raw.count(b"\n") == raw.count(b"\r\n"), "a bare LF slipped in"


def test_z_rows_have_nine_fields_with_four_digit_exponents(tmp_path, drt):
    _, rows = read_z(write_drt_z(tmp_path / "a.z", drt))

    for row in rows:
        fields = [f.strip() for f in row.split(",")]
        assert len(fields) == 9
        # The seven floats, then Err and Range as bare integers.
        for field in fields[:7]:
            assert re.fullmatch(r"-?\d\.\d{6}E[+-]\d{4}", field), field
        assert fields[7:] == ["0", "0"]


def test_z_carries_the_drt_as_frequency_and_real(tmp_path, drt):
    """tau -> f = 1/(2*pi*tau), gamma -> Z'(a), with Z''(b) left at zero."""
    taus, gammas = drt.get_drt_data()[:2]
    _, rows = read_z(write_drt_z(tmp_path / "a.z", drt))

    values = np.array([[float(f) for f in row.split(",")[:6]] for row in rows])
    assert np.allclose(values[:, 0], 1.0 / (2 * np.pi * taus), rtol=1e-5)
    assert np.allclose(values[:, 4], gammas, rtol=1e-5)
    assert np.all(values[:, 5] == 0.0)


def test_z_frequency_descends(tmp_path, drt):
    """Ascending tau has to come out as descending frequency, the order ZView's
    own files use."""
    _, rows = read_z(write_drt_z(tmp_path / "a.z", drt))
    frequencies = np.array([float(row.split(",")[0]) for row in rows])
    assert np.all(np.diff(frequencies) < 0)


# ---- model file ----

def test_model_is_one_rc_pair_per_peak(tmp_path, peaks):
    text = write_drt_model(tmp_path / "m.mdl", peaks).read_bytes().decode("ascii")
    pairs = peak_rc_pairs(peaks)
    assert len(pairs) >= 2, "the two RC branches should survive as peaks"

    # Rs, then four elements (open, C, R, close) for each pair.
    expected = 1 + 4 * len(pairs)
    assert f"Begin Circuit Model:          {expected}" in text
    assert f"End Circuit Model:            {expected}" in text
    assert text.count("Begin_Parallel") == len(pairs)
    assert text.count("End_Parallel") == len(pairs)


def test_model_elements_are_numbered_without_gaps(tmp_path, peaks):
    text = write_drt_model(tmp_path / "m.mdl", peaks).read_bytes().decode("ascii")
    numbers = [int(n) for n in re.findall(r"Element #(\d+) Type:", text)]
    assert numbers == list(range(len(numbers)))


def test_model_values_are_the_peaks(tmp_path, peaks):
    """R is the peak area and C = tau/R, so the RC pair reproduces the peak's
    time constant."""
    text = write_drt_model(tmp_path / "m.mdl", peaks).read_bytes().decode("ascii")

    resistances = [
        float(v) for v in re.findall(r"Element #\d+,0 Value:\s+(\S+)", text)
    ]
    capacitances = [
        float(v) for v in re.findall(r"Element #\d+,1 Value:\s+(\S+)", text)
    ]
    pairs = peak_rc_pairs(peaks)

    # The first resistance is Rs, which is not a peak.
    assert np.allclose(resistances[1:], [r for r, _ in pairs], rtol=1e-5)
    assert np.allclose(capacitances, [c for _, c in pairs], rtol=1e-5)

    peak_taus = np.array(resistances[1:]) * np.array(capacitances)
    reference = peaks.to_peaks_dataframe()["tau (s)"].to_numpy()
    assert np.allclose(sorted(peak_taus), sorted(reference[: len(peak_taus)]), rtol=1e-3)


def test_series_resistance_seeds_rs(tmp_path, dataset, peaks):
    rs = estimate_series_resistance(dataset)
    # The circuit's own R0, recovered from the high-frequency end.
    assert rs == pytest.approx(10.0, rel=1e-3)

    text = (
        write_drt_model(tmp_path / "m.mdl", peaks, series_resistance=rs)
        .read_bytes()
        .decode("ascii")
    )
    assert "Element #0 Name:            Rs" in text
    first = float(re.search(r"Element #0,0 Value:\s+(\S+)", text).group(1))
    assert first == pytest.approx(rs, rel=1e-5)


def test_model_without_usable_peaks_refuses(tmp_path):
    class _Empty:
        peaks = ()

    with pytest.raises(ValueError, match="no circuit"):
        write_drt_model(tmp_path / "m.mdl", _Empty())


# ---- conformance against ZView's own files ----

@pytest.mark.skipif(
    not (ZVIEW_SAMPLES / "ZData/Example Demo Data/DEMO1.Z").exists(),
    reason="ZView is not installed, so its reference files are unavailable",
)
def test_z_matches_zviews_own_data_file(tmp_path, drt):
    """Line-for-line structural comparison with a file ZView wrote."""
    reference = (ZVIEW_SAMPLES / "ZData/Example Demo Data/DEMO1.Z").read_bytes()
    ref_lines = reference.decode("ascii").split("\r\n")
    ours, rows = read_z(write_drt_z(tmp_path / "a.z", drt))

    for line in (0, 1, 2, 7, 10):
        assert ours[line] == ref_lines[line], f"line {line + 1} differs"
    # Line 9 is a fixed prefix plus the frequency limits.
    assert ours[8].startswith("0,2,0,1,")
    assert ref_lines[8].startswith("0,2,0,1,")

    # A reference row and ours must agree field for field in shape.
    def shape(row):
        return [
            "num" if re.fullmatch(r"-?\d\.\d{6}E[+-]\d{4}", f.strip()) else f.strip()
            for f in row.split(",")
        ]

    assert shape(rows[0]) == shape(ref_lines[11])


@pytest.mark.skipif(
    not (ZVIEW_SAMPLES / "ZModels/12861 Dummy Cell.mdl").exists(),
    reason="ZView is not installed, so its reference files are unavailable",
)
def test_model_matches_zviews_own_dummy_cell(tmp_path, peaks):
    """That model is an Rs(RC)(RC), the same shape a two-peak DRT produces, so
    the settings block and element grammar can be compared directly."""
    reference = (ZVIEW_SAMPLES / "ZModels/12861 Dummy Cell.mdl").read_bytes()
    ref_lines = reference.decode("ascii").split("\r\n")
    ours = (
        write_drt_model(tmp_path / "m.mdl", peaks, series_resistance=10.0)
        .read_bytes()
        .decode("ascii")
        .split("\r\n")
    )

    # The settings block: same keys in the same order, values aside.
    def keys(lines):
        return [line.split(":")[0] for line in lines[: lines.index("End ZView:                    2.0") + 1]]

    assert keys(ours) == keys(ref_lines)

    # Element grammar: the type codes and marker names must line up.
    assert ours[0] == ref_lines[0], "the format version banner differs"
    for marker in (
        "  Element #0 Type:            1",
        "  Element #0,0 Name:          R",
    ):
        assert marker in ref_lines and marker in ours

    def type_of(lines, name):
        for i, line in enumerate(lines):
            if line.strip().endswith(name):
                return lines[i - 1].split(":")[1].strip()
        return None

    for name in ("Begin_Parallel", "End_Parallel"):
        assert type_of(ours, name) == type_of(ref_lines, name)
