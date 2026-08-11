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
    write_spectrum_z,
    write_validation_fit_z,
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


# ---- validated spectrum ----

def values_of(rows):
    """The seven float columns of each data line, as an array."""
    return np.array([[float(f) for f in row.split(",")[:7]] for row in rows])


def test_spectrum_z_writes_the_impedance_columns(tmp_path, dataset):
    """Z' and Z'' land in their own columns, Z'' signed as ZView stores it."""
    _, rows = read_z(write_spectrum_z(tmp_path / "s.z", dataset))
    Z = dataset.data.get_impedances(masked=False)

    values = values_of(rows)
    # Written highest frequency first, which the fixture's logspace is not.
    order = np.argsort(dataset.data.get_frequencies(masked=False))[::-1]
    assert np.allclose(values[:, 4], Z[order].real, rtol=1e-5)
    assert np.allclose(values[:, 5], Z[order].imag, rtol=1e-5)
    # A capacitive sweep: ZView's own files keep Z'' negative rather than -Z''.
    assert np.all(values[:, 5] <= 0)
    assert np.all(np.diff(values[:, 0]) < 0)


def test_spectrum_z_writes_kept_points_only(tmp_path, dataset):
    """The rejected points are absent from the file, not flagged in it -- that
    is what makes this an after-validation export."""
    from copy import deepcopy

    masked = EISDataset(
        deepcopy(dataset.data), dataset.index, dataset.source_file, dataset.file_id
    )
    masked.data.set_mask({0: True, 5: True, 6: True})

    header, rows = read_z(write_spectrum_z(tmp_path / "s.z", masked))
    total = masked.data.get_num_points(masked=None)
    assert len(rows) == total - 3
    assert int(header[9]) == len(rows), "the point count must follow the removal"

    dropped = masked.data.get_frequencies(masked=True)
    written = values_of(rows)[:, 0]
    for f in dropped:
        assert not np.any(np.isclose(written, f, rtol=1e-5))


def test_spectrum_z_can_write_the_unmasked_sweep(tmp_path, dataset):
    from copy import deepcopy

    masked = EISDataset(
        deepcopy(dataset.data), dataset.index, dataset.source_file, dataset.file_id
    )
    masked.data.set_mask({0: True})

    _, rows = read_z(write_spectrum_z(tmp_path / "s.z", masked, kept_only=False))
    assert len(rows) == masked.data.get_num_points(masked=None)


def test_spectrum_z_refuses_a_fully_masked_sweep(tmp_path, dataset):
    from copy import deepcopy

    empty = EISDataset(
        deepcopy(dataset.data), dataset.index, dataset.source_file, dataset.file_id
    )
    empty.data.set_mask({i: True for i in range(empty.data.get_num_points(masked=None))})

    with pytest.raises(ValueError, match="nothing to write"):
        write_spectrum_z(tmp_path / "s.z", empty)


def test_comment_cannot_break_the_header(tmp_path, dataset):
    """The header is quote-delimited and line-positional, so a quote or newline
    in the free text would shift every line after it."""
    header, rows = read_z(
        write_spectrum_z(
            tmp_path / "s.z", dataset, comment='he said "12 of 40"\nkept'
        )
    )
    assert len(header) == 11
    assert header[5] == '"he said \'12 of 40\' kept"'
    assert int(header[9]) == len(rows)


# ---- validation fit ----

@pytest.fixture(scope="module")
def kk(dataset):
    from core.validation import run_kk_test

    return run_kk_test(dataset)


def test_validation_fit_z_carries_the_reconstruction(tmp_path, kk):
    _, rows = read_z(write_validation_fit_z(tmp_path / "f.z", kk))
    frequencies = kk.get_frequencies()
    Z = kk.get_impedances()
    order = np.argsort(frequencies)[::-1]

    values = values_of(rows)
    assert len(rows) == len(frequencies)
    assert np.allclose(values[:, 0], frequencies[order], rtol=1e-5)
    assert np.allclose(values[:, 4], Z[order].real, rtol=1e-5)
    assert np.allclose(values[:, 5], Z[order].imag, rtol=1e-5)


def test_validation_fit_z_overlays_the_spectrum(tmp_path, dataset, kk):
    """With nothing masked the two files share their frequencies in the same
    order, so ZView plots them on top of each other. Once points are rejected
    the fit keeps its own, wider coverage -- see write_validation_fit_z."""
    _, spectrum = read_z(write_spectrum_z(tmp_path / "s.z", dataset))
    _, fit = read_z(write_validation_fit_z(tmp_path / "f.z", kk))

    assert np.allclose(values_of(fit)[:, 0], values_of(spectrum)[:, 0], rtol=1e-5)


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
