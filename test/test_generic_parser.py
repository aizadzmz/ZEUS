"""The loosely-structured .txt/.csv importer, the only way data enters the app
for instruments with no supported export format.

It guesses the delimiter, where the table starts, what each column means, and
where one sweep ends. Every one of those guesses is silent when it goes wrong,
so they are pinned here -- including on the demo files that ship with the app.
"""
from pathlib import Path

import numpy as np
import pytest

from core.generic_parser import (
    _choose_delimiter,
    _split_rows,
    classify_header,
    guess_column_roles,
    parse_generic_file,
    sniff_columns,
)
from core.io_utils import EISParseError

REPO = Path(__file__).resolve().parent.parent

HEADER = ["freq/Hz", "Z'/Ohm", "-Z''/Ohm"]
DATA = [["1000", "0.10", "0.02"], ["100", "0.20", "0.05"], ["10", "0.30", "0.09"]]


def _write(tmp_path, lines, name="sweep.txt"):
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _tabbed(rows):
    return ["\t".join(row) for row in rows]


# --- the shipped demo files -------------------------------------------------


@pytest.mark.parametrize(
    "name,sweeps,points", [("Demo File 1.txt", 2, 121), ("Demo File 2.txt", 1, 121)]
)
def test_the_demo_files_parse_as_they_always_have(name, sweeps, points):
    datasets = parse_generic_file(REPO / name)

    assert len(datasets) == sweeps
    assert datasets[0].num_points == points
    # Sorted high -> low, and every point a real measurement.
    assert datasets[0].frequencies[0] > datasets[0].frequencies[-1]
    assert np.all(datasets[0].frequencies > 0)


def test_the_demo_headers_are_still_recognised():
    headers, rows, roles = sniff_columns(REPO / "Demo File 1.txt")

    assert headers[0] == "Freq(Hz)"
    assert roles["frequency"] == 0
    assert "re" in roles and "im" in roles
    assert len(rows) == 5


# --- finding where the table starts ----------------------------------------


def test_a_plain_header_and_body(tmp_path):
    path = _write(tmp_path, _tabbed([HEADER] + DATA))
    headers, rows = _split_rows(path.read_text().splitlines())

    assert headers == HEADER
    assert rows == DATA


def test_a_title_block_above_the_header_is_skipped(tmp_path):
    """The failure this was written for: the title became the headers and
    every real row was then dropped for not matching their width."""
    lines = [
        "Some Instrument Export v2.1",
        "Date: Jan 1, 2026",
        "Cell area: 1.0 cm2",
    ] + _tabbed([HEADER] + DATA)

    headers, rows = _split_rows(lines)
    assert headers == HEADER
    assert rows == DATA


def test_a_preamble_file_parses_end_to_end(tmp_path):
    lines = ["Instrument: ACME", "Operator: nobody", ""] + _tabbed([HEADER] + DATA)
    path = _write(tmp_path, lines)

    datasets = parse_generic_file(path)
    assert len(datasets) == 1
    assert datasets[0].num_points == 3
    assert np.allclose(datasets[0].frequencies, [1000, 100, 10])
    # -Z'' column, so the stored imaginary part is negated.
    assert np.allclose(datasets[0].impedances.imag, [-0.02, -0.05, -0.09])


def test_a_headerless_table_gets_positional_names(tmp_path):
    headers, rows = _split_rows(_tabbed(DATA))

    assert headers == ["Column 1", "Column 2", "Column 3"]
    assert rows == DATA


def test_a_file_with_no_numbers_still_raises_a_useful_error(tmp_path):
    path = _write(tmp_path, ["notes", "nothing numeric here at all"])

    with pytest.raises(EISParseError):
        parse_generic_file(path)


# --- delimiters -------------------------------------------------------------


def test_prose_commas_do_not_decide_the_delimiter():
    """A comma in a title used to pick CSV for a tab-separated file."""
    lines = ["Exported: Jan 1, 2026, 09:00"] + _tabbed([HEADER] + DATA)

    assert _choose_delimiter(lines) == "\t"


@pytest.mark.parametrize("delimiter", ["\t", ",", ";"])
def test_each_supported_delimiter(delimiter):
    lines = [delimiter.join(row) for row in [HEADER] + DATA]
    headers, rows = _split_rows(lines)

    assert headers == HEADER
    assert rows == DATA


def test_whitespace_separated_files_still_work():
    headers, rows = _split_rows([" ".join(row) for row in [HEADER] + DATA])

    assert headers == HEADER
    assert rows == DATA


def test_quoted_csv_fields_are_unquoted():
    """Excel's 'CSV UTF-8' wraps every field in quotes."""
    lines = [",".join(f'"{cell}"' for cell in row) for row in [HEADER] + DATA]
    headers, rows = _split_rows(lines)

    assert headers == HEADER
    assert rows == DATA


def test_a_trailing_delimiter_does_not_drop_every_row():
    """BioLogic ends every row with a tab, leaving a phantom empty field."""
    lines = ["\t".join(row) + "\t" for row in [HEADER] + DATA]
    headers, rows = _split_rows(lines)

    assert headers == HEADER
    assert rows == DATA


# --- column roles -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,role,negated",
    [
        ("freq/Hz", "frequency", False),
        ("Frequency (Hz)", "frequency", False),
        ("Z'(a)", "re", False),
        ("-Z''(Ohm)", "im", True),
        ("|Z|/Ohm", "mag", False),
        ("-Phase(deg)", "phase", True),
        ("Range", None, False),
    ],
)
def test_header_classification(raw, role, negated):
    assert classify_header(raw) == (role, negated)


def test_the_first_column_of_a_role_wins():
    """setdefault, so a duplicate later column does not displace the first."""
    roles = guess_column_roles(["freq/Hz", "Z'/Ohm", "-Z''/Ohm", "Z'/Ohm"])

    assert roles["re"] == 1


def test_magnitude_and_phase_are_converted(tmp_path):
    lines = _tabbed(
        [["freq/Hz", "|Z|/Ohm", "Phase(deg)"], ["1000", "2.0", "0"], ["100", "2.0", "-90"]]
    )
    path = _write(tmp_path, lines)

    ds = parse_generic_file(path)[0]
    assert np.allclose(ds.impedances.real, [2.0, 0.0], atol=1e-9)
    assert np.allclose(ds.impedances.imag, [0.0, -2.0], atol=1e-9)


def test_an_explicit_mapping_overrides_the_guess(tmp_path):
    """What the import dialog hands back."""
    lines = _tabbed([["a", "b", "c"], ["1000", "0.1", "0.02"], ["100", "0.2", "0.05"]])
    path = _write(tmp_path, lines)

    ds = parse_generic_file(path, column_roles={"frequency": 0, "re": 1, "im": 2})[0]
    assert np.allclose(ds.frequencies, [1000, 100])


def test_a_file_without_impedance_columns_is_refused(tmp_path):
    path = _write(tmp_path, _tabbed([["freq/Hz", "Ampl"], ["1000", "0.01"]]))

    with pytest.raises(EISParseError, match="impedance"):
        parse_generic_file(path)


# --- sweeps -----------------------------------------------------------------


def test_a_reversal_starts_a_new_sweep(tmp_path):
    rows = [["1000", "0.1", "0.02"], ["100", "0.2", "0.05"],
            ["1000", "0.1", "0.02"], ["100", "0.2", "0.05"]]
    path = _write(tmp_path, _tabbed([HEADER] + rows))

    datasets = parse_generic_file(path)
    assert len(datasets) == 2
    assert all(ds.num_points == 2 for ds in datasets)


def test_padding_rows_of_zeros_are_dropped(tmp_path):
    """Instruments pad a sweep out to a fixed row count."""
    rows = DATA + [["0", "0", "0"], ["0", "0", "0"]]
    path = _write(tmp_path, _tabbed([HEADER] + rows))

    assert parse_generic_file(path)[0].num_points == 3
