"""Parse generic, loosely-structured EIS exports (plain .txt or .csv).

Unlike the BioLogic-specific parsers in io_utils/mb_parser, these files have
no fixed schema: headers vary ("freq/Hz" vs "Frequency (Hz)"), the impedance
may be given as Re/Im or as magnitude/phase, and sign conventions differ
(e.g. a column literally named "-Z''" stores the negative of Im(Z)). Column
roles are guessed from the header text but callers can override the guess
with an explicit ``column_roles`` mapping (e.g. from a GUI confirmation
dialog) when the guess is wrong or ambiguous.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyimpspec import DataSet

from core.io_utils import EISDataset, EISParseError

# Role -> set of normalized header "bodies" that identify it. Units/symbols
# are stripped before matching (see _normalize_header), so "freq/Hz" and
# "Frequency (Hz)" both normalize to "freq"/"frequency".
_ROLE_ALIASES = {
    "frequency": {"freq", "frequency", "f"},
    "re": {"re", "z'", "zre", "real"},
    "im": {"im", "z''", "zim", "imag", "imaginary"},
    "mag": {"z", "|z|", "zmod", "modz", "mod"},
    "phase": {"phase", "phz", "theta"},
    "time": {"time", "t"},
    "index": {"index", "pt", "point", "no", "#", "cycle"},
}

_DELIMITER_CANDIDATES = ("\t", ",", ";")


def _normalize_header(raw: str) -> Tuple[str, bool]:
    """'-Z'' (Ohm)' -> ("z''", True); 'freq/Hz' -> ("freq", False)."""
    core = re.split(r"[\(/]", raw.strip())[0].strip().lower()
    core = re.sub(r"\s+", "", core)
    core = core.replace('"', "''").replace("″", "''").replace("′", "'")
    neg = core.startswith("-")
    body = core[1:] if neg else core
    return body, neg


def classify_header(raw: str) -> Tuple[Optional[str], bool]:
    """Guess which physical quantity a column header represents.

    Returns (role, is_negated). role is one of "frequency", "re", "im",
    "mag", "phase", "time", "index", or None if unrecognized.
    """
    body, neg = _normalize_header(raw)
    for role, aliases in _ROLE_ALIASES.items():
        if body in aliases:
            return role, neg
    return None, neg


def guess_column_roles(headers: Sequence[str]) -> Dict[str, int]:
    """Best-effort header -> role mapping. Values are column indices.

    Negated "im"/"phase" columns (e.g. "-Z''", "-Phase") are keyed as
    "neg_im"/"neg_phase" so callers know to flip sign when computing Z.
    The first match wins per role/sign combination.
    """
    mapping: Dict[str, int] = {}
    for i, header in enumerate(headers):
        role, neg = classify_header(header)
        if role is None or role == "index":
            continue
        key = f"neg_{role}" if neg and role in ("im", "phase") else role
        mapping.setdefault(key, i)
    return mapping


def _read_lines(path: Path, encoding: str) -> List[str]:
    with open(path, "r", encoding=encoding, newline="") as f:
        text = f.read()

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise EISParseError(f"'{path.name}' is empty.")
    return lines


def _read_rows(path: Path, encoding: str) -> Tuple[List[str], List[List[str]]]:
    """Read a file and split it into (headers, data_rows)."""
    return _split_rows(_read_lines(path, encoding))


def _split_rows(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Hand-split every line/cell in pure Python.

    This used to hand off delimited files to pandas' C parser. Measured on
    the exports this actually sees, that was the wrong trade: ~3x *slower*
    than this on a few-hundred-row sweep file, and the ~0.6s it saved on an
    86k-row one was less than the ~0.9s pandas costs to import -- an import
    that, being deferred to first use, landed inside the click that opens
    the file. Splitting here keeps the parse dependency-free and the cost
    proportional to the file.
    """
    delimiter = next((c for c in _DELIMITER_CANDIDATES if c in lines[0]), None)
    if delimiter:
        # csv.reader (not str.split) so quoted fields - e.g. Excel's "CSV
        # UTF-8" export wrapping every field in "..." - are unquoted
        # correctly instead of leaving literal quote characters behind.
        rows = list(csv.reader(lines, delimiter=delimiter))
    else:
        # str.split() with no args splits on any run of whitespace and is
        # implemented in C, unlike re.split(r"\s+", ...) - matters here
        # since this runs once per line.
        rows = [ln.split() for ln in lines]

    # A trailing delimiter on a line yields a phantom empty field (BioLogic
    # exports end every row with a tab). Trim those so the header width
    # matches the real data width instead of silently dropping every row.
    headers = _rstrip_blanks([h.strip() for h in rows[0]])
    ncols = len(headers)
    data_rows = [
        cells
        for cells in ([c.strip() for c in r] for r in rows[1:])
        if len(_rstrip_blanks(cells)) == ncols
    ]
    return headers, [r[:ncols] for r in data_rows]


def _rstrip_blanks(cells: List[str]) -> List[str]:
    """Drop trailing empty fields left behind by a trailing delimiter."""
    end = len(cells)
    while end > 0 and not cells[end - 1]:
        end -= 1
    return cells[:end]


def sniff_columns(
    file_path: str | Path, encoding: str = "utf-8-sig"
) -> Tuple[List[str], List[List[str]], Dict[str, int]]:
    """Read headers + a few sample rows and guess column roles, without
    parsing the whole file. Intended for a GUI confirmation dialog."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        headers, data_rows = _read_rows(path, encoding)
    except UnicodeDecodeError:
        headers, data_rows = _read_rows(path, "latin-1")

    return headers, data_rows[:5], guess_column_roles(headers)


def _column_to_floats(rows: List[List[str]], col: int) -> np.ndarray:
    """Convert one column of a hand-split data grid to a float array.

    Tries a single bulk conversion first (fast path: the common case of a
    clean numeric export). Only falls back to converting cell-by-cell -
    turning unparseable cells into NaN rather than raising - if the bulk
    conversion hits a bad cell somewhere in the column.
    """
    cells = [row[col] for row in rows]
    try:
        return np.array(cells, dtype=np.float64)
    except ValueError:
        out = np.full(len(cells), np.nan)
        for i, cell in enumerate(cells):
            try:
                out[i] = float(cell)
            except ValueError:
                pass
        return out


def parse_generic_file(
    file_path: str | Path,
    column_roles: Optional[Dict[str, int]] = None,
    encoding: str = "utf-8-sig",
    file_id: int = 0,
) -> List[EISDataset]:
    """Parse a generic single- or multi-sweep EIS export (plain .txt or
    .csv) with arbitrary column headers.

    column_roles maps role names to column indices: "frequency" (required),
    plus either ("re" and "im"/"neg_im") or ("mag" and "phase"/"neg_phase").
    If omitted, roles are guessed from the header row; pass an explicit
    mapping (e.g. from sniff_columns + user confirmation) to override a bad
    guess.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        lines = _read_lines(path, encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        lines = _read_lines(path, encoding)

    headers, data_rows = _split_rows(lines)

    roles = column_roles if column_roles is not None else guess_column_roles(headers)

    if "frequency" not in roles:
        raise EISParseError(
            f"Could not identify a frequency column in '{path.name}'. "
            f"Found headers: {headers}"
        )

    has_re_im = "re" in roles and ("im" in roles or "neg_im" in roles)
    has_mag_phase = "mag" in roles and ("phase" in roles or "neg_phase" in roles)
    if not has_re_im and not has_mag_phase:
        raise EISParseError(
            f"Could not identify impedance columns in '{path.name}'. Need "
            f"either Z'/Z'' (real+imaginary) or Z/phase (magnitude+phase). "
            f"Found headers: {headers}"
        )

    freq_col = roles["frequency"]
    if has_re_im:
        re_col = roles["re"]
        im_col, im_neg = (
            (roles["im"], False) if "im" in roles else (roles["neg_im"], True)
        )
    else:
        mag_col = roles["mag"]
        phase_col, phase_neg = (
            (roles["phase"], False) if "phase" in roles else (roles["neg_phase"], True)
        )

    def column(idx: int) -> np.ndarray:
        return _column_to_floats(data_rows, idx)

    freqs = column(freq_col)
    if has_re_im:
        re_vals = column(re_col)
        im_vals = column(im_col)
        if im_neg:
            im_vals = -im_vals
        z = re_vals + 1j * im_vals
    else:
        mag = column(mag_col)
        phase_deg = column(phase_col)
        if phase_neg:
            phase_deg = -phase_deg
        theta = np.radians(phase_deg)
        z = mag * np.cos(theta) + 1j * mag * np.sin(theta)

    # Instruments pad a sweep out to a fixed row count with all-zero rows
    # (and unparseable cells above became NaN); f <= 0 is not a measurable
    # EIS point in any case.
    valid = np.isfinite(freqs) & (freqs > 0.0) & np.isfinite(z)
    frequencies = freqs[valid]
    impedances = z[valid]

    if frequencies.size == 0:
        raise EISParseError(f"No numeric data rows found in '{path.name}'.")

    sweeps = _split_into_sweeps(frequencies, impedances)

    datasets = []
    for i, (freqs, zs) in enumerate(sweeps):
        freqs, zs = _drop_duplicate_frequencies(freqs, zs)
        ds = DataSet(np.array(freqs), np.array(zs), label=f"Sweep {i + 1}", path=str(path))
        datasets.append(EISDataset(ds, index=i, source_file=path.stem, file_id=file_id))
    return datasets


def _drop_duplicate_frequencies(
    frequencies: np.ndarray, impedances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the first row per frequency, preserving sweep order.

    pyimpspec's DataSet rejects a repeated frequency outright, but real
    exports do repeat one (a point measured twice for averaging, duplicated
    rows). Dropping the repeat costs one point; letting it through raises
    ValueError out of the parser and used to take the whole app down.
    """
    _, first_occurrence = np.unique(frequencies, return_index=True)
    keep = np.sort(first_occurrence)
    return frequencies[keep], impedances[keep]


def _split_into_sweeps(
    frequencies: np.ndarray, impedances: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split rows into separate sweeps whenever the frequency direction
    reverses. A single EIS sweep moves monotonically high->low or
    low->high, so a reversal marks the start of the next sweep."""
    if len(frequencies) < 2:
        return [(frequencies, impedances)]

    sweeps: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    direction = 0
    for i in range(1, len(frequencies)):
        delta = frequencies[i] - frequencies[i - 1]
        if delta == 0:
            continue
        cur_dir = 1 if delta > 0 else -1
        if direction == 0:
            direction = cur_dir
        elif cur_dir != direction:
            # A down-then-up (or up-then-down) run usually measures the
            # turning-point frequency twice, once per direction. Cut between
            # those two rows so each sweep keeps its own copy -- cutting at i
            # instead would leave both in the outgoing sweep, i.e. a repeated
            # frequency that DataSet rejects.
            cut = i - 1 if frequencies[i - 1] == frequencies[i - 2] else i
            sweeps.append((frequencies[start:cut], impedances[start:cut]))
            start = cut
            direction = 0
    sweeps.append((frequencies[start:], impedances[start:]))
    return sweeps
