"""Parse generic, loosely-structured EIS exports (plain .txt or .csv)."""
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
    """Guess which physical quantity a column header represents."""
    body, neg = _normalize_header(raw)
    for role, aliases in _ROLE_ALIASES.items():
        if body in aliases:
            return role, neg
    return None, neg


def guess_column_roles(headers: Sequence[str]) -> Dict[str, int]:
    """Best-effort header -> role mapping. Values are column indices."""
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


# How much of a row must parse as a number for it to be data rather than
# prose. Not all of it: instruments write text flags ("OK", "ovl", a range
# code) into otherwise numeric rows, and the header itself is occasionally
# half numeric ("I/mA", "1", "2").
_DATA_ROW_FRACTION = 0.6

# Below this many columns a numeric line is not a sweep row -- it is a
# metadata line that happens to hold a number, like "Cell area: 1.0".
_MIN_DATA_COLUMNS = 2


def _numeric_fraction(cells: List[str]) -> float:
    """How much of a row parses as a number, 0.0 for an empty row."""
    if not cells:
        return 0.0
    numeric = 0
    for cell in cells:
        try:
            float(cell)
        except ValueError:
            pass
        else:
            numeric += 1
    return numeric / len(cells)


def _choose_delimiter(lines: List[str]) -> Optional[str]:
    """The character separating columns, judged over the whole file.

    Not from the first line alone, which is only the header when nothing sits
    above it. Instruments routinely write a title block first, and a title is
    prose -- the comma in 'Date: Jan 1, 2026' would otherwise decide that the
    file is comma-separated and split every real row in the wrong places.

    A true delimiter appears the *same* number of times on almost every line,
    since every row has the same columns; prose punctuation does not. That
    consistency is what is scored here, rather than mere presence.
    """
    best: Optional[str] = None
    best_score = 0
    for candidate in _DELIMITER_CANDIDATES:
        counts = [line.count(candidate) for line in lines]
        most_common = max(set(counts), key=counts.count)
        if most_common == 0:
            continue
        score = counts.count(most_common)
        if score > best_score:
            best, best_score = candidate, score
    # Must hold for most of the file, or it is punctuation, not structure.
    return best if best_score > len(lines) / 2 else None


def _first_data_row(rows: List[List[str]]) -> Optional[int]:
    """Index of the first row that reads as measurements rather than text."""
    for i, cells in enumerate(rows):
        trimmed = _rstrip_blanks(cells)
        if (
            len(trimmed) >= _MIN_DATA_COLUMNS
            and _numeric_fraction(trimmed) >= _DATA_ROW_FRACTION
        ):
            return i
    return None


def _split_rows(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Hand-split every line/cell in pure Python, and find where the table
    starts.

    The header is located rather than assumed to be line one: a plain .txt
    export often carries an instrument title, a timestamp and a settings block
    above its column names. Taking line one on faith turned those into the
    headers and then dropped every real row for not matching their width, so
    the file arrived with no data and a column dialog listing words from a
    sentence.
    """
    delimiter = _choose_delimiter(lines)
    if delimiter:
        # csv.reader, not str.split, so quoted fields (Excel's "CSV UTF-8"
        # export wraps every field in "...") are unquoted correctly.
        rows = list(csv.reader(lines, delimiter=delimiter))
    else:
        # Bare str.split() splits on any run of whitespace and runs in C,
        # unlike re.split(r"\s+", ...); this runs once per line.
        rows = [ln.split() for ln in lines]

    rows = [[cell.strip() for cell in row] for row in rows]

    # No numeric row anywhere: fall back to the old assumption so the caller
    # raises its own "no numeric data rows" against real headers.
    first_data = _first_data_row(rows)
    if first_data is None:
        first_data = 1

    # A trailing delimiter yields a phantom empty field (BioLogic exports end
    # every row with a tab). Trim so the header width matches the data width,
    # instead of silently dropping every row.
    if first_data > 0:
        headers = _rstrip_blanks(rows[first_data - 1])
    else:
        # The table starts on line one, so there are no names to read. Give
        # the columns positional ones: the import dialog needs something to
        # label them with, and the caller can still be handed a mapping.
        headers = [f"Column {i + 1}" for i in range(len(_rstrip_blanks(rows[0])))]

    ncols = len(headers)
    data_rows = [
        cells
        for cells in rows[first_data:]
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
    """Read headers + a few sample rows and guess column roles, without parsing
    the whole file. Intended for a GUI confirmation dialog."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        headers, data_rows = _read_rows(path, encoding)
    except UnicodeDecodeError:
        headers, data_rows = _read_rows(path, "latin-1")

    return headers, data_rows[:5], guess_column_roles(headers)


def _column_to_floats(rows: List[List[str]], col: int) -> np.ndarray:
    """Convert one column of a hand-split data grid to a float array."""
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
    """Parse a generic single- or multi-sweep EIS export (plain .txt or .csv)
    with arbitrary column headers."""
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
    """Keep only the first row per frequency, preserving sweep order."""
    _, first_occurrence = np.unique(frequencies, return_index=True)
    keep = np.sort(first_occurrence)
    return frequencies[keep], impedances[keep]


def _split_into_sweeps(
    frequencies: np.ndarray, impedances: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split rows into separate sweeps wherever the frequency direction
    reverses, since one sweep is monotonic."""
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
