"""Parse generic, loosely-structured EIS exports (plain .txt or .csv).

Unlike the BioLogic-specific parsers in io_utils/mb_parser, these files have
no fixed schema: headers vary ("freq/Hz" vs "Frequency (Hz)"), the impedance
may be given as Re/Im or as magnitude/phase, and sign conventions differ
(e.g. a column literally named "-Z''" stores the negative of Im(Z)). Column
roles are guessed from the header text but callers can override the guess
with an explicit ``column_roles`` mapping (e.g. from a GUI confirmation
dialog) when the guess is wrong or ambiguous.
"""
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


def _read_rows(path: Path, encoding: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding=encoding, newline="") as f:
        text = f.read()

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise EISParseError(f"'{path.name}' is empty.")

    delimiter = next((c for c in _DELIMITER_CANDIDATES if c in lines[0]), None)
    if delimiter:
        # csv.reader (not str.split) so quoted fields - e.g. Excel's "CSV
        # UTF-8" export wrapping every field in "..." - are unquoted
        # correctly instead of leaving literal quote characters behind.
        rows = list(csv.reader(lines, delimiter=delimiter))
    else:
        rows = [re.split(r"\s+", ln.strip()) for ln in lines]

    headers = [h.strip() for h in rows[0]]
    ncols = len(headers)
    data_rows = [r for r in rows[1:] if len(r) == ncols]
    return headers, data_rows


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


def parse_generic_file(
    file_path: str | Path,
    column_roles: Optional[Dict[str, int]] = None,
    encoding: str = "utf-8-sig",
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
        headers, data_rows = _read_rows(path, encoding)
    except UnicodeDecodeError:
        headers, data_rows = _read_rows(path, "latin-1")

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

    frequencies: List[float] = []
    impedances: List[complex] = []
    for row in data_rows:
        try:
            f = float(row[freq_col])
            if has_re_im:
                re_val = float(row[re_col])
                im_val = float(row[im_col])
                if im_neg:
                    im_val = -im_val
                z = complex(re_val, im_val)
            else:
                mag = float(row[mag_col])
                phase_deg = float(row[phase_col])
                if phase_neg:
                    phase_deg = -phase_deg
                theta = np.radians(phase_deg)
                z = complex(mag * np.cos(theta), mag * np.sin(theta))
        except (ValueError, IndexError):
            continue
        frequencies.append(f)
        impedances.append(z)

    if not frequencies:
        raise EISParseError(f"No numeric data rows found in '{path.name}'.")

    sweeps = _split_into_sweeps(frequencies, impedances)

    datasets = []
    for i, (freqs, zs) in enumerate(sweeps):
        ds = DataSet(np.array(freqs), np.array(zs), label=f"Sweep {i + 1}", path=str(path))
        datasets.append(EISDataset(ds, index=i, source_file=path.stem))
    return datasets


def _split_into_sweeps(
    frequencies: List[float], impedances: List[complex]
) -> List[Tuple[List[float], List[complex]]]:
    """Split rows into separate sweeps whenever the frequency direction
    reverses. A single EIS sweep moves monotonically high->low or
    low->high, so a reversal marks the start of the next sweep."""
    if len(frequencies) < 2:
        return [(frequencies, impedances)]

    sweeps: List[Tuple[List[float], List[complex]]] = []
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
            sweeps.append((frequencies[start:i], impedances[start:i]))
            start = i
            direction = 0
    sweeps.append((frequencies[start:], impedances[start:]))
    return sweeps
