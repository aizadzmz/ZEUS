#parse modulo bat style files
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyimpspec import DataSet

from core.io_utils import EISDataset, EISParseError

_EIS_TECHNIQUES = {"PEIS", "GEIS"}


def parse_modulobat_file(
    file_path: str | Path, encoding: str = "latin-1", file_id: int = 0
) -> List[EISDataset]:
    """Parse a BioLogic/BT-Lab Modulo Bat .mpt export, extracting the PEIS/GEIS
    sub-measurements embedded in the cycling sequence."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()

    header_count_match = re.search(r"Nb header lines\s*:\s*(\d+)", lines[1])
    if not header_count_match:
        raise EISParseError(f"Could not find 'Nb header lines' in '{path.name}'.")
    n_header = int(header_count_match.group(1))

    header_lines = lines[:n_header]
    column_header = header_lines[-1].rstrip("\n").split("\t")
    data_lines = lines[n_header:]

    eis_ns = _find_eis_sequence_steps(header_lines)
    if not eis_ns:
        raise EISParseError(
            f"No PEIS/GEIS steps found in the Modulo Bat sequence of '{path.name}'."
        )

    try:
        ns_col = column_header.index("Ns")
        zcycle_col = column_header.index("z cycle")
        freq_col = column_header.index("freq/Hz")
        re_col = column_header.index("Re(Z)/Ohm")
        neg_im_col = column_header.index("-Im(Z)/Ohm")
    except ValueError as exc:
        raise EISParseError(f"Expected column missing in '{path.name}': {exc}") from exc

    sweeps: Dict[str, List[Tuple[float, float, float]]] = {}
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if parts[ns_col] not in eis_ns:
            continue
        zcycle = parts[zcycle_col]
        sweeps.setdefault(zcycle, []).append(
            (float(parts[freq_col]), float(parts[re_col]), float(parts[neg_im_col]))
        )

    if not sweeps:
        raise EISParseError(f"No EIS data rows found in '{path.name}'.")

    datasets = []
    for i, zcycle in enumerate(sorted(sweeps, key=int)):
        rows = sweeps[zcycle]
        frequencies = np.array([r[0] for r in rows])
        impedances = np.array([complex(r[1], -r[2]) for r in rows])
        ds = DataSet(frequencies, impedances, label=f"z cycle {zcycle}", path=str(path))
        datasets.append(EISDataset(ds, index=i, source_file=path.stem, file_id=file_id))

    return datasets


def _find_eis_sequence_steps(header_lines: List[str]) -> set:
    """Map each 'Ns' sequence-step index to its ctrl_type and return the
    PEIS/GEIS ones."""
    ns_values = None
    ctrl_types = None
    for line in header_lines:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "Ns":
            ns_values = tokens[1:]
        elif tokens[0] == "ctrl_type":
            ctrl_types = tokens[1:]

    if ns_values is None or ctrl_types is None:
        return set()

    return {ns for ns, ctrl in zip(ns_values, ctrl_types) if ctrl in _EIS_TECHNIQUES}
