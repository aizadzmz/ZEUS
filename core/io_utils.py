#parse typical PEIS/GEIS style data
from pathlib import Path
from typing import List

import pyimpspec
from pyimpspec import DataSet


class EISParseError(Exception):
    """Raised when parsing an EIS .mpt file fails."""


class EISDataset:
    """Wraps a single pyimpspec DataSet (one frequency sweep / experiment)."""

    def __init__(self, dataset: DataSet, index: int, source_file: str, file_id: int = 0):
        self._dataset = dataset
        self.index = index
        self.source_file = source_file
        # Identifies which loaded file this sweep came from. Distinct from
        # source_file (a filename stem, which two different files can share)
        # -- it's what makes `key` unique when several files are open at
        # once. Defaults to 0 for single-file callers (scripts, Streamlit).
        self.file_id = file_id

    @property
    def label(self) -> str:
        """Short label for legends, e.g. 'Set 01'. Unique only within one file
        -- use `key` across files."""
        return f"Set {self.index + 1:02d}"

    @property
    def full_label(self) -> str:
        """Full label for titles/filenames, e.g. 'my_file_set01'."""
        return f"{self.source_file}_set{self.index + 1:02d}"

    @property
    def key(self) -> str:
        """Stable identity for dict keys and signals, unique across every
        loaded file, unlike `label`."""
        return f"{self.file_id}:{self.index}"

    @property
    def qualified_label(self) -> str:
        """Display label that disambiguates across files, e.g. 'my_file — Set
        01'."""
        return f"{self.source_file} · {self.label}"

    @property
    def frequencies(self):
        """Frequencies in Hz, sorted high → low (numpy array)."""
        return self._dataset.get_frequencies()

    @property
    def impedances(self):
        """Complex impedances in Ohm (numpy array)."""
        return self._dataset.get_impedances()

    @property
    def num_points(self) -> int:
        return self._dataset.get_num_points()

    @property
    def data(self) -> DataSet:
        """Direct access to the underlying pyimpspec DataSet if needed."""
        return self._dataset

    def to_dict(self) -> dict:
        Z = self.impedances
        return {
            "key":            self.key,
            "label":          self.label,
            "full_label":     self.full_label,
            "index":          self.index,
            "file_id":        self.file_id,
            "num_points":     self.num_points,
            "frequencies_hz": self.frequencies.tolist(),
            "re_z_ohm":       Z.real.tolist(),
            "im_z_ohm":       Z.imag.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"<EISDataset index={self.index} "
            f"label='{self.label}' "
            f"points={self.num_points}>"
        )

_SUPPORTED_EXTENSIONS = (".mpt", ".txt")


def parse_eis_file(file_path: str | Path, file_id: int = 0) -> List[EISDataset]:
    """Parse a BioLogic .mpt file into a list of EISDataset objects."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Expected one of {_SUPPORTED_EXTENSIONS}, got '{path.suffix}'."
        )

    try:
        raw_datasets: List[DataSet] = pyimpspec.parse_data(str(path))
    except Exception as exc:
        raise EISParseError(
            f"Failed to parse '{path.name}': {exc}"
        ) from exc

    if not raw_datasets:
        raise EISParseError(
            f"No EIS datasets found in '{path.name}'. "
            f"The file may be empty or malformed."
        )

    return [
        EISDataset(ds, index=i, source_file=path.stem, file_id=file_id)
        for i, ds in enumerate(raw_datasets)
    ]