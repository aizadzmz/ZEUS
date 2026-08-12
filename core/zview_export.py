#Export analysis output in the formats Scribner's ZView reads
"""Write measured and derived spectra as ZView data files (.z), and fitted DRT
peaks as a ZView equivalent-circuit model (.mdl), in layouts taken from the
files ZView itself ships rather than from a published specification. The
layouts below note which parts the reader is known to be strict about.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from core.io_utils import EISDataset

# ZView's own files end every line with CRLF and its parser is line-position based, so line endings are not left to the platform.
_EOL = "\r\n"

# The nine columns every ZView data file carries, verbatim from its own files.
_Z_COLUMN_HEADER = (
    '"  Freq (Hz)    Ampl     Bias   Time(Sec)   Z\'(a)    Z\'\'(b)    GD   Err   Range"'
)


def _z_num(value) -> str:
    """One number in ZView's fixed-width scientific notation, e.g.
    ' 6.500000E+0004'. The exponent is padded to four digits, which is what
    ZView writes; Python's own %E gives two."""
    number = float(value)
    if not np.isfinite(number):
        number = 0.0
    mantissa, _, exponent = f"{number:.6E}".partition("E")
    sign, digits = exponent[0], exponent[1:]
    return f"{mantissa}E{sign}{digits.zfill(4)}"


def _z_text(text: str) -> str:
    """One free-text header field, made safe to sit inside ZView's quotes: the
    fields are quote-delimited and one per line, so an embedded quote or newline
    would shift every line after it."""
    return " ".join(str(text).replace('"', "'").split())


def _z_row(frequency, z_real, z_imag, index: int) -> str:
    """One data line. Ampl, Bias, GD, Err and Range are not recorded by anything
    this app exports, so they are zero; Time carries the row number, as in
    ZView's demo files."""
    fields = [
        _z_num(frequency),
        _z_num(0.0),  # Ampl
        _z_num(0.0),  # Bias
        _z_num(index),  # Time(Sec)
        _z_num(z_real),
        _z_num(z_imag),
        _z_num(0.0),  # GD
    ]
    # Two spaces after each comma, then Err and Range as bare integers: exactly what ZView writes.
    return " " + ",  ".join(fields) + ", 0, 0"


def _write_z(
    path: Path,
    rows: List[Tuple[float, float, float]],
    comment: str,
    source: str = "eis-batch-analysis",
) -> Path:
    """Write the eleven-line header and the data block. The header length is
    fixed: ZView takes the point count from line 10 and the first data row from
    line 12, so no line here may be dropped or added. Line 5 names what produced
    the data and line 6 is the one free-text slot.
    """
    frequencies = [row[0] for row in rows] or [0.0]
    now = datetime.now()

    header = [
        '"ZView Calculated Data File: Version 1.1"',
        '"Raw Data"',
        '"Sweep Frequency: Control Voltage"',
        # The stray inner quote is ZView's, not a typo -- its files all carry it.
        f'"Date: {now.month}-{now.day}-{now.year}     "Time: {now.hour}:{now.minute}:{now.second}"',
        f'"{_z_text(source)}"',
        f'"{_z_text(comment)}"',
        f'"{path.name}"',
        '"Frequency"',
        f"0,2,0,1,{min(frequencies):.6G},{max(frequencies):.6G}",
        str(len(rows)),
        _Z_COLUMN_HEADER,
    ]

    lines = header + [
        _z_row(frequency, real, imag, index)
        for index, (frequency, real, imag) in enumerate(rows, start=1)
    ]
    # newline="" keeps Python from translating the explicit CRLFs into CRCRLF.
    with open(path, "w", newline="", encoding="ascii", errors="replace") as handle:
        handle.write(_EOL.join(lines) + _EOL)
    return path


def _impedance_rows(frequencies, impedances) -> List[Tuple[float, float, float]]:
    """(f, Z', Z'') rows, highest frequency first. Z'' keeps its sign rather
    than being negated, as in ZView's own files, and the descending sort is what
    stops ZView drawing the sweep as a zigzag.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    impedances = np.asarray(impedances, dtype=complex)
    order = np.argsort(frequencies)[::-1]
    return [
        (float(frequencies[i]), float(impedances[i].real), float(impedances[i].imag))
        for i in order
    ]


def write_spectrum_z(
    path: Path, dataset: EISDataset, kept_only: bool = True, comment: str = ""
) -> Path:
    """A measured spectrum as a ZView data file, so ZView can fit an equivalent
    circuit to it.

    kept_only writes the points that survived masking, which is what makes this
    an "after validation" export; pass False to write the sweep as measured.
    comment goes in the file's one free-text header line.
    """
    masked = False if kept_only else None
    rows = _impedance_rows(
        dataset.data.get_frequencies(masked=masked),
        dataset.data.get_impedances(masked=masked),
    )
    if not rows:
        raise ValueError(
            "Every point in this sweep is masked, so there is nothing to write."
        )
    return _write_z(
        Path(path),
        rows,
        comment or "measured impedance, masked points removed",
        source=f"eis-batch-analysis: {dataset.full_label}",
    )


def write_validation_fit_z(path: Path, result, comment: str = "") -> Path:
    """A validation result's reconstructed impedance as a ZView data file: the
    lin-KK model's response, or the modulus Z-HIT rebuilt from the phase.

    Written on the frequencies the validation was *fitted* on, which need not
    be the ones write_spectrum_z keeps, so ZView draws the fit straight through
    the gaps the spectrum has. The residuals are not written; export the CSV.
    """
    rows = _impedance_rows(result.get_frequencies(), result.get_impedances())
    return _write_z(
        Path(path),
        rows,
        comment or "reconstructed impedance from the validation fit",
        source="eis-batch-analysis: validation fit",
    )


def write_drt_z(path: Path, result) -> Path:
    """A DRT curve as a ZView data file. ZView has no notion of a distribution,
    so tau becomes a frequency through f = 1/(2*pi*tau) and gamma is written
    into Z'(a) with Z''(b) left at zero. Credible intervals have nowhere to go
    in this format; export the CSV instead.
    """
    taus, gammas = result.get_drt_data()
    taus = np.asarray(taus, dtype=float)
    # tau ascending gives frequency descending, the order ZView's own files are written in.
    frequencies = 1.0 / (2.0 * np.pi * taus)

    gamma_real, gamma_imag = np.asarray(gammas), np.zeros_like(taus)
    comment = "gamma(tau) [ohm] in the Z'(a) column; f = 1/(2*pi*tau)"

    rows = [
        (float(f), float(re), float(im))
        for f, re, im in zip(frequencies, gamma_real, gamma_imag)
    ]
    return _write_z(
        Path(path),
        rows,
        comment,
        source="eis-batch-analysis: distribution of relaxation times",
    )


# ---- equivalent-circuit model ----

# ZView's model files open with a block of solver settings between these markers, copied from the files ZView ships; none describes the circuit, but the reader expects it.
_MDL_SETTINGS = """ZView Circuit Model:          2.0
  IOPT:                       0
  DINP:                       1
  DFIT:                       1
  PINP:                       4
  PFIT:                       4
  FREEQ:                      1
  NEG:                        1
  FUN:                        99
  DATATYPE:                   0
  IPAR:                       0
  IFP:                        0
  IRE:                        0
  CELCAP:                     0
  ROE:                        0
  ATEMP:                      293
  M:                          0
  N:                          34
  MAXFEV:                     100
  NPRINT:                     9
  IRCH:                       -3
  MODE:                       0
  ICP:                        0
  IPRINT:                     1
  IGACC:                      2
  WEIGHTU:                    0
  WEIGHTX:                    1
  MAX FREQ:                   0.0005
  MIN FREQ:                   100000
  FIT MODE:                   0
  RANGE MODE:                 1
  FIT BEEP:                   0
  Batch Previous:             0
  Batch Spectra:              1
  Batch Print:                0
  Batch Output:
End ZView:                    2.0"""

# Element type codes, as used in ZView's own model files.
_TYPE_RESISTOR = 1
_TYPE_CAPACITOR = 2
_TYPE_BEGIN_PARALLEL = -1
_TYPE_END_PARALLEL = -2


def _mdl_num(value) -> str:
    """A parameter value. ZView writes plain decimals and uppercase
    exponents."""
    return f"{float(value):.6G}"


def _element(index: int, type_code: int, name: str, parameters=()) -> List[str]:
    """The lines describing one element. The parameter slot number is fixed per
    element type -- a resistor's R is slot 0, a capacitor's T is slot 1 -- so
    callers pass it rather than relying on position."""
    lines = [
        f"  Element #{index} Type:            {type_code}",
        f"  Element #{index} Name:            {name}",
    ]
    for slot, label, value in parameters:
        lines += [
            f"  Element #{index},{slot} Name:          {label}",
            f"  Element #{index},{slot} Value:         {_mdl_num(value)}",
            f"  Element #{index},{slot} Free:          1",
        ]
    return lines


def peak_rc_pairs(peaks) -> List[Tuple[float, float]]:
    """(resistance, capacitance) for each fitted DRT peak: a peak's area is the
    resistance it contributes and its position the time constant, so
    C = tau / R. Peaks with a non-positive area are dropped.
    """
    pairs: List[Tuple[float, float]] = []
    for index, peak in enumerate(peaks.peaks):
        # Same reconstruction as core.bdf_export.write_drt_peaks: the peak stores a position and height relative to the scaled fit.
        tau = 10 ** ((peak.position * peak.x_scale) + peak.x_offset)
        resistance = float(peaks.get_peak_area(index))
        if not np.isfinite(resistance) or resistance <= 0 or not np.isfinite(tau):
            continue
        pairs.append((resistance, float(tau) / resistance))
    return pairs


def write_drt_model(path: Path, peaks, series_resistance: Optional[float] = None) -> Path:
    """Fitted DRT peaks as a ZView equivalent-circuit model: one parallel R-C
    pair per peak, giving Rs(R1C1)(R2C2)..., with every parameter left free.

    series_resistance seeds Rs (see estimate_series_resistance); None writes Rs
    as 0, still free. Raises ValueError when no peak survives peak_rc_pairs,
    since a model with no elements is not something ZView can open.
    """
    pairs = peak_rc_pairs(peaks)
    if not pairs:
        raise ValueError(
            "No DRT peak has a positive resistance, so there is no circuit to "
            "write. Run peak extraction again, or export the CSV instead."
        )

    lines: List[str] = []
    lines += _element(
        0,
        _TYPE_RESISTOR,
        "Rs",
        [(0, "R", 0.0 if series_resistance is None else series_resistance)],
    )

    # Elements sit in an implicit series at the top level, so each R-C pair needs only its own parallel block.
    index = 1
    for number, (resistance, capacitance) in enumerate(pairs, start=1):
        lines += _element(index, _TYPE_BEGIN_PARALLEL, "Begin_Parallel")
        index += 1
        lines += _element(
            index, _TYPE_CAPACITOR, f"C{number}", [(1, "T", capacitance)]
        )
        index += 1
        lines += _element(index, _TYPE_RESISTOR, f"R{number}", [(0, "R", resistance)])
        index += 1
        lines += _element(index, _TYPE_END_PARALLEL, "End_Parallel")
        index += 1

    # The count brackets the block and must equal the number of elements in it, counting the parallel markers.
    document = [
        *_MDL_SETTINGS.split("\n"),
        f"Begin Circuit Model:          {index}",
        *lines,
        f"End Circuit Model:            {index}",
    ]
    path = Path(path)
    with open(path, "w", newline="", encoding="ascii", errors="replace") as handle:
        handle.write(_EOL.join(document) + _EOL)
    return path


def estimate_series_resistance(dataset: EISDataset) -> Optional[float]:
    """The sweep's ohmic resistance, which the DRT itself does not report, for
    seeding Rs in the exported model. None when the sweep has no points left.
    """
    from core.ecm import ohmic_resistance

    frequencies = dataset.data.get_frequencies(masked=False)
    impedances = dataset.data.get_impedances(masked=False)
    if len(frequencies) == 0:
        return None
    return ohmic_resistance(frequencies, impedances)
