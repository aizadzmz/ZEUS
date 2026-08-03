import re

import numpy as np
from pyimpspec import DataSet

from core.circuit_diagram import (
    build_fit_diagram,
    build_preview_diagram,
    format_quantity,
)
from core.ecm import CIRCUIT_PRESETS, run_ecm_fit
from core.io_utils import EISDataset

# Same synthetic spectrum as test_ecm.py: R0 + two parallel RC pairs, so the
# values annotated on the schematic below have a known right answer.
R0_TRUE, R1_TRUE, R2_TRUE = 10.0, 50.0, 30.0
f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = R0_TRUE + R1_TRUE / (1 + 1j * w * 5e-3) + R2_TRUE / (1 + 1j * w * 0.3)
dataset = EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file="synthetic")

TEXTS = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
TEXT_TAGS = re.compile(r'<text[^>]*\bx="(-?[\d.]+)"[^>]*\by="(-?[\d.]+)"[^>]*>')
VIEWBOX = re.compile(r'viewBox="(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)"')
FONTSIZE = re.compile(r'font-size="([\d.]+)"')


def labels(svg: bytes):
    """Every label string in a diagram, tags stripped."""
    return [re.sub(r"<[^>]+>", "", t) for t in TEXTS.findall(svg.decode("utf-8"))]


# --- Value formatting ---
# Engineering prefixes, because nobody quotes a double-layer capacitance in
# scientific notation.
assert format_quantity(10.0, "ohm") == "10 Ω"
assert format_quantity(1234.0, "ohm") == "1.234 kΩ"
assert format_quantity(1.2e-6, "F") == "1.2 µF"
assert format_quantity(9.9e-10, "F") == "990 pF"
assert format_quantity(0.0, "ohm") == "0 Ω"
# Dimensionless quantities (the CPE exponent) keep their plain spelling: a
# prefix with no unit after it would read as a typo.
assert format_quantity(0.9012, "") == "0.9012"
assert format_quantity(float("nan"), "ohm") == "—"
# Past the last prefix the mantissa is allowed to leave 1-999 rather than the
# unit losing its meaning.
assert format_quantity(1e-18, "F").endswith(" fF"), format_quantity(1e-18, "F")
print("format_quantity OK")


# --- Every shipped preset draws ---
# The presets populate a dropdown, and the tab previews whichever is picked,
# so an element the drawing code cannot place is a crash on selection.
for name, cdc in CIRCUIT_PRESETS:
    svg = build_preview_diagram(cdc)
    assert svg.startswith(b"<svg"), name
    # Every element is named, whether or not it has a symbol of its own --
    # unmapped ones (Warburg, Gerischer) fall back to a box, and the name is
    # then the only thing saying what the box is.
    assert len(labels(svg)) >= cdc.count("R"), name
print(f"all {len(CIRCUIT_PRESETS)} presets draw OK")

# A half-typed code is the caller's problem to catch (the GUI's live
# validation does), not something to be silently drawn as an empty circuit.
try:
    build_preview_diagram("R(RC")
except Exception:
    pass
else:
    raise AssertionError("malformed CDC should not produce a diagram")

# Unfitted circuits are annotated with their initial values, which is what
# makes the preview useful for a circuit built from DRT peaks.
preview = labels(build_preview_diagram("R{R=12.5}(RC)"))
assert "12.5 Ω" in preview, preview
print("preview draws initial values OK")


# --- A fit's values land on its components ---
result = run_ecm_fit(dataset, "R(RC)(RQ)")
svg = build_fit_diagram(result)
text = labels(svg)

# Names use unicode subscripts rather than pyimpspec's "R_1" spelling.
assert "R₁" in text and "Q₁" in text, text
assert not any("_" in label for label in text), text

# The fitted resistances are on the drawing, in the same order the circuit
# puts them, and carry an error bar each.
resistances = [label for label in text if label.endswith("%") and "Ω" in label]
assert len(resistances) == 3, text
for label, expected in zip(resistances, (R0_TRUE, R1_TRUE, R2_TRUE)):
    value = float(label.split()[0])
    assert abs(value - expected) / expected < 0.02, (label, expected)
print("fitted values annotate their components OK:", resistances)

# A two-parameter element gets one <text> per parameter. They used to share
# one multi-line label, whose <tspan dy=...> line breaks Qt's SVG renderer
# ignores -- printing Y and n on top of each other.
assert any(label.startswith("Y = ") for label in text), text
assert any(label.startswith("n = ") for label in text), text
print("multi-parameter elements stack one line per parameter OK")

# Qt implements SVG 1.2 Tiny, which has no dominant-baseline: if schemdraw
# emits it, every label lands a line-height from where it belongs.
assert b"dominant-baseline" not in svg
print("SVG is Qt-compatible OK (no dominant-baseline)")


# --- Nothing is drawn outside the canvas ---
# The bounding box schemdraw computes underestimates label extents, which
# used to clip the top row of element names off the image entirely.
def assert_labels_inside(svg: bytes, what: str) -> None:
    document = svg.decode("utf-8")
    x0, y0, width, height = (float(v) for v in VIEWBOX.search(document).groups())
    sizes = [float(s) for s in FONTSIZE.findall(document)]
    slack = max(sizes) if sizes else 0.0
    for x, y in TEXT_TAGS.findall(document):
        x, y = float(x), float(y)
        assert x0 <= x <= x0 + width, (what, "x", x, (x0, width))
        # y is the text's baseline, so allow a line either side of it.
        assert y0 - slack <= y <= y0 + height + slack, (what, "y", y, (y0, height))


assert_labels_inside(svg, "R(RC)(RQ) fit")
for name, cdc in CIRCUIT_PRESETS:
    assert_labels_inside(build_preview_diagram(cdc), name)
print("every label falls inside the canvas OK")

print("ALL CIRCUIT DIAGRAM CHECKS PASSED")
