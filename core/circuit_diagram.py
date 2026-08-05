"""Schematic drawings of an equivalent circuit, annotated per component.

The walk can also report where each part landed (see HitRegion), which is what
lets gui/circuit_canvas.py map a click back to a core.circuit_model node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import repeat
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pyimpspec import Circuit
    from pyimpspec.analysis.fitting import FitResult

    from core.circuit_model import ConnectionNode

# Element body length and parallel-branch separation, in drawing units. Larger
# than pyimpspec's defaults (2.0, 1.5) because every element carries a label
# above *and* below: the width keeps "1.234 mS·sⁿ" clear of its neighbor, the
# height keeps a two-line value block off the branch below.
UNIT_WIDTH = 3.4
NODE_HEIGHT = 2.6

# Label sizes in points; values smaller than names so the name reads as the
# heading of the pair.
NAME_FONTSIZE = 12.0
VALUE_FONTSIZE = 10.0
# Label distance from the element body, in drawing units. schemdraw's default
# (0.1) does not clear the wire once the label is a two-line block.
LABEL_OFFSET = 0.45
# Gap between value lines under one element, in drawing units (1 unit = 36 pt).
LINE_SPACING = 0.36

# Blank space in points around the finished drawing. schemdraw underestimates
# text extents when sizing the SVG and clips the top label without this.
SVG_PADDING = 16.0

# --- hit-region geometry, in drawing units -----------------------------------
# A placed element spans the full UNIT_WIDTH including leads, so neighbours in a
# series touch; insetting its box leaves room for the gap target between them.
ELEMENT_INSET = 0.6
# Half-extents of a gap box, kept under ELEMENT_INSET so the two never overlap.
GAP_HALF_WIDTH = 0.5
GAP_HALF_HEIGHT = 0.5
# A parallel is grabbed by the vertical rail joining its branches.
RAIL_HALF_WIDTH = 0.35
MIN_HALF_HEIGHT = 0.5

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Engineering prefixes, keyed by the power of ten they stand for.
_PREFIXES = {
    12: "T", 9: "G", 6: "M", 3: "k", 0: "",
    -3: "m", -6: "µ", -9: "n", -12: "p", -15: "f",
}
_MIN_EXPONENT, _MAX_EXPONENT = min(_PREFIXES), max(_PREFIXES)

# pyimpspec spells units in ASCII; these are the typeset forms the rest of the
# app uses. Anything unlisted falls back to _unit_text's generic cleanup.
_UNIT_TEXT = {
    "": "",
    "ohm": "Ω",
    "S*s^n": "S·sⁿ",
    "S*s^(1/2)": "S·s^½",
    "ohm*s^(1/2)": "Ω·s^½",
}


@dataclass(frozen=True)
class HitRegion:
    """One clickable area, in the finished SVG's viewBox points. kind is
    "element", "connection" (a parallel's rail), or "gap" -- an insertion point
    in the series `node_id`, at position `index`."""

    kind: str
    node_id: int
    rect: Tuple[float, float, float, float]  # x, y, width, height
    index: int = -1

    def contains(self, x: float, y: float) -> bool:
        left, top, width, height = self.rect
        return left <= x <= left + width and top <= y <= top + height

    @property
    def center(self) -> Tuple[float, float]:
        left, top, width, height = self.rect
        return left + width / 2.0, top + height / 2.0


@dataclass
class CircuitDrawing:
    """A schematic plus the map from its pixels back to the circuit tree."""

    svg: bytes
    viewbox: Tuple[float, float, float, float]  # x, y, width, height
    regions: List[HitRegion] = field(default_factory=list)

    def region_at(self, x: float, y: float) -> Optional[HitRegion]:
        """The region under a viewBox point; elements win ties."""
        hits = [region for region in self.regions if region.contains(x, y)]
        if not hits:
            return None
        return min(hits, key=lambda region: 0 if region.kind == "element" else 1)


def unit_text(unit: str) -> str:
    """pyimpspec's ASCII unit string as something readable on a diagram."""
    if unit in _UNIT_TEXT:
        return _UNIT_TEXT[unit]
    return unit.replace("*", "·")


_unit_text = unit_text


def format_quantity(value: float, unit: str) -> str:
    """One fitted value as it appears under its component, e.g. 10.02 Ω, 1.234
    µF, or a bare 0.9012 for the dimensionless CPE exponent."""
    if not math.isfinite(value):
        return "—"

    text_unit = _unit_text(unit)
    if not text_unit:
        return f"{value:.4g}"
    if value == 0:
        return f"0 {text_unit}"

    exponent = math.floor(math.log10(abs(value)))
    # Round down to a multiple of three so the mantissa lands in 1-999, then
    # clamp -- past femto/tera there is no prefix left, and the plain %g
    # spelling the clamp produces is still honest.
    scale = max(_MIN_EXPONENT, min(_MAX_EXPONENT, (exponent // 3) * 3))
    return f"{value / 10.0**scale:.4g} {_PREFIXES[scale]}{text_unit}"


def _pretty_name(name: str) -> str:
    """'R_1' -> 'R₁'. Labels from the extended CDC syntax (R{R=5:ct} -> 'R_ct')
    keep their text, since only digits have subscript glyphs."""
    base, _, index = name.partition("_")
    return base + index.translate(_SUBSCRIPT) if index else name


def _value_lines(
    element_name: str,
    element,
    parameters: Optional[Dict[str, Dict[str, object]]],
    show_errors: bool,
) -> List[str]:
    """The lines drawn under one component."""
    units = element.get_units()
    fitted = (parameters or {}).get(element_name)

    lines: List[str] = []
    if fitted is not None:
        items = [(symbol, p.get_value(), p.get_unit(), p) for symbol, p in fitted.items()]
    else:
        items = [(symbol, value, units.get(symbol, ""), None) for symbol, value in element.get_values().items()]

    named = len(items) > 1
    for symbol, value, unit, parameter in items:
        text = format_quantity(value, unit)
        if named:
            text = f"{symbol} = {text}"
        if parameter is not None and show_errors:
            text += _error_text(parameter)
        lines.append(text)
    return lines


def _error_text(parameter) -> str:
    """The ' ±x%' suffix on a fitted value, or '' when the fit produced no
    covariance matrix. Floored at '<0.01%'."""
    relative = parameter.get_relative_error() * 100.0
    if not math.isfinite(relative):
        return ""
    if relative < 0.01:
        return " ±<0.01%"
    return f" ±{relative:.2g}%"


def _schemdraw_symbols() -> dict:
    """pyimpspec element type -> schemdraw symbol class. Unmapped types draw as
    a plain box, labelled with their name."""
    import schemdraw.elements as elm
    from pyimpspec import (
        Capacitor,
        ConstantPhaseElement,
        Inductor,
        ModifiedInductor,
        Resistor,
    )

    return {
        Resistor: elm.ResistorIEEE,
        Capacitor: elm.Capacitor,
        ConstantPhaseElement: elm.CPE,
        Inductor: elm.Inductor2,
        ModifiedInductor: elm.Inductor2,
    }


def _build_svg(
    circuit: Circuit,
    parameters: Optional[Dict[str, Dict[str, object]]],
    foreground: str,
    accent: str,
    show_errors: bool,
    tree: Optional[ConnectionNode] = None,
) -> CircuitDrawing:
    """Walk the circuit and draw it. Passing the core.circuit_model `tree` the
    circuit was built from turns on hit-region collection -- the two are walked
    in lockstep, since to_circuit builds connections child-for-child in order."""
    import schemdraw.elements as elm
    from pyimpspec import Element, Parallel, Series
    from schemdraw import Drawing
    from schemdraw.backends import svg as svg_backend
    from schemdraw.backends.svg import config as svg_config

    # Qt's SVG renderer is SVG 1.2 Tiny, which has no dominant-baseline, so
    # every label would be drawn a line-height off. schemdraw's "Batik" mode
    # positions text by computed y instead.
    svg_config.useBatik = True

    symbols = _schemdraw_symbols()
    # Collected in drawing units and converted once the scale is known.
    boxes: List[Tuple[str, int, int, Tuple[float, float, float, float]]] = []

    def record(kind: str, node, bounds, index: int = -1) -> None:
        if node is not None:
            boxes.append((kind, node.node_id, index, bounds))

    def children_of(node) -> object:
        return node.children if node is not None else repeat(None)

    def draw_element(element, drawing: Drawing, node=None) -> None:
        name = circuit.get_element_name(element)
        symbol = symbols.get(type(element), elm.ResistorIEC)()
        symbol.label(
            _pretty_name(name),
            loc="top",
            ofst=LABEL_OFFSET,
            fontsize=NAME_FONTSIZE,
            color=foreground,
        )
        # One label call per line, each pushed further from the element. A list
        # would space them *along* the element, and a newline-joined string
        # becomes <tspan dy=...>, which Qt's SVG 1.2 Tiny renderer ignores --
        # both collapse a two-parameter element onto one line.
        for offset, line in enumerate(_value_lines(name, element, parameters, show_errors)):
            symbol.label(
                line,
                loc="bottom",
                ofst=LABEL_OFFSET + offset * LINE_SPACING,
                fontsize=VALUE_FONTSIZE,
                color=accent,
            )
        drawing.add(symbol.right())
        if node is not None:
            # includetext=False: labels sit clear of the body and would make
            # adjacent components' targets overlap.
            left, bottom, right, top = symbol.get_bbox(transform=True, includetext=False)
            record("element", node, (left + ELEMENT_INSET, bottom, right - ELEMENT_INSET, top))

    def width(item) -> float:
        if isinstance(item, Element):
            return UNIT_WIDTH
        if isinstance(item, Series):
            # The +1.0 is the half-unit of lead drawn either side of a
            # parallel block nested inside a series one (see draw_series).
            return sum(
                width(child) + (1.0 if isinstance(child, Parallel) else 0.0)
                for child in item
            )
        return max(width(child) for child in item)

    def height(item) -> float:
        if isinstance(item, Element):
            return NODE_HEIGHT
        if isinstance(item, Series):
            return max(height(child) for child in item)
        return sum(height(child) for child in item)

    def draw_parallel(parallel, drawing: Drawing, node=None) -> None:
        branches = list(parallel)
        heights = [height(branch) for branch in branches]
        top_x, top_y = drawing.here

        # Drop to the lowest branch, remembering each rung on the way down,
        # so the branches can be drawn bottom-up and popped back to the node.
        for index in range(len(branches) - 1):
            drawing.push()
            drawing.add(elm.Line(l=heights[index]).down())

        # The rail just descended is the parallel's own handle.
        drop = max(sum(heights[:-1]), 2 * MIN_HALF_HEIGHT)
        record(
            "connection",
            node,
            (top_x - RAIL_HALF_WIDTH, top_y - drop, top_x + RAIL_HALF_WIDTH, top_y),
        )

        total = width(parallel)
        for index, (branch, child) in reversed(
            list(enumerate(zip(branches, children_of(node))))
        ):
            draw_branch(branch, drawing, child)
            padding = total - width(branch)
            if padding > 0:
                drawing.add(elm.Line(l=padding).right())
            if index > 0:
                drawing.add(elm.Line(l=heights[index - 1]).up())
                drawing.pop()

    def draw_branch(item, drawing: Drawing, node=None) -> None:
        if isinstance(item, Element):
            draw_element(item, drawing, node)
        elif isinstance(item, Series):
            draw_series(item, drawing, node)
        else:
            draw_parallel(item, drawing, node)

    def draw_series(series, drawing: Drawing, node=None, outermost: bool = False) -> None:
        items = list(series)
        for index, (item, child) in enumerate(zip(items, children_of(node))):
            gap_at(index, drawing, node)
            if isinstance(item, Parallel):
                # Short leads keep adjacent parallel blocks (the usual
                # R(RQ)(RQ) shape) off a shared vertical rail, which would
                # read as one four-branch node.
                if not outermost:
                    drawing.add(elm.Line(l=0.5).right())
                draw_parallel(item, drawing, child)
                if not outermost or (
                    index < len(items) - 1 and isinstance(items[index + 1], Parallel)
                ):
                    drawing.add(elm.Line(l=0.5).right())
            else:
                draw_branch(item, drawing, child)
        gap_at(len(items), drawing, node)

    def gap_at(index: int, drawing: Drawing, node) -> None:
        """An insertion target on the wire at the cursor's current position."""
        x, y = drawing.here
        record(
            "gap",
            node,
            (x - GAP_HALF_WIDTH, y - GAP_HALF_HEIGHT, x + GAP_HALF_WIDTH, y + GAP_HALF_HEIGHT),
            index,
        )

    drawing = Drawing(canvas="svg")
    drawing.config(
        unit=UNIT_WIDTH,
        fontsize=NAME_FONTSIZE,
        font="sans-serif",
        color=foreground,
        lw=1.8,
    )
    drawing.add(elm.Dot(open=True))
    drawing.add(elm.Line(l=1.0).right())
    # A Circuit always reports exactly one outermost Series (the implicit
    # square brackets of the CDC), which is the connection to walk.
    draw_series(circuit.get_connections(recursive=False)[0], drawing, tree, outermost=True)
    drawing.add(elm.Line(l=1.0).right())
    drawing.add(elm.Dot(open=True))

    data = _pad_svg(drawing.get_imagedata("svg"), SVG_PADDING)
    # The svg backend maps drawing units to viewBox points as (x, -y) * scale.
    scale = getattr(svg_backend, "PT_PER_IN", 72.0) * drawing.dwgparams.get(
        "inches_per_unit", 0.5
    )
    return CircuitDrawing(
        svg=data,
        viewbox=_svg_viewbox(data),
        regions=[_to_region(kind, node_id, index, bounds, scale) for kind, node_id, index, bounds in boxes],
    )


def _to_region(kind, node_id, index, bounds, scale: float) -> HitRegion:
    left, bottom, right, top = bounds
    half = max(MIN_HALF_HEIGHT, (top - bottom) / 2.0)
    middle = (top + bottom) / 2.0
    # y flips, so the drawing's top edge becomes the rect's smaller coordinate.
    return HitRegion(
        kind=kind,
        node_id=node_id,
        index=index,
        rect=(
            left * scale,
            -(middle + half) * scale,
            (right - left) * scale,
            2 * half * scale,
        ),
    )


def _svg_viewbox(data: bytes) -> Tuple[float, float, float, float]:
    """Read the finished SVG's viewBox, so regions and picture share a frame."""
    import re

    box = re.search(rb'viewBox="(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)"', data[:2000])
    if box is None:
        return (0.0, 0.0, 1.0, 1.0)
    return tuple(float(value) for value in box.groups())  # type: ignore[return-value]


def _pad_svg(data: bytes, padding: float) -> bytes:
    """Grow an SVG's viewBox (and declared size) by `padding` points on every
    side, leaving the drawing untouched."""
    import re

    header = re.match(rb"<svg[^>]*>", data)
    if header is None:
        return data
    box = re.search(
        rb'viewBox="(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)"', header.group()
    )
    if box is None:
        return data

    x, y, width, height = (float(value) for value in box.groups())
    x, y = x - padding, y - padding
    width, height = width + 2 * padding, height + 2 * padding

    updated = header.group()
    updated = re.sub(
        rb'viewBox="[^"]*"',
        f'viewBox="{x:g} {y:g} {width:g} {height:g}"'.encode(),
        updated,
    )
    updated = re.sub(rb'\bwidth="[^"]*"', f'width="{width:g}pt"'.encode(), updated)
    updated = re.sub(rb'\bheight="[^"]*"', f'height="{height:g}pt"'.encode(), updated)
    return updated + data[header.end():]


def build_fit_diagram(
    result: FitResult,
    *,
    foreground: str = "#252931",
    accent: str = "#2f5fbf",
    show_errors: bool = True,
) -> bytes:
    """The fitted circuit as SVG, each component labelled with its fitted
    value(s) and relative standard error."""
    return _build_svg(
        result.circuit,
        result.parameters,
        foreground=foreground,
        accent=accent,
        show_errors=show_errors,
    ).svg


def build_preview_diagram(
    cdc: str,
    *,
    foreground: str = "#252931",
    accent: str = "#6b7280",
) -> bytes:
    """The circuit a description code spells out, before anything is fitted,
    labelled with its initial values."""
    from pyimpspec import parse_cdc

    return _build_svg(
        parse_cdc(cdc),
        None,
        foreground=foreground,
        accent=accent,
        show_errors=False,
    ).svg


def build_editor_drawing(
    tree: ConnectionNode,
    *,
    parameters: Optional[Dict[str, Dict[str, object]]] = None,
    foreground: str = "#252931",
    accent: str = "#6b7280",
    show_errors: bool = True,
) -> CircuitDrawing:
    """A circuit tree drawn with its hit regions, for the editable canvas.
    Pass a fit result's `parameters` to annotate with fitted values instead of
    the tree's own initial ones."""
    from core.circuit_model import to_circuit

    return _build_svg(
        to_circuit(tree),
        parameters,
        foreground=foreground,
        accent=accent,
        show_errors=show_errors,
        tree=tree,
    )
