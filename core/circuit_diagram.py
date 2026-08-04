"""Schematic drawings of an equivalent circuit, annotated per component."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from pyimpspec import Circuit
    from pyimpspec.analysis.fitting import FitResult

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


def _unit_text(unit: str) -> str:
    """pyimpspec's ASCII unit string as something readable on a diagram."""
    if unit in _UNIT_TEXT:
        return _UNIT_TEXT[unit]
    return unit.replace("*", "·")


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
) -> bytes:
    """Walk the circuit and emit the schematic as SVG bytes."""
    import schemdraw.elements as elm
    from pyimpspec import Element, Parallel, Series
    from schemdraw import Drawing
    from schemdraw.backends.svg import config as svg_config

    # Qt's SVG renderer is SVG 1.2 Tiny, which has no dominant-baseline, so
    # every label would be drawn a line-height off. schemdraw's "Batik" mode
    # positions text by computed y instead.
    svg_config.useBatik = True

    symbols = _schemdraw_symbols()

    def draw_element(element, drawing: Drawing) -> None:
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

    def draw_parallel(parallel, drawing: Drawing) -> None:
        branches = list(parallel)
        heights = [height(branch) for branch in branches]

        # Drop to the lowest branch, remembering each rung on the way down,
        # so the branches can be drawn bottom-up and popped back to the node.
        for index in range(len(branches) - 1):
            drawing.push()
            drawing.add(elm.Line(l=heights[index]).down())

        total = width(parallel)
        for index, branch in reversed(list(enumerate(branches))):
            draw_branch(branch, drawing)
            padding = total - width(branch)
            if padding > 0:
                drawing.add(elm.Line(l=padding).right())
            if index > 0:
                drawing.add(elm.Line(l=heights[index - 1]).up())
                drawing.pop()

    def draw_branch(item, drawing: Drawing) -> None:
        if isinstance(item, Element):
            draw_element(item, drawing)
        elif isinstance(item, Series):
            draw_series(item, drawing)
        else:
            draw_parallel(item, drawing)

    def draw_series(series, drawing: Drawing, outermost: bool = False) -> None:
        items = list(series)
        for index, item in enumerate(items):
            if isinstance(item, Parallel):
                # Short leads keep adjacent parallel blocks (the usual
                # R(RQ)(RQ) shape) off a shared vertical rail, which would
                # read as one four-branch node.
                if not outermost:
                    drawing.add(elm.Line(l=0.5).right())
                draw_parallel(item, drawing)
                if not outermost or (
                    index < len(items) - 1 and isinstance(items[index + 1], Parallel)
                ):
                    drawing.add(elm.Line(l=0.5).right())
            else:
                draw_branch(item, drawing)

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
    draw_series(circuit.get_connections(recursive=False)[0], drawing, outermost=True)
    drawing.add(elm.Line(l=1.0).right())
    drawing.add(elm.Dot(open=True))

    return _pad_svg(drawing.get_imagedata("svg"), SVG_PADDING)


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
    )


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
    )
