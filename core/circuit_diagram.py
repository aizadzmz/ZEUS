"""Schematic drawings of an equivalent circuit, annotated per component.

Renders a circuit to a standalone SVG (bytes) that gui/figure_panes.py hosts
in the ECM Parameters tab. The point of drawing it at all is that a circuit
description code is a topology written sideways -- "R(RQ)(RQ)" says nothing
about which semicircle is which -- and the point of annotating it is that a
fitted value only means something once you can see *which* component it
belongs to. So each element carries its name above the symbol and its fitted
value(s) below, in place of the flat parameter table.

pyimpspec offers Circuit.to_drawing(), but it allows one label string per
element and bakes in its own styling, so the walk over the circuit's
Series/Parallel tree is reimplemented here (the layout arithmetic follows
pyimpspec's). That buys three things the table cannot show at a glance:
name and value in different colors on opposite sides of the symbol, a
theme-aware stroke color, and element widths sized for the value text.

Like core/ecm.py, every pyimpspec/schemdraw import lives inside a function:
the GUI must be able to import this module while building its sidebar without
paying pyimpspec's ~4 s import (see core/__init__.py).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from pyimpspec import Circuit
    from pyimpspec.analysis.fitting import FitResult

# Element bodies are UNIT_WIDTH drawing units long and branches of a parallel
# connection sit NODE_HEIGHT apart. Both are wider/taller than pyimpspec's
# defaults (2.0 and 1.5) because every element here carries a label above
# *and* below: the extra width keeps "1.234 mS·sⁿ" from running into its
# neighbor, and the extra height keeps a two-line value block off the branch
# below it.
UNIT_WIDTH = 3.4
NODE_HEIGHT = 2.6

# Label sizes in points. Values are set a little smaller than names so the
# name reads as the heading of the pair.
NAME_FONTSIZE = 12.0
VALUE_FONTSIZE = 10.0
# How far a label sits off the element body, in drawing units. Enough to
# clear the wire the element sits on, which the schemdraw default (0.1) is
# not once the label is a two-line block.
LABEL_OFFSET = 0.45
# Gap between successive value lines under one element, in drawing units
# (1 unit = 36 pt), sized for VALUE_FONTSIZE plus a little leading.
LINE_SPACING = 0.36

# Points of blank space added around the finished drawing. schemdraw sizes
# the SVG from a bounding box that estimates text extents, and underestimates
# the top label enough to clip it; this is the slack that keeps every label
# inside the viewBox (and stops the diagram from butting against the frame).
SVG_PADDING = 16.0

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Engineering prefixes, keyed by the power of ten they stand for.
_PREFIXES = {
    12: "T", 9: "G", 6: "M", 3: "k", 0: "",
    -3: "m", -6: "µ", -9: "n", -12: "p", -15: "f",
}
_MIN_EXPONENT, _MAX_EXPONENT = min(_PREFIXES), max(_PREFIXES)

# pyimpspec spells units in ASCII; these are the same units set in the type
# the rest of the app uses (core.plotting's axis labels use Ω too). Anything
# not listed falls back to _unit_text's generic cleanup.
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
    """One fitted value as it appears under its component, e.g. 10.02 Ω,
    1.234 µF, or a bare 0.9012 for the dimensionless CPE exponent.

    Dimensionless quantities keep their plain decimal spelling -- "0.9 " with
    a prefix glued to nothing would read as a typo -- while everything else
    is scaled to an engineering prefix, which is how these values are quoted
    in practice (nobody writes 1.2e-06 F).
    """
    if not math.isfinite(value):
        return "—"

    text_unit = _unit_text(unit)
    if not text_unit:
        return f"{value:.4g}"
    if value == 0:
        return f"0 {text_unit}"

    exponent = math.floor(math.log10(abs(value)))
    # Round down to a multiple of three so the mantissa lands in 1-999, then
    # clamp: past femto/tera there is no prefix left and the plain %g
    # spelling (which the clamped branch produces, e.g. "0.001 f") is still
    # honest. Values that extreme mean the fit diverged anyway.
    scale = max(_MIN_EXPONENT, min(_MAX_EXPONENT, (exponent // 3) * 3))
    return f"{value / 10.0**scale:.4g} {_PREFIXES[scale]}{text_unit}"


def _pretty_name(name: str) -> str:
    """'R_1' -> 'R₁'. Labels from the extended CDC syntax (R{R=5:ct} ->
    'R_ct') keep their text, since only digits have subscript glyphs."""
    base, _, index = name.partition("_")
    return base + index.translate(_SUBSCRIPT) if index else name


def _value_lines(
    element_name: str,
    element,
    parameters: Optional[Dict[str, Dict[str, object]]],
    show_errors: bool,
) -> List[str]:
    """The lines drawn under one component.

    With a fit's `parameters` in hand these are the fitted values (plus their
    relative standard errors); without one they are the element's current
    values, which for a circuit built from DRT peaks are real starting
    estimates rather than pyimpspec's generic defaults.

    Single-parameter elements (R, C, L) show the bare value -- "10.02 Ω"
    under a resistor needs no "R =" to explain it -- while multi-parameter
    ones (Q, W, Zarc) name each, since "0.9" alone would be ambiguous.
    """
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
    """The ' ±x%' suffix on a fitted value, or '' when the fitting method
    produced no covariance matrix to estimate it from (see
    core.ecm.run_ecm_fit -- gradient-free methods report NaN here, which is
    a property of the method rather than of this parameter, and is called
    out once in the text report instead of on every component).

    Held to two significant figures and floored at '<0.01%': the diagram is
    for reading the fit at a glance, and '±0.00%' looks like a bug rather
    than like a tightly-determined parameter."""
    relative = parameter.get_relative_error() * 100.0
    if not math.isfinite(relative):
        return ""
    if relative < 0.01:
        return " ±<0.01%"
    return f" ±{relative:.2g}%"


def _schemdraw_symbols() -> dict:
    """pyimpspec element type -> schemdraw symbol class. Anything unmapped
    (Warburg in its several forms, Gerischer, Havriliak-Negami, the
    transmission lines) is drawn as a plain box, which is the conventional
    way to show a distributed element; its name label says which it is."""
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
    """Walk the circuit and emit the schematic as SVG bytes.

    The width/height arithmetic and the push/pop bracketing of parallel
    branches follow pyimpspec's Circuit.to_drawing: a series connection is
    laid out left to right, a parallel one drops a vertical rail, draws each
    branch padded to the widest, and climbs back up. What differs is only
    what gets attached to each element (two labels rather than one) and that
    nothing here is styled by schemdraw's global defaults.
    """
    import schemdraw.elements as elm
    from pyimpspec import Element, Parallel, Series
    from schemdraw import Drawing
    from schemdraw.backends.svg import config as svg_config

    # Qt's SVG renderer implements SVG 1.2 Tiny, which has no
    # dominant-baseline: left on, every label would be drawn a line-height
    # away from where schemdraw meant it. schemdraw's "Batik" mode is exactly
    # the workaround (it positions text by computed y instead). Set here
    # rather than at import time so the flag is on for any caller of this
    # module, and only for callers of this module.
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
        # One label call per line, each pushed a little further from the
        # element, rather than one multi-line label or the list schemdraw
        # also accepts. A list would space the strings *along* the element
        # (printing a CPE's Y and n side by side), and a newline-joined
        # string becomes <tspan dy=...>, which Qt's SVG 1.2 Tiny renderer
        # ignores -- both collapse a two-parameter element onto one line.
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
                # Short leads keep two adjacent parallel blocks -- the usual
                # R(RQ)(RQ) shape -- from sharing a vertical rail, which
                # would read as one four-branch node.
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
    """Grow an SVG's viewBox (and its declared size to match) by `padding`
    points on every side, leaving the drawing itself untouched.

    Done to the finished document rather than by asking schemdraw for a
    larger margin because the margin is applied in drawing units *before*
    the text-extent estimate that undersizes the box in the first place.
    Returns the document unchanged if the header does not look as expected,
    since a diagram with a clipped label still beats no diagram at all.
    """
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
    value(s) and relative standard error.

    A "±0%" is a value pinned to more precision than two decimals of percent
    can show; a "—" is a parameter the fitting method could not put an error
    on at all (see core.ecm.run_ecm_fit on gradient-free methods).
    """
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
    labelled with its initial values.

    Raises whatever pyimpspec's parser raises for a malformed code; callers
    that draw as the user types should validate first (core.ecm.validate_cdc).
    """
    from pyimpspec import parse_cdc

    return _build_svg(
        parse_cdc(cdc),
        None,
        foreground=foreground,
        accent=accent,
        show_errors=False,
    )
