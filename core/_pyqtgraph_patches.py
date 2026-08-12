"""Monkey patch for pyqtgraph's SVG exporter. Call apply() before exporting;
gui.figure_panes does so.

Unlike core._pyimpspec_patches, a target that has moved is not an error here:
this works around an upstream bug, so a pyqtgraph that no longer carries the
broken code is the outcome being aimed at rather than a problem. The patch
stands down and leaves whatever that version does in place.
"""

import importlib
import re

# The pyqtgraph this was written against; kept for the next reader to check the
# bug against, since apply() stands down silently rather than pinning it.
PATCHED_VERSION = "0.14.0"

# The SVG path commands that carry no coordinates. correctCoordinates splits a
# path's `d` on whitespace and unpacks two comma-separated numbers out of every
# token, so one bare closepath aborts the entire export with "not enough values
# to unpack" -- and Qt ends every filled rectangle with one, including the
# ViewBox border that every plot this app draws has.
_CLOSEPATH = re.compile(r"\s*[Zz]\s*$")

_applied = False


def _strip_closepath(node) -> list:
    """Take the trailing closepath off every <path> under `node`, returning what
    was taken so it can be put back. These paths all close onto the point they
    opened on and so draw identically without it, but it is restored regardless
    rather than quietly altering the exported drawing.
    """
    stripped = []
    for element in node.getElementsByTagName("path"):
        data = element.getAttribute("d")
        match = _CLOSEPATH.search(data)
        if match is None:
            continue
        remainder = data[: match.start()]
        # correctCoordinates skips an empty `d`, which would leave the closepath
        # below as the whole path; leave those alone rather than emptying them.
        if not remainder.strip():
            continue
        element.setAttribute("d", remainder)
        # Normalised to the bare command: the rewritten `d` comes back with a
        # trailing space of its own, and two in a row would read as an empty
        # token to anything that splits it again.
        stripped.append((element, match.group().strip()))
    return stripped


def apply() -> bool:
    """Patch pyqtgraph once, and report whether the workaround is in place.

    False means the exporter no longer looks like the version this was written
    against, and is left alone -- see the module docstring for why that is an
    outcome rather than a failure.
    """
    global _applied
    if _applied:
        return True

    try:
        module = importlib.import_module("pyqtgraph.exporters.SVGExporter")
    except ImportError:
        return False
    original = getattr(module, "correctCoordinates", None)
    if original is None:
        return False

    def correctCoordinates(node, defs, item, options):
        stripped = _strip_closepath(node)
        # Restored even when the export fails, so a half-patched DOM is never
        # what a caller retries against.
        try:
            return original(node, defs, item, options)
        finally:
            for element, closepath in stripped:
                element.setAttribute(
                    "d", f"{element.getAttribute('d').rstrip()} {closepath}"
                )

    module.correctCoordinates = correctCoordinates
    _applied = True
    return True
