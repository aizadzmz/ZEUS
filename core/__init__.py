"""Public surface of the analysis core.

Every name here is resolved on first attribute access rather than at import
time (PEP 562). Importing any submodule -- `core.plotting`, say -- runs this
file first, so eager re-exports made every one of them pay for pyimpspec, and
through it scipy.signal and sympy: ~4 s that the GUI spent before its window
appeared. See gui/app.py, which now shows the window off the light stack and
warms the heavy modules in the background.

`from core import parse_eis_file` still works exactly as before; it just
imports core.io_utils at that moment instead of at startup.
"""

from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    "EISDataset": "core.io_utils",
    "EISParseError": "core.io_utils",
    "parse_eis_file": "core.io_utils",
    "parse_generic_file": "core.generic_parser",
}

__all__ = list(_LAZY_EXPORTS)

if TYPE_CHECKING:  # keeps type checkers and IDE completion working
    from core.generic_parser import parse_generic_file
    from core.io_utils import EISDataset, EISParseError, parse_eis_file


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    # Cache on the package so later lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
