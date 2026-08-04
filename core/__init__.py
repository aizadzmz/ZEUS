"""Public surface of the analysis core."""

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
