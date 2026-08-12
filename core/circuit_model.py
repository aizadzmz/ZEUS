"""An editable tree standing in for a circuit description code.

A CDC is parsed into plain dataclasses, those are edited, and the result is
serialized back through pyimpspec, so the CDC stays the app's source of truth.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import count
from typing import Dict, Iterator, List, Optional, Tuple, Union

# Per-process only, never serialized: it exists so a click can name the node it hit.
_next_id = count(1).__next__


@dataclass
class ElementNode:
    """One circuit component, with everything the extended CDC syntax can pin."""

    symbol: str  # registry key: "R", "Q", "Ws", ...
    values: Dict[str, float] = field(default_factory=dict)
    lower: Dict[str, float] = field(default_factory=dict)
    upper: Dict[str, float] = field(default_factory=dict)
    fixed: Dict[str, bool] = field(default_factory=dict)
    label: str = ""
    # Container elements hold whole connections; carried through as one opaque box.
    subcircuits: Dict[str, object] = field(default_factory=dict)
    node_id: int = field(default_factory=_next_id)

    def copy(self) -> "ElementNode":
        return replace(
            self,
            values=dict(self.values),
            lower=dict(self.lower),
            upper=dict(self.upper),
            fixed=dict(self.fixed),
            subcircuits=deepcopy(self.subcircuits),
            node_id=_next_id(),
        )


@dataclass
class ConnectionNode:
    """A series or parallel grouping of elements and further connections."""

    kind: str  # "series" | "parallel"
    children: List["Node"] = field(default_factory=list)
    node_id: int = field(default_factory=_next_id)

    def copy(self) -> "ConnectionNode":
        return ConnectionNode(self.kind, [child.copy() for child in self.children])


Node = Union[ElementNode, ConnectionNode]


# --------------------------------------------------------------- construction


def empty_root() -> ConnectionNode:
    """The starting point for a circuit built from nothing: an empty series, the
    same implicit outermost brackets every CDC has."""
    return ConnectionNode("series", [])


def element_symbols() -> Dict[str, type]:
    """{symbol: pyimpspec element class} for the canvas's 'add element' menu.
    Each class carries get_description() for the menu entry's tooltip."""
    from pyimpspec.circuit.registry import get_elements

    return get_elements()


def new_element(symbol: str) -> ElementNode:
    """A component of the given type at pyimpspec's default values and bounds.
    Read off a freshly constructed instance rather than the class, so container
    elements bring their default subcircuits with them."""
    return _node_from(_element_class(symbol)())  # type: ignore[return-value]


def default_values(symbol: str) -> Dict[str, float]:
    """An element type's factory-default parameter values, for the editor's
    'reset' action."""
    return dict(_element_class(symbol).get_default_values())


def _element_class(symbol: str) -> type:
    classes = element_symbols()
    if symbol not in classes:
        raise ValueError(f"Unknown circuit element {symbol!r}.")
    return classes[symbol]


# ------------------------------------------------------------ serialization


def from_cdc(cdc: str) -> ConnectionNode:
    """Parse a circuit description code into an editable tree."""
    from pyimpspec import parse_cdc

    return from_circuit(parse_cdc(cdc))


def from_circuit(circuit) -> ConnectionNode:
    """As from_cdc, for an already-parsed Circuit -- e.g. a fit result's."""
    # A Circuit always reports one outermost Series, the CDC's implicit brackets.
    return _node_from(circuit.get_connections(recursive=False)[0])


def _node_from(item) -> Node:
    from pyimpspec import Element, Parallel, Series  # noqa: F401

    if isinstance(item, Element):
        return ElementNode(
            symbol=type(item).get_symbol(),
            values=dict(item.get_values()),
            lower=dict(item.get_lower_limits()),
            upper=dict(item.get_upper_limits()),
            fixed=dict(item.are_fixed()),
            label=item.get_label(),
            subcircuits=(
                deepcopy(item.get_subcircuits()) if hasattr(item, "get_subcircuits") else {}
            ),
        )

    kind = "series" if isinstance(item, Series) else "parallel"
    if not isinstance(item, (Series, Parallel)):
        raise TypeError(f"Unsupported connection type {type(item).__name__}.")
    return ConnectionNode(kind, [_node_from(child) for child in item])


def to_circuit(root: ConnectionNode):
    """Build a pyimpspec Circuit from the tree."""
    from pyimpspec import Circuit, Series

    connection = _connection_from(root)
    # Circuit's contract is an outermost Series; a root that is not one gets wrapped.
    if not isinstance(connection, Series):
        connection = Series([connection])
    return Circuit(connection)


def _connection_from(node: ConnectionNode):
    from pyimpspec import Parallel, Series

    children = [
        _element_from(child) if isinstance(child, ElementNode) else _connection_from(child)
        for child in node.children
    ]
    return Series(children) if node.kind == "series" else Parallel(children)


def _element_from(node: ElementNode):
    """One node as a pyimpspec Element, following Element.__copy__'s pattern."""
    element = _element_class(node.symbol)(
        **node.values, **deepcopy(node.subcircuits)
    )
    element.set_lower_limits(**node.lower)
    element.set_upper_limits(**node.upper)
    element.set_fixed(**node.fixed)
    element.set_label(node.label)
    return element


def to_cdc(root: ConnectionNode, decimals: int = 12) -> str:
    """The tree as an extended circuit description code. pyimpspec does the
    formatting, so values, bounds, fixed flags and labels all survive."""
    if not root.children:
        return ""
    return to_circuit(root).to_string(decimals)


# -------------------------------------------------------------- inspection


def walk(node: Node) -> Iterator[Node]:
    """Every node in the tree, parents before children."""
    yield node
    if isinstance(node, ConnectionNode):
        for child in node.children:
            yield from walk(child)


def find(root: ConnectionNode, node_id: int) -> Optional[Node]:
    for node in walk(root):
        if node.node_id == node_id:
            return node
    return None


def _locate(root: ConnectionNode, node_id: int) -> Optional[Tuple[ConnectionNode, int]]:
    """The (parent, index) of a node, or None for the root / a missing id."""
    for node in walk(root):
        if isinstance(node, ConnectionNode):
            for index, child in enumerate(node.children):
                if child.node_id == node_id:
                    return node, index
    return None


def element_count(root: ConnectionNode) -> int:
    return sum(1 for node in walk(root) if isinstance(node, ElementNode))


# ------------------------------------------------------------- edit actions
# Each action returns a new root, so a failed edit cannot leave a half-modified tree behind.


def _rebuilt(root: ConnectionNode) -> Tuple[ConnectionNode, Dict[int, Node]]:
    """A deep copy plus {node_id: new node}, so an action can find its target."""
    mapping: Dict[int, Node] = {}

    def clone(node: Node) -> Node:
        if isinstance(node, ElementNode):
            copy: Node = replace(
                node,
                values=dict(node.values),
                lower=dict(node.lower),
                upper=dict(node.upper),
                fixed=dict(node.fixed),
                subcircuits=deepcopy(node.subcircuits),
            )
        else:
            copy = ConnectionNode(
                node.kind, [clone(child) for child in node.children], node.node_id
            )
        mapping[node.node_id] = copy
        return copy

    # Ids survive the clone, so the canvas keeps its selection across an edit.
    return clone(root), mapping  # type: ignore[return-value]


def insert_at(
    root: ConnectionNode, connection_id: int, index: int, symbol: str
) -> ConnectionNode:
    """Insert a new element into a connection at the given position -- what a
    click on a gap between two components does."""
    new_root, mapping = _rebuilt(root)
    target = mapping.get(connection_id)
    if not isinstance(target, ConnectionNode):
        raise ValueError("Insertion target is not a connection.")
    target.children.insert(max(0, min(index, len(target.children))), new_element(symbol))
    return new_root


def insert_after(root: ConnectionNode, node_id: int, symbol: str) -> ConnectionNode:
    """Insert a new element directly after an existing node, in its parent."""
    located = _locate(root, node_id)
    if located is None:
        raise ValueError("Cannot insert after a node that is not in the tree.")
    parent, index = located
    return insert_at(root, parent.node_id, index + 1, symbol)


def append_element(root: ConnectionNode, symbol: str) -> ConnectionNode:
    """Add an element at the end of the outermost series."""
    return insert_at(root, root.node_id, len(root.children), symbol)


def add_in_series(root: ConnectionNode, node_id: int, symbol: str) -> ConnectionNode:
    """Put a new element in series with an existing one. Inside a parallel
    branch that means growing a series around it: (RC) becomes (R[CW])."""
    located = _locate(root, node_id)
    if located is None:
        raise ValueError("Cannot add in series to a node that is not in the tree.")
    parent, _ = located
    if parent.kind == "series":
        return insert_after(root, node_id, symbol)

    new_root, mapping = _rebuilt(root)
    target = mapping[node_id]
    parent, index = _locate(new_root, node_id)  # type: ignore[misc]
    parent.children[index] = ConnectionNode("series", [target, new_element(symbol)])
    return new_root


def add_branch(root: ConnectionNode, connection_id: int, symbol: str) -> ConnectionNode:
    """Add another branch to a parallel connection."""
    new_root, mapping = _rebuilt(root)
    target = mapping.get(connection_id)
    if not isinstance(target, ConnectionNode) or target.kind != "parallel":
        raise ValueError("Branches can only be added to a parallel connection.")
    target.children.append(new_element(symbol))
    return new_root


def wrap_in_parallel(root: ConnectionNode, node_id: int, symbol: str) -> ConnectionNode:
    """Put a node in parallel with a new element: R becomes (RC), the single most
    common edit in EIS modelling."""
    new_root, mapping = _rebuilt(root)
    target = mapping.get(node_id)
    if target is None:
        raise ValueError("Cannot wrap a node that is not in the tree.")
    located = _locate(new_root, node_id)
    if located is None:
        raise ValueError("Cannot wrap the outermost connection.")
    parent, index = located
    parent.children[index] = ConnectionNode("parallel", [target, new_element(symbol)])
    return new_root


def duplicate(root: ConnectionNode, node_id: int) -> ConnectionNode:
    """Copy a node -- element or whole connection -- in beside itself. Copying an
    (RQ) pair is how a two-time-constant circuit gets its second one."""
    new_root, mapping = _rebuilt(root)
    target = mapping.get(node_id)
    if target is None:
        raise ValueError("Cannot duplicate a node that is not in the tree.")
    located = _locate(new_root, node_id)
    if located is None:
        raise ValueError("Cannot duplicate the outermost connection.")
    parent, index = located
    parent.children.insert(index + 1, target.copy())
    return new_root


def delete(root: ConnectionNode, node_id: int) -> ConnectionNode:
    """Remove a node, then tidy up what it left behind."""
    new_root, _ = _rebuilt(root)
    located = _locate(new_root, node_id)
    if located is None:
        raise ValueError("Cannot delete a node that is not in the tree.")
    parent, index = located
    del parent.children[index]
    _collapse(new_root)
    return new_root


def _collapse(node: ConnectionNode) -> None:
    """Fold away what a deletion made meaningless -- a one-branch parallel, an
    empty connection, a series inside a series -- so the tree still round-trips
    through a CDC."""
    tidied: List[Node] = []
    for child in node.children:
        if isinstance(child, ElementNode):
            tidied.append(child)
            continue

        # Depth first, so a promoted grandchild is already tidy.
        _collapse(child)
        if not child.children:
            continue
        if len(child.children) == 1:
            promoted = child.children[0]
            if (
                isinstance(promoted, ConnectionNode)
                and promoted.kind == "series"
                and node.kind == "series"
            ):
                tidied.extend(promoted.children)
            else:
                tidied.append(promoted)
        elif child.kind == "series" and node.kind == "series":
            tidied.extend(child.children)
        else:
            tidied.append(child)
    node.children = tidied


def replace_element(root: ConnectionNode, node_id: int, symbol: str) -> ConnectionNode:
    """Change a component's type in place, keeping the values of any parameters
    the new type shares with the old one."""
    new_root, mapping = _rebuilt(root)
    target = mapping.get(node_id)
    if not isinstance(target, ElementNode):
        raise ValueError("Only elements have a type to replace.")
    if target.symbol == symbol:
        return new_root

    replacement = new_element(symbol)
    for key in replacement.values:
        if key in target.values:
            replacement.values[key] = target.values[key]
            replacement.lower[key] = target.lower[key]
            replacement.upper[key] = target.upper[key]
            replacement.fixed[key] = target.fixed[key]
    replacement.label = target.label
    replacement.node_id = target.node_id

    parent, index = _locate(new_root, node_id)  # type: ignore[misc]
    parent.children[index] = replacement
    return new_root


def set_element(
    root: ConnectionNode,
    node_id: int,
    *,
    values: Optional[Dict[str, float]] = None,
    lower: Optional[Dict[str, float]] = None,
    upper: Optional[Dict[str, float]] = None,
    fixed: Optional[Dict[str, bool]] = None,
    label: Optional[str] = None,
) -> ConnectionNode:
    """Update a component's parameters. Each argument is a partial update, so the
    editor can send one changed field at a time."""
    new_root, mapping = _rebuilt(root)
    target = mapping.get(node_id)
    if not isinstance(target, ElementNode):
        raise ValueError("Only elements have parameters to set.")

    for updates, current in (
        (values, target.values),
        (lower, target.lower),
        (upper, target.upper),
        (fixed, target.fixed),
    ):
        if updates:
            unknown = set(updates) - set(current)
            if unknown:
                raise ValueError(
                    f"{target.symbol} has no parameter(s) {', '.join(sorted(unknown))}."
                )
            current.update(updates)

    if label is not None:
        target.label = validate_label(label)
    return new_root


def validate_label(label: str) -> str:
    """What Element.set_label accepts, checked here so the editor can reject a
    bad label as it is typed rather than at serialization time."""
    label = label.strip()
    if not label:
        return ""
    if not label.isascii():
        raise ValueError("A label can only contain ASCII characters.")
    if label.isdigit():
        raise ValueError("A label cannot be only digits.")
    return label
