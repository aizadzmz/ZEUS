import numpy as np
import pytest
from pyimpspec import DataSet, parse_cdc

from core.circuit_model import (
    ConnectionNode,
    ElementNode,
    add_branch,
    add_in_series,
    append_element,
    delete,
    duplicate,
    element_count,
    empty_root,
    find,
    from_cdc,
    insert_after,
    insert_at,
    new_element,
    replace_element,
    set_element,
    to_cdc,
    validate_label,
    walk,
    wrap_in_parallel,
)
from core.ecm import CIRCUIT_PRESETS, canonical_cdc, run_ecm_fit
from core.io_utils import EISDataset

# Same synthetic spectrum as test_ecm.py and test_circuit_diagram.py: R0 plus two
# parallel RC pairs, so a circuit assembled purely by edit actions below has a
# known right answer to converge on.
R0_TRUE, R1_TRUE, R2_TRUE = 10.0, 50.0, 30.0
f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = R0_TRUE + R1_TRUE / (1 + 1j * w * 5e-3) + R2_TRUE / (1 + 1j * w * 0.3)
dataset = EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file="synthetic")


def structure(node):
    """The tree's shape and element types, ignoring node ids and values."""
    if isinstance(node, ElementNode):
        return node.symbol
    return (node.kind, tuple(structure(child) for child in node.children))


def first_element(root):
    return next(n for n in walk(root) if isinstance(n, ElementNode))


def elements(root):
    return [n for n in walk(root) if isinstance(n, ElementNode)]


# ------------------------------------------------------------- round-tripping


@pytest.mark.parametrize("name,cdc", CIRCUIT_PRESETS, ids=[n for n, _ in CIRCUIT_PRESETS])
def test_presets_round_trip(name, cdc):
    root = from_cdc(cdc)
    again = from_cdc(to_cdc(root))
    assert structure(again) == structure(root)
    # The CDC the tree serializes to must describe the same circuit as the one
    # it was parsed from -- this is what lets the canvas write back into the
    # CDC field without changing what gets fitted.
    assert canonical_cdc(to_cdc(root)) == canonical_cdc(cdc)


def test_extended_syntax_survives_a_round_trip():
    root = from_cdc("R{R=5F:s}(R{R=100/1/1e4}Q{Y=1e-6,n=0.9})")
    restored = from_cdc(to_cdc(root))

    series_r, parallel = restored.children
    assert series_r.values["R"] == pytest.approx(5.0)
    assert series_r.fixed["R"] is True
    assert series_r.label == "s"

    charge_transfer, cpe = parallel.children
    assert charge_transfer.values["R"] == pytest.approx(100.0)
    assert charge_transfer.lower["R"] == pytest.approx(1.0)
    assert charge_transfer.upper["R"] == pytest.approx(1e4)
    assert cpe.values["Y"] == pytest.approx(1e-6)
    assert cpe.values["n"] == pytest.approx(0.9)


def test_empty_root_serializes_to_an_empty_code():
    assert to_cdc(empty_root()) == ""


def test_structure_of_a_nested_preset():
    # R(C[RW]): a resistor in series with a parallel pair, one branch of which is
    # itself a series connection.
    assert structure(from_cdc("R(C[RW])")) == (
        "series",
        ("R", ("parallel", ("C", ("series", ("R", "W"))))),
    )


# --------------------------------------------------------------- edit actions


def test_append_and_insert():
    root = append_element(append_element(empty_root(), "R"), "C")
    assert structure(root) == ("series", ("R", "C"))

    root = insert_after(root, first_element(root).node_id, "L")
    assert structure(root) == ("series", ("R", "L", "C"))

    root = insert_at(root, root.node_id, 0, "Q")
    assert structure(root) == ("series", ("Q", "R", "L", "C"))
    assert parse_cdc(to_cdc(root)) is not None


def test_edits_do_not_mutate_the_original_tree():
    root = from_cdc("R(RC)")
    before = to_cdc(root)
    append_element(root, "L")
    delete(root, first_element(root).node_id)
    assert to_cdc(root) == before


def test_wrap_in_parallel():
    root = from_cdc("RR")
    root = wrap_in_parallel(root, elements(root)[1].node_id, "C")
    assert structure(root) == ("series", ("R", ("parallel", ("R", "C"))))
    assert canonical_cdc(to_cdc(root)) == canonical_cdc("R(RC)")


def test_add_in_series_inside_a_series():
    root = from_cdc("RC")
    root = add_in_series(root, first_element(root).node_id, "L")
    assert structure(root) == ("series", ("R", "L", "C"))


def test_add_in_series_inside_a_parallel_branch():
    root = from_cdc("(RC)")
    capacitor = next(n for n in elements(root) if n.symbol == "C")
    root = add_in_series(root, capacitor.node_id, "W")
    assert structure(root) == ("series", (("parallel", ("R", ("series", ("C", "W")))),))
    assert canonical_cdc(to_cdc(root)) == canonical_cdc("(R[CW])")


def test_add_branch_to_a_parallel():
    root = from_cdc("R(RC)")
    parallel = next(
        n for n in walk(root) if isinstance(n, ConnectionNode) and n.kind == "parallel"
    )
    root = add_branch(root, parallel.node_id, "L")
    assert structure(root) == ("series", ("R", ("parallel", ("R", "C", "L"))))


def test_add_branch_rejects_a_series():
    root = from_cdc("R(RC)")
    with pytest.raises(ValueError):
        add_branch(root, root.node_id, "C")


def test_duplicate_an_element_and_a_connection():
    root = from_cdc("R(RQ)")
    root = duplicate(root, first_element(root).node_id)
    assert structure(root) == ("series", ("R", "R", ("parallel", ("R", "Q"))))

    parallel = next(
        n for n in walk(root) if isinstance(n, ConnectionNode) and n.kind == "parallel"
    )
    root = duplicate(root, parallel.node_id)
    assert structure(root) == (
        "series",
        ("R", "R", ("parallel", ("R", "Q")), ("parallel", ("R", "Q"))),
    )
    # The copy is independent: editing it must not touch the original.
    copies = [n for n in walk(root) if isinstance(n, ConnectionNode) and n.kind == "parallel"]
    assert copies[0].node_id != copies[1].node_id
    assert copies[0].children[0].node_id != copies[1].children[0].node_id


def test_duplicated_values_are_independent():
    root = from_cdc("R{R=7}")
    root = duplicate(root, first_element(root).node_id)
    original, copy = elements(root)
    root = set_element(root, copy.node_id, values={"R": 99.0})
    assert find(root, original.node_id).values["R"] == pytest.approx(7.0)
    assert find(root, copy.node_id).values["R"] == pytest.approx(99.0)


def test_delete_collapses_a_single_branch_parallel():
    root = from_cdc("R(RC)")
    capacitor = next(n for n in elements(root) if n.symbol == "C")
    root = delete(root, capacitor.node_id)
    # The parallel is meaningless with one branch left, so it folds into the
    # series and the code becomes a plain RR.
    assert structure(root) == ("series", ("R", "R"))
    assert canonical_cdc(to_cdc(root)) == canonical_cdc("RR")


def test_delete_removes_an_emptied_connection():
    root = from_cdc("R(RC)")
    for element in elements(root):
        if element.symbol in ("R", "C") and element is not elements(root)[0]:
            root = delete(root, element.node_id)
    assert structure(root) == ("series", ("R",))


def test_delete_a_whole_connection():
    root = from_cdc("R(RQ)(RQ)")
    parallel = next(
        n for n in walk(root) if isinstance(n, ConnectionNode) and n.kind == "parallel"
    )
    root = delete(root, parallel.node_id)
    assert structure(root) == ("series", ("R", ("parallel", ("R", "Q"))))


def test_delete_flattens_a_nested_series():
    root = from_cdc("R(C[RW])")
    capacitor = next(n for n in elements(root) if n.symbol == "C")
    root = delete(root, capacitor.node_id)
    # (\[RW]) leaves one branch holding a series pair, which flattens all the way
    # up into the outermost series rather than nesting [ [R W] ].
    assert structure(root) == ("series", ("R", "R", "W"))
    assert parse_cdc(to_cdc(root)) is not None


def test_delete_everything_leaves_an_empty_root():
    root = from_cdc("R(RC)")
    for element in elements(root):
        root = delete(root, element.node_id)
    assert element_count(root) == 0
    assert to_cdc(root) == ""


def test_delete_rejects_an_unknown_node():
    with pytest.raises(ValueError):
        delete(from_cdc("RC"), -1)


def test_replace_element_keeps_shared_parameters():
    root = from_cdc("Q{Y=1.5e-3,n=0.8}")
    cpe = first_element(root)
    root = replace_element(root, cpe.node_id, "Ws")

    replaced = first_element(root)
    assert replaced.symbol == "Ws"
    # Ws shares Y and n with Q, so both carry over; its own B, which Q has no
    # counterpart for, comes in at the Ws default.
    assert replaced.values["Y"] == pytest.approx(1.5e-3)
    assert replaced.values["n"] == pytest.approx(0.8)
    assert replaced.values["B"] == pytest.approx(new_element("Ws").values["B"])
    # The node keeps its identity, so the canvas's selection survives the swap.
    assert replaced.node_id == cpe.node_id


def test_replace_element_with_a_type_sharing_nothing():
    root = from_cdc("R{R=42}")
    root = replace_element(root, first_element(root).node_id, "C")
    replaced = first_element(root)
    assert replaced.symbol == "C"
    assert replaced.values == new_element("C").values


def test_replace_element_with_the_same_type_is_a_no_op():
    root = from_cdc("R{R=42}")
    root = replace_element(root, first_element(root).node_id, "R")
    assert first_element(root).values["R"] == pytest.approx(42.0)


def test_set_element_updates_one_field_at_a_time():
    root = from_cdc("R")
    node = first_element(root)
    root = set_element(root, node.node_id, values={"R": 12.5})
    root = set_element(root, node.node_id, lower={"R": 1.0}, upper={"R": 100.0})
    root = set_element(root, node.node_id, fixed={"R": True}, label="ct")

    updated = first_element(root)
    assert updated.values["R"] == pytest.approx(12.5)
    assert updated.lower["R"] == pytest.approx(1.0)
    assert updated.upper["R"] == pytest.approx(100.0)
    assert updated.fixed["R"] is True
    assert updated.label == "ct"
    # And all of it survives the trip out through the CDC.
    restored = from_cdc(to_cdc(root)).children[0]
    assert restored.fixed["R"] is True
    assert restored.label == "ct"


def test_set_element_rejects_an_unknown_parameter():
    root = from_cdc("R")
    with pytest.raises(ValueError, match="no parameter"):
        set_element(root, first_element(root).node_id, values={"C": 1.0})


def test_validate_label():
    assert validate_label("  ct  ") == "ct"
    assert validate_label("") == ""
    with pytest.raises(ValueError):
        validate_label("123")
    with pytest.raises(ValueError):
        validate_label("Ω")


def test_new_element_rejects_an_unknown_symbol():
    with pytest.raises(ValueError, match="Unknown circuit element"):
        new_element("Nope")


# ------------------------------------------------------- end-to-end with a fit


def test_a_circuit_built_by_clicking_fits_the_spectrum():
    """Assemble R(RC)(RC) the way the canvas would -- add, wrap, duplicate -- and
    check the result is a circuit that fits, and recovers the true values."""
    root = append_element(empty_root(), "R")
    root = append_element(root, "R")
    root = wrap_in_parallel(root, elements(root)[1].node_id, "C")
    parallel = next(
        n for n in walk(root) if isinstance(n, ConnectionNode) and n.kind == "parallel"
    )
    root = duplicate(root, parallel.node_id)
    assert canonical_cdc(to_cdc(root)) == canonical_cdc("R(RC)(RC)")

    result = run_ecm_fit(dataset, to_cdc(root))
    fitted = sorted(
        p.get_value()
        for name, parameters in result.parameters.items()
        for symbol, p in parameters.items()
        if symbol == "R"
    )
    assert fitted == pytest.approx([R0_TRUE, R2_TRUE, R1_TRUE], rel=1e-3)
