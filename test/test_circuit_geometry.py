import pytest

from core.circuit_diagram import build_editor_drawing, build_preview_diagram
from core.circuit_model import ConnectionNode, ElementNode, from_cdc, walk
from core.ecm import CIRCUIT_PRESETS


def regions(drawing, kind):
    return [r for r in drawing.regions if r.kind == kind]


def overlaps(a, b):
    ax, ay, aw, ah = a.rect
    bx, by, bw, bh = b.rect
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def inside_viewbox(drawing, region):
    vx, vy, vw, vh = drawing.viewbox
    x, y = region.center
    return vx <= x <= vx + vw and vy <= y <= vy + vh


@pytest.mark.parametrize("name,cdc", CIRCUIT_PRESETS, ids=[n for n, _ in CIRCUIT_PRESETS])
def test_every_node_gets_exactly_one_region(name, cdc):
    tree = from_cdc(cdc)
    drawing = build_editor_drawing(tree)

    expected = {
        node.node_id
        for node in walk(tree)
        if isinstance(node, ElementNode) or node.kind == "parallel"
    }
    tagged = {r.node_id for r in drawing.regions if r.kind in ("element", "connection")}
    assert tagged == expected

    counts = [r.node_id for r in drawing.regions if r.kind in ("element", "connection")]
    assert len(counts) == len(set(counts))


@pytest.mark.parametrize("name,cdc", CIRCUIT_PRESETS, ids=[n for n, _ in CIRCUIT_PRESETS])
def test_regions_land_on_the_drawing(name, cdc):
    drawing = build_editor_drawing(from_cdc(cdc))
    assert drawing.regions
    for region in drawing.regions:
        assert inside_viewbox(drawing, region), f"{region.kind} fell outside the viewBox"
        assert region.rect[2] > 0 and region.rect[3] > 0


@pytest.mark.parametrize("cdc", ["R(RQ)(RQ)", "RCL", "R(C[RW])"])
def test_element_regions_are_disjoint(cdc):
    drawing = build_editor_drawing(from_cdc(cdc))
    elements = regions(drawing, "element")
    for i, a in enumerate(elements):
        for b in elements[i + 1 :]:
            assert not overlaps(a, b), f"{a.rect} overlaps {b.rect}"


@pytest.mark.parametrize("cdc", ["R(RQ)(RQ)", "RCL", "R(C[RW])"])
def test_gaps_do_not_overlap_elements(cdc):
    drawing = build_editor_drawing(from_cdc(cdc))
    for gap in regions(drawing, "gap"):
        for element in regions(drawing, "element"):
            assert not overlaps(gap, element)


def test_a_series_gets_a_gap_on_each_side_of_every_item():
    tree = from_cdc("RCL")
    drawing = build_editor_drawing(tree)
    gaps = regions(drawing, "gap")
    assert [g.index for g in gaps] == [0, 1, 2, 3]
    assert {g.node_id for g in gaps} == {tree.node_id}


def test_nested_series_gets_its_own_gaps():
    tree = from_cdc("R(C[RW])")
    nested = next(
        n
        for n in walk(tree)
        if isinstance(n, ConnectionNode) and n.kind == "series" and n is not tree
    )
    gaps = regions(build_editor_drawing(tree), "gap")
    assert sorted(g.index for g in gaps if g.node_id == nested.node_id) == [0, 1, 2]


def test_element_regions_run_left_to_right_in_series_order():
    tree = from_cdc("RCL")
    drawing = build_editor_drawing(tree)
    by_node = {r.node_id: r for r in regions(drawing, "element")}
    lefts = [by_node[child.node_id].rect[0] for child in tree.children]
    assert lefts == sorted(lefts)


def test_parallel_branches_are_stacked_vertically():
    tree = from_cdc("(RC)")
    drawing = build_editor_drawing(tree)
    parallel = tree.children[0]
    tops = [
        next(r for r in drawing.regions if r.node_id == branch.node_id).rect[1]
        for branch in parallel.children
    ]
    assert tops[0] != tops[1]


def test_no_regions_without_a_tree():
    # The plain preview path is untouched and still returns bytes.
    assert build_preview_diagram("R(RC)").startswith(b"<svg")


def test_editor_drawing_matches_the_preview_picture():
    cdc = "R(RQ)"
    assert build_editor_drawing(from_cdc(cdc)).svg == build_preview_diagram(cdc)


def test_fitted_values_can_annotate_the_editor_drawing():
    from test.test_circuit_diagram import dataset

    from core.ecm import run_ecm_fit

    result = run_ecm_fit(dataset, "R(RC)(RC)")
    tree = from_cdc("R(RC)(RC)")
    drawing = build_editor_drawing(tree, parameters=result.parameters)
    assert b"\xc2\xb1" in drawing.svg  # the +/- of an error bar
    assert len(regions(drawing, "element")) == 5
