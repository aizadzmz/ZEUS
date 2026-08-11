"""The DRT step's diffusion-tail subtraction: fitting a diffusion model to a
sweep and removing the fitted element's own impedance.

Distinct from the inductive filter it sits beside, which masks points. This
one rewrites them, so the tests below care about three things the masking
filter never has to: that the arithmetic recovers the rest of the circuit,
that the sweep it was asked about is left alone, and that a topology where the
subtraction would be meaningless is refused rather than approximated.
"""
import os

# Must precede any QApplication: the suite runs without a display in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from pyimpspec import DataSet, parse_cdc, simulate_spectrum

from core.filtering import (
    _series_diffusion_elements,
    describe_diffusion_fit,
    diffusion_impedance,
    diffusion_subtracted,
    impedance_subtracted,
)
from core.io_utils import EISDataset

FREQUENCIES = np.logspace(5, -2, 60)

# (label, the whole cell, the subtraction circuit, what should be left over).
CASES = [
    (
        "Ws",
        "R{R=10}(R{R=50}Q{Y=1e-4,n=0.85})Ws{Y=0.02,B=8}",
        "R(RQ)Ws",
        "R{R=10}(R{R=50}Q{Y=1e-4,n=0.85})",
    ),
    (
        "W",
        "R{R=5}(R{R=30}Q{Y=5e-4,n=0.9})W{Y=0.05}",
        "R(RQ)W",
        "R{R=5}(R{R=30}Q{Y=5e-4,n=0.9})",
    ),
    (
        "Wo",
        "R{R=8}(R{R=40}Q{Y=2e-4,n=0.88})Wo{Y=0.03,B=5}",
        "R(RQ)Wo",
        "R{R=8}(R{R=40}Q{Y=2e-4,n=0.88})",
    ),
    (
        "G",
        "R{R=12}(R{R=45}Q{Y=1e-4,n=0.9})G{Y=0.04,k=0.5}",
        "R(RQ)G",
        "R{R=12}(R{R=45}Q{Y=1e-4,n=0.9})",
    ),
]


def _sweep(cdc: str, frequencies=FREQUENCIES) -> EISDataset:
    return EISDataset(simulate_spectrum(parse_cdc(cdc), frequencies), 0, "sim")


# A cell whose two time constants (1e-4 s, 1e-2 s) both sit inside a
# 0.1 Hz - 100 kHz window, so both arcs are actually resolvable, plus a
# Warburg tail of comparable size to the arcs.
_TWO_ARC_CELL = (
    "R{R=0.005}(R{R=0.002}Q{Y=0.199,n=0.85})(R{R=0.003}Q{Y=5.28,n=0.9})W{Y=548}"
)
# The same circuit minus the tail, seeded ~1.5x off, as a DRT-built circuit is.
_KINETIC_SEED = "R{R=0.006}(R{R=0.003}Q{Y=0.28,n=0.85})(R{R=0.0045}Q{Y=7.4,n=0.9})"


def _two_arc_cell() -> EISDataset:
    frequencies = np.logspace(5, -1, 60)
    impedances = simulate_spectrum(parse_cdc(_TWO_ARC_CELL), frequencies).get_impedances()
    rng = np.random.default_rng(1)
    noisy = impedances * (
        1
        + rng.normal(0, 0.005, impedances.shape)
        + 1j * rng.normal(0, 0.005, impedances.shape)
    )
    return EISDataset(DataSet(frequencies, noisy), 0, "two-arc")


@pytest.mark.parametrize("name,cell,subtraction,remainder", CASES, ids=[c[0] for c in CASES])
def test_subtraction_leaves_the_rest_of_the_circuit(name, cell, subtraction, remainder):
    """The point of the feature: for a cell that really is the model plus a
    series diffusion element, removing the element leaves the model."""
    subtracted, result = diffusion_subtracted(_sweep(cell), subtraction)

    expected = parse_cdc(remainder).get_impedances(FREQUENCIES)
    residual = np.abs(subtracted.impedances - expected).max() / np.abs(expected).max()
    assert residual < 1e-3, f"{name}: {residual:.2e} of the spectrum left over"
    assert result.pseudo_chisqr < 1e-4


def test_seeding_is_what_makes_the_fit_land():
    """Regression on the reason _fit_diffusion_circuit exists at all.

    From pyimpspec's default initial values (Ws starts at Y=1, B=1) the fit
    converges on Y~1e5 -- a tail nobody measured, subtracted from data that
    still looks like data afterwards. Asserted against the true parameters
    rather than against chi-squared, because the bad fit reports a converged
    result too; only the values give it away.
    """
    _, cell, subtraction, _ = CASES[0]
    _, result = diffusion_subtracted(_sweep(cell), subtraction)

    element = _series_diffusion_elements(result.circuit)[0]
    values = element.get_values()
    assert values["Y"] == pytest.approx(0.02, rel=0.05)
    assert values["B"] == pytest.approx(8.0, rel=0.05)


def test_the_sweep_it_was_asked_about_is_untouched():
    """A per-analysis filter, like inductive_tail_removed: Validation and ECM
    read the same sweep and must not see the tail disappear from under them."""
    _, cell, subtraction, _ = CASES[0]
    dataset = _sweep(cell)
    before = dataset.impedances.copy()

    subtracted, _ = diffusion_subtracted(dataset, subtraction)

    assert np.array_equal(dataset.impedances, before)
    assert not np.allclose(subtracted.impedances, before)


def test_points_are_kept_not_dropped():
    """What separates this from the inductive filter beside it."""
    _, cell, subtraction, _ = CASES[0]
    dataset = _sweep(cell)
    subtracted, _ = diffusion_subtracted(dataset, subtraction)

    assert subtracted.num_points == dataset.num_points


def test_diffusion_inside_a_parallel_branch_is_refused():
    """R(Q[RWs]) is a real preset in CIRCUIT_PRESETS and fits happily, but the
    Ws does not contribute its own impedance to the total, so subtracting it
    would produce a plausible-looking spectrum that means nothing."""
    _, cell, _, _ = CASES[0]

    with pytest.raises(ValueError, match="in series"):
        diffusion_subtracted(_sweep(cell), "R(Q[RWs])")


def test_a_circuit_with_no_diffusion_element_is_refused():
    _, cell, _, _ = CASES[0]

    with pytest.raises(ValueError, match="no diffusion element"):
        diffusion_subtracted(_sweep(cell), "R(RQ)")


def test_masked_points_are_subtracted_from_too():
    """subtract_impedances writes to the DataSet's full impedance array, so the
    model has to be evaluated at every frequency even though the fit only saw
    the unmasked ones. Getting this wrong misaligns the whole spectrum."""
    _, cell, subtraction, remainder = CASES[0]
    dataset = _sweep(cell)
    dataset.data.set_mask({i: True for i in range(5)})

    subtracted, _ = diffusion_subtracted(dataset, subtraction)

    # The mask travels with the copy, so compare against everything.
    all_impedances = subtracted.data.get_impedances(masked=None)
    expected = parse_cdc(remainder).get_impedances(FREQUENCIES)
    assert len(all_impedances) == len(FREQUENCIES)
    assert sum(subtracted.data.get_mask().values()) == 5
    residual = np.abs(all_impedances - expected).max() / np.abs(expected).max()
    assert residual < 1e-2


def test_noise_degrades_gracefully():
    """1% scatter should cost accuracy, not correctness -- the subtraction is
    only useful if it survives data that was actually measured."""
    _, cell, subtraction, remainder = CASES[0]
    clean = simulate_spectrum(parse_cdc(cell), FREQUENCIES).get_impedances()
    rng = np.random.default_rng(0)
    noisy = clean * (
        1 + rng.normal(0, 0.01, clean.shape) + 1j * rng.normal(0, 0.01, clean.shape)
    )
    dataset = EISDataset(DataSet(FREQUENCIES, noisy), 0, "noisy")

    subtracted, _ = diffusion_subtracted(dataset, subtraction)

    expected = parse_cdc(remainder).get_impedances(FREQUENCIES)
    residual = np.abs(subtracted.impedances - expected).max() / np.abs(expected).max()
    assert residual < 0.1


def test_impedance_subtracted_replays_a_cached_fit():
    """The GUI caches the fit and re-applies it on every redraw; that path has
    to land on the same spectrum as the one-shot call."""
    _, cell, subtraction, _ = CASES[0]
    dataset = _sweep(cell)

    impedances, _ = diffusion_impedance(dataset, subtraction)
    replayed = impedance_subtracted(dataset, impedances)
    one_shot, _ = diffusion_subtracted(dataset, subtraction)

    assert np.allclose(replayed.impedances, one_shot.impedances)


def test_describe_diffusion_fit_names_the_element_and_its_quality():
    _, cell, subtraction, _ = CASES[0]
    _, result = diffusion_subtracted(_sweep(cell), subtraction)

    text = describe_diffusion_fit(result)
    assert "Ws" in text
    assert "Y=" in text and "B=" in text
    assert "χ²" in text


def test_the_description_is_two_lines_with_the_quality_second():
    """The panel row reserves exactly two lines, and the fit quality is the
    one that has to survive -- it is what says whether to trust the result."""
    _, cell, subtraction, _ = CASES[0]
    _, result = diffusion_subtracted(_sweep(cell), subtraction)

    lines = describe_diffusion_fit(result).split("\n")
    assert len(lines) == 2
    assert "χ²" not in lines[0]
    assert lines[1].startswith("pseudo χ²")


def test_fixed_parameters_are_left_out_of_the_description():
    """n is never fitted, so it carried no information while taking up room
    the fitted values needed."""
    _, cell, subtraction, _ = CASES[0]
    _, result = diffusion_subtracted(_sweep(cell), subtraction)

    element = _series_diffusion_elements(result.circuit)[0]
    assert element.is_fixed("n"), "fixture assumes n is the fixed one"

    first_line = describe_diffusion_fit(result).split("\n")[0]
    assert "n=" not in first_line
    assert "Y=" in first_line and "B=" in first_line


# --- measured data, which is where the synthetic cases were not enough ------
#
# Every test above passes a spectrum that is exactly the subtraction circuit
# plus noise. A real cell is not: this one spends its first 12 points above the
# real axis on cabling inductance, and both bugs the synthetic cases missed
# lived there.

MEASURED = "test/data/example_EIS_data.txt"


@pytest.fixture
def measured():
    from core.io_utils import parse_eis_file

    return parse_eis_file(MEASURED)[0]


def test_ohmic_resistance_is_the_axis_crossing_not_the_first_point(measured):
    """The bug that made the first version useless on real data.

    Re(Z) at the highest frequency is the inductive tail's, not the cell's,
    and seeding R_s from it drove the polarisation resistance negative and the
    Warburg to Y=1.6e13 -- a fit so far off that the impedance it subtracted
    was indistinguishable from zero.
    """
    from core.filtering import _ohmic_resistance

    frequencies, impedances = measured.frequencies, measured.impedances
    order = np.argsort(frequencies)[::-1]
    real, imag = impedances.real[order], impedances.imag[order]
    ohmic = _ohmic_resistance(frequencies, impedances)

    assert (imag > 0).any(), "fixture is meant to have an inductive tail"

    # Between the two points that straddle the axis. Asserted as a bracket
    # rather than as "below the first point": which side of the crossing the
    # inductive branch runs is the cell's business -- this fixture sweeps left
    # of it, the demo cells sweep right -- and the estimator is about landing
    # on the crossing, not about a direction.
    i = np.flatnonzero(np.sign(imag[:-1]) * np.sign(imag[1:]) < 0)[0]
    assert min(real[i], real[i + 1]) <= ohmic <= max(real[i], real[i + 1])
    # And materially away from the point the old estimator returned.
    assert abs(ohmic - real[0]) > 0.1 * abs(ohmic)


def test_ohmic_resistance_falls_back_when_nothing_crosses():
    """A sweep entirely below the axis has no crossing to interpolate, and its
    highest-frequency point is the right answer."""
    from core.filtering import _ohmic_resistance

    _, cell, _, _ = CASES[0]
    sweep = _sweep(cell)
    assert (sweep.impedances.imag <= 0).all()

    ohmic = _ohmic_resistance(sweep.frequencies, sweep.impedances)
    assert ohmic == pytest.approx(sweep.impedances[np.argmax(sweep.frequencies)].real)


def test_measured_sweep_fits_sanely(measured):
    """The end-to-end regression. Before the fix this fitted at chi-squared 27
    and subtracted nothing; anything in that range means the seeding has come
    undone again."""
    _, result = diffusion_subtracted(measured, "R(RQ)W")

    assert result.pseudo_chisqr < 0.1
    element = _series_diffusion_elements(result.circuit)[0]
    # The runaway signature: Y climbing until the element vanishes.
    assert element.get_values()["Y"] < 1e6


def test_the_subtraction_actually_moves_a_measured_spectrum(measured):
    """The failure was silent -- a converged fit whose impedance was so close
    to zero that the 'subtracted' spectrum was the raw one."""
    from core.filtering import inductive_tail_removed

    before = inductive_tail_removed(measured)
    after, _ = diffusion_subtracted(before, "R(RQ)W")

    # The tail is what should move, and it should move towards the axis.
    lowest = np.argmin(before.frequencies)
    assert abs(after.impedances[lowest]) < abs(before.impedances[lowest])
    assert not np.allclose(after.impedances, before.impedances)


def test_inductive_points_are_held_out_of_the_fit(measured):
    """None of the presets has an inductor, so the inductive points cannot be
    represented at any parameter value -- all they can do is drag the fit. They
    are held out whether or not the DRT step's own filter is on, which is what
    this asserts: the two calls must agree."""
    from core.filtering import inductive_tail_removed

    with_tail, _ = diffusion_subtracted(measured, "R(RQ)W")
    without_tail, _ = diffusion_subtracted(inductive_tail_removed(measured), "R(RQ)W")

    unmasked = without_tail.data.get_mask()
    kept = [i for i in range(with_tail.num_points) if not unmasked.get(i, False)]
    assert np.allclose(
        with_tail.data.get_impedances(masked=None)[kept],
        without_tail.data.get_impedances(masked=None)[kept],
    )


def test_a_sweep_with_too_few_fittable_points_is_refused(measured):
    """Rather than fitting five free parameters to whatever survived."""
    from core.filtering import detached_copy

    stripped = detached_copy(measured)
    stripped.data.set_mask({i: True for i in range(2, stripped.num_points)})

    with pytest.raises(ValueError, match="Nothing to subtract"):
        diffusion_subtracted(stripped, "R(RQ)W")


# --- putting the tail back into the ECM model -------------------------------
#
# A DRT computed on a subtracted sweep yields a circuit with no diffusion
# element, and the ECM step fits the sweep as measured, tail included. These
# pin down why that gap has to be closed in the model rather than in the data.


def test_a_diffusion_free_circuit_fitted_to_a_tailed_sweep_is_badly_wrong():
    """The failure that pseudo chi-squared does not show.

    Two resolvable arcs plus a Warburg, fitted with a circuit that has no
    Warburg: the R-CPE pairs absorb the tail because nothing else can, and the
    resistances come out by tens of percent while chi-squared stays at a value
    that reads as a perfectly good fit.
    """
    from pyimpspec import Resistor

    from core.ecm import run_ecm_fit

    result = run_ecm_fit(_two_arc_cell(), _KINETIC_SEED)

    resistances = [
        e.get_values()["R"] for e in result.circuit.get_elements() if isinstance(e, Resistor)
    ]
    assert result.pseudo_chisqr < 0.1, "chi-squared looks fine -- that is the point"
    assert abs(resistances[1] - 0.002) / 0.002 > 0.3
    assert abs(resistances[2] - 0.003) / 0.003 > 0.3


def test_the_element_in_the_model_recovers_the_kinetic_resistances():
    """The fix, and the reason it is the model that gains the element rather
    than the data losing the tail: fitting the same kinetic circuit to the
    *subtracted* sweep still leaves both resistances tens of percent out,
    because the subtraction's own residual lands in them instead."""
    from pyimpspec import Resistor

    from core.ecm import run_ecm_fit

    def resistances(result):
        return [
            e.get_values()["R"]
            for e in result.circuit.get_elements()
            if isinstance(e, Resistor)
        ]

    cell = _two_arc_cell()
    subtracted, _ = diffusion_subtracted(cell, "R(RQ)W")

    on_subtracted = resistances(run_ecm_fit(subtracted, _KINETIC_SEED))
    with_element = resistances(run_ecm_fit(cell, _KINETIC_SEED + "W{Y=548}"))

    # Subtracting the data helps, but not enough to trust a parameter.
    assert abs(on_subtracted[1] - 0.002) / 0.002 > 0.2
    # Modelling the tail recovers both to within a few percent.
    assert abs(with_element[1] - 0.002) / 0.002 < 0.05
    assert abs(with_element[2] - 0.003) / 0.003 < 0.05


def test_diffusion_element_cdc_is_an_appendable_fragment():
    """It is concatenated onto a circuit built from DRT peaks, so it has to
    parse in that position and carry the values it was fitted with."""
    from pyimpspec import parse_cdc as parse

    from core.filtering import diffusion_element_cdc

    _, cell, subtraction, _ = CASES[0]
    _, result = diffusion_subtracted(_sweep(cell), subtraction)

    fragment = diffusion_element_cdc(result)
    combined = parse("R(RQ)" + fragment)

    element = _series_diffusion_elements(combined)[0]
    assert element.get_symbol() == "Ws"
    assert element.get_values()["Y"] == pytest.approx(0.02, rel=0.05)


# --- the DRT step's wiring --------------------------------------------------


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    """A MainWindow holding three copies of the Ws cell, all selected."""
    from gui.main_window import MainWindow

    _, cell, _, _ = CASES[0]
    window = MainWindow()
    window._datasets = [
        EISDataset(simulate_spectrum(parse_cdc(cell), FREQUENCIES), i, "sim", 0)
        for i in range(3)
    ]
    window._selection.set_datasets(window._datasets)
    window._selection.set_all(True)
    # Index 1 is R(RQ)Ws, the circuit these sweeps were built from.
    window.drt_step.diffusion_cdc_combo.setCurrentIndex(1)
    return window


def test_filter_off_leaves_the_drt_inputs_alone(window):
    selected = window._selected_datasets()
    assert window._drt_inputs(selected) == selected


def test_filter_on_subtracts_from_every_drt_input(window):
    _, _, _, remainder = CASES[0]
    selected = window._selected_datasets()
    window.drt_step.subtract_diffusion_check.setChecked(True)

    expected = parse_cdc(remainder).get_impedances(FREQUENCIES)
    for sweep in window._drt_inputs(selected):
        residual = np.abs(sweep.impedances - expected).max() / np.abs(expected).max()
        assert residual < 1e-3


def test_the_step_leaves_the_sweeps_the_other_steps_read_alone(window):
    """The whole reason this is scoped to the DRT. ECM and Validation read
    window's own datasets, which must still carry their tail."""
    selected = window._selected_datasets()
    before = [ds.impedances.copy() for ds in selected]
    window.drt_step.subtract_diffusion_check.setChecked(True)

    window._drt_inputs(selected, detached=True)

    assert all(np.array_equal(ds.impedances, was) for ds, was in zip(selected, before))


def test_fits_are_cached_across_redraws(window):
    """_drt_inputs runs on every redraw, and each miss is a CNLS fit per
    sweep -- without the cache, paging through sweeps refits the lot."""
    selected = window._selected_datasets()
    window.drt_step.subtract_diffusion_check.setChecked(True)

    window._drt_inputs(selected)
    assert len(window._diffusion_fits) == len(selected)

    window._drt_inputs(selected)
    assert len(window._diffusion_fits) == len(selected)


def test_changing_the_mask_refits(window):
    """A stale tail is worse than a slow one: the fit the cache holds was made
    against a point set the sweep no longer has."""
    selected = window._selected_datasets()
    window.drt_step.subtract_diffusion_check.setChecked(True)
    window._drt_inputs(selected)
    before = len(window._diffusion_fits)

    selected[0].data.set_mask({0: True, 1: True})
    window._drt_inputs(selected)

    assert len(window._diffusion_fits) == before + 1


def test_a_failed_fit_passes_the_sweep_through(window):
    """A sweep that cannot be fitted still reaches the DRT, unsubtracted --
    dropping it would leave a hole on the plot with nothing to explain it."""
    selected = window._selected_datasets()
    window.drt_step.subtract_diffusion_check.setChecked(True)
    # Stands in for any fit failure, which is what the GUI stores this way.
    key = (selected[0].key, "R(RQ)Ws", frozenset())
    window._diffusion_fits[key] = (None, "singular matrix")

    inputs = window._drt_inputs(selected)

    assert np.allclose(inputs[0].impedances, selected[0].impedances)
    assert inputs[0] is not selected[0]
    window._update_diffusion_label([selected[0]])
    assert "singular matrix" in window.drt_step.diffusion_status_label.text()


def test_the_readout_reports_the_fit_behind_the_subtraction(window):
    selected = window._selected_datasets()
    window.drt_step.subtract_diffusion_check.setChecked(True)
    window._drt_inputs(selected)

    window._update_diffusion_label([selected[0]])
    assert "Ws" in window.drt_step.diffusion_status_label.text()

    # Several on screen have several fits, so there is nothing single to show.
    window._update_diffusion_label(selected)
    assert "each fitted separately" in window.drt_step.diffusion_status_label.text()


def test_the_readout_never_needs_more_room_than_it_has(window, app):
    """The reported fault: the pseudo chi-squared wrapped onto a line the row
    had no height for and was clipped. Checked for every model, because it is
    the two-parameter ones whose first line is long enough to trigger it, and
    for the failure message, which is longer still."""
    from PySide6.QtCore import QRect, Qt

    window.show()
    window.step_stack.setCurrentIndex(2)
    window.drt_step.subtract_diffusion_check.setChecked(True)
    label = window.drt_step.diffusion_status_label
    app.processEvents()

    def assert_fits(context):
        app.processEvents()
        metrics = label.fontMetrics()
        lines = label.text().split("\n")
        assert len(lines) <= 2, f"{context}: {len(lines)} lines"
        assert label.height() >= metrics.lineSpacing() * len(lines), context
        for line in lines:
            assert metrics.horizontalAdvance(line) <= label.width(), f"{context}: {line!r}"

    selected = window._selected_datasets()[:1]
    for index in range(window.drt_step.diffusion_cdc_combo.count()):
        window.drt_step.diffusion_cdc_combo.setCurrentIndex(index)
        window._drt_inputs(selected)
        window._update_diffusion_label(selected)
        assert_fits(window.drt_step.diffusion_cdc_combo.currentData())

    window._diffusion_shown[selected[0].key] = (
        "Fitting failed: the covariance matrix is singular at iteration 431 of 500"
    )
    window._update_diffusion_label(selected)
    assert_fits("failure")
    # Elided in the label, whole in the tooltip.
    assert label.toolTip().endswith("431 of 500")
    assert label.full_text.endswith("431 of 500")


def test_the_readout_puts_the_quality_on_the_second_line(window, app):
    window.show()
    window.step_stack.setCurrentIndex(2)
    window.drt_step.subtract_diffusion_check.setChecked(True)
    selected = window._selected_datasets()[:1]
    window._drt_inputs(selected)
    window._update_diffusion_label(selected)
    app.processEvents()

    lines = window.drt_step.diffusion_status_label.full_text.split("\n")
    assert len(lines) == 2
    assert lines[1].startswith("pseudo χ²")


def test_the_model_row_greys_out_with_the_filter(window):
    assert not window.drt_step.diffusion_cdc_combo.isEnabled()
    window.drt_step.subtract_diffusion_check.setChecked(True)
    assert window.drt_step.diffusion_cdc_combo.isEnabled()


def _run_drt_through_the_window(window):
    """Drive a DRT the way the Run button does, so the per-sweep diffusion fit
    is recorded the way _on_drt_worker_result records it.

    Goes through _start_drt_run rather than repeating its steps here. That is
    where the batch's diffusion fits are frozen, and a helper that rebuilt the
    sequence by hand would go on passing if the freeze were dropped -- which
    is the failure test_diffusion_record covers.
    """
    from core.drt import analyze_drt_peaks, run_drt
    from gui.workers import DRTWorker

    selected = window._selected_datasets()
    # The thread would only move run_drt off the UI thread. Results are
    # delivered below instead: synchronously, and in a known order.
    start = DRTWorker.start
    DRTWorker.start = lambda self: None
    try:
        window._start_drt_run(selected, {"mode": "complex"}, "DRT")
        inputs = list(window._drt_worker._datasets)
    finally:
        DRTWorker.start = start

    for source, prepared in zip(selected, inputs):
        window._on_drt_worker_result(source.key, run_drt(prepared, lambda_value=1e-3))
        window._drt_peaks[source.key] = analyze_drt_peaks(window._drt_results[source.key])
    return selected[0]


def test_a_built_circuit_regains_the_tail_that_was_subtracted(window):
    """Without this the ECM step fits a diffusion-free circuit to a sweep that
    still has its tail, and the R-CPE resistances absorb it."""
    window.drt_step.subtract_diffusion_check.setChecked(True)
    source = _run_drt_through_the_window(window)

    assert window._drt_params[source.key]["diffusion_element"]

    window._build_circuit_from_drt()
    cdc = window.ecm_step.cdc_edit.text()

    assert _series_diffusion_elements(parse_cdc(cdc)), cdc
    assert "fitted diffusion element" in window.statusBar().currentMessage()


def test_a_built_circuit_gains_nothing_when_nothing_was_subtracted(window):
    """The element is put back because it was taken out -- with the filter off
    there is no tail missing from the DRT, and appending one would be inventing
    a process the peaks never saw."""
    window.drt_step.subtract_diffusion_check.setChecked(False)
    source = _run_drt_through_the_window(window)

    assert window._drt_params[source.key]["diffusion_element"] is None

    window._build_circuit_from_drt()
    assert not _series_diffusion_elements(parse_cdc(window.ecm_step.cdc_edit.text()))


def test_the_run_record_carries_the_circuit_not_just_the_flag(window):
    """A subtraction cannot be reconstructed from the saved DRT result: the
    data it was computed from no longer holds the tail that was removed."""
    window.drt_step.subtract_diffusion_check.setChecked(True)
    record = window._drt_record({"mode": "complex"})

    assert record["subtract_diffusion"] is True
    assert record["diffusion_cdc"] == "R(RQ)Ws"

    window.drt_step.subtract_diffusion_check.setChecked(False)
    assert window._drt_record({"mode": "complex"})["diffusion_cdc"] is None
