def mask_inductive_points(dataset) -> None: #remove inductive tail
    """Mask points with a positive imaginary impedance (inductive artifacts),
    in place."""
    Z = dataset.data.get_impedances(masked=None)  # all points, incl. masked
    dataset.data.set_mask({i: bool(z.imag > 0) for i, z in enumerate(Z)})


def clear_mask(dataset) -> None: #re-add the inductive points
    """Unmask all points of an EISDataset (in place)."""
    dataset.data.set_mask({})


def detached_copy(dataset):
    """A copy of an EISDataset that no longer shares its point data, so masking
    one leaves the other alone.

    The copy keeps the original's index and file, and so its key: results
    computed from it still file under the sweep it came from."""
    from copy import deepcopy

    from core.io_utils import EISDataset

    return EISDataset(
        deepcopy(dataset.data), dataset.index, dataset.source_file, dataset.file_id
    )


def inductive_tail_removed(dataset):
    """A detached copy of an EISDataset with the inductive tail (Im(Z) > 0)
    masked on top of whatever is already masked.

    A copy rather than an in-place mask, unlike mask_inductive_points: this is
    a per-analysis filter, and the shared mask is what validation results are
    checked against, so moving it would mark them stale."""
    filtered = detached_copy(dataset)
    Z = filtered.data.get_impedances(masked=None)  # all points, incl. masked
    inductive = {i: True for i, z in enumerate(Z) if z.imag > 0}
    # Guarded: set_mask({}) means "unmask everything", so an empty dict here
    # would hand back a copy with the original's masking undone.
    if inductive:
        filtered.data.set_mask(inductive)
    return filtered


def mask_points(dataset, indices) -> None:
    """Force the given point indices masked, in place, leaving every other
    point's state alone. Used to replay an iterative prune's removals, which
    _refresh cannot re-derive without re-running the validation."""
    if indices:
        dataset.data.set_mask({int(i): True for i in indices})


def apply_manual_overrides(dataset, masked, kept) -> None:
    """Force the given point indices masked / unmasked, in place, on top of
    whatever the automatic filters have already decided."""
    overrides = {int(i): True for i in masked}
    overrides.update({int(i): False for i in kept})
    if overrides:
        dataset.data.set_mask(overrides)

# --- diffusion tail ---------------------------------------------------------
#
# Unlike the filters above, which mask points, this one rewrites them: a
# diffusion model is fitted to the sweep and the fitted element's impedance is
# subtracted from every point. The point count is unchanged -- the tail is
# flattened, not dropped -- which is the whole reason to prefer it to masking
# when the low-frequency points still carry a time constant worth resolving.


def _top_series_elements(circuit):
    """The elements sitting directly in a circuit's outermost series
    connection, parallel branches excluded."""
    from pyimpspec import Series

    return [
        element
        for connection in circuit.get_connections(recursive=False)
        if isinstance(connection, Series)
        for element in connection.get_elements(recursive=False)
    ]


def _series_diffusion_elements(circuit):
    """The diffusion elements sitting directly in a circuit's outermost series
    connection.

    Only those can be subtracted. Z_measured - Z_diffusion leaves the rest of
    the circuit behind only when the two are in series; for a diffusion element
    inside a parallel branch -- CIRCUIT_PRESETS' "R(Q[RWs])", say -- the branch
    contributes something other than that element's own impedance to the total,
    so removing it is not a subtraction and the result means nothing. Hence
    this guard rather than a search over every element in the circuit.
    """
    from core.ecm import DIFFUSION_SYMBOLS

    return [
        element
        for element in _top_series_elements(circuit)
        if element.get_symbol() in DIFFUSION_SYMBOLS
    ]


# How the sweep's polarisation resistance is split between the kinetic arc and
# the diffusion tail for the starting guesses. Which way it actually splits is
# the one thing that cannot be read off the spectrum, so the fit is started
# from a spread and the best result kept; see _fit_diffusion_circuit.
_SEED_DIFFUSION_FRACTIONS = (0.25, 0.5, 0.75)

# Below this many points left to fit, the sweep is refused rather than fitted
# on whatever survives -- the presets carry four or five free parameters.
_MIN_FIT_POINTS = 8


def _seed_diffusion_element(element, r_diffusion: float, tau: float) -> None:
    """Set one diffusion element's initial values from a resistance and a time
    constant, in place.

    Each expression below inverts that element's own impedance at the limit
    where the two quantities are readable, with n at its (fixed) default:

    - W  = 1/(Y(jw)^n)      -> |Z| = R_d at w = 1/tau
    - Ws = tanh((Bjw)^n)/(Yjw)^n  -> Z(0) = (B/Y)^n, knee at w = 1/B
    - Wo = coth(...)/...    -> same knee; no DC limit, so Ws's relation is
                               reused for the magnitude, which is the right
                               order either way
    - G  = 1/(Y(k+jw)^n)    -> Z(0) = 1/(Yk^n), knee at w = k

    Elements with richer parameterisations (de Levie, the transmission lines)
    are left on pyimpspec's defaults rather than seeded badly.
    """
    symbol = element.get_symbol()
    n = element.get_values().get("n", 0.5)

    if symbol == "W":
        element.set_values(Y=tau**n / r_diffusion)
    elif symbol in ("Ws", "Wo"):
        element.set_values(B=tau, Y=tau / r_diffusion ** (1.0 / n))
    elif symbol == "G":
        k = 1.0 / tau
        element.set_values(k=k, Y=1.0 / (r_diffusion * k**n))


def _ohmic_resistance(frequencies, impedances) -> float:
    """The sweep's ohmic resistance, as core.ecm.ohmic_resistance measures it.

    This wrapper is all that is left of a second implementation that lived
    here: seeding a diffusion fit needed the axis crossing, while
    core.ecm.series_resistance still read the first point, and the two
    disagreed by a factor of four on a cell with cabling inductance. The
    crossing was the right one, so it moved to core.ecm and the callers that
    were wrong now share it.
    """
    from core.ecm import ohmic_resistance

    return ohmic_resistance(frequencies, impedances)


def _seeded_cdcs(dataset, cdc):
    """The subtraction circuit, respelled with initial values taken from the
    sweep, once per starting guess.

    Same trick as core.ecm.seed_cdc -- values are carried in the CDC itself --
    but sourced from the data rather than from a previous fit, because there
    is no previous fit the first time round.
    """
    import numpy as np
    from pyimpspec import Resistor, parse_cdc

    impedances = dataset.impedances
    frequencies = dataset.frequencies
    r_series = _ohmic_resistance(frequencies, impedances)
    r_low = float(impedances[np.argmin(frequencies)].real)
    # Floored, not trusted: a sweep that never turns over, or one already
    # subtracted from, can put the low-frequency intercept below the ohmic
    # one and leave every seed negative.
    r_polarisation = max(r_low - r_series, abs(r_series)) or 1.0
    # The tail is what the lowest measured frequency is still resolving, so
    # its time constant is that frequency's, to an order of magnitude.
    tau = 1.0 / (2.0 * np.pi * float(np.min(frequencies)))

    for fraction in _SEED_DIFFUSION_FRACTIONS:
        circuit = parse_cdc(cdc)
        series_elements = _top_series_elements(circuit)
        diffusion = _series_diffusion_elements(circuit)

        # The first series resistor is the ohmic one; any others, and the
        # resistors inside parallel branches, share what is left.
        ohmic = next((e for e in series_elements if isinstance(e, Resistor)), None)
        if ohmic is not None:
            ohmic.set_values(R=r_series)
        arc_resistors = [
            e
            for e in circuit.get_elements()
            if isinstance(e, Resistor) and e is not ohmic
        ]
        if arc_resistors:
            share = r_polarisation * (1.0 - fraction) / len(arc_resistors)
            for element in arc_resistors:
                element.set_values(R=share)

        r_diffusion = r_polarisation * fraction / max(len(diffusion), 1)
        for element in diffusion:
            _seed_diffusion_element(element, r_diffusion, tau)

        yield circuit.to_string(12)


def _fit_diffusion_circuit(dataset, cdc, **fit_kwargs):
    """Fit a subtraction circuit to a sweep, from several data-derived starting
    points, and return the best of them.

    Not a plain run_ecm_fit, because pyimpspec's default initial values for the
    diffusion elements (Ws starts at Y=1, B=1) are nowhere near a real cell's.
    From those, every fitting method here walks off to Y~1e6 and reports a
    converged fit that subtracts a tail nobody measured -- the worst failure
    this feature can have, since the output still looks like data. Seeding from
    the sweep's own intercepts recovers the true parameters instead.

    The inductive points are held out of the fit whether or not the DRT step's
    own filter is on, because none of the subtraction circuits contains an
    inductor: they are points the model cannot represent at any parameter
    value, so all they can do is drag it. On the demo cells that is 46 of 121
    points, enough on its own to send the Warburg to Y=1.6e13. They are still
    subtracted from -- held out of the fit is not dropped from the sweep.
    """
    from core.ecm import run_ecm_fit

    target = inductive_tail_removed(dataset)
    if target.num_points < _MIN_FIT_POINTS:
        raise ValueError(
            f"Only {target.num_points} non-inductive point(s) to fit "
            f"'{cdc}' to. Nothing to subtract a diffusion tail from."
        )

    best = None
    failures = []
    for candidate in _seeded_cdcs(target, cdc):
        try:
            result = run_ecm_fit(target, candidate, **fit_kwargs)
        except Exception as exc:
            failures.append(str(exc))
            continue
        if best is None or result.pseudo_chisqr < best.pseudo_chisqr:
            best = result

    if best is None:
        raise ValueError(
            f"Could not fit '{cdc}' to this sweep: {failures[0] if failures else 'no result'}"
        )
    return best


def diffusion_impedance(dataset, cdc, **fit_kwargs):
    """Fit `cdc` to a sweep and evaluate the fitted diffusion element's own
    impedance at each of the sweep's frequencies.

    Returns (impedances, fit_result). The fit sees the unmasked points only,
    as any other fit would, but the impedances returned cover *every* point:
    DataSet.subtract_impedances writes to the full impedance array, so handing
    it a shorter one would either raise or silently misalign.
    """
    import numpy as np
    from pyimpspec import parse_cdc

    # Checked before the fit, which is the expensive half: a circuit whose
    # diffusion element is not in series can never be used here, however well
    # it happens to fit.
    if not _series_diffusion_elements(parse_cdc(cdc)):
        raise ValueError(
            f"'{cdc}' has no diffusion element in series with the rest of the "
            f"circuit. Subtracting a tail needs one: a diffusion element "
            f"inside a parallel branch does not contribute its own impedance "
            f"to the total, so removing it is not a subtraction."
        )

    result = _fit_diffusion_circuit(dataset, cdc, **fit_kwargs)
    frequencies = dataset.data.get_frequencies(masked=None)
    impedances = np.zeros(len(frequencies), dtype=np.complex128)
    # Summed, because several series diffusion elements are still in series
    # with each other.
    for element in _series_diffusion_elements(result.circuit):
        impedances += element.get_impedances(frequencies)
    return impedances, result


def impedance_subtracted(dataset, impedances):
    """A detached copy of an EISDataset with `impedances` subtracted point by
    point.

    A copy, for the same reason inductive_tail_removed is one: this is a
    per-analysis filter, and the sweep the other steps read must not move.
    Split out from diffusion_subtracted so a caller holding a cached fit can
    apply it again without paying for the fit."""
    filtered = detached_copy(dataset)
    filtered.data.subtract_impedances(impedances)
    return filtered


def diffusion_subtracted(dataset, cdc, **fit_kwargs):
    """A detached copy of an EISDataset with a fitted diffusion element's
    impedance subtracted, and the fit it came from.

    The fit comes back with it because a subtraction rewrites measured data:
    how well the model fitted is the only thing that says whether the result
    is a cleaned spectrum or an invented one."""
    impedances, result = diffusion_impedance(dataset, cdc, **fit_kwargs)
    return impedance_subtracted(dataset, impedances), result


def diffusion_element_cdc(result) -> str:
    """The fitted diffusion element(s) as a CDC fragment, values included, for
    appending to a circuit built from a DRT of the subtracted sweep.

    Fitting a diffusion-free circuit to a sweep that still has its tail drives
    the kinetic resistances badly out -- on a two-arc test cell with known
    values, by 77% and 206% -- because the R-CPE pairs absorb the tail when
    nothing else in the model can. Note that pseudo chi-squared does not show
    this: it stays around 0.07 while the parameters are meaningless.

    Subtracting the tail from the data before fitting is not the fix either;
    that leaves 44% and 72%, the subtraction's own residual having been baked
    into the parameters instead. Putting the element back into the *model* and
    fitting the measured sweep is what recovers them, to 0.4% and 0.2% -- and
    the values below are already fitted, so they seed it as well.
    """
    return "".join(
        element.to_string(6) for element in _series_diffusion_elements(result.circuit)
    )


def describe_diffusion_fit(result) -> str:
    """What was subtracted and how well it fitted, for the DRT panel's
    readout, as exactly two lines: the fitted element(s), then the fit quality.

    Split across two lines because the settings panel is narrow enough that
    one line wrapped at an unpredictable point, and what fell past the bottom
    was the pseudo chi-squared -- the half that says whether to trust the
    subtraction at all. Fixed parameters are left out of the first line for
    the same reason: n is never fitted, so it carried no information while
    pushing everything after it onto the next row.
    """
    circuit = result.circuit
    parts = []
    for element in _series_diffusion_elements(circuit):
        values = ", ".join(
            f"{key}={value:.4g}"
            for key, value in element.get_values().items()
            if not element.is_fixed(key)
        )
        parts.append(f"{circuit.get_element_name(element)} ({values})")
    return f"{'; '.join(parts)}\npseudo χ² {result.pseudo_chisqr:.3g}"