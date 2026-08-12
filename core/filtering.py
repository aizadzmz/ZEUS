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
    one leaves the other alone. It keeps the original's index and file, and so
    its key."""
    from copy import deepcopy

    from core.io_utils import EISDataset

    return EISDataset(
        deepcopy(dataset.data), dataset.index, dataset.source_file, dataset.file_id
    )


def inductive_tail_removed(dataset):
    """A detached copy of an EISDataset with the inductive tail (Im(Z) > 0)
    masked on top of whatever is already masked. A copy rather than an in-place
    mask, so validation results are not marked stale."""
    filtered = detached_copy(dataset)
    Z = filtered.data.get_impedances(masked=None)  # all points, incl. masked
    inductive = {i: True for i, z in enumerate(Z) if z.imag > 0}
    # Guarded: set_mask({}) means "unmask everything", undoing the original's masking.
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
# Rewrites points rather than masking them: a fitted diffusion impedance is subtracted from every one, so the tail is flattened rather than dropped.


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
    connection -- the only ones a subtraction can leave the rest of the circuit
    behind, since Z_measured - Z_diffusion only works in series.
    """
    from core.ecm import DIFFUSION_SYMBOLS

    return [
        element
        for element in _top_series_elements(circuit)
        if element.get_symbol() in DIFFUSION_SYMBOLS
    ]


# How the polarisation resistance splits between arc and tail for the starting guesses; unreadable from the spectrum, so the fit tries a spread and keeps the best.
_SEED_DIFFUSION_FRACTIONS = (0.25, 0.5, 0.75)

# Below this many points the sweep is refused: the presets carry four or five parameters.
_MIN_FIT_POINTS = 8


def _seed_diffusion_element(element, r_diffusion: float, tau: float) -> None:
    """Set one diffusion element's initial values from a resistance and a time
    constant, in place. Each expression inverts that element's own impedance at
    the limit where the two are readable, with n at its (fixed) default:

    - W  = 1/(Y(jw)^n)            -> |Z| = R_d at w = 1/tau
    - Ws = tanh((Bjw)^n)/(Yjw)^n  -> Z(0) = (B/Y)^n, knee at w = 1/B
    - Wo = coth(...)/...          -> same knee; Ws's relation reused for |Z|
    - G  = 1/(Y(k+jw)^n)          -> Z(0) = 1/(Yk^n), knee at w = k

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
    """The sweep's ohmic resistance, as core.ecm.ohmic_resistance measures it."""
    from core.ecm import ohmic_resistance

    return ohmic_resistance(frequencies, impedances)


def _seeded_cdcs(dataset, cdc):
    """The subtraction circuit, respelled with initial values taken from the
    sweep, once per starting guess. Same trick as core.ecm.seed_cdc, but sourced
    from the data rather than from a previous fit.
    """
    import numpy as np
    from pyimpspec import Resistor, parse_cdc

    impedances = dataset.impedances
    frequencies = dataset.frequencies
    r_series = _ohmic_resistance(frequencies, impedances)
    r_low = float(impedances[np.argmin(frequencies)].real)
    # Floored, not trusted: a sweep that never turns over can leave every seed negative.
    r_polarisation = max(r_low - r_series, abs(r_series)) or 1.0
    # The tail is what the lowest measured frequency resolves, so it sets the time constant.
    tau = 1.0 / (2.0 * np.pi * float(np.min(frequencies)))

    for fraction in _SEED_DIFFUSION_FRACTIONS:
        circuit = parse_cdc(cdc)
        series_elements = _top_series_elements(circuit)
        diffusion = _series_diffusion_elements(circuit)

        # The first series resistor is the ohmic one; the rest share what is left.
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

    The inductive points are held out of the fit whether or not the DRT step's
    own filter is on -- none of the subtraction circuits contains an inductor.
    They are still subtracted from: held out of the fit is not dropped.
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
    but the impedances cover *every* point, as DataSet.subtract_impedances
    requires.
    """
    import numpy as np
    from pyimpspec import parse_cdc

    # Checked before the fit: a non-series diffusion element can never be used here.
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
    # Summed, because several series diffusion elements are still in series.
    for element in _series_diffusion_elements(result.circuit):
        impedances += element.get_impedances(frequencies)
    return impedances, result


def impedance_subtracted(dataset, impedances):
    """A detached copy of an EISDataset with `impedances` subtracted point by
    point. Split out from diffusion_subtracted so a caller holding a cached fit
    can apply it again without paying for the fit."""
    filtered = detached_copy(dataset)
    filtered.data.subtract_impedances(impedances)
    return filtered


def diffusion_subtracted(dataset, cdc, **fit_kwargs):
    """A detached copy of an EISDataset with a fitted diffusion element's
    impedance subtracted, and the fit it came from -- how well the model fitted
    being the only thing that says whether the result is cleaned or invented."""
    impedances, result = diffusion_impedance(dataset, cdc, **fit_kwargs)
    return impedance_subtracted(dataset, impedances), result


def diffusion_element_cdc(result) -> str:
    """The fitted diffusion element(s) as a CDC fragment, values included, for
    appending to a circuit built from a DRT of the subtracted sweep. The values
    are already fitted, so they seed it as well.
    """
    return "".join(
        element.to_string(6) for element in _series_diffusion_elements(result.circuit)
    )


def describe_diffusion_fit(result) -> str:
    """What was subtracted and how well it fitted, for the DRT panel's readout,
    as exactly two lines: the fitted element(s), then the fit quality. Fixed
    parameters are left out of the first line.
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