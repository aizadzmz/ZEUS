import numpy as np
from pyimpspec import DataSet

from core.drt import analyze_drt_peaks, run_drt
from core.ecm import (
    CIRCUIT_PRESETS,
    canonical_cdc,
    circuit_from_drt_peaks,
    format_fit_report,
    run_ecm_fit,
    run_ecm_fit_seeded,
    seed_cdc,
    series_resistance,
    validate_cdc,
)
from core.io_utils import EISDataset
from core.session import SCHEMA_VERSION, load_session, save_session

# --- Synthetic spectrum: R0 + two parallel RC elements (tau = 5 ms, 300 ms) ---
# Same construction as test_drt.py/test_session.py, but here the ground truth
# is the point: R(RC)(RC) is the *true* model of this spectrum, so a fit of it
# must recover R0=10, R1=50, R2=30 and must beat any simpler circuit.
R0_TRUE, R1_TRUE, R2_TRUE = 10.0, 50.0, 30.0
rng = np.random.default_rng(42)
f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = R0_TRUE + R1_TRUE / (1 + 1j * w * 5e-3) + R2_TRUE / (1 + 1j * w * 0.3)
Z = Z * (1 + 0.001 * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))))

dataset = EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file="synthetic")

TRUE_CDC = "R(RC)(RC)"


def resistances(result):
    """The fitted resistor values, sorted. Sorted because pyimpspec does not
    guarantee which parallel branch of the circuit lands on which RC pair --
    the fit is of a set of time constants, not an ordered list."""
    return sorted(
        p.value
        for name, params in result.parameters.items()
        for symbol, p in params.items()
        if symbol == "R"
    )


# --- CDC validation and canonicalization ---
ok, message = validate_cdc(TRUE_CDC)
assert ok, message
assert "5 parameter" in message, message

for bad in ("R(RC", "R(XY)", "", "   "):
    ok, message = validate_cdc(bad)
    assert not ok, f"{bad!r} should not validate (got {message!r})"
print("CDC validation OK (accepts valid, rejects malformed/unknown/empty)")

# Harmless typing differences must collapse to one cache key, or fitting the
# "same" circuit twice would silently produce two entries.
assert canonical_cdc("R(RC)") == canonical_cdc("R( RC )") == canonical_cdc("[R(RC)]")
assert canonical_cdc("R(RC)") != canonical_cdc("R(RC)(RC)")
print("CDC canonicalization OK, R(RC) ->", canonical_cdc("R(RC)"))

# Every shipped preset must parse -- they populate a dropdown, so a typo in
# one is a dead menu entry rather than something the user can work around.
for name, cdc in CIRCUIT_PRESETS:
    ok, message = validate_cdc(cdc)
    assert ok, f"preset {name!r} ({cdc}) does not parse: {message}"
print(f"all {len(CIRCUIT_PRESETS)} circuit presets parse OK")

# --- Fitting the true circuit recovers the ground truth ---
result = run_ecm_fit(dataset, TRUE_CDC)
fitted = resistances(result)
expected = sorted([R0_TRUE, R1_TRUE, R2_TRUE])
assert np.allclose(fitted, expected, rtol=0.02), f"{fitted} != {expected}"
assert result.pseudo_chisqr < 1e-3, result.pseudo_chisqr
print(
    "fit recovers ground truth OK:",
    [round(v, 3) for v in fitted], "vs", expected,
    f"(chi2 = {result.pseudo_chisqr:.2e})",
)

# The default method is chosen specifically so error bars exist -- see
# core.ecm.run_ecm_fit. A NaN here means that default regressed.
errors = [p.stderr for params in result.parameters.values() for p in params.values()]
assert all(e == e for e in errors), f"expected finite error bars, got {errors}"
print("all parameters have finite error bars OK, max =", f"{max(errors):.4g}%")

# --- Seeding from a previous fit ---
# A perturbed sweep, as a later cycle in a degradation series would be.
Z_b = Z * 1.05
dataset_b = EISDataset(
    DataSet(frequencies=f, impedances=Z_b), index=1, source_file="synthetic"
)

seeded_cdc = seed_cdc(TRUE_CDC, result)
assert seeded_cdc != TRUE_CDC, "seeding should rewrite the CDC with fitted values"
assert canonical_cdc(seeded_cdc) == canonical_cdc(TRUE_CDC), "topology must not change"
# A previous fit of a different circuit is not seedable and must be ignored
# rather than mismatched element-by-element.
assert seed_cdc("R(RC)", result) == "R(RC)"
assert seed_cdc(TRUE_CDC, None) == TRUE_CDC
print("seed_cdc OK (rewrites values, preserves topology, falls back on mismatch)")

seeded = run_ecm_fit_seeded(dataset_b, TRUE_CDC, result)
unseeded = run_ecm_fit(dataset_b, TRUE_CDC)
assert np.allclose(resistances(seeded), resistances(unseeded), rtol=0.01)
assert np.allclose(resistances(seeded), [v * 1.05 for v in expected], rtol=0.02)
print(
    "seeded fit converges to the same answer as unseeded OK:",
    [round(v, 3) for v in resistances(seeded)],
)

# --- Two circuits fitted to one sweep both survive ---
# The regression test for keying the cache by (circuit, sweep) rather than
# sweep alone: fitting a second circuit must not evict the first.
simple = run_ecm_fit(dataset, "R(RC)")
ecm_results = {
    (canonical_cdc("R(RC)"), dataset.key): simple,
    (canonical_cdc(TRUE_CDC), dataset.key): result,
}
assert len(ecm_results) == 2, "second circuit overwrote the first"
assert {key for _, key in ecm_results} == {dataset.key}
# R(RC)(RC) is the true model, so it must fit better than the single-RC one.
assert result.pseudo_chisqr < simple.pseudo_chisqr
print(
    "two circuits coexist for one sweep OK:",
    f"R(RC) chi2={simple.pseudo_chisqr:.2e} > {TRUE_CDC} chi2={result.pseudo_chisqr:.2e}",
)

# --- Report formatting ---
report = format_fit_report(result, "Set 01")
assert "Set 01" in report and "[R(RC)(RC)]" in report
assert "pseudo chi-squared" in report
assert "Std. err. (%)" in report          # parameters dataframe
assert "Akaike info. criterion" in report  # statistics dataframe
print("format_fit_report OK")

# --- Building a circuit from DRT peaks ---
# The DRT supplies what an ECM cannot infer on its own: how many processes
# there are. On this spectrum the answer is known to be two.
peaks = analyze_drt_peaks(run_drt(dataset))
r_series = series_resistance(dataset)
assert np.isclose(r_series, R0_TRUE, rtol=0.02), r_series

generated = circuit_from_drt_peaks(peaks, r_series)
ok, message = validate_cdc(generated)
assert ok, f"generated circuit does not parse: {generated} ({message})"
# Two real peaks -> two R-CPE pairs, whatever peak fitting reported in total.
assert canonical_cdc(generated) == "[R(RQ)(RQ)]", canonical_cdc(generated)
assert peaks.get_num_peaks() > 2, "expected negligible edge peaks to be present"
print(
    f"circuit from DRT OK: {peaks.get_num_peaks()} raw peak(s) -> "
    f"{canonical_cdc(generated)}"
)

# Keeping every peak must produce a pair per peak -- i.e. the trimming above
# is the threshold doing its job, not the peak count being small.
kept_all = circuit_from_drt_peaks(peaks, r_series, min_area_fraction=0.0)
assert canonical_cdc(kept_all).count("(") == peaks.get_num_peaks()
print("min_area_fraction OK: 0.0 keeps all", peaks.get_num_peaks(), "peaks")

# The ideal-capacitor variant describes the same topology with C in place of Q.
generated_c = circuit_from_drt_peaks(peaks, r_series, use_cpe=False)
assert canonical_cdc(generated_c) == "[R(RC)(RC)]", canonical_cdc(generated_c)

# What matters is that the generated starting point actually fits: both
# variants must recover the ground truth, from values the DRT supplied rather
# than from pyimpspec's generic defaults.
for label, cdc in (("R(RQ)(RQ)", generated), ("R(RC)(RC)", generated_c)):
    generated_fit = run_ecm_fit(dataset, cdc)
    assert np.allclose(resistances(generated_fit), expected, rtol=0.02), (
        label, resistances(generated_fit)
    )
    print(
        f"  {label} from DRT fits to {[round(v, 3) for v in resistances(generated_fit)]}"
        f" (chi2 = {generated_fit.pseudo_chisqr:.2e})"
    )

# --- Session round-trip ---
ecm_params = {
    (canonical_cdc(TRUE_CDC), dataset.key): {"method": "least_squares", "weight": "boukamp"},
    (canonical_cdc("R(RC)"), dataset.key): {"method": "least_squares", "weight": "boukamp"},
}
path = "test/data/_session_ecm_tmp.eisz"
save_session(
    path, [dataset], {}, {}, {}, {}, {}, {},
    ecm_results=ecm_results,
    ecm_params=ecm_params,
)

(
    loaded_datasets,
    _,
    _,
    _,
    _,
    _,
    _,
    loaded_ecm,
    loaded_ecm_params,
) = load_session(path)

assert SCHEMA_VERSION == 4, SCHEMA_VERSION
assert set(loaded_ecm) == set(ecm_results), f"{set(loaded_ecm)} != {set(ecm_results)}"
assert loaded_ecm_params == ecm_params

restored = loaded_ecm[(canonical_cdc(TRUE_CDC), dataset.key)]
assert np.allclose(resistances(restored), resistances(result))
assert np.isclose(restored.pseudo_chisqr, result.pseudo_chisqr)
assert restored.method == result.method and restored.weight == result.weight
assert np.allclose(restored.frequencies, result.frequencies)
assert np.allclose(restored.impedances, result.impedances)
assert np.allclose(restored.residuals, result.residuals)
# The fitted values travel with the topology, so the restored circuit
# reproduces the curve rather than just its shape.
assert np.allclose(
    restored.get_impedances(num_per_decade=20),
    result.get_impedances(num_per_decade=20),
)
# Error bars and the lmfit-derived statistics must survive too -- both are
# reconstructed by hand, so this is what proves the stand-in works.
restored_errors = [p.stderr for params in restored.parameters.values() for p in params.values()]
assert np.allclose(sorted(restored_errors), sorted(errors))
assert "Akaike info. criterion" in format_fit_report(restored, "Set 01")
print("ECM session round-trip OK (values, errors, statistics, curve)")

# A saved session with no fits must still load, and old sessions predate the
# ecm_results block entirely.
path_empty = "test/data/_session_ecm_empty_tmp.eisz"
save_session(path_empty, [dataset], {}, {}, {})
assert load_session(path_empty)[7] == {}
print("session without ECM fits loads OK")

print("ALL ECM CHECKS PASSED")
