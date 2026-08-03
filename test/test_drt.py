import time

import numpy as np
from pyimpspec import DataSet

from core.drt import run_drt
from core.io_utils import EISDataset

# --- Synthetic spectrum: R0 + two parallel RC elements (tau = 5 ms, 300 ms) ---
rng = np.random.default_rng(42)
f = np.logspace(5, -1, 40)
w = 2 * np.pi * f
Z = 10 + 50 / (1 + 1j * w * 5e-3) + 30 / (1 + 1j * w * 0.3)
Z = Z * (1 + 0.001 * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))))

dataset = EISDataset(DataSet(frequencies=f, impedances=Z), index=0, source_file="synthetic")

# --- DRT (TR-RBF, Simple Run: fast, deterministic point estimate) ---
t0 = time.time()
result = run_drt(dataset)
print(f"DRT finished in {time.time() - t0:.3f} s")
print(f"lambda = {result.lambda_value:.3g}")

tau, gamma = result.get_drt_data()
peak_taus, peak_gammas = result.get_peaks()
print(f"Peaks at tau = {peak_taus} s (expected ~5e-3 and ~3e-1)")
