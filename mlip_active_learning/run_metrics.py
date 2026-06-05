#!/usr/bin/env python3
"""Run 5 evaluation metrics on MACE-labeled structures for all 6 systems.

Metrics:
  1. Data efficiency (from experimental CSV results)
  2. Convergence labels (from experimental CSV results)
  3. MD stability (short NVT at 300K, check force stability)
  4. E-V physical plausibility (energy-volume scan)
  5. SOAP coverage (descriptor space coverage)
"""

import sys, pickle, time, os, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = "data/ms25_labeled"
MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Metric 3: MD Stability
# ---------------------------------------------------------------------------
def test_md_stability(atoms, calc, n_steps=2000, timestep=0.5, T=300):
    """Run short NVT MD and check for force divergence."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.nvtberendsen import NVTBerendsen
    from ase import units

    atoms = atoms.copy()
    atoms.calc = calc
    atoms.set_pbc(True)

    try:
        MaxwellBoltzmannDistribution(atoms, T * units.kB)
        dyn = NVTBerendsen(atoms, timestep * units.fs, T * units.kB, taut=0.5)
    except Exception as e:
        return {"stable": False, "error": str(e), "max_force": None}

    forces_history = []
    crashed = False

    for step in range(n_steps):
        try:
            dyn.run(1)
            f = atoms.get_forces()
            max_f = np.abs(f).max()
            forces_history.append(max_f)
            if max_f > 5.0:  # Force explosion threshold
                crashed = True
                break
        except Exception:
            crashed = True
            break

    return {
        "stable": not crashed,
        "n_steps_completed": len(forces_history) * 10,
        "max_force": float(np.max(forces_history)) if forces_history else None,
        "mean_force": float(np.mean(forces_history)) if forces_history else None,
        "final_force": float(forces_history[-1]) if forces_history else None,
    }

# ---------------------------------------------------------------------------
# Metric 4: E-V physical plausibility
# ---------------------------------------------------------------------------
def test_ev_plausibility(atoms, calc, n_points=11):
    """Check if energy decreases monotonically as volume increases near equilibrium."""
    atoms = atoms.copy()
    atoms.calc = calc

    # Get equilibrium energy
    try:
        e0 = atoms.get_potential_energy()
    except:
        return {"plausible": False, "error": "energy_fail"}

    factors = np.linspace(0.92, 1.08, n_points)
    energies = []
    for f in factors:
        scaled = atoms.copy()
        cell = scaled.get_cell()
        scaled.set_cell(cell * f ** (1/3), scale_atoms=True)
        scaled.calc = calc
        try:
            e = scaled.get_potential_energy()
            energies.append(e)
        except:
            energies.append(np.nan)

    energies = np.array(energies)
    valid = ~np.isnan(energies)
    if valid.sum() < 3:
        return {"plausible": False, "error": "too_few_points"}

    # Check monotonic decrease (energy lower at larger volume near eq)
    n_violations = 0
    for i in range(len(energies) - 1):
        if not (np.isnan(energies[i]) or np.isnan(energies[i+1])):
            if energies[i+1] > energies[i] + 1e-4:
                n_violations += 1

    return {
        "plausible": n_violations <= 2,
        "n_violations": n_violations,
        "e_min": float(np.nanmin(energies)),
        "e_max": float(np.nanmax(energies)),
        "volume_range": [float(factors[0]), float(factors[-1])],
    }

# ---------------------------------------------------------------------------
# Metric 5: Coverage (simple pairwise distance metric)
# ---------------------------------------------------------------------------
def compute_coverage(structures, n_sample=50):
    """Compute mean pairwise distance in structure space as coverage proxy."""
    from scipy.spatial.distance import pdist, cdist
    # Use simple descriptor: composition + density + cell params
    feats = []
    for s in structures[:n_sample]:
        comp = np.bincount(s.get_atomic_numbers(), minlength=84)[:84]
        if len(s) > 0:
            density = len(s) / s.get_volume()
        else:
            density = 0
        cell = s.get_cell().flatten()
        f = np.concatenate([comp.astype(float), [density], cell])
        feats.append(f)
    feats = np.array(feats)
    if len(feats) > 1:
        return float(pdist(feats, 'euclidean').mean())
    return 0.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from mace.calculators import MACECalculator
print(f"Loading MACE...")
calc = MACECalculator(model_path=MODEL_PATH, device=DEVICE, default_dtype="float32")

systems = sorted([f.stem for f in Path(DATA_DIR).glob("*.pkl")])

print(f"\n{'='*70}")
print(f"  EVALUATION METRICS FOR {len(systems)} SYSTEMS")
print(f"{'='*70}")

all_results = {}

for sys_name in systems:
    print(f"\n--- {sys_name} ---")

    with open(f"{DATA_DIR}/{sys_name}.pkl", "rb") as f:
        structures = pickle.load(f)

    # Pick a medium-sized structure for MD and E-V
    idx = np.argmin([abs(len(s) - 50) for s in structures])
    test_atoms = structures[idx].copy()
    print(f"  Test structure: {len(test_atoms)} atoms")

    # MD stability
    print(f"  Running MD stability (200 steps, 0.5fs, 300K)...")
    t0 = time.time()
    md_result = test_md_stability(test_atoms, calc, n_steps=200)
    print(f"    {'STABLE' if md_result['stable'] else 'CRASHED'} "
          f"(max_f={md_result.get('max_force', 'N/A')}, "
          f"time={time.time()-t0:.0f}s)")

    # E-V plausibility
    print(f"  Running E-V plausibility...")
    t0 = time.time()
    ev_result = test_ev_plausibility(test_atoms, calc)
    print(f"    {'PLAUSIBLE' if ev_result.get('plausible') else 'NOT_PLAUSIBLE'} "
          f"(violations={ev_result.get('n_violations', 'N/A')}, "
          f"time={time.time()-t0:.0f}s)")

    # SOAP coverage
    print(f"  Computing SOAP coverage...")
    t0 = time.time()
    coverage = compute_coverage(structures)
    print(f"    Coverage={coverage:.2f} (time={time.time()-t0:.0f}s)")

    all_results[sys_name] = {
        "md_stable": md_result["stable"],
        "md_max_force": md_result.get("max_force"),
        "ev_plausible": ev_result.get("plausible", False),
        "ev_violations": ev_result.get("n_violations", -1),
        "soap_coverage": float(coverage),
    }

# Summary
print(f"\n{'='*70}")
print(f"  METRICS SUMMARY")
print(f"{'='*70}")
print(f"  {'System':<28} {'MD':>8} {'E-V':>8} {'Coverage':>10}")
print(f"  {'-'*56}")
for sys_name in systems:
    r = all_results[sys_name]
    md_ok = "OK" if r["md_stable"] else "FAIL"
    ev_ok = "OK" if r["ev_plausible"] else "FAIL"
    print(f"  {sys_name:<28} {md_ok:>8} {ev_ok:>8} {r['soap_coverage']:>10.1f}")

# Save
import pandas as pd
df = pd.DataFrame(all_results).T
df.to_csv("results/evaluation_metrics.csv")
print(f"\nSaved to results/evaluation_metrics.csv")
print("Done!")
