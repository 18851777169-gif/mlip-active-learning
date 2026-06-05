#!/usr/bin/env python
"""Run fast_experiment with multiple seeds and aggregate results."""
import subprocess, sys, json, numpy as np, pandas as pd
from pathlib import Path

SEEDS = [42, 52, 62]
SCRIPT = Path(__file__).parent / "fast_experiment.py"

all_curves = {}  # strategy -> list of curves (one per seed)

for seed in SEEDS:
    print(f"\n{'#'*50}\n# SEED {seed}\n{'#'*50}")
    result = subprocess.run(
        [sys.executable, "-u", str(SCRIPT), str(seed)],
        capture_output=True, text=True, cwd=SCRIPT.parent,
        timeout=3600,
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"SEED {seed} FAILED:\n{result.stderr[-500:]}")
        continue

    # Parse CSV output
    try:
        df = pd.read_csv(SCRIPT.parent / "results" / f"fast_experiment_seed{seed}.csv")
        for col in df.columns[1:]:
            all_curves.setdefault(col, []).append(df[col].values)
    except Exception as e:
        print(f"Parse error: {e}")

# Aggregate
print(f"\n{'='*60}")
print(f"AGGREGATE ({len(all_curves.get('A_random', []))} seeds)")
print(f"{'='*60}")
print(f"{'Strategy':<30} {'Final MAE':>16} {'Best MAE':>16}")
print("-" * 62)

agg = {}
for s in ["A_random", "C_uncertainty", "E_diversity", "G_hybrid_weighted", "H_hybrid_twostage"]:
    if s not in all_curves or len(all_curves[s]) == 0:
        continue
    curves = np.array(all_curves[s])
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    final_mu, final_std = mean_curve[-1], std_curve[-1]
    best_mu = np.nanmean(np.nanmin(curves, axis=1))
    print(f"  {s:<28} {final_mu:.4f} ± {final_std:.4f}   {best_mu:.4f}")
    agg[s] = {"mean": mean_curve, "std": std_curve}

# Save aggregate
df_mean = pd.DataFrame({s: agg[s]["mean"] for s in agg})
df_mean.index.name = "iteration"
df_mean.to_csv(SCRIPT.parent / "results" / "aggregate_curves_mean.csv")

df_std = pd.DataFrame({s: agg[s]["std"] for s in agg})
df_std.to_csv(SCRIPT.parent / "results" / "aggregate_curves_std.csv")

# Report best strategy
finals = {s: agg[s]["mean"][-1] for s in agg}
best = min(finals, key=finals.get)
print(f"\nBest strategy: {best} ({finals[best]:.4f} eV)")
print("Done!")
