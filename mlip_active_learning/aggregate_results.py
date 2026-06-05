"""Aggregate multi-seed experiment results."""
import pandas as pd, numpy as np, sys
from pathlib import Path

seeds = [42, 52, 62]
strategies = ["A_random", "C_uncertainty", "E_diversity", "G_hybrid_weighted", "H_hybrid_twostage"]
all_curves = {s: [] for s in strategies}

for seed in seeds:
    try:
        df = pd.read_csv(f"results/fast_experiment_seed{seed}.csv")
        for s in strategies:
            if s in df.columns:
                all_curves[s].append(df[s].values)
    except FileNotFoundError:
        print(f"Seed {seed} CSV not found, skipping")

print(f"Aggregate over {len(all_curves['A_random'])} seeds:")
print(f"{'Strategy':<30} {'Final MAE':>18} {'Best MAE':>18}")
print("-" * 68)
for s in strategies:
    curves = np.array(all_curves[s])
    finals = curves[:, -1]
    bests = np.nanmin(curves, axis=1)
    final_mu, final_std = np.nanmean(finals), np.nanstd(finals)
    best_mu, best_std = np.nanmean(bests), np.nanstd(bests)
    print(f"  {s:<28} {final_mu:.4f} +/- {final_std:.4f}   {best_mu:.4f} +/- {best_std:.4f}")

final_means = {s: np.nanmean(np.array(all_curves[s])[:, -1]) for s in strategies}
ranked = sorted(final_means, key=final_means.get)
print("\nRanking (by final MAE):")
for i, s in enumerate(ranked):
    print(f"  {i+1}. {s}: {final_means[s]:.4f}")

random_best_mean = np.nanmean([np.nanmin(np.array(all_curves["A_random"])[i]) for i in range(len(seeds))])
print(f"\nRandom best MAE (avg): {random_best_mean:.4f}")
for s in strategies:
    if s == "A_random":
        continue
    curves = np.array(all_curves[s])
    s_best = np.nanmean(np.nanmin(curves, axis=1))
    imp = (random_best_mean - s_best) / random_best_mean * 100
    print(f"  {s}: best MAE={s_best:.4f}, vs Random: {imp:+.1f}%")

# Save aggregate CSV
agg_mean = {}
for s in strategies:
    curves = np.array(all_curves[s])
    agg_mean[s] = np.nanmean(curves, axis=0)
df_out = pd.DataFrame(agg_mean)
df_out.index.name = "iteration"
df_out.to_csv("results/aggregate_3seeds_mean.csv", index=False)
print("\nSaved aggregate_3seeds_mean.csv")
