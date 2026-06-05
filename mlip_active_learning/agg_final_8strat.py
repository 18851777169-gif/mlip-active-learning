"""Aggregate final 8-strategy 3-seed MS25 results with GPU-updated MgO/ZrO2."""
import pandas as pd, numpy as np

systems = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
strategies = ["A_random","B_gmm_uncertainty","C_ensemble_qbc","D_mc_dropout","E_diversity","F_latent_clustering","G_hybrid_weighted","H_hybrid_twostage"]

print(f"{'='*70}")
print(f"  MS25 FINAL — 8 strategies x 3 seeds")
print(f"{'='*70}")

cross_sys = {s: [] for s in strategies}

for sys_name in systems:
    all_best = {s: [] for s in strategies}
    for seed in [42, 52, 62]:
        # Check GPU results first for MgO/ZrO2
        gpu_csv = f"results/ms25_gpu_{sys_name}_seed{seed}.csv" if sys_name in ["MgO_surface","Zr_oxide_amorphous"] else None
        csv_path = f"results/ms25_{sys_name}_seed{seed}.csv"
        try:
            df = pd.read_csv(csv_path)
        except:
            # New GPU results might be in different location
            df = pd.read_csv(f"results/ms25_gpu_{sys_name}_seed{seed}.csv")
        for s in strategies:
            if s in df.columns:
                all_best[s].append(np.min(df[s].values))

    print(f"\n--- {sys_name} ---")
    print(f"  {'Strategy':<28} {'Best MAE':>14} {'vs Random':>10}")
    print(f"  {'-'*54}")

    rand_best = np.mean(all_best["A_random"])
    for s in strategies:
        if not all_best[s]:
            continue
        b = np.array(all_best[s])
        mu, std = b.mean(), b.std()
        imp = (rand_best - mu) / rand_best * 100
        marker = " <--" if s.startswith("G_") or s.startswith("H_") else ""
        print(f"  {s:<28} {mu:>8.2f} +/- {std:.2f}   {imp:>+8.1f}%{marker}")
        cross_sys[s].append(imp)

print(f"\n{'='*70}")
print(f"  CROSS-SYSTEM (avg improvement vs Random)")
print(f"{'='*70}")
for s in strategies:
    imps = np.array(cross_sys[s]) if cross_sys[s] else np.array([0])
    better = sum(1 for x in imps if x > 0)
    print(f"  {s:<28} {imps.mean():+5.1f}% +/- {imps.std():.1f}%  ({better}/{len(systems)})")

avg_imp = {s: np.mean(cross_sys[s]) if cross_sys[s] else 0 for s in strategies}
ranked = sorted(avg_imp, key=avg_imp.get, reverse=True)
print(f"\n  Ranking:")
for i, s in enumerate(ranked):
    print(f"  {i+1}. {s}: {avg_imp[s]:+.1f}%")
