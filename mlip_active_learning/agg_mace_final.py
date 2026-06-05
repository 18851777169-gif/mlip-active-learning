"""Aggregate MACE fine-tuning AL results."""
import pandas as pd, numpy as np

systems = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
strategies = ["A_random","C_uncertainty","E_diversity","G_hybrid_weighted"]

print(f"{'='*70}")
print(f"  MACE FINE-TUNING AL — 3 seeds x 6 systems x 4 strategies")
print(f"{'='*70}")

cross_sys = {s: [] for s in strategies}

for sys_name in systems:
    all_best = {s: [] for s in strategies}
    for seed in [42, 52, 62]:
        try:
            df = pd.read_csv(f"results/mace_al_{sys_name}_seed{seed}.csv")
            for s in strategies:
                if s in df.columns:
                    all_best[s].append(np.min(df[s].values))
        except Exception as e:
            pass

    print(f"\n--- {sys_name} ---")
    print(f"  {'Strategy':<28} {'Best MAE':>14} {'vs Random':>10}")
    print(f"  {'-'*54}")

    if not all_best["A_random"]:
        print("  (no data)")
        continue

    rand_best = np.mean(all_best["A_random"])
    for s in strategies:
        if not all_best[s]:
            continue
        b = np.array(all_best[s])
        mu, std = b.mean(), b.std()
        imp = (rand_best - mu) / rand_best * 100
        marker = " <--" if s.startswith("G_") else ""
        print(f"  {s:<28} {mu:>8.4f} +/- {std:.4f}   {imp:>+8.1f}%{marker}")
        cross_sys[s].append(imp)

print(f"\n{'='*70}")
print(f"  CROSS-SYSTEM (avg improvement vs Random)")
print(f"{'='*70}")
for s in strategies:
    imps = np.array(cross_sys[s])
    if len(imps) == 0:
        continue
    better = sum(1 for x in imps if x > 0)
    print(f"  {s:<28} {imps.mean():+5.1f}% +/- {imps.std():.1f}%  ({better}/{len(systems)})")

avg_imp = {s: np.mean(cross_sys[s]) for s in strategies if cross_sys[s]}
ranked = sorted(avg_imp, key=avg_imp.get, reverse=True)
print(f"\n  Ranking:")
for i, s in enumerate(ranked):
    print(f"  {i+1}. {s}: {avg_imp[s]:+.1f}%")
