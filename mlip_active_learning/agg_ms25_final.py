"""Aggregate 3-seed MS25 results: mean +/- std across seeds."""
import pandas as pd, numpy as np

systems = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
strategies = ["A_random","C_uncertainty","E_diversity","G_hybrid_weighted","H_hybrid_twostage"]

print(f"{'='*70}")
print(f"  MS25 3-SEED AGGREGATE RESULTS")
print(f"{'='*70}")

cross_sys = {s: [] for s in strategies}

for sys_name in systems:
    print(f"\n--- {sys_name} ---")
    print(f"  {'Strategy':<28} {'Best MAE':>16} {'vs Random':>10}")
    print(f"  {'-'*54}")

    all_best = {s: [] for s in strategies}
    for seed in [42, 52, 62]:
        df = pd.read_csv(f"results/ms25_{sys_name}_seed{seed}.csv")
        for s in strategies:
            all_best[s].append(np.min(df[s].values))

    rand_best = np.mean(all_best["A_random"])
    for s in strategies:
        b = np.array(all_best[s])
        mu, std = b.mean(), b.std()
        imp = (rand_best - mu) / rand_best * 100
        marker = " <--" if s.startswith("G_") or s.startswith("H_") else ""
        print(f"  {s:<28} {mu:>8.2f} +/- {std:.2f}   {imp:>+8.1f}%{marker}")
        cross_sys[s].append(imp)

print(f"\n{'='*70}")
print(f"  CROSS-SYSTEM SUMMARY (avg improvement vs Random)")
print(f"{'='*70}")
for s in strategies:
    imps = np.array(cross_sys[s])
    better = sum(1 for x in imps if x > 0)
    print(f"  {s:<28} {imps.mean():+5.1f}% +/- {imps.std():.1f}%  (better in {better}/{len(systems)})")

# Overall ranking
avg_imp = {s: np.mean(cross_sys[s]) for s in strategies}
ranked = sorted(avg_imp, key=avg_imp.get, reverse=True)
print(f"\n  Overall ranking (avg improvement):")
for i, s in enumerate(ranked):
    print(f"  {i+1}. {s}: {avg_imp[s]:+.1f}%")
