"""Final aggregation: 6 systems x 3 seeds x 8 strategies."""
import pandas as pd, numpy as np

systems = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
strategies = ["A_random","B_gmm_uncertainty","C_ensemble_qbc","D_mc_dropout","E_diversity","F_latent_clustering","G_hybrid_weighted","H_hybrid_twostage"]

print("="*70)
print("  FINAL 8-STRATEGY 3-SEED RESULTS")
print("="*70)

cross = {s: [] for s in strategies}

for s in systems:
    best = {k: [] for k in strategies}
    for seed in [42,52,62]:
        df = pd.read_csv(f"results/ms25_{s}_seed{seed}.csv")
        for k in strategies:
            if k in df.columns:
                best[k].append(np.nanmin(df[k].values))

    if not best["A_random"]:
        continue
    rb = np.mean(best["A_random"])

    print(f"\n--- {s} ---")
    print(f"  {'Strategy':<28} {'Best':>12} {'vs Random':>10}")
    print(f"  {'-'*52}")
    for k in strategies:
        if not best[k]: continue
        b = np.array(best[k]); mu, std = b.mean(), b.std()
        imp = (rb - mu) / rb * 100
        m = " <--" if k.startswith("G_") or k.startswith("H_") else ""
        print(f"  {k:<28} {mu:>8.3f}+/-{std:.3f} {imp:+8.1f}%{m}")
        cross[k].append(imp)

print(f"\n{'='*70}")
print(f"  CROSS-SYSTEM RANKING")
print(f"{'='*70}")
avg = {}
for k in strategies:
    if not cross[k]: continue
    imps = np.array(cross[k])
    n = sum(1 for x in imps if x > 0)
    avg[k] = imps.mean()
    print(f"  {k:<28} {imps.mean():+5.1f}% +/- {imps.std():.1f}% ({n}/{len(systems)})")

ranked = sorted(avg, key=avg.get, reverse=True)
print(f"\n  Ranking:")
for i, k in enumerate(ranked):
    print(f"  {i+1}. {k}: {avg[k]:+.1f}%")
