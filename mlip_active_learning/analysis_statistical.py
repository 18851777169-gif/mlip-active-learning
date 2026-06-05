#!/usr/bin/env python3
"""Statistical analysis: Friedman + Wilcoxon + effect sizes + LaTeX table."""
import pandas as pd, numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, ttest_1samp

DATA_DIR = "results_mace_al"
SYSTEMS = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
S2 = ["A_random","C_uncertainty","E_diversity","G_hybrid_weighted","I_aud_rank","J_aud_batch","K_aud_bald","L_rho_diagnostic"]
SEEDS = [42,52,62]

data = {}
for s in SYSTEMS:
    data[s] = {}
    for k in S2:
        vals = []
        for seed in SEEDS:
            try:
                df = pd.read_csv(f"{DATA_DIR}/mace_al_{s}_seed{seed}.csv")
                if k in df.columns: vals.append(df[k].iloc[-1])
            except: pass
        if vals: data[s][k] = vals

print("=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# 1. Friedman
print("\n--- 1. Friedman Test ---")
for s in SYSTEMS:
    samples = [data[s][k] for k in S2 if k in data[s] and len(data[s][k])==3]
    if len(samples)>=3:
        st, p = friedmanchisquare(*samples)
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        print(f"  {s:<28s}: chi2={st:.2f}, p={p:.4f} {sig}")

# 2. Wilcoxon vs Random
print("\n--- 2. Pairwise Wilcoxon vs Random ---")
for s in SYSTEMS:
    if "A_random" not in data[s]: continue
    base = data[s]["A_random"]
    for k in S2:
        if k=="A_random" or k not in data[s]: continue
        comp = data[s][k]
        if len(base)==len(comp) and len(base)>=3:
            st, p = wilcoxon(base, comp, zero_method="wilcox")
            n=len(base); g=sum(1 for a,b in zip(base,comp) if a>b)
            e=sum(1 for a,b in zip(base,comp) if a==b); l=n-g-e
            d=(g-l)/n
            eff = "large" if abs(d)>0.474 else ("medium" if abs(d)>0.33 else ("small" if abs(d)>0.147 else "neg"))
            sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
            print(f"  {s:<28s} vs {k:<25s}: W={st:.0f} p={p:.4f}{sig} d={d:+.3f}({eff})")

# 3. Cross-system meta
print("\n--- 3. Cross-System Meta ---")
cross = {}
for s in SYSTEMS:
    if "A_random" not in data[s]: continue
    rb = np.mean(data[s]["A_random"])
    for k in S2:
        if k=="A_random" or k not in data[s]: continue
        cross.setdefault(k,[]).append((rb-np.mean(data[s][k]))/rb*100)

print(f"  {'Strategy':<28s} {'Mean':>8s} {'Std':>8s} {'Better':>8s} {'p(vs0)':>8s}")
for k in S2:
    if k=="A_random" or k not in cross: continue
    imps = cross[k]; mu,std=np.mean(imps),np.std(imps)
    n=sum(1 for x in imps if x>0)
    try: _,p=ttest_1samp(imps,0)
    except: p=1.0
    print(f"  {k:<28s} {mu:+7.1f}% {std:>7.1f}% {n:>5}/{len(imps)} {p:>8.4f}")

# 4. LaTeX
print("\n--- 4. LaTeX ---")
for s in SYSTEMS:
    if s not in data or "A_random" not in data[s]: continue
    rb=np.mean(data[s]["A_random"])
    best_k, best_imp = "A_random", 0
    for k in S2:
        if k=="A_random" or k not in data[s]: continue
        imp=(rb-np.mean(data[s][k]))/rb*100
        if imp>best_imp: best_imp,best_k=imp,k
    print(f"  {s.replace('_','\_')} & {best_k} & {best_imp:+.1f}\% \\\\")

print("\nDone!")
