#!/usr/bin/env python3
"""Zeolite deep-dive: learning curve analysis + embedding diversity."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import os
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/mace_al_zeolite_seed42.csv")
print(f"Zeolite AL: {list(df.columns)}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for col in df.columns:
    ls = '-' if col.startswith(('G_','I_','J_','K_','L_')) else '--'
    lw = 2 if col.startswith(('G_','I_','J_','K_','L_')) else 1
    axes[0].plot(df[col].values, ls=ls, lw=lw, label=col, alpha=0.8)
axes[0].set_xlabel("AL Iteration"); axes[0].set_ylabel("MAE (eV)")
axes[0].set_title("Zeolite - All Strategies"); axes[0].legend(fontsize=6, ncol=2)
axes[0].grid(True, alpha=0.3)

rand_final = df["A_random"].iloc[-1]
imps = {c:(rand_final-df[c].iloc[-1])/rand_final*100 for c in df.columns if c!="A_random"}
sorted_imps = sorted(imps.items(), key=lambda x: x[1], reverse=True)
colors = ['#27AE60' if x[0].startswith(('G_','L_')) else '#3498DB' if x[0].startswith(('I_','J_','K_')) else '#95A5A6' for x in sorted_imps]
axes[1].barh([x[0] for x in sorted_imps], [x[1] for x in sorted_imps], color=colors)
axes[1].axvline(x=0, color='gray', linestyle='--')
axes[1].set_xlabel("Improvement over Random (%)"); axes[1].set_title("Zeolite - Final Improvement")

for col in ["A_random","G_hybrid_weighted","I_aud_rank","K_aud_bald","L_rho_diagnostic"]:
    if col in df.columns: axes[2].plot(df[col].values, '-o', label=col, markersize=3)
axes[2].set_xlabel("AL Iteration"); axes[2].set_ylabel("MAE (eV)")
axes[2].set_title("Zeolite - Key Strategies"); axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig("figures/zeolite_analysis.png", dpi=150)
print("Saved figures/zeolite_analysis.png")

best_mae = min(df[c].min() for c in df.columns if c!="A_random")
target = best_mae*1.1
print(f"Best MAE: {best_mae:.4f}, Target: {target:.4f}")
for c in df.columns:
    for i,v in enumerate(df[c].values):
        if v<=target: print(f"  {c}: reached at iter {i} (MAE={v:.4f})"); break
    else: print(f"  {c}: NEVER")
print("Done!")
