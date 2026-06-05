#!/usr/bin/env python3
"""Main figures: heatmap + learning curves + ranking boxplot."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import os

DATA = "../results"
S = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
SN = ["HEA","MgO","Pt","ZrO2","Water","Zeolite"]
SK = ["A_random","C_uncertainty","E_diversity","G_hybrid_weighted","I_aud_rank","J_aud_batch","K_aud_bald","L_rho_diagnostic"]
KN = ["Random","QBC","Div","Hybrid-W","AUD-R","AUD-B","AUD-Bald","Rho-Diag"]
seeds = [42,52,62]

# Figure 1: Heatmap
fig, ax = plt.subplots(figsize=(12,7))
hm = np.zeros((len(S), len(SK)))
for i,s in enumerate(S):
    vals = {k:[] for k in SK}
    for seed in seeds:
        try:
            df = pd.read_csv(f"{DATA}/mace_al_{s}_seed{seed}.csv")
            for k in SK:
                if k in df.columns: vals[k].append(df[k].iloc[-1])
        except: pass
    rb = np.mean(vals["A_random"]) if vals["A_random"] else 1
    for j,k in enumerate(SK):
        if k=="A_random": hm[i,j]=0
        elif vals[k]: hm[i,j]=(rb-np.mean(vals[k]))/rb*100
im = ax.imshow(hm, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
ax.set_xticks(range(len(SK))); ax.set_xticklabels(KN, rotation=45, ha='right')
ax.set_yticks(range(len(S))); ax.set_yticklabels(SN)
for i in range(len(S)):
    for j in range(len(SK)):
        v=hm[i,j]; c='white' if abs(v)>15 else 'black'
        ax.text(j,i,f'{v:+.0f}%',ha='center',va='center',fontsize=9,color=c)
ax.set_title("MACE AL Improvement over Random (%)")
plt.colorbar(im, ax=ax, label='Improvement %')
plt.tight_layout(); plt.savefig("heatmap.png", dpi=150); plt.close()
print("Saved heatmap.png")

# Figure 2: Learning curves
fig, axes = plt.subplots(2,3,figsize=(15,10))
for idx,s in enumerate(S):
    ax=axes[idx//3,idx%3]
    for k,color in [("A_random","#888"),("G_hybrid_weighted","#27AE60"),("L_rho_diagnostic","#8E44AD")]:
        curves=[]
        for seed in seeds:
            try:
                df=pd.read_csv(f"{DATA}/mace_al_{s}_seed{seed}.csv")
                if k in df.columns: curves.append(df[k].values)
            except: pass
        if curves:
            st=np.array(curves); mu=np.mean(st,axis=0); std=np.std(st,axis=0)
            x=np.arange(len(mu))
            ax.plot(x,mu,'-o',label=k,color=color,markersize=3)
            ax.fill_between(x,mu-std,mu+std,alpha=0.15,color=color)
    ax.set_title(SN[idx]); ax.set_xlabel("Iter"); ax.set_ylabel("MAE (eV)")
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig("learning_curves.png", dpi=150); plt.close()
print("Saved learning_curves.png")

# Figure 3: Ranking boxplot
imps={k:[] for k in SK if k!="A_random"}
for s in S:
    vals={k:[] for k in SK}
    for seed in seeds:
        try:
            df=pd.read_csv(f"{DATA}/mace_al_{s}_seed{seed}.csv")
            for k in SK:
                if k in df.columns: vals[k].append(df[k].iloc[-1])
        except: pass
    rb=np.mean(vals["A_random"]) if vals["A_random"] else 1
    for k in SK:
        if k=="A_random" or not vals[k]: continue
        for v in vals[k]: imps[k].append((rb-v)/rb*100)

fig,ax=plt.subplots(figsize=(12,6))
data_plot=[imps[k] for k in imps]
labels=[KN[SK.index(k)] for k in imps]
bp=ax.boxplot(data_plot,labels=labels,patch_artist=True)
for patch,k in zip(bp['boxes'],imps):
    patch.set_facecolor('#3498DB' if 'AUD' in k else '#27AE60' if 'Hybrid' in k or 'Rho' in k else '#95A5A6')
ax.axhline(y=0,color='gray',linestyle='--')
ax.set_ylabel("Improvement over Random (%)"); ax.set_title("Strategy Performance Distribution")
plt.xticks(rotation=45,ha='right')
plt.tight_layout(); plt.savefig("ranking_boxplot.png", dpi=150); plt.close()
print("Saved ranking_boxplot.png")
print("All figures done!")
