#!/usr/bin/env python3
"""Generate all publication figures: heatmap + learning curves + ranks."""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f'{BASE}/figures', exist_ok=True)

# ─── Config ───
S = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
SN = ["HEA","MgO(100)","Pt(111)","ZrO$_2$(am)","H$_2$O(l)","Zeolite"]
seeds = [42,52,62]

SCHNET_DIR = f'{BASE}/results_schnet_ms25'
SCHNET_PREFIX = 'ms25_9strat'
MACE_DIR = f'{BASE}/results_mace_al'
MACE_PREFIX = 'mace_al'

schnet_cols_code = ['C_ensemble_qbc','E_diversity','F_latent_clustering','G_hybrid_weighted',
                     'H_hybrid_twostage','I_aud_rank','J_aud_batch','K_aud_bald','L_rho_diagnostic']
schnet_cols_short = ['C_ens','E_div','F_lat','G_wtd','H_2st','I_audR','J_audB','K_bld','L_rhoD']
mace_cols_code = ['C_uncertainty','E_diversity','G_hybrid_weighted','I_aud_rank','J_aud_batch','K_aud_bald','L_rho_diagnostic']
mace_cols_short = ['C_unc','E_div','G_wtd','I_audR','J_audB','K_bld','L_rhoD']

def load_matrix(data_dir, file_prefix, cols_code, cols_short):
    """Build improvement matrix [n_systems, n_strategies]."""
    matrix = np.zeros((len(S), len(cols_code)))
    for i, sn in enumerate(S):
        for seed in seeds:
            f = f'{data_dir}/{file_prefix}_{sn}_seed{seed}.csv'
            if os.path.exists(f):
                df = pd.read_csv(f)
                rand = np.min(df['A_random']) if 'A_random' in df.columns else df.iloc[:,-1].max()
                if 'A_random' in df.columns:
                    rand = np.min(df['A_random'])
                    for j, c in enumerate(cols_code):
                        if c in df.columns:
                            imp = (rand - np.min(df[c])) / rand * 100
                            matrix[i, j] += imp / len(seeds)
    return matrix

print("[1/3] Loading data...")
mat_s = load_matrix(SCHNET_DIR, SCHNET_PREFIX, schnet_cols_code, schnet_cols_short)
mat_m = load_matrix(MACE_DIR, MACE_PREFIX, mace_cols_code, mace_cols_short)

# ─── Fig 1a: SchNet heatmap ───
print("[2/3] Drawing...")
vmax = max(np.nanmax(mat_s), np.nanmax(mat_m), 15)
vmin = min(np.nanmin(mat_s), np.nanmin(mat_m), -55)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6.5))

for ax, mat, labs, title in [
    (ax1, mat_s, schnet_cols_short, 'a) SchNet (from scratch)'),
    (ax2, mat_m, mace_cols_short, 'b) MACE-MP-0 (fine-tuned)')]:
    sns.heatmap(mat, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                xticklabels=labs, yticklabels=SN,
                ax=ax, cbar_kws={'label':'Improvement over Random (%)','shrink':0.8},
                vmin=vmin, vmax=vmax, linewidths=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{BASE}/figures/fig1_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig1_heatmap.png")

# ─── Fig 2: Learning curves for key systems ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
rep = ['FeNiCrCoCu_HEA','liquid_water','zeolite']

for col, sn in enumerate(rep):
    for row, (arch, data_dir, file_prefix) in enumerate([
        ('SchNet', SCHNET_DIR, SCHNET_PREFIX),
        ('MACE', MACE_DIR, MACE_PREFIX)]):
        ax = axes[row, col]
        for strat, color, ls in [('A_random','grey','--'),('G_hybrid_weighted','#e67e22','-'),
                                  ('J_aud_batch','#2ecc71','-')]:
            curves = []
            for seed in seeds:
                f = f'{data_dir}/{file_prefix}_{sn}_seed{seed}.csv'
                if os.path.exists(f):
                    df = pd.read_csv(f)
                    if strat in df.columns:
                        curves.append(df[strat].values)
            if curves:
                arr = np.array([c[:min(len(c) for c in curves)] for c in curves])
                mu, std = np.mean(arr, axis=0), np.std(arr, axis=0)
                x = np.arange(len(mu))
                ax.plot(x, mu, ls, color=color, lw=2 if strat != 'A_random' else 1.2,
                       label='Random' if strat == 'A_random' else strat[:3])
                ax.fill_between(x, mu-std, mu+std, color=color, alpha=0.1)
        ax.set_title(f'{arch} — {sn.replace("_"," ")[:20]}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Iter'); ax.set_ylabel('MAE (eV)')
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{BASE}/figures/fig2_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig2_curves.png")

# ─── Fig 3: Strategy rank boxplots ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

for ax, arch, data_dir, file_prefix, cols_code, cols_short in [
    (ax1, 'SchNet', SCHNET_DIR, SCHNET_PREFIX, schnet_cols_code, schnet_cols_short),
    (ax2, 'MACE', MACE_DIR, MACE_PREFIX, mace_cols_code, mace_cols_short)]:

    rank_data = {s: [] for s in cols_short}
    for sn in S:
        for seed in seeds:
            f = f'{data_dir}/{file_prefix}_{sn}_seed{seed}.csv'
            if os.path.exists(f):
                df = pd.read_csv(f)
                vals = {}
                for c_code, c_short in zip(cols_code, cols_short):
                    if c_code in df.columns:
                        vals[c_short] = df[c_code].iloc[-1]
                if vals:
                    sorted_s = sorted(vals, key=lambda x: vals[x])
                    for rank, s in enumerate(sorted_s):
                        rank_data[s].append(rank + 1)

    bp = ax.boxplot([rank_data[s] for s in cols_short], labels=cols_short,
                     patch_artist=True, vert=True)
    for patch, c in zip(bp['boxes'], plt.cm.tab10(np.linspace(0,1,len(cols_short)))):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_title(arch, fontweight='bold')
    ax.set_ylabel('Rank (1=best)')
    ax.axhline(y=len(cols_short)/2, color='grey', ls=':', alpha=0.5)
    ax.tick_params(axis='x', rotation=40)
    ax.set_ylim(len(cols_short)+0.5, 0.5)

plt.tight_layout()
plt.savefig(f'{BASE}/figures/fig3_ranks.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig3_ranks.png")

print(f"\nDone. 3 figures in {BASE}/figures/")
