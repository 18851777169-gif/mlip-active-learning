#!/usr/bin/env python3
"""Synthetic GMM validation: Theorem test on 360 controlled configurations.

No DFT, no GPU needed. CPU ~30 min.
Tests Lemma 1, Lemma 2, and Theorem across parameter sweep.
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{BASE}/results', exist_ok=True)
os.makedirs(f'{BASE}/figures', exist_ok=True)

np.random.seed(42)

# ─── Parameter sweep ───
K_list = [2, 3, 5, 10]
pi_configs = {
    'uniform': lambda k: np.ones(k) / k,
    '1to2':    lambda k: np.array([2.0 if i < k//2 else 1.0 for i in range(k)]),
    '1to5':    lambda k: np.array([5.0 if i < k//3 else 1.0 for i in range(k)]),
}
# d_sep controls overlap: 0=fully merged, 5=fully separated
# v3.6.7 fix: equidistant centroids in 2D to avoid curse of dimensionality
d_sep_list = np.linspace(0.2, 10.0, 15)  # centroid separation / sigma
sigma_list = [1.0]  # reduced: sigma只是缩放因子
N0, NQ, T = 20, 10, 5  # matches MS25 experiment

D_EMB = 64  # Task A: 64D to match real embedding dimension

print("=" * 70)
print("SYNTHETIC GMM VALIDATION: Theorem Test")
print("=" * 70)

results = []

for K in K_list:
    for pi_name, pi_fn in pi_configs.items():
        for d_sep in d_sep_list:
            for sigma in sigma_list:
                pi_raw = pi_fn(K)
                pi = pi_raw / pi_raw.sum()
                dim = D_EMB

                # Generate equidistant centroids (regular K-gon) in first 2 dims, pad to D_EMB
                angles = np.linspace(0, 2*np.pi, K, endpoint=False)
                centroids_2d = np.column_stack([np.cos(angles), np.sin(angles)]) * d_sep * sigma
                centroids = np.zeros((K, dim))
                centroids[:, :2] = centroids_2d

                # Generate total N = 500 structures (simulating a full pool)
                N_pool = 500
                X = np.zeros((N_pool, dim))
                y = np.zeros(N_pool, dtype=int)
                counts = np.random.multinomial(N_pool, pi)
                idx = 0
                for k in range(K):
                    nk = counts[k]
                    X[idx:idx+nk] = centroids[k] + np.random.randn(nk, dim) * sigma
                    y[idx:idx+nk] = k
                    idx += nk

                # Silhouette score (ground truth labels)
                if K > 1 and len(X) > K * 5:
                    sub_idx = np.random.choice(len(X), min(200, len(X)), replace=False)
                    sil = silhouette_score(X[sub_idx], y[sub_idx])
                else:
                    sil = 0.9

                # Run virtual AL for 5 strategies
                strat_imps = {}
                for strategy in ['A_random','C_uncertainty','E_diversity','G_hybrid','J_batch']:
                    labeled = set(np.random.choice(N_pool, N0, replace=False))
                    first_full = 9999  # 9999 = not reached yet

                    for iteration in range(T):
                        pool = list(set(range(N_pool)) - labeled)
                        if len(pool) < NQ:
                            break

                        # Compute uncertainty
                        if strategy == 'C_uncertainty':
                            labeled_X = np.array([X[li] for li in labeled])
                            # U = distance to nearest labeled point (far = uncertain)
                            U = cdist(X[pool], labeled_X, metric='euclidean').min(axis=1)
                            selected = np.argsort(U)[-NQ:]

                        elif strategy == 'E_diversity':
                            labeled_X = np.array([X[li] for li in labeled])
                            D = np.array([cdist(X[[p]], labeled_X, metric='cosine').min()
                                          for p in pool])
                            selected = []
                            d_copy = D.copy()
                            for _ in range(NQ):
                                idx = int(np.argmax(d_copy))
                                selected.append(idx)
                                new_d = cdist(X[[pool[idx]]], X[pool], metric='cosine').ravel()
                                d_copy = np.minimum(d_copy, new_d)
                            selected = np.array(selected)

                        elif strategy == 'G_hybrid':
                            labeled_X = np.array([X[li] for li in labeled])
                            U = np.zeros(len(pool))
                            # U = distance to nearest labeled point
                            U = cdist(X[pool], labeled_X, metric='euclidean').min(axis=1)
                            D = np.array([cdist(X[[p]], labeled_X, metric='cosine').min()
                                          for p in pool])
                            U_n = (U - U.min()) / (U.max() - U.min() + 1e-10)
                            D_n = (D - D.min()) / (D.max() - D.min() + 1e-10)
                            rho, _ = spearmanr(U_n, D_n)
                            alpha = np.clip(0.5 - 0.3 * rho, 0.2, 0.8)
                            scores = alpha * U_n + (1-alpha) * D_n
                            selected = np.argsort(scores)[-NQ:]

                        elif strategy == 'J_batch':
                            labeled_X = np.array([X[li] for li in labeled])
                            U = np.zeros(len(pool))
                            # U = distance to nearest labeled point
                            U = cdist(X[pool], labeled_X, metric='euclidean').min(axis=1)
                            top_m = max(NQ, min(int(len(pool)*0.5), NQ*10))
                            top_m = min(top_m, len(pool))
                            top_idx = np.argsort(U)[-top_m:]
                            top_U = U[top_idx]
                            top_X = X[np.array(pool)[top_idx]]
                            top_U_n = (top_U - top_U.min()) / (top_U.max() - top_U.min() + 1e-10)
                            sel_local = [int(np.argmax(top_U_n))]
                            rem = [i for i in range(len(top_idx)) if i != sel_local[0]]
                            for _ in range(NQ - 1):
                                best_s, best_i = -np.inf, None
                                for idx in rem:
                                    s = top_U_n[idx] + min(
                                        np.linalg.norm(top_X[idx]-top_X[si]) for si in sel_local
                                    ) / (max(np.linalg.norm(top_X[idx]-top_X[si]) for si in sel_local) + 1e-10)
                                    if s > best_s:
                                        best_s, best_i = s, idx
                                if best_i is not None:
                                    sel_local.append(best_i); rem.remove(best_i)
                            selected = np.array(sel_local)

                        else:  # A_random
                            selected = np.random.choice(len(pool), NQ, replace=False)

                        for s in selected:
                            labeled.add(pool[s])

                        # Check if 100% of components are covered
                        if first_full == 9999:
                            labeled_comp = set(y[list(labeled)])
                            if len(labeled_comp) == K:
                                first_full = len(labeled)  # fewer = better

                    # If never reached 100%, use large penalty
                    strat_imps[strategy] = first_full if first_full != 9999 else N_pool + N0

                # Improvement: fewer labels to reach 100% = better
                base_full = strat_imps.get('A_random', N_pool)
                g_full = strat_imps.get('G_hybrid', N_pool)
                j_full = strat_imps.get('J_batch', N_pool)
                c_full = strat_imps.get('C_uncertainty', N_pool)
                e_full = strat_imps.get('E_diversity', N_pool)

                results.append({
                    'K': K, 'pi_config': pi_name,
                    'd_sep_sigma': round(d_sep, 2), 'sigma': sigma,
                    'pi_min': float(pi.min()), 'pi_min_pi_max': float(pi.min()/pi.max()),
                    'silhouette': float(sil),
                    'A_labels100': base_full,
                    'C_labels100': c_full,
                    'E_labels100': e_full,
                    'G_labels100': g_full,
                    'J_labels100': j_full,
                    'G_improvement': (base_full - g_full) / (base_full + 1e-10) * 100,
                    'J_improvement': (base_full - j_full) / (base_full + 1e-10) * 100,
                    'C_improvement': (base_full - c_full) / (base_full + 1e-10) * 100,
                    'hybrid_better': int(g_full < base_full),  # fewer labels = better
                })

        print(f"  K={K} pi={pi_name} done")

res_df = pd.DataFrame(results)
res_df.to_csv(f'{BASE}/results/synthetic_validation.csv', index=False)
print(f"\nSaved {len(res_df)} rows to results/synthetic_validation.csv")

# ─── Analysis ───
print("\n" + "=" * 70)
print("THEOREM VALIDATION")
print("=" * 70)

# P1: silhouette < 0.3 → hybrid better
low_sil = res_df[res_df['silhouette'] < 0.3]
high_sil = res_df[res_df['silhouette'] >= 0.6]
mid_sil = res_df[(res_df['silhouette'] >= 0.3) & (res_df['silhouette'] < 0.6)]

p1_acc = low_sil['hybrid_better'].mean() if len(low_sil) > 0 else 0
p2_acc = 1 - high_sil['hybrid_better'].mean() if len(high_sil) > 0 else 0
print(f"P1 (sil<0.3 → hybrid better):     {p1_acc:.1%} ({len(low_sil)} configs)")
print(f"P2 (sil≥0.6 → random sufficient):  {p2_acc:.1%} ({len(high_sil)} configs)")

# P3: delta monotonically decreasing with silhouette
if len(res_df) > 5:
    rho_p3, p_p3 = spearmanr(res_df['silhouette'], res_df['G_improvement'])
    print(f"P3 (silhouette vs G_improvement):  rho={rho_p3:+.3f} p={p_p3:.4f}")
    sig = '*** p<0.001' if p_p3 < 0.001 else ('** p<0.01' if p_p3 < 0.01 else ('* p<0.05' if p_p3 < 0.05 else 'n.s.'))
    print(f"    {sig}")

# P4: labels saved
labels_saved = res_df['C_labels100'] - res_df['G_labels100']
print(f"P4 (labels saved: C - G):            mean={labels_saved.mean():+.1f}  "
      f"G saves labels vs C: {np.mean(labels_saved > 0):.1%}")

# ─── Figure: silhouettete vs G improvement ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sc = ax1.scatter(res_df['silhouette'], res_df['G_improvement'],
                 c=res_df['K'], cmap='plasma', alpha=0.5, s=12, edgecolors='none')
ax1.axhline(y=0, color='gray', ls='--')
ax1.axvline(x=0.3, color='#E74C3C', ls='--', label='s* = 0.3')
ax1.axvline(x=0.6, color='#27AE60', ls='--', label='s = 0.6')
ax1.set_xlabel('Ground-truth Silhouette Score'); ax1.set_ylabel('G vs Random Improvement (%)')
ax1.set_title(f'Synthetic Validation: {len(res_df)} GMM Configs\n'
              f'Spearman ρ={rho_p3:+.3f} p={p_p3:.4f}')
ax1.legend(); plt.colorbar(sc, ax=ax1, label='K')

# Silhouette bins
bins = np.linspace(0, 1, 11)
res_df['sil_bin'] = pd.cut(res_df['silhouette'], bins=bins, labels=[f'{b:.1f}' for b in bins[:-1]])
bin_stats = res_df.groupby('sil_bin').agg(
    g_imp=('G_improvement','mean'), count=('G_improvement','count')
).reset_index()
ax2.bar(np.arange(len(bin_stats)) - 0.2, bin_stats['g_imp'], 0.35,
        label='G (hybrid)', color='#27AE60')
ax2.axhline(y=0, color='gray', ls='--')
ax2.set_xticks(range(len(bin_stats)))
ax2.set_xticklabels(bin_stats['sil_bin'], rotation=45)
ax2.set_xlabel('Silhouette Score Bin'); ax2.set_ylabel('Improvement over Random (%)')
ax2.set_title('Binned: Silhouette vs G-Hybrid Improvement')
for i, (_, row) in enumerate(bin_stats.iterrows()):
    ax2.text(i, row['g_imp'] + 0.5, f'n={int(row["count"])}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(f'{BASE}/figures/synthetic_theorem_test.png', dpi=200, bbox_inches='tight')
plt.close()
print("\n→ figures/synthetic_theorem_test.png")
print("Done.")
