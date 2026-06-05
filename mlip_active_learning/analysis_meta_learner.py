#!/usr/bin/env python3
"""Meta-learner v2: 36-row LOSO, only pre-AL features, no data leakage

Features: system features (SOAP + rho + chemistry) — all computable pre-AL
Labels: G_hybrid_weighted better than A_random? — from AL results
LOSO: grouped by system, each fold tests 3seeds x 2arch = 6 rows
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import binomtest

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{BASE}/figures', exist_ok=True)

# ─── 1. Load features ───
feat_df = pd.read_csv(f'{BASE}/results/system_features_clean.csv')
print(f"Features: {len(feat_df)} rows, {len(feat_df.system.unique())} systems")

# ─── 2. Build labels ───
S = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation",
     "Zr_oxide_amorphous","liquid_water","zeolite"]
seeds = [42, 52, 62]

rows = []
for arch, ddir, prefix in [
    ('SchNet', 'results_schnet_ms25', 'ms25_9strat'),
    ('MACE',  'results_mace_al',   'mace_al')]:
    for sn in S:
        for seed in seeds:
            f = f'{BASE}/{ddir}/{prefix}_{sn}_seed{seed}.csv'
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            ar = df['A_random'].iloc[-1]
            g = df.get('G_hybrid_weighted', pd.Series([np.nan])).iloc[-1]
            j = df.get('J_aud_batch', pd.Series([np.nan])).iloc[-1]
            row = {
                'system': sn, 'seed': seed, 'architecture': arch,
                'G_better': int(not np.isnan(g) and g < ar),
                'J_better': int(not np.isnan(j) and j < ar),
                'any_better': int(any(
                    df[c].iloc[-1] < ar for c in df.columns
                    if c != 'A_random' and not pd.isna(df[c].iloc[-1])
                )),
            }
            rows.append(row)

label_df = pd.DataFrame(rows)
print(f"Labels: {len(label_df)} rows")
print(f"  G_better: {label_df['G_better'].sum()}/{len(label_df)} ({label_df['G_better'].mean():.0%})")
print(f"  J_better: {label_df['J_better'].sum()}/{len(label_df)} ({label_df['J_better'].mean():.0%})")

# ─── 3. Merge ───
data = label_df.merge(feat_df, on=['system', 'seed'], how='inner')
print(f"Merged: {len(data)} rows\n")

# ─── 4. Valid feature columns (all pre-AL) ───
feat_cols = [
    'n_elements', 'n_atoms_mean',
    'force_std', 'force_skew', 'force_range',
    'silhouette_k2', 'silhouette_k3', 'silhouette_k4',
    'pairwise_cosine_mean', 'pairwise_cosine_std',
    'rho_mean', 'rho_std', 'rho_anti_frac',
]
feat_cols = [c for c in feat_cols if c in data.columns and data[c].notna().mean() > 0.5]
print(f"Features ({len(feat_cols)}): {feat_cols}")

X_all = data[feat_cols].fillna(data[feat_cols].median())
groups = data['system'].values

# ─── 5. LOSO ───
def run_loso(X, y, groups, label_name):
    logo = LeaveOneGroupOut()
    correct, total = 0, 0
    preds_all = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        s = StandardScaler()
        clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                                      random_state=42, class_weight='balanced')
        clf.fit(s.fit_transform(X_tr), y_tr)
        y_pred = clf.predict(s.transform(X_te))
        for sys_name, true, pred in zip(groups[test_idx], y_te, y_pred):
            preds_all.append({'system': sys_name, 'true': int(true), 'pred': int(pred)})
            total += 1
            if true == pred:
                correct += 1

    acc = correct / total if total > 0 else 0
    p = binomtest(correct, total, p=0.5, alternative='greater').pvalue if total > 0 else 1
    sig = '*** SIGNIFICANT' if p < 0.05 else '(n.s.)'

    print(f"  {label_name}: {correct}/{total} = {acc:.1%}  p={p:.4f} {sig}")

    s_full = StandardScaler()
    clf_full = RandomForestClassifier(n_estimators=300, max_depth=4,
                                       random_state=42, class_weight='balanced')
    clf_full.fit(s_full.fit_transform(X), y)
    imps = sorted(zip(feat_cols, clf_full.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    print(f"    Top: {', '.join(f'{f}={v:.3f}' for f,v in imps[:4])}")

    return acc, p, preds_all, imps

print("=" * 60)
print("LOSO CROSS-VALIDATION (grouped by system)")
print("=" * 60)
acc_g, p_g, preds_g, imps_g = run_loso(X_all, data['G_better'].values, groups,
                                        'G > Random')
acc_j, p_j, preds_j, imps_j = run_loso(X_all, data['J_better'].values, groups,
                                        'J > Random')

# ─── 6. Cross-architecture ───
print("\n" + "=" * 60)
print("CROSS-ARCHITECTURE")
print("=" * 60)
for train_arch in ['SchNet', 'MACE']:
    test_arch = 'MACE' if train_arch == 'SchNet' else 'SchNet'
    train_mask = data['architecture'] == train_arch
    test_mask = data['architecture'] == test_arch
    if train_mask.sum() < 5 or test_mask.sum() < 3:
        continue
    X_tr = X_all[train_mask]; y_tr = data.loc[train_mask, 'G_better'].values
    X_te = X_all[test_mask]; y_te = data.loc[test_mask, 'G_better'].values
    s = StandardScaler()
    clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                                  random_state=42, class_weight='balanced')
    clf.fit(s.fit_transform(X_tr), y_tr)
    y_pred = clf.predict(s.transform(X_te))
    acc = (y_pred == y_te).mean()
    print(f"  {train_arch} → {test_arch}: {acc:.1%} (n={len(y_te)})")

# ─── 7. Figure ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
top = imps_g[:6]
ax.barh([f[0] for f in top[::-1]], [f[1] for f in top[::-1]], color='#3498DB')
ax.set_xlabel('Importance'); ax.set_title('G > Random: Feature Importance')

ax = axes[1]
sys_acc = {}
for p in preds_g:
    sys_acc.setdefault(p['system'], {'correct': 0, 'total': 0})
    sys_acc[p['system']]['correct'] += int(p['true'] == p['pred'])
    sys_acc[p['system']]['total'] += 1
sorted_sys = sorted(sys_acc.items(), key=lambda x: x[1]['correct'] / x[1]['total'])
labels = [s[0][:20] for s in sorted_sys]
accs = [s[1]['correct'] / s[1]['total'] for s in sorted_sys]
colors = ['#27AE60' if a > 0.5 else '#E74C3C' for a in accs]
ax.barh(labels, accs, color=colors)
ax.axvline(x=0.5, color='gray', ls='--', label='Random baseline')
ax.axvline(x=acc_g, color='#27AE60', lw=2, label=f'Overall ({acc_g:.0%})')
ax.set_xlabel('Accuracy'); ax.set_xlim(0, 1)
ax.set_title(f'G > Random: LOSO (p={p_g:.4f})')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{BASE}/figures/meta_learner.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n→ figures/meta_learner.png")

# ─── 8. Results CSV ───
pd.DataFrame([
    {'metric': 'LOSO_task', 'value': 'G_hybrid_weighted > Random'},
    {'metric': 'LOSO_accuracy', 'value': f'{acc_g:.1%}'},
    {'metric': 'Binomial_p', 'value': f'{p_g:.4f}'},
    {'metric': 'Binomial_significant', 'value': p_g < 0.05},
    {'metric': 'N_rows', 'value': len(data)},
    {'metric': 'N_systems', 'value': len(S)},
    {'metric': 'N_features', 'value': len(feat_cols)},
    {'metric': 'Leakage_check', 'value': 'All features computable pre-AL'},
    {'metric': 'Top_feature_1', 'value': f'{imps_g[0][0]}={imps_g[0][1]:.4f}'},
    {'metric': 'Top_feature_2', 'value': f'{imps_g[1][0]}={imps_g[1][1]:.4f}'},
]).to_csv(f'{BASE}/results/meta_learner_results.csv', index=False)
print("→ results/meta_learner_results.csv")
print("\nDone.")
