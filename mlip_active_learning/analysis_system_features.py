#!/usr/bin/env python3
"""提取可事前获取的体系特征（无数据泄漏）

每行 = (体系, 种子) 6×3=18行
特征全部来自: ①50随机结构 SOAP ②ρ分布 ③化学基本信息
不依赖任何 AL 结果
"""
import pickle, numpy as np, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{BASE}/results', exist_ok=True)

S = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation",
     "Zr_oxide_amorphous","liquid_water","zeolite"]
seeds = [42, 52, 62]

print("=" * 60)
print("SYSTEM FEATURES (pre-AL only, no leakage)")
print("=" * 60)

# ─── SOAP ───
try:
    from dscribe.descriptors import SOAP
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist
    HAS_SOAP = True
except ImportError:
    print("WARNING: pip install dscribe")
    HAS_SOAP = False

if HAS_SOAP:
    all_species = set()
    for sn in S:
        with open(f'{BASE}/data/ms25_labeled/{sn}.pkl', 'rb') as f:
            structs = pickle.load(f)
        for s in structs[:100]:
            all_species.update(s.get_atomic_numbers())
    soap = SOAP(species=list(all_species), r_cut=5.0, n_max=8, l_max=6,
                periodic=True, sparse=False)
    print(f"SOAP ready: {len(all_species)} species")

rows = []
for sn in S:
    with open(f'{BASE}/data/ms25_labeled/{sn}.pkl', 'rb') as f:
        structures = pickle.load(f)

    all_species = set()
    for s in structures:
        all_species.update(s.get_chemical_symbols())
    n_elements = len(all_species)
    n_atoms_mean = float(np.mean([len(s) for s in structures[:100]]))

    for seed in seeds:
        np.random.seed(seed)
        n_sample = min(50, len(structures))
        idx = np.random.choice(len(structures), n_sample, replace=False)
        subset = [structures[i] for i in idx]

        # ── 力分布（来自 MACE 标注，非 AL 结果）──
        forces_all = []
        for s in subset:
            if 'forces' in s.info:
                forces_all.extend(s.info['forces'].ravel())
        forces_all = np.array(forces_all)
        f_std = float(np.std(forces_all)) if len(forces_all) > 1 else 0.0
        f_skew = float(pd.Series(forces_all).skew()) if len(forces_all) > 3 else 0.0
        f_range = float(np.percentile(forces_all, 95) - np.percentile(forces_all, 5)) if len(forces_all) > 0 else 0.0

        # ── SOAP 嵌入多模态 ──
        sil_k2, sil_k3, sil_k4 = np.nan, np.nan, np.nan
        pw_mean, pw_std = np.nan, np.nan
        if HAS_SOAP and len(subset) >= 10:
            embs_raw = soap.create(subset)
            # Pad to max length for consistent shape
            max_len = max(len(np.asarray(e).flatten()) for e in embs_raw)
            embs = np.array([np.pad(np.asarray(e).flatten(), (0, max_len-len(np.asarray(e).flatten()))) for e in embs_raw])
            pw = pdist(embs, metric='cosine')
            pw_mean = float(np.mean(pw))
            pw_std = float(np.std(pw))
            for k, store in [(2,'sil_k2'),(3,'sil_k3'),(4,'sil_k4')]:
                if len(embs) >= max(k*5, 10):
                    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
                    labels = km.fit_predict(embs)
                    try:
                        locals()[store] = float(silhouette_score(embs, labels))
                    except:
                        pass

        # ── ρ 分布（来自 results_rho）──
        rho_mean, rho_std, rho_anti_frac = np.nan, np.nan, np.nan
        rho_file = f'{BASE}/results_rho/rho_stats_{sn}_seed{seed}.csv'
        if os.path.exists(rho_file):
            rho_df = pd.read_csv(rho_file)
            rhos = rho_df['rho'].dropna().values
            rhos = rhos[rhos != 0] if len(rhos) > 1 else rhos
            if len(rhos) > 0:
                rho_mean = float(np.mean(rhos))
                rho_std = float(np.std(rhos))
                rho_anti_frac = float(np.mean(rhos < -0.3))

        rows.append({
            'system': sn, 'seed': seed,
            'n_elements': n_elements, 'n_atoms_mean': n_atoms_mean,
            'force_std': f_std, 'force_skew': f_skew, 'force_range': f_range,
            'silhouette_k2': sil_k2, 'silhouette_k3': sil_k3,
            'silhouette_k4': sil_k4,
            'pairwise_cosine_mean': pw_mean, 'pairwise_cosine_std': pw_std,
            'rho_mean': rho_mean, 'rho_std': rho_std,
            'rho_anti_frac': rho_anti_frac,
        })

df = pd.DataFrame(rows)
df.to_csv(f'{BASE}/results/system_features_clean.csv', index=False)
print(f"Saved {len(df)} rows ({len(df['system'].unique())} systems)")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing rate:")
for c in df.columns:
    miss = df[c].isna().mean()
    if miss > 0:
        print(f"  {c}: {miss:.0%}")
print("\nDone.")
