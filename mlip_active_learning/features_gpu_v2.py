#!/usr/bin/env python3
"""GPU feature extraction: forces + scipy SOAP-alternative + chemistry + rho."""
import pickle, numpy as np, pandas as pd, os
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans2

S = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
rows = []

for sn in S:
    with open(f"data/ms25_labeled/{sn}.pkl","rb") as f:
        structures = pickle.load(f)

    all_sym = set()
    for s in structures: all_sym.update(s.get_chemical_symbols())
    n_elem = len(all_sym)
    n_atoms_mean = float(np.mean([len(s) for s in structures[:100]]))

    for seed in [42,52,62]:
        np.random.seed(seed)
        n_sample = min(50, len(structures))
        idx = np.random.choice(len(structures), n_sample, replace=False)
        subset = [structures[i] for i in idx]

        # Forces
        forces = np.concatenate([s.info.get('forces',np.zeros((len(s),3))).ravel() for s in subset])
        f_std = float(np.std(forces))
        f_skew = float(pd.Series(forces).skew()) if len(forces)>3 else 0.0
        f_range = float(np.percentile(forces,95)-np.percentile(forces,5))

        # Simple structural descriptors (Coulomb-like matrix distances)
        feats = []
        for s in subset:
            z = np.array(s.get_atomic_numbers())
            pos = s.positions
            cell = s.get_cell()
            # Pairwise distances in periodic cell
            diff = pos[:,None,:] - pos[None,:,:]
            for c in range(3):
                diff[:,:,c] -= cell[c,c] * np.round(diff[:,:,c]/cell[c,c])
            dists = np.sqrt((diff**2).sum(axis=2))
            # Coulomb-like: z_i * z_j / r_ij for r > 0
            mask = dists > 1e-10
            coulomb = np.zeros_like(dists)
            coulomb[mask] = (z[:,None] * z[None,:])[mask] / dists[mask]
            # Upper triangle eigenvalues as fingerprint
            tri = coulomb[np.triu_indices_from(coulomb, k=1)]
            if len(tri) > 0:
                feats.append([float(np.mean(tri)), float(np.std(tri)),
                              float(np.percentile(tri,10)), float(np.percentile(tri,90))])
            else:
                feats.append([0,0,0,0])
        feats = np.array(feats)

        # Silhouette scores using k-means on Coulomb features
        sil_k2, sil_k3, sil_k4 = np.nan, np.nan, np.nan
        pw_mean = float(pdist(feats, 'cosine').mean()) if len(feats)>1 else np.nan
        pw_std = float(pdist(feats, 'cosine').std()) if len(feats)>1 else np.nan
        for k in [2,3,4]:
            if len(feats) >= max(k*5, 10):
                try:
                    centroids, labels = kmeans2(feats, k, minit='points', seed=seed)
                    d_intra = np.array([np.linalg.norm(feats[labels==l]-centroids[l],axis=1).mean() for l in range(k)])
                    d_inter = pdist(centroids, 'euclidean').mean()
                    sil = (d_inter - d_intra.mean()) / max(d_inter, d_intra.mean())
                    if k==2: sil_k2 = float(sil)
                    elif k==3: sil_k3 = float(sil)
                    else: sil_k4 = float(sil)
                except: pass

        # Rho
        rho_mean, rho_std, rho_anti_frac = np.nan, np.nan, np.nan
        rho_file = f"results_rho/rho_stats_{sn}_seed42.csv"
        if os.path.exists(rho_file):
            rho_df = pd.read_csv(rho_file)
            rhos = rho_df['rho'].dropna().values
            if len(rhos)>0:
                rho_mean = float(np.mean(rhos)); rho_std = float(np.std(rhos))
                rho_anti_frac = float(np.mean(rhos < -0.3))

        rows.append({
            'system': sn, 'seed': seed,
            'n_elements': n_elem, 'n_atoms_mean': n_atoms_mean,
            'force_std': f_std, 'force_skew': f_skew, 'force_range': f_range,
            'silhouette_k2': sil_k2, 'silhouette_k3': sil_k3, 'silhouette_k4': sil_k4,
            'pairwise_cosine_mean': pw_mean, 'pairwise_cosine_std': pw_std,
            'rho_mean': rho_mean, 'rho_std': rho_std, 'rho_anti_frac': rho_anti_frac,
        })

pd.DataFrame(rows).to_csv("results/system_features_clean.csv", index=False)
print(f"Saved {len(rows)} rows")
miss = pd.DataFrame(rows).isna().mean()
print(f"Missing: {dict(miss[miss>0])}")
