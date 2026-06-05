#!/usr/bin/env python3
"""MACE Embedding Diagnostic: compute silhouette from MACE-MP-0 features.

No fine-tuning, no DFT. Forward pass only (~5 min per system).
Validates P1-P3 using model-native embeddings instead of SOAP proxy.
"""
import pickle, numpy as np, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{BASE}/results', exist_ok=True)

# ─── Load MACE-MP-0 ───
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"

print(f"Device: {DEVICE}")
print(f"Loading MACE-MP-0 from {MODEL_PATH}...")

try:
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_path=MODEL_PATH, device=DEVICE, default_dtype="float32")
    mace_model = calc.models[0]
    print("MACE-MP-0 loaded.")
except Exception as e:
    print(f"MACE loading failed: {e}")
    print("Trying fallback path...")
    import sys
    sys.exit(1)

def get_mace_embedding(atoms_list):
    """Extract node features from MACE-MP-0, return per-structure mean-pooled embeddings."""
    mace_model.eval()
    embeddings = []
    with torch.no_grad():
        for atoms in atoms_list:
            batch = calc._atoms_to_batch(atoms)
            batch = calc._clone_batch(batch)
            batch_dict = batch.to_dict()
            # Forward pass, return node features
            output = mace_model(batch_dict, training=False, compute_force=False)
            node_feats = output.get('node_feats', None)
            if node_feats is None:
                embeddings.append(np.zeros(1))
            else:
                embeddings.append(node_feats.mean(dim=0).cpu().numpy())
    return np.array(embeddings)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_silhouette(embs, seed=42):
    """Compute silhouette scores for k=2,3,4."""
    results = {}
    for k in [2, 3, 4]:
        if len(embs) >= k * 5:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(embs)
            try:
                results[f'sil_k{k}'] = float(silhouette_score(embs, labels))
            except:
                results[f'sil_k{k}'] = np.nan
    return results

# ─── Process MS25 systems ───
MS25 = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation",
        "Zr_oxide_amorphous","liquid_water","zeolite"]
MS25_LABELS = ["HEA","MgO(100)","Pt(111)","ZrO2(am)","H2O(l)","Zeolite"]

print("\n" + "="*70)
print("MS25: MACE Embedding Silhouette")
print("="*70)

results = []
for sn, label in zip(MS25, MS25_LABELS):
    pkl_path = f'{BASE}/data/ms25_labeled/{sn}.pkl'
    if not os.path.exists(pkl_path):
        print(f"  {sn}: NOT FOUND")
        continue

    with open(pkl_path, 'rb') as f:
        structures = pickle.load(f)

    # Multiple seeds
    for seed in [42, 52, 62]:
        np.random.seed(seed)
        n_sample = min(50, len(structures))
        idx = np.random.choice(len(structures), n_sample, replace=False)
        subset = [structures[i] for i in idx]

        embs = get_mace_embedding(subset)
        sil = compute_silhouette(embs, seed)

        n_elem = len(set(structures[0].get_chemical_symbols()))
        results.append({
            'system': sn, 'label': label, 'seed': seed,
            'dataset': 'MS25',
            'n_elements': n_elem, 'n_atoms': len(structures[0]),
            **sil
        })

    avg_sil = np.mean([r['sil_k3'] for r in results if r['system']==sn and not np.isnan(r.get('sil_k3', np.nan))])
    print(f"  {sn:<30s}: sil_k3 = {avg_sil:.4f}")

# ─── Process MP-ALOE systems ───
mp_aloe_dir = f'{BASE}/data/mp_aloe'
mp_systems = []
if os.path.exists(mp_aloe_dir):
    for fname in os.listdir(mp_aloe_dir):
        if fname.endswith('.pkl'):
            mp_systems.append(fname.replace('.pkl',''))

print(f"\nMP-ALOE systems: {mp_systems}")

for sn in mp_systems:
    pkl_path = f'{mp_aloe_dir}/{sn}.pkl'
    try:
        with open(pkl_path, 'rb') as f:
            structures = pickle.load(f)
    except:
        print(f"  {sn}: load failed")
        continue

    for seed in [42, 52, 62]:
        np.random.seed(seed)
        n_sample = min(50, len(structures))
        idx = np.random.choice(len(structures), n_sample, replace=False)
        subset = [structures[i] for i in idx]

        embs = get_mace_embedding(subset)
        sil = compute_silhouette(embs, seed)

        n_elem = len(set(structures[0].get_chemical_symbols()))
        results.append({
            'system': sn, 'label': sn, 'seed': seed,
            'dataset': 'MP-ALOE',
            'n_elements': n_elem, 'n_atoms': len(structures[0]),
            **sil
        })

    avg_sil = np.mean([r['sil_k3'] for r in results if r['system']==sn and not np.isnan(r.get('sil_k3', np.nan))])
    print(f"  {sn:<30s}: sil_k3 = {avg_sil:.4f}")

# ─── Save ───
df = pd.DataFrame(results)
df.to_csv(f'{BASE}/results/mace_embedding_silhouette.csv', index=False)
print(f"\nSaved {len(df)} rows to results/mace_embedding_silhouette.csv")

# ─── Summary ───
summary = df.groupby(['system','dataset']).agg(
    sil_k3_mean=('sil_k3','mean'),
    sil_k3_std=('sil_k3','std'),
).reset_index().sort_values('sil_k3_mean')

print("\n" + "="*70)
print("MACE EMBEDDING SILHOUETTE SUMMARY")
print("="*70)
print(f"{'System':<30s} {'Dataset':<10s} {'sil_k3':>8s}")
print("-"*50)
for _, row in summary.iterrows():
    cat = 'HIGH' if row['sil_k3_mean'] > 0.6 else ('LOW' if row['sil_k3_mean'] < 0.3 else 'MID')
    print(f"{row['system']:<30s} {row['dataset']:<10s} {row['sil_k3_mean']:>8.4f} [{cat}]")

print("\nDone.")
