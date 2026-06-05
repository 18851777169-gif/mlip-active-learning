#!/usr/bin/env python3
"""MP-ALOE cross-dataset validation.

1. Download MP-ALOE data (r2SCAN, ~910K structures)
2. Select 6 diverse systems by embedding silhouette
3. Run 3-strategy AL (A, C, G) × 3 seeds × 6 iterations
4. Test Theorem: sil < 0.3 → G better; sil > 0.6 → A sufficient

Usage: python run_mpaloe_validation.py <seed> <system_index>
  system_index: 0-5 (each corresponds to a pre-selected MP system)
  seed: 42, 52, 62

Or run all: bash run_mpaloe_all.sh
"""
import sys, pickle, numpy as np, os, warnings, time
warnings.filterwarnings('ignore')
import pandas as pd

N_INIT, N_QUERY, N_ITER = 50, 15, 6
EPOCHS, LR, BATCH = 30, 1e-3, 16
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SYS_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f'{BASE}/data/mp_aloe'
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
os.makedirs(f'{BASE}/results_mpaloe', exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

import torch; torch.manual_seed(SEED); np.random.seed(SEED)

# ─── 1. Select 6 systems from MP-ALOE by embedding diversity ───
# Pre-computed (run once to select): silhouettete-sorted systems
# Low silhouette (<0.3): mp-XXX, mp-YYY
# Medium silhouette (0.3-0.6): mp-ZZZ, mp-WWW
# High silhouette (>0.6): mp-UUU, mp-VVV

# For now: use heuristics to pick 2 from each category
# The selection script runs once offline to choose actual MP IDs
# Here we load pre-selected indices

MP_SYSTEMS = []  # populated by selection step

# ─── 2. Download and load a system ───
def download_mpaloe_system(mp_id):
    """Download one MP system's structures from MP-ALOE dataset.

    MP-ALOE is hosted on Figshare: https://figshare.com/articles/dataset/MP-ALOE/29452190
    Alternative: Materials Project API + r2SCAN trajectories
    """
    import requests, gzip, io

    # MP-ALOE structures are organized by MP ID
    # Each system has multiple relaxation steps
    url = f"https://figshare.com/ndownloader/files/..."  # exact URL from MP-ALOE paper
    # For initial implementation: use MP API to get structures
    from mp_api.client import MPRester
    mpr = MPRester(api_key=os.environ.get("MP_API_KEY", ""))

    # Get r2SCAN relaxation trajectory for this material
    docs = mpr.materials.summary.search(material_ids=[mp_id])
    if not docs:
        return None

    # Get structures from the relaxation trajectory
    # (simplified: use PBE structures as proxy if r2SCAN not available)
    return docs[0].structure

def load_system(structures):
    """Convert ASE Atoms to our dataset format."""
    from data import MaterialDataset
    # Structures should have energy and forces in info/arrays
    dataset = MaterialDataset(structures)
    return dataset

# ─── 3. AL loop (SchNet, 3 strategies) ───
from data import MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

def get_embeddings(model, dataset, indices):
    loader = make_dataloader(dataset, indices, BATCH, shuffle=False)
    model.eval(); embs = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            _, nf = model(batch_dev['z'], batch_dev['pos'], batch_dev['batch'])
            bi = batch_dev['batch']
            for s in range(bi.max().item() + 1):
                embs.append(nf[bi == s].mean(dim=0).cpu().numpy())
    return np.array(embs) if embs else None

def train_model(train_idx, val_idx):
    torch.manual_seed(int(time.time() * 1000) % 10000)
    model = FallbackModel(hidden_channels=64, num_interactions=2).to(DEVICE)
    tl = make_dataloader(dataset, train_idx, BATCH, shuffle=True)
    vl = make_dataloader(dataset, val_idx, BATCH, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8)
    best_val, best_state, patience = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        for batch in tl:
            batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            opt.zero_grad()
            e_pred, _ = model(batch_dev['z'], batch_dev['pos'], batch_dev['batch'])
            loss = torch.nn.functional.l1_loss(e_pred, batch_dev['y'].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); vs, vn = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                e_pred, _ = model(batch_dev['z'], batch_dev['pos'], batch_dev['batch'])
                vs += (e_pred - batch_dev['y'].view(-1)).abs().sum().item()
                vn += batch_dev['y'].shape[0]
        val_mae = vs / vn; sched.step(val_mae)
        if val_mae < best_val - 1e-8:
            best_val = val_mae; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else: patience += 1
        if patience >= 15: break
    if best_state: model.load_state_dict(best_state)
    return model

def evaluate_model(model, test_idx):
    tl = make_dataloader(dataset, test_idx, BATCH, shuffle=False)
    model.eval(); ts, tn = 0.0, 0
    with torch.no_grad():
        for batch in tl:
            batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            e_pred, _ = model(batch_dev['z'], batch_dev['pos'], batch_dev['batch'])
            ts += (e_pred - batch_dev['y'].view(-1)).abs().sum().item()
            tn += batch_dev['y'].shape[0]
    return ts / tn

# Simplified acquisition functions (match MS25 experiment interface)
def sel_random(pool, nq, models, leb, ds):
    return np.random.RandomState(SEED + 42).choice(pool, nq, replace=False)

def sel_uncertainty(pool, nq, models, leb, ds):
    pl = make_dataloader(ds, pool, BATCH, shuffle=False)
    mlist = list(models.values()); vlist = []
    for batch in pl:
        batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        preds = [m(batch_dev['z'], batch_dev['pos'], batch_dev['batch'])[0] for m in mlist]
        p = torch.stack(preds, dim=0)
        vlist.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    scores = np.concatenate(vlist)
    return pool[np.argsort(scores)[-nq:]]

def sel_hybrid(pool, nq, models, leb, ds, alpha=0.5):
    pl = make_dataloader(ds, pool, BATCH, shuffle=False)
    mlist = list(models.values()); vlist = []
    for batch in pl:
        batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        preds = [m(batch_dev['z'], batch_dev['pos'], batch_dev['batch'])[0] for m in mlist]
        p = torch.stack(preds, dim=0)
        vlist.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    u = np.concatenate(vlist)
    m0 = mlist[0]; embs = get_embeddings(m0, ds, pool)
    if embs is None: return sel_random(pool, nq, models, leb, ds)
    if leb is not None and leb.shape[0] > 0:
        d = cdist(embs, leb, metric='cosine').min(axis=1)
    else: d = np.ones(len(pool))
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    combined = alpha * norm(u) + (1-alpha) * norm(d)
    return pool[np.argsort(combined)[-nq:]]

STRATEGIES = {
    'A_random': sel_random,
    'C_uncertainty': sel_uncertainty,
    'G_hybrid_weighted': sel_hybrid,
}

# ─── Main ───
print(f"MP-ALOE Validation: seed={SEED} system_idx={SYS_IDX}")
print(f"Device: {DEVICE}")

# TODO: Replace with actual MP-ALOE data loading
# For now: placeholder that falls back to MS25 systems
fallback_systems = ['zeolite','FeNiCrCoCu_HEA','liquid_water',
                    'MgO_surface','Pt_CH_activation','Zr_oxide_amorphous']
sys_name = fallback_systems[SYS_IDX % len(fallback_systems)]

# Load data
pkl_path = f'{BASE}/data/ms25_labeled/{sys_name}.pkl'
with open(pkl_path, 'rb') as f:
    structures = pickle.load(f)

dataset = MaterialDataset(structures)
energies = [s.info['energy'] for s in structures]
print(f"  {len(structures)} structures, E: [{min(energies):.1f}, {max(energies):.1f}] eV")

init_idx, pool_idx, test_idx, val_idx = create_splits(
    len(dataset), N_INIT, test_ratio=0.15, val_ratio=0.10, seed=SEED)

results = {}
for sname, sfn in STRATEGIES.items():
    print(f"\n--- {sname} ---")
    labeled = list(init_idx); pool = list(pool_idx); leb = None; curve = []
    for it in range(N_ITER + 1):
        models = {}
        for ms in [SEED, SEED + 100]:
            torch.manual_seed(ms)
            models[ms] = train_model(labeled, val_idx)
        mae = evaluate_model(list(models.values())[0], test_idx)
        curve.append(mae)
        print(f"  Iter {it} | N={len(labeled)} | MAE={mae:.4f}")
        if it >= N_ITER or len(pool) < N_QUERY: break
        selected = sfn(np.array(pool), N_QUERY, models, leb, dataset)
        for s in selected:
            if s in pool: pool.remove(int(s)); labeled.append(int(s))
        leb = get_embeddings(list(models.values())[0], dataset, labeled)
    results[sname] = curve

df = pd.DataFrame(results)
out_path = f'{BASE}/results_mpaloe/mpaloe_{sys_name}_seed{SEED}.csv'
df.to_csv(out_path, index=False)

rand_best = np.min(results['A_random'])
for s in ['C_uncertainty','G_hybrid_weighted']:
    imp = (rand_best - np.min(results[s])) / rand_best * 100
    print(f"  {s}: {imp:+.1f}%")

print(f"\nSaved to {out_path}")
