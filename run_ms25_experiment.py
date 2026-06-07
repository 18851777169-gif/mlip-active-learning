#!/usr/bin/env python3
"""Full MS25 experiment: 6 systems x 5 strategies x active learning.

Loads MACE-labeled MS25 structures, trains SchNet to mimic MACE,
compares acquisition strategies per system.
"""

import os, sys, pickle, time, warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

N_INIT, N_QUERY, N_ITER = 50, 15, 8
HIDDEN, EPOCHS, LR, BATCH = 64, 40, 1e-3, 16
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DATA_DIR = "data/ms25_labeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

from data import MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel
from scipy.spatial.distance import cdist

def to_dev(batch):
    return {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------
def get_embeddings(model, dataset, indices):
    loader = make_dataloader(dataset, indices, BATCH, shuffle=False)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = to_dev(batch)
            _, node_feats = model(batch["z"], batch["pos"], batch["batch"])
            batch_idx = batch["batch"]
            for s in range(batch_idx.max().item() + 1):
                mask = batch_idx == s
                embs.append(node_feats[mask].mean(dim=0).cpu().numpy())
    if embs:
        return np.array(embs)
    return None

def sel_random(pool, nq, models, leb, ds):
    return np.random.RandomState(SEED).choice(pool, nq, replace=False)

def sel_uncertainty(pool, nq, models, leb, ds):
    pl = make_dataloader(ds, pool, BATCH, shuffle=False)
    mlist = list(models.values())
    vars_list = []
    for batch in pl:
        batch = to_dev(batch)
        preds = []
        for m in mlist:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        p = torch.stack(preds, dim=0)
        vars_list.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    scores = np.concatenate(vars_list)
    return pool[np.argsort(scores)[-nq:]]

def sel_diversity(pool, nq, models, leb, ds):
    m0 = list(models.values())[0]
    embs = get_embeddings(m0, ds, pool)
    if embs is None:
        return sel_random(pool, nq, models, leb, ds)
    if leb is not None and leb.shape[0] > 0:
        dists = cdist(embs, leb, metric="cosine").min(axis=1)
    else:
        dists = np.ones(len(pool))
    selected, d = [], dists.copy()
    for _ in range(nq):
        idx = int(np.argmax(d))
        selected.append(idx)
        nd = cdist(embs, embs[[idx]], metric="cosine").ravel()
        d = np.minimum(d, nd)
    return pool[np.array(selected)]

def sel_hybrid_weighted(pool, nq, models, leb, ds, alpha=0.5):
    pl = make_dataloader(ds, pool, BATCH, shuffle=False)
    mlist = list(models.values())
    vars_list = []
    for batch in pl:
        batch = to_dev(batch)
        preds = []
        for m in mlist:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        p = torch.stack(preds, dim=0)
        vars_list.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    u = np.concatenate(vars_list)
    m0 = mlist[0]
    embs = get_embeddings(m0, ds, pool)
    if embs is None:
        return sel_random(pool, nq, models, leb, ds)
    if leb is not None and leb.shape[0] > 0:
        d = cdist(embs, leb, metric="cosine").min(axis=1)
    else:
        d = np.ones(len(pool))
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)
    combined = alpha * norm(u) + (1 - alpha) * norm(d)
    return pool[np.argsort(combined)[-nq:]]

def sel_hybrid_twostage(pool, nq, models, leb, ds, topk=0.3):
    pl = make_dataloader(ds, pool, BATCH, shuffle=False)
    mlist = list(models.values())
    vars_list = []
    for batch in pl:
        batch = to_dev(batch)
        preds = []
        for m in mlist:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        p = torch.stack(preds, dim=0)
        vars_list.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    u = np.concatenate(vars_list)
    n_keep = max(nq * 3, int(len(pool) * topk))
    filtered = pool[np.argsort(u)[-n_keep:]]
    return sel_diversity(filtered, nq, models, leb, ds)

# B: GMM uncertainty — cluster with k-means, score by distance to centroid
def sel_gmm_uncertainty(pool, nq, models, leb, ds):
    from scipy.cluster.vq import kmeans2
    m0 = list(models.values())[0]
    pool_embs = get_embeddings(m0, ds, pool)
    if pool_embs is None or len(pool_embs) < 3:
        return sel_random(pool, nq, models, leb, ds)
    n_clusters = min(5, len(pool_embs)//5)
    centroids, labels = kmeans2(pool_embs, n_clusters, minit='points', seed=42)
    scores = np.array([np.linalg.norm(pool_embs[i] - centroids[labels[i]])
                       for i in range(len(pool_embs))])
    return pool[np.argsort(scores)[-nq:]]

# D: MC-Dropout — multi-pass with model.train() noise
def sel_mc_dropout(pool, nq, models, leb, ds, n_passes=10):
    pl = make_dataloader(ds, pool, BATCH, shuffle=False)
    mlist = list(models.values())
    vars_list = []
    for batch in pl:
        batch = to_dev(batch)
        preds = []
        for m in mlist:
            m.train()
            with torch.no_grad():
                for _ in range(n_passes):
                    e, _ = m(batch["z"], batch["pos"], batch["batch"])
                    preds.append(e)
            m.eval()
        p = torch.stack(preds, dim=0)
        vars_list.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    scores = np.concatenate(vars_list)
    return pool[np.argsort(scores)[-nq:]]

# F: Latent clustering — k-means, pick nearest to each centroid
def sel_latent_clustering(pool, nq, models, leb, ds):
    from scipy.cluster.vq import kmeans2
    m0 = list(models.values())[0]
    pool_embs = get_embeddings(m0, ds, pool)
    if pool_embs is None or len(pool_embs) < nq:
        return sel_random(pool, nq, models, leb, ds)
    k = min(nq, len(pool_embs))
    centroids, _ = kmeans2(pool_embs, k, minit='points', seed=42)
    dists = cdist(centroids, pool_embs, metric="euclidean")
    selected = dists.argmin(axis=1)
    return pool[selected]

STRATEGIES = {
    "A_random": sel_random,
    "B_gmm_uncertainty": sel_gmm_uncertainty,
    "C_ensemble_qbc": sel_uncertainty,
    "D_mc_dropout": sel_mc_dropout,
    "E_diversity": sel_diversity,
    "F_latent_clustering": sel_latent_clustering,
    "G_hybrid_weighted": sel_hybrid_weighted,
    "H_hybrid_twostage": sel_hybrid_twostage,
}

# ---------------------------------------------------------------------------
# Train & Evaluate
# ---------------------------------------------------------------------------
def train_and_eval(dataset, train_idx, val_idx, test_idx):
    model = FallbackModel(hidden_channels=HIDDEN, num_interactions=2).to(DEVICE)
    train_loader = make_dataloader(dataset, train_idx, BATCH, shuffle=True)
    val_loader = make_dataloader(dataset, val_idx, BATCH, shuffle=False)
    test_loader = make_dataloader(dataset, test_idx, BATCH, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    best_val, best_state, patience = float("inf"), None, 0

    for ep in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad()
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                val_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
                val_n += batch["y"].shape[0]
        val_mae = val_sum / val_n
        sched.step(val_mae)
        if val_mae < best_val - 1e-8:
            best_val, best_state = val_mae, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 15:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_sum, test_n = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            test_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
            test_n += batch["y"].shape[0]
    return model, test_sum / test_n

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
systems = sorted([f.stem for f in Path(DATA_DIR).glob("*.pkl")])
# Accept optional system name as second argument
if len(sys.argv) > 2:
    target_system = sys.argv[2]
    systems = [s for s in systems if s == target_system]
    if not systems:
        print(f"System {target_system} not found in {DATA_DIR}")
        sys.exit(1)
print(f"Systems: {systems}")
print(f"Seed: {SEED}")

all_system_results = {}

for sys_name in systems:
    print(f"\n{'#'*60}\n#  {sys_name}\n{'#'*60}")

    # Load MACE-labeled data
    pkl_path = os.path.join(DATA_DIR, f"{sys_name}.pkl")
    with open(pkl_path, "rb") as f:
        structures = pickle.load(f)

    dataset = MaterialDataset(structures)
    energies = [s.info["energy"] for s in structures]
    print(f"  {len(structures)} structures, Energy: [{min(energies):.1f}, {max(energies):.1f}] eV")

    init_idx, pool_idx, test_idx, val_idx = create_splits(
        len(dataset), N_INIT, test_ratio=0.15, val_ratio=0.10, seed=SEED)

    sys_results = {}
    for sname, sfn in STRATEGIES.items():
        print(f"\n  --- {sname} ---")
        labeled = list(init_idx)
        pool = list(pool_idx)
        leb = None
        curve = []

        for it in range(N_ITER + 1):
            models = {}
            for ms in [SEED, SEED + 100]:
                torch.manual_seed(ms)
                model, test_mae = train_and_eval(dataset, labeled, val_idx, test_idx)
                models[ms] = model

            curve.append(test_mae)
            print(f"    Iter {it} | N={len(labeled)} | Test MAE={test_mae:.2f} eV")

            if it >= N_ITER or len(pool) < N_QUERY:
                break

            selected = sfn(np.array(pool), N_QUERY, models, leb, dataset)
            for s in selected:
                if s in pool:
                    pool.remove(int(s))
                    labeled.append(int(s))
            leb = get_embeddings(list(models.values())[0], dataset, labeled)

        sys_results[sname] = curve

    all_system_results[sys_name] = sys_results

# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  FINAL RESULTS — {len(systems)} Systems x {len(STRATEGIES)} Strategies")
print(f"{'='*70}")

import pandas as pd

for sys_name in systems:
    sr = all_system_results[sys_name]
    rand_best = np.min(sr["A_random"])
    print(f"\n--- {sys_name} ---")
    print(f"  {'Strategy':<28} {'Best MAE':>12} {'vs Random':>10}")
    for sname in ["A_random","B_gmm_uncertainty","C_ensemble_qbc","D_mc_dropout","E_diversity","F_latent_clustering","G_hybrid_weighted","H_hybrid_twostage"]:
        c = sr[sname]
        best = np.min(c)
        imp = (rand_best - best) / rand_best * 100
        marker = " <--" if sname.startswith("G_") or sname.startswith("H_") else ""
        print(f"  {sname:<28} {best:>12.2f} {imp:>+9.1f}%{marker}")

    # Save per-system CSV
    df = pd.DataFrame(sr)
    df.to_csv(f"results/ms25_{sys_name}_seed{SEED}.csv", index=False)

# Cross-system summary
print(f"\n{'='*70}")
print(f"  CROSS-SYSTEM SUMMARY")
print(f"{'='*70}")

summary = {}
for sname in ["A_random","B_gmm_uncertainty","C_ensemble_qbc","D_mc_dropout","E_diversity","F_latent_clustering","G_hybrid_weighted","H_hybrid_twostage"]:
    improvements = []
    for sys_name in systems:
        sr = all_system_results[sys_name]
        rand_best = np.min(sr["A_random"])
        s_best = np.min(sr[sname])
        improvements.append((rand_best - s_best) / rand_best * 100)
    avg_imp = np.mean(improvements)
    std_imp = np.std(improvements)
    n_better = sum(1 for x in improvements if x > 0)
    print(f"  {sname:<28} {avg_imp:+5.1f}% +/- {std_imp:.1f}%  "
          f"(better in {n_better}/{len(systems)} systems)")

print(f"\n  Done! Results in results/ms25_*_seed{SEED}.csv")
