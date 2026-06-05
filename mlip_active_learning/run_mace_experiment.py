#!/usr/bin/env python3
"""Active learning experiment on MACE-labeled data.

Loads MACE-generated structures, trains SchNet to mimic MACE predictions,
compares 5 acquisition strategies.
"""

import sys, time, pickle, os, warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_PATH = "data/mace_labeled.pkl"
N_INIT, N_QUERY, N_ITER = 50, 15, 8
N_TOTAL = 800
HIDDEN, EPOCHS, LR, BATCH = 64, 40, 1e-3, 16
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load MACE-labeled data
print(f"Loading MACE data from {DATA_PATH}...")
with open(DATA_PATH, "rb") as f:
    all_structs = pickle.load(f)
print(f"Loaded {len(all_structs)} structures")

# Build dataset
from data import MaterialDataset, make_dataloader, create_splits
dataset = MaterialDataset(all_structs)
init_idx, pool_idx, test_idx, val_idx = create_splits(
    len(dataset), N_INIT, test_ratio=0.15, val_ratio=0.10, seed=SEED)
print(f"Splits: init={len(init_idx)}, pool={len(pool_idx)}, "
      f"test={len(test_idx)}, val={len(val_idx)}")

# Show stats
energies = [s.info["energy"] for s in all_structs]
print(f"Energy range: [{min(energies):.1f}, {max(energies):.1f}] eV, "
      f"mean={np.mean(energies):.1f} +/- {np.std(energies):.1f}")

from model_fallback import FallbackModel

test_loader = make_dataloader(dataset, test_idx, BATCH, shuffle=False)

def train_model(train_indices):
    model = FallbackModel(hidden_channels=HIDDEN, num_interactions=2)
    train_loader = make_dataloader(dataset, train_indices, BATCH, shuffle=True)
    val_loader = make_dataloader(dataset, val_idx, BATCH, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    best_val, best_state, patience = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        for batch in train_loader:
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
                e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                val_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
                val_n += batch["y"].shape[0]
        val_mae = val_sum / val_n
        sched.step(val_mae)
        if val_mae < best_val - 1e-8:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            test_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
            test_n += batch["y"].shape[0]
    return model, test_sum / test_n

def get_embeddings(model, indices):
    loader = make_dataloader(dataset, indices, BATCH, shuffle=False)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            _, node_feats = model(batch["z"], batch["pos"], batch["batch"])
            batch_idx = batch["batch"]
            for s in range(batch_idx.max().item() + 1):
                mask = batch_idx == s
                embs.append(node_feats[mask].mean(dim=0).cpu().numpy())
    return np.array(embs)

# Acquisition functions
from scipy.spatial.distance import cdist

def sel_random(pool, nq, models, leb):
    return np.random.RandomState(SEED).choice(pool, nq, replace=False)

def sel_uncertainty(pool, nq, models, leb):
    pl = make_dataloader(dataset, pool, BATCH, shuffle=False)
    vars_ = []
    mlist = list(models.values())
    for batch in pl:
        preds = []
        for m in mlist:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        p = torch.stack(preds, dim=0)
        vars_.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    top = np.argsort(np.concatenate(vars_))[-nq:]
    return pool[top]

def sel_diversity(pool, nq, models, leb):
    m0 = list(models.values())[0]
    embs = get_embeddings(m0, pool)
    if leb is not None and leb.shape[0] > 0:
        dists = cdist(embs, leb, metric="cosine").min(axis=1)
    else:
        dists = np.ones(len(pool))
    # FPS
    selected, d = [], dists.copy()
    for _ in range(nq):
        idx = int(np.argmax(d))
        selected.append(idx)
        nd = cdist(embs, embs[[idx]], metric="cosine").ravel()
        d = np.minimum(d, nd)
    return pool[np.array(selected)]

def sel_hybrid_weighted(pool, nq, models, leb, alpha=0.5):
    pl = make_dataloader(dataset, pool, BATCH, shuffle=False)
    mlist = list(models.values())
    vars_ = []
    for batch in pl:
        preds = []
        for m in mlist:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        p = torch.stack(preds, dim=0)
        vars_.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    u = np.concatenate(vars_)
    m0 = mlist[0]
    embs = get_embeddings(m0, pool)
    if leb is not None and leb.shape[0] > 0:
        d = cdist(embs, leb, metric="cosine").min(axis=1)
    else:
        d = np.ones(len(pool))

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)
    combined = alpha * norm(u) + (1 - alpha) * norm(d)
    top = np.argsort(combined)[-nq:]
    return pool[top]

def sel_hybrid_twostage(pool, nq, models, leb, topk=0.3):
    pl = make_dataloader(dataset, pool, BATCH, shuffle=False)
    mlist = list(models.values())
    vars_ = []
    for batch in pl:
        preds = []
        for m in mlist:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        p = torch.stack(preds, dim=0)
        vars_.append(p.std(dim=0).cpu().numpy() if p.shape[0] > 1 else np.zeros(p.shape[1]))
    u = np.concatenate(vars_)
    n_keep = max(nq * 3, int(len(pool) * topk))
    top_unc = np.argsort(u)[-n_keep:]
    filtered = pool[top_unc]
    m0 = mlist[0]
    embs = get_embeddings(m0, filtered)
    if leb is not None and leb.shape[0] > 0:
        dists = cdist(embs, leb, metric="cosine").min(axis=1)
    else:
        dists = np.ones(len(filtered))
    selected, d = [], dists.copy()
    for _ in range(min(nq, len(filtered))):
        idx = int(np.argmax(d))
        selected.append(idx)
        nd = cdist(embs, embs[[idx]], metric="cosine").ravel()
        d = np.minimum(d, nd)
    return filtered[np.array(selected)]

STRATEGIES = {
    "A_random": sel_random,
    "C_uncertainty": sel_uncertainty,
    "E_diversity": sel_diversity,
    "G_hybrid_weighted": sel_hybrid_weighted,
    "H_hybrid_twostage": sel_hybrid_twostage,
}

print(f"\n{'='*60}")
print(f"  MACE-LABELED EXPERIMENT (seed={SEED})")
print(f"{'='*60}")

results = {}
for sname, sfn in STRATEGIES.items():
    print(f"\n--- {sname} ---")
    labeled = list(init_idx)
    pool = list(pool_idx)
    leb = None
    curve = []

    for it in range(N_ITER + 1):
        models = {}
        for ms in [SEED, SEED + 100]:
            torch.manual_seed(ms)
            model, test_mae = train_model(labeled)
            models[ms] = model

        curve.append(test_mae)
        n_lab = len(labeled)
        print(f"  Iter {it} | N={n_lab} | Test MAE={test_mae:.4f} eV")

        if it >= N_ITER or len(pool) < N_QUERY:
            break

        selected = sfn(np.array(pool), N_QUERY, models, leb)
        for s in selected:
            if s in pool:
                pool.remove(int(s))
                labeled.append(int(s))
        leb = get_embeddings(list(models.values())[0], labeled)

    results[sname] = curve

# Report
print(f"\n{'='*60}")
print(f"  RESULTS (MACE-labeled data, {len(all_structs)} structs)")
print(f"{'='*60}")
print(f"{'Strategy':<30} {'Final MAE (eV)':>16} {'Best MAE':>16}")
print("-" * 62)
for s in ["A_random", "C_uncertainty", "E_diversity", "G_hybrid_weighted", "H_hybrid_twostage"]:
    c = results[s]
    print(f"  {s:<28} {c[-1]:>16.4f} {np.min(c):>16.4f}")

# Save
import pandas as pd
df = pd.DataFrame(results)
df.index.name = "iteration"
out_csv = f"results/mace_experiment_seed{SEED}.csv"
df.to_csv(out_csv)
print(f"\nSaved: {out_csv}")

# Efficiency
rand_best = np.min(results["A_random"])
target = rand_best * 0.95
print(f"\nRandom best: {rand_best:.2f}, Target (95%): {target:.2f}")
for s in results:
    if s == "A_random":
        continue
    c = results[s]
    imp = (rand_best - np.min(c)) / rand_best * 100
    print(f"  {s}: best={np.min(c):.2f}, vs Random: {imp:+.1f}%")
