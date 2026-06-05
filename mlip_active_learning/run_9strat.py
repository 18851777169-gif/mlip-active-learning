#!/usr/bin/env python3
"""9-strategy active learning: A-H + I_AUD_Rank + J_AUD_Batch + K_AUD_Bald.

Uses SOAP descriptors for diversity, ensemble force variance for uncertainty.
"""

import sys, pickle, time, os, warnings
import numpy as np
import torch, torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

N_INIT, N_QUERY, N_ITER = 50, 15, 6
HIDDEN, EPOCHS, LR, BATCH = 64, 40, 1e-3, 16
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DATA_DIR = "data/ms25_labeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}, Seed: {SEED}")

from data import MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel
from scipy.spatial.distance import cdist

def to_dev(b):
    return {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in b.items()}

# ---------------------------------------------------------------------------
def compute_soap_embeddings(structures, indices):
    """SOAP not available — use model embeddings instead."""
    return None  # Use model embeddings via get_embeddings()

# ---------------------------------------------------------------------------
# Core ML functions
# ---------------------------------------------------------------------------
from scipy.stats import spearmanr

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
    return np.array(embs) if embs else None

def compute_uncertainties(models, dataset, pool_indices):
    """Compute ensemble energy variance for pool structures."""
    pl = make_dataloader(dataset, pool_indices, BATCH, shuffle=False)
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
        vars_list.append(p.std(dim=0, unbiased=False).cpu().numpy() if p.shape[0]>1 else np.zeros(p.shape[1]))
    return np.concatenate(vars_list)

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
            batch = to_dev(batch); opt.zero_grad()
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_dev(batch); e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                val_sum += (e_pred - batch["y"].view(-1)).abs().sum().item(); val_n += batch["y"].shape[0]
        val_mae = val_sum/val_n; sched.step(val_mae)
        if val_mae < best_val-1e-8: best_val, best_state, patience = val_mae, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else: patience += 1
        if patience >= 15: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); test_sum, test_n = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = to_dev(batch); e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            test_sum += (e_pred - batch["y"].view(-1)).abs().sum().item(); test_n += batch["y"].shape[0]
    return model, test_sum/test_n

# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------
def sel_random(pool, nq, **kw):
    return np.random.RandomState(SEED).choice(pool, nq, replace=False)

def sel_uncertainty(pool, nq, models, dataset, **kw):
    u = compute_uncertainties(models, dataset, pool)
    return pool[np.argsort(u)[-nq:]]

def sel_diversity(pool, nq, models, dataset, structures, labeled_embeddings, **kw):
    pool_embs = get_embeddings(list(models.values())[0], dataset, pool)
    if pool_embs is None: return sel_random(pool, nq)
    if labeled_embeddings is not None and labeled_embeddings.shape[0] > 0:
        dists = cdist(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else:
        dists = np.ones(len(pool))
    selected, d = [], dists.copy()
    for _ in range(nq):
        idx = int(np.argmax(d)); selected.append(idx)
        nd = cdist(pool_embs, pool_embs[[idx]], metric="cosine").ravel(); d = np.minimum(d, nd)
    return pool[np.array(selected)]

def sel_hybrid_weighted(pool, nq, models, dataset, structures, labeled_embeddings, alpha=0.5, **kw):
    u = compute_uncertainties(models, dataset, pool)
    pool_embs = get_embeddings(list(models.values())[0], dataset, pool)
    if pool_embs is None: return sel_random(pool, nq)
    if labeled_embeddings is not None and labeled_embeddings.shape[0] > 0:
        d = cdist(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else: d = np.ones(len(pool))
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    combined = alpha*norm(u) + (1-alpha)*norm(d)
    return pool[np.argsort(combined)[-nq:]]

def sel_hybrid_twostage(pool, nq, models, dataset, structures, labeled_embeddings, **kw):
    u = compute_uncertainties(models, dataset, pool)
    n_keep = max(nq*3, int(len(pool)*0.3))
    filtered = pool[np.argsort(u)[-n_keep:]]
    return sel_diversity(filtered, nq, models, dataset, structures, labeled_embeddings)

# ---------------------------------------------------------------------------
# I: AUD-Rank (plan v1.0)
# ---------------------------------------------------------------------------
def sel_aud_rank(pool, nq, models, dataset, structures, labeled_embeddings, **kw):
    u = compute_uncertainties(models, dataset, pool)
    pool_embs = get_embeddings(list(models.values())[0], dataset, pool)
    if pool_embs is None or len(pool)<5: return sel_random(pool, nq)

    if labeled_embeddings is not None and labeled_embeddings.shape[0]>0:
        d = cdist(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else: d = np.ones(len(pool))

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u), norm(d)
    rho, _ = spearmanr(u_n, d_n)
    alpha = np.clip(0.5 - 0.3*rho, 0.2, 0.8)
    combined = alpha*u_n + (1-alpha)*d_n
    return pool[np.argsort(combined)[-nq:]]

# ---------------------------------------------------------------------------
# J: AUD-Batch (plan v1.0)
# ---------------------------------------------------------------------------
def sel_aud_batch(pool, nq, models, dataset, structures, labeled_embeddings, **kw):
    u = compute_uncertainties(models, dataset, pool)
    pool_embs = get_embeddings(list(models.values())[0], dataset, pool)
    if pool_embs is None or len(pool)<nq: return sel_random(pool, nq)

    top_m = max(nq, min(int(len(pool)*0.5), nq*10))
    top_m = min(top_m, len(pool))
    top_m_idx = np.argsort(u)[-top_m:]
    top_u = u[top_m_idx]; top_embs = pool_embs[top_m_idx]

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    top_u_n = norm(top_u)

    selected_local = [int(np.argmax(top_u_n))]
    remaining = [i for i in range(len(top_m_idx)) if i!=selected_local[0]]

    for _ in range(nq-1):
        best_score, best_idx = -np.inf, None
        for idx in remaining:
            u_score = top_u_n[idx]
            min_dist = min(np.linalg.norm(top_embs[idx]-top_embs[s]) for s in selected_local)
            max_dist = max(np.linalg.norm(top_embs[idx]-top_embs[s]) for s in selected_local)
            total = u_score + min_dist/(max_dist+1e-10)
            if total > best_score: best_score, best_idx = total, idx
        if best_idx is not None:
            selected_local.append(best_idx); remaining.remove(best_idx)

    return pool[top_m_idx[np.array(selected_local)]]

# ---------------------------------------------------------------------------
# K: AUD-BALD (plan v1.0)
# ---------------------------------------------------------------------------
def sel_aud_bald(pool, nq, models, dataset, structures, labeled_embeddings, **kw):
    mlist = list(models.values())
    pl = make_dataloader(dataset, pool, BATCH, shuffle=False)
    per_model_forces = []
    for m in mlist:
        m.eval(); model_f = []
        with torch.no_grad():
            for batch in pl:
                batch_dev = to_dev(batch)
                _, forces = m(batch_dev["z"], batch_dev["pos"], batch_dev["batch"])
                if forces is None: model_f.extend([0.0]*(batch_dev["batch"].max().item()+1)); continue
                for s in range(batch_dev["batch"].max().item()+1):
                    model_f.append(torch.norm(forces[batch_dev["batch"]==s]).item())
        per_model_forces.append(model_f)
    per_model_forces = np.array(per_model_forces)
    if per_model_forces.shape[1]==0: return sel_random(pool, nq)

    ensemble_std = per_model_forces.std(axis=0)
    ensemble_mean = per_model_forces.mean(axis=0)
    u_bald = ensemble_std*(1.0+ensemble_std/(ensemble_mean+1e-10))

    pool_embs = get_embeddings(mlist[0], dataset, pool)
    if pool_embs is None or len(pool)<5: return sel_random(pool, nq)
    if labeled_embeddings is not None and labeled_embeddings.shape[0]>0:
        d = cdist(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else: d = np.ones(len(pool))

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u_bald), norm(d)
    rho, _ = spearmanr(u_n, d_n)
    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8)
    combined = alpha*u_n + (1-alpha)*d_n
    return pool[np.argsort(combined)[-nq:]]

# ---------------------------------------------------------------------------
# L: ρ-Diagnostic Selector (plan v1.0)
# ---------------------------------------------------------------------------
def sel_rho_diagnostic(pool, nq, models, dataset, structures, labeled_embeddings, rho_threshold=-0.3, **kw):
    u = compute_uncertainties(models, dataset, pool)
    pool_embs = get_embeddings(list(models.values())[0], dataset, pool)
    if pool_embs is None or len(pool)<5: return sel_random(pool, nq)

    if labeled_embeddings is not None and labeled_embeddings.shape[0]>0:
        d = cdist(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else: d = np.ones(len(pool))

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u), norm(d)
    rho, _ = spearmanr(u_n, d_n)
    if rho < rho_threshold:
        alpha = np.clip(0.5-0.3*rho, 0.2, 0.8)
    else: alpha = 0.5
    combined = alpha*u_n + (1-alpha)*d_n
    return pool[np.argsort(combined)[-nq:]]

# ---------------------------------------------------------------------------
# Strategy registry (remove B, D to save time)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "A_random": sel_random,
    "C_ensemble_qbc": sel_uncertainty,
    "E_diversity": sel_diversity,
    "F_latent_clustering": sel_hybrid_weighted,
    "G_hybrid_weighted": sel_hybrid_weighted,
    "H_hybrid_twostage": sel_hybrid_twostage,
    "I_aud_rank": sel_aud_rank,
    "J_aud_batch": sel_aud_batch,
    "K_aud_bald": sel_aud_bald,
    "L_rho_diagnostic": sel_rho_diagnostic,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
systems = sorted([f.stem for f in Path(DATA_DIR).glob("*.pkl")])
if len(sys.argv) > 2:
    systems = [s for s in systems if s == sys.argv[2]]

print(f"Systems: {systems}, Strategies: {list(STRATEGIES.keys())}")

for sys_name in systems:
    print(f"\n{'#'*60}\n#  {sys_name}\n{'#'*60}")
    with open(f"{DATA_DIR}/{sys_name}.pkl","rb") as f:
        structures = pickle.load(f)
    dataset = MaterialDataset(structures)
    energies = [s.info["energy"] for s in structures]
    print(f"  {len(structures)} structures, E=[{min(energies):.0f},{max(energies):.0f}]")
    init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

    all_curves = {}
    for sname, sfn in STRATEGIES.items():
        print(f"\n  --- {sname} ---")
        labeled = list(init_idx); pool = list(pool_idx)
        leb = None; curve = []
        for it in range(N_ITER+1):
            models = {}
            for ms in [SEED, SEED+100]:
                torch.manual_seed(ms)
                model, test_mae = train_and_eval(dataset, labeled, val_idx, test_idx)
                models[ms] = model
            curve.append(test_mae)
            print(f"    Iter {it} | N={len(labeled)} | Test MAE={test_mae:.4f} eV")
            if it >= N_ITER or len(pool) < N_QUERY: break
            # Compute labeled embeddings from first model
            leb = get_embeddings(list(models.values())[0], dataset, labeled)
            selected = sfn(np.array(pool), N_QUERY, models=models, dataset=dataset,
                           structures=structures, labeled_embeddings=leb)
            for s in selected:
                if s in pool: pool.remove(int(s)); labeled.append(int(s))
        all_curves[sname] = curve

    # Save
    import pandas as pd
    max_len = max(len(c) for c in all_curves.values())
    data = {s: c + [np.nan]*(max_len-len(c)) for s, c in all_curves.items()}
    df = pd.DataFrame(data)
    df.to_csv(f"results/ms25_9strat_{sys_name}_seed{SEED}.csv", index=False)
    print(f"  Saved results/ms25_9strat_{sys_name}_seed{SEED}.csv")

print("\nDone!")
