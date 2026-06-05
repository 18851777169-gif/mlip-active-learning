#!/usr/bin/env python3
"""MACE AL — single strategy, fresh process. Called by bash wrapper."""
import sys, pickle, time, os, warnings, copy
import numpy as np
import torch, torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

N_INIT, N_QUERY, N_ITER = 50, 15, 6
EPOCHS, LR, BATCH = 20, 5e-4, 4
SEED = int(sys.argv[1])
SYS = sys.argv[2]
STRAT = sys.argv[3]
DATA_DIR = "data/ms25_labeled"
MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"
DEVICE = "cuda"

torch.manual_seed(SEED); np.random.seed(SEED)

from data import MaterialDataset, create_splits
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# Load data
with open(f"{DATA_DIR}/{SYS}.pkl", "rb") as f:
    structures = pickle.load(f)
dataset = MaterialDataset(structures)
init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

# MACE wrapper
from mace.calculators import MACECalculator
from ase import Atoms

class MACEFineTuner:
    def __init__(self, model_path, seed=42):
        torch.manual_seed(seed)
        self.calc = MACECalculator(model_path=model_path, device=DEVICE, default_dtype="float32")
        self.model = self.calc.models[0]
        for n, p in self.model.named_parameters():
            p.requires_grad = ("readouts" in n or "products.1" in n or "interactions.1.linear_up" in n)
        self._n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _atoms_to_batch_dict(self, atoms):
        batch = self.calc._atoms_to_batch(atoms)
        return self.calc._clone_batch(batch).to_dict()

    def predict_batch(self, batch_dict):
        return self.model(batch_dict, training=self.model.training, compute_force=False)["energy"]

    def finetune(self, structures, train_idx, val_idx):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params: return float("inf")
        opt = torch.optim.Adam(params, lr=LR)
        train_s = [structures[i] for i in train_idx]
        val_s = [structures[i] for i in val_idx]
        best_val, best_state, patience = float("inf"), None, 0

        for ep in range(EPOCHS):
            self.model.train()
            for i in range(0, len(train_s), BATCH):
                batch_s = train_s[i:i+BATCH]
                combined = batch_s[0]  # one at a time
                bd = self._atoms_to_batch_dict(combined)
                opt.zero_grad()
                e_pred = self.predict_batch(bd)
                e_true = torch.tensor([s.info["energy"] for s in batch_s], dtype=torch.float32, device=DEVICE)
                loss = nn.functional.l1_loss(e_pred.view(-1), e_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            self.model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_s), BATCH):
                    batch_s = val_s[i:i+BATCH]
                    combined = batch_s[0]
                    bd = self._atoms_to_batch_dict(combined)
                    e_pred = self.predict_batch(bd)
                    e_true = torch.tensor([s.info["energy"] for s in batch_s], dtype=torch.float32, device=DEVICE)
                    v_loss += (e_pred.view(-1) - e_true).abs().sum().item()
                    v_n += len(batch_s)
            val_mae = v_loss / v_n
            if val_mae < best_val - 1e-8:
                best_val, best_state = val_mae, copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 5: break

        if best_state: self.model.load_state_dict(best_state)
        return best_val

    def evaluate(self, structures, test_idx):
        self.model.eval()
        t_loss, t_n = 0.0, 0
        with torch.no_grad():
            for i in test_idx:
                bd = self._atoms_to_batch_dict(structures[i])
                e_pred = self.predict_batch(bd)
                e_true = torch.tensor(structures[i].info["energy"], dtype=torch.float32, device=DEVICE)
                t_loss += (e_pred.view(-1) - e_true).abs().item()
                t_n += 1
        return t_loss / t_n

    def get_embedding(self, atoms):
        bd = self._atoms_to_batch_dict(atoms)
        self.model.eval()
        with torch.no_grad():
            out = self.model(bd, training=False, compute_force=False)
            feats = out.get("node_feats", None)
            if feats is not None: return feats.mean(dim=0).cpu().numpy()
        return None

# Acquisition functions
def sel_random(pool, nq, **kw):
    return np.random.RandomState(SEED).choice(pool, nq, replace=False)

def sel_uncertainty(pool, nq, tuners, structures, **kw):
    scores = []
    for i in pool:
        bd = tuners[0]._atoms_to_batch_dict(structures[i])
        preds = [t.predict_batch(bd).view(-1).item() for t in tuners]
        scores.append(np.std(preds) if len(preds)>1 else 0)
    return pool[np.argsort(scores)[-nq:]]

def sel_diversity(pool, nq, tuners, structures, labeled_embs=None, **kw):
    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        dists = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: dists = np.ones(len(valid))
    sel, d = [], dists.copy()
    for _ in range(nq):
        idx = int(np.argmax(d)); sel.append(idx)
        d = np.minimum(d, cdist(embs, embs[[idx]], metric="cosine").ravel())
    return np.array(valid)[np.array(sel)]

def sel_hybrid(pool, nq, tuners, structures, labeled_embs=None, alpha=0.5, **kw):
    u_scores = np.array([np.std([t.predict_batch(tuners[0]._atoms_to_batch_dict(structures[i])).view(-1).item() for t in tuners]) for i in pool])
    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        d = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: d = np.ones(len(valid))
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    combined = alpha*norm(u_scores) + (1-alpha)*norm(d)
    return np.array(valid)[np.argsort(combined)[-nq:]]

def sel_aud_rank(pool, nq, tuners, structures, labeled_embs=None, **kw):
    u = np.array([np.std([t.predict_batch(tuners[0]._atoms_to_batch_dict(structures[i])).view(-1).item() for t in tuners]) for i in pool])
    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        d = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: d = np.ones(len(valid))
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u), norm(d)
    rho, _ = spearmanr(u_n, d_n)
    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8)
    combined = alpha*u_n + (1-alpha)*d_n
    return np.array(valid)[np.argsort(combined)[-nq:]]

def sel_aud_batch(pool, nq, tuners, structures, labeled_embs=None, **kw):
    u = np.array([np.std([t.predict_batch(tuners[0]._atoms_to_batch_dict(structures[i])).view(-1).item() for t in tuners]) for i in pool])
    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    top_m = max(nq, min(int(len(valid)*0.5), nq*10)); top_m = min(top_m, len(valid))
    top_idx = np.argsort(u)[-top_m:]
    top_u, top_embs = u[top_idx], embs[top_idx]
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    top_u_n = norm(top_u)
    sel = [int(np.argmax(top_u_n))]
    rem = [i for i in range(len(top_idx)) if i!=sel[0]]
    for _ in range(nq-1):
        best, best_i = -np.inf, None
        for idx in rem:
            sc = top_u_n[idx] + min(np.linalg.norm(top_embs[idx]-top_embs[s]) for s in sel)/(max(np.linalg.norm(top_embs[idx]-top_embs[s]) for s in sel)+1e-10)
            if sc > best: best, best_i = sc, idx
        if best_i is not None: sel.append(best_i); rem.remove(best_i)
    return np.array(valid)[top_idx[np.array(sel)]]

# Simple proxy for K/L (skip force-based BALD to avoid crash)
def sel_aud_bald_proxy(pool, nq, tuners, structures, labeled_embs=None, **kw):
    """BALD energy-based proxy — avoids ASE force calls."""
    u = np.array([np.std([t.predict_batch(tuners[0]._atoms_to_batch_dict(structures[i])).view(-1).item() for t in tuners]) for i in pool])
    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n = norm(u)
    # BALD-like: amplify with 1+std/mean
    u_bald = u * (1.0 + u/(u.mean()+1e-10))
    u_bald_n = norm(u_bald)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        d = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: d = np.ones(len(valid))
    d_n = norm(d)
    rho, _ = spearmanr(u_bald_n, d_n)
    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8)
    combined = alpha*u_bald_n + (1-alpha)*d_n
    return np.array(valid)[np.argsort(combined)[-nq:]]

def sel_rho_diagnostic(pool, nq, tuners, structures, labeled_embs=None, rho_threshold=-0.3, **kw):
    u = np.array([np.std([t.predict_batch(tuners[0]._atoms_to_batch_dict(structures[i])).view(-1).item() for t in tuners]) for i in pool])
    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        d = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: d = np.ones(len(valid))
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u), norm(d)
    rho, _ = spearmanr(u_n, d_n)
    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8) if rho < rho_threshold else 0.5
    combined = alpha*u_n + (1-alpha)*d_n
    return np.array(valid)[np.argsort(combined)[-nq:]]

STRATEGIES = {
    "A_random": lambda p, nq, t, s, le: sel_random(p, nq),
    "C_uncertainty": sel_uncertainty,
    "E_diversity": sel_diversity,
    "G_hybrid_weighted": lambda p, nq, t, s, le: sel_hybrid(p, nq, t, s, le),
    "I_aud_rank": sel_aud_rank,
    "J_aud_batch": sel_aud_batch,
    "K_aud_bald": sel_aud_bald_proxy,
    "L_rho_diagnostic": sel_rho_diagnostic,
}

# ===== RUN ONE STRATEGY =====
ens_seeds = [SEED, SEED+100]
sfn = STRATEGIES[STRAT]
print(f"Strategy: {STRAT}, Seed: {SEED}, System: {SYS}")

labeled = list(init_idx); pool = list(pool_idx); leb = None; curve = []
for it in range(N_ITER+1):
    tuners = []
    for eseed in ens_seeds:
        ft = MACEFineTuner(MODEL_PATH, seed=eseed)
        ft.finetune(structures, labeled, val_idx)
        tuners.append(ft)
    test_mae = tuners[0].evaluate(structures, test_idx)
    curve.append(test_mae)
    print(f"  Iter {it} | N={len(labeled)} | Test MAE={test_mae:.4f} eV")
    if it >= N_ITER or len(pool) < N_QUERY: break
    selected = sfn(np.array(pool), N_QUERY, tuners=tuners, structures=structures, labeled_embs=leb)
    for s in selected:
        if s in pool: pool.remove(int(s)); labeled.append(int(s))
    leb_vals = [tuners[0].get_embedding(structures[i]) for i in labeled]
    leb = np.array([e for e in leb_vals if e is not None])
    del tuners; torch.cuda.empty_cache()

# Save to CSV (append mode)
import pandas as pd
csv_path = f"results/mace_al_{SYS}_seed{SEED}.csv"
try:
    existing = pd.read_csv(csv_path)
    existing[STRAT] = curve[:len(existing)]
    existing.to_csv(csv_path, index=False)
except:
    pd.DataFrame({STRAT: curve}).to_csv(csv_path, index=False)
print(f"  Saved {STRAT} to {csv_path}")
