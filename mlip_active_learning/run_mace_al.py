#!/usr/bin/env python3
"""MACE Fine-tuning Active Learning — Full Experiment.

Fine-tunes pre-trained MACE-MP-0 on actively selected structures.
Compares 5 acquisition strategies with 3 seeds.
"""

import sys, pickle, time, os, warnings, copy
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

warnings.filterwarnings("ignore")

N_INIT, N_QUERY, N_ITER = 50, 15, 6
EPOCHS, LR, BATCH = 20, 5e-4, 4
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DATA_DIR = "data/ms25_labeled"
MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"
DEVICE = "cuda"

torch.manual_seed(SEED); np.random.seed(SEED)

from data import MaterialDataset, make_dataloader, create_splits
from scipy.spatial.distance import cdist
from ase import Atoms

# ---------------------------------------------------------------------------
# MACE Fine-Tuner with ensemble support
# ---------------------------------------------------------------------------
class MACEFineTuner:
    def __init__(self, model_path, seed=42):
        from mace.calculators import MACECalculator
        torch.manual_seed(seed)
        self.calc = MACECalculator(model_path=model_path, device=DEVICE, default_dtype="float32")
        self.model = self.calc.models[0]
        for n, p in self.model.named_parameters():
            p.requires_grad = ("readouts" in n or "products.1" in n or
                               "interactions.1.linear_up" in n)
        self._n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    [MACE FT] trainable params: {self._n_trainable}")
        self._n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _atoms_to_batch_dict(self, atoms):
        batch = self.calc._atoms_to_batch(atoms)
        batch = self.calc._clone_batch(batch)
        return batch.to_dict()

    def predict_batch(self, batch_dict):
        return self.model(batch_dict, training=self.model.training, compute_force=False)["energy"]

    def finetune(self, structures, train_idx, val_idx, epochs=EPOCHS, lr=LR):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        train_s = [structures[i] for i in train_idx]
        val_s = [structures[i] for i in val_idx]
        best_val, best_state, patience = float("inf"), None, 0

        for ep in range(epochs):
            self.model.train()
            for i in range(0, len(train_s), BATCH):
                batch_s = train_s[i:i+BATCH]
                combined = batch_s[0]  # process one at a time for MACE
                bd = self._atoms_to_batch_dict(combined)
                opt.zero_grad()
                e_pred = self.predict_batch(bd)
                e_true = torch.tensor([s.info["energy"] for s in batch_s],
                                      dtype=torch.float32, device=DEVICE)
                loss = nn.functional.l1_loss(e_pred.view(-1), e_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            self.model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_s), BATCH):
                    batch_s = val_s[i:i+BATCH]
                    combined = batch_s[0]  # process one at a time for MACE
                    bd = self._atoms_to_batch_dict(combined)
                    e_pred = self.predict_batch(bd)
                    e_true = torch.tensor([s.info["energy"] for s in batch_s],
                                          dtype=torch.float32, device=DEVICE)
                    v_loss += (e_pred.view(-1) - e_true).abs().sum().item()
                    v_n += len(batch_s)
            val_mae = v_loss / v_n
            if val_mae < best_val - 1e-8:
                best_val, best_state = val_mae, copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 5:
                break
        if best_state:
            self.model.load_state_dict(best_state)
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
            if feats is not None:
                return feats.mean(dim=0).cpu().numpy()
        return None

# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------
def sel_random(pool, nq):
    return np.random.RandomState(SEED).choice(pool, nq, replace=False)

def sel_uncertainty(pool, nq, tuners, structures, leb=None):
    scores = []
    for i in pool:
        bd = tuners[0]._atoms_to_batch_dict(structures[i])
        preds = []
        for t in tuners:
            t.model.eval()
            with torch.no_grad():
                e = t.predict_batch(bd)
                preds.append(e.view(-1).item())
        scores.append(np.std(preds) if len(preds) > 1 else 0)
    top = np.argsort(scores)[-nq:]
    return pool[top]

def sel_diversity(pool, nq, tuners, structures, labeled_embs):
    embs = np.array([tuners[0].get_embedding(structures[i])
                     for i in pool if tuners[0].get_embedding(structures[i]) is not None])
    valid_pool = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    if len(valid_pool) < nq:
        return sel_random(pool, nq)

    if labeled_embs is not None and labeled_embs.shape[0] > 0:
        dists = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else:
        dists = np.ones(len(valid_pool))

    selected_local, d = [], dists.copy()
    for _ in range(nq):
        idx = int(np.argmax(d))
        selected_local.append(idx)
        nd = cdist(embs, embs[[idx]], metric="cosine").ravel()
        d = np.minimum(d, nd)
    return np.array(valid_pool)[np.array(selected_local)]

def sel_hybrid(pool, nq, tuners, structures, labeled_embs, alpha=0.5):
    # Uncertainty
    u_scores = []
    for i in pool:
        bd = tuners[0]._atoms_to_batch_dict(structures[i])
        preds = []
        for t in tuners:
            t.model.eval()
            with torch.no_grad():
                e = t.predict_batch(bd)
                preds.append(e.view(-1).item())
        u_scores.append(np.std(preds) if len(preds) > 1 else 0)
    u_scores = np.array(u_scores)

    # Diversity
    embs = np.array([tuners[0].get_embedding(structures[i])
                     for i in pool if tuners[0].get_embedding(structures[i]) is not None])
    if labeled_embs is not None and labeled_embs.shape[0] > 0:
        d_scores = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else:
        d_scores = np.ones(len(pool))

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)
    combined = alpha * norm(u_scores) + (1 - alpha) * norm(d_scores)
    top = np.argsort(combined)[-nq:]
    return pool[top]

from scipy.stats import spearmanr

# ── I: AUD-Rank (MACE) ──
def sel_aud_rank_mace(pool, nq, tuners, structures, labeled_embs):
    u_scores = []
    for i in pool:
        bd = tuners[0]._atoms_to_batch_dict(structures[i]); preds = []
        for t in tuners:
            t.model.eval()
            with torch.no_grad(): preds.append(t.predict_batch(bd).view(-1).item())
        u_scores.append(np.std(preds) if len(preds)>1 else 0)
    u_scores = np.array(u_scores)

    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        d_scores = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: d_scores = np.ones(len(valid))

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u_scores), norm(d_scores)
    rho, _ = spearmanr(u_n, d_n)
    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8)
    combined = alpha*u_n + (1-alpha)*d_n
    return np.array(valid)[np.argsort(combined)[-nq:]]

# ── J: AUD-Batch (MACE) ──
def sel_aud_batch_mace(pool, nq, tuners, structures, labeled_embs):
    u_scores = []
    for i in pool:
        bd = tuners[0]._atoms_to_batch_dict(structures[i]); preds = []
        for t in tuners:
            t.model.eval()
            with torch.no_grad(): preds.append(t.predict_batch(bd).view(-1).item())
        u_scores.append(np.std(preds) if len(preds)>1 else 0)
    u_scores = np.array(u_scores)

    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)

    top_m = max(nq, min(int(len(valid)*0.5), nq*10))
    top_m = min(top_m, len(valid))
    top_m_idx = np.argsort(u_scores)[-top_m:]
    top_u = u_scores[top_m_idx]; top_embs = embs[top_m_idx]

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
        if best_idx is not None: selected_local.append(best_idx); remaining.remove(best_idx)
    return np.array(valid)[top_m_idx[np.array(selected_local)]]

# ── K: AUD-BALD (MACE) ──
def sel_aud_bald_mace(pool, nq, tuners, structures, labeled_embs):
    per_model_forces = []
    for t in tuners:
        t.model.eval(); model_f = []
        with torch.no_grad():
            for i in pool:
                a = structures[i].copy(); a.calc = t.calc
                f = a.get_forces(); model_f.append(np.linalg.norm(f))
        per_model_forces.append(model_f)
    per_model_forces = np.array(per_model_forces)
    if per_model_forces.shape[1]==0: return sel_random(pool, nq)

    ensemble_std = per_model_forces.std(axis=0)
    ensemble_mean = per_model_forces.mean(axis=0)
    u_bald = ensemble_std*(1.0+ensemble_std/(ensemble_mean+1e-10))

    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        d_scores = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: d_scores = np.ones(len(valid))

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u_bald), norm(d_scores)
    rho, _ = spearmanr(u_n, d_n)
    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8)
    combined = alpha*u_n + (1-alpha)*d_n
    return np.array(valid)[np.argsort(combined)[-nq:]]

# ── L: ρ-Diagnostic (MACE) ──
def sel_rho_diagnostic_mace(pool, nq, tuners, structures, labeled_embs, rho_threshold=-0.3):
    u_scores = []
    for i in pool:
        bd = tuners[0]._atoms_to_batch_dict(structures[i]); preds = []
        for t in tuners:
            t.model.eval()
            with torch.no_grad(): preds.append(t.predict_batch(bd).view(-1).item())
        u_scores.append(np.std(preds) if len(preds)>1 else 0)
    u_scores = np.array(u_scores)

    valid = [i for i in pool if tuners[0].get_embedding(structures[i]) is not None]
    embs = np.array([tuners[0].get_embedding(structures[i]) for i in valid])
    if len(valid) < nq: return sel_random(pool, nq)
    if labeled_embs is not None and labeled_embs.shape[0]>0:
        d_scores = cdist(embs, labeled_embs, metric="cosine").min(axis=1)
    else: d_scores = np.ones(len(valid))

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u_scores), norm(d_scores)
    rho, _ = spearmanr(u_n, d_n)
    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8) if rho < rho_threshold else 0.5
    combined = alpha*u_n + (1-alpha)*d_n
    return np.array(valid)[np.argsort(combined)[-nq:]]

# ── Strategy registry ──
STRATEGIES = {
    "A_random": lambda p, nq, t, s, le: sel_random(p, nq),
    "C_uncertainty": sel_uncertainty,
    "E_diversity": sel_diversity,
    "G_hybrid_weighted": sel_hybrid,
    "I_aud_rank": sel_aud_rank_mace,
    "J_aud_batch": sel_aud_batch_mace,
    "K_aud_bald": sel_aud_bald_mace,
    "L_rho_diagnostic": sel_rho_diagnostic_mace,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
systems = sorted([f.stem for f in Path(DATA_DIR).glob("*.pkl")])
if len(sys.argv) > 2:
    systems = [s for s in systems if s == sys.argv[2]]

ens_seeds = [SEED, SEED + 100]
print(f"Seed: {SEED}, Systems: {systems}")

for sys_name in systems:
    print(f"\n{'#'*60}\n#  {sys_name}\n{'#'*60}")

    with open(f"{DATA_DIR}/{sys_name}.pkl", "rb") as f:
        structures = pickle.load(f)

    # Compute MACE energies (use existing if already labeled with MACE)
    energies_present = all("energy" in s.info and s.info.get("_mace_labeled") for s in structures)
    if not energies_present:
        print(f"  Computing MACE energies...")
        t0 = time.time()
        from mace.calculators import MACECalculator
        ec = MACECalculator(model_path=MODEL_PATH, device=DEVICE, default_dtype="float32")
        for atoms in structures:
            a = atoms.copy()
            a.calc = ec
            atoms.info["energy"] = a.get_potential_energy()
            atoms.info["_mace_labeled"] = True
        print(f"  Done in {time.time()-t0:.0f}s")

    dataset = MaterialDataset(structures)
    init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

    all_curves = {}
    import pandas as pd
    for sname, sfn in STRATEGIES.items():
        torch.cuda.empty_cache()
        print(f"\n  --- {sname} ---")
        labeled = list(init_idx)
        pool = list(pool_idx)
        leb = None
        curve = []

        for it in range(N_ITER + 1):
            tuners = []
            for eseed in ens_seeds:
                ft = MACEFineTuner(MODEL_PATH, seed=eseed)
                ft.finetune(structures, labeled, val_idx)
                tuners.append(ft)

            test_mae = tuners[0].evaluate(structures, test_idx)
            curve.append(test_mae)
            print(f"    Iter {it} | N={len(labeled)} | Test MAE={test_mae:.4f} eV")

            if it >= N_ITER or len(pool) < N_QUERY:
                break

            selected = sfn(np.array(pool), N_QUERY, tuners, structures, leb)
            for s in selected:
                if s in pool:
                    pool.remove(int(s))
                    labeled.append(int(s))

            leb_vals = [tuners[0].get_embedding(structures[i]) for i in labeled]
            leb = np.array([e for e in leb_vals if e is not None])

        all_curves[sname] = curve
        # Incremental save after each strategy
        max_len = max(len(c) for c in all_curves.values())
        data = {}
        for s, c in all_curves.items():
            data[s] = c + [np.nan] * (max_len - len(c))
        pd.DataFrame(data).to_csv(f"results/mace_al_{sys_name}_seed{SEED}.csv", index=False)

print("\nDone!")
