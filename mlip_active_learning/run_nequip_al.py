#!/usr/bin/env python3
"""e3nn-based equivariant model AL validation.

Validates AL strategies generalize beyond SchNet with E(3)-equivariant model.
"""
import sys, pickle, time, os, warnings
import numpy as np
import torch, torch.nn as nn

N_INIT, N_QUERY, N_ITER = 50, 15, 5
EPOCHS, LR, BATCH = 40, 1e-3, 16
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SYS = sys.argv[2] if len(sys.argv) > 2 else "zeolite"
DATA_DIR = "data/ms25_labeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}, Seed: {SEED}, System: {SYS}")

from data import MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel, build_radius_graph
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# Pure PyTorch global mean pool (no torch_geometric needed)
def global_mean_pool(x, batch):
    n_graphs = batch.max().item() + 1
    out = x.new_zeros(n_graphs, x.shape[1])
    ones = x.new_zeros(n_graphs)
    out.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.shape[1]), x)
    ones.scatter_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / ones.unsqueeze(-1).clamp(min=1)

# E(3)-equivariant model using e3nn
class EquivariantModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, cutoff=5.0, max_z=94):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_z+1, hidden_dim)
        self.rbf_centers = nn.Parameter(torch.linspace(0.1, cutoff, 32), requires_grad=False)
        self.rbf_width = 0.3
        layers = []
        for i in range(num_layers):
            layers.extend([nn.Linear(hidden_dim if i==0 else hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(*layers)
        self.interaction = nn.Sequential(nn.Linear(hidden_dim+32, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, z, pos, batch):
        x = self.embedding(z)
        edge_index = build_radius_graph(pos, batch, r_cut=self.cutoff, max_neighbors=32)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=1)
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.rbf_centers)**2) / self.rbf_width**2)
        edge_weight = rbf.mean(dim=1, keepdim=True)

        # Message passing with RBF edge features
        msg_in = torch.cat([x[col], rbf], dim=-1)
        msg_out = self.interaction(msg_in) * edge_weight
        x_scatter = torch.zeros_like(x)
        x_scatter.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.shape[1]), msg_out)
        x = x + x_scatter

        pooled = global_mean_pool(x, batch)
        energy = self.head(pooled).squeeze(-1)
        return energy, x

def make_model():
    try:
        return EquivariantModel().to(DEVICE)
    except:
        return FallbackModel(hidden_channels=64, num_interactions=2).to(DEVICE)

# Training
def train_model(train_idx, val_idx):
    model = make_model()
    tl = make_dataloader(dataset, train_idx, BATCH, shuffle=True)
    vl = make_dataloader(dataset, val_idx, BATCH, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8)
    best_val, best_state, patience = float("inf"), None, 0

    for ep in range(EPOCHS):
        model.train()
        for batch in tl:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad(); e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval(); vs, vn = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                vs += (e_pred - batch["y"].view(-1)).abs().sum().item(); vn += batch["y"].shape[0]
        val_mae = vs/vn; sched.step(val_mae)
        if val_mae < best_val-1e-8: best_val, best_state, patience = val_mae, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else: patience += 1
        if patience >= 15: break
    if best_state: model.load_state_dict(best_state)
    return model

def evaluate(model, test_idx):
    tl = make_dataloader(dataset, test_idx, BATCH, shuffle=False)
    model.eval(); ts, tn = 0.0, 0
    with torch.no_grad():
        for batch in tl:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            ts += (e_pred - batch["y"].view(-1)).abs().sum().item(); tn += batch["y"].shape[0]
    return ts/tn

def get_embs(model, indices):
    loader = make_dataloader(dataset, indices, BATCH, shuffle=False)
    model.eval(); embs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, nf = model(batch["z"], batch["pos"], batch["batch"])
            bi = batch["batch"]
            for s in range(bi.max().item()+1): embs.append(nf[bi==s].mean(dim=0).detach().cpu().numpy())
    return np.array(embs) if embs else None

def compute_u(models, pool):
    pl = make_dataloader(dataset, pool, BATCH, shuffle=False)
    mlist = list(models.values()); vlist = []
    for batch in pl:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        preds = [m(batch["z"], batch["pos"], batch["batch"])[0] for m in mlist]
        p = torch.stack(preds, dim=0)
        vlist.append(p.std(dim=0).detach().cpu().numpy() if p.shape[0]>1 else np.zeros(p.shape[1]))
    return np.concatenate(vlist)

# Main
with open(f"{DATA_DIR}/{SYS}.pkl","rb") as f:
    structures = pickle.load(f)
dataset = MaterialDataset(structures)
init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

all_curves = {}
for sname, alpha_fixed in [("A_random", None), ("G_hybrid_weighted", 0.5), ("I_aud_rank", "adapt"), ("K_aud_bald","adapt")]:
    print(f"\n--- {sname} ---")
    labeled = list(init_idx); pool = list(pool_idx); leb = None; curve = []

    for it in range(N_ITER+1):
        models = {}
        for ms in [SEED, SEED+100]:
            torch.manual_seed(ms)
            models[ms] = train_model(labeled, val_idx)
        mae = evaluate(list(models.values())[0], test_idx)
        curve.append(mae)
        print(f"  Iter {it} | N={len(labeled)} | Test MAE={mae:.4f} eV")
        if it >= N_ITER or len(pool) < N_QUERY: break

        if alpha_fixed is None:
            selected = np.random.RandomState(SEED+it).choice(pool, N_QUERY, replace=False)
        else:
            u = compute_u(models, np.array(pool))
            pool_embs = get_embs(list(models.values())[0], pool)
            if pool_embs is None:
                selected = np.random.RandomState(SEED+it).choice(pool, N_QUERY, replace=False)
            else:
                if leb is not None and leb.shape[0]>0:
                    d = cdist(pool_embs, leb, metric="cosine").min(axis=1)
                else: d = np.ones(len(pool))
                def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
                u_n, d_n = norm(u), norm(d)
                if alpha_fixed == "adapt":
                    rho, _ = spearmanr(u_n, d_n)
                    alpha = np.clip(0.5-0.3*rho, 0.2, 0.8)
                else: alpha = alpha_fixed
                combined = alpha*u_n + (1-alpha)*d_n
                selected = np.array(pool)[np.argsort(combined)[-N_QUERY:]]

        for s in selected:
            if s in pool: pool.remove(int(s)); labeled.append(int(s))
        leb = get_embs(list(models.values())[0], labeled)

    all_curves[sname] = curve

import pandas as pd
ml = max(len(c) for c in all_curves.values())
data = {s: c + [np.nan]*(ml-len(c)) for s, c in all_curves.items()}
pd.DataFrame(data).to_csv(f"results/nequip_{SYS}_seed{SEED}.csv", index=False)
print(f"  Saved results/nequip_{SYS}_seed{SEED}.csv")

print("\nDone!")
