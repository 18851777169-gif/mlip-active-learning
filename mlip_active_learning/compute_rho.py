"""Compute Spearman ρ distribution from SchNet 10-strat experiment.

Loads a completed experiment, re-computes U/D at each AL iteration,
and logs ρ values for statistical analysis.
"""

import sys, pickle, numpy as np
import torch, torch.nn as nn

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SYS = sys.argv[2] if len(sys.argv) > 2 else "FeNiCrCoCu_HEA"
DATA_DIR = "data/ms25_labeled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); np.random.seed(SEED)

from data import MaterialDataset, create_splits, make_dataloader
from model_fallback import FallbackModel
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# Load data
with open(f"{DATA_DIR}/{SYS}.pkl", "rb") as f:
    structures = pickle.load(f)
dataset = MaterialDataset(structures)

N_INIT, N_QUERY, N_ITER = 50, 15, 6
init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

def train_model(train_idx, val_idx):
    model = FallbackModel(hidden_channels=64, num_interactions=2).to(DEVICE)
    tl = make_dataloader(dataset, train_idx, 16, shuffle=True)
    vl = make_dataloader(dataset, val_idx, 16, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8)
    best_val, best_state, patience = float("inf"), None, 0
    for ep in range(40):
        model.train()
        for batch in tl:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad()
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vs, vn = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                vs += (e_pred - batch["y"].view(-1)).abs().sum().item()
                vn += batch["y"].shape[0]
        val_mae = vs/vn; sched.step(val_mae)
        if val_mae < best_val-1e-8: best_val, best_state, patience = val_mae, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else: patience += 1
        if patience >= 15: break
    if best_state: model.load_state_dict(best_state)
    return model

def get_embeddings(model, indices):
    loader = make_dataloader(dataset, indices, 16, shuffle=False)
    model.eval(); embs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, nf = model(batch["z"], batch["pos"], batch["batch"])
            bi = batch["batch"]
            for s in range(bi.max().item() + 1):
                embs.append(nf[bi==s].mean(dim=0).cpu().numpy())
    return np.array(embs)

def compute_u(models, pool_indices):
    pl = make_dataloader(dataset, pool_indices, 16, shuffle=False)
    mlist = list(models.values()); vars_list = []
    for batch in pl:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        preds = []
        for m in mlist:
            m.eval()
            with torch.no_grad(): preds.append(m(batch["z"], batch["pos"], batch["batch"])[0])
        p = torch.stack(preds, dim=0)
        vars_list.append(p.std(dim=0, unbiased=False).cpu().numpy() if p.shape[0]>1 else np.zeros(p.shape[1]))
    return np.concatenate(vars_list)

# Run AL and collect ρ values
print(f"Computing ρ distribution for {SYS} (seed={SEED})...")
labeled = list(init_idx); pool = list(pool_idx); leb = None

rho_values = []  # (strategy_name, iteration, rho, alpha)

for iteration in range(N_ITER + 1):
    models = {}
    for ms in [SEED, SEED+100]:
        torch.manual_seed(ms)
        models[ms] = train_model(labeled, val_idx)

    if iteration >= N_ITER or len(pool) < N_QUERY: break

    # Compute U and D
    u = compute_u(models, np.array(pool))
    pool_embs = get_embeddings(list(models.values())[0], pool)
    if leb is not None and leb.shape[0] > 0:
        d = cdist(pool_embs, leb, metric="cosine").min(axis=1)
    else:
        d = np.ones(len(pool))

    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-10)
    u_n, d_n = norm(u), norm(d)
    rho, pval = spearmanr(u_n, d_n)
    if np.isnan(rho): rho = 0.0

    alpha_adaptive = np.clip(0.5 - 0.3*rho, 0.2, 0.8)
    # L strategy: only adaptive if rho < -0.3
    alpha_l = np.clip(0.5 - 0.3*rho, 0.2, 0.8) if rho < -0.3 else 0.5

    rho_values.append({
        "iteration": iteration, "n_labeled": len(labeled),
        "rho": rho, "rho_pval": pval,
        "alpha_i": alpha_adaptive, "alpha_l": alpha_l,
    })

    print(f"  Iter {iteration}: N={len(labeled)}, ρ={rho:+.3f}, α_I={alpha_adaptive:.2f}, α_L={alpha_l:.2f}")

    # Select using G_hybrid (fixed α=0.5) to match experiment conditions
    combined = 0.5*u_n + 0.5*d_n
    selected = np.array(pool)[np.argsort(combined)[-N_QUERY:]]
    for s in selected:
        if s in pool: pool.remove(int(s)); labeled.append(int(s))
    leb = get_embeddings(list(models.values())[0], labeled)

# Statistics
import pandas as pd
rhos = [r["rho"] for r in rho_values]
print(f"\n===== ρ Statistics for {SYS} =====")
print(f"  N iterations: {len(rhos)}")
print(f"  ρ range: [{min(rhos):+.3f}, {max(rhos):+.3f}]")
print(f"  ρ mean: {np.mean(rhos):+.3f} ± {np.std(rhos):.3f}")
print(f"  ρ < -0.3 (anti-correlated): {sum(1 for r in rhos if r < -0.3)}/{len(rhos)} ({sum(1 for r in rhos if r < -0.3)/len(rhos)*100:.0f}%)")
print(f"  ρ < 0 (negative): {sum(1 for r in rhos if r < 0)}/{len(rhos)} ({sum(1 for r in rhos if r < 0)/len(rhos)*100:.0f}%)")
print(f"  ρ > 0 (positive): {sum(1 for r in rhos if r > 0)}/{len(rhos)} ({sum(1 for r in rhos if r > 0)/len(rhos)*100:.0f}%)")
print(f"  α_I range: [{min(r['alpha_i'] for r in rho_values):.2f}, {max(r['alpha_i'] for r in rho_values):.2f}]")
print(f"  α_L = 0.5 fraction: {sum(1 for r in rho_values if r['alpha_l'] == 0.5)}/{len(rho_values)}")
print(f"  α_L adaptive fraction: {sum(1 for r in rho_values if r['alpha_l'] != 0.5)}/{len(rho_values)}")

df = pd.DataFrame(rho_values)
csv_path = f"results/rho_stats_{SYS}_seed{SEED}.csv"
df.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")
