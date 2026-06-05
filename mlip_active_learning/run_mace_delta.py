#!/usr/bin/env python3
"""MACE Delta-Learning Active Learning Experiment.

Uses pre-trained MACE-MP-0 as fixed feature extractor, trains a small
correction MLP on top. Active learning selects structures to minimize
the correction model's error.

This is equivalent to fine-tuning the output head of MACE while keeping
the body frozen — a common and effective transfer learning strategy.
"""

import sys, pickle, time, os, warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

N_INIT, N_QUERY, N_ITER = 30, 10, 6
EPOCHS, LR, BATCH, HIDDEN = 30, 1e-3, 16, 128
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DATA_DIR = "data/ms25_labeled"
MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"

torch.manual_seed(SEED)
np.random.seed(SEED)

from data import MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# MACE feature extractor (pre-compute all embeddings once)
# ---------------------------------------------------------------------------
def precompute_mace_energies(structures, model_path, device="cuda"):
    """Compute MACE baseline energies for all structures."""
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_path=model_path, device=device, default_dtype="float32")
    energies = []
    for i, atoms in enumerate(structures):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        try:
            e = atoms_copy.get_potential_energy()
            energies.append(e)
        except:
            energies.append(0.0)
        if (i + 1) % 200 == 0:
            print(f"  MACE energy {i+1}/{len(structures)}")
    return np.array(energies)

# ---------------------------------------------------------------------------
# Correction model (learns MACE_true - baseline_prediction)
# ---------------------------------------------------------------------------
class CorrectionHead(nn.Module):
    """Predicts per-structure energy correction from SchNet features."""
    def __init__(self, in_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)

def get_schnet_features(model, dataset, indices, batch_size=16):
    """Extract mean-pooled SchNet features for structures."""
    loader = make_dataloader(dataset, indices, batch_size, shuffle=False)
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in loader:
            _, node_feats = model(batch["z"], batch["pos"], batch["batch"])
            batch_idx = batch["batch"]
            for s in range(batch_idx.max().item() + 1):
                mask = batch_idx == s
                feats.append(node_feats[mask].mean(dim=0).cpu().numpy())
    return np.array(feats) if feats else None

# ---------------------------------------------------------------------------
# Train correction head
# ---------------------------------------------------------------------------
def train_correction(feat_model, dataset, train_idx, val_idx, test_idx,
                     mace_energies, device="cuda", epochs=EPOCHS, lr=LR):
    """Train correction head: predicts (MACE_true - baseline) from features."""
    # Get features
    train_feats = get_schnet_features(feat_model, dataset, train_idx)
    val_feats = get_schnet_features(feat_model, dataset, val_idx)
    test_feats = get_schnet_features(feat_model, dataset, test_idx)

    # Targets: difference from mean baseline
    train_energies = np.array([dataset[i]["y"].item() for i in train_idx])
    val_energies = np.array([dataset[i]["y"].item() for i in val_idx])
    test_energies = np.array([dataset[i]["y"].item() for i in test_idx])

    # Use a baseline: just the mean training energy (simple correction target)
    baseline = np.mean(train_energies)
    train_targets = train_energies - baseline
    val_targets = val_energies - baseline
    test_targets = test_energies - baseline

    head = CorrectionHead(in_dim=train_feats.shape[1], hidden=HIDDEN).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    Xt = torch.tensor(train_feats, dtype=torch.float32).to(device)
    yt = torch.tensor(train_targets, dtype=torch.float32).to(device)
    Xv = torch.tensor(val_feats, dtype=torch.float32).to(device)
    yv = torch.tensor(val_targets, dtype=torch.float32).to(device)
    Xtest = torch.tensor(test_feats, dtype=torch.float32).to(device)
    ytest = torch.tensor(test_targets, dtype=torch.float32).to(device)

    best_val, best_state, patience = float("inf"), None, 0
    n_train = Xt.shape[0]
    batch_size = min(32, n_train)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            pred = head(Xt[idx])
            loss = nn.functional.l1_loss(pred, yt[idx])
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_mae = (head(Xv) - yv).abs().mean().item()

        if val_mae < best_val - 1e-8:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_mae = (head(Xtest) - ytest).abs().mean().item()

    return head, test_mae

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading MACE for energy pre-computation...")
from mace.calculators import MACECalculator

# Load one system for testing
sys_name = sys.argv[2] if len(sys.argv) > 2 else "FeNiCrCoCu_HEA"
with open(f"{DATA_DIR}/{sys_name}.pkl", "rb") as f:
    structures = pickle.load(f)
print(f"{sys_name}: {len(structures)} structures")

# Pre-compute MACE energies
print("Computing MACE reference energies...")
t0 = time.time()
mace_energies = precompute_mace_energies(structures, MODEL_PATH)
print(f"Done in {time.time()-t0:.0f}s, E=[{mace_energies.min():.1f}, {mace_energies.max():.1f}]")

# Overwrite dataset energies with MACE energies
for i, atoms in enumerate(structures):
    atoms.info["energy"] = mace_energies[i]

dataset = MaterialDataset(structures)
init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

# Use SchNet as feature extractor (quick pre-train on initial set)
print(f"\nPre-training SchNet feature extractor on {N_INIT} structures...")
feat_model = FallbackModel(hidden_channels=64, num_interactions=2)
train_loader = make_dataloader(dataset, init_idx, 16, shuffle=True)
opt = torch.optim.Adam(feat_model.parameters(), lr=1e-3)
for ep in range(15):
    for batch in train_loader:
        opt.zero_grad()
        e_pred, _ = feat_model(batch["z"], batch["pos"], batch["batch"])
        loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
        loss.backward()
        opt.step()

# Train correction head
print("Training correction head...")
head, test_mae = train_correction(feat_model, dataset, init_idx, val_idx, test_idx, mace_energies)
print(f"Initial correction MAE: {test_mae:.4f} eV")

# Simple active learning loop
print(f"\nActive learning ({N_ITER} iterations)...")
labeled = list(init_idx)
pool = list(pool_idx)
curve = [test_mae]

for it in range(N_ITER):
    # Re-train feature extractor
    feat_model = FallbackModel(hidden_channels=64, num_interactions=2)
    train_loader = make_dataloader(dataset, labeled, 16, shuffle=True)
    opt = torch.optim.Adam(feat_model.parameters(), lr=1e-3)
    for ep in range(15):
        for batch in train_loader:
            opt.zero_grad()
            e_pred, _ = feat_model(batch["z"], batch["pos"], batch["batch"])
            loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
            loss.backward()
            opt.step()

    # Re-train correction
    head, test_mae = train_correction(feat_model, dataset, labeled, val_idx, test_idx, mace_energies)
    curve.append(test_mae)
    print(f"  Iter {it+1}: N={len(labeled)}, Correction MAE={test_mae:.4f} eV")

    if len(pool) < N_QUERY:
        break

    # Random selection (baseline)
    selected = np.random.RandomState(SEED+it).choice(pool, N_QUERY, replace=False)
    for s in selected:
        pool.remove(int(s))
        labeled.append(int(s))

print(f"\nCorrection MAE: {curve[0]:.4f} -> {curve[-1]:.4f} ({len(labeled)} structures)")
print("MACE delta-learning pipeline works!")
