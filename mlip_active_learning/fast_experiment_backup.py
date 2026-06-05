#!/usr/bin/env python
"""Fast active learning experiment — minimal abstractions, maximum speed.

Compares Random vs Uncertainty vs Diversity vs Hybrid-Weighted
on Cu LJ clusters using the lightweight SchNet fallback model.
Completes in ~5 minutes on CPU.
"""

import sys, time, json, warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

# Local imports
from data import generate_synthetic_structures, MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel
from descriptors import SOAPDescriptors, farthest_point_sampling

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_INIT = 50
N_QUERY = 15
N_ITER = 8
N_TOTAL = 1200
HIDDEN = 64
EPOCHS = 30
LR = 1e-3
BATCH = 16
DEVICE = "cpu"
N_SEEDS = 2   # Repeat experiment with different seeds for error bars
BASE_SEED = 42

torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Generating data...")
structs = generate_synthetic_structures("MgO_surface", n_structures=N_TOTAL, seed=BASE_SEED)
dataset = MaterialDataset(structs)
init_idx, pool_idx, test_idx, val_idx = create_splits(
    len(dataset), N_INIT, test_ratio=0.15, val_ratio=0.10, seed=BASE_SEED)
print(f"Total: {len(dataset)}, Init: {len(init_idx)}, Pool: {len(pool_idx)}, "
      f"Test: {len(test_idx)}, Val: {len(val_idx)}")

test_loader = make_dataloader(dataset, test_idx, BATCH, shuffle=False)

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_model(train_indices, val_indices, epochs=EPOCHS, lr=LR):
    """Train a FallbackModel and return best val MAE + trained model."""
    model = FallbackModel(hidden_channels=HIDDEN, num_interactions=2)
    train_loader = make_dataloader(dataset, train_indices, BATCH, shuffle=True)
    val_loader = make_dataloader(dataset, val_indices, BATCH, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8)

    best_val_mae = float("inf")
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_mae_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                val_mae_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
                val_n += batch["y"].shape[0]
        val_mae = val_mae_sum / val_n
        scheduler.step(val_mae)

        if val_mae < best_val_mae - 1e-8:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= 15:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_mae_sum = 0.0
    test_n = 0
    with torch.no_grad():
        for batch in test_loader:
            e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
            test_mae_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
            test_n += batch["y"].shape[0]
    test_mae = test_mae_sum / test_n

    return model, test_mae

# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------
def get_embeddings(model, indices):
    """Compute structure embeddings for given indices."""
    loader = make_dataloader(dataset, indices, BATCH, shuffle=False)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            _, node_feats = model(batch["z"], batch["pos"], batch["batch"])
            # Mean pool per structure
            batch_idx = batch["batch"]
            n_structs = batch_idx.max().item() + 1
            for s in range(n_structs):
                mask = batch_idx == s
                embs.append(node_feats[mask].mean(dim=0).cpu().numpy())
    return np.array(embs)

def select_random(pool_indices, n_query):
    return np.random.RandomState(BASE_SEED).choice(pool_indices, n_query, replace=False)

def select_uncertainty(pool_indices, n_query, models, n_models=2):
    """Select by ensemble variance (train 2 models)."""
    models_list = list(models.values())
    if len(models_list) < 2:
        return select_random(pool_indices, n_query)

    pool_loader = make_dataloader(dataset, pool_indices, BATCH, shuffle=False)
    variances = []
    for batch in pool_loader:
        preds = []
        for m in models_list:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        preds = torch.stack(preds, dim=0)
        variances.append(preds.std(dim=0).cpu().numpy())
    scores = np.concatenate(variances)
    top_k = np.argsort(scores)[-n_query:]
    return pool_indices[top_k]

def select_diversity(pool_indices, n_query, _models, labeled_embeddings):
    """Farthest point sampling in embedding space."""
    pool_embs = get_embeddings(list(_models.values())[0], pool_indices)
    if labeled_embeddings is not None and labeled_embeddings.shape[0] > 0:
        from scipy.spatial.distance import cdist
        # Score = min distance to any labeled structure
        dists = cdist(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else:
        dists = np.ones(len(pool_indices))
    top_k = np.argsort(dists)[-n_query:]
    return pool_indices[top_k]

def select_hybrid_twostage(pool_indices, n_query, _models, labeled_embeddings, topk_frac=0.3):
    """Two-stage: Top K% by uncertainty, then FPS diversity within filtered set."""
    models_list = list(_models.values())
    pool_loader = make_dataloader(dataset, pool_indices, BATCH, shuffle=False)

    # Stage 1: Compute uncertainty scores
    variances = []
    for batch in pool_loader:
        preds = []
        for m in models_list:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        preds = torch.stack(preds, dim=0)
        v = preds.std(dim=0).cpu().numpy() if len(models_list) > 1 else np.zeros(preds.shape[1])
        variances.append(v)
    u_scores = np.concatenate(variances)

    # Filter to top topk_frac
    n_keep = max(n_query * 3, int(len(pool_indices) * topk_frac))
    top_unc = np.argsort(u_scores)[-n_keep:]
    filtered_pool = pool_indices[top_unc]

    # Stage 2: FPS diversity within filtered
    from scipy.spatial.distance import cdist as cdist_fn
    pool_embs = get_embeddings(list(_models.values())[0], filtered_pool)
    if labeled_embeddings is not None and labeled_embeddings.shape[0] > 0:
        dists = cdist_fn(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else:
        dists = np.ones(len(filtered_pool))

    # Farthest point sampling
    selected_local = []
    dists = dists.copy()
    for _ in range(min(n_query, len(filtered_pool))):
        idx = int(np.argmax(dists))
        selected_local.append(idx)
        new_dists = cdist_fn(pool_embs, pool_embs[[idx]], metric="cosine").ravel()
        dists = np.minimum(dists, new_dists)

    return filtered_pool[np.array(selected_local)]


def select_hybrid_weighted(pool_indices, n_query, models, labeled_embeddings, alpha=0.5):
    """Weighted combination of uncertainty and diversity."""
    # Uncertainty
    models_list = list(models.values())
    pool_loader = make_dataloader(dataset, pool_indices, BATCH, shuffle=False)
    variances = []
    for batch in pool_loader:
        preds = []
        for m in models_list:
            m.eval()
            with torch.no_grad():
                e, _ = m(batch["z"], batch["pos"], batch["batch"])
                preds.append(e)
        preds = torch.stack(preds, dim=0)
        v = preds.std(dim=0).cpu().numpy() if len(models_list) > 1 else np.zeros(preds.shape[1])
        variances.append(v)
    u_scores = np.concatenate(variances)

    # Diversity
    pool_embs = get_embeddings(list(models.values())[0], pool_indices)
    if labeled_embeddings is not None and labeled_embeddings.shape[0] > 0:
        from scipy.spatial.distance import cdist
        d_scores = cdist(pool_embs, labeled_embeddings, metric="cosine").min(axis=1)
    else:
        d_scores = np.ones(len(pool_indices))

    # Normalize and combine
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    combined = alpha * norm(u_scores) + (1 - alpha) * norm(d_scores)
    top_k = np.argsort(combined)[-n_query:]
    return pool_indices[top_k]

STRATEGIES = {
    "A_random": lambda pool, nq, models, leb: select_random(pool, nq),
    "C_uncertainty": select_uncertainty,
    "E_diversity": select_diversity,
    "G_hybrid_weighted": select_hybrid_weighted,
    "H_hybrid_twostage": select_hybrid_twostage,
}

# ---------------------------------------------------------------------------
def run_one_experiment(exp_seed):
    """Run a single seeded experiment, return dict of strategy->[mae_list]."""
    rng = np.random.RandomState(exp_seed)
    structs = generate_synthetic_structures("MgO_surface", n_structures=N_TOTAL, seed=exp_seed)
    dataset_local = MaterialDataset(structs)
    init_idx, pool_idx, test_idx, val_idx = create_splits(
        len(dataset_local), N_INIT, test_ratio=0.15, val_ratio=0.10, seed=exp_seed)

    # Pre-create test loader (same for all strategies in this run)
    test_loader_local = make_dataloader(dataset_local, test_idx, BATCH, shuffle=False)

    def train_model_local(train_indices):
        """Train a FallbackModel using this experiment's data."""
        model = FallbackModel(hidden_channels=HIDDEN, num_interactions=2)
        train_loader = make_dataloader(dataset_local, train_indices, BATCH, shuffle=True)
        val_loader = make_dataloader(dataset_local, val_idx, BATCH, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8)
        best_val_mae = float("inf")
        best_state = None
        patience = 0
        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                loss = nn.functional.l1_loss(e_pred, batch["y"].view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            val_mae_sum, val_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                    val_mae_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
                    val_n += batch["y"].shape[0]
            val_mae = val_mae_sum / val_n
            scheduler.step(val_mae)
            if val_mae < best_val_mae - 1e-8:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 15:
                break
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        test_mae_sum, test_n = 0.0, 0
        with torch.no_grad():
            for batch in test_loader_local:
                e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
                test_mae_sum += (e_pred - batch["y"].view(-1)).abs().sum().item()
                test_n += batch["y"].shape[0]
        return model, test_mae_sum / test_n

    def get_embeddings_local(model, indices):
        loader = make_dataloader(dataset_local, indices, BATCH, shuffle=False)
        model.eval()
        embs = []
        with torch.no_grad():
            for batch in loader:
                _, node_feats = model(batch["z"], batch["pos"], batch["batch"])
                batch_idx = batch["batch"]
                n_structs = batch_idx.max().item() + 1
                for s in range(n_structs):
                    mask = batch_idx == s
                    embs.append(node_feats[mask].mean(dim=0).cpu().numpy())
        return np.array(embs)

    exp_results = {}
    for strategy_name, select_fn in STRATEGIES.items():
        torch.manual_seed(exp_seed)
        np.random.seed(exp_seed)

        labeled = list(init_idx)
        pool = list(pool_idx)
        labeled_embs = None
        curve = []

        for iteration in range(N_ITER + 1):
            models = {}
            for model_seed in [exp_seed, exp_seed + 100]:
                torch.manual_seed(model_seed)
                model, test_mae = train_model_local(labeled)
                models[model_seed] = model

            curve.append(test_mae)
            if iteration >= N_ITER or len(pool) < N_QUERY:
                break

            selected = select_fn(np.array(pool), N_QUERY, models, labeled_embs)
            for s in selected:
                if s in pool:
                    pool.remove(int(s))
                    labeled.append(int(s))
            labeled_embs = get_embeddings_local(list(models.values())[0], labeled)

        exp_results[strategy_name] = curve
    return exp_results

# ---------------------------------------------------------------------------
# Run multiple seeds
# ---------------------------------------------------------------------------
print(f"Running {N_SEEDS} seeds x {len(STRATEGIES)} strategies x {N_ITER+1} iterations...")
all_seeds_results = []

for run_i in range(N_SEEDS):
    seed = BASE_SEED + run_i * 10
    print(f"\n{'#'*50}")
    print(f"#  Seed {seed} (run {run_i+1}/{N_SEEDS})")
    print(f"{'#'*50}")
    t0 = time.time()
    seed_results = run_one_experiment(seed)
    all_seeds_results.append(seed_results)
    print(f"  Run time: {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# Aggregate & Report
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  AGGREGATE RESULTS (mean ± std over {N_SEEDS} seeds)")
print(f"{'='*60}")

import pandas as pd
strategy_names = list(STRATEGIES.keys())
all_finals = {s: [] for s in strategy_names}

for seed_res in all_seeds_results:
    for s in strategy_names:
        curve = seed_res[s]
        all_finals[s].append(curve[-1] if curve else float("nan"))

print(f"\n{'Strategy':<30} {'Final MAE':>16} {'Best MAE':>16}")
print("-" * 62)
for s in strategy_names:
    finals = np.array(all_finals[s])
    mu, std = np.nanmean(finals), np.nanstd(finals)
    # Aggregate best across seeds
    bests = [np.min(seed_res[s]) for seed_res in all_seeds_results if s in seed_res]
    best_mu, best_std = np.nanmean(bests), np.nanstd(bests)
    print(f"  {s:<28} {mu:.4f} ± {std:.4f}   {best_mu:.4f} ± {best_std:.4f}")

# Save aggregate learning curves
max_iter = N_ITER + 1
agg_data = {}
for s in strategy_names:
    curves_matrix = []
    for seed_res in all_seeds_results:
        c = seed_res.get(s, [])
        curves_matrix.append(c)
    # Pad to same length
    max_len = max(len(c) for c in curves_matrix)
    padded = np.full((len(curves_matrix), max_len), np.nan)
    for i, c in enumerate(curves_matrix):
        padded[i, :len(c)] = c
    agg_data[s] = padded

df_curves = pd.DataFrame({
    s: np.nanmean(agg_data[s], axis=0) for s in strategy_names
})
df_curves.index.name = "iteration"
df_curves.to_csv("results/fast_experiment_curves.csv")

df_std = pd.DataFrame({
    s: np.nanstd(agg_data[s], axis=0) for s in strategy_names
})
df_std.to_csv("results/fast_experiment_std.csv")

# Print final comparison
print(f"\n  Results saved: fast_experiment_curves.csv, fast_experiment_std.csv")
print(f"  Done! ({len(all_seeds_results)} seeds completed)")
