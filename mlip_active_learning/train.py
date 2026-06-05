"""Training loop for MACE MLIP model.

Supports:
  - Single model training
  - Ensemble training (multiple seeds)
  - Fine-tuning from pretrained MACE-MP-0
  - Early stopping with patience
  - EMA loss tracking
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import time

from model import MACEWrapper, EnsembleMACE
from metrics import evaluate_model


def train_single_model(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    config,
    model_name: str = "model",
) -> Dict:
    """Train a single MACE model.

    Returns:
        dict with training history: train_losses, val_maes, best_epoch, etc.
    """
    device = config.device
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=config.patience // 2, min_lr=1e-6)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = {"train_loss": [], "val_mae": [], "lr": []}

    for epoch in range(config.max_epochs):
        # Training
        model.train()
        epoch_losses = []

        for batch_data in train_dataloader:
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_data.items()}

            optimizer.zero_grad()
            energy_pred, forces_pred, _ = model(batch_data)

            # Energy loss
            energy_true = batch_data["y"].view(-1)
            e_loss = nn.functional.l1_loss(energy_pred, energy_true)

            # Force loss
            f_loss = torch.tensor(0.0, device=device)
            forces_true = batch_data.get("forces", None)
            if forces_pred is not None and forces_true is not None:
                f_loss = nn.functional.l1_loss(forces_pred, forces_true)

            loss = e_loss + config.force_weight * f_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        history["train_loss"].append(avg_loss)

        # Validation
        val_results = evaluate_model(model, val_dataloader, device)
        val_mae = val_results["energy_mae"]
        history["val_mae"].append(val_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_mae)

        # Early stopping
        if val_mae < best_val_mae - 1e-8:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 0 or epoch % 5 == 0 or epoch == best_epoch:
            print(f"    Epoch {epoch:3d} | Loss {avg_loss:.6f} | "
                  f"Val MAE {val_mae:.6f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= config.patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_val_mae"] = best_val_mae

    return history


def train_ensemble(
    config,
    train_dataloader,
    val_dataloader,
) -> EnsembleMACE:
    """Train an ensemble of MACE models with different random seeds.

    Returns:
        EnsembleMACE with all members trained.
    """
    ensemble = EnsembleMACE(
        ensemble_size=config.ensemble_size,
        seeds=config.ensemble_seeds,
        model_name=config.mace_model,
        pretrained=config.pretrained,
        r_max=config.r_max,
        dtype=config.mace_dtype,
        device=config.device,
        use_mace=config.use_mace,
    )

    for i, seed in enumerate(config.ensemble_seeds[:config.ensemble_size]):
        print(f"  Training ensemble member {i+1}/{config.ensemble_size} (seed={seed})...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        history = train_single_model(
            ensemble.members[i],
            train_dataloader,
            val_dataloader,
            config,
            model_name=f"ensemble_{i}",
        )
        print(f"    Best val MAE: {history['best_val_mae']:.6f}")

    return ensemble


def save_checkpoint(model, path: str):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model_class, path: str, **kwargs):
    """Load model from checkpoint."""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
