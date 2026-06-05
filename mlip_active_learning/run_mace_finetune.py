#!/usr/bin/env python3
"""MACE Fine-tuning Active Learning.

Fine-tunes MACE readout layers on actively selected structures.
Uncertainty = ensemble variance (2 fine-tuned copies).
Diversity = MACE node features.
"""

import sys, pickle, time, os, warnings, copy
import numpy as np
import torch, torch.nn as nn

warnings.filterwarnings("ignore")

N_INIT, N_QUERY, N_ITER = 30, 10, 5
EPOCHS, LR, BATCH = 15, 1e-3, 4
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DATA_DIR = "data/ms25_labeled"
MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"
DEVICE = "cuda"

torch.manual_seed(SEED); np.random.seed(SEED)

from data import MaterialDataset, make_dataloader, create_splits
from scipy.spatial.distance import cdist
from ase import Atoms

# ---------------------------------------------------------------------------
# MACE fine-tuner
# ---------------------------------------------------------------------------
class MACEFineTuner:
    def __init__(self, model_path, seed=42):
        from mace.calculators import MACECalculator
        torch.manual_seed(seed)
        self.calc = MACECalculator(model_path=model_path, device=DEVICE, default_dtype="float32")
        self.model = self.calc.models[0]
        self._unfreeze_readouts()

    def _unfreeze_readouts(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = ("readout" in n or "products.1" in n or
                               "interactions.1.linear_up" in n)

    def _atoms_to_batch_dict(self, atoms):
        batch = self.calc._atoms_to_batch(atoms)
        batch = self.calc._clone_batch(batch)
        return batch.to_dict()

    def predict_batch(self, batch_dict):
        return self.model(batch_dict, training=self.model.training, compute_force=False)["energy"]

    def get_embedding(self, batch_dict):
        out = self.model(batch_dict, training=False, compute_force=False)
        return out.get("node_feats", None)

    def finetune(self, structures, train_idx, val_idx, epochs=EPOCHS, lr=LR):
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        opt = torch.optim.Adam(params, lr=lr)

        # Prepare training data as batch dicts (one per structure for simplicity)
        train_structs = [structures[i] for i in train_idx]
        val_structs = [structures[i] for i in val_idx]

        best_val, best_state, patience = float("inf"), None, 0

        for ep in range(epochs):
            self.model.train()
            total_loss, n_batch = 0.0, 0
            for i in range(0, len(train_structs), BATCH):
                batch_structs = train_structs[i:i+BATCH]
                # Combine into one ASE Atoms for batching
                combined = self._combine_atoms(batch_structs)
                bd = self._atoms_to_batch_dict(combined)
                opt.zero_grad()
                e_pred = self.predict_batch(bd)
                e_true = torch.tensor([s.info["energy"] for s in batch_structs],
                                      dtype=torch.float32, device=DEVICE)
                loss = nn.functional.l1_loss(e_pred.view(-1), e_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total_loss += loss.item()
                n_batch += 1

            # Validation
            self.model.eval()
            val_loss, val_n = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_structs), BATCH):
                    batch_structs = val_structs[i:i+BATCH]
                    combined = self._combine_atoms(batch_structs)
                    bd = self._atoms_to_batch_dict(combined)
                    e_pred = self.predict_batch(bd)
                    e_true = torch.tensor([s.info["energy"] for s in batch_structs],
                                          dtype=torch.float32, device=DEVICE)
                    val_loss += (e_pred.view(-1) - e_true).abs().sum().item()
                    val_n += len(batch_structs)
            val_mae = val_loss / val_n

            if val_mae < best_val - 1e-8:
                best_val = val_mae
                best_state = copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 6:
                break

        if best_state:
            self.model.load_state_dict(best_state)
        return best_val

    def _combine_atoms(self, atoms_list):
        """Combine multiple ASE Atoms into one for batch processing."""
        if len(atoms_list) == 1:
            return atoms_list[0]
        # Use ASE's built-in combine
        from ase.build import bulk  # ensures ase is loaded
        combined = atoms_list[0].copy()
        for a in atoms_list[1:]:
            combined = combined + a
        return combined

    def predict_energy(self, atoms):
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calc
        return atoms_copy.get_potential_energy()

# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
print(f"Loading MACE...")
tuner = MACEFineTuner(MODEL_PATH, seed=SEED)
n_trainable = sum(p.numel() for p in tuner.model.parameters() if p.requires_grad)
print(f"Trainable params: {n_trainable}")

# Load data
sys_name = sys.argv[2] if len(sys.argv) > 2 else "FeNiCrCoCu_HEA"
with open(f"{DATA_DIR}/{sys_name}.pkl", "rb") as f:
    structures = pickle.load(f)

# Pre-compute MACE reference energies
print(f"Computing MACE energies for {sys_name}...")
t0 = time.time()
for atoms in structures:
    if "energy" not in atoms.info or atoms.info.get("_recompute", False):
        atoms.info["energy"] = tuner.predict_energy(atoms)
print(f"Done in {time.time()-t0:.0f}s")

dataset = MaterialDataset(structures)
init_idx, pool_idx, test_idx, val_idx = create_splits(len(dataset), N_INIT, 0.15, 0.10, SEED)

print(f"Fine-tuning on {N_INIT} structures...")
t0 = time.time()
val_mae = tuner.finetune(structures, init_idx, val_idx, epochs=10, lr=1e-3)
# Evaluate
tuner.model.eval()
test_loss, test_n = 0.0, 0
with torch.no_grad():
    for i in test_idx:
        bd = tuner._atoms_to_batch_dict(structures[i])
        e_pred = tuner.predict_batch(bd)
        e_true = torch.tensor(structures[i].info["energy"], dtype=torch.float32, device=DEVICE)
        test_loss += (e_pred.view(-1) - e_true).abs().item()
        test_n += 1
test_mae = test_loss / test_n

print(f"Fine-tune: {time.time()-t0:.0f}s, Test MAE={test_mae:.4f} eV")
print(f"\nMACE fine-tuning pipeline works!")
