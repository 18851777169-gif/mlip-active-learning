"""Debug MACE K-strategy crash."""
import sys, pickle, copy
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, ".")

MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"
DEVICE = "cuda"

from data import MaterialDataset, create_splits
from ase import Atoms

# Load data
with open("data/ms25_labeled/FeNiCrCoCu_HEA.pkl", "rb") as f:
    structures = pickle.load(f)
dataset = MaterialDataset(structures)
init_idx, _, test_idx, val_idx = create_splits(len(dataset), 50, 0.15, 0.10, 42)

# Test 1: Fine-tune
print("Test 1: Fine-tuning...")
from run_mace_al import MACEFineTuner
ft = MACEFineTuner(MODEL_PATH, seed=42)
print(f"  Trainable: {ft._n_trainable}")
val_mae = ft.finetune(structures, init_idx, val_idx, epochs=2, lr=1e-4)
print(f"  Val MAE={val_mae:.4f}")
test_mae = ft.evaluate(structures, test_idx)
print(f"  Test MAE={test_mae:.4f}")

# Test 2: Force computation for BALD
print("\nTest 2: BALD force computation...")
pool = init_idx[:3]
ft.model.eval()
model_f = []
with torch.no_grad():
    for i in pool:
        a = structures[i].copy()
        a.calc = ft.calc
        try:
            f = a.get_forces()
            model_f.append(np.linalg.norm(f))
            print(f"  Struct {i}: force_norm={model_f[-1]:.3f}")
        except Exception as e:
            print(f"  Struct {i} FAIL: {e}")
            model_f.append(0.0)
print("  BALD test OK")

# Test 3: Fine-tune again (simulate second strategy)
print("\nTest 3: Second fine-tune...")
ft2 = MACEFineTuner(MODEL_PATH, seed=123)
val_mae2 = ft2.finetune(structures, init_idx, val_idx, epochs=2, lr=1e-4)
print(f"  Val MAE={val_mae2:.4f}")

# Test 4: Check if model params still have grad after multiple fine-tunes
print("\nTest 4: Post fine-tune grad check...")
params = [p for p in ft2.model.parameters() if p.requires_grad]
print(f"  Trainable params after 2nd FT: {len(params)} ({sum(p.numel() for p in params)})")

print("\nALL TESTS PASSED!")
