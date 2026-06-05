"""Debug and fix MACE K/L crash."""
import sys, pickle, torch, copy
sys.path.insert(0, ".")

MODEL_PATH = "/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model"
DEVICE = "cuda"

from data import MaterialDataset, create_splits

with open("data/ms25_labeled/FeNiCrCoCu_HEA.pkl", "rb") as f:
    structures = pickle.load(f)
dataset = MaterialDataset(structures)
init_idx, _, test_idx, val_idx = create_splits(len(dataset), 50, 0.15, 0.10, 42)

from run_mace_al import MACEFineTuner

print("=== Simulating multi-strategy fine-tuning ===")
for strat_idx in range(8):
    print(f"\nStrategy {strat_idx}: creating fresh tuner...")
    torch.cuda.empty_cache()
    ft = MACEFineTuner(MODEL_PATH, seed=42+strat_idx)
    print(f"  Tuner created, trainable={ft._n_trainable}")

    # Check if params actually have requires_grad
    trainable = [n for n, p in ft.model.named_parameters() if p.requires_grad]
    print(f"  Trainable names: {len(trainable)}")
    if not trainable:
        print(f"  ERROR: No trainable params!")
        break

    # Fine-tune
    try:
        val_mae = ft.finetune(structures, init_idx, val_idx, epochs=1, lr=1e-4)
        print(f"  Fine-tune OK, val_mae={val_mae:.4f}")

        # Test evaluate
        mae = ft.evaluate(structures, test_idx)
        print(f"  Evaluate OK, test_mae={mae:.4f}")

        # Test force computation (K-style)
        a = structures[init_idx[0]].copy()
        a.calc = ft.calc
        f = a.get_forces()
        print(f"  Force OK, max|f|={abs(f).max():.3f}")

        # Test embedding (I/J-style)
        emb = ft.get_embedding(structures[init_idx[0]])
        print(f"  Embedding OK, shape={emb.shape if emb is not None else 'None'}")

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        break

    del ft
    torch.cuda.empty_cache()

print("\nDone!")
