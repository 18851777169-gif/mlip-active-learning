import pickle, sys, time
sys.path.insert(0, ".")
from data import MaterialDataset, make_dataloader, create_splits
from model_fallback import FallbackModel
import torch, numpy as np

# Test each system
for sys_name in ["liquid_water", "zeolite", "FeNiCrCoCu_HEA", "MgO_surface"]:
    path = f"data/ms25_labeled/{sys_name}.pkl"
    with open(path, "rb") as f:
        structs = pickle.load(f)
    energies = [s.info["energy"] for s in structs]
    print(f"{sys_name}: {len(structs)} structs, "
          f"E=[{min(energies):.0f},{max(energies):.0f}], sizes=[{len(structs[0])},{len(structs[-1])}]")

# Quick training test on water
print("\nQuick training test (water, 50 structs)...")
with open("data/ms25_labeled/liquid_water.pkl", "rb") as f:
    structs = pickle.load(f)
dataset = MaterialDataset(structs)
init, pool, test, val = create_splits(len(dataset), 50, 0.15, 0.10, 42)
model = FallbackModel(hidden_channels=64, num_interactions=2)
loader = make_dataloader(dataset, init, 16, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
for ep in range(3):
    for batch in loader:
        opt.zero_grad()
        e_pred, _ = model(batch["z"], batch["pos"], batch["batch"])
        loss = torch.nn.functional.l1_loss(e_pred, batch["y"].view(-1))
        loss.backward()
        opt.step()
    print(f"  Epoch {ep}, loss={loss.item():.1f}")
print(f"Training OK! ({time.time()-t0:.0f}s)")
