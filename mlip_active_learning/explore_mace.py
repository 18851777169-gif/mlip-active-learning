"""Explore MACE calculator internals to understand training API."""
from mace.calculators import MACECalculator
import inspect, torch

# 1. Check calculate method
src = inspect.getsource(MACECalculator.calculate)
print("=== calculate (first 2k chars) ===")
print(src[:2000])

# 2. Check model forward signature
print("\n=== ScaleShiftMACE.forward signature ===")
from mace.modules.models import ScaleShiftMACE
sig = inspect.signature(ScaleShiftMACE.forward)
print(sig)

# 3. Check model parameters
print("\n=== Model parameter names ===")
calc = MACECalculator(
    model_path="/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model",
    device="cuda", default_dtype="float32"
)
model = calc.models[0]
for i, (name, p) in enumerate(model.named_parameters()):
    print(f"  {name}: {list(p.shape)} requires_grad={p.requires_grad}")
    if i > 20:
        break

# 4. Try to call model.forward with edge_index
print("\n=== Testing direct model.forward ===")
import numpy as np
from ase.build import molecule
atoms = molecule("H2O")
atoms.calc = calc
e_ref = atoms.get_potential_energy()
print(f"ASE reference energy: {e_ref:.4f} eV")
print("ASE path works, now need to trace graph preparation...")
