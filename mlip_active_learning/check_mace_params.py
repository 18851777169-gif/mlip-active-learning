from mace.calculators import MACECalculator
calc = MACECalculator(model_path="/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model", device="cuda", default_dtype="float32")
m = calc.models[0]
print("=== All parameter names ===")
for n, p in m.named_parameters():
    print(f"  {n}: {list(p.shape)}")

print("\n=== Matching our patterns ===")
for pattern in ["scale", "shift", "readout", "atomic", "products.1", "interactions.1.linear_up"]:
    matches = [n for n, p in m.named_parameters() if pattern in n.lower()]
    print(f"  '{pattern}': {matches}")
