from mace.calculators import MACECalculator
calc = MACECalculator(
    model_path="/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model",
    device="cuda", default_dtype="float32")
m = calc.models[0]
total = 0
for n, p in m.named_parameters():
    shape = list(p.shape)
    num = p.numel()
    total += num
    print(f"{n}: {shape}  ({num:,})")
print(f"\nTotal: {total:,} params")
