#!/usr/bin/env python3
"""Cost-benefit analysis: DFT hours saved vs AL overhead."""
import pandas as pd, numpy as np

# Experimental data
# SchNet + MACE experiments: average best strategy improvement
# Assume DFT cost ~10 CPU-hours/structure, AL overhead ~1h per iteration
DFT_COST_PER_STRUCT = 10  # CPU-hours per DFT calculation
AL_OVERHEAD_PER_ITER = 0.5  # GPU-hours per AL iteration (training + acquisition)

# Load MACE results to compute savings
S = ["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
seeds = [42,52,62]

print("=" * 60)
print("COST-BENEFIT ANALYSIS: DFT Savings from Active Learning")
print("=" * 60)

rows = []
for sn in S:
    # Load MACE AL results
    best_improvs = []
    for seed in seeds:
        try:
            df = pd.read_csv(f"results/mace_al_{sn}_seed{seed}.csv")
            rand = df["A_random"].iloc[-1]
            best = rand
            for c in df.columns:
                if c != "A_random" and df[c].iloc[-1] < best:
                    best = df[c].iloc[-1]
            best_improvs.append((rand - best) / rand * 100 if rand > 0 else 0)
        except: pass

    avg_imp = np.mean(best_improvs) if best_improvs else 0

    # Cost model: to reach same MAE
    # Without AL: need N_random structures
    # With AL: need N_random * (1 - imp%/100) structures
    # Each AL experiment runs N_iter iterations
    N_AL = 7  # iterations
    N_structures = 170  # total labeled

    dft_without = N_structures * DFT_COST_PER_STRUCT
    al_overhead = N_AL * 2 * AL_OVERHEAD_PER_ITER  # 2 models per iter
    dft_saved = N_structures * (avg_imp / 100) * DFT_COST_PER_STRUCT
    net_saving = dft_saved - al_overhead

    rows.append({
        'system': sn,
        'avg_improvement_pct': avg_imp,
        'dft_without_AL': dft_without,
        'al_compute_cost': al_overhead,
        'dft_hours_saved': dft_saved,
        'net_saving_hours': net_saving,
        'roi': net_saving / al_overhead if al_overhead > 0 else 0,
    })

res = pd.DataFrame(rows)
res.to_csv("results/cost_benefit.csv", index=False)

print(f"\n{'System':<25s} {'Improve':>8s} {'DFT Saved':>10s} {'AL Cost':>8s} {'Net':>8s} {'ROI':>6s}")
print("-" * 68)
for _, r in res.iterrows():
    print(f"{r['system']:<25s} {r['avg_improvement_pct']:+6.1f}% {r['dft_hours_saved']:>8.0f}h "
          f"{r['al_compute_cost']:>6.0f}h {r['net_saving_hours']:+7.0f}h {r['roi']:>5.1f}x")

print(f"\nTotal DFT saved: {res['dft_hours_saved'].sum():.0f} hours")
print(f"Total AL cost: {res['al_compute_cost'].sum():.0f} hours")
print(f"Net saving: {res['net_saving_hours'].sum():.0f} hours")
print(f"Saved results/cost_benefit.csv")
