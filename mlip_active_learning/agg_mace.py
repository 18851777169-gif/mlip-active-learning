import pandas as pd, numpy as np
seeds = [42, 52, 62]
strats = ["A_random","C_uncertainty","E_diversity","G_hybrid_weighted","H_hybrid_twostage"]
all_best = {s: [] for s in strats}
for seed in seeds:
    df = pd.read_csv(f"results/mace_experiment_seed{seed}.csv")
    for s in strats:
        all_best[s].append(np.min(df[s].values))
rand_best = np.mean(all_best["A_random"])
print("Aggregate (3 seeds) - MACE-labeled Cu clusters:")
print(f"{'Strategy':<28} {'Best':>12} {'vs Random':>10}")
print("-" * 52)
for s in strats:
    b = np.array(all_best[s])
    imp = (rand_best - b.mean()) / rand_best * 100
    print(f"{s:<28} {b.mean():.2f}+/-{b.std():.2f}  {imp:+.1f}%")
ranked = sorted(strats, key=lambda s: np.mean(all_best[s]))
print("\nRanking:")
for i, s in enumerate(ranked):
    print(f"  {i+1}. {s}: {np.mean(all_best[s]):.2f}")
