import re, numpy as np, pandas as pd

BASE = "results_mpaloe"
systems = ['mp-570316','mp-22046','mp-729184','mp-632401']
strategies = ['A_random','G_hybrid_weighted','I_aud_rank']
seeds = [42,52,62]

print("=" * 70)
print("MP-ALOE CROSS-VALIDATION RESULTS (parsed from logs)")
print("=" * 70)

all_rows = []

for mid in systems:
    print(f"\n--- {mid} ---")
    for s in strategies:
        vals = []
        for seed in seeds:
            log_path = f"{BASE}/{mid}_seed{seed}.log"
            try:
                with open(log_path) as f:
                    text = f.read()
                # Extract block for this strategy
                pattern = rf"--- {re.escape(s)} ---\n(.*?)(?:\n---|\nDone|\Z)"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    block = match.group(1)
                    maes = re.findall(r"MAE=([\d.]+)", block)
                    if maes:
                        vals.append(float(maes[-1]))
            except FileNotFoundError:
                print(f"  MISSING: {log_path}")
        if vals:
            mu = np.mean(vals)
            std = np.std(vals)
            all_rows.append({'system': mid, 'strategy': s,
                             'mean_mae': mu, 'std_mae': std,
                             'seeds': len(vals), 'raw_maes': vals})
            print(f"  {s:25s} MAE={mu:.4f} +/- {std:.4f}  ({len(vals)} seeds: {[f'{v:.4f}' for v in vals]})")

# Compute improvement over random
print("\n" + "=" * 70)
print("IMPROVEMENT OVER A_RANDOM")
print("=" * 70)

for mid in systems:
    rows_mid = [r for r in all_rows if r['system'] == mid]
    rb_row = next((r for r in rows_mid if r['strategy'] == 'A_random'), None)
    if rb_row is None:
        continue
    rb = rb_row['mean_mae']
    print(f"\n{mid} (baseline A_random MAE={rb:.4f}):")
    for r in rows_mid:
        if r['strategy'] == 'A_random':
            continue
        imp = (rb - r['mean_mae']) / rb * 100
        better = "BETTER" if imp > 0 else "WORSE"
        print(f"  {r['strategy']:25s} MAE={r['mean_mae']:.4f}  Δ={imp:+.1f}%  {better}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: % systems where G_hybrid > A_random")
print("=" * 70)
g_wins = 0
for mid in systems:
    rows_mid = [r for r in all_rows if r['system'] == mid]
    rb_row = next((r for r in rows_mid if r['strategy'] == 'A_random'), None)
    g_row = next((r for r in rows_mid if r['strategy'] == 'G_hybrid_weighted'), None)
    if rb_row and g_row:
        if g_row['mean_mae'] < rb_row['mean_mae']:
            g_wins += 1
            print(f"  {mid}: G_hybrid BETTER ({g_row['mean_mae']:.4f} < {rb_row['mean_mae']:.4f})")
        else:
            print(f"  {mid}: G_hybrid WORSE  ({g_row['mean_mae']:.4f} >= {rb_row['mean_mae']:.4f})")
print(f"\n  G_hybrid wins: {g_wins}/{len(systems)} = {g_wins/len(systems):.1%}")

# Save
df = pd.DataFrame(all_rows)
df.to_csv(f'{BASE}/mpaloe_aggregated.csv', index=False)
print(f"\nSaved: {BASE}/mpaloe_aggregated.csv")
print("Done.")
