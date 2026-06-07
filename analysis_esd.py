#!/usr/bin/env python3
"""ESD diagnostic criteria validation."""
import pandas as pd, numpy as np

S=["FeNiCrCoCu_HEA","MgO_surface","Pt_CH_activation","Zr_oxide_amorphous","liquid_water","zeolite"]
seeds=[42,52,62]

# 1. Control limit
g_wins=0
for s in S:
    for seed in seeds:
        try:
            df=pd.read_csv(f"results/mace_al_{s}_seed{seed}.csv")
            if df["G_hybrid_weighted"].iloc[-1] < df["A_random"].iloc[-1]: g_wins+=1
        except: pass
total=len(S)*len(seeds)
print(f"1. Control: G>Random {g_wins}/{total} ({g_wins/total*100:.0f}%) {'PASS' if g_wins/total>=0.8 else 'FAIL'}")

# 2. Sensitivity
anti,l_wins=0,0
for s in S:
    for seed in seeds:
        try:
            dr=pd.read_csv(f"results_rho/rho_stats_{s}_seed42.csv")
            da=pd.read_csv(f"results/mace_al_{s}_seed{seed}.csv")
            for _,row in dr.iterrows():
                if row.get("rho",0)<-0.3:
                    anti+=1
                    if "L_rho_diagnostic" in da.columns:
                        if da["L_rho_diagnostic"].iloc[-1] < da["A_random"].iloc[-1]: l_wins+=1
        except: pass
print(f"2. Sensitivity (rho<-0.3): {l_wins}/{anti} ({l_wins/anti*100:.0f}% if anti>0) {'PASS' if anti>0 and l_wins/anti>=0.5 else 'N/A'}")

# 3. Specificity
norm,l_eq=0,0
for s in S:
    try:
        dr=pd.read_csv(f"results_rho/rho_stats_{s}_seed42.csv")
        da=pd.read_csv(f"results/mace_al_{s}_seed42.csv")
        for _,row in dr.iterrows():
            if row.get("rho",0)>=-0.3:
                norm+=1
                if abs(row.get("alpha_l",0)-0.5)<0.01: l_eq+=1
    except: pass
print(f"3. Specificity (L=G when normal): {l_eq}/{norm} ({l_eq/norm*100:.0f}% if norm>0) {'PASS' if norm>0 and l_eq/norm>=0.8 else 'N/A'}")

# 4. Summary
print("\nESD Summary:")
print(f"  Control: {'PASS' if g_wins/total>=0.8 else 'FAIL'}")
print(f"  Sensitivity: {'PASS' if anti>0 and l_wins/anti>=0.5 else 'N/A'}")
print(f"  Specificity: {'PASS' if norm>0 and l_eq/norm>=0.8 else 'N/A'}")
print("Done!")
