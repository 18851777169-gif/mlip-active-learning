#!/bin/bash
source /share/home/tm949679661250000/a954358970/gpumace/setup_env.sh
cd /share/home/tm949679661250000/a954358970/gpumace/mlip_active_learning

STRATS="A_random C_uncertainty E_diversity G_hybrid_weighted I_aud_rank J_aud_batch K_aud_bald L_rho_diagnostic"

for seed in 42 52 62; do
  for sys in FeNiCrCoCu_HEA MgO_surface Pt_CH_activation Zr_oxide_amorphous liquid_water zeolite; do
    (
      for strat in $STRATS; do
        echo "=== $sys seed=$seed strat=$strat ==="
        timeout 1800 python3 -u run_mace_single.py $seed $sys $strat 2>&1 | grep -E "(Strategy:|Iter|Saved|Error|Trace)"
        RC=$?
        if [ $RC -ne 0 ]; then
          echo "FAILED: $sys seed=$seed strat=$strat (exit $RC)" >> results/mace_failures.log
        fi
      done
      echo "DONE: $sys seed=$seed"
    ) &
  done
  wait
  echo "=== Seed $seed complete ==="
done
echo "ALL DONE"
