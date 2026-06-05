#!/bin/bash
source /share/home/tm949679661250000/a954358970/gpumace/setup_env.sh
cd /share/home/tm949679661250000/a954358970/gpumace/mlip_active_learning
for seed in 42 52 62; do
  for sys in FeNiCrCoCu_HEA MgO_surface Pt_CH_activation Zr_oxide_amorphous liquid_water zeolite; do
    nohup python3 -u run_9strat.py $seed $sys > results/ms25_9strat_${sys}_seed${seed}.log 2>&1 &
    echo "Launched $sys seed=$seed"
  done
done
echo "All 18 launched"
