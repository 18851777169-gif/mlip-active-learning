#!/bin/bash
source /share/home/tm949679661250000/a954358970/gpumace/setup_env.sh
cd /share/home/tm949679661250000/a954358970/gpumace/mlip_active_learning
for seed in 42 52 62; do
  for sys in FeNiCrCoCu_HEA Pt_CH_activation liquid_water zeolite; do
    nohup python3 -u run_ms25_experiment.py $seed $sys > results/ms25_8strat_${sys}_seed${seed}.log 2>&1 &
    echo "Launched $sys seed=$seed"
  done
done
echo "All 12 launched"
