#!/bin/bash
source /share/home/tm949679661250000/a954358970/gpumace/setup_env.sh
cd /share/home/tm949679661250000/a954358970/gpumace/mlip_active_learning

SYSTEMS="FeNiCrCoCu_HEA MgO_surface Pt_CH_activation Zr_oxide_amorphous liquid_water zeolite"

for seed in 42 52 62; do
    for sys in $SYSTEMS; do
        LOG="results/ms25_${sys}_seed${seed}.log"
        echo "Start $sys seed=$seed"
        nohup python3 -u -c "
import sys; sys.argv = ['', '$seed', '$sys']
exec(open('run_ms25_experiment.py').read())
" > "$LOG" 2>&1 &
    done
done
echo "All 18 processes launched"
wait
echo "ALL DONE"
