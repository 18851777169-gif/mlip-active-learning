#!/bin/bash
source /share/home/tm949679661250000/a954358970/gpumace/setup_env.sh
cd /share/home/tm949679661250000/a954358970/gpumace/mlip_active_learning
for seed in 42 52 62; do
    echo "=== Seed $seed ==="
    python3 -u run_ms25_experiment.py $seed 2>&1 | grep -E '(Iter 0|Iter 8|Seed|CROSS|SUMMARY|better|Done)'
done
