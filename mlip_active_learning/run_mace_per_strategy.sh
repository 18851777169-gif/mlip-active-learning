#!/bin/bash
# Run MACE AL with each strategy in a fresh process
source /share/home/tm949679661250000/a954358970/gpumace/setup_env.sh
cd /share/home/tm949679661250000/a954358970/gpumace/mlip_active_learning

SEED=$1
SYS=$2

ALL_STRATS="A_random C_uncertainty E_diversity G_hybrid_weighted I_aud_rank J_aud_batch K_aud_bald L_rho_diagnostic"

for strat in $ALL_STRATS; do
    echo "=== Strategy $strat (seed=$SEED sys=$SYS) ==="
    timeout 3600 python3 -u -c "
import sys, pickle, torch, numpy as np
sys.argv = ['', '$SEED', '$SYS', '$strat']
torch.cuda.empty_cache()

# Re-initialize everything fresh
from data import MaterialDataset, create_splits
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from ase import Atoms
import copy

MODEL_PATH = '/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model'
DEVICE = 'cuda'
SEED = $SEED
DATA_DIR = 'data/ms25_labeled'
N_INIT, N_QUERY, N_ITER = 50, 15, 6
EPOCHS, LR, BATCH = 20, 5e-4, 4

torch.manual_seed(SEED); np.random.seed(SEED)

sys_name = '$SYS'
target_strat = '$strat'

# Run one strategy
exec(open('run_mace_al_single.py').read())
print(f'DONE: {target_strat}')
" 2>&1
    RC=$?
    echo "Exit code: $RC"
    if [ $RC -ne 0 ]; then
        echo "Strategy $strat FAILED, continuing to next..."
    fi
    # Cleanup between strategies
    sleep 2
done
echo "ALL STRATEGIES DONE for seed=$SEED sys=$SYS"
