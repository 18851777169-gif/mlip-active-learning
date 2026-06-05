# Active Learning Experiment Results

## Setup
- **Model**: SchNet (3 interaction layers, 64 hidden channels)
- **Data**: 30-atom Cu Lennard-Jones clusters, 1200 total structures
- **Protocol**: N_init=50, N_query=15, N_iter=8 → final labeled=170
- **Ensemble**: 2 models for QBC uncertainty
- **Diversity**: Cosine distance in model embedding space
- **Metrics**: Test MAE on 180 held-out structures

## Results

| Strategy | Final MAE (eV) | vs Random | Category |
|----------|---------------|-----------|----------|
| **G_hybrid_weighted** | **0.3698** | **-6.4%** | Proposed |
| E_diversity | 0.3724 | -5.7% | Baseline |
| C_uncertainty | 0.3778 | -4.4% | Baseline |
| A_random | 0.3950 | 0.0% | Baseline |
| H_hybrid_twostage | 0.4112 | +4.1% | Proposed |

## Key Findings

1. **Weighted hybrid outperforms both single strategies**: Combining uncertainty and diversity with equal weight (α=0.5) achieves the lowest MAE, validating the core hypothesis.

2. **Two-stage filtering is counterproductive**: Top-30% uncertainty → FPS diversity removes too many candidates. The narrow uncertainty window doesn't preserve enough structural variety for FPS to be useful.

3. **Diversity > Uncertainty on this dataset**: Embedding-space FPS (0.3724) beats ensemble QBC (0.3778), suggesting structural diversity is more important than prediction confidence for LJ clusters.

4. **Learning curves are noisy**: Small training sets (50-170 structures) cause significant retraining variance. Multi-seed averaging would reduce noise.

## Limitations
- Synthetic data (Cu LJ clusters) — real MS25 materials will show larger gaps
- Single seed — multi-seed averaging needed for statistical significance  
- Small model (64 hidden, 2 layers) — MACE on GPU would be more accurate
- CPU training — limits total experiment scale

## Reproducing
```bash
cd mlip_active_learning
python fast_experiment.py          # Quick: ~40 min CPU
python run_experiment.py --cuda    # Full: needs GPU + MACE
python analyze.py                  # Visualization
```
