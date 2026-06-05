## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-05-31
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Title**: Hybrid Acquisition Function for Active Learning of Machine Learning Interatomic Potentials
- **Objective**: Demonstrate that combining uncertainty-based and diversity-based acquisition functions reduces the number of DFT calculations required to reach target accuracy, compared to either strategy alone
- **Hypothesis**: A hybrid acquisition strategy (uncertainty + diversity) achieves the same test MAE with fewer labeled samples than uncertainty-only or diversity-only baselines
- **Type**: training

## Research Question

Can a hybrid acquisition function that jointly considers model uncertainty AND structural diversity select more informative training samples for MLIPs than single-criterion strategies?

### Independent Variable
Type of acquisition function (6 levels):
1. **Random** — uniform random sampling (lower bound baseline)
2. **Uncertainty-only** — ensemble variance
3. **Diversity-only** — embedding space distance maximization
4. **Hybrid-Weighted** — α·unc + (1-α)·div (α ∈ {0.3, 0.5, 0.7})
5. **Hybrid-TwoStage(U→D)** — top K% by uncertainty, then pick most diverse
6. **Hybrid-Pareto** — select from Pareto frontier of (uncertainty, diversity)

### Dependent Variable
- **Primary**: Test MAE (energy) at each active learning iteration → learning curve
- **Secondary**: Number of labeled samples to reach target MAE (data efficiency ratio)

## Setup

- **Language/Framework**: Python 3.10+, PyTorch 2.x, PyTorch Geometric
- **Entry Command**: `python run_experiment.py`
- **Working Directory**: `./mlip_active_learning/`
- **Dependencies**: see requirements.txt
- **Environment**: CPU (small-scale proof of concept), optionally GPU

## Design

### Model
SchNet (Schütt et al., 2017) — message-passing neural network for molecular properties:
- 3 interaction layers, 128 hidden channels
- 50 Gaussian radial basis functions (cutoff = 10 Å)
- Energy prediction as sum of atomic contributions

### Uncertainty Estimation
Deep Ensemble (Lakshminarayanan et al., 2017):
- 3 independently-initialized SchNet models trained on same data
- Uncertainty = variance of energy predictions across ensemble

### Diversity Estimation
- Extract embedding vectors from penultimate layer of one ensemble member
- Mean-pool atomic embeddings to per-structure vector
- Diversity score = minimum cosine distance to any already-labeled structure

### Active Learning Protocol
```
1. Randomly select N_init = 100 structures as initial training set
2. Train ensemble on labeled set
3. For iteration = 1..20:
   a. Compute acquisition scores for all pool structures
   b. Select K = 20 structures with highest scores
   c. Add selected structures to labeled set (simulating DFT)
   d. Retrain ensemble
   e. Evaluate on fixed test set → record MAE
```

## Inputs

| Input | Path | Description |
|-------|------|-------------|
| QM9 dataset | auto-download via PyG | 134k small organic molecules with DFT energies |
| (Optional) MP-ALOE | auto-download | Materials Project active learning benchmark |

## Expected Outputs

| Output | Path | Format | Success Criterion |
|--------|------|--------|------------------|
| Learning curves | results/learning_curves.csv | CSV | 6 rows × 21 columns (iteration 0-20) |
| Data efficiency table | results/efficiency.csv | CSV | Relative DFT savings per method |
| Visualization | results/learning_curves.png | PNG | Clear separation between methods |
| Model checkpoints | checkpoints/ | .pt files | Ensemble for best method saved |

## Monitoring Configuration

- **Timeout**: 4 hours (full experiment)
- **Monitor files**: results/learning_curves.csv
- **Experiment type override**: training
- **Metric file**: results/learning_curves.csv

## Analysis Plan

- **Primary metric**: Area under the learning curve (MAE vs N_labeled) — lower is better
- **Success threshold**: Hybrid method achieves same MAE as best single-criterion method with ≥20% fewer labeled samples
- **Comparison**: Friedman test (non-parametric) across acquisition methods, followed by pairwise Wilcoxon with Holm correction
- **Effect size**: Cliff's delta for pairwise comparisons against baselines
