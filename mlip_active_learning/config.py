"""Configuration for MLIP active learning experiments.

Follows the RQ Brief & Methodology Blueprint:
  - 8 acquisition strategies (A-H)
  - 6 material systems from MS25
  - MACE equivariant MLIP framework
  - 5 evaluation metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Material systems (MS25 benchmark)
# ---------------------------------------------------------------------------
MATERIAL_SYSTEMS: Dict[str, dict] = {
    "MgO_surface": {
        "type": "ionic_crystal_surface",
        "description": "MgO(100) surface reconstruction",
        "challenge": "surface reconstruction",
        "n_atoms_typical": 64,
    },
    "liquid_water": {
        "type": "molecular_liquid",
        "description": "Liquid water at 300K",
        "challenge": "hydrogen bond network",
        "n_atoms_typical": 96,
    },
    "zeolite": {
        "type": "porous_framework",
        "description": "Zeolite CHA/FAU/LTA/MFI frameworks",
        "challenge": "large unit cell + multi-element",
        "n_atoms_typical": 144,
    },
    "Pt_CH_activation": {
        "type": "catalytic_metal_surface",
        "description": "Pt(111) C-H bond cleavage",
        "challenge": "reaction transition state",
        "n_atoms_typical": 72,
    },
    "FeNiCrCoCu_HEA": {
        "type": "high_entropy_alloy",
        "description": "FeNiCrCoCu high-entropy alloy",
        "challenge": "chemical disorder",
        "n_atoms_typical": 108,
    },
    "Zr_oxide_amorphous": {
        "type": "amorphous_oxide",
        "description": "Zr-oxide amorphous structure",
        "challenge": "structural disorder",
        "n_atoms_typical": 96,
    },
}

# ---------------------------------------------------------------------------
# Acquisition strategies (8 types, A-H)
# ---------------------------------------------------------------------------
ACQUISITION_STRATEGIES: Dict[str, dict] = {
    "A_random": {
        "label": "Random (Baseline)",
        "category": "baseline",
        "description": "Uniform random sampling from pool",
    },
    "B_gmm_uncertainty": {
        "label": "GMM Uncertainty",
        "category": "uncertainty",
        "description": "MACE built-in: Mahalanobis distance in latent space via GMM per atom type",
    },
    "C_ensemble_qbc": {
        "label": "Ensemble QBC",
        "category": "uncertainty",
        "description": "Query-by-committee: 3 MACE models with different seeds, max variance",
    },
    "D_mc_dropout": {
        "label": "MC-Dropout",
        "category": "uncertainty",
        "description": "Monte Carlo dropout: multiple forward passes, variance across passes",
    },
    "E_fps_soap": {
        "label": "FPS + SOAP",
        "category": "diversity",
        "description": "Farthest point sampling in SOAP descriptor space",
    },
    "F_latent_clustering": {
        "label": "Latent Space Clustering",
        "category": "diversity",
        "description": "k-means clustering in MACE feature space, sample from clusters",
    },
    "G_hybrid_weighted": {
        "label": "Hybrid-Weighted (Proposed)",
        "category": "hybrid",
        "description": "Score = alpha * U_norm + (1-alpha) * D_norm",
    },
    "H_hybrid_twostage": {
        "label": "Hybrid-TwoStage (Proposed)",
        "category": "hybrid",
        "description": "Top 30% by uncertainty -> FPS diversity filter to K",
    },
}


@dataclass
class Config:
    """Master configuration for the experiment."""

    # ---- Active learning protocol ----
    n_init: int = 100           # Initial labeled pool size
    n_query: int = 20           # Structures selected per AL iteration
    n_iterations: int = 20      # Number of AL iterations
    # Final labeled set = n_init + n_query * n_iterations = 500

    # ---- Model ----
    use_mace: bool = False       # Use MACE (GPU recommended); False = fallback SchNet
    mace_model: str = "small"   # "small", "medium", "large", or path to checkpoint
    pretrained: str = "MACE-MP-0"  # Pretrained model to fine-tune from (ignored if use_mace=False)
    mace_dtype: str = "float32"
    r_max: float = 5.0          # Cutoff radius (Å)

    # ---- Ensemble ----
    ensemble_size: int = 3      # For QBC (strategy C)
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # ---- MC-Dropout ----
    mc_passes: int = 10         # Forward passes for MC-Dropout uncertainty
    mc_dropout_rate: float = 0.1

    # ---- GMM uncertainty ----
    gmm_n_components: int = 5

    # ---- SOAP descriptor ----
    soap_rcut: float = 5.0
    soap_nmax: int = 6
    soap_lmax: int = 4

    # ---- Latent clustering ----
    n_clusters: int = 20        # k for k-means (matches n_query)

    # ---- Hybrid strategies ----
    hybrid_alphas: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    two_stage_topk: float = 0.3  # Fraction kept after first-stage filter

    # ---- Training ----
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 30           # Early stopping patience
    ema_decay: float = 0.99      # Exponential moving average for loss tracking
    force_weight: float = 10.0   # Weight of force loss relative to energy loss

    # ---- MD stability test ----
    md_temperature: float = 300.0   # K
    md_timestep: float = 0.5       # fs
    md_steps: int = 400            # 200 ps at 0.5 fs
    md_crash_threshold: float = 1.0  # eV/Å force above which MD is "crashed"

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Compute ----
    device: str = "cuda"         # "cuda" or "cpu"
    n_workers: int = 4

    # ---- Output ----
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "./data"

    # ---- Active systems to run (subset for faster testing) ----
    active_systems: List[str] = field(default_factory=lambda: [
        "MgO_surface",
        "liquid_water",
        "zeolite",
        "Pt_CH_activation",
        "FeNiCrCoCu_HEA",
        "Zr_oxide_amorphous",
    ])

    # ---- Active strategies to run ----
    active_strategies: List[str] = field(default_factory=lambda: [
        "A_random",
        "B_gmm_uncertainty",
        "C_ensemble_qbc",
        "D_mc_dropout",
        "E_fps_soap",
        "F_latent_clustering",
        "G_hybrid_weighted",
        "H_hybrid_twostage",
    ])
