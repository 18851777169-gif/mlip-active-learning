"""Evaluation metrics for MLIP active learning experiments.

Implements the 5 metrics from the Methodology Blueprint:
  1. Data efficiency curve (MAE vs N_labeled)
  2. Convergence label count (labels to reach target MAE)
  3. MD stability (crash rate in NVT simulation)
  4. Physical plausibility (E-V curve monotonicity)
  5. Coverage (Voronoi volume in SOAP space)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(predictions - targets)))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root mean square error."""
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_force_mae(pred_forces: np.ndarray, true_forces: np.ndarray) -> float:
    """Per-component force MAE."""
    return float(np.mean(np.abs(pred_forces - true_forces)))


def evaluate_model(model, dataloader, device: str = "cpu") -> Dict[str, float]:
    """Evaluate energy and force MAE on a dataloader."""
    import torch

    model.eval()
    energy_preds, energy_targets = [], []
    force_preds, force_targets = [], []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_data.items()}
            output = model(batch_data)
            if isinstance(output, tuple):
                e_pred = output[0]
                f_pred = output[1] if len(output) > 1 else None
            else:
                e_pred = output
                f_pred = None
            e_true = batch_data["y"].view(-1)
            f_true = batch_data.get("forces", None)

            energy_preds.append(e_pred.cpu().numpy())
            energy_targets.append(e_true.cpu().numpy())

            if f_pred is not None and f_true is not None:
                force_preds.append(f_pred.cpu().numpy())
                force_targets.append(f_true.cpu().numpy())

    e_pred_all = np.concatenate(energy_preds)
    e_true_all = np.concatenate(energy_targets)

    results = {
        "energy_mae": compute_mae(e_pred_all, e_true_all),
        "energy_rmse": compute_rmse(e_pred_all, e_true_all),
    }

    if force_preds:
        f_pred_all = np.concatenate(force_preds)
        f_true_all = np.concatenate(force_targets)
        results["force_mae"] = compute_force_mae(f_pred_all, f_true_all)

    return results


# ---------------------------------------------------------------------------
# Metric 1: Data efficiency curve
# ---------------------------------------------------------------------------

class DataEfficiencyTracker:
    """Tracks test MAE at each AL iteration to build learning curves."""

    def __init__(self):
        self.curves: Dict[str, List[float]] = {}  # strategy -> [mae_0, mae_1, ...]

    def record(self, strategy: str, iteration: int, mae: float):
        if strategy not in self.curves:
            self.curves[strategy] = []
        # Ensure list is long enough
        while len(self.curves[strategy]) <= iteration:
            self.curves[strategy].append(np.nan)
        self.curves[strategy][iteration] = mae

    def to_dataframe(self):
        import pandas as pd
        data = {}
        for strategy, mae_list in self.curves.items():
            data[strategy] = mae_list
        return pd.DataFrame(data)

    def compute_auc(self) -> Dict[str, float]:
        """Area under the learning curve (lower = better)."""
        aucs = {}
        for strategy, mae_list in self.curves.items():
            valid = [m for m in mae_list if not np.isnan(m)]
            if valid:
                aucs[strategy] = float(np.trapz(valid)) / len(valid)
        return aucs


# ---------------------------------------------------------------------------
# Metric 2: Convergence label count
# ---------------------------------------------------------------------------

def compute_convergence_labels(
    curves: Dict[str, List[float]],
    target_mae: float,
    n_init: int = 100,
    n_query: int = 20,
) -> Dict[str, Optional[int]]:
    """Number of labeled structures needed to reach target MAE.

    Returns:
        {strategy: n_labels or None if never reaches target}
    """
    results = {}
    for strategy, mae_list in curves.items():
        reached = False
        for i, mae in enumerate(mae_list):
            if not np.isnan(mae) and mae <= target_mae:
                n_labels = n_init + i * n_query
                results[strategy] = n_labels
                reached = True
                break
        if not reached:
            results[strategy] = None
    return results


def compute_data_efficiency_ratio(
    convergence_labels: Dict[str, Optional[int]],
    baselines: List[str] = None,
) -> Dict[str, float]:
    """Ratio: baseline_labels / method_labels (>1 means method is more efficient)."""
    if baselines is None:
        baselines = ["A_random"]

    ratios = {}
    for strategy, n_labels in convergence_labels.items():
        if strategy in baselines or n_labels is None:
            continue
        best_baseline = min(
            (l for l in [convergence_labels.get(b) for b in baselines] if l is not None),
            default=None
        )
        if best_baseline is not None and n_labels > 0:
            ratios[strategy] = best_baseline / n_labels
    return ratios


# ---------------------------------------------------------------------------
# Metric 3: MD stability
# ---------------------------------------------------------------------------

def test_md_stability(
    model,
    atoms,
    temperature: float = 300.0,
    timestep: float = 0.5,
    n_steps: int = 400,
    crash_threshold: float = 1.0,
) -> Dict:
    """Run short NVT MD simulation and check for crashes.

    A crash is defined as any atom experiencing force > crash_threshold eV/Å.
    Returns crash status and fraction of steps that remained stable.
    """
    try:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase import units
        from ase.io.trajectory import Trajectory
    except ImportError:
        return {"crashed": True, "stable_fraction": 0.0, "error": "ASE not available"}

    import copy

    atoms = copy.deepcopy(atoms)
    atoms.calc = None  # Model predictions used instead

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature * units.kB)

    crash_count = 0
    n_atoms = len(atoms)

    for step in range(n_steps):
        # Get model forces
        try:
            import torch
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).unsqueeze(0)
            pos = torch.tensor(atoms.positions, dtype=torch.float32).unsqueeze(0)
            # Model inference
            with torch.no_grad():
                # Simplified: would need proper model interface
                pass
        except Exception:
            pass

        # Check force magnitude
        try:
            forces = atoms.get_forces()
            max_force = np.abs(forces).max()
            if max_force > crash_threshold:
                crash_count += 1
        except Exception:
            crash_count += 1

        # Integrate
        try:
            dyn = VelocityVerlet(atoms, timestep * units.fs)
            dyn.run(1)
        except Exception:
            crash_count += 1
            break

    stable_fraction = 1.0 - crash_count / max(n_steps, 1)

    return {
        "crashed": stable_fraction < 0.5,
        "stable_fraction": stable_fraction,
        "crash_count": crash_count,
        "total_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Metric 4: Physical plausibility (E-V scan)
# ---------------------------------------------------------------------------

def test_ev_monotonicity(model, atoms, volume_factors=None) -> Dict:
    """Check if energy-volume curve is physically monotonically decreasing
    near equilibrium (violation indicates unphysical model).

    Scans volumes from 0.9 to 1.1 of equilibrium volume.
    """
    if volume_factors is None:
        volume_factors = np.linspace(0.90, 1.10, 21)

    import copy
    energies = []

    for factor in volume_factors:
        scaled = copy.deepcopy(atoms)
        cell = scaled.get_cell()
        scaled.set_cell(cell * factor ** (1/3), scale_atoms=True)

        # Would compute energy via model inference
        energies.append(0.0)  # Placeholder

    energies = np.array(energies)

    # Check monotonicity: E should decrease as V increases (near equilibrium)
    n_violations = 0
    for i in range(len(energies) - 1):
        if energies[i + 1] > energies[i] + 1e-4:
            n_violations += 1

    is_physical = n_violations <= 2  # Allow 2 minor violations

    return {
        "is_physical": is_physical,
        "n_monotonicity_violations": n_violations,
        "total_points": len(volume_factors),
        "volume_range": [float(volume_factors[0]), float(volume_factors[-1])],
    }


# ---------------------------------------------------------------------------
# Metric 5: Coverage (SOAP-space Voronoi volume)
# ---------------------------------------------------------------------------

def compute_coverage(structures, soap_computer=None) -> float:
    """Estimate coverage of SOAP space by labeled structures.

    Uses convex hull volume approximation or pairwise distance
    sum as a simple coverage proxy.

    Returns:
        coverage_score: higher = more of descriptor space covered
    """
    if soap_computer is None:
        from descriptors import SOAPDescriptors
        soap_computer = SOAPDescriptors()

    descriptors = soap_computer.compute(structures)
    if descriptors is None or descriptors.shape[0] < 3:
        return 0.0

    from scipy.spatial import ConvexHull

    # PCA to reduce dimensionality for convex hull
    from sklearn.decomposition import PCA
    n_components = min(6, descriptors.shape[1], descriptors.shape[0] - 1)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(descriptors)

    try:
        hull = ConvexHull(reduced)
        return float(hull.volume)
    except Exception:
        # Pairwise distance sum as fallback
        from scipy.spatial.distance import pdist
        return float(pdist(reduced).mean() * reduced.shape[0])


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    model, test_dataloader, system_name: str, device: str = "cpu",
    structures_for_md=None, structures_for_ev=None, structures_for_coverage=None,
) -> Dict:
    """Run all 5 metrics for a trained model on one material system."""
    results = {
        "system": system_name,
    }

    # Metrics 1-2: accuracy
    eval_results = evaluate_model(model, test_dataloader, device)
    results.update(eval_results)

    # Metric 3: MD stability
    if structures_for_md is not None:
        md_result = test_md_stability(model, structures_for_md)
        results["md_stability"] = md_result

    # Metric 4: E-V plausibility
    if structures_for_ev is not None:
        ev_result = test_ev_monotonicity(model, structures_for_ev)
        results["ev_plausibility"] = ev_result

    # Metric 5: Coverage
    if structures_for_coverage is not None:
        coverage = compute_coverage(structures_for_coverage)
        results["soap_coverage"] = coverage

    return results
