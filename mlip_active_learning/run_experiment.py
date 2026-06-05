#!/usr/bin/env python
"""Main experiment script for MLIP active learning.

Runs the core active learning loop comparing 8 acquisition strategies
across 6 material systems (MS25 benchmark).

Usage:
    python run_experiment.py                          # Full experiment
    python run_experiment.py --systems MgO_surface    # Single system
    python run_experiment.py --strategies A_random,G_hybrid_weighted  # Subset
    python run_experiment.py --test                   # Quick test mode
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings

from config import Config, MATERIAL_SYSTEMS, ACQUISITION_STRATEGIES
from data import (
    MaterialDataset, create_splits, make_dataloader,
    load_or_generate_data, generate_synthetic_structures,
)
from model import MACEWrapper, EnsembleMACE
from acquisition import create_acquisition_function
from metrics import (
    evaluate_model, DataEfficiencyTracker,
    compute_convergence_labels, compute_data_efficiency_ratio,
)
from train import train_single_model, train_ensemble


def parse_args():
    p = argparse.ArgumentParser(description="MLIP Active Learning Experiment")
    p.add_argument("--systems", type=str, default=None,
                   help="Comma-separated system names (default: all 6)")
    p.add_argument("--strategies", type=str, default=None,
                   help="Comma-separated strategy IDs (default: all 8)")
    p.add_argument("--test", action="store_true",
                   help="Quick test mode: fewer iterations, synthetic data")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device: cpu or cuda")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Output directory")
    return p.parse_args()


def run_active_learning_loop(
    config: Config,
    dataset: MaterialDataset,
    system_name: str,
    strategy_id: str,
    init_indices: np.ndarray,
    pool_indices: np.ndarray,
    test_indices: np.ndarray,
    val_indices: np.ndarray,
) -> dict:
    """Run one active learning experiment for a given strategy on a system.

    Returns:
        dict with learning_curve, final_mae, training_times, etc.
    """
    strategy_info = ACQUISITION_STRATEGIES[strategy_id]
    print(f"\n{'='*60}")
    print(f"  System: {system_name} | Strategy: {strategy_info['label']}")
    print(f"{'='*60}")

    n_query = config.n_query
    n_iterations = config.n_iterations

    labeled = list(init_indices)
    pool = list(pool_indices)

    # Data structures for tracking
    learning_curve = []  # [(n_labeled, test_mae, test_rmse)]
    timing = []

    # Initial training set
    init_dataloader = make_dataloader(dataset, labeled, config.batch_size,
                                      shuffle=True, n_workers=config.n_workers)
    val_dataloader = make_dataloader(dataset, val_indices, config.batch_size,
                                     shuffle=False, n_workers=config.n_workers)
    test_dataloader = make_dataloader(dataset, test_indices, config.batch_size,
                                      shuffle=False, n_workers=config.n_workers)

    # Create acquisition function
    acq_fn = create_acquisition_function(strategy_id, alpha=0.5, topk_frac=0.3)

    # Train initial ensemble
    print(f"\n  [Initial training] {len(labeled)} structures")
    import time as _time
    t0 = _time.time()

    try:
        ensemble = train_ensemble(config, init_dataloader, val_dataloader)
    except Exception as e:
        print(f"  [WARNING] Training failed: {e}")
        print(f"  Using fallback evaluation")
        ensemble = None

    train_time = _time.time() - t0
    timing.append(train_time)

    # Evaluate initial model
    if ensemble is not None:
        eval_results = evaluate_model(ensemble, test_dataloader, config.device)
        test_mae = eval_results["energy_mae"]
    else:
        test_mae = float("inf")

    learning_curve.append({"iteration": 0, "n_labeled": len(labeled),
                           "test_mae": test_mae})
    print(f"  Iter 0 | N={len(labeled)} | Test MAE={test_mae:.6f} | "
          f"Train time={train_time:.1f}s")

    # Active learning iterations
    for iteration in range(1, n_iterations + 1):
        print(f"\n  --- Iteration {iteration}/{n_iterations} ---")

        if len(pool) < n_query:
            print(f"  Pool exhausted ({len(pool)} < {n_query}). Stopping.")
            break

        # Prepare pool dataloader for scoring
        pool_dataloader = make_dataloader(dataset, pool, config.batch_size,
                                          shuffle=False, n_workers=config.n_workers)

        # Prepare pool structures (for SOAP-based methods)
        pool_structures = [dataset.get_atoms(i) for i in pool]

        # Select next batch
        t0 = _time.time()
        selected_local = acq_fn.select(
            pool_indices=np.array(pool),
            n_query=n_query,
            model=ensemble,
            pool_dataloader=pool_dataloader,
            pool_structures=pool_structures,
            labeled_structures=[dataset.get_atoms(i) for i in labeled],
            labeled_dataloader=make_dataloader(dataset, labeled, config.batch_size,
                                               shuffle=False, n_workers=config.n_workers),
            device=config.device,
        )
        select_time = _time.time() - t0

        # Move selected from pool to labeled
        selected = sorted(set(int(s) for s in selected_local))
        for s in selected:
            if s in pool:
                pool.remove(s)
                labeled.append(s)

        print(f"  Selected {len(selected)} structures in {select_time:.1f}s")
        print(f"  Labeled: {len(labeled)}, Pool: {len(pool)}")

        # Retrain
        train_dataloader = make_dataloader(dataset, labeled, config.batch_size,
                                           shuffle=True, n_workers=config.n_workers)
        t0 = _time.time()

        try:
            if isinstance(ensemble, EnsembleMACE) and ensemble is not None:
                ensemble = train_ensemble(config, train_dataloader, val_dataloader)
            else:
                # Single model retraining
                model = MACEWrapper(
                    model_name=config.mace_model,
                    pretrained=config.pretrained,
                    r_max=config.r_max,
                    device=config.device,
                    use_mace=config.use_mace,
                )
                train_single_model(model, train_dataloader, val_dataloader, config)
        except Exception as e:
            print(f"  [WARNING] Retraining failed: {e}")

        train_time = _time.time() - t0
        timing.append(train_time)

        # Evaluate
        if ensemble is not None:
            eval_results = evaluate_model(ensemble, test_dataloader, config.device)
            test_mae = eval_results["energy_mae"]
        else:
            test_mae = float("inf")

        learning_curve.append({"iteration": iteration, "n_labeled": len(labeled),
                               "test_mae": test_mae})
        print(f"  Iter {iteration} | N={len(labeled)} | Test MAE={test_mae:.6f} | "
              f"Train time={train_time:.1f}s")

    return {
        "strategy": strategy_id,
        "system": system_name,
        "learning_curve": learning_curve,
        "final_mae": learning_curve[-1]["test_mae"],
        "total_train_time": sum(timing),
        "n_final_labeled": len(labeled),
    }


def run_full_experiment(config: Config, args) -> dict:
    """Run the complete experiment across all systems and strategies."""

    systems = config.active_systems
    strategies = config.active_strategies

    if args.systems:
        systems = [s.strip() for s in args.systems.split(",")]
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]

    if args.test:
        systems = [systems[0]]
        strategies = ["A_random", "G_hybrid_weighted"]  # baseline vs proposed
        config.n_iterations = 3
        config.n_init = 50
        config.n_query = 10
        config.ensemble_size = 2
        config.max_epochs = 80
        config.patience = 15
        config.learning_rate = 1e-3
        config.hidden_channels = 64
        config.pool_size = 600
        config.use_mace = False
        config._test_mode = True
        print("=" * 60)
        print("  QUICK TEST MODE")
        print(f"  Systems: {len(systems)} | Strategies: {len(strategies)} | "
              f"Iterations: {config.n_iterations} | Cu LJ clusters")
        print("=" * 60)

    print(f"\nMaterials systems ({len(systems)}):")
    for s in systems:
        info = MATERIAL_SYSTEMS[s]
        print(f"  {s} ({info['type']}) — {info['challenge']}")

    print(f"\nAcquisition strategies ({len(strategies)}):")
    for s in strategies:
        info = ACQUISITION_STRATEGIES[s]
        print(f"  {s} [{info['category']}] — {info['label']}")

    all_results = {}

    for system_name in systems:
        print(f"\n{'#'*60}")
        print(f"#  SYSTEM: {system_name}")
        print(f"{'#'*60}")

        # Load data
        print(f"\n  Loading data...")
        dataset = load_or_generate_data(system_name, config)
        print(f"  Total structures: {len(dataset)}")

        # Create shared splits
        n_total = len(dataset)
        init_idx, pool_idx, test_idx, val_idx = create_splits(
            n_total, config.n_init,
            test_ratio=0.15, val_ratio=0.10, seed=config.seed,
        )
        print(f"  Splits: init={len(init_idx)}, pool={len(pool_idx)}, "
              f"test={len(test_idx)}, val={len(val_idx)}")

        system_results = {}

        for strategy_id in strategies:
            result = run_active_learning_loop(
                config, dataset, system_name, strategy_id,
                init_idx.copy(), pool_idx.copy(),
                test_idx.copy(), val_idx.copy(),
            )
            system_results[strategy_id] = result

            # Save intermediate results
            save_intermediate_results(config, system_name, system_results)

        all_results[system_name] = system_results

    return all_results


def save_intermediate_results(config, system_name, system_results):
    """Save results after each system completes (safety checkpoint)."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save learning curves CSV
    curves_data = {}
    for strategy_id, result in system_results.items():
        curve = result["learning_curve"]
        curves_data[strategy_id] = [c["test_mae"] for c in curve]

    df = pd.DataFrame(curves_data)
    df.index.name = "iteration"
    df.to_csv(results_dir / f"{system_name}_learning_curves.csv")

    # Save summary JSON
    summary = {}
    for strategy_id, result in system_results.items():
        summary[strategy_id] = {
            "strategy": result["strategy"],
            "final_mae": result["final_mae"],
            "total_train_time": result["total_train_time"],
            "n_final_labeled": result["n_final_labeled"],
        }

    with open(results_dir / f"{system_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def generate_final_report(all_results: dict, config: Config):
    """Generate final experiment report."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n\n{'#'*60}")
    print(f"#  FINAL RESULTS")
    print(f"{'#'*60}")

    # Per-system summary
    for system_name, system_results in all_results.items():
        print(f"\n--- {system_name} ---")
        print(f"{'Strategy':<30} {'Final MAE':>12} {'Train Time':>12}")
        print("-" * 56)

        for strategy_id, result in system_results.items():
            label = ACQUISITION_STRATEGIES[strategy_id]["label"]
            print(f"{label:<30} {result['final_mae']:>12.6f} "
                  f"{result['total_train_time']:>10.1f}s")

    # Compute data efficiency
    print(f"\n--- Data Efficiency (vs Random Baseline) ---")
    for system_name, system_results in all_results.items():
        curves = {}
        for strategy_id, result in system_results.items():
            curves[strategy_id] = [c["test_mae"] for c in result["learning_curve"]]

        # Find target MAE as 90% of best random final MAE
        random_final = curves.get("A_random", [1.0])[-1]
        target_mae = random_final * 0.9

        convergence = compute_convergence_labels(curves, target_mae, config.n_init, config.n_query)
        ratios = compute_data_efficiency_ratio(convergence, baselines=["A_random"])

        print(f"  {system_name} (target MAE={target_mae:.4f}):")
        for strategy_id, ratio in ratios.items():
            label = ACQUISITION_STRATEGIES[strategy_id]["label"]
            print(f"    {label}: {ratio:.2f}x more efficient than random")

    # Save combined results
    all_curves = {}
    for system_name, system_results in all_results.items():
        for strategy_id, result in system_results.items():
            key = f"{system_name}/{strategy_id}"
            all_curves[key] = [c["test_mae"] for c in result["learning_curve"]]

    df = pd.DataFrame(all_curves)
    df.index.name = "iteration"
    df.to_csv(results_dir / "all_learning_curves.csv")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_init": config.n_init,
            "n_query": config.n_query,
            "n_iterations": config.n_iterations,
            "model": config.mace_model,
            "pretrained": config.pretrained,
        },
        "systems": list(all_results.keys()),
        "strategies": list(list(all_results.values())[0].keys())
        if all_results else [],
    }
    with open(results_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")
    return all_results


def main():
    args = parse_args()
    config = Config()
    config.device = args.device
    config.results_dir = args.results_dir

    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 60)
    print("  MLIP Active Learning Experiment")
    print("  Hybrid Acquisition Functions for DFT Data Efficiency")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    all_results = run_full_experiment(config, args)
    generate_final_report(all_results, config)

    print(f"\n  Experiment complete: {datetime.now().isoformat()}")
    return all_results


if __name__ == "__main__":
    main()
