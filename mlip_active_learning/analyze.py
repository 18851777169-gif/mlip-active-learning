#!/usr/bin/env python
"""Analysis and visualization of MLIP active learning experiment results.

Produces:
  1. Learning curves (MAE vs N_labeled) — per system + aggregate
  2. Data efficiency bar chart (labels to reach target MAE)
  3. Strategy ranking table (Friedman + pairwise Wilcoxon)
  4. Category comparison (uncertainty vs diversity vs hybrid)
  5. Per-system radar chart of relative efficiency
"""

import sys
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from config import ACQUISITION_STRATEGIES, MATERIAL_SYSTEMS


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "baseline": "#888888",
    "uncertainty": "#E74C3C",
    "diversity": "#3498DB",
    "hybrid": "#27AE60",
}

STRATEGY_COLORS = {
    "A_random": "#888888",
    "B_gmm_uncertainty": "#E74C3C",
    "C_ensemble_qbc": "#C0392B",
    "D_mc_dropout": "#F1948A",
    "E_fps_soap": "#3498DB",
    "F_latent_clustering": "#2980B9",
    "G_hybrid_weighted": "#27AE60",
    "H_hybrid_twostage": "#1E8449",
}


def load_results(results_dir: str) -> dict:
    """Load experiment results from output directory."""
    results_dir = Path(results_dir)

    # Load combined curves
    curves_path = results_dir / "all_learning_curves.csv"
    if curves_path.exists():
        df_curves = pd.read_csv(curves_path, index_col=0)
    else:
        print(f"[ERROR] Results file not found: {curves_path}")
        sys.exit(1)

    # Load metadata
    meta_path = results_dir / "experiment_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    # Reconstruct results dict
    all_results = {}
    for system_name in metadata["systems"]:
        all_results[system_name] = {}

        # Load per-system summary
        summary_path = results_dir / f"{system_name}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        for strategy_id in metadata["strategies"]:
            key = f"{system_name}/{strategy_id}"
            if key in df_curves.columns:
                curve = df_curves[key].dropna().values
                final_mae = curve[-1] if len(curve) > 0 else np.nan

                all_results[system_name][strategy_id] = {
                    "learning_curve": curve,
                    "final_mae": final_mae,
                }

                if system_name in all_results and summary and strategy_id in summary:
                    all_results[system_name][strategy_id].update(summary[strategy_id])

    return all_results, metadata


# ---------------------------------------------------------------------------
# Plot 1: Learning curves per system
# ---------------------------------------------------------------------------

def plot_learning_curves(all_results: dict, config_dict: dict, output_dir: str):
    """Plot MAE vs N_labeled for each material system."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_init = config_dict.get("n_init", 100)
    n_query = config_dict.get("n_query", 20)

    systems = list(all_results.keys())

    for system_name in systems:
        fig, ax = plt.subplots(figsize=(10, 6))

        system_results = all_results[system_name]

        for strategy_id in sorted(system_results.keys()):
            curve = system_results[strategy_id].get("learning_curve", [])
            if not isinstance(curve, np.ndarray):
                curve = np.array(curve)

            if len(curve) == 0 or np.all(np.isnan(curve)):
                continue

            n_labeled = [n_init + i * n_query for i in range(len(curve))]

            label = ACQUISITION_STRATEGIES[strategy_id]["label"]
            color = STRATEGY_COLORS.get(strategy_id, "#333333")
            lw = 2.5 if "hybrid" in ACQUISITION_STRATEGIES[strategy_id]["category"] else 1.5

            ax.plot(n_labeled, curve, "-o", label=label, color=color,
                    linewidth=lw, markersize=4, markevery=2)

        ax.set_xlabel("Number of Labeled Structures", fontsize=12)
        ax.set_ylabel("Energy MAE (eV/atom)", fontsize=12)
        ax.set_title(f"Learning Curves — {system_name}", fontsize=14)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"learning_curve_{system_name}.png", dpi=150)
        plt.close(fig)

    print(f"  Saved {len(systems)} learning curve plots to {output_dir}/")


# ---------------------------------------------------------------------------
# Plot 2: Aggregate comparison (average over systems)
# ---------------------------------------------------------------------------

def plot_aggregate_curves(all_results: dict, config_dict: dict, output_dir: str):
    """Average learning curves across all systems."""
    output_dir = Path(output_dir)

    n_init = config_dict.get("n_init", 100)
    n_query = config_dict.get("n_query", 20)

    # Collect curves per strategy
    strategy_curves: Dict[str, List[np.ndarray]] = {}

    for system_name, system_results in all_results.items():
        for strategy_id, result in system_results.items():
            curve = result.get("learning_curve", [])
            if not isinstance(curve, np.ndarray):
                curve = np.array(curve)
            if len(curve) > 0 and not np.all(np.isnan(curve)):
                strategy_curves.setdefault(strategy_id, []).append(curve)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: All curves averaged
    for strategy_id in sorted(strategy_curves.keys()):
        curves = strategy_curves[strategy_id]
        max_len = max(len(c) for c in curves)
        padded = []
        for c in curves:
            pad = np.full(max_len - len(c), np.nan)
            padded.append(np.concatenate([c, pad]))
        stacked = np.array(padded)
        mean_curve = np.nanmean(stacked, axis=0)
        std_curve = np.nanstd(stacked, axis=0)

        n_labeled = [n_init + i * n_query for i in range(len(mean_curve))]

        label = ACQUISITION_STRATEGIES[strategy_id]["label"]
        color = STRATEGY_COLORS.get(strategy_id, "#333333")
        lw = 2.5 if "hybrid" in ACQUISITION_STRATEGIES[strategy_id]["category"] else 1.5

        ax1.plot(n_labeled, mean_curve, "-o", label=label, color=color,
                 linewidth=lw, markersize=4, markevery=2)
        ax1.fill_between(n_labeled, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.1, color=color)

    ax1.set_xlabel("Number of Labeled Structures")
    ax1.set_ylabel("Energy MAE (eV/atom)")
    ax1.set_title("Average Learning Curves (All Systems)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right: Bar chart of final MAE by strategy
    final_maes = {}
    for strategy_id, curves in strategy_curves.items():
        finals = [c[-1] for c in curves if not np.isnan(c[-1])]
        if finals:
            final_maes[strategy_id] = (np.mean(finals), np.std(finals))

    strategies_ordered = sorted(final_maes.keys(), key=lambda s: final_maes[s][0])
    labels = [ACQUISITION_STRATEGIES[s]["label"] for s in strategies_ordered]
    means = [final_maes[s][0] for s in strategies_ordered]
    stds = [final_maes[s][1] for s in strategies_ordered]
    colors = [STRATEGY_COLORS.get(s, "#333") for s in strategies_ordered]

    bars = ax2.barh(labels, means, xerr=stds, color=colors, capsize=3)
    ax2.set_xlabel("Final Energy MAE (eV/atom)")
    ax2.set_title("Final Test MAE by Strategy")
    ax2.axvline(x=means[0], color="gray", linestyle="--", alpha=0.5,
                label=f"Best: {means[0]:.4f}")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved aggregate comparison to {output_dir}/")


# ---------------------------------------------------------------------------
# Plot 3: Data efficiency ratio
# ---------------------------------------------------------------------------

def plot_data_efficiency(all_results: dict, config_dict: dict, output_dir: str):
    """Bar chart: data efficiency ratio vs random baseline."""
    output_dir = Path(output_dir)
    n_init = config_dict.get("n_init", 100)
    n_query = config_dict.get("n_query", 20)

    # Find target MAE per system (best random final MAE * 0.9)
    from metrics import compute_convergence_labels, compute_data_efficiency_ratio

    efficiency_data = {}
    for system_name, system_results in all_results.items():
        curves = {}
        for strategy_id, result in system_results.items():
            curve = result.get("learning_curve", [])
            if not isinstance(curve, np.ndarray):
                curve = np.array(curve)
            curves[strategy_id] = list(curve)

        random_final = curves.get("A_random", [1.0])[-1]
        target_mae = random_final * 0.9

        convergence = compute_convergence_labels(curves, target_mae, n_init, n_query)
        ratios = compute_data_efficiency_ratio(convergence, baselines=["A_random"])

        for strategy_id, ratio in ratios.items():
            efficiency_data.setdefault(strategy_id, []).append(ratio)

    if not efficiency_data:
        print("  No efficiency data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    strategies_ordered = sorted(efficiency_data.keys(),
                                key=lambda s: np.mean(efficiency_data[s]),
                                reverse=True)
    labels = [ACQUISITION_STRATEGIES[s]["label"] for s in strategies_ordered]
    means = [np.mean(efficiency_data[s]) for s in strategies_ordered]
    stds = [np.std(efficiency_data[s]) for s in strategies_ordered]
    colors = [STRATEGY_COLORS.get(s, "#333") for s in strategies_ordered]

    bars = ax.barh(labels, means, xerr=stds, color=colors, capsize=3)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7, label="Random baseline (=1.0)")
    ax.set_xlabel("Data Efficiency Ratio (higher = fewer DFTs needed)")
    ax.set_title("Data Efficiency Relative to Random Sampling")

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{mean:.2f}x", va="center", fontsize=9)

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(output_dir / "data_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"  Saved data efficiency plot to {output_dir}/")


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def run_statistical_tests(all_results: dict):
    """Friedman test + pairwise Wilcoxon with Holm correction."""
    from scipy.stats import friedmanchisquare, wilcoxon
    from scipy.stats import combine_pvalues

    systems = list(all_results.keys())
    strategies = list(all_results[systems[0]].keys()) if systems else []

    if len(systems) < 2 or len(strategies) < 2:
        print("  Insufficient data for statistical testing")
        return

    print("\n--- Statistical Analysis ---")

    # Build matrix: systems × strategies
    final_maes = {}
    for system_name in systems:
        for strategy_id in strategies:
            result = all_results[system_name].get(strategy_id, {})
            mae = result.get("final_mae", np.nan)
            final_maes.setdefault(strategy_id, []).append(mae)

    # Friedman test
    samples = [np.array(final_maes[s]) for s in strategies if s in final_maes]
    if len(samples) >= 3 and all(len(s) == len(samples[0]) for s in samples):
        try:
            stat, p = friedmanchisquare(*samples)
            print(f"  Friedman test: chi2={stat:.3f}, p={p:.4f}")
            if p < 0.05:
                print(f"  -> Significant difference across strategies (p < 0.05)")
        except Exception as e:
            print(f"  Friedman test failed: {e}")

    # Pairwise Wilcoxon vs best baseline (A_random)
    if "A_random" in final_maes and "G_hybrid_weighted" in final_maes:
        try:
            stat, p = wilcoxon(final_maes["G_hybrid_weighted"],
                               final_maes["A_random"])
            print(f"  Hybrid-Weighted vs Random: W={stat:.1f}, p={p:.4f}")
        except Exception as e:
            print(f"  Wilcoxon test failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Analyze MLIP AL experiment results")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Path to experiment results directory")
    p.add_argument("--output-dir", type=str, default="results/figures",
                   help="Output directory for figures")
    args = p.parse_args()

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    print("Loading results...")
    all_results, metadata = load_results(args.results_dir)
    config_dict = metadata.get("config", {})

    print(f"  Systems: {list(all_results.keys())}")
    print(f"  Strategies: {list(list(all_results.values())[0].keys())}")

    # Generate plots
    plot_learning_curves(all_results, config_dict, args.output_dir)
    plot_aggregate_curves(all_results, config_dict, args.output_dir)
    plot_data_efficiency(all_results, config_dict, args.output_dir)

    # Statistical tests
    run_statistical_tests(all_results)

    print(f"\n  All figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
