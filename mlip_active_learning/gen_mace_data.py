#!/usr/bin/env python3
"""Generate training data using MACE as ground-truth potential.

Creates random Cu clusters and labels them with MACE-MP-0 energies/forces.
This produces a complex, realistic potential energy surface for testing
active learning strategies.
"""

import sys, os, time, pickle, argparse
import numpy as np
import torch
from ase import Atoms
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-structures", type=int, default=1500)
    p.add_argument("--min-atoms", type=int, default=15)
    p.add_argument("--max-atoms", type=int, default=45)
    p.add_argument("--output", type=str, default="data/mace_labeled.pkl")
    p.add_argument("--model-path", type=str,
                   default="/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def generate_structure(rng, min_atoms, max_atoms, elements=None):
    """Generate a random periodic cluster with minimum distance constraint."""
    if elements is None:
        elements = ["Cu", "Ni", "Fe", "Al", "Si", "Mg", "O", "Pt", "Zr", "Ti"]

    n_atoms = rng.randint(min_atoms, max_atoms + 1)
    n_types = rng.randint(1, min(4, len(elements)))
    chosen = list(rng.choice(elements, size=n_types, replace=False))
    symbols = [chosen[rng.randint(n_types)] for _ in range(n_atoms)]

    min_dist = 1.8
    box_size = (n_atoms * 30.0) ** (1/3)
    positions = np.zeros((n_atoms, 3))
    positions[0] = rng.uniform(0, box_size, 3)
    for j in range(1, n_atoms):
        for _ in range(300):
            candidate = rng.uniform(0, box_size, 3)
            delta = positions[:j] - candidate
            delta = delta - box_size * np.round(delta / box_size)
            if np.sqrt((delta ** 2).sum(axis=1)).min() > min_dist:
                positions[j] = candidate
                break
        else:
            positions[j] = rng.uniform(0, box_size, 3)

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_cell([box_size] * 3)
    atoms.set_pbc(True)
    return atoms

def main():
    args = parse_args()
    print(f"Generating {args.n_structures} structures ({args.min_atoms}-{args.max_atoms} atoms)...")

    from mace.calculators import MACECalculator
    print(f"Loading MACE model from {args.model_path}...")
    calc = MACECalculator(model_path=args.model_path, device=args.device, default_dtype="float32")
    print(f"MACE loaded: {calc.num_models} model(s) on {args.device}")

    rng = np.random.RandomState(42)
    structures = []
    n_failed = 0

    t0 = time.time()
    for i in range(args.n_structures):
        atoms = generate_structure(rng, args.min_atoms, args.max_atoms)
        atoms.calc = calc
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            atoms.info["energy"] = energy
            atoms.info["forces"] = forces
            structures.append(atoms)
        except Exception as e:
            n_failed += 1
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.n_structures - i - 1) / rate
            print(f"  {i+1}/{args.n_structures} ({rate:.1f} structs/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"Done: {len(structures)} structures in {elapsed:.0f}s ({len(structures)/elapsed:.1f}/s)")
    if n_failed:
        print(f"Failed: {n_failed}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(structures, f)
    print(f"Saved to {args.output}")

    # Stats
    energies = [s.info["energy"] for s in structures]
    sizes = [len(s) for s in structures]
    print(f"Energy: [{np.min(energies):.2f}, {np.max(energies):.2f}] eV, "
          f"mean={np.mean(energies):.2f} +/- {np.std(energies):.2f}")
    print(f"Size: [{np.min(sizes)}, {np.max(sizes)}]")

if __name__ == "__main__":
    main()
