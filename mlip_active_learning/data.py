"""Data loading for MS25, MP-ALOE, MatPES datasets.

Each material system provides ASE Atoms objects. Data loading follows
a unified interface so any dataset can be plugged in.

For the proof-of-concept / testing phase, we also include a synthetic
data generator for each of the 6 MS25 material system types.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle

from config import MATERIAL_SYSTEMS


# ---------------------------------------------------------------------------
# Lennard-Jones energy calculator for physically meaningful synthetic data
# ---------------------------------------------------------------------------

# LJ parameters for common elements (epsilon in eV, sigma in Å)
LJ_PARAMS = {
    "H": {"epsilon": 0.001, "sigma": 1.0},
    "C": {"epsilon": 0.005, "sigma": 1.7},
    "N": {"epsilon": 0.005, "sigma": 1.55},
    "O": {"epsilon": 0.006, "sigma": 1.52},
    "Mg": {"epsilon": 0.03, "sigma": 2.5},
    "Al": {"epsilon": 0.04, "sigma": 2.5},
    "Si": {"epsilon": 0.04, "sigma": 2.3},
    "P": {"epsilon": 0.03, "sigma": 2.1},
    "S": {"epsilon": 0.025, "sigma": 2.0},
    "Ca": {"epsilon": 0.02, "sigma": 2.8},
    "Ti": {"epsilon": 0.05, "sigma": 2.5},
    "Cr": {"epsilon": 0.05, "sigma": 2.3},
    "Fe": {"epsilon": 0.06, "sigma": 2.3},
    "Co": {"epsilon": 0.06, "sigma": 2.2},
    "Ni": {"epsilon": 0.06, "sigma": 2.2},
    "Cu": {"epsilon": 0.05, "sigma": 2.2},
    "Zn": {"epsilon": 0.03, "sigma": 2.3},
    "Zr": {"epsilon": 0.06, "sigma": 2.6},
    "Pt": {"epsilon": 0.07, "sigma": 2.5},
    "default": {"epsilon": 0.03, "sigma": 2.4},
}

def _compute_lj_energy_forces(atoms):
    """Compute Lennard-Jones energy and forces for ASE atoms.

    Uses geometric mixing rules: sigma_ij = sqrt(sigma_i * sigma_j)
                                epsilon_ij = sqrt(epsilon_i * epsilon_j)
    V_LJ(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    """
    import numpy as np

    positions = atoms.positions
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    energy = 0.0
    forces = np.zeros((n, 3))

    for i in range(n):
        si = symbols[i]
        eps_i = LJ_PARAMS.get(si, LJ_PARAMS["default"])["epsilon"]
        sig_i = LJ_PARAMS.get(si, LJ_PARAMS["default"])["sigma"]

        for j in range(i + 1, n):
            sj = symbols[j]
            eps_j = LJ_PARAMS.get(sj, LJ_PARAMS["default"])["epsilon"]
            sig_j = LJ_PARAMS.get(sj, LJ_PARAMS["default"])["sigma"]

            eps_ij = np.sqrt(eps_i * eps_j)
            sig_ij = np.sqrt(sig_i * sig_j)

            # Minimum image convention for periodic systems
            delta = positions[i] - positions[j]
            if pbc.any():
                delta = delta - cell.T @ np.round(np.linalg.solve(cell.T, delta))
            r = np.sqrt(np.sum(delta * delta))

            if r < 1e-10 or r > 10.0:
                continue

            # LJ energy
            sr = sig_ij / r
            sr6 = sr ** 6
            sr12 = sr6 * sr6
            e_pair = 4.0 * eps_ij * (sr12 - sr6)
            energy += e_pair

            # LJ force
            f_mag = 24.0 * eps_ij * (2.0 * sr12 - sr6) / r
            f_vec = f_mag * delta / r
            forces[i] += f_vec
            forces[j] -= f_vec

    atoms.info["energy"] = energy
    atoms.info["forces"] = forces
    return atoms


# ---------------------------------------------------------------------------
# Synthetic structure generators (for testing when real data unavailable)
# ---------------------------------------------------------------------------

def _random_bulk_structure(symbols, lattice_constant, size=(2, 2, 2)):
    """Generate a perturbed bulk crystal structure."""
    from ase import Atoms
    from ase.build import bulk

    atoms = bulk(symbols[0], crystalstructure="fcc", a=lattice_constant)
    if len(symbols) > 1:
        atoms = atoms.repeat(size)
        # Replace some atoms with other species
        rng = np.random.RandomState(42)
        for i in range(len(atoms)):
            if rng.random() < 0.2:
                atoms[i].symbol = symbols[rng.randint(len(symbols))]
    else:
        atoms = atoms.repeat(size)

    # Perturb positions
    atoms.positions += np.random.RandomState(42).normal(0, 0.05, atoms.positions.shape)
    return atoms


def generate_synthetic_structures(system_name: str, n_structures: int = 100, seed: int = 42):
    """Generate synthetic ASE structures for a material system type.

    This is a FALLBACK for when actual MS25/MP-ALOE data is not locally
    available. Real experiments should use actual DFT data.
    """
    import ase
    from ase import Atoms
    from ase.build import bulk, molecule, fcc111, surface
    import ase.io

    rng = np.random.RandomState(seed)

    system = MATERIAL_SYSTEMS[system_name]
    sys_type = system["type"]
    n_atoms = system["n_atoms_typical"]

    structures = []
    for i in range(n_structures):
        # Fixed atom count, single element → energy variation purely from geometry
        n_atoms = 30
        symbols = ["Cu"] * n_atoms

        # Random positions with minimum distance constraint
        min_dist = 2.0  # Minimum interatomic distance (Å)
        box_size = (n_atoms * 25.0) ** (1/3)  # ~9.1 Å for 30 atoms
        positions = np.zeros((n_atoms, 3))
        positions[0] = rng.uniform(0, box_size, 3)
        for j in range(1, n_atoms):
            for _ in range(200):
                candidate = rng.uniform(0, box_size, 3)
                delta = positions[:j] - candidate
                delta = delta - box_size * np.round(delta / box_size)
                dists = np.sqrt((delta ** 2).sum(axis=1))
                if dists.min() > min_dist:
                    positions[j] = candidate
                    break
            else:
                positions[j] = rng.uniform(0, box_size, 3)

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.set_cell([box_size] * 3)
        atoms.set_pbc(True)

        # Compute energy with Lennard-Jones for physical consistency
        atoms.info["system"] = system_name
        atoms = _compute_lj_energy_forces(atoms)

        structures.append(atoms)

    return structures


# ---------------------------------------------------------------------------
# Unified dataset class
# ---------------------------------------------------------------------------

class MaterialDataset(Dataset):
    """Dataset wrapping ASE structures for training.

    Each item returns:
        atomic_numbers: LongTensor [n_atoms]
        positions: FloatTensor [n_atoms, 3]
        energy: FloatTensor [1]
        forces: FloatTensor [n_atoms, 3]
        cell: FloatTensor [3, 3] (optional)
    """

    def __init__(self, structures: List, compute_forces: bool = True):
        self.structures = structures
        self.compute_forces = compute_forces

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        atoms = self.structures[idx]
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        energy = torch.tensor(atoms.info.get("energy", 0.0), dtype=torch.float32).view(1)
        forces = torch.tensor(atoms.info.get("forces", np.zeros((len(atoms), 3))),
                              dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array if hasattr(atoms, 'cell') else np.eye(3),
                            dtype=torch.float32)

        return {
            "z": z,
            "pos": pos,
            "y": energy,
            "forces": forces,
            "cell": cell,
            "natoms": torch.tensor(len(atoms), dtype=torch.long),
        }

    def get_atoms(self, idx):
        return self.structures[idx]


def collate_fn(batch):
    """Collate variable-size structures into a batch."""
    z_list, pos_list, y_list, f_list, cell_list, natoms_list = [], [], [], [], [], []
    batch_idx = []
    cumsum = 0

    for i, item in enumerate(batch):
        n = item["z"].shape[0]
        z_list.append(item["z"])
        pos_list.append(item["pos"])
        y_list.append(item["y"])
        f_list.append(item["forces"])
        cell_list.append(item["cell"])
        natoms_list.append(item["natoms"])
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        cumsum += n

    return {
        "z": torch.cat(z_list),
        "pos": torch.cat(pos_list),
        "y": torch.stack(y_list),
        "forces": torch.cat(f_list),
        "cell": torch.stack(cell_list),
        "natoms": torch.stack(natoms_list),
        "batch": torch.cat(batch_idx),
        "ptr": torch.tensor([0] + [item["z"].shape[0] for item in batch]).cumsum(0),
    }


def make_dataloader(dataset, indices, batch_size, shuffle=True, n_workers=0):
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, drop_last=False,
        num_workers=n_workers,
    )


# ---------------------------------------------------------------------------
# Split generation for active learning
# ---------------------------------------------------------------------------

def create_splits(n_total: int, n_init: int, test_ratio: float = 0.15,
                  val_ratio: float = 0.10, seed: int = 42):
    """Create init/pool/test/val splits for active learning.

    Returns:
        init_idx: initial labeled set (size n_init)
        pool_idx: unlabeled pool for AL querying
        test_idx: fixed test set
        val_idx: fixed validation set
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)

    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    remaining = indices[n_test + n_val:]

    init_idx = remaining[:n_init]
    pool_idx = remaining[n_init:]

    return init_idx, pool_idx, test_idx, val_idx


# ---------------------------------------------------------------------------
# Data loading from files (ASE trajectories, extxyz, etc.)
# ---------------------------------------------------------------------------

def load_structures_from_file(path: str, system_name: str = None) -> List:
    """Load structures from ASE-compatible file."""
    import ase.io
    structures = ase.io.read(path, index=":")
    if system_name:
        for s in structures:
            s.info["system"] = system_name
    return structures


def load_or_generate_data(system_name: str, config) -> MaterialDataset:
    """Load real data if available, otherwise generate synthetic structures
    for pipeline testing.

    Priority:
    1. User-specified data path in config
    2. MS25 data in standard locations
    3. Synthetic fallback (with warning)
    """
    data_dir = Path(config.data_dir) / system_name

    # Check for existing data
    for ext in [".extxyz", ".xyz", ".db", ".traj"]:
        pattern = str(data_dir / f"*{ext}")
        matches = list(Path(".").glob(pattern))
        if matches:
            print(f"  Loading {len(matches)} structures from {matches[0]}...")
            all_structures = []
            for path in matches:
                all_structures.extend(load_structures_from_file(str(path), system_name))
            return MaterialDataset(all_structures)

    # Check for pre-processed pickle
    pkl_path = data_dir / "structures.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            structures = pickle.load(f)
        print(f"  Loaded {len(structures)} structures from {pkl_path}")
        return MaterialDataset(structures)

    # Fallback: synthetic data for pipeline testing
    print(f"  [WARNING] No data found for {system_name}, generating synthetic structures")
    print(f"  Results with synthetic data are NOT scientifically valid - ")
    print(f"  replace with MS25/MP-ALOE structures for real experiments.")
    n_total = config.pool_size + config.n_init + 500  # 500 for test+val
    structures = generate_synthetic_structures(system_name, n_structures=n_total, seed=config.seed)
    return MaterialDataset(structures)
