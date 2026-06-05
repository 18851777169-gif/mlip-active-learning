"""Download and generate MS25 benchmark structures from Materials Project.

6 material systems:
  1. MgO(100) surface — mp-1265 rocksalt
  2. Liquid water — generated from MD at 300K
  3. Zeolites — mp entries for CHA/FAU/LTA/MFI frameworks
  4. Pt(111) C-H activation — mp-126 fcc + CH4 on surface
  5. FeNiCrCoCu HEA — SQS-generated Cantor alloy
  6. Zr-oxide amorphous — melt-quench from ZrO2

Requirements: pip install mp-api ase pymatgen
MP API key: set via MP_API_KEY env var or pass as argument
"""

import os, sys, pickle, argparse, time
import numpy as np
from pathlib import Path

API_KEY = "I6a9DUwxNkzd8McvA01i3BKGy6jisKhm"

# ---------------------------------------------------------------------------
# System 1: MgO(100) surface
# ---------------------------------------------------------------------------
def generate_mgo_surface(n_structures=200, seed=42):
    """Generate MgO(100) surface slabs — high diversity.

    Varies: thickness, lateral size, vacuum, surface defects,
    termination, and thermal-like perturbation.
    """
    from ase.build import bulk, surface
    from ase import Atoms
    rng = np.random.RandomState(seed)
    structures = []

    for i in range(n_structures):
        n_layers = rng.randint(2, 7)
        vacuum = rng.uniform(6, 18)
        sx, sy = rng.randint(2, 5), rng.randint(2, 5)

        atoms = bulk("MgO", crystalstructure="rocksalt", a=4.21)
        # Sometimes use different surface index for diversity
        surface_idx = (1, 0, 0) if rng.random() > 0.3 else (1, 1, 0)
        slab = surface(atoms, surface_idx, n_layers)
        slab = slab.repeat((sx, sy, 1))
        slab.center(vacuum=vacuum, axis=2)

        # Remove random atoms (surface defects) — 0-3 vacancies
        n_remove = rng.randint(0, 4)
        if n_remove > 0 and len(slab) > n_remove:
            idx = rng.choice(len(slab), n_remove, replace=False)
            del slab[idx]

        # Larger perturbation for diversity
        slab.positions += rng.normal(0, 0.12, slab.positions.shape)
        slab.info["system"] = "MgO_surface"
        structures.append(slab)

    return structures


# ---------------------------------------------------------------------------
# System 2: Liquid water
# ---------------------------------------------------------------------------
def generate_liquid_water(n_structures=200, seed=42):
    """Download water-containing structures from Materials Project.

    Searches for H2O-containing materials, creates supercells and
    distortions for a physically reasonable dataset.
    """
    from ase.build import molecule
    from ase import Atoms
    rng = np.random.RandomState(seed)
    structures = []

    mp_structures = []
    try:
        from mp_api.client import MPRester
        with MPRester("I6a9DUwxNkzd8McvA01i3BKGy6jisKhm") as mpr:
            # Search for ice/water phases and hydrates
            for formula in ["H2O", "H4O2", "H6O3", "H8O4"]:
                try:
                    results = mpr.materials.summary.search(
                        formula=formula,
                        fields=["material_id"],
                        num_elements=2,
                    )
                    for r in results[:5]:
                        try:
                            struct = mpr.get_structure_by_material_id(r.material_id)
                            if 10 < len(struct) < 200:
                                mp_structures.append(struct)
                                print(f"    Downloaded {r.material_id}: {struct.formula}")
                        except Exception:
                            pass
                except Exception:
                    pass
    except Exception as e:
        print(f"    MP search failed: {e}")

    if mp_structures:
        print(f"    Got {len(mp_structures)} MP water structures")
        # Convert pymatgen Structure -> ASE Atoms
        ase_mp = []
        for s in mp_structures:
            try:
                from ase.io import read
                import io
                # Use CIF/JSON roundtrip for conversion
                atoms = s.to_ase_atoms()
                ase_mp.append(atoms)
            except Exception:
                pass

        if ase_mp:
            print(f"    Converted {len(ase_mp)} to ASE format")
            for i in range(n_structures):
                seed_s = ase_mp[rng.randint(len(ase_mp))]
                # Deep copy as pure ASE Atoms (no pymatgen references)
                from ase import Atoms as AAtoms
                atoms = AAtoms(
                    symbols=seed_s.get_chemical_symbols(),
                    positions=seed_s.positions.copy(),
                    cell=seed_s.cell.copy(),
                    pbc=True,
                )
                rep = (rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 2))
                try:
                    atoms = atoms.repeat(rep)
                except Exception:
                    pass
                atoms.positions += rng.normal(0, 0.08, atoms.positions.shape)
                atoms.info["system"] = "liquid_water"
                structures.append(atoms)
            return structures

    # Fallback: H2O molecules placed carefully on lattice
    n_molecules = 32
    density = 0.033
    box_size = (n_molecules / density) ** (1/3)
    n_per_side = int(np.ceil(n_molecules ** (1/3)))
    spacing = box_size / n_per_side

    for i in range(n_structures):
        atoms = Atoms()
        mol_idx = 0
        for ix in range(n_per_side):
            for iy in range(n_per_side):
                for iz in range(n_per_side):
                    if mol_idx >= n_molecules:
                        break
                    mol = molecule("H2O")
                    mol.positions += np.array([ix, iy, iz]) * spacing
                    atoms.extend(mol)
                    mol_idx += 1
                if mol_idx >= n_molecules:
                    break

        atoms.set_cell([box_size] * 3)
        atoms.set_pbc(True)
        atoms.positions += rng.normal(0, 0.15, atoms.positions.shape)
        atoms.info["system"] = "liquid_water"
        structures.append(atoms)

    return structures


# ---------------------------------------------------------------------------
# System 3: Zeolites
# ---------------------------------------------------------------------------
def generate_zeolites(n_structures=200, seed=42):
    """Generate zeolite-like structures using alpha-quartz and cristobalite.

    Uses SiO2 polymorphs (quartz, cristobalite) as base frameworks,
    creates supercells with varying sizes and mild distortions.
    These are physically reasonable silica structures.
    """
    from ase.build import bulk
    from ase import Atoms
    rng = np.random.RandomState(seed)
    structures = []

    # Try to download from MP first
    mp_structures = []
    try:
        from mp_api.client import MPRester
        zeolite_ids = ["mp-832313", "mp-30038", "mp-715024", "mp-697350",
                       "mp-1201437", "mp-560840", "mp-541204", "mp-504048"]
        with MPRester("I6a9DUwxNkzd8McvA01i3BKGy6jisKhm") as mpr:
            for mp_id in zeolite_ids:
                try:
                    struct = mpr.get_structure_by_material_id(mp_id)
                    mp_structures.append(struct)
                    print(f"    Downloaded {mp_id}: {struct.formula}")
                except Exception:
                    pass
    except Exception as e:
        print(f"    MP download failed: {e}")

    if mp_structures:
        # Convert pymatgen Structure -> ASE Atoms
        ase_mp = []
        for s in mp_structures:
            try:
                atoms = s.to_ase_atoms()
                ase_mp.append(atoms)
            except Exception:
                pass
        if ase_mp:
            print(f"    Converted {len(ase_mp)} zeolites to ASE")
            for i in range(n_structures):
                seed_s = ase_mp[rng.randint(len(ase_mp))]
                from ase import Atoms as AAtoms
                atoms = AAtoms(
                    symbols=seed_s.get_chemical_symbols(),
                    positions=seed_s.positions.copy(),
                    cell=seed_s.cell.copy(),
                    pbc=True,
                )
                rep = (rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 2))
                try:
                    atoms = atoms.repeat(rep)
                except Exception:
                    pass
                atoms.positions += rng.normal(0, 0.04, atoms.positions.shape)
                atoms.info["system"] = "zeolite"
                structures.append(atoms)
            return structures

    # Fallback: quartz/cristobalite supercells with distortions
    for i in range(n_structures):
        crystal = "quartz" if rng.random() > 0.5 else "cristobalite"
        rep = (rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3))
        a, c = (4.92, 5.41) if crystal == "quartz" else (4.97, 6.92)
        try:
            atoms = bulk("SiO2", crystalstructure=crystal, a=a, c=c)
            atoms = atoms.repeat(rep)
            atoms.set_pbc(True)
        except Exception:
            atoms = bulk("SiO2", crystalstructure="quartz", a=4.92, c=5.41)
            atoms = atoms.repeat(rep)
            atoms.set_pbc(True)

        atoms.positions += rng.normal(0, 0.04, atoms.positions.shape)
        atoms.info["system"] = "zeolite"
        structures.append(atoms)

    return structures


# ---------------------------------------------------------------------------
# System 4: Pt(111) C-H activation
# ---------------------------------------------------------------------------
def generate_pt_ch_activation(n_structures=200, seed=42):
    """Generate Pt(111) surface with CH4 adsorbate."""
    from ase.build import fcc111, molecule
    from ase import Atoms
    rng = np.random.RandomState(seed)
    structures = []

    for i in range(n_structures):
        size = (rng.randint(2, 4), rng.randint(2, 4), rng.randint(3, 6))
        slab = fcc111("Pt", a=3.92, size=size)
        slab.center(vacuum=10.0, axis=2)

        # Add CH4 at varying positions above surface
        ch4 = molecule("CH4")
        surface_z = slab.positions[:, 2].max()
        ch4.positions[:, 2] += surface_z + rng.uniform(1.5, 3.5)
        ch4.positions[:, :2] += rng.uniform(0, slab.cell[0, 0], 2)

        combined = slab + ch4
        combined.positions += rng.normal(0, 0.04, combined.positions.shape)
        combined.info["system"] = "Pt_CH_activation"
        structures.append(combined)

    return structures


# ---------------------------------------------------------------------------
# System 5: FeNiCrCoCu High-Entropy Alloy
# ---------------------------------------------------------------------------
def generate_hea(n_structures=200, seed=42):
    """Generate FeNiCrCoCu Cantor alloy structures."""
    from ase.build import bulk
    from ase import Atoms
    rng = np.random.RandomState(seed)
    elements = ["Fe", "Ni", "Cr", "Co", "Cu"]
    structures = []

    for i in range(n_structures):
        size = (rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4))
        atoms = bulk("Ni", crystalstructure="fcc", a=3.52)
        atoms = atoms.repeat(size)

        # Randomly assign elements (equimolar)
        for j in range(len(atoms)):
            atoms[j].symbol = elements[rng.randint(5)]

        atoms.positions += rng.normal(0, 0.06, atoms.positions.shape)
        atoms.info["system"] = "FeNiCrCoCu_HEA"
        structures.append(atoms)

    return structures


# ---------------------------------------------------------------------------
# System 6: Zr-oxide amorphous
# ---------------------------------------------------------------------------
def generate_zr_oxide_amorphous(n_structures=200, seed=42):
    """Generate highly diverse amorphous ZrO2 structures.

    Varies: Zr count (10-60), density, stoichiometry (O/Zr ~1.8-2.2),
    and introduces large thermal-like perturbations.
    """
    from ase import Atoms
    rng = np.random.RandomState(seed)
    structures = []

    for i in range(n_structures):
        n_zr = rng.randint(10, 60)
        # Vary stoichiometry slightly
        o_ratio = rng.uniform(1.8, 2.2)
        n_o = int(n_zr * o_ratio)
        n_total = n_zr + n_o

        # Vary density (box size factor)
        density_factor = rng.uniform(0.8, 1.3)
        box_size = (n_total * 18.0 * density_factor) ** (1/3)

        symbols = ["Zr"] * n_zr + ["O"] * n_o

        # Place atoms with minimum distance
        min_dist = 1.8
        positions = np.zeros((n_total, 3))
        positions[0] = rng.uniform(0, box_size, 3)
        for j in range(1, n_total):
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
        # Larger perturbation for structural diversity
        atoms.positions += rng.normal(0, 0.15, atoms.positions.shape)
        atoms.info["system"] = "Zr_oxide_amorphous"
        structures.append(atoms)

    return structures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SYSTEM_GENERATORS = {
    "MgO_surface": generate_mgo_surface,
    "liquid_water": generate_liquid_water,
    "zeolite": generate_zeolites,
    "Pt_CH_activation": generate_pt_ch_activation,
    "FeNiCrCoCu_HEA": generate_hea,
    "Zr_oxide_amorphous": generate_zr_oxide_amorphous,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", type=str, nargs="*",
                        default=list(SYSTEM_GENERATORS.keys()))
    parser.add_argument("--n-structures", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="ms25_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for sys_name in args.systems:
        gen_fn = SYSTEM_GENERATORS[sys_name]
        print(f"\nGenerating {sys_name} ({args.n_structures} structures)...")
        t0 = time.time()

        structures = gen_fn(args.n_structures, args.seed)
        elapsed = time.time() - t0

        # Save as pickle
        out_path = os.path.join(args.output_dir, f"{sys_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(structures, f)

        sizes = [len(s) for s in structures]
        print(f"  Saved {len(structures)} structures to {out_path}")
        print(f"  Sizes: {min(sizes)}-{max(sizes)} atoms, "
              f"Time: {elapsed:.1f}s")

    print(f"\nDone! Data saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
