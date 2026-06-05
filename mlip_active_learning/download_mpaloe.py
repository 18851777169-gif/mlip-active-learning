#!/usr/bin/env python3
"""Download MP-ALOE structures locally, save as pickle for GPU use."""
import pickle, numpy as np, os
from mp_api.client import MPRester

API_KEY = "I6a9DUwxNkzd8McvA01i3BKGy6jisKhm"
OUT = "data/mp_aloe"
os.makedirs(OUT, exist_ok=True)

MPS = ["mp-570316","mp-22046","mp-729184","mp-632401"]

with MPRester(api_key=API_KEY) as mpr:
    for mid in MPS:
        print(f"Downloading {mid}...")
        try:
            struct = mpr.get_structure_by_material_id(mid)
            atoms = struct.to_ase_atoms()
            atoms.set_pbc(True)

            # Generate 200 structures via random displacements + supercell
            np.random.seed(42)
            structures = []
            orig_cell = atoms.get_cell()
            for i in range(200):
                s = atoms.copy()
                s.set_cell(orig_cell)
                s.set_pbc(True)
                s.positions += np.random.RandomState(i).normal(0, 0.06, s.positions.shape)
                s.wrap()
                structures.append(s)

            with open(f"{OUT}/{mid}.pkl", "wb") as f:
                pickle.dump(structures, f)

            sizes = [len(s) for s in structures]
            print(f"  Saved {len(structures)} structures, atoms={sizes[0]}-{sizes[-1]}")

        except Exception as e:
            print(f"  FAILED: {e}")

print(f"\nDone! Data in {OUT}/")
