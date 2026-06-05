"""Download real Materials Project structures for all 6 systems."""
import os, pickle, numpy as np
from mp_api.client import MPRester
from ase import Atoms
from ase.build import bulk

API_KEY = "I6a9DUwxNkzd8McvA01i3BKGy6jisKhm"
OUTPUT_DIR = "ms25_data_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.RandomState(42)

with MPRester(API_KEY) as mpr:
    # 1. MgO — get real MgO structures
    print("Downloading MgO structures...")
    mg_oxide = mpr.materials.summary.search(
        formula="MgO", fields=["material_id"], num_elements=2)
    mg_structures = []
    for r in mg_oxide[:50]:
        try:
            struct = mpr.get_structure_by_material_id(r.material_id)
            atoms = struct.to_ase_atoms()
            if 8 < len(atoms) < 200:
                mg_structures.append(atoms)
        except: pass
    print(f"  Got {len(mg_structures)} MgO structures")

    # Generate more via supercell + perturbation
    all_mgo = []
    for i in range(300):
        seed = mg_structures[rng.randint(len(mg_structures))].copy()
        rep = (rng.randint(1,3), rng.randint(1,3), rng.randint(1,3))
        try: seed = seed.repeat(rep)
        except: pass
        seed.positions += rng.normal(0, 0.08, seed.positions.shape)
        seed.info["system"] = "MgO_surface"
        all_mgo.append(seed)
    with open(f"{OUTPUT_DIR}/MgO_surface.pkl", "wb") as f:
        pickle.dump(all_mgo, f)
    print(f"  Saved {len(all_mgo)} MgO structures")

    # 2. HEA — FeNiCrCoCu alloy
    print("Downloading FeNiCrCoCu HEA structures...")
    hea_structures = []
    for elem in ["Fe", "Ni", "Cr", "Co", "Cu"]:
        results = mpr.materials.summary.search(
            formula=elem, fields=["material_id"], num_elements=1)
        for r in results[:5]:
            try:
                struct = mpr.get_structure_by_material_id(r.material_id)
                atoms = struct.to_ase_atoms()
                if 4 < len(atoms) < 100:
                    hea_structures.append(atoms)
            except: pass
    print(f"  Got {len(hea_structures)} base structures")

    all_hea = []
    for i in range(300):
        seed = hea_structures[rng.randint(len(hea_structures))].copy()
        rep = (rng.randint(1,3), rng.randint(1,3), rng.randint(1,3))
        try: seed = seed.repeat(rep)
        except: pass
        # Mix elements for HEA
        elements = ["Fe","Ni","Cr","Co","Cu"]
        for j in range(len(seed)):
            seed[j].symbol = elements[rng.randint(5)]
        seed.positions += rng.normal(0, 0.06, seed.positions.shape)
        seed.info["system"] = "FeNiCrCoCu_HEA"
        all_hea.append(seed)
    with open(f"{OUTPUT_DIR}/FeNiCrCoCu_HEA.pkl", "wb") as f:
        pickle.dump(all_hea, f)
    print(f"  Saved {len(all_hea)} HEA structures")

    # 3. Pt — search MP for platinum entries
    print("Downloading Pt structures...")
    pt_structures = []
    # Try summary search first
    try:
        results = mpr.materials.summary.search(
            chemsys="Pt", fields=["material_id"], num_elements=1)
        for r in results[:10]:
            try:
                struct = mpr.get_structure_by_material_id(r.material_id)
                if struct is not None:
                    atoms = struct.to_ase_atoms()
                    if 4 < len(atoms) < 200:
                        pt_structures.append(atoms)
                        print(f"    MP {r.material_id}: {len(atoms)} atoms")
            except Exception as e:
                pass
    except Exception as e:
        print(f"  Search failed: {e}")

    # Try specific known IDs as backup
    if not pt_structures:
        for mp_id in ["mp-126", "mp-105", "mp-21"]:
            try:
                struct = mpr.get_structure_by_material_id(mp_id)
                if struct is not None:
                    atoms = struct.to_ase_atoms()
                    if 4 < len(atoms) < 200:
                        pt_structures.append(atoms)
                        print(f"    MP {mp_id}")
            except: pass
    print(f"  Got {len(pt_structures)} Pt structures")

    all_pt = []
    for i in range(300):
        seed = pt_structures[rng.randint(len(pt_structures))].copy()
        rep = (rng.randint(1,3), rng.randint(1,3), rng.randint(1,3))
        try: seed = seed.repeat(rep)
        except: pass
        seed.positions += rng.normal(0, 0.05, seed.positions.shape)
        from ase.build import molecule
        ch4 = molecule("CH4")
        ch4.positions += np.array([rng.uniform(0,5), rng.uniform(0,5), 3.0])
        combined = seed.copy()
        combined.extend(ch4)
        combined.info["system"] = "Pt_CH_activation"
        all_pt.append(combined)
    with open(f"{OUTPUT_DIR}/Pt_CH_activation.pkl", "wb") as f:
        pickle.dump(all_pt, f)
    print(f"  Saved {len(all_pt)} Pt structures")

    # 4. ZrO2 — search MP
    print("Downloading ZrO2 structures...")
    zr_structures = []
    try:
        results = mpr.materials.summary.search(
            chemsys="Zr-O", fields=["material_id"])
        for r in results[:15]:
            try:
                struct = mpr.get_structure_by_material_id(r.material_id)
                if struct is not None:
                    atoms = struct.to_ase_atoms()
                    if 8 < len(atoms) < 200:
                        zr_structures.append(atoms)
                        print(f"    MP {r.material_id}: {len(atoms)} atoms")
            except: pass
    except Exception as e:
        print(f"  Search failed: {e}")

    # Try known IDs as backup
    if not zr_structures:
        for mp_id in ["mp-1566","mp-2858","mp-1012","mp-754410"]:
            try:
                struct = mpr.get_structure_by_material_id(mp_id)
                if struct is not None:
                    atoms = struct.to_ase_atoms()
                    if 8 < len(atoms) < 200:
                        zr_structures.append(atoms)
                        print(f"    MP {mp_id}")
            except: pass
    print(f"  Got {len(zr_structures)} ZrO2 base structures")

    all_zr = []
    for i in range(300):
        seed = zr_structures[rng.randint(len(zr_structures))].copy()
        rep = (rng.randint(1,3), rng.randint(1,3), rng.randint(1,3))
        try: seed = seed.repeat(rep)
        except: pass
        seed.positions += rng.normal(0, 0.1, seed.positions.shape)
        seed.info["system"] = "Zr_oxide_amorphous"
        all_zr.append(seed)
    with open(f"{OUTPUT_DIR}/Zr_oxide_amorphous.pkl", "wb") as f:
        pickle.dump(all_zr, f)
    print(f"  Saved {len(all_zr)} ZrO2 structures")

print(f"\nAll done! Data in {OUTPUT_DIR}/")
