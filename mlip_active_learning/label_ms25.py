"""Label MS25 structures with MACE-MP-0 energies on GPU."""
import os, sys, pickle, time, argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="ms25_data")
    parser.add_argument("--output-dir", default="data/ms25_labeled")
    parser.add_argument("--model-path",
        default="/share/home/tm949679661250000/a954358970/gpumace/mace_offline_package/mace-mp-0-medium.model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--systems", nargs="*", default=None)
    args = parser.parse_args()

    from mace.calculators import MACECalculator
    print(f"Loading MACE from {args.model_path}...")
    calc = MACECalculator(model_path=args.model_path, device=args.device, default_dtype="float32")
    print(f"MACE ready: {calc.num_models} model(s) on {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    systems = args.systems or [f.stem for f in Path(args.input_dir).glob("*.pkl")]
    print(f"Systems: {systems}")

    for sys_name in systems:
        pkl_path = os.path.join(args.input_dir, f"{sys_name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  SKIP {sys_name}: not found")
            continue

        print(f"\nLabeling {sys_name}...")
        with open(pkl_path, "rb") as f:
            structures = pickle.load(f)

        n_fail = 0
        t0 = time.time()
        for i, atoms in enumerate(structures):
            atoms.calc = calc
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                atoms.info["energy"] = energy
                atoms.info["forces"] = forces
            except Exception as e:
                n_fail += 1

            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(structures)} ({rate:.1f}/s)")

        elapsed = time.time() - t0
        out_path = os.path.join(args.output_dir, f"{sys_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(structures, f)

        energies = [s.info["energy"] for s in structures]
        print(f"  Done: {len(structures)} in {elapsed:.0f}s "
              f"({len(structures)/elapsed:.1f}/s), {n_fail} failed")
        print(f"  Energy: [{min(energies):.1f}, {max(energies):.1f}] eV")

    print(f"\nAll done! Labeled data in {args.output_dir}/")

if __name__ == "__main__":
    main()
