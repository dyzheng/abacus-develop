"""
NEB calculation using pyabacus AbacusCalculator in ESolver mode.

ESolver mode keeps the ABACUS ESolver alive in memory between SCF calls,
avoiding repeated initialization overhead. This is expected to be faster
for geometry-optimization-like workloads (NEB, BFGS, etc.).

Usage:
    # Default (IS_CONTCAR.txt / FS_CONTCAR.txt, serial)
    python neb_esolver.py

    # Custom structures + 4 OpenMP threads
    python neb_esolver.py --is-file my_IS.vasp --fs-file my_FS.vasp --nthreads 4

    # With MPI (ESolver runs in-process, so use mpirun to launch)
    OMP_NUM_THREADS=2 mpirun -np 4 python neb_esolver.py --nthreads 2

    # 8 NEB images, tighter convergence
    python neb_esolver.py --n-images 8 --fmax 0.03

Output:
    - neb_esolver.traj      : NEB trajectory
    - IS_esolver.traj       : Initial-state relaxation trajectory
    - FS_esolver.traj       : Final-state relaxation trajectory
    - neb_esolver_result.npz : Barrier, energies, forces, timing
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

from ase.mep import NEB, NEBTools
from ase.optimize import BFGS

# Ensure pyabacus is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neb_common import (
    build_arg_parser,
    read_contcar,
    get_selective_dynamics,
    write_abacus_input_dir,
    NEB_K,
    NEB_FMAX_PHASE1, NEB_STEPS_PHASE1, NEB_STEPS_PHASE2,
    RELAX_STEPS,
)
from pyabacus.ase import AbacusCalculator


def run_neb_esolver(args):
    """Run NEB with ESolver-mode AbacusCalculator."""
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)

    # ---- Apply OMP_NUM_THREADS for ESolver in-process parallelism ----
    if args.nthreads > 1:
        os.environ["OMP_NUM_THREADS"] = str(args.nthreads)

    timing = {}
    results = {}

    # ------------------------------------------------------------------
    # 1. Read initial / final structures
    # ------------------------------------------------------------------
    initial = read_contcar(args.is_file)
    final = read_contcar(args.fs_file)
    move_flags_is = get_selective_dynamics(args.is_file)
    move_flags_fs = get_selective_dynamics(args.fs_file)

    print(f"System: {initial.get_chemical_formula()}")
    print(f"  Atoms  : {len(initial)}")
    print(f"  Cell   : {initial.cell.lengths()}")
    print(f"  Mode   : ESolver (memory-persistent)")
    print(f"  nthreads (OMP): {args.nthreads}")
    if args.nprocs > 1:
        print(f"  NOTE: ESolver runs in-process. For MPI parallelism use:")
        print(f"        mpirun -np {args.nprocs} python {sys.argv[0]} ...")
    print()

    # ------------------------------------------------------------------
    # 2. Prepare ABACUS input directories
    # ------------------------------------------------------------------
    is_dir = os.path.join(work_dir, "esolver_IS")
    fs_dir = os.path.join(work_dir, "esolver_FS")
    neb_dir = os.path.join(work_dir, "esolver_NEB")

    write_abacus_input_dir(initial, is_dir, move_flags=move_flags_is)
    write_abacus_input_dir(final, fs_dir, move_flags=move_flags_fs)

    # ------------------------------------------------------------------
    # 3. Relax endpoints
    # ------------------------------------------------------------------
    print("=== Relaxing initial state (ESolver mode) ===")
    t0 = time.perf_counter()

    calc_is = AbacusCalculator(input_dir=is_dir, mode="esolver", gamma_only=True)
    initial.calc = calc_is
    opt_is = BFGS(initial, trajectory="IS_esolver.traj")
    opt_is.run(fmax=args.relax_fmax, steps=RELAX_STEPS)

    timing["relax_IS"] = time.perf_counter() - t0
    results["energy_IS"] = initial.get_potential_energy()
    results["converged_IS"] = opt_is.converged()
    print(f"  Energy: {results['energy_IS']:.6f} eV")
    print(f"  Time  : {timing['relax_IS']:.1f} s")
    calc_is.cleanup()
    print()

    print("=== Relaxing final state (ESolver mode) ===")
    t0 = time.perf_counter()

    calc_fs = AbacusCalculator(input_dir=fs_dir, mode="esolver", gamma_only=True)
    final.calc = calc_fs
    opt_fs = BFGS(final, trajectory="FS_esolver.traj")
    opt_fs.run(fmax=args.relax_fmax, steps=RELAX_STEPS)

    timing["relax_FS"] = time.perf_counter() - t0
    results["energy_FS"] = final.get_potential_energy()
    results["converged_FS"] = opt_fs.converged()
    print(f"  Energy: {results['energy_FS']:.6f} eV")
    print(f"  Time  : {timing['relax_FS']:.1f} s")
    calc_fs.cleanup()
    print()

    # ------------------------------------------------------------------
    # 4. Build NEB images
    # ------------------------------------------------------------------
    n_images = args.n_images
    print(f"=== Building NEB ({n_images} images) ===")

    os.makedirs(neb_dir, exist_ok=True)
    write_abacus_input_dir(initial, neb_dir, move_flags=move_flags_is)

    images = [initial.copy()]
    for _ in range(n_images):
        images.append(initial.copy())
    images.append(final.copy())

    for img in images:
        calc = AbacusCalculator(
            input_dir=neb_dir, mode="esolver", gamma_only=True
        )
        img.calc = calc

    neb = NEB(images, k=NEB_K, climb=False, allow_shared_calculator=True)
    neb.interpolate()
    print("  Interpolation done.")
    print()

    # ------------------------------------------------------------------
    # 5. NEB Phase 1: coarse relaxation
    # ------------------------------------------------------------------
    print(f"=== NEB Phase 1 (fmax={NEB_FMAX_PHASE1}) ===")
    t0 = time.perf_counter()

    opt_neb = BFGS(neb, trajectory="neb_esolver.traj")
    conv1 = opt_neb.run(fmax=NEB_FMAX_PHASE1, steps=NEB_STEPS_PHASE1)

    timing["neb_phase1"] = time.perf_counter() - t0
    print(f"  Converged: {conv1}")
    print(f"  Time     : {timing['neb_phase1']:.1f} s")
    print()

    # ------------------------------------------------------------------
    # 6. NEB Phase 2: CI-NEB
    # ------------------------------------------------------------------
    if conv1:
        print(f"=== NEB Phase 2 CI-NEB (fmax={args.fmax}) ===")
        t0 = time.perf_counter()

        neb.climb = True
        conv2 = opt_neb.run(fmax=args.fmax, steps=NEB_STEPS_PHASE2)

        timing["neb_phase2"] = time.perf_counter() - t0
        print(f"  Converged: {conv2}")
        print(f"  Time     : {timing['neb_phase2']:.1f} s")
        print()
    else:
        timing["neb_phase2"] = 0.0

    # ------------------------------------------------------------------
    # 7. Collect results
    # ------------------------------------------------------------------
    neb_tool = NEBTools(neb.images)
    e_a, de = neb_tool.get_barrier()
    results["barrier_forward"] = e_a
    results["barrier_reverse"] = de
    results["image_energies"] = [
        img.get_potential_energy() for img in neb.images
    ]

    timing["total"] = sum(timing.values())

    print("=== Results (ESolver mode) ===")
    print(f"  Forward barrier : {e_a:.4f} eV")
    print(f"  Reverse barrier : {de:.4f} eV")
    print(f"  IS energy       : {results['energy_IS']:.6f} eV")
    print(f"  FS energy       : {results['energy_FS']:.6f} eV")
    print(f"  Total time      : {timing['total']:.1f} s")
    print()

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    np.savez(
        "neb_esolver_result.npz",
        **{k: np.array(v) for k, v in results.items()},
    )
    with open("neb_esolver_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print("Saved: neb_esolver_result.npz, neb_esolver_timing.json")

    # Cleanup all calculators
    for img in images:
        if hasattr(img.calc, "cleanup"):
            img.calc.cleanup()

    return results, timing


if __name__ == "__main__":
    parser = build_arg_parser("NEB with pyabacus ESolver mode")
    run_neb_esolver(parser.parse_args())
