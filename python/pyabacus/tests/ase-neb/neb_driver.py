"""
NEB calculation using pyabacus AbacusCalculator in Driver mode.

Driver mode runs a fresh ABACUS process for each energy/force evaluation,
reading/writing files on disk. This is the "traditional" approach with
full isolation between SCF calls, but higher overhead from repeated
process startup and I/O.

Because the driver reads STRU from disk, we subclass AbacusCalculator to
re-write the STRU file with updated atomic positions before each call.

Usage:
    # Default (IS_CONTCAR.txt / FS_CONTCAR.txt, serial)
    python neb_driver.py

    # Custom structures + 4 MPI + 2 threads
    python neb_driver.py --is-file my_IS.vasp --fs-file my_FS.vasp \
                         --nprocs 4 --nthreads 2

    # 8 NEB images, tighter convergence
    python neb_driver.py --n-images 8 --fmax 0.03

Output:
    - neb_driver.traj      : NEB trajectory
    - IS_driver.traj       : Initial-state relaxation trajectory
    - FS_driver.traj       : Final-state relaxation trajectory
    - neb_driver_result.npz : Barrier, energies, forces, timing
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
    _write_stru,
    NEB_K,
    NEB_FMAX_PHASE1, NEB_STEPS_PHASE1, NEB_STEPS_PHASE2,
    RELAX_STEPS,
)
from pyabacus.ase import AbacusCalculator
from pyabacus.ase.calculator import all_changes


class DriverNEBCalculator(AbacusCalculator):
    """AbacusCalculator (driver mode) that re-writes STRU before each call.

    Standard driver mode reads INPUT/STRU/KPT from input_dir but does not
    update STRU when ASE changes positions. This subclass overrides
    calculate() to write the updated structure to disk first.
    """

    def __init__(self, input_dir, move_flags=None, **kwargs):
        kwargs["mode"] = "driver"
        super().__init__(input_dir=input_dir, **kwargs)
        self._move_flags = move_flags

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if properties is None:
            properties = ["energy", "forces"]
        if system_changes is None:
            system_changes = all_changes

        # Update self.atoms from argument
        if atoms is not None:
            self.atoms = atoms.copy()

        # Re-write STRU with current positions before calling ABACUS
        if self.atoms is not None:
            _write_stru(self.atoms, self.input_dir, self._move_flags)

        super().calculate(self.atoms, properties, system_changes)


def run_neb_driver(args):
    """Run NEB with Driver-mode AbacusCalculator."""
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)

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
    print(f"  Atoms   : {len(initial)}")
    print(f"  Cell    : {initial.cell.lengths()}")
    print(f"  Mode    : Driver (subprocess, file-based I/O)")
    print(f"  nprocs  : {args.nprocs}")
    print(f"  nthreads: {args.nthreads}")
    print()

    # ------------------------------------------------------------------
    # 2. Prepare ABACUS input directories (one per endpoint)
    # ------------------------------------------------------------------
    is_dir = os.path.join(work_dir, "driver_IS")
    fs_dir = os.path.join(work_dir, "driver_FS")

    write_abacus_input_dir(initial, is_dir, move_flags=move_flags_is)
    write_abacus_input_dir(final, fs_dir, move_flags=move_flags_fs)

    # Prepare per-image directories for NEB
    n_images = args.n_images
    neb_dirs = []
    for i in range(n_images + 2):
        d = os.path.join(work_dir, f"driver_NEB_{i:02d}")
        write_abacus_input_dir(initial, d, move_flags=move_flags_is)
        neb_dirs.append(d)

    # ------------------------------------------------------------------
    # 3. Relax endpoints
    # ------------------------------------------------------------------
    print("=== Relaxing initial state (Driver mode) ===")
    t0 = time.perf_counter()

    calc_is = DriverNEBCalculator(
        input_dir=is_dir, move_flags=move_flags_is,
        nprocs=args.nprocs, nthreads=args.nthreads,
    )
    initial.calc = calc_is
    opt_is = BFGS(initial, trajectory="IS_driver.traj")
    opt_is.run(fmax=args.relax_fmax, steps=RELAX_STEPS)

    timing["relax_IS"] = time.perf_counter() - t0
    results["energy_IS"] = initial.get_potential_energy()
    results["converged_IS"] = opt_is.converged()
    print(f"  Energy: {results['energy_IS']:.6f} eV")
    print(f"  Time  : {timing['relax_IS']:.1f} s")
    print()

    print("=== Relaxing final state (Driver mode) ===")
    t0 = time.perf_counter()

    calc_fs = DriverNEBCalculator(
        input_dir=fs_dir, move_flags=move_flags_fs,
        nprocs=args.nprocs, nthreads=args.nthreads,
    )
    final.calc = calc_fs
    opt_fs = BFGS(final, trajectory="FS_driver.traj")
    opt_fs.run(fmax=args.relax_fmax, steps=RELAX_STEPS)

    timing["relax_FS"] = time.perf_counter() - t0
    results["energy_FS"] = final.get_potential_energy()
    results["converged_FS"] = opt_fs.converged()
    print(f"  Energy: {results['energy_FS']:.6f} eV")
    print(f"  Time  : {timing['relax_FS']:.1f} s")
    print()

    # ------------------------------------------------------------------
    # 4. Build NEB images
    # ------------------------------------------------------------------
    print(f"=== Building NEB ({n_images} images) ===")

    images = [initial.copy()]
    for _ in range(n_images):
        images.append(initial.copy())
    images.append(final.copy())

    # Each image gets its own calculator with its own input directory
    for i, img in enumerate(images):
        img.calc = DriverNEBCalculator(
            input_dir=neb_dirs[i], move_flags=move_flags_is,
            nprocs=args.nprocs, nthreads=args.nthreads,
        )

    neb = NEB(images, k=NEB_K, climb=False)
    neb.interpolate()
    print("  Interpolation done.")
    print()

    # ------------------------------------------------------------------
    # 5. NEB Phase 1: coarse relaxation
    # ------------------------------------------------------------------
    print(f"=== NEB Phase 1 (fmax={NEB_FMAX_PHASE1}) ===")
    t0 = time.perf_counter()

    opt_neb = BFGS(neb, trajectory="neb_driver.traj")
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

    print("=== Results (Driver mode) ===")
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
        "neb_driver_result.npz",
        **{k: np.array(v) for k, v in results.items()},
    )
    with open("neb_driver_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print("Saved: neb_driver_result.npz, neb_driver_timing.json")

    return results, timing


if __name__ == "__main__":
    parser = build_arg_parser("NEB with pyabacus Driver mode")
    run_neb_driver(parser.parse_args())
