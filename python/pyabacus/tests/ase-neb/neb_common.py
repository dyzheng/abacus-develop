"""
Shared utilities for NEB test scripts.

Provides functions to:
- Read VASP CONTCAR files and return ASE Atoms
- Generate ABACUS input directories (INPUT, STRU, KPT) from Atoms
- Common NEB parameters and CLI argument parser
"""

import argparse
import os
import shutil
import numpy as np
from ase.io import read as ase_read

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PP_ORB_DIR = "/root/abacus-develop/tests/PP_ORB"

# Pseudopotential mapping: element -> filename
PP_MAP = {
    "Ce": "58_Ce.UPF",
    "C": "C_ONCV_PBE-1.0.upf",
    "O": "O_ONCV_PBE-1.0.upf",
}

# Orbital mapping: element -> filename
ORB_MAP = {
    "Ce": "Ce_gga_8au_80Ry_4s2p2d2f.Orb",
    "C": "C_gga_8au_100Ry_2s2p1d.orb",
    "O": "O_gga_7au_100Ry_2s2p1d.orb",
}

# ABACUS INPUT parameters for LCAO-SCF
INPUT_PARAMS = {
    "calculation": "scf",
    "basis_type": "lcao",
    "gamma_only": 1,
    "ecutwfc": 80,
    "scf_thr": 1.0e-6,
    "scf_nmax": 200,
    "smearing_method": "gaussian",
    "smearing_sigma": 0.01,
    "mixing_type": "broyden",
    "mixing_beta": 0.3,
    "cal_force": 1,
    "cal_stress": 0,
    "pseudo_dir": PP_ORB_DIR,
    "orbital_dir": PP_ORB_DIR,
    "suffix": "PYABACUS",
    "out_level": "ie",
}

# NEB default parameters
NEB_K = 1.0                 # Spring constant
NEB_N_IMAGES = 5            # Intermediate images (total = N_IMAGES + 2)
NEB_FMAX_PHASE1 = 0.45      # eV/Ang for initial relaxation
NEB_FMAX_PHASE2 = 0.05      # eV/Ang for CI-NEB
NEB_STEPS_PHASE1 = 200
NEB_STEPS_PHASE2 = 300
RELAX_FMAX = 0.05           # eV/Ang for endpoint relaxation
RELAX_STEPS = 200


# ---------------------------------------------------------------------------
# CLI argument parser (shared by both scripts)
# ---------------------------------------------------------------------------

def build_arg_parser(description="Run ASE-NEB with pyabacus"):
    """Return an ArgumentParser with common NEB arguments.

    Both neb_esolver.py and neb_driver.py call this to define a
    consistent CLI interface.
    """
    p = argparse.ArgumentParser(description=description)

    # ---- Structure files ----
    p.add_argument(
        "--is-file", default="IS_CONTCAR.txt",
        help="Path to initial-state VASP CONTCAR/POSCAR (default: IS_CONTCAR.txt)",
    )
    p.add_argument(
        "--fs-file", default="FS_CONTCAR.txt",
        help="Path to final-state VASP CONTCAR/POSCAR (default: FS_CONTCAR.txt)",
    )

    # ---- Parallelism ----
    p.add_argument(
        "--nprocs", type=int, default=1,
        help="Number of MPI processes (default: 1). "
             "Driver mode: passed to subprocess. "
             "ESolver mode: must launch with mpirun -np N instead.",
    )
    p.add_argument(
        "--nthreads", type=int, default=1,
        help="Number of OpenMP threads per MPI process (default: 1)",
    )

    # ---- NEB parameters ----
    p.add_argument(
        "--n-images", type=int, default=NEB_N_IMAGES,
        help=f"Number of intermediate NEB images (default: {NEB_N_IMAGES})",
    )
    p.add_argument(
        "--fmax", type=float, default=NEB_FMAX_PHASE2,
        help=f"CI-NEB force convergence in eV/Ang (default: {NEB_FMAX_PHASE2})",
    )
    p.add_argument(
        "--relax-fmax", type=float, default=RELAX_FMAX,
        help=f"Endpoint relaxation force convergence in eV/Ang (default: {RELAX_FMAX})",
    )

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_contcar(filename):
    """Read a VASP CONTCAR/POSCAR and return an ASE Atoms object."""
    return ase_read(filename, format="vasp")


def get_selective_dynamics(filename):
    """Read selective-dynamics flags from a VASP CONTCAR.

    Returns a list of [mx, my, mz] per atom, where 1 = movable, 0 = fixed.
    Returns None if no selective dynamics.
    """
    with open(filename) as f:
        lines = f.readlines()

    # Check for selective dynamics line
    idx = 7  # line after atom counts
    if lines[idx].strip().lower().startswith("s"):
        idx += 1  # skip "Selective dynamics"

    coord_line = lines[idx].strip()
    if coord_line.lower().startswith(("d", "c", "k")):
        idx += 1
    else:
        return None

    flags = []
    for line in lines[idx:]:
        tokens = line.split()
        if len(tokens) < 6:
            break
        mx = 1 if tokens[3].upper() == "T" else 0
        my = 1 if tokens[4].upper() == "T" else 0
        mz = 1 if tokens[5].upper() == "T" else 0
        flags.append([mx, my, mz])

    return flags if flags else None


def write_abacus_input_dir(atoms, output_dir, move_flags=None,
                           extra_params=None):
    """Write ABACUS INPUT, STRU, KPT files for the given Atoms.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to write.
    output_dir : str
        Directory to write files into.
    move_flags : list of [int,int,int], optional
        Per-atom selective dynamics flags (1=move, 0=fix).
    extra_params : dict, optional
        Override or add to INPUT_PARAMS.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Merge INPUT parameters
    params = dict(INPUT_PARAMS)
    if extra_params:
        params.update(extra_params)

    # ---- Write INPUT ----
    input_path = os.path.join(output_dir, "INPUT")
    with open(input_path, "w") as f:
        f.write("INPUT_PARAMETERS\n")
        for key, val in params.items():
            f.write(f"{key} {val}\n")

    # ---- Write KPT (Gamma only) ----
    kpt_path = os.path.join(output_dir, "KPT")
    with open(kpt_path, "w") as f:
        f.write("K_POINTS\n0\nGamma\n1 1 1 0 0 0\n")

    # ---- Write STRU ----
    _write_stru(atoms, output_dir, move_flags)


def _write_stru(atoms, output_dir, move_flags=None):
    """Write ABACUS STRU file from ASE Atoms."""
    from pyabacus.constants import ANG_TO_BOHR

    symbols = atoms.get_chemical_symbols()
    cell_ang = atoms.get_cell()
    cell_bohr = cell_ang * ANG_TO_BOHR
    scaled_pos = atoms.get_scaled_positions()

    # Build per-type info
    species_order = []
    for s in symbols:
        if s not in species_order:
            species_order.append(s)

    stru_path = os.path.join(output_dir, "STRU")
    with open(stru_path, "w") as f:
        # ATOMIC_SPECIES
        f.write("ATOMIC_SPECIES\n")
        for elem in species_order:
            mass = atoms[symbols.index(elem)].mass
            pp = PP_MAP[elem]
            f.write(f"{elem} {mass:.4f} {pp}\n")

        # NUMERICAL_ORBITAL
        f.write("\nNUMERICAL_ORBITAL\n")
        for elem in species_order:
            f.write(f"{ORB_MAP[elem]}\n")

        # LATTICE_CONSTANT
        f.write("\nLATTICE_CONSTANT\n1.0\n")

        # LATTICE_VECTORS (in Bohr)
        f.write("\nLATTICE_VECTORS\n")
        for row in cell_bohr:
            f.write(f"{row[0]:17.11f} {row[1]:17.11f} {row[2]:17.11f}\n")

        # ATOMIC_POSITIONS
        f.write("\nATOMIC_POSITIONS\nDirect\n")

        atom_idx = 0
        for elem in species_order:
            # Collect indices of this element
            indices = [i for i, s in enumerate(symbols) if s == elem]
            f.write(f"\n{elem}\n")
            f.write("0.0\n")  # magmom
            f.write(f"{len(indices)}\n")
            for i in indices:
                sx, sy, sz = scaled_pos[i]
                f.write(f"{sx:17.11f} {sy:17.11f} {sz:17.11f} ")
                if move_flags is not None and i < len(move_flags):
                    mx, my, mz = move_flags[i]
                    f.write(f"{mx} {my} {mz}")
                else:
                    f.write("1 1 1")
                f.write("\n")
