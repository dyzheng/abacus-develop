"""JSON serialisation and CIF / POSCAR conversion utilities."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ..constants import BOHR_TO_ANG, ANG_TO_BOHR


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def to_json(data: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """Serialise a unified dict to a JSON string.

    If *filepath* is given the JSON is also written to that file.
    """
    text = json.dumps(data, indent=2, default=str)
    if filepath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as fh:
            fh.write(text)
    return text


def from_json(source: str) -> Dict[str, Any]:
    """Load a unified dict from a JSON string or file path."""
    if os.path.isfile(source):
        with open(source) as fh:
            return json.load(fh)
    return json.loads(source)


# ---------------------------------------------------------------------------
# CIF / POSCAR
# ---------------------------------------------------------------------------

def _atoms_to_stru(atoms: Any,
                   pp_dict: Optional[Dict[str, str]] = None,
                   orb_dict: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Convert an ASE ``Atoms`` object to a flat stru dict.

    Cell is converted from Angstrom to Bohr.  Coordinates are fractional
    (Direct).
    """
    cell_ang = atoms.cell.tolist()
    cell_bohr = [[v * ANG_TO_BOHR for v in row] for row in cell_ang]
    scaled = atoms.get_scaled_positions().tolist()

    # Build per-type info
    symbols = list(atoms.get_chemical_symbols())
    seen: list = []
    label: list = []
    atom_number: list = []
    mass_list: list = []

    for s in symbols:
        if s not in seen:
            seen.append(s)
            label.append(s)
            atom_number.append(0)
            mass_list.append(atoms[symbols.index(s)].mass)
        atom_number[seen.index(s)] += 1

    # Re-order coords so atoms of the same type are grouped
    ordered_coords: list = []
    for lbl in label:
        for idx, s in enumerate(symbols):
            if s == lbl:
                ordered_coords.append(scaled[idx])

    pp = [pp_dict.get(l) for l in label] if pp_dict else None
    orb = [orb_dict.get(l) for l in label] if orb_dict else None

    return {
        "label": label,
        "atom_number": atom_number,
        "mass": mass_list,
        "pp": pp,
        "orb": orb,
        "paw": None,
        "cell": cell_bohr,
        "coord": ordered_coords,
        "lattice_constant": 1.0,
        "cartesian": False,
        "move": None,
        "magmom": [0.0] * len(label),
        "magmom_atom": None,
        "velocity": None,
        "angle1": None,
        "angle2": None,
        "constrain": None,
        "lambda_": None,
        "dpks": None,
    }


def from_cif(
    cif_path: str,
    input_params: Optional[Dict[str, Any]] = None,
    pp_dict: Optional[Dict[str, str]] = None,
    orb_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a unified dict from a CIF file.

    Requires **ASE** (``ase.io.read``) or **pymatgen**.
    """
    atoms = _read_structure(cif_path)
    stru = _atoms_to_stru(atoms, pp_dict=pp_dict, orb_dict=orb_dict)
    inp = dict(input_params) if input_params else {
        "calculation": "scf", "basis_type": "pw"}
    kpt = {"mode": "gamma", "grid": [1, 1, 1], "shift": [0, 0, 0]}
    return {"input": inp, "stru": stru, "kpt": kpt}


def from_poscar(
    poscar_path: str,
    input_params: Optional[Dict[str, Any]] = None,
    pp_dict: Optional[Dict[str, str]] = None,
    orb_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a unified dict from a VASP POSCAR file.  Requires **ASE**."""
    atoms = _read_structure(poscar_path, fmt="vasp")
    stru = _atoms_to_stru(atoms, pp_dict=pp_dict, orb_dict=orb_dict)
    inp = dict(input_params) if input_params else {
        "calculation": "scf", "basis_type": "pw"}
    kpt = {"mode": "gamma", "grid": [1, 1, 1], "shift": [0, 0, 0]}
    return {"input": inp, "stru": stru, "kpt": kpt}


def _read_structure(path: str, fmt: Optional[str] = None) -> Any:
    """Read a structure file via ASE, falling back to pymatgen."""
    try:
        from ase.io import read as ase_read
        return ase_read(path, format=fmt)
    except ImportError:
        pass
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        struct = Structure.from_file(path)
        return AseAtomsAdaptor.get_atoms(struct)
    except ImportError:
        raise ImportError(
            "Either ASE or pymatgen is required for CIF/POSCAR conversion. "
            "Install one with: pip install ase   or   pip install pymatgen")
