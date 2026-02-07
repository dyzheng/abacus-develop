"""Read / write ABACUS STRU files (flat dict format)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

_STRU_KEYWORDS = {
    "ATOMIC_SPECIES",
    "NUMERICAL_ORBITAL",
    "LATTICE_CONSTANT",
    "LATTICE_VECTORS",
    "ATOMIC_POSITIONS",
    "NUMERICAL_DESCRIPTOR",
    "PAW_FILES",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_block(lines: List[str], keyname: str) -> Optional[List[str]]:
    """Extract the block of non-empty, non-comment lines following *keyname*."""
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if line.split("#")[0].split("//")[0].strip() == keyname:
            block: List[str] = []
            for j in range(i + 1, len(lines)):
                raw = lines[j]
                stripped = raw.strip()
                if stripped == "":
                    continue
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                if stripped in _STRU_KEYWORDS:
                    return block
                block.append(raw.split("#")[0].split("//")[0].strip())
            return block
    return None


def _parse_position(pos_line: str) -> Tuple:
    """Parse a single atom position line.

    Returns (pos, move, velocity, magmom, angle1, angle2, constrain, lambda_).
    """
    tokens = pos_line.split()
    pos = [float(tokens[i]) for i in range(3)]
    move = None
    velocity = None
    magmom = None
    angle1 = None
    angle2 = None
    constrain = None
    lambda_ = None

    if len(tokens) <= 3:
        return pos, move, velocity, magmom, angle1, angle2, constrain, lambda_

    # Collect tagged sublists
    move_l: List[int] = []
    vel_l: List[float] = []
    mag_l: List[float] = []
    a1_l: List[float] = []
    a2_l: List[float] = []
    con_l: List[bool] = []
    lam_l: List[float] = []
    label = "move"  # default: first unlabelled numbers are move flags

    for tok in tokens[3:]:
        if tok == "m":
            label = "move"; move_l = []; continue
        if tok in ("v", "vel", "velocity"):
            label = "velocity"; vel_l = []; continue
        if tok in ("mag", "magmom"):
            label = "magmom"; mag_l = []; continue
        if tok == "angle1":
            label = "angle1"; a1_l = []; continue
        if tok == "angle2":
            label = "angle2"; a2_l = []; continue
        if tok in ("constrain", "sc"):
            label = "constrain"; con_l = []; continue
        if tok == "lambda":
            label = "lambda"; lam_l = []; continue
        # value
        if label == "move":
            move_l.append(int(tok))
        elif label == "velocity":
            vel_l.append(float(tok))
        elif label == "magmom":
            mag_l.append(float(tok))
        elif label == "angle1":
            a1_l.append(float(tok))
        elif label == "angle2":
            a2_l.append(float(tok))
        elif label == "constrain":
            con_l.append(bool(int(tok)))
        elif label == "lambda":
            lam_l.append(float(tok))

    if len(move_l) == 3:
        move = move_l
    if len(vel_l) == 3:
        velocity = vel_l
    if len(mag_l) in (1, 3):
        magmom = mag_l if len(mag_l) == 3 else mag_l[0]
    if len(a1_l) == 1:
        angle1 = a1_l[0]
    if len(a2_l) == 1:
        angle2 = a2_l[0]
    if len(con_l) == 3:
        constrain = con_l
    elif len(con_l) == 1:
        constrain = con_l[0]
    if len(lam_l) == 3:
        lambda_ = lam_l
    elif len(lam_l) == 1:
        lambda_ = lam_l[0]

    return pos, move, velocity, magmom, angle1, angle2, constrain, lambda_


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_stru(filepath: str) -> Dict[str, Any]:
    """Read an ABACUS STRU file and return a flat dict.

    Keys follow the abacus-test convention (see plan doc for schema).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"STRU file not found: {filepath}")

    with open(filepath) as fh:
        lines = fh.readlines()

    atomic_species = _get_block(lines, "ATOMIC_SPECIES")
    numerical_orbital = _get_block(lines, "NUMERICAL_ORBITAL")
    lattice_constant_block = _get_block(lines, "LATTICE_CONSTANT")
    lattice_vectors = _get_block(lines, "LATTICE_VECTORS")
    atom_positions = _get_block(lines, "ATOMIC_POSITIONS")
    dpks_block = _get_block(lines, "NUMERICAL_DESCRIPTOR")
    paw_block = _get_block(lines, "PAW_FILES")

    lat_const = 1.0 if lattice_constant_block is None else float(
        lattice_constant_block[0].split()[0])
    dpks = None if dpks_block is None else dpks_block[0].strip()

    # --- species ---
    labels: List[str] = []
    mass_list: List[float] = []
    pp_list: List[Optional[str]] = []
    for line in (atomic_species or []):
        parts = line.split()
        labels.append(parts[0])
        mass_list.append(float(parts[1]))
        pp_list.append(parts[2] if len(parts) > 2 else None)

    # --- orbitals ---
    orb_list: Optional[List[str]] = None
    if numerical_orbital is not None:
        orb_list = [l.split()[0] for l in numerical_orbital]

    # --- PAW files ---
    paw_list: Optional[List[str]] = None
    if paw_block is not None:
        paw_list = [l.strip() for l in paw_block]

    # --- cell ---
    cell: List[List[float]] = []
    if lattice_vectors:
        for line in lattice_vectors:
            cell.append([float(x) for x in line.split()[:3]])

    # --- positions ---
    if atom_positions is None or len(atom_positions) == 0:
        raise ValueError("ATOMIC_POSITIONS block missing or empty")

    coord_type = atom_positions[0].split()[0].lower()
    cartesian = coord_type.startswith("cart")

    atom_number: List[int] = []
    coords: List[List[float]] = []
    magmom_global: List[float] = []
    magmom_atom: List[Any] = []
    move_list: List[Any] = []
    velocity_list: List[Any] = []
    angle1_list: List[Any] = []
    angle2_list: List[Any] = []
    constrain_list: List[Any] = []
    lambda_list: List[Any] = []

    real_labels: List[str] = []
    real_pp: List[Optional[str]] = []
    real_orb: List[Optional[str]] = []
    real_paw: List[Optional[str]] = []
    real_mass: List[float] = []

    i = 1
    while i < len(atom_positions):
        lbl = atom_positions[i].strip()
        if lbl not in labels:
            raise ValueError(
                f"Label '{lbl}' not found in ATOMIC_SPECIES: {labels}")
        an = int(atom_positions[i + 2].split()[0])
        if an == 0:
            i += 3
            continue

        lbl_idx = labels.index(lbl)
        real_labels.append(lbl)
        real_mass.append(mass_list[lbl_idx])
        real_pp.append(pp_list[lbl_idx] if pp_list else None)
        real_orb.append(orb_list[lbl_idx] if orb_list else None)
        real_paw.append(paw_list[lbl_idx] if paw_list else None)
        magmom_global.append(float(atom_positions[i + 1].split()[0]))
        atom_number.append(an)

        i += 3
        for j in range(an):
            (pos, mv, vel, mag, a1, a2, con, lam) = _parse_position(
                atom_positions[i + j])
            coords.append(pos)
            move_list.append(mv)
            velocity_list.append(vel)
            magmom_atom.append(mag)
            angle1_list.append(a1)
            angle2_list.append(a2)
            constrain_list.append(con)
            lambda_list.append(lam)
        i += an

    # Collapse per-atom lists to None if all entries are None
    def _collapse(lst):
        return None if all(v is None for v in lst) else lst

    return {
        "label": real_labels,
        "atom_number": atom_number,
        "mass": real_mass,
        "pp": real_pp if any(v is not None for v in real_pp) else None,
        "orb": real_orb if any(v is not None for v in real_orb) else None,
        "paw": real_paw if any(v is not None for v in real_paw) else None,
        "cell": cell,
        "coord": coords,
        "lattice_constant": lat_const,
        "cartesian": cartesian,
        "move": _collapse(move_list),
        "magmom": magmom_global,
        "magmom_atom": _collapse(magmom_atom),
        "velocity": _collapse(velocity_list),
        "angle1": _collapse(angle1_list),
        "angle2": _collapse(angle2_list),
        "constrain": _collapse(constrain_list),
        "lambda_": _collapse(lambda_list),
        "dpks": dpks,
    }


def write_stru(stru_data: Dict[str, Any], filepath: str) -> None:
    """Write a flat stru dict to an ABACUS STRU file.

    Values are written as-is (no unit conversion).
    """
    label = stru_data["label"]
    atom_number = stru_data["atom_number"]
    coord = stru_data["coord"]
    cell = stru_data["cell"]
    lat_const = stru_data.get("lattice_constant", 1.0)
    cartesian = stru_data.get("cartesian", False)
    mass = stru_data.get("mass")
    pp = stru_data.get("pp")
    orb = stru_data.get("orb")
    paw = stru_data.get("paw")
    magmom_global = stru_data.get("magmom")
    move = stru_data.get("move")
    magmom_atom = stru_data.get("magmom_atom")
    velocity = stru_data.get("velocity")
    angle1 = stru_data.get("angle1")
    angle2 = stru_data.get("angle2")
    constrain = stru_data.get("constrain")
    lambda_ = stru_data.get("lambda_")
    dpks = stru_data.get("dpks")

    ntype = len(label)
    natom = sum(atom_number)
    assert len(coord) == natom

    cc = "ATOMIC_SPECIES\n"
    for i, lbl in enumerate(label):
        m = mass[i] if mass and i < len(mass) else 1.0
        p = pp[i] if pp and i < len(pp) and pp[i] is not None else ""
        cc += f"{lbl} {m} {p}\n"

    # NUMERICAL_ORBITAL
    if orb and any(o is not None for o in orb):
        cc += "\nNUMERICAL_ORBITAL\n"
        for o in orb:
            cc += f"{o}\n"

    # PAW_FILES
    if paw and any(p is not None for p in paw):
        cc += "\nPAW_FILES\n"
        for p in paw:
            cc += f"{p}\n"

    # LATTICE_CONSTANT
    cc += f"\nLATTICE_CONSTANT\n{lat_const}\n"

    # LATTICE_VECTORS
    cc += "\nLATTICE_VECTORS\n"
    for row in cell:
        cc += "%17.11f %17.11f %17.11f\n" % tuple(row)

    # ATOMIC_POSITIONS
    cc += "\nATOMIC_POSITIONS\n"
    cc += "Cartesian\n" if cartesian else "Direct\n"

    idx = 0
    for i, lbl in enumerate(label):
        cc += f"\n{lbl}\n"
        mg = magmom_global[i] if magmom_global and i < len(magmom_global) else 0.0
        cc += f"{mg}\n"
        cc += f"{atom_number[i]}\n"
        for j in range(atom_number[i]):
            ai = idx + j
            cc += "%17.11f %17.11f %17.11f " % tuple(coord[ai])
            # move flags
            if move and ai < len(move) and move[ai] is not None:
                cc += "%d %d %d " % tuple(move[ai])
            # velocity
            if velocity and ai < len(velocity) and velocity[ai] is not None:
                cc += "v %f %f %f " % tuple(velocity[ai])
            # magmom
            if magmom_atom and ai < len(magmom_atom) and magmom_atom[ai] is not None:
                mag = magmom_atom[ai]
                if isinstance(mag, list):
                    if len(mag) == 3:
                        cc += "mag %12.8f %12.8f %12.8f " % tuple(mag)
                    elif len(mag) == 1:
                        cc += "mag %12.8f " % mag[0]
                else:
                    cc += "mag %12.8f " % mag
            # angle1 / angle2
            if angle1 and ai < len(angle1) and angle1[ai] is not None:
                cc += "angle1 %f " % angle1[ai]
            if angle2 and ai < len(angle2) and angle2[ai] is not None:
                cc += "angle2 %f " % angle2[ai]
            # constrain
            if constrain and ai < len(constrain) and constrain[ai] is not None:
                cv = constrain[ai]
                if isinstance(cv, list) and len(cv) == 3:
                    cc += "sc " + " ".join(
                        "1" if c else "0" for c in cv) + " "
                elif isinstance(cv, list) and len(cv) == 1:
                    cc += "sc " + ("1" if cv[0] else "0") + " "
                elif not isinstance(cv, list):
                    cc += "sc " + ("1" if cv else "0") + " "
            # lambda
            if lambda_ and ai < len(lambda_) and lambda_[ai] is not None:
                lv = lambda_[ai]
                if isinstance(lv, list):
                    cc += "lambda " + " ".join(str(x) for x in lv) + " "
                else:
                    cc += "lambda " + str(lv) + " "
            cc += "\n"
        idx += atom_number[i]

    # NUMERICAL_DESCRIPTOR
    if dpks:
        cc += "\nNUMERICAL_DESCRIPTOR\n"
        cc += dpks + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as fh:
        fh.write(cc)
