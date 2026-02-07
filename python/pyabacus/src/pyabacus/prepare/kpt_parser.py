"""Read / write ABACUS KPT (K_POINTS) files."""

from __future__ import annotations

import os
from typing import Any, Dict, List


def _normalise_mode(raw: str) -> str:
    """Map the first non-blank word of the mode line to a canonical name."""
    low = raw.strip().lower()
    if low.startswith("g"):
        return "gamma"
    if low.startswith("m"):
        return "mp"
    if low.startswith("d"):
        return "direct"
    if low == "line_cartesian":
        return "line_cartesian"
    if low.startswith("l"):
        return "line"
    if low.startswith("c"):
        return "cartesian"
    raise ValueError(f"Unknown KPT mode: {raw!r}")


def read_kpt(filepath: str) -> Dict[str, Any]:
    """Read an ABACUS KPT file and return a dict.

    Returns
    -------
    dict with key ``"mode"`` plus mode-specific fields:
      * gamma / mp  → ``grid`` (3-int list), ``shift`` (3-float list)
      * direct / cartesian → ``points`` list of ``[kx, ky, kz, weight]``
      * line / line_cartesian → ``points`` list of ``[kx, ky, kz, npts]``
        or ``[kx, ky, kz, npts, label]``
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"KPT file not found: {filepath}")

    with open(filepath) as fh:
        lines = [l for l in fh.readlines()
                 if l.split("#")[0].strip()]

    if len(lines) < 3:
        raise ValueError(f"KPT file too short: {filepath}")

    mode = _normalise_mode(lines[2].split()[0])

    if mode in ("gamma", "mp"):
        tokens = lines[3].split()
        grid = [int(t) for t in tokens[:3]]
        shift = [float(t) for t in tokens[3:6]] if len(tokens) >= 6 else [0, 0, 0]
        return {"mode": mode, "grid": grid, "shift": shift}

    if mode in ("direct", "cartesian"):
        nk = int(lines[1].split()[0])
        points: List[List[float]] = []
        for i in range(nk):
            tokens = lines[3 + i].split()
            points.append([float(t) for t in tokens[:4]])
        return {"mode": mode, "points": points}

    # line / line_cartesian
    points_line: List[list] = []
    for line in lines[3:]:
        if not line.strip():
            break
        tokens = line.split()
        entry: list = [float(tokens[0]), float(tokens[1]),
                       float(tokens[2]), int(tokens[3])]
        if len(tokens) > 4:
            entry.append(" ".join(tokens[4:]))
        points_line.append(entry)
    return {"mode": mode, "points": points_line}


def write_kpt(kpt_data: Dict[str, Any], filepath: str) -> None:
    """Write a kpt dict to an ABACUS KPT file."""
    mode = kpt_data["mode"]
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w") as fh:
        if mode in ("gamma", "mp"):
            mode_str = "Gamma" if mode == "gamma" else "MP"
            grid = kpt_data["grid"]
            shift = kpt_data.get("shift", [0, 0, 0])
            vals = list(grid) + list(shift)
            fh.write(f"K_POINTS\n0\n{mode_str}\n")
            fh.write(" ".join(str(v) for v in vals) + "\n")

        elif mode in ("direct", "cartesian"):
            pts = kpt_data["points"]
            fh.write(f"K_POINTS\n{len(pts)}\n{mode.capitalize()}\n")
            for p in pts:
                fh.write("%17.11f %17.11f %17.11f %17.11f\n" % tuple(p[:4]))

        elif mode in ("line", "line_cartesian"):
            pts = kpt_data["points"]
            mode_str = "Line" if mode == "line" else "Line_Cartesian"
            fh.write(f"K_POINTS\n{len(pts)}\n{mode_str}\n")
            for p in pts:
                fh.write("%17.11f %17.11f %17.11f %4d" % tuple(p[:4]))
                if len(p) > 4:
                    label = p[4]
                    if not (label.startswith("#") or label.startswith("//")):
                        label = "#" + label
                    fh.write(f" {label}")
                fh.write("\n")
        else:
            raise ValueError(f"Unknown KPT mode: {mode!r}")
