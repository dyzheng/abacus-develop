"""Read / write a complete ABACUS input directory (INPUT + STRU + KPT)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pyabacus.prepare.input_parser import read_input, write_input
from pyabacus.prepare.kpt_parser import read_kpt, write_kpt
from pyabacus.prepare.stru_parser import read_stru, write_stru


def read_directory(dir_path: str) -> Dict[str, Any]:
    """Read INPUT + STRU + KPT from *dir_path* and return a unified dict.

    The returned dict has three top-level keys: ``"input"``, ``"stru"``,
    ``"kpt"`` (the last may be ``None`` when no KPT file exists).
    """
    input_path = os.path.join(dir_path, "INPUT")
    inp = read_input(input_path)

    stru_file = inp.get("stru_file", "STRU")
    kpt_file = inp.get("kpoint_file", "KPT")

    stru_path = os.path.join(dir_path, stru_file)
    stru = read_stru(stru_path)

    kpt_path = os.path.join(dir_path, kpt_file)
    kpt: Optional[Dict[str, Any]] = None
    if os.path.isfile(kpt_path):
        kpt = read_kpt(kpt_path)

    return {"input": inp, "stru": stru, "kpt": kpt}


def write_directory(data: Dict[str, Any], dir_path: str) -> None:
    """Write a unified dict to INPUT + STRU + KPT files in *dir_path*."""
    os.makedirs(dir_path, exist_ok=True)

    inp = data.get("input", {})
    stru_file = inp.get("stru_file", "STRU")
    kpt_file = inp.get("kpoint_file", "KPT")

    write_input(inp, os.path.join(dir_path, "INPUT"))

    if "stru" in data and data["stru"] is not None:
        write_stru(data["stru"], os.path.join(dir_path, stru_file))

    if "kpt" in data and data["kpt"] is not None:
        write_kpt(data["kpt"], os.path.join(dir_path, kpt_file))
