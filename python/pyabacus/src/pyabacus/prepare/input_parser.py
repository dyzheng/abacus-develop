"""Read / write ABACUS INPUT files."""

from __future__ import annotations

import os
import re
from typing import Any, Dict


def _auto_convert(value_str: str) -> Any:
    """Try int → float → str for a single token, list for multi-token."""
    tokens = value_str.split()
    if len(tokens) > 1:
        try:
            return [int(t) for t in tokens]
        except ValueError:
            pass
        try:
            return [float(t) for t in tokens]
        except ValueError:
            return tokens
    # single token
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        return value_str


def read_input(filepath: str) -> Dict[str, Any]:
    """Read an ABACUS INPUT file and return a dict.

    Keys are lowercased.  Values are auto-converted (int / float / str).
    Multi-value entries become lists.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"INPUT file not found: {filepath}")

    result: Dict[str, Any] = {}
    with open(filepath) as fh:
        for line in fh:
            stripped = line.split("#")[0].strip()
            if not stripped:
                continue
            # Split on first whitespace/tab
            parts = re.split(r"[ \t]+", stripped, maxsplit=1)
            if len(parts) != 2:
                continue  # header line like INPUT_PARAMETERS
            key = parts[0].lower()
            val = _auto_convert(parts[1].strip())
            result[key] = val
    return result


def write_input(params: Dict[str, Any], filepath: str) -> None:
    """Write a dict to an ABACUS INPUT file.

    Lists/tuples are joined with spaces.  ``None`` values are written as
    comment lines.
    """
    lines = ["INPUT_PARAMETERS\n"]
    for key, val in params.items():
        if val is None:
            lines.append(f"#{key}\n")
        else:
            if isinstance(val, (list, tuple)):
                val_str = " ".join(str(v) for v in val)
            else:
                val_str = str(val)
            lines.append(f"{key}\t\t\t{val_str}\n")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as fh:
        fh.writelines(lines)
