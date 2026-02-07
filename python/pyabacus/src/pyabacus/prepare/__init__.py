"""pyabacus.prepare — Read, write, and convert ABACUS input files.

Public API
----------
read_directory / write_directory   Round-trip INPUT + STRU + KPT as a dict.
read_input / write_input           INPUT file only.
read_kpt / write_kpt               KPT file only.
read_stru / write_stru             STRU file only.
to_json / from_json                JSON serialisation.
from_cif / from_poscar             Build dict from CIF / POSCAR (needs ASE).
"""

from pyabacus.prepare.input_parser import read_input, write_input
from pyabacus.prepare.kpt_parser import read_kpt, write_kpt
from pyabacus.prepare.stru_parser import read_stru, write_stru
from pyabacus.prepare.directory import read_directory, write_directory
from pyabacus.prepare.convert import to_json, from_json, from_cif, from_poscar

__all__ = [
    "read_input",
    "write_input",
    "read_kpt",
    "write_kpt",
    "read_stru",
    "write_stru",
    "read_directory",
    "write_directory",
    "to_json",
    "from_json",
    "from_cif",
    "from_poscar",
]
