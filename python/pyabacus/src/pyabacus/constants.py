"""Unit conversion constants for PyABACUS.

All values must match ``source/source_base/constants.h`` in the C++ codebase.
"""

RY_TO_EV = 13.605698              # 1 Ry = 13.605698 eV
BOHR_TO_ANG = 0.529177249         # 1 Bohr = 0.529177249 Angstrom
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG  # 1 Angstrom in Bohr
RY_BOHR_TO_EV_ANG = RY_TO_EV / BOHR_TO_ANG  # ~25.7112, force conversion
KBAR_TO_EV_ANG3 = 1.0 / 1602.1766208        # kbar -> eV/Angstrom^3

ENERGY_FIELDS = [
    'etot', 'eband', 'hartree_energy', 'etxc',
    'ewald_energy', 'demet', 'exx', 'evdw',
]
