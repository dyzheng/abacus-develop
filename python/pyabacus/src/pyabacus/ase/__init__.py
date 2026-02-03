"""
ASE Calculator interface for pyabacus.

This module provides an ASE-compatible Calculator that uses pyabacus ESolver
directly for efficient geometry optimization with ASE optimizers.

Two calculation modes are supported:
- ESolver Mode (default): Memory persistent, efficient for sequential calculations
- Driver Mode: Independent calculations via file I/O

Example
-------
>>> from ase import Atoms
>>> from ase.optimize import BFGS
>>> from pyabacus.ase import AbacusCalculator, CalculatorMode
>>>
>>> # ESolver mode (default) with context manager
>>> with AbacusCalculator(input_dir='./Si_scf/') as calc:
...     atoms.calc = calc
...     opt = BFGS(atoms)
...     opt.run(fmax=0.01)
>>>
>>> # Driver mode for independent calculations
>>> calc = AbacusCalculator(input_dir='./Si_scf/', mode=CalculatorMode.DRIVER)
>>> atoms.calc = calc
>>> energy = atoms.get_potential_energy()
"""

from .calculator import AbacusCalculator, CalculatorMode

__all__ = ['AbacusCalculator', 'CalculatorMode']
