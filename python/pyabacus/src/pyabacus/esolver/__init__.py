"""
PyABACUS ESolver Module
=======================

This module provides Python bindings for ABACUS ESolver classes,
enabling Python-controlled SCF workflows with breakpoint support.

Main Classes
------------
ESolverLCAO_gamma : ESolver for gamma-only LCAO calculations
ESolverLCAO_multi_k : ESolver for multi-k LCAO calculations
ESolverPW_cf : ESolver for plane wave calculations (single precision)
ESolverPW_cd : ESolver for plane wave calculations (double precision)
LCAOWorkflow : High-level workflow wrapper for LCAO with callback support
PWWorkflow : High-level workflow wrapper for plane wave calculations

Example (LCAO)
--------------
>>> from pyabacus.esolver import LCAOWorkflow
>>>
>>> workflow = LCAOWorkflow("./")
>>> workflow.initialize()
>>> result = workflow.run_scf(max_iter=100)
>>> print(f"Total energy: {result.energy.etot}")

Example (PW)
------------
>>> from pyabacus.esolver import PWWorkflow
>>>
>>> workflow = PWWorkflow("./")
>>> workflow.initialize()
>>> result = workflow.run_scf(max_iter=100)
>>> print(f"Total energy: {result.energy.etot}")
"""

from .workflow import LCAOWorkflow
from .pw_workflow import PWWorkflow
from .data_types import ChargeData, EnergyData, HamiltonianData, DensityMatrixData, SCFResult, ForceData, StressData

# Import C++ bindings
ForceAccessor = None
StressAccessor = None
ESolverPW_cf = None
ESolverPW_cd = None
try:
    from ._esolver_pack import (
        ESolverLCAO_gamma,
        ESolverLCAO_multi_k,
        ChargeAccessor,
        EnergyAccessor,
        HamiltonianAccessor_gamma,
        HamiltonianAccessor_multi_k,
        DensityMatrixAccessor_gamma,
        DensityMatrixAccessor_multi_k,
    )
    # Try to import new accessors (may not be available in older builds)
    try:
        from ._esolver_pack import ForceAccessor, StressAccessor
    except ImportError:
        pass  # ForceAccessor/StressAccessor not available in this build
    # Try to import PW ESolver classes
    try:
        from ._esolver_pack import ESolverPW_cf, ESolverPW_cd
    except ImportError:
        pass  # PW ESolver not available in this build
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import _esolver_pack: {e}. "
                  "ESolver bindings may not be available.")

    # Define placeholder classes for documentation
    ESolverLCAO_gamma = None
    ESolverLCAO_multi_k = None
    ChargeAccessor = None
    EnergyAccessor = None
    HamiltonianAccessor_gamma = None
    HamiltonianAccessor_multi_k = None
    DensityMatrixAccessor_gamma = None
    DensityMatrixAccessor_multi_k = None

__all__ = [
    # High-level interfaces
    'LCAOWorkflow',
    'PWWorkflow',

    # Data types
    'ChargeData',
    'EnergyData',
    'HamiltonianData',
    'DensityMatrixData',
    'SCFResult',
    'ForceData',
    'StressData',

    # Low-level C++ bindings (LCAO)
    'ESolverLCAO_gamma',
    'ESolverLCAO_multi_k',
    'ChargeAccessor',
    'EnergyAccessor',
    'ForceAccessor',
    'StressAccessor',
    'HamiltonianAccessor_gamma',
    'HamiltonianAccessor_multi_k',
    'DensityMatrixAccessor_gamma',
    'DensityMatrixAccessor_multi_k',

    # Low-level C++ bindings (PW)
    'ESolverPW_cf',
    'ESolverPW_cd',
]
