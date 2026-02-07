"""
High-level runner interface for ABACUS calculations.

This module provides the `abacus()` function - the main entry point
for running ABACUS DFT calculations from Python.

Two implementations are available:
1. C++ bindings (if available) - direct library calls
2. Subprocess fallback - calls the abacus executable
"""

from typing import Optional
from pathlib import Path
import numpy as np

from .result import CalculationResult
from .subprocess_runner import run_abacus_subprocess
from ..constants import RY_TO_EV, BOHR_TO_ANG, ENERGY_FIELDS


def abacus(
    input_dir: Optional[str] = None,
    *,
    input_file: Optional[str] = None,
    stru_file: Optional[str] = None,
    kpt_file: Optional[str] = None,
    pseudo_dir: Optional[str] = None,
    orbital_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    calculate_force: bool = True,
    calculate_stress: bool = False,
    verbosity: int = 1,
    nprocs: int = 1,
    nthreads: int = 1,
) -> CalculationResult:
    """
    Run an ABACUS DFT calculation.

    This is the main entry point for running ABACUS calculations from Python.
    It provides the same functionality as the ABACUS command-line program.

    Parameters
    ----------
    input_dir : str, optional
        Directory containing INPUT, STRU, KPT files.
        If not specified, uses current directory.
    input_file : str, optional
        Explicit path to INPUT file. Overrides input_dir/INPUT.
    stru_file : str, optional
        Explicit path to STRU file. Overrides value in INPUT.
    kpt_file : str, optional
        Explicit path to KPT file. Overrides value in INPUT.
    pseudo_dir : str, optional
        Directory containing pseudopotential files.
        Overrides pseudo_dir in INPUT.
    orbital_dir : str, optional
        Directory containing orbital files (for LCAO).
        Overrides orbital_dir in INPUT.
    output_dir : str, optional
        Directory for output files. Default: "OUT.PYABACUS"
    calculate_force : bool, optional
        Whether to calculate forces. Default: True
    calculate_stress : bool, optional
        Whether to calculate stress tensor. Default: False
    verbosity : int, optional
        Output verbosity level:
        - 0: Silent (no output)
        - 1: Normal (default)
        - 2: Verbose (detailed output)
    nprocs : int, optional
        Number of MPI processes. Default: 1
        Equivalent to: mpirun -np nprocs abacus
    nthreads : int, optional
        Number of OpenMP threads. Default: 1
        Equivalent to: OMP_NUM_THREADS=nthreads

    Returns
    -------
    CalculationResult
        Object containing all calculation results including:
        - converged: Whether SCF converged
        - etot: Total energy (eV)
        - etot_ev: Total energy (eV)
        - forces: Forces on atoms (if calculate_force=True)
        - stress: Stress tensor (if calculate_stress=True)
        - energies: Dictionary of energy components

    Raises
    ------
    FileNotFoundError
        If INPUT file is not found
    RuntimeError
        If calculation fails or ABACUS is not installed

    Examples
    --------
    Basic SCF calculation:

    >>> result = pyabacus.abacus("./Si_scf/")
    >>> print(f"Energy: {result.etot_ev:.6f} eV")
    >>> print(f"Converged: {result.converged}")

    Calculate forces and stress:

    >>> result = pyabacus.abacus(
    ...     "./Si_relax/",
    ...     calculate_force=True,
    ...     calculate_stress=True,
    ... )
    >>> print(f"Max force: {np.max(np.abs(result.forces_ev_ang)):.4f} eV/Ang")

    Parallel calculation with MPI and OpenMP:

    >>> result = pyabacus.abacus(
    ...     "./Si_scf/",
    ...     nprocs=4,      # 4 MPI processes
    ...     nthreads=2,    # 2 OpenMP threads per process
    ... )

    Silent mode:

    >>> result = pyabacus.abacus("./Si_scf/", verbosity=0)
    """
    # Try to use C++ driver first
    try:
        from ._driver_pack import PyDriver
        _HAS_CPP_DRIVER = True
    except ImportError:
        _HAS_CPP_DRIVER = False

    # Determine input directory
    if input_dir is None:
        if input_file is not None:
            input_dir = str(Path(input_file).parent)
        else:
            input_dir = "."

    # Validate input directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Check for INPUT file
    if input_file is None:
        input_file_path = input_path / "INPUT"
        if not input_file_path.exists():
            raise FileNotFoundError(
                f"INPUT file not found in {input_dir}. "
                "Please provide input_file parameter or ensure INPUT exists."
            )

    # Set default output directory
    if output_dir is None:
        output_dir = "OUT.PYABACUS"

    if _HAS_CPP_DRIVER:
        # Use C++ driver
        driver = PyDriver()

        cpp_result = driver.run(
            input_dir=str(input_dir),
            input_file=input_file or "",
            stru_file=stru_file or "",
            kpt_file=kpt_file or "",
            pseudo_dir=pseudo_dir or "",
            orbital_dir=orbital_dir or "",
            output_dir=output_dir or "",
            calculate_force=calculate_force,
            calculate_stress=calculate_stress,
            verbosity=verbosity,
        )

        # Convert C++ result to Python dataclass
        # C++ CalculationResult stores energies in Rydberg;
        # Python CalculationResult stores energies in eV.
        energy_kwargs = {f: getattr(cpp_result, f) * RY_TO_EV for f in ENERGY_FIELDS}
        result = CalculationResult(
            converged=cpp_result.converged,
            niter=cpp_result.niter,
            drho=cpp_result.drho,
            **energy_kwargs,
            fermi_energy=cpp_result.fermi_energy,
            bandgap=cpp_result.bandgap,
            nat=cpp_result.nat,
            ntype=cpp_result.ntype,
            nbands=cpp_result.nbands,
            nks=cpp_result.nks,
        )

        # Copy forces if available
        if cpp_result.has_forces:
            result.forces = np.array(cpp_result.forces)

        # Copy stress if available
        if cpp_result.has_stress:
            result.stress = np.array(cpp_result.stress)

        # Copy output tracking fields
        result.output_dir = cpp_result.output_dir
        result.log_file = cpp_result.log_file
        result.output_files = dict(cpp_result.output_files)
    else:
        # Use subprocess fallback
        result = run_abacus_subprocess(
            input_dir=str(input_dir),
            output_dir=output_dir,
            verbosity=verbosity,
            calculate_force=calculate_force,
            calculate_stress=calculate_stress,
            nprocs=nprocs,
            nthreads=nthreads,
        )

    return result


def run_scf(
    input_dir: str,
    **kwargs
) -> CalculationResult:
    """
    Convenience function for running SCF calculation.

    This is an alias for `abacus()` with default parameters
    suitable for single-point SCF calculations.

    Parameters
    ----------
    input_dir : str
        Directory containing input files
    **kwargs
        Additional arguments passed to `abacus()`

    Returns
    -------
    CalculationResult
        Calculation results
    """
    return abacus(input_dir, **kwargs)


def run_relax(
    input_dir: str,
    **kwargs
) -> CalculationResult:
    """
    Convenience function for running geometry optimization.

    This is an alias for `abacus()` with force calculation enabled.

    Parameters
    ----------
    input_dir : str
        Directory containing input files
    **kwargs
        Additional arguments passed to `abacus()`

    Returns
    -------
    CalculationResult
        Calculation results
    """
    kwargs.setdefault('calculate_force', True)
    return abacus(input_dir, **kwargs)
