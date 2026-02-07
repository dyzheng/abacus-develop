"""
ASE Calculator implementation using pyabacus ESolver.

This module provides an ASE-compatible Calculator that uses pyabacus ESolver
directly, enabling efficient geometry optimization with ASE optimizers (BFGS,
FIRE, etc.) using ABACUS as the DFT engine.

Unlike ase-abacus (subprocess-based), this calculator uses pyabacus directly
for lower overhead and better integration.

Two calculation modes are supported:
1. Driver Mode: Uses abacus() function for complete, independent calculations
2. ESolver Mode: Uses LCAOWorkflow directly with memory persistence
"""

from enum import Enum, auto
from typing import List, Optional, Any, Union
import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase import Atoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Calculator = object
    all_changes = ['positions', 'numbers', 'cell', 'pbc']
    Atoms = None

from ..esolver import LCAOWorkflow
from ..esolver.data_types import StressData
from ..constants import RY_TO_EV, BOHR_TO_ANG, RY_BOHR_TO_EV_ANG, KBAR_TO_EV_ANG3, ENERGY_FIELDS


class CalculatorMode(Enum):
    """
    Calculator execution mode.

    Attributes
    ----------
    DRIVER : auto
        Full driver mode using abacus() function. Each calculation is
        independent and uses file-based I/O. No memory persistence between
        calculations. Suitable for single-point calculations or when
        isolation between calculations is needed.
    ESOLVER : auto
        Direct ESolver mode using LCAOWorkflow. Memory is persistent between
        calculations, allowing efficient sequential calculations (e.g.,
        geometry optimization). Requires explicit cleanup() call or use
        as context manager to release resources.
    """
    DRIVER = auto()
    ESOLVER = auto()


class AbacusCalculator(Calculator):
    """
    ASE Calculator interface for ABACUS via pyabacus.

    This calculator supports two modes of operation:

    1. **ESolver Mode** (default): Uses LCAOWorkflow directly with memory
       persistence for efficient sequential calculations (e.g., geometry
       optimization). Requires explicit cleanup() or use as context manager.

    2. **Driver Mode**: Uses abacus() function for complete, independent
       calculations via file I/O. No memory persistence between calculations.

    Parameters
    ----------
    input_dir : str
        Directory containing INPUT, STRU, KPT files
    gamma_only : bool
        Use gamma-only calculation (default: True). Only used in ESolver mode.
    mode : CalculatorMode or str
        Calculation mode: CalculatorMode.ESOLVER (default) or CalculatorMode.DRIVER.
        Can also be specified as string 'esolver' or 'driver' (case-insensitive).
    nprocs : int
        Number of MPI processes (default: 1). Only used in Driver mode.
    nthreads : int
        Number of OpenMP threads (default: 1). Only used in Driver mode.
    restart : str, optional
        Path to restart file (not currently used)
    label : str
        Calculator label (default: 'abacus')
    atoms : Atoms, optional
        ASE Atoms object to attach

    Example
    -------
    ESolver mode (default, memory persistent):

    >>> from ase.optimize import BFGS
    >>> from pyabacus.ase import AbacusCalculator
    >>>
    >>> with AbacusCalculator(input_dir='./Si_scf/') as calc:
    ...     atoms.calc = calc
    ...     opt = BFGS(atoms)
    ...     opt.run(fmax=0.01)

    Driver mode (independent calculations):

    >>> calc = AbacusCalculator(input_dir='./Si_scf/', mode='driver')
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()

    Notes
    -----
    The calculator automatically handles unit conversions:
    - Energy: Rydberg -> eV (ESolver mode) or already in eV (Driver mode)
    - Forces: Ry/Bohr -> eV/Å (ESolver mode) or already in eV/Å (Driver mode)
    - Stress: kbar -> eV/Å³ (Voigt notation)
    """

    # Properties this calculator can compute
    implemented_properties = ['energy', 'forces', 'stress']

    # Default parameters
    default_parameters = {
        'input_dir': '.',
        'gamma_only': True,
        'mode': CalculatorMode.ESOLVER,
    }

    def __init__(
        self,
        input_dir: str = '.',
        gamma_only: bool = True,
        mode: Union[CalculatorMode, str] = CalculatorMode.ESOLVER,
        nprocs: int = 1,
        nthreads: int = 1,
        restart: Optional[str] = None,
        label: str = 'abacus',
        atoms: Optional[Any] = None,
        **kwargs
    ):
        """Initialize AbacusCalculator."""
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE is required for AbacusCalculator. "
                "Install it with: pip install ase"
            )

        # Initialize base Calculator
        super().__init__(restart=restart, label=label, atoms=atoms, **kwargs)

        # Parse mode if string
        if isinstance(mode, str):
            try:
                mode = CalculatorMode[mode.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid mode '{mode}'. Must be 'driver' or 'esolver'."
                )

        # Store parameters
        self.input_dir = input_dir
        self.gamma_only = gamma_only
        self.mode = mode
        self.nprocs = nprocs
        self.nthreads = nthreads

        # Internal state
        self._workflow: Optional[LCAOWorkflow] = None
        self._initialized = False
        self._scf_converged = False

    def __enter__(self) -> 'AbacusCalculator':
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager, cleanup resources."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self) -> None:
        """
        Release resources (ESolver mode only).

        This method releases the LCAOWorkflow and associated memory.
        After cleanup, the calculator can be reinitialized by running
        another calculation.

        In Driver mode, this is a no-op since each calculation is independent.
        """
        if self._workflow is not None:
            self._workflow.cleanup()
            self._workflow = None
        self._initialized = False
        self._scf_converged = False

    def _ensure_initialized(self) -> None:
        """Initialize the workflow if not already done (ESolver mode only)."""
        if not self._initialized:
            self._workflow = LCAOWorkflow(self.input_dir, self.gamma_only)
            self._workflow.initialize()
            self._initialized = True

    def _sync_positions(self) -> None:
        """Synchronize ASE Atoms positions to ABACUS (ESolver mode only)."""
        if self.atoms is None:
            return

        # Update positions
        positions = self.atoms.get_positions()
        self._workflow.update_positions(positions)

        # Update cell if periodic
        if any(self.atoms.pbc):
            cell = self.atoms.get_cell()[:]
            self._workflow.update_cell(np.array(cell))

    def calculate(
        self,
        atoms: Optional[Any] = None,
        properties: List[str] = ['energy'],
        system_changes: List[str] = all_changes
    ) -> None:
        """
        Calculate requested properties.

        This is the main method called by ASE to compute energy, forces,
        and stress. It dispatches to the appropriate mode-specific method.

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object. If provided, updates self.atoms.
        properties : list of str
            Properties to calculate. Can include 'energy', 'forces', 'stress'.
        system_changes : list of str
            List of changes since last calculation.
        """
        # Call parent's calculate to handle atoms assignment
        super().calculate(atoms, properties, system_changes)

        if self.mode == CalculatorMode.DRIVER:
            self._calculate_driver(properties)
        else:
            self._calculate_esolver(properties, system_changes)

    def _calculate_driver(self, properties: List[str]) -> None:
        """
        Driver mode calculation using abacus() function.

        Each calculation is independent and uses file-based I/O.
        Results are already in ASE units (eV, eV/Å).
        """
        from ..driver import abacus

        result = abacus(
            input_dir=self.input_dir,
            calculate_force='forces' in properties,
            calculate_stress='stress' in properties,
            nprocs=self.nprocs,
            nthreads=self.nthreads,
        )

        # Energy from driver is already in eV
        self.results['energy'] = result.etot

        # Forces from driver are already in eV/Å
        if result.forces is not None:
            self.results['forces'] = result.forces

        # Stress from driver is in kbar, convert to eV/Å³ Voigt notation
        if result.stress is not None:
            stress_data = StressData(stress=result.stress)
            self.results['stress'] = stress_data.to_eV_Ang3()

    def _calculate_esolver(
        self,
        properties: List[str],
        system_changes: List[str]
    ) -> None:
        """
        ESolver mode calculation using LCAOWorkflow.

        Memory is persistent between calculations for efficiency.
        Results need unit conversion from ABACUS units.
        """
        # Initialize workflow if needed
        self._ensure_initialized()

        # Sync positions if they changed
        if 'positions' in system_changes or 'cell' in system_changes:
            self._sync_positions()
            self._scf_converged = False

        # Run SCF if not converged
        # Note: run_scf() internally calls before_scf() and after_scf()
        if not self._scf_converged:
            result = self._workflow.run_scf()
            self._scf_converged = result.converged

        # Get energy (Ry -> eV)
        energy_data = self._workflow.energy
        self.results['energy'] = energy_data.etot * RY_TO_EV

        # Get forces if requested (Ry/Bohr -> eV/Å)
        if 'forces' in properties:
            self._workflow.cal_force()
            force_data = self._workflow.force
            self.results['forces'] = force_data.to_eV_Ang()

        # Get stress if requested (kbar -> eV/Å³, Voigt notation)
        if 'stress' in properties:
            self._workflow.cal_stress()
            stress_data = self._workflow.stress
            self.results['stress'] = stress_data.to_eV_Ang3()

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self._scf_converged = False

    def get_potential_energy(
        self,
        atoms: Optional[Any] = None,
        force_consistent: bool = False
    ) -> float:
        """
        Get potential energy.

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object
        force_consistent : bool
            If True, return force-consistent energy (not implemented)

        Returns
        -------
        float
            Potential energy in eV
        """
        return self.get_property('energy', atoms)

    def get_forces(self, atoms: Optional[Any] = None) -> np.ndarray:
        """
        Get forces on atoms.

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object

        Returns
        -------
        np.ndarray
            Forces with shape (nat, 3) in eV/Å
        """
        return self.get_property('forces', atoms)

    def get_stress(self, atoms: Optional[Any] = None) -> np.ndarray:
        """
        Get stress tensor in Voigt notation.

        Parameters
        ----------
        atoms : Atoms, optional
            ASE Atoms object

        Returns
        -------
        np.ndarray
            Stress in Voigt notation (6,): xx, yy, zz, yz, xz, xy in eV/Å³
        """
        return self.get_property('stress', atoms)

    @property
    def workflow(self) -> Optional[LCAOWorkflow]:
        """
        Get the underlying LCAOWorkflow instance.

        Returns None in Driver mode since no workflow is used.
        """
        return self._workflow

    def get_energy_components(self) -> dict:
        """
        Get detailed energy components.

        Only available in ESolver mode after a calculation.

        Returns
        -------
        dict
            Dictionary with energy components in eV:
            - etot: Total energy
            - eband: Band energy
            - hartree_energy: Hartree energy
            - etxc: Exchange-correlation energy
            - ewald_energy: Ewald energy
            - demet: -TS term for metals
            - exx: Exact exchange energy
            - evdw: van der Waals energy

        Raises
        ------
        RuntimeError
            If calculator not initialized or in Driver mode
        """
        if self.mode == CalculatorMode.DRIVER:
            raise RuntimeError(
                "get_energy_components() is only available in ESolver mode."
            )
        if not self._initialized:
            raise RuntimeError("Calculator not initialized. Run a calculation first.")

        energy_data = self._workflow.energy
        return {f: getattr(energy_data, f) * RY_TO_EV for f in ENERGY_FIELDS}
