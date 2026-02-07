"""
High-level workflow interface for plane wave (PW) calculations.

This module provides a Pythonic interface for controlling PW calculations
with support for callbacks and breakpoints.
"""

from typing import Callable, Optional
from pathlib import Path
import numpy as np

from .callbacks import CallbackMixin
from .data_types import SCFResult, EnergyData, ChargeData, ForceData, StressData


class PWWorkflow(CallbackMixin):
    """
    High-level workflow wrapper for plane wave calculations.

    This class provides a Pythonic interface for controlling PW calculations
    with support for callbacks at various stages of the SCF loop.

    Parameters
    ----------
    input_dir : str
        Directory containing INPUT, STRU, and other input files
    precision : str, optional
        Precision for calculations: "single" or "double" (default: "double")

    Example
    -------
    >>> workflow = PWWorkflow("./")
    >>> workflow.initialize()
    >>>
    >>> # Register callback for breakpoint before after_scf
    >>> def inspect_state(wf):
    ...     print(f"Energy: {wf.energy.etot}")
    >>>
    >>> workflow.register_callback('before_after_scf', inspect_state)
    >>> result = workflow.run_scf(max_iter=100)
    >>> print(result.summary())
    """

    def __init__(self, input_dir: str, precision: str = "double"):
        """
        Initialize PWWorkflow.

        Parameters
        ----------
        input_dir : str
            Directory containing input files
        precision : str
            Precision: "single" (complex<float>) or "double" (complex<double>)
        """
        self._input_dir = str(Path(input_dir).resolve())
        self._precision = precision
        self._esolver = None
        self._initialized = False
        self._scf_running = False

        # Initialize callback registry from mixin
        self._init_callbacks()

    def initialize(self) -> None:
        """
        Initialize the calculation.

        This must be called before running any SCF calculations.
        """
        # Import the appropriate ESolver class based on precision
        try:
            if self._precision == "single":
                from ._esolver_pack import ESolverPW_cf
                self._esolver = ESolverPW_cf()
            else:
                from ._esolver_pack import ESolverPW_cd
                self._esolver = ESolverPW_cd()
        except ImportError as e:
            raise ImportError(
                f"Could not import ESolver PW bindings: {e}. "
                "Make sure pyabacus is properly installed with ESolver support."
            ) from e

        self._esolver.initialize(self._input_dir)
        self._esolver.before_all_runners()
        self._initialized = True

    def run_scf(
        self,
        max_iter: int = 100,
        istep: int = 0,
        callback: Optional[Callable[['PWWorkflow', int], None]] = None
    ) -> SCFResult:
        """
        Run SCF calculation with callback support.

        Parameters
        ----------
        max_iter : int
            Maximum number of SCF iterations
        istep : int
            Ion step index (for MD/relaxation)
        callback : Callable, optional
            Additional callback called after each iteration.
            Receives (workflow, iter_num) as arguments.

        Returns
        -------
        SCFResult
            Result of the SCF calculation

        Raises
        ------
        RuntimeError
            If workflow is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")

        self._scf_running = True

        # before_scf
        self._esolver.before_scf(istep)
        self._fire_callbacks('before_scf')

        # SCF loop
        for iter_num in range(1, max_iter + 1):
            self._esolver.run_scf_iteration(iter_num)

            # Fire after_iter callbacks
            self._fire_callbacks('after_iter', iter_num)

            # Call user-provided callback
            if callback is not None:
                callback(self, iter_num)

            # Check convergence or oscillation (matching ESolver_KS::runner())
            if self._esolver.is_converged():
                break
            if hasattr(self._esolver, 'is_oscillating') and self._esolver.is_oscillating():
                break

        # Breakpoint before after_scf - this is the main inspection point
        self._fire_callbacks('before_after_scf')

        # Collect result before after_scf
        result = self._collect_result()

        # after_scf
        self._esolver.after_scf(istep)
        self._fire_callbacks('after_scf')

        self._scf_running = False

        return result

    def run_scf_step(self, iter_num: int) -> None:
        """
        Run a single SCF iteration.

        This is useful for manual control of the SCF loop.

        Parameters
        ----------
        iter_num : int
            Iteration number (1-based)
        """
        if not self._scf_running:
            raise RuntimeError(
                "SCF not started. Call before_scf() first or use run_scf()."
            )
        self._esolver.run_scf_iteration(iter_num)

    def before_scf(self, istep: int = 0) -> None:
        """
        Prepare for SCF calculation.

        Call this before manually running SCF iterations.

        Parameters
        ----------
        istep : int
            Ion step index
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        self._esolver.before_scf(istep)
        self._scf_running = True
        self._fire_callbacks('before_scf')

    def after_scf(self, istep: int = 0) -> None:
        """
        Finalize SCF calculation.

        Call this after manually running SCF iterations.

        Parameters
        ----------
        istep : int
            Ion step index
        """
        self._fire_callbacks('before_after_scf')
        self._esolver.after_scf(istep)
        self._fire_callbacks('after_scf')
        self._scf_running = False

    def _collect_result(self) -> SCFResult:
        """Collect SCF result from current state."""
        return SCFResult(
            converged=self._esolver.is_converged(),
            niter=self._esolver.niter,
            drho=self._esolver.drho,
            energy=self.energy,
            charge=self.charge,
        )

    # ==================== Data Accessors ====================

    @property
    def energy(self) -> EnergyData:
        """
        Get energy data.

        Returns
        -------
        EnergyData
            Energy components from the calculation
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        accessor = self._esolver.get_energy()
        return EnergyData(
            etot=accessor.etot,
            eband=accessor.eband,
            hartree_energy=accessor.hartree_energy,
            etxc=accessor.etxc,
            ewald_energy=accessor.ewald_energy,
            demet=accessor.demet,
            exx=accessor.exx,
            evdw=accessor.evdw,
        )

    @property
    def charge(self) -> ChargeData:
        """
        Get charge density data.

        Returns
        -------
        ChargeData
            Charge density from the calculation
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        accessor = self._esolver.get_charge()
        return ChargeData(
            rho=accessor.get_rho(),
            rhog=accessor.get_rhog(),
            rho_core=accessor.get_rho_core(),
            nspin=accessor.nspin,
            nrxx=accessor.nrxx,
            ngmc=accessor.ngmc,
        )

    @property
    def force(self) -> ForceData:
        """
        Get force data.

        Returns
        -------
        ForceData
            Forces on atoms (call cal_force() first)
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        accessor = self._esolver.get_force()
        return ForceData(
            forces=accessor.get_forces(),
            nat=accessor.nat,
        )

    @property
    def stress(self) -> StressData:
        """
        Get stress tensor data.

        Returns
        -------
        StressData
            Stress tensor (call cal_stress() first)
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        accessor = self._esolver.get_stress()
        return StressData(
            stress=accessor.get_stress(),
        )

    # ==================== Force and Stress ====================

    def cal_force(self) -> None:
        """
        Calculate forces on atoms.

        Must be called after SCF convergence before accessing force property.
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        self._esolver.cal_force()

    def cal_stress(self) -> None:
        """
        Calculate stress tensor.

        Must be called after SCF convergence before accessing stress property.
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        self._esolver.cal_stress()

    # ==================== Position and Cell Update ====================

    def update_positions(self, positions: np.ndarray) -> None:
        """
        Update atomic positions.

        After updating positions, you must call before_scf() and run_scf()
        to recalculate the electronic structure.

        Parameters
        ----------
        positions : np.ndarray
            Atomic positions with shape (nat, 3) in Angstrom (Cartesian)
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        self._esolver.update_positions(positions)

    def update_cell(self, cell: np.ndarray) -> None:
        """
        Update cell vectors.

        After updating the cell, you must call before_scf() and run_scf()
        to recalculate the electronic structure.

        Parameters
        ----------
        cell : np.ndarray
            Cell vectors with shape (3, 3) in Angstrom
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        self._esolver.update_cell(cell)

    def get_positions(self) -> np.ndarray:
        """
        Get atomic positions.

        Returns
        -------
        np.ndarray
            Atomic positions with shape (nat, 3) in Angstrom (Cartesian)
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.get_positions()

    def get_cell(self) -> np.ndarray:
        """
        Get cell vectors.

        Returns
        -------
        np.ndarray
            Cell vectors with shape (3, 3) in Angstrom
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.get_cell()

    # ==================== Wave Function Access ====================

    def get_psi(self, ik: int) -> np.ndarray:
        """
        Get wave function coefficients for k-point ik.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        np.ndarray
            Wave function coefficients with shape (nbands, npw)
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.get_psi(ik)

    def get_eigenvalues(self, ik: int) -> np.ndarray:
        """
        Get eigenvalues for k-point ik.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        np.ndarray
            Eigenvalues with shape (nbands,)
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.get_eigenvalues(ik)

    def get_occupations(self, ik: int) -> np.ndarray:
        """
        Get occupation numbers for k-point ik.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        np.ndarray
            Occupation numbers with shape (nbands,)
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.get_occupations(ik)

    # ==================== System Information ====================

    @property
    def nks(self) -> int:
        """Number of k-points."""
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.nks

    @property
    def nbands(self) -> int:
        """Number of bands."""
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.nbands

    @property
    def nspin(self) -> int:
        """Number of spin channels."""
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.nspin

    @property
    def nat(self) -> int:
        """Number of atoms."""
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.nat

    @property
    def npwx(self) -> int:
        """Maximum number of plane waves."""
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.npwx

    def get_npw(self, ik: int) -> int:
        """
        Get number of plane waves for k-point ik.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        int
            Number of plane waves
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self._esolver.get_npw(ik)

    def cleanup(self) -> None:
        """
        Release ESolver resources and reset state.

        This method should be called when the workflow is no longer needed
        to free up memory and resources. After cleanup, the workflow can
        be reinitialized by calling initialize() again.
        """
        if self._esolver is not None:
            self._esolver.cleanup()
            self._esolver = None
        self._initialized = False
        self._scf_running = False
        self.clear_callbacks()
