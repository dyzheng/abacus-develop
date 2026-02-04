"""
High-level workflow interface for LCAO calculations.

This module provides a Pythonic interface for controlling LCAO calculations
with support for callbacks and breakpoints.
"""

from typing import Callable, Optional
import numpy as np

from .callbacks import CallbackMixin
from .data_access import DataAccessMixin
from .data_types import SCFResult


class LCAOWorkflow(CallbackMixin, DataAccessMixin):
    """
    High-level workflow wrapper for LCAO calculations.

    This class provides a Pythonic interface for controlling LCAO calculations
    with support for callbacks at various stages of the SCF loop.

    Parameters
    ----------
    input_dir : str
        Directory containing INPUT, STRU, and other input files
    gamma_only : bool, optional
        Whether to use gamma-only calculation (default: True)

    Example
    -------
    >>> workflow = LCAOWorkflow("./")
    >>> workflow.initialize()
    >>>
    >>> # Register callback for breakpoint before after_scf
    >>> def inspect_state(wf):
    ...     print(f"Energy: {wf.energy.etot}")
    ...     np.save("charge.npy", wf.charge.rho)
    >>>
    >>> workflow.register_callback('before_after_scf', inspect_state)
    >>> result = workflow.run_scf(max_iter=100)
    >>> print(result.summary())
    """

    def __init__(self, input_dir: str, gamma_only: bool = True):
        """
        Initialize LCAOWorkflow.

        Parameters
        ----------
        input_dir : str
            Directory containing input files
        gamma_only : bool
            Use gamma-only calculation if True, multi-k if False
        """
        self._input_dir = input_dir
        self._gamma_only = gamma_only
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
        # Import the appropriate ESolver class
        try:
            if self._gamma_only:
                from ._esolver_pack import ESolverLCAO_gamma
                self._esolver = ESolverLCAO_gamma()
            else:
                from ._esolver_pack import ESolverLCAO_multi_k
                self._esolver = ESolverLCAO_multi_k()
        except ImportError as e:
            raise ImportError(
                f"Could not import ESolver bindings: {e}. "
                "Make sure pyabacus is properly installed with ESolver support."
            ) from e

        self._esolver.initialize(self._input_dir)
        self._esolver.before_all_runners()
        self._initialized = True

    def run_scf(
        self,
        max_iter: int = 100,
        istep: int = 0,
        callback: Optional[Callable[['LCAOWorkflow', int], None]] = None
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

            # Check convergence
            if self._esolver.is_converged():
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

    def cleanup(self) -> None:
        """
        Release ESolver resources and reset state.

        This method should be called when the workflow is no longer needed
        to free up memory and resources. After cleanup, the workflow can
        be reinitialized by calling initialize() again.
        """
        if self._esolver is not None:
            self._esolver = None
        self._initialized = False
        self._scf_running = False
        self.clear_callbacks()
