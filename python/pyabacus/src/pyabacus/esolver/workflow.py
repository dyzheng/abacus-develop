"""
High-level workflow interfaces for ABACUS calculations.

This module provides:
- ``_BaseWorkflow``: shared logic for LCAO and PW workflows
- ``LCAOWorkflow``: Pythonic interface for controlling LCAO calculations
  with support for callbacks and breakpoints
"""

from typing import Callable, Optional
import os
import numpy as np

from .callbacks import CallbackMixin
from .data_access import DataAccessMixin
from .data_types import SCFResult


class _BaseWorkflow(CallbackMixin, DataAccessMixin):
    """Base class containing methods shared by LCAO and PW workflows."""

    def __init__(self, input_dir: str):
        self._input_dir = input_dir
        self._esolver = None
        self._initialized = False
        self._scf_running = False
        self._init_callbacks()

    def run_scf_step(self, iter_num: int) -> None:
        """
        Run a single SCF iteration.

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

    def update_positions(self, positions: np.ndarray) -> None:
        """
        Update atomic positions.

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

        After cleanup, the workflow can be reinitialized by calling
        initialize() again.
        """
        if self._esolver is not None:
            self._esolver.cleanup()
            self._esolver = None
        self._initialized = False
        self._scf_running = False
        self.clear_callbacks()


class LCAOWorkflow(_BaseWorkflow):
    """
    High-level workflow wrapper for LCAO calculations.

    Parameters
    ----------
    input_dir : str
        Directory containing INPUT, STRU, and other input files
    gamma_only : bool or None, optional
        Whether to use gamma-only calculation.  When ``None`` (the default),
        the value is read from the INPUT file in *input_dir* during
        ``initialize()``.

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

    def __init__(self, input_dir: str, gamma_only: Optional[bool] = None):
        super().__init__(input_dir)
        self._gamma_only = gamma_only

    @staticmethod
    def _read_gamma_only(input_dir: str) -> bool:
        """Read gamma_only from the INPUT file in *input_dir*."""
        from pyabacus.prepare import read_input
        inp = read_input(os.path.join(input_dir, "INPUT"))
        return bool(inp.get("gamma_only", 1))

    def initialize(self) -> None:
        """
        Initialize the calculation.

        This must be called before running any SCF calculations.
        If ``gamma_only`` was not specified at construction time, it is
        determined from the INPUT file in *input_dir*.
        """
        if self._gamma_only is None:
            self._gamma_only = self._read_gamma_only(self._input_dir)

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
        """
        if not self._initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")

        self._scf_running = True
        self._esolver.run_scf(max_iter)
        result = self._collect_result()
        self._scf_running = False
        return result
