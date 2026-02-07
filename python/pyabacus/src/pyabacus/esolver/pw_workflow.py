"""
High-level workflow interface for plane wave (PW) calculations.

This module provides a Pythonic interface for controlling PW calculations
with support for callbacks and breakpoints.
"""

from typing import Callable, Optional
from pathlib import Path
import numpy as np

from .workflow import _BaseWorkflow
from .data_types import SCFResult


class PWWorkflow(_BaseWorkflow):
    """
    High-level workflow wrapper for plane wave calculations.

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
        super().__init__(str(Path(input_dir).resolve()))
        self._precision = precision

    def initialize(self) -> None:
        """
        Initialize the calculation.

        This must be called before running any SCF calculations.
        """
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

            self._fire_callbacks('after_iter', iter_num)

            if callback is not None:
                callback(self, iter_num)

            if self._esolver.is_converged():
                break
            if hasattr(self._esolver, 'is_oscillating') and self._esolver.is_oscillating():
                break

        # Breakpoint before after_scf
        self._fire_callbacks('before_after_scf')

        result = self._collect_result()

        # after_scf
        self._esolver.after_scf(istep)
        self._fire_callbacks('after_scf')

        self._scf_running = False

        return result

    # ==================== PW-specific properties ====================

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
