"""
Test: Driver vs ESolver mode energy comparison.

Verifies that Driver mode and ESolver mode produce consistent results:
- Total energy (with Ry->eV conversion for ESolver)
- SCF convergence status
- Number of SCF iterations
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest_abacus import (
    setup_calculation_directory,
    verify_setup,
    WorkingDirectory,
    GAMMA_TEST_CASE,
    MULTIK_TEST_CASE,
    RY_TO_EV,
)


def run_esolver_calculation(work_dir: Path, gamma_only: bool):
    """
    Run ESolver mode calculation.

    Returns
    -------
    dict with keys: energy_ev, niter, converged
    """
    try:
        from pyabacus.esolver import LCAOWorkflow
    except ImportError as e:
        pytest.skip(f"ESolver module not available: {e}")

    with WorkingDirectory(work_dir):
        workflow = LCAOWorkflow("./", gamma_only=gamma_only)
        workflow.initialize()
        result = workflow.run_scf(max_iter=100)

        energy_ev = result.energy.etot * RY_TO_EV
        niter = result.niter
        converged = result.converged

        workflow.cleanup()

    return {
        'energy_ev': energy_ev,
        'niter': niter,
        'converged': converged,
    }


def run_driver_calculation(work_dir: Path):
    """
    Run Driver mode calculation.

    Returns
    -------
    dict with keys: energy_ev, niter, converged
    """
    try:
        from pyabacus.driver import abacus
    except ImportError as e:
        pytest.skip(f"Driver module not available: {e}")

    with WorkingDirectory(work_dir):
        result = abacus(
            input_dir="./",
            calculate_force=False,
            calculate_stress=False,
            verbosity=0
        )

    return {
        'energy_ev': result.etot,
        'niter': result.niter,
        'converged': result.converged,
    }


@pytest.mark.integration
class TestDriverVsESolverEnergy:
    """Compare energy results between Driver and ESolver modes."""

    def test_gamma_only_energy(self, tmp_path):
        """Verify energy consistency for gamma-only Si2 calculation."""
        # Step 1: Setup calculation directory
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")

        work_dir = tmp_path / "gamma_energy_test"
        success = setup_calculation_directory(GAMMA_TEST_CASE, work_dir)
        assert success, "Failed to setup calculation directory"
        assert verify_setup(work_dir), "Calculation directory verification failed"

        # Step 2: Run ESolver calculation
        esolver_result = run_esolver_calculation(work_dir, gamma_only=True)

        # Step 3: Verify ESolver convergence
        assert esolver_result['converged'], \
            f"ESolver SCF did not converge after {esolver_result['niter']} iterations"

        # Step 4: Run Driver calculation (fresh directory)
        work_dir2 = tmp_path / "gamma_energy_test_driver"
        setup_calculation_directory(GAMMA_TEST_CASE, work_dir2)
        driver_result = run_driver_calculation(work_dir2)

        # Step 5: Verify Driver convergence
        assert driver_result['converged'], \
            f"Driver SCF did not converge after {driver_result['niter']} iterations"

        # Step 6: Compare results
        energy_diff = abs(esolver_result['energy_ev'] - driver_result['energy_ev'])

        print(f"\n=== Gamma-only Energy Comparison ===")
        print(f"ESolver: E={esolver_result['energy_ev']:.8f} eV, "
              f"niter={esolver_result['niter']}")
        print(f"Driver:  E={driver_result['energy_ev']:.8f} eV, "
              f"niter={driver_result['niter']}")
        print(f"Energy difference: {energy_diff:.2e} eV")

        assert energy_diff < 1e-5, \
            f"Energy mismatch: {energy_diff:.2e} eV > 1e-5 eV"
        assert esolver_result['niter'] == driver_result['niter'], \
            f"Iteration mismatch: {esolver_result['niter']} vs {driver_result['niter']}"

    def test_multi_k_energy(self, tmp_path):
        """Verify energy consistency for multi-k Si2 calculation."""
        # Step 1: Setup calculation directory
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        # Use init_chg=atomic for fair comparison (not DM restart)
        work_dir = tmp_path / "multik_energy_test"
        success = setup_calculation_directory(
            MULTIK_TEST_CASE, work_dir,
            input_modifications={'init_chg': 'atomic'}
        )
        assert success, "Failed to setup calculation directory"

        # Step 2: Run ESolver calculation
        esolver_result = run_esolver_calculation(work_dir, gamma_only=False)
        assert esolver_result['converged'], "ESolver SCF did not converge"

        # Step 3: Run Driver calculation
        work_dir2 = tmp_path / "multik_energy_test_driver"
        setup_calculation_directory(
            MULTIK_TEST_CASE, work_dir2,
            input_modifications={'init_chg': 'atomic'}
        )
        driver_result = run_driver_calculation(work_dir2)
        assert driver_result['converged'], "Driver SCF did not converge"

        # Step 4: Compare results
        energy_diff = abs(esolver_result['energy_ev'] - driver_result['energy_ev'])

        print(f"\n=== Multi-k Energy Comparison ===")
        print(f"ESolver: E={esolver_result['energy_ev']:.8f} eV, "
              f"niter={esolver_result['niter']}")
        print(f"Driver:  E={driver_result['energy_ev']:.8f} eV, "
              f"niter={driver_result['niter']}")
        print(f"Energy difference: {energy_diff:.2e} eV")

        assert energy_diff < 1e-5, f"Energy mismatch: {energy_diff:.2e} eV"
        assert esolver_result['niter'] == driver_result['niter'], \
            f"Iteration mismatch"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
