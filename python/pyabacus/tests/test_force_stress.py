"""
Test: Force and stress comparison between Driver and ESolver modes.

Verifies that both modes produce consistent force and stress results.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest_abacus import (
    setup_calculation_directory,
    WorkingDirectory,
    MULTIK_TEST_CASE,
    RY_TO_EV,
)


@pytest.mark.integration
class TestForceComparison:
    """Compare force results between Driver and ESolver modes."""

    def test_force_consistency(self, tmp_path):
        """
        Verify force consistency between ESolver and Driver modes.

        Workflow:
        1. Run ESolver SCF + force calculation
        2. Run Driver SCF + force calculation
        3. Compare force arrays
        """
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        try:
            from pyabacus.esolver import LCAOWorkflow
            from pyabacus.driver import abacus
        except ImportError as e:
            pytest.skip(f"Module not available: {e}")

        # === ESolver calculation ===
        work_dir1 = tmp_path / "force_esolver"
        setup_calculation_directory(
            MULTIK_TEST_CASE, work_dir1,
            input_modifications={'init_chg': 'atomic'}
        )

        with WorkingDirectory(work_dir1):
            workflow = LCAOWorkflow("./", gamma_only=False)
            workflow.initialize()
            result = workflow.run_scf(max_iter=100)
            assert result.converged, "ESolver SCF did not converge"

            workflow.cal_force()
            esolver_forces = workflow.force.to_eV_Ang()
            workflow.cleanup()

        # === Driver calculation ===
        work_dir2 = tmp_path / "force_driver"
        setup_calculation_directory(
            MULTIK_TEST_CASE, work_dir2,
            input_modifications={'init_chg': 'atomic'}
        )

        with WorkingDirectory(work_dir2):
            driver_result = abacus(
                input_dir="./",
                calculate_force=True,
                calculate_stress=False,
                verbosity=0
            )

        assert driver_result.converged, "Driver SCF did not converge"
        assert driver_result.has_forces, "Driver should have forces"
        driver_forces = driver_result.forces

        # === Compare results ===
        print(f"\n=== Force Comparison ===")
        print(f"ESolver forces shape: {esolver_forces.shape}")
        print(f"Driver forces shape: {driver_forces.shape}")
        print(f"ESolver forces (eV/Ang):\n{esolver_forces}")
        print(f"Driver forces (eV/Ang):\n{driver_forces}")

        assert esolver_forces.shape == driver_forces.shape, \
            "Force array shapes should match"

        force_diff = np.abs(esolver_forces - driver_forces)
        max_diff = np.max(force_diff)
        print(f"Max force difference: {max_diff:.2e} eV/Ang")

        assert max_diff < 1e-4, \
            f"Force difference too large: {max_diff:.2e} eV/Ang"


@pytest.mark.integration
class TestStressComparison:
    """Compare stress results between Driver and ESolver modes."""

    def test_stress_consistency(self, tmp_path):
        """
        Verify stress consistency between ESolver and Driver modes.

        Workflow:
        1. Run ESolver SCF + stress calculation
        2. Run Driver SCF + stress calculation
        3. Compare stress tensors (Voigt notation)
        """
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        try:
            from pyabacus.esolver import LCAOWorkflow
            from pyabacus.driver import abacus
        except ImportError as e:
            pytest.skip(f"Module not available: {e}")

        # === ESolver calculation ===
        work_dir1 = tmp_path / "stress_esolver"
        setup_calculation_directory(
            MULTIK_TEST_CASE, work_dir1,
            input_modifications={'init_chg': 'atomic'}
        )

        with WorkingDirectory(work_dir1):
            workflow = LCAOWorkflow("./", gamma_only=False)
            workflow.initialize()
            result = workflow.run_scf(max_iter=100)
            assert result.converged, "ESolver SCF did not converge"

            workflow.cal_stress()
            esolver_stress = workflow.stress.to_voigt()  # kbar
            workflow.cleanup()

        # === Driver calculation ===
        work_dir2 = tmp_path / "stress_driver"
        setup_calculation_directory(
            MULTIK_TEST_CASE, work_dir2,
            input_modifications={'init_chg': 'atomic'}
        )

        with WorkingDirectory(work_dir2):
            driver_result = abacus(
                input_dir="./",
                calculate_force=False,
                calculate_stress=True,
                verbosity=0
            )

        assert driver_result.converged, "Driver SCF did not converge"
        assert driver_result.has_stress, "Driver should have stress"

        # Convert 3x3 matrix to Voigt notation
        s = driver_result.stress
        driver_stress = np.array([
            s[0, 0], s[1, 1], s[2, 2],
            s[1, 2], s[0, 2], s[0, 1]
        ])

        # === Compare results ===
        print(f"\n=== Stress Comparison (Voigt, kbar) ===")
        print(f"ESolver stress: {esolver_stress}")
        print(f"Driver stress:  {driver_stress}")

        stress_diff = np.abs(esolver_stress - driver_stress)
        max_diff = np.max(stress_diff)
        print(f"Max stress difference: {max_diff:.2e} kbar")

        assert max_diff < 0.1, \
            f"Stress difference too large: {max_diff:.2e} kbar"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
