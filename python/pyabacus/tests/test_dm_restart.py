"""
Test: Density matrix restart acceleration.

Verifies that using density matrix restart (init_chg=dm) reduces
SCF iterations compared to starting from atomic charge (init_chg=atomic).
"""

import pytest
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest_abacus import (
    setup_calculation_directory,
    verify_setup,
    WorkingDirectory,
    MULTIK_TEST_CASE,
)


def find_dmr_output_file(work_dir: Path) -> Path:
    """Find the output dmrs1_nao.csr file after calculation."""
    candidates = [
        work_dir / "OUT.autotest" / "dmrs1_nao.csr",
        work_dir / "OUT.PYABACUS" / "dmrs1_nao.csr",
        work_dir / "dmrs1_nao.csr",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@pytest.mark.integration
class TestDensityMatrixRestart:
    """Test density matrix restart acceleration."""

    def test_dm_restart_reduces_iterations(self, tmp_path):
        """
        Verify DM restart reduces SCF iterations.

        Workflow:
        1. Run calculation with init_chg=atomic, out_dmr=1
        2. Copy output dmrs1_nao.csr to new directory
        3. Run calculation with init_chg=dm
        4. Verify second run has fewer iterations
        """
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        try:
            from pyabacus.driver import abacus
        except ImportError as e:
            pytest.skip(f"Driver module not available: {e}")

        # === Run 1: Generate density matrix ===
        run1_dir = tmp_path / "run1_generate_dm"
        success = setup_calculation_directory(
            MULTIK_TEST_CASE, run1_dir,
            input_modifications={
                'init_chg': 'atomic',
                'out_dmr': '1',
            }
        )
        assert success, "Failed to setup run1 directory"

        with WorkingDirectory(run1_dir):
            result1 = abacus(input_dir="./", verbosity=0)

        assert result1.converged, \
            f"Run1 did not converge after {result1.niter} iterations"

        niter1 = result1.niter
        energy1 = result1.etot

        print(f"\n=== Run 1: Generate DM (init_chg=atomic) ===")
        print(f"Iterations: {niter1}")
        print(f"Energy: {energy1:.8f} eV")

        # Find output DM file
        dmr_file = find_dmr_output_file(run1_dir)
        if dmr_file is None:
            # Use pre-existing file from test case
            dmr_file = MULTIK_TEST_CASE / 'dmrs1_nao.csr'
            if not dmr_file.exists():
                pytest.skip("No dmrs1_nao.csr available for restart test")

        # === Run 2: Use density matrix restart ===
        run2_dir = tmp_path / "run2_dm_restart"
        success = setup_calculation_directory(
            MULTIK_TEST_CASE, run2_dir,
            input_modifications={
                'init_chg': 'dm',
                'read_file_dir': './',
            }
        )
        assert success, "Failed to setup run2 directory"

        # Copy DM file to run2 directory
        shutil.copy(dmr_file, run2_dir / 'dmrs1_nao.csr')

        with WorkingDirectory(run2_dir):
            result2 = abacus(input_dir="./", verbosity=0)

        assert result2.converged, \
            f"Run2 did not converge after {result2.niter} iterations"

        niter2 = result2.niter
        energy2 = result2.etot

        print(f"\n=== Run 2: DM Restart (init_chg=dm) ===")
        print(f"Iterations: {niter2}")
        print(f"Energy: {energy2:.8f} eV")

        # === Verify results ===
        print(f"\n=== Comparison ===")
        print(f"Iteration reduction: {niter1} -> {niter2}")
        print(f"Energy difference: {abs(energy1 - energy2):.2e} eV")

        assert niter2 < niter1, \
            f"DM restart should reduce iterations: {niter2} >= {niter1}"
        assert abs(energy1 - energy2) < 1e-5, \
            f"Energy should be consistent: |{energy1} - {energy2}| > 1e-5"

    def test_dm_restart_esolver_vs_driver(self, tmp_path):
        """
        Verify ESolver and Driver give same results with DM restart.

        Both modes should read the same dmrs1_nao.csr and produce
        identical energy and iteration count.
        """
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        dmr_source = MULTIK_TEST_CASE / 'dmrs1_nao.csr'
        if not dmr_source.exists():
            pytest.skip("dmrs1_nao.csr not found for restart test")

        try:
            from pyabacus.esolver import LCAOWorkflow
            from pyabacus.driver import abacus
        except ImportError as e:
            pytest.skip(f"Module not available: {e}")

        from conftest_abacus import RY_TO_EV

        # Setup directory with DM restart
        work_dir = tmp_path / "dm_restart_comparison"
        setup_calculation_directory(MULTIK_TEST_CASE, work_dir)
        shutil.copy(dmr_source, work_dir / 'dmrs1_nao.csr')

        # Run ESolver
        with WorkingDirectory(work_dir):
            workflow = LCAOWorkflow("./", gamma_only=False)
            workflow.initialize()
            es_result = workflow.run_scf(max_iter=100)
            es_energy = es_result.energy.etot * RY_TO_EV
            es_niter = es_result.niter
            workflow.cleanup()

        # Run Driver (fresh directory)
        work_dir2 = tmp_path / "dm_restart_comparison_driver"
        setup_calculation_directory(MULTIK_TEST_CASE, work_dir2)
        shutil.copy(dmr_source, work_dir2 / 'dmrs1_nao.csr')

        with WorkingDirectory(work_dir2):
            dr_result = abacus(input_dir="./", verbosity=0)
            dr_energy = dr_result.etot
            dr_niter = dr_result.niter

        print(f"\n=== DM Restart: ESolver vs Driver ===")
        print(f"ESolver: E={es_energy:.8f} eV, niter={es_niter}")
        print(f"Driver:  E={dr_energy:.8f} eV, niter={dr_niter}")

        assert abs(es_energy - dr_energy) < 1e-5, "Energy mismatch"
        assert es_niter == dr_niter, "Iteration count mismatch"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
