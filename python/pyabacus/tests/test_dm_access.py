"""
Test: Density matrix and Hamiltonian data access in ESolver mode.

Verifies that ESolver mode provides correct access to:
- Density matrix (DMK) with proper structure
- Hamiltonian matrices (Hk, Sk)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest_abacus import (
    setup_calculation_directory,
    WorkingDirectory,
    GAMMA_TEST_CASE,
)


@pytest.mark.integration
class TestDensityMatrixAccess:
    """Test density matrix data access in ESolver mode."""

    def test_density_matrix_structure(self, tmp_path):
        """
        Verify density matrix has correct structure after SCF.

        Checks:
        - nks, nrow, ncol are positive
        - DMK list length matches nks
        - Each DMK matrix has correct shape
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")

        try:
            from pyabacus.esolver import LCAOWorkflow
        except ImportError as e:
            pytest.skip(f"ESolver module not available: {e}")

        # Step 1: Setup and run calculation
        work_dir = tmp_path / "dm_structure_test"
        success = setup_calculation_directory(GAMMA_TEST_CASE, work_dir)
        assert success, "Failed to setup calculation directory"

        with WorkingDirectory(work_dir):
            workflow = LCAOWorkflow("./", gamma_only=True)
            workflow.initialize()
            result = workflow.run_scf(max_iter=100)

            # Step 2: Verify convergence
            assert result.converged, "SCF did not converge"

            # Step 3: Access and verify density matrix
            dm_data = workflow.density_matrix

            print(f"\n=== Density Matrix Structure ===")
            print(f"nks (k-points): {dm_data.nks}")
            print(f"nrow: {dm_data.nrow}")
            print(f"ncol: {dm_data.ncol}")
            print(f"DMK list length: {len(dm_data.DMK)}")

            assert dm_data.nks > 0, "nks should be positive"
            assert dm_data.nrow > 0, "nrow should be positive"
            assert dm_data.ncol > 0, "ncol should be positive"
            assert len(dm_data.DMK) == dm_data.nks, \
                f"DMK length {len(dm_data.DMK)} != nks {dm_data.nks}"

            # Verify each DMK matrix shape
            for ik, dmk in enumerate(dm_data.DMK):
                expected_shape = (dm_data.nrow, dm_data.ncol)
                assert dmk.shape == expected_shape, \
                    f"DMK[{ik}] shape {dmk.shape} != {expected_shape}"

            workflow.cleanup()

    def test_density_matrix_trace(self, tmp_path):
        """
        Verify density matrix trace is physically reasonable.

        For Si2 with 8 valence electrons, trace should be ~8.
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")

        try:
            from pyabacus.esolver import LCAOWorkflow
        except ImportError as e:
            pytest.skip(f"ESolver module not available: {e}")

        work_dir = tmp_path / "dm_trace_test"
        setup_calculation_directory(GAMMA_TEST_CASE, work_dir)

        with WorkingDirectory(work_dir):
            workflow = LCAOWorkflow("./", gamma_only=True)
            workflow.initialize()
            result = workflow.run_scf(max_iter=100)
            assert result.converged, "SCF did not converge"

            dm_data = workflow.density_matrix

            print(f"\n=== Density Matrix Trace ===")
            for ik in range(dm_data.nks):
                trace = dm_data.trace(ik)
                print(f"DM trace[{ik}]: {trace}")

                # Trace should be positive (related to electron count)
                assert trace.real > 0, \
                    f"DM trace should be positive, got {trace}"

            workflow.cleanup()


@pytest.mark.integration
class TestHamiltonianAccess:
    """Test Hamiltonian matrix access in ESolver mode."""

    def test_hamiltonian_structure(self, tmp_path):
        """
        Verify Hamiltonian matrices have correct structure.

        Checks:
        - Hk and Sk lists have correct length
        - Matrix shapes match nbasis
        - S matrix diagonal is positive (overlap)
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")

        try:
            from pyabacus.esolver import LCAOWorkflow
        except ImportError as e:
            pytest.skip(f"ESolver module not available: {e}")

        work_dir = tmp_path / "ham_structure_test"
        setup_calculation_directory(GAMMA_TEST_CASE, work_dir)

        with WorkingDirectory(work_dir):
            workflow = LCAOWorkflow("./", gamma_only=True)
            workflow.initialize()
            result = workflow.run_scf(max_iter=100)
            assert result.converged, "SCF did not converge"

            ham_data = workflow.hamiltonian

            print(f"\n=== Hamiltonian Structure ===")
            print(f"nks: {ham_data.nks}")
            print(f"nbasis: {ham_data.nbasis}")
            print(f"Hk list length: {len(ham_data.Hk)}")
            print(f"Sk list length: {len(ham_data.Sk)}")

            assert ham_data.nks > 0, "nks should be positive"
            assert ham_data.nbasis > 0, "nbasis should be positive"
            assert len(ham_data.Hk) == ham_data.nks, "Hk length mismatch"
            assert len(ham_data.Sk) == ham_data.nks, "Sk length mismatch"

            # Verify matrix shapes and properties
            for ik in range(ham_data.nks):
                Hk = ham_data.get_Hk(ik)
                Sk = ham_data.get_Sk(ik)

                expected_shape = (ham_data.nbasis, ham_data.nbasis)
                assert Hk.shape == expected_shape, \
                    f"Hk[{ik}] shape {Hk.shape} != {expected_shape}"
                assert Sk.shape == expected_shape, \
                    f"Sk[{ik}] shape {Sk.shape} != {expected_shape}"

                # S matrix diagonal should be positive
                S_diag = np.diag(Sk).real
                assert np.all(S_diag > 0), \
                    f"S matrix diagonal should be positive"

            workflow.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
