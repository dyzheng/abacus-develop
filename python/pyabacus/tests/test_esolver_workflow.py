"""
Unit tests for PyABACUS ESolver workflow module.

This module contains tests for the LCAOWorkflow class and related ESolver functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import os
import sys

# Import data types directly to avoid C++ binding issues during testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pyabacus.esolver.data_types import (
    ForceData,
    StressData,
    EnergyData,
    ChargeData,
    SCFResult,
    HamiltonianData,
    DensityMatrixData,
    RY_TO_EV,
    BOHR_TO_ANG,
    RY_BOHR_TO_EV_ANG,
    KBAR_TO_EV_ANG3,
)


# ============================================================================
# Unit Tests for Data Types
# ============================================================================

class TestUnitConversionConstants:
    """Tests for unit conversion constants."""

    def test_ry_to_ev(self):
        """Test Rydberg to eV conversion constant."""
        # 1 Ry = 13.6057 eV (approximately)
        assert 13.60 < RY_TO_EV < 13.61

    def test_bohr_to_ang(self):
        """Test Bohr to Angstrom conversion constant."""
        # 1 Bohr = 0.529177 Angstrom
        assert 0.529 < BOHR_TO_ANG < 0.530

    def test_ry_bohr_to_ev_ang(self):
        """Test Ry/Bohr to eV/Ang conversion constant."""
        # Should be approximately RY_TO_EV / BOHR_TO_ANG
        expected = RY_TO_EV / BOHR_TO_ANG
        assert abs(RY_BOHR_TO_EV_ANG - expected) < 0.01


class TestSCFResult:
    """Tests for SCFResult dataclass."""

    def test_scf_result_creation(self):
        """Test SCFResult can be created."""
        energy = EnergyData(etot=-10.0, eband=-5.0)
        result = SCFResult(
            converged=True,
            niter=50,
            drho=1e-9,
            energy=energy,
        )

        assert result.converged is True
        assert result.niter == 50
        assert result.drho == pytest.approx(1e-9)
        assert result.energy.etot == -10.0

    def test_scf_result_summary(self):
        """Test SCFResult summary method."""
        energy = EnergyData(etot=-10.0, eband=-5.0)
        result = SCFResult(
            converged=True,
            niter=50,
            drho=1e-9,
            energy=energy,
        )

        summary = result.summary()
        assert "converged" in summary.lower()
        assert "50" in summary
        assert "drho" in summary.lower()


class TestHamiltonianData:
    """Tests for HamiltonianData dataclass."""

    def test_hamiltonian_data_creation(self):
        """Test HamiltonianData can be created."""
        Hk = [np.eye(10) * (-1.0) for _ in range(4)]
        Sk = [np.eye(10) for _ in range(4)]

        ham_data = HamiltonianData(
            Hk=Hk,
            Sk=Sk,
            HR=None,
            SR=None,
            nbasis=10,
            nks=4,
        )

        assert ham_data.nbasis == 10
        assert ham_data.nks == 4
        assert len(ham_data.Hk) == 4
        assert len(ham_data.Sk) == 4

    def test_hamiltonian_get_Hk(self):
        """Test HamiltonianData.get_Hk method."""
        Hk = [np.eye(10) * (i + 1) for i in range(4)]
        Sk = [np.eye(10) for _ in range(4)]

        ham_data = HamiltonianData(
            Hk=Hk,
            Sk=Sk,
            HR=None,
            SR=None,
            nbasis=10,
            nks=4,
        )

        H0 = ham_data.get_Hk(0)
        assert H0[0, 0] == pytest.approx(1.0)

        H2 = ham_data.get_Hk(2)
        assert H2[0, 0] == pytest.approx(3.0)


class TestDensityMatrixData:
    """Tests for DensityMatrixData dataclass."""

    def test_density_matrix_creation(self):
        """Test DensityMatrixData can be created."""
        DMK = [np.eye(10) * 0.5 for _ in range(4)]

        dm_data = DensityMatrixData(
            DMK=DMK,
            DMR=None,
            nks=4,
            nrow=10,
            ncol=10,
        )

        assert dm_data.nks == 4
        assert dm_data.nrow == 10
        assert dm_data.ncol == 10

    def test_density_matrix_trace(self):
        """Test DensityMatrixData.trace method."""
        DMK = [np.eye(10) * 0.5 for _ in range(4)]

        dm_data = DensityMatrixData(
            DMK=DMK,
            DMR=None,
            nks=4,
            nrow=10,
            ncol=10,
        )

        trace = dm_data.trace(0)
        assert trace == pytest.approx(5.0)  # 10 * 0.5


class TestChargeData:
    """Tests for ChargeData dataclass."""

    def test_charge_data_creation(self):
        """Test ChargeData can be created."""
        rho = np.ones((100,)) * 0.01

        charge_data = ChargeData(
            rho=rho,
            nspin=1,
            nrxx=100,
        )

        assert charge_data.nspin == 1
        assert charge_data.nrxx == 100
        assert charge_data.rho.shape == (100,)

    def test_charge_total_charge(self):
        """Test ChargeData.total_charge method."""
        rho = np.ones((100,)) * 0.01

        charge_data = ChargeData(
            rho=rho,
            nspin=1,
            nrxx=100,
        )

        total = charge_data.total_charge()
        assert total == pytest.approx(1.0)  # 100 * 0.01


# ============================================================================
# Mock-based Tests for LCAOWorkflow
# ============================================================================

class TestLCAOWorkflowWithMock:
    """Tests for LCAOWorkflow using mock ESolver."""

    @pytest.fixture
    def mock_esolver(self):
        """Create a mock ESolver."""
        esolver = Mock()

        # Mock basic properties
        esolver.nks = 4
        esolver.nbands = 10
        esolver.nbasis = 26
        esolver.nspin = 1
        esolver.nat = 2

        # Mock SCF methods
        esolver.is_converged.return_value = True
        esolver.niter = 50
        esolver.drho = 1e-9

        # Mock eigenvalues
        esolver.get_eigenvalues.return_value = np.linspace(-0.5, 0.5, 10)

        return esolver

    def test_workflow_import(self):
        """Test that LCAOWorkflow can be imported."""
        try:
            from pyabacus.esolver import LCAOWorkflow
            assert LCAOWorkflow is not None
        except ImportError as e:
            if "ESolver" in str(e) or "_esolver_pack" in str(e):
                pytest.skip("ESolver module not available")
            raise

    def test_workflow_data_types_import(self):
        """Test that data types can be imported."""
        from pyabacus.esolver.data_types import (
            SCFResult,
            EnergyData,
            ForceData,
            StressData,
        )

        assert SCFResult is not None
        assert EnergyData is not None
        assert ForceData is not None
        assert StressData is not None

    def test_callbacks_mixin(self):
        """Test CallbackMixin functionality."""
        from pyabacus.esolver.callbacks import CallbackMixin

        class TestClass(CallbackMixin):
            def __init__(self):
                self._init_callbacks()

        obj = TestClass()

        # Test callback registration
        callback_called = []

        def test_callback(workflow):
            callback_called.append(True)

        obj.register_callback('before_scf', test_callback)
        obj._fire_callbacks('before_scf')

        assert len(callback_called) == 1

    def test_callbacks_unregister(self):
        """Test callback unregistration."""
        from pyabacus.esolver.callbacks import CallbackMixin

        class TestClass(CallbackMixin):
            def __init__(self):
                self._init_callbacks()

        obj = TestClass()

        callback_called = []

        def test_callback(workflow):
            callback_called.append(True)

        obj.register_callback('before_scf', test_callback)
        obj.unregister_callback('before_scf', test_callback)
        obj._fire_callbacks('before_scf')

        assert len(callback_called) == 0

    def test_callbacks_clear(self):
        """Test clearing callbacks."""
        from pyabacus.esolver.callbacks import CallbackMixin

        class TestClass(CallbackMixin):
            def __init__(self):
                self._init_callbacks()

        obj = TestClass()

        def test_callback(workflow):
            pass

        obj.register_callback('before_scf', test_callback)
        obj.register_callback('after_scf', test_callback)
        obj.clear_callbacks()

        # All callbacks should be cleared
        assert len(obj._callbacks['before_scf']) == 0
        assert len(obj._callbacks['after_scf']) == 0


# ============================================================================
# Integration Tests (require ESolver module)
# ============================================================================

@pytest.mark.integration
class TestLCAOWorkflowIntegration:
    """Integration tests requiring ESolver module."""

    @pytest.fixture
    def si2_input_dir(self, tmp_path):
        """Create Si2 test input files."""
        # Create minimal INPUT file
        input_content = """INPUT_PARAMETERS
calculation scf
basis_type lcao
gamma_only 1
ecutwfc 20
scf_thr 1e-6
scf_nmax 100
pseudo_dir ../../PP_ORB
orbital_dir ../../PP_ORB
"""
        (tmp_path / "INPUT").write_text(input_content)

        # Create minimal STRU file
        stru_content = """ATOMIC_SPECIES
Si 28.085 Si_ONCV_PBE-1.0.upf upf201

NUMERICAL_ORBITAL
Si_gga_8au_60Ry_2s2p1d.orb

LATTICE_CONSTANT
10.2

LATTICE_VECTORS
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5

ATOMIC_POSITIONS
Direct

Si
0.0
2
0.00 0.00 0.00 1 1 1
0.25 0.25 0.25 1 1 1
"""
        (tmp_path / "STRU").write_text(stru_content)

        # Create KPT file
        kpt_content = """K_POINTS
0
Gamma
1 1 1 0 0 0
"""
        (tmp_path / "KPT").write_text(kpt_content)

        return tmp_path

    @pytest.mark.skip(reason="Requires full ABACUS installation with ESolver")
    def test_workflow_scf(self, si2_input_dir):
        """Test basic SCF workflow."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(str(si2_input_dir), gamma_only=True)
        workflow.initialize()

        result = workflow.run_scf(max_iter=100)

        assert result.niter > 0
        assert result.energy.etot < 0

        workflow.cleanup()

    @pytest.mark.skip(reason="Requires full ABACUS installation with ESolver")
    def test_workflow_with_callback(self, si2_input_dir):
        """Test workflow with callback."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(str(si2_input_dir), gamma_only=True)
        workflow.initialize()

        iterations = []

        def track_iterations(wf, iter_num):
            iterations.append(iter_num)

        workflow.register_callback('after_iter', track_iterations)
        result = workflow.run_scf(max_iter=100)

        assert len(iterations) > 0
        assert iterations[-1] == result.niter

        workflow.cleanup()

    @pytest.mark.skip(reason="Requires full ABACUS installation with ESolver")
    def test_workflow_eigenvalues(self, si2_input_dir):
        """Test eigenvalue access."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(str(si2_input_dir), gamma_only=True)
        workflow.initialize()

        result = workflow.run_scf(max_iter=100)

        eigenvalues = workflow.get_eigenvalues(0)
        assert eigenvalues is not None
        assert len(eigenvalues) == workflow.nbands

        workflow.cleanup()

    @pytest.mark.skip(reason="Requires full ABACUS installation with ESolver")
    def test_workflow_force_stress(self, si2_input_dir):
        """Test force and stress calculation."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(str(si2_input_dir), gamma_only=True)
        workflow.initialize()

        result = workflow.run_scf(max_iter=100)

        workflow.cal_force()
        workflow.cal_stress()

        forces = workflow.force.to_eV_Ang()
        stress = workflow.stress.to_voigt()

        assert forces.shape == (workflow.nat, 3)
        assert stress.shape == (6,)

        workflow.cleanup()


# ============================================================================
# Test for ESolver module availability
# ============================================================================

class TestESolverModuleAvailability:
    """Tests for ESolver module availability detection."""

    def test_esolver_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # This test verifies that the module handles missing ESolver gracefully
        try:
            from pyabacus.esolver import LCAOWorkflow
            # If import succeeds, ESolver is available
            assert LCAOWorkflow is not None
        except ImportError as e:
            # Import error is expected if ESolver is not built
            assert "_esolver_pack" in str(e) or "ESolver" in str(e)

    def test_data_types_always_available(self):
        """Test that data types are always available."""
        # Data types should be importable even without ESolver
        from pyabacus.esolver.data_types import (
            SCFResult,
            EnergyData,
            ForceData,
            StressData,
            ChargeData,
            HamiltonianData,
            DensityMatrixData,
        )

        assert SCFResult is not None
        assert EnergyData is not None
        assert ForceData is not None
        assert StressData is not None
        assert ChargeData is not None
        assert HamiltonianData is not None
        assert DensityMatrixData is not None

    def test_callbacks_always_available(self):
        """Test that callbacks module is always available."""
        from pyabacus.esolver.callbacks import CallbackMixin

        assert CallbackMixin is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
