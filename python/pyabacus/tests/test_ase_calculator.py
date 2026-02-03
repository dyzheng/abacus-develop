"""
Tests for ASE Calculator integration with pyabacus.

This module contains unit tests for the AbacusCalculator class using mock
ESolver objects, as well as integration test markers for tests that require
a full ABACUS installation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import warnings

# Import data types directly to avoid C++ binding issues during testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pyabacus.esolver.data_types import (
    ForceData,
    StressData,
    EnergyData,
    ChargeData,
    SCFResult,
)


# ============================================================================
# Unit Tests for Data Types
# ============================================================================

class TestForceData:
    """Tests for ForceData class."""

    def test_force_data_creation(self):
        """Test ForceData can be created with forces array."""
        forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        force_data = ForceData(forces=forces, nat=2)

        assert force_data.nat == 2
        assert force_data.forces.shape == (2, 3)
        np.testing.assert_array_equal(force_data.forces, forces)

    def test_force_conversion_to_eV_Ang(self):
        """Test force unit conversion from Ry/Bohr to eV/Ang."""
        # 1 Ry/Bohr = 25.7112 eV/Ang (approximately)
        forces_ry_bohr = np.array([[1.0, 0.0, 0.0]])
        force_data = ForceData(forces=forces_ry_bohr, nat=1)

        forces_eV_Ang = force_data.to_eV_Ang()

        # Check conversion factor
        expected = forces_ry_bohr * ForceData.RY_BOHR_TO_EV_ANG
        np.testing.assert_allclose(forces_eV_Ang, expected, rtol=1e-6)

        # Verify the conversion factor is approximately 25.7112
        assert 25.7 < ForceData.RY_BOHR_TO_EV_ANG < 25.8

    def test_force_to_dict(self):
        """Test ForceData.to_dict() method."""
        forces = np.array([[0.1, 0.2, 0.3]])
        force_data = ForceData(forces=forces, nat=1)

        result = force_data.to_dict()

        assert 'forces' in result
        assert 'forces_eV_Ang' in result
        assert 'nat' in result
        assert result['nat'] == 1


class TestStressData:
    """Tests for StressData class."""

    def test_stress_data_creation(self):
        """Test StressData can be created with stress tensor."""
        stress = np.array([
            [10.0, 1.0, 2.0],
            [1.0, 20.0, 3.0],
            [2.0, 3.0, 30.0]
        ])
        stress_data = StressData(stress=stress)

        assert stress_data.stress.shape == (3, 3)
        np.testing.assert_array_equal(stress_data.stress, stress)

    def test_stress_to_voigt(self):
        """Test stress conversion to Voigt notation."""
        stress = np.array([
            [10.0, 1.0, 2.0],   # xx, xy, xz
            [1.0, 20.0, 3.0],   # yx, yy, yz
            [2.0, 3.0, 30.0]    # zx, zy, zz
        ])
        stress_data = StressData(stress=stress)

        voigt = stress_data.to_voigt()

        # Voigt notation: xx, yy, zz, yz, xz, xy
        expected = np.array([10.0, 20.0, 30.0, 3.0, 2.0, 1.0])
        np.testing.assert_array_equal(voigt, expected)

    def test_stress_conversion_to_eV_Ang3(self):
        """Test stress unit conversion from kbar to eV/Ang^3."""
        stress = np.array([
            [1602.1766208, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        stress_data = StressData(stress=stress)

        stress_eV_Ang3 = stress_data.to_eV_Ang3()

        # 1602.1766208 kbar = 1 eV/Ang^3
        np.testing.assert_allclose(stress_eV_Ang3[0], 1.0, rtol=1e-6)

    def test_stress_to_dict(self):
        """Test StressData.to_dict() method."""
        stress = np.eye(3) * 10.0
        stress_data = StressData(stress=stress)

        result = stress_data.to_dict()

        assert 'stress' in result
        assert 'stress_voigt' in result
        assert 'stress_eV_Ang3' in result


class TestEnergyData:
    """Tests for EnergyData class."""

    def test_energy_to_eV(self):
        """Test energy conversion from Ry to eV."""
        energy = EnergyData(etot=-10.0, eband=-5.0)

        energy_eV = energy.to_eV()

        # 1 Ry = 13.6057 eV
        assert energy_eV.etot == pytest.approx(-10.0 * 13.605693122994)
        assert energy_eV.eband == pytest.approx(-5.0 * 13.605693122994)


# ============================================================================
# Mock-based Tests for ASE Calculator
# ============================================================================

class TestAbacusCalculatorWithMock:
    """Tests for AbacusCalculator using mock ESolver."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock LCAOWorkflow."""
        workflow = Mock()

        # Mock energy accessor
        energy_accessor = Mock()
        energy_accessor.etot = -10.0  # Ry
        energy_accessor.eband = -5.0
        energy_accessor.hartree_energy = 2.0
        energy_accessor.etxc = -3.0
        energy_accessor.ewald_energy = 1.0
        energy_accessor.demet = 0.0
        energy_accessor.exx = 0.0
        energy_accessor.evdw = 0.0
        workflow.energy = EnergyData(
            etot=-10.0, eband=-5.0, hartree_energy=2.0,
            etxc=-3.0, ewald_energy=1.0
        )

        # Mock force accessor
        force_accessor = Mock()
        force_accessor.get_forces.return_value = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        force_accessor.nat = 2
        workflow.force = ForceData(
            forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
            nat=2
        )

        # Mock stress accessor
        stress_accessor = Mock()
        stress_accessor.get_stress.return_value = np.eye(3) * 10.0
        workflow.stress = StressData(stress=np.eye(3) * 10.0)

        # Mock SCF result
        workflow.run_scf.return_value = SCFResult(
            converged=True, niter=10, drho=1e-8,
            energy=workflow.energy
        )

        return workflow

    @pytest.fixture
    def mock_atoms(self):
        """Create mock ASE Atoms object."""
        try:
            from ase import Atoms
            return Atoms('Si2', positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                        cell=[5.43, 5.43, 5.43], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

    def test_calculator_import(self):
        """Test that AbacusCalculator can be imported."""
        try:
            from pyabacus.ase import AbacusCalculator
            assert AbacusCalculator is not None
        except ImportError as e:
            if "ASE" in str(e):
                pytest.skip("ASE not installed")
            raise

    def test_calculator_initialization(self, tmp_path):
        """Test calculator initialization."""
        try:
            from pyabacus.ase import AbacusCalculator
        except ImportError:
            pytest.skip("ASE not installed")

        calc = AbacusCalculator(input_dir=str(tmp_path), gamma_only=True)

        assert calc.input_dir == str(tmp_path)
        assert calc.gamma_only is True
        assert calc._initialized is False

    def test_implemented_properties(self):
        """Test that calculator declares correct implemented properties."""
        try:
            from pyabacus.ase import AbacusCalculator
        except ImportError:
            pytest.skip("ASE not installed")

        assert 'energy' in AbacusCalculator.implemented_properties
        assert 'forces' in AbacusCalculator.implemented_properties
        assert 'stress' in AbacusCalculator.implemented_properties

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_energy_calculation(self, mock_workflow_class, mock_workflow, mock_atoms, tmp_path):
        """Test energy calculation with unit conversion."""
        try:
            from pyabacus.ase import AbacusCalculator
        except ImportError:
            pytest.skip("ASE not installed")

        mock_workflow_class.return_value = mock_workflow

        calc = AbacusCalculator(input_dir=str(tmp_path))
        mock_atoms.calc = calc

        energy = mock_atoms.get_potential_energy()

        # Energy should be converted from Ry to eV
        expected_energy = -10.0 * 13.605693122994
        assert energy == pytest.approx(expected_energy, rel=1e-6)

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_force_calculation(self, mock_workflow_class, mock_workflow, mock_atoms, tmp_path):
        """Test force calculation with unit conversion."""
        try:
            from pyabacus.ase import AbacusCalculator
        except ImportError:
            pytest.skip("ASE not installed")

        mock_workflow_class.return_value = mock_workflow

        calc = AbacusCalculator(input_dir=str(tmp_path))
        mock_atoms.calc = calc

        forces = mock_atoms.get_forces()

        # Forces should be converted from Ry/Bohr to eV/Ang
        assert forces.shape == (2, 3)
        # First atom force in x direction: 0.1 Ry/Bohr * 25.7112 = 2.57112 eV/Ang
        expected_fx = 0.1 * ForceData.RY_BOHR_TO_EV_ANG
        assert forces[0, 0] == pytest.approx(expected_fx, rel=1e-6)

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_stress_calculation(self, mock_workflow_class, mock_workflow, mock_atoms, tmp_path):
        """Test stress calculation with unit conversion and Voigt notation."""
        try:
            from pyabacus.ase import AbacusCalculator
        except ImportError:
            pytest.skip("ASE not installed")

        mock_workflow_class.return_value = mock_workflow

        calc = AbacusCalculator(input_dir=str(tmp_path))
        mock_atoms.calc = calc

        stress = mock_atoms.get_stress()

        # Stress should be in Voigt notation (6,)
        assert stress.shape == (6,)
        # Diagonal elements should be 10 kbar converted to eV/Ang^3
        expected_diag = 10.0 * StressData.KBAR_TO_EV_ANG3
        assert stress[0] == pytest.approx(expected_diag, rel=1e-6)


# ============================================================================
# Integration Test Markers (require full ABACUS installation)
# ============================================================================

@pytest.mark.integration
class TestAbacusCalculatorIntegration:
    """Integration tests requiring full ABACUS installation."""

    @pytest.fixture
    def si2_input_dir(self, tmp_path):
        """Create Si2 test input files."""
        # Create minimal INPUT file
        input_content = """INPUT_PARAMETERS
calculation scf
basis_type lcao
gamma_only 1
ecutwfc 50
scf_thr 1e-6
scf_nmax 100
"""
        (tmp_path / "INPUT").write_text(input_content)

        # Create minimal STRU file
        stru_content = """ATOMIC_SPECIES
Si 28.085 Si_ONCV_PBE-1.0.upf

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

    @pytest.mark.skip(reason="Requires full ABACUS installation")
    def test_bfgs_relaxation(self, si2_input_dir):
        """Test BFGS geometry optimization."""
        from ase.optimize import BFGS
        from pyabacus.ase import AbacusCalculator

        calc = AbacusCalculator(input_dir=str(si2_input_dir))

        # Create Si2 atoms
        from ase import Atoms
        atoms = Atoms('Si2',
                     positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                     cell=[5.43, 5.43, 5.43],
                     pbc=True)
        atoms.calc = calc

        opt = BFGS(atoms)
        opt.run(fmax=0.05)

        assert opt.converged()

    @pytest.mark.skip(reason="Requires full ABACUS installation")
    def test_fire_relaxation(self, si2_input_dir):
        """Test FIRE geometry optimization."""
        from ase.optimize import FIRE
        from pyabacus.ase import AbacusCalculator

        calc = AbacusCalculator(input_dir=str(si2_input_dir))

        from ase import Atoms
        atoms = Atoms('Si2',
                     positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                     cell=[5.43, 5.43, 5.43],
                     pbc=True)
        atoms.calc = calc

        opt = FIRE(atoms)
        opt.run(fmax=0.05)

        assert opt.converged()

    @pytest.mark.skip(reason="Requires full ABACUS installation")
    def test_cell_relaxation(self, si2_input_dir):
        """Test cell + ion relaxation using UnitCellFilter."""
        from ase.optimize import BFGS
        from ase.constraints import UnitCellFilter
        from pyabacus.ase import AbacusCalculator

        calc = AbacusCalculator(input_dir=str(si2_input_dir))

        from ase import Atoms
        atoms = Atoms('Si2',
                     positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                     cell=[5.43, 5.43, 5.43],
                     pbc=True)
        atoms.calc = calc

        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf)
        opt.run(fmax=0.05)

        assert opt.converged()


# ============================================================================
# Test Fixtures for conftest.py
# ============================================================================

@pytest.fixture
def sample_forces():
    """Sample force data for testing."""
    return np.array([
        [0.01, 0.02, 0.03],
        [-0.01, -0.02, -0.03],
    ])


@pytest.fixture
def sample_stress():
    """Sample stress tensor for testing."""
    return np.array([
        [10.0, 1.0, 2.0],
        [1.0, 20.0, 3.0],
        [2.0, 3.0, 30.0],
    ])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
