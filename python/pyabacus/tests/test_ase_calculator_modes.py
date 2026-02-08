"""
Tests for ASE Calculator dual-mode enhancement.

This module tests the CalculatorMode enum and the dual-mode functionality
of AbacusCalculator (Driver mode vs ESolver mode).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Import data types directly to avoid C++ binding issues during testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pyabacus.esolver.data_types import (
    ForceData,
    StressData,
    EnergyData,
    SCFResult,
)


# ============================================================================
# Mode Selection Tests
# ============================================================================

class TestCalculatorModeEnum:
    """Tests for CalculatorMode enum."""

    def test_calculator_mode_enum_exists(self):
        """Test CalculatorMode.DRIVER and ESOLVER exist."""
        from pyabacus.ase import CalculatorMode

        assert hasattr(CalculatorMode, 'DRIVER')
        assert hasattr(CalculatorMode, 'ESOLVER')

    def test_calculator_mode_values_are_distinct(self):
        """Test that DRIVER and ESOLVER have distinct values."""
        from pyabacus.ase import CalculatorMode

        assert CalculatorMode.DRIVER != CalculatorMode.ESOLVER

    def test_default_mode_is_esolver(self, tmp_path):
        """Test backward compatibility - default mode is ESOLVER."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        calc = AbacusCalculator(input_dir=str(tmp_path))
        assert calc.mode == CalculatorMode.ESOLVER

    def test_mode_can_be_set_by_string_driver(self, tmp_path):
        """Test mode can be set using 'driver' string."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        calc = AbacusCalculator(input_dir=str(tmp_path), mode='driver')
        assert calc.mode == CalculatorMode.DRIVER

    def test_mode_can_be_set_by_string_esolver(self, tmp_path):
        """Test mode can be set using 'esolver' string."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        calc = AbacusCalculator(input_dir=str(tmp_path), mode='esolver')
        assert calc.mode == CalculatorMode.ESOLVER

    def test_mode_can_be_set_by_string_case_insensitive(self, tmp_path):
        """Test mode string is case insensitive."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        calc1 = AbacusCalculator(input_dir=str(tmp_path), mode='DRIVER')
        calc2 = AbacusCalculator(input_dir=str(tmp_path), mode='Driver')
        calc3 = AbacusCalculator(input_dir=str(tmp_path), mode='ESOLVER')

        assert calc1.mode == CalculatorMode.DRIVER
        assert calc2.mode == CalculatorMode.DRIVER
        assert calc3.mode == CalculatorMode.ESOLVER

    def test_mode_can_be_set_by_enum(self, tmp_path):
        """Test mode can be set using CalculatorMode enum directly."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)
        assert calc.mode == CalculatorMode.DRIVER

    def test_invalid_mode_raises_error(self, tmp_path):
        """Test that invalid mode string raises ValueError."""
        from pyabacus.ase import AbacusCalculator

        with pytest.raises((ValueError, KeyError)):
            AbacusCalculator(input_dir=str(tmp_path), mode='invalid_mode')


# ============================================================================
# Driver Mode Tests
# ============================================================================

class TestDriverMode:
    """Tests for Driver mode functionality."""

    @pytest.fixture
    def mock_abacus_result(self):
        """Create a mock CalculationResult from driver."""
        from pyabacus.driver import CalculationResult

        result = CalculationResult(
            converged=True,
            niter=15,
            drho=1e-9,
            etot=-136.05,  # Already in eV from driver
            eband=-50.0,
            hartree_energy=20.0,
            etxc=-30.0,
            ewald_energy=10.0,
            nat=2,
        )
        result.forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])  # eV/Ang
        result.stress = np.array([
            [10.0, 1.0, 2.0],
            [1.0, 20.0, 3.0],
            [2.0, 3.0, 30.0]
        ])  # kbar
        return result

    @pytest.fixture
    def mock_atoms(self):
        """Create mock ASE Atoms object."""
        try:
            from ase import Atoms
            return Atoms('Si2', positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                        cell=[5.43, 5.43, 5.43], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

    @patch('pyabacus.driver.abacus')
    def test_driver_mode_calls_abacus_function(self, mock_abacus, mock_abacus_result, mock_atoms, tmp_path):
        """Test that Driver mode calls the abacus() function."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_abacus.return_value = mock_abacus_result

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)
        mock_atoms.calc = calc

        _ = mock_atoms.get_potential_energy()

        mock_abacus.assert_called_once()

    @patch('pyabacus.driver.abacus')
    def test_driver_mode_no_memory_persistence(self, mock_abacus, mock_abacus_result, mock_atoms, tmp_path):
        """Test that Driver mode calls abacus() for each calculation."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_abacus.return_value = mock_abacus_result

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)
        mock_atoms.calc = calc

        # First calculation
        _ = mock_atoms.get_potential_energy()
        assert mock_abacus.call_count == 1

        # Trigger recalculation by changing positions
        mock_atoms.positions[0, 0] += 0.01
        _ = mock_atoms.get_potential_energy()
        assert mock_abacus.call_count == 2

    @patch('pyabacus.driver.abacus')
    def test_driver_mode_energy_already_in_eV(self, mock_abacus, mock_abacus_result, mock_atoms, tmp_path):
        """Test that Driver mode energy is already in eV (no conversion needed)."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_abacus.return_value = mock_abacus_result

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)
        mock_atoms.calc = calc

        energy = mock_atoms.get_potential_energy()

        # Energy from driver is already in eV
        assert energy == pytest.approx(-136.05, rel=1e-6)

    @patch('pyabacus.driver.abacus')
    def test_driver_mode_forces_already_in_eV_Ang(self, mock_abacus, mock_abacus_result, mock_atoms, tmp_path):
        """Test that Driver mode forces are already in eV/Ang."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_abacus.return_value = mock_abacus_result

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)
        mock_atoms.calc = calc

        forces = mock_atoms.get_forces()

        # Forces from driver are already in eV/Ang
        np.testing.assert_allclose(forces[0], [0.1, 0.2, 0.3], rtol=1e-6)

    @patch('pyabacus.driver.abacus')
    def test_driver_mode_stress_voigt_notation(self, mock_abacus, mock_abacus_result, mock_atoms, tmp_path):
        """Test that Driver mode stress is converted to Voigt notation in eV/Ang^3."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_abacus.return_value = mock_abacus_result

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)
        mock_atoms.calc = calc

        stress = mock_atoms.get_stress()

        # Stress should be in Voigt notation (6,)
        assert stress.shape == (6,)

    @patch('pyabacus.driver.abacus')
    def test_driver_mode_passes_nprocs_nthreads(self, mock_abacus, mock_abacus_result, mock_atoms, tmp_path):
        """Test that Driver mode passes nprocs and nthreads to abacus()."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_abacus.return_value = mock_abacus_result

        calc = AbacusCalculator(
            input_dir=str(tmp_path),
            mode=CalculatorMode.DRIVER,
            nprocs=4,
            nthreads=2,
        )
        mock_atoms.calc = calc

        _ = mock_atoms.get_potential_energy()

        # Check that abacus was called with nprocs and nthreads
        call_kwargs = mock_abacus.call_args[1]
        assert call_kwargs.get('nprocs') == 4
        assert call_kwargs.get('nthreads') == 2


# ============================================================================
# ESolver Mode Tests
# ============================================================================

class TestESolverMode:
    """Tests for ESolver mode functionality."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock LCAOWorkflow."""
        workflow = Mock()

        # Mock energy accessor
        workflow.energy = EnergyData(
            etot=-10.0, eband=-5.0, hartree_energy=2.0,
            etxc=-3.0, ewald_energy=1.0
        )

        # Mock force accessor
        workflow.force = ForceData(
            forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
            nat=2
        )

        # Mock stress accessor
        workflow.stress = StressData(stress=np.eye(3) * 10.0)

        # Mock SCF result
        workflow.run_scf.return_value = SCFResult(
            converged=True, niter=10, drho=1e-8,
            energy=workflow.energy
        )

        # Mock cleanup method
        workflow.cleanup = Mock()

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

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_esolver_mode_uses_workflow(self, mock_workflow_class, mock_workflow, mock_atoms, tmp_path):
        """Test that ESolver mode instantiates LCAOWorkflow."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow_class.return_value = mock_workflow

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER)
        mock_atoms.calc = calc

        _ = mock_atoms.get_potential_energy()

        mock_workflow_class.assert_called_once()

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_esolver_mode_memory_persistence(self, mock_workflow_class, mock_workflow, mock_atoms, tmp_path):
        """Test that ESolver mode reuses workflow between calculations."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow_class.return_value = mock_workflow

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER)
        mock_atoms.calc = calc

        # First calculation
        _ = mock_atoms.get_potential_energy()
        # Second calculation (forces)
        _ = mock_atoms.get_forces()

        # Workflow should only be instantiated once
        assert mock_workflow_class.call_count == 1

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_esolver_mode_position_update(self, mock_workflow_class, mock_workflow, mock_atoms, tmp_path):
        """Test that ESolver mode calls update_positions() on position change."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow_class.return_value = mock_workflow

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER)
        mock_atoms.calc = calc

        # First calculation
        _ = mock_atoms.get_potential_energy()

        # Change positions
        mock_atoms.positions[0, 0] += 0.01

        # Second calculation
        _ = mock_atoms.get_potential_energy()

        # update_positions should have been called
        assert mock_workflow.update_positions.called

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_esolver_mode_energy_conversion(self, mock_workflow_class, mock_workflow, mock_atoms, tmp_path):
        """Test that ESolver mode converts energy from Ry to eV."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow_class.return_value = mock_workflow

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER)
        mock_atoms.calc = calc

        energy = mock_atoms.get_potential_energy()

        # Energy should be converted from Ry to eV
        # -10.0 Ry * 13.605698 = -136.05698 eV
        expected_energy = -10.0 * 13.605698
        assert energy == pytest.approx(expected_energy, rel=1e-6)


# ============================================================================
# Cleanup Tests
# ============================================================================

class TestCleanup:
    """Tests for cleanup functionality."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock LCAOWorkflow with cleanup."""
        workflow = Mock()
        workflow.energy = EnergyData(etot=-10.0)
        workflow.force = ForceData(forces=np.array([[0.1, 0.0, 0.0]]), nat=1)
        workflow.stress = StressData(stress=np.eye(3) * 10.0)
        workflow.run_scf.return_value = SCFResult(
            converged=True, niter=10, drho=1e-8,
            energy=workflow.energy
        )
        workflow.cleanup = Mock()
        return workflow

    def test_cleanup_method_exists(self, tmp_path):
        """Test that cleanup() method exists on AbacusCalculator."""
        from pyabacus.ase import AbacusCalculator

        calc = AbacusCalculator(input_dir=str(tmp_path))
        assert hasattr(calc, 'cleanup')
        assert callable(calc.cleanup)

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_cleanup_releases_workflow(self, mock_workflow_class, mock_workflow, tmp_path):
        """Test that cleanup() releases the workflow (sets to None)."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow_class.return_value = mock_workflow

        try:
            from ase import Atoms
            atoms = Atoms('Si', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER)
        atoms.calc = calc

        # Run a calculation to initialize workflow
        _ = atoms.get_potential_energy()
        assert calc._workflow is not None

        # Cleanup
        calc.cleanup()

        # Workflow should be None
        assert calc._workflow is None
        assert calc._initialized is False

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_cleanup_allows_reinitialization(self, mock_workflow_class, mock_workflow, tmp_path):
        """Test that calculation can run again after cleanup."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow_class.return_value = mock_workflow

        try:
            from ase import Atoms
            atoms = Atoms('Si', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER)
        atoms.calc = calc

        # First calculation
        energy1 = atoms.get_potential_energy()

        # Cleanup
        calc.cleanup()

        # Second calculation after cleanup
        energy2 = atoms.get_potential_energy()

        # Both should work
        assert energy1 == energy2

    def test_cleanup_is_idempotent(self, tmp_path):
        """Test that multiple cleanup() calls are safe."""
        from pyabacus.ase import AbacusCalculator

        calc = AbacusCalculator(input_dir=str(tmp_path))

        # Multiple cleanups should not raise
        calc.cleanup()
        calc.cleanup()
        calc.cleanup()

    @patch('pyabacus.driver.abacus')
    def test_cleanup_in_driver_mode_is_noop(self, mock_abacus, tmp_path):
        """Test that cleanup() in Driver mode is a no-op (no error)."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode
        from pyabacus.driver import CalculationResult

        mock_abacus.return_value = CalculationResult(converged=True, etot=-100.0)

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)

        # Cleanup should not raise in driver mode
        calc.cleanup()
        calc.cleanup()


# ============================================================================
# Context Manager Tests
# ============================================================================

class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_protocol(self, tmp_path):
        """Test that __enter__ and __exit__ exist."""
        from pyabacus.ase import AbacusCalculator

        calc = AbacusCalculator(input_dir=str(tmp_path))

        assert hasattr(calc, '__enter__')
        assert hasattr(calc, '__exit__')
        assert callable(calc.__enter__)
        assert callable(calc.__exit__)

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_context_manager_cleanup_on_exit(self, mock_workflow_class, tmp_path):
        """Test that context manager calls cleanup on exit."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow = Mock()
        mock_workflow.energy = EnergyData(etot=-10.0)
        mock_workflow.run_scf.return_value = SCFResult(
            converged=True, niter=10, drho=1e-8,
            energy=mock_workflow.energy
        )
        mock_workflow.cleanup = Mock()
        mock_workflow_class.return_value = mock_workflow

        try:
            from ase import Atoms
            atoms = Atoms('Si', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

        with AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER) as calc:
            atoms.calc = calc
            _ = atoms.get_potential_energy()
            assert calc._workflow is not None

        # After exiting context, workflow should be cleaned up
        assert calc._workflow is None

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_context_manager_cleanup_on_exception(self, mock_workflow_class, tmp_path):
        """Test that context manager calls cleanup even on exception."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow = Mock()
        mock_workflow.energy = EnergyData(etot=-10.0)
        mock_workflow.run_scf.return_value = SCFResult(
            converged=True, niter=10, drho=1e-8,
            energy=mock_workflow.energy
        )
        mock_workflow.cleanup = Mock()
        mock_workflow_class.return_value = mock_workflow

        try:
            from ase import Atoms
            atoms = Atoms('Si', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

        calc = None
        try:
            with AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER) as calc:
                atoms.calc = calc
                _ = atoms.get_potential_energy()
                raise ValueError("Test exception")
        except ValueError:
            pass

        # After exception, workflow should still be cleaned up
        assert calc._workflow is None

    def test_context_manager_returns_self(self, tmp_path):
        """Test that __enter__ returns self."""
        from pyabacus.ase import AbacusCalculator

        calc = AbacusCalculator(input_dir=str(tmp_path))

        with calc as c:
            assert c is calc


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock LCAOWorkflow."""
        workflow = Mock()
        workflow.energy = EnergyData(etot=-10.0)
        workflow.force = ForceData(forces=np.array([[0.1, 0.0, 0.0]]), nat=1)
        workflow.stress = StressData(stress=np.eye(3) * 10.0)
        workflow.run_scf.return_value = SCFResult(
            converged=True, niter=10, drho=1e-8,
            energy=workflow.energy
        )
        workflow.cleanup = Mock()
        return workflow

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_existing_api_unchanged(self, mock_workflow_class, mock_workflow, tmp_path):
        """Test that existing API without mode parameter works as before."""
        from pyabacus.ase import AbacusCalculator

        mock_workflow_class.return_value = mock_workflow

        try:
            from ase import Atoms
            atoms = Atoms('Si', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

        # Old API without mode parameter
        calc = AbacusCalculator(input_dir=str(tmp_path), gamma_only=True)
        atoms.calc = calc

        energy = atoms.get_potential_energy()

        # Should work as before (ESolver mode)
        assert energy is not None
        mock_workflow_class.assert_called_once()

    @patch('pyabacus.ase.calculator.LCAOWorkflow')
    def test_workflow_property_still_works(self, mock_workflow_class, mock_workflow, tmp_path):
        """Test that workflow property works in ESolver mode."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode

        mock_workflow_class.return_value = mock_workflow

        try:
            from ase import Atoms
            atoms = Atoms('Si', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        except ImportError:
            pytest.skip("ASE not installed")

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.ESOLVER)
        atoms.calc = calc

        _ = atoms.get_potential_energy()

        # workflow property should return the workflow
        assert calc.workflow is mock_workflow

    @patch('pyabacus.driver.abacus')
    def test_workflow_property_none_in_driver_mode(self, mock_abacus, tmp_path):
        """Test that workflow property is None in Driver mode."""
        from pyabacus.ase import AbacusCalculator, CalculatorMode
        from pyabacus.driver import CalculationResult

        mock_abacus.return_value = CalculationResult(converged=True, etot=-100.0)

        calc = AbacusCalculator(input_dir=str(tmp_path), mode=CalculatorMode.DRIVER)

        # workflow property should be None in driver mode
        assert calc.workflow is None


# ============================================================================
# LCAOWorkflow Cleanup Tests
# ============================================================================

class TestLCAOWorkflowCleanup:
    """Tests for LCAOWorkflow.cleanup() method."""

    def test_workflow_cleanup_method_exists(self):
        """Test that cleanup() method exists on LCAOWorkflow."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(input_dir='.', gamma_only=True)
        assert hasattr(workflow, 'cleanup')
        assert callable(workflow.cleanup)

    def test_workflow_cleanup_resets_state(self):
        """Test that cleanup() resets internal state."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(input_dir='.', gamma_only=True)

        # Manually set some state
        workflow._initialized = True
        workflow._scf_running = True

        # Cleanup
        workflow.cleanup()

        # State should be reset
        assert workflow._initialized is False
        assert workflow._scf_running is False
        assert workflow._esolver is None

    def test_workflow_cleanup_clears_callbacks(self):
        """Test that cleanup() clears callbacks."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(input_dir='.', gamma_only=True)

        # Register a callback
        def dummy_callback(wf):
            pass

        workflow.register_callback('before_scf', dummy_callback)
        assert len(workflow._callbacks['before_scf']) == 1

        # Cleanup
        workflow.cleanup()

        # Callbacks should be cleared
        assert len(workflow._callbacks['before_scf']) == 0

    def test_workflow_cleanup_is_idempotent(self):
        """Test that multiple cleanup() calls are safe."""
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow(input_dir='.', gamma_only=True)

        # Multiple cleanups should not raise
        workflow.cleanup()
        workflow.cleanup()
        workflow.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
