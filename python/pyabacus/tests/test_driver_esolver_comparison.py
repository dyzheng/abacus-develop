"""
Integration tests comparing Driver mode and ESolver mode results.

Tests verify:
1. Energy and convergence consistency between modes
2. Density matrix restart acceleration
3. Density matrix data access in ESolver mode

Each ESolver/Driver calculation runs in a **subprocess** so that C++ global
variables (PARAM, GlobalV, static class members set by Input_Conv::Convert())
are fully isolated between runs.  This avoids the state-leakage problem that
occurs when running multiple calculations in a single Python process.

Test logic: build a unified dict via ``pyabacus.prepare.read_directory``
first, apply any modifications to the dict, then write it out with
``write_directory`` before running the Driver / ESolver interface.
The dict is the **single source of truth** — ``gamma_only`` is read
automatically from the INPUT file by LCAOWorkflow, not passed externally.
"""

import json
import subprocess
import sys
import textwrap

import pytest
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pyabacus.prepare import read_directory, write_directory

from conftest_abacus import (
    WorkingDirectory,
    GAMMA_TEST_CASE,
    MULTIK_TEST_CASE,
    PP_ORB_DIR,
    RY_TO_EV,
)

# Path to pyabacus src directory (for subprocess sys.path)
_SRC_PATH = str(Path(__file__).parent.parent / 'src')


# ============================================================================
# Helper Functions
# ============================================================================

def build_test_data(src_dir: Path, input_modifications: dict = None) -> dict:
    """Read a test-case directory into a unified dict and patch paths.

    This is the single place where the dict is constructed.  All path
    fixups (``pseudo_dir``, ``orbital_dir``) and optional INPUT tweaks
    are applied here so that later steps only consume the dict.
    """
    data = read_directory(str(src_dir))
    data["input"]["pseudo_dir"] = str(PP_ORB_DIR)
    data["input"]["orbital_dir"] = str(PP_ORB_DIR)
    if input_modifications:
        data["input"].update(input_modifications)
    return data


def write_test_directory(data: dict, dst_dir: Path) -> None:
    """Write a unified dict to *dst_dir* and create the output subdir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    write_directory(data, str(dst_dir))
    suffix = data["input"].get("suffix", "autotest")
    (dst_dir / f"OUT.{suffix}").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Subprocess runners — each calculation runs in a fresh Python process
# so that ALL C++ global / static state is completely isolated.
# ---------------------------------------------------------------------------

def run_esolver_subprocess(work_dir: Path) -> dict:
    """Run ESolver calculation in a subprocess.

    Returns dict with keys: energy_ev, niter, converged.
    """
    result_file = work_dir / "_esolver_result.json"
    script = textwrap.dedent(f"""\
        import sys, json, os
        sys.path.insert(0, {_SRC_PATH!r})
        os.chdir({str(work_dir)!r})

        from mpi4py import MPI
        from pyabacus.esolver import LCAOWorkflow

        workflow = LCAOWorkflow("./")
        workflow.initialize()
        result = workflow.run_scf(max_iter=100)

        data = {{
            "energy_ev": result.energy.etot * {RY_TO_EV},
            "niter": result.niter,
            "converged": result.converged,
        }}

        workflow.cleanup()

        with open({str(result_file)!r}, "w") as f:
            json.dump(data, f)
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )

    if proc.returncode != 0:
        # Check if it's an import error (module not built)
        if "ImportError" in proc.stderr or "ModuleNotFoundError" in proc.stderr:
            pytest.skip(f"ESolver module not available:\n{proc.stderr[-500:]}")
        pytest.fail(
            f"ESolver subprocess failed (exit {proc.returncode}):\n"
            f"STDOUT (last 2000 chars):\n{proc.stdout[-2000:]}\n"
            f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}"
        )

    with open(result_file) as f:
        return json.load(f)


def run_driver_subprocess(work_dir: Path) -> dict:
    """Run Driver calculation in a subprocess.

    Returns dict with keys: energy_ev, niter, converged.
    """
    result_file = work_dir / "_driver_result.json"
    script = textwrap.dedent(f"""\
        import sys, json, os
        sys.path.insert(0, {_SRC_PATH!r})
        os.chdir({str(work_dir)!r})

        from mpi4py import MPI
        from pyabacus.driver import abacus

        result = abacus(
            input_dir="./",
            calculate_force=False,
            calculate_stress=False,
            verbosity=0,
        )

        data = {{
            "energy_ev": result.etot,
            "niter": result.niter,
            "converged": result.converged,
        }}

        with open({str(result_file)!r}, "w") as f:
            json.dump(data, f)
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )

    if proc.returncode != 0:
        if "ImportError" in proc.stderr or "ModuleNotFoundError" in proc.stderr:
            pytest.skip(f"Driver module not available:\n{proc.stderr[-500:]}")
        pytest.fail(
            f"Driver subprocess failed (exit {proc.returncode}):\n"
            f"STDOUT (last 2000 chars):\n{proc.stdout[-2000:]}\n"
            f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}"
        )

    with open(result_file) as f:
        return json.load(f)


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


# ============================================================================
# Test 1: Driver vs ESolver Mode Comparison (subprocess-isolated)
# ============================================================================

@pytest.mark.integration
class TestDriverVsESolverComparison:
    """Compare results between driver mode and ESolver mode.

    Each run executes in a **separate subprocess** for complete C++ global
    state isolation.
    """

    def test_gamma_only_energy_consistency(self, tmp_path):
        """
        Verify energy and niter consistency for gamma-only calculation.

        Uses tests/02_NAO_Gamma/001_NO_GO_OHK (Si2, gamma-only).
        Reference energy: -204.106280612482 eV
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")

        # --- dict construction ---
        data = build_test_data(GAMMA_TEST_CASE)
        assert data["input"].get("gamma_only", 1) == 1, \
            "Expected gamma_only=1 from dict"

        # --- ESolver run (subprocess) ---
        es_dir = tmp_path / "gamma_esolver"
        write_test_directory(data, es_dir)
        esolver_result = run_esolver_subprocess(es_dir)
        assert esolver_result['converged'], \
            f"ESolver SCF did not converge after {esolver_result['niter']} iters"

        # --- Driver run (subprocess, same data, fresh dir) ---
        dr_dir = tmp_path / "gamma_driver"
        write_test_directory(data, dr_dir)
        driver_result = run_driver_subprocess(dr_dir)
        assert driver_result['converged'], \
            f"Driver SCF did not converge after {driver_result['niter']} iters"

        # --- compare ---
        energy_diff = abs(esolver_result['energy_ev'] - driver_result['energy_ev'])

        print(f"\n=== Gamma-only Energy Comparison ===")
        print(f"ESolver: E={esolver_result['energy_ev']:.8f} eV, "
              f"niter={esolver_result['niter']}")
        print(f"Driver:  E={driver_result['energy_ev']:.8f} eV, "
              f"niter={driver_result['niter']}")
        print(f"Energy difference: {energy_diff:.2e} eV")

        assert energy_diff < 1e-4, \
            f"Energy mismatch: {energy_diff:.2e} eV > 1e-4 eV"
        assert esolver_result['niter'] == driver_result['niter'], \
            f"Iteration mismatch: {esolver_result['niter']} vs {driver_result['niter']}"

    def test_multi_k_energy_consistency(self, tmp_path):
        """
        Verify energy and niter consistency for multi-k calculation.

        Uses tests/03_NAO_multik/02_NO_KP_15 (Si2, 2x2x2 k-points).
        Reference energy: -204.8201568334281149 eV
        """
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        # --- dict construction ---
        data = build_test_data(
            MULTIK_TEST_CASE,
            input_modifications={'init_chg': 'atomic'},
        )
        assert data["input"].get("gamma_only", 1) == 0, \
            "Expected gamma_only=0 from dict"

        # --- ESolver run (subprocess) ---
        es_dir = tmp_path / "multik_esolver"
        write_test_directory(data, es_dir)
        esolver_result = run_esolver_subprocess(es_dir)
        assert esolver_result['converged'], "ESolver SCF did not converge"

        # --- Driver run (subprocess) ---
        dr_dir = tmp_path / "multik_driver"
        write_test_directory(data, dr_dir)
        driver_result = run_driver_subprocess(dr_dir)
        assert driver_result['converged'], "Driver SCF did not converge"

        # --- compare ---
        energy_diff = abs(esolver_result['energy_ev'] - driver_result['energy_ev'])

        print(f"\n=== Multi-k Energy Comparison ===")
        print(f"ESolver: E={esolver_result['energy_ev']:.8f} eV, "
              f"niter={esolver_result['niter']}")
        print(f"Driver:  E={driver_result['energy_ev']:.8f} eV, "
              f"niter={driver_result['niter']}")
        print(f"Energy difference: {energy_diff:.2e} eV")

        assert energy_diff < 1e-2, \
            f"Energy mismatch: {energy_diff:.2e} eV > 1e-2 eV"
        assert esolver_result['niter'] == driver_result['niter'], \
            f"Iteration mismatch: {esolver_result['niter']} vs {driver_result['niter']}"


# ============================================================================
# Test 2: Density Matrix Restart Acceleration (subprocess-isolated)
# ============================================================================

@pytest.mark.integration
class TestDensityMatrixRestart:
    """Test density matrix restart acceleration."""

    def test_dm_restart_reduces_iterations_driver_mode(self, tmp_path):
        """
        Verify DM restart reduces SCF iterations in driver mode.

        Workflow:
        1. Build dict with init_chg=atomic, out_dmr=1
        2. Run calculation, locate output dmrs1_nao.csr
        3. Build dict with init_chg=dm, copy DM file, run again
        4. Verify second run has fewer iterations
        """
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        # --- dict construction: run 1 (generate DM) ---
        data_run1 = build_test_data(
            MULTIK_TEST_CASE,
            input_modifications={'init_chg': 'atomic', 'out_dmr': '1'},
        )

        run1_dir = tmp_path / "run1_generate_dm"
        write_test_directory(data_run1, run1_dir)
        result1 = run_driver_subprocess(run1_dir)

        assert result1['converged'], \
            f"Run1 did not converge after {result1['niter']} iterations"

        niter1 = result1['niter']
        energy1 = result1['energy_ev']

        print(f"\n=== Run 1: Generate DM (init_chg=atomic) ===")
        print(f"Iterations: {niter1}")
        print(f"Energy: {energy1:.8f} eV")

        # Find output DM file
        dmr_file = find_dmr_output_file(run1_dir)
        if dmr_file is None:
            dmr_file = MULTIK_TEST_CASE / 'dmrs1_nao.csr'
            if not dmr_file.exists():
                pytest.skip("No dmrs1_nao.csr available for restart test")

        # --- dict construction: run 2 (DM restart) ---
        data_run2 = build_test_data(
            MULTIK_TEST_CASE,
            input_modifications={'init_chg': 'dm', 'read_file_dir': './'},
        )

        run2_dir = tmp_path / "run2_dm_restart"
        write_test_directory(data_run2, run2_dir)
        shutil.copy(dmr_file, run2_dir / 'dmrs1_nao.csr')

        result2 = run_driver_subprocess(run2_dir)

        assert result2['converged'], \
            f"Run2 did not converge after {result2['niter']} iterations"

        niter2 = result2['niter']
        energy2 = result2['energy_ev']

        print(f"\n=== Run 2: DM Restart (init_chg=dm) ===")
        print(f"Iterations: {niter2}")
        print(f"Energy: {energy2:.8f} eV")

        # --- verify ---
        print(f"\n=== Comparison ===")
        print(f"Iteration reduction: {niter1} -> {niter2}")
        print(f"Energy difference: {abs(energy1 - energy2):.2e} eV")

        assert niter2 < niter1, \
            f"DM restart should reduce iterations: {niter2} >= {niter1}"
        assert abs(energy1 - energy2) < 1e-5, \
            f"Energy should be consistent: |{energy1} - {energy2}| > 1e-5"

    def test_dm_restart_esolver_vs_driver_consistency(self, tmp_path):
        """
        Verify ESolver and Driver modes give same results with DM restart.

        Both modes should read the same dmrs1_nao.csr and produce
        identical energy and iteration count.
        """
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        dmr_source = MULTIK_TEST_CASE / 'dmrs1_nao.csr'
        if not dmr_source.exists():
            pytest.skip("dmrs1_nao.csr not found for restart test")

        # --- dict construction (shared by both runs) ---
        data = build_test_data(MULTIK_TEST_CASE)

        # --- ESolver run (subprocess) ---
        es_dir = tmp_path / "dm_restart_esolver"
        write_test_directory(data, es_dir)
        shutil.copy(dmr_source, es_dir / 'dmrs1_nao.csr')
        es_result = run_esolver_subprocess(es_dir)

        es_energy = es_result['energy_ev']
        es_niter = es_result['niter']

        # --- Driver run (subprocess) ---
        dr_dir = tmp_path / "dm_restart_driver"
        write_test_directory(data, dr_dir)
        shutil.copy(dmr_source, dr_dir / 'dmrs1_nao.csr')
        dr_result = run_driver_subprocess(dr_dir)

        dr_energy = dr_result['energy_ev']
        dr_niter = dr_result['niter']

        print(f"\n=== DM Restart: ESolver vs Driver ===")
        print(f"ESolver: E={es_energy:.8f} eV, niter={es_niter}")
        print(f"Driver:  E={dr_energy:.8f} eV, niter={dr_niter}")

        assert abs(es_energy - dr_energy) < 1e-4, \
            f"Energy mismatch: |{es_energy} - {dr_energy}| > 1e-4"
        assert es_niter == dr_niter, \
            f"Iteration count mismatch: {es_niter} vs {dr_niter}"


# ============================================================================
# Test 3: Density Matrix Access (in-process, needs direct object access)
# ============================================================================

@pytest.mark.integration
class TestDensityMatrixAccess:
    """Test density matrix data access in ESolver mode.

    These tests require direct access to the LCAOWorkflow Python object,
    so they run in-process rather than in a subprocess.
    """

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

        # --- dict construction ---
        data = build_test_data(GAMMA_TEST_CASE)

        # --- ESolver run ---
        work_dir = tmp_path / "dm_structure_test"
        write_test_directory(data, work_dir)

        with WorkingDirectory(work_dir):
            workflow = LCAOWorkflow("./")
            workflow.initialize()
            result = workflow.run_scf(max_iter=100)

            assert result.converged, "SCF did not converge"

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

            for ik, dmk in enumerate(dm_data.DMK):
                expected_shape = (dm_data.nrow, dm_data.ncol)
                assert dmk.shape == expected_shape, \
                    f"DMK[{ik}] shape {dmk.shape} != {expected_shape}"

            workflow.cleanup()

    def test_density_matrix_trace(self, tmp_path):
        """
        Verify density matrix trace is physically reasonable.

        For Si2 with 8 valence electrons, trace should be positive
        and related to the electron count.
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")

        try:
            from pyabacus.esolver import LCAOWorkflow
        except ImportError as e:
            pytest.skip(f"ESolver module not available: {e}")

        # --- dict construction ---
        data = build_test_data(GAMMA_TEST_CASE)

        # --- ESolver run ---
        work_dir = tmp_path / "dm_trace_test"
        write_test_directory(data, work_dir)

        with WorkingDirectory(work_dir):
            workflow = LCAOWorkflow("./")
            workflow.initialize()
            result = workflow.run_scf(max_iter=100)
            assert result.converged, "SCF did not converge"

            dm_data = workflow.density_matrix

            print(f"\n=== Density Matrix Trace ===")
            for ik in range(dm_data.nks):
                trace = dm_data.trace(ik)
                print(f"DM trace[{ik}]: {trace}")

                assert trace.real > 0, \
                    f"DM trace should be positive, got {trace}"

            workflow.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
