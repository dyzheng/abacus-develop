"""
Sequential multi-run tests for ESolver mode.

Each ESolver / Driver calculation runs in a **subprocess** so that C++ global
variables are fully isolated.  The tests verify that ESolver results are
identical to Driver results across different calculation types and orderings.

Test matrix:
1. gamma -> multi-k (different basis/k-grid)
2. multi-k -> gamma (reversed order)
3. same calculation twice (reproducibility)
4. nspin=1 -> nspin=2 (different spin settings)
"""

import json
import subprocess
import sys
import textwrap

import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pyabacus.prepare import read_directory, write_directory

from conftest_abacus import (
    GAMMA_TEST_CASE,
    MULTIK_TEST_CASE,
    GAMMA_SPIN2_TEST_CASE,
    PP_ORB_DIR,
    RY_TO_EV,
)

# Path to pyabacus src directory (for subprocess sys.path)
_SRC_PATH = str(Path(__file__).parent.parent / 'src')


# ============================================================================
# Helper Functions
# ============================================================================

def build_test_data(src_dir: Path, input_modifications: dict = None) -> dict:
    """Read a test-case directory into a unified dict and patch paths."""
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


# ============================================================================
# Sequential Multi-Run Tests (subprocess-isolated)
# ============================================================================

@pytest.mark.integration
class TestSequentialMultiRun:
    """Test ESolver calculations across different types.

    Each calculation runs in its own subprocess for complete C++ global
    state isolation.  The tests verify that ESolver produces correct
    results for various calculation types and that each result matches
    its Driver reference.
    """

    def test_sequential_gamma_then_multik(self, tmp_path):
        """
        Run gamma ESolver -> multi-k ESolver.

        Each ESolver result must match its driver reference.
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        # --- Build data dicts ---
        gamma_data = build_test_data(GAMMA_TEST_CASE)
        multik_data = build_test_data(
            MULTIK_TEST_CASE,
            input_modifications={'init_chg': 'atomic'},
        )

        # --- ESolver runs ---
        gamma_es_dir = tmp_path / "gamma_esolver"
        write_test_directory(gamma_data, gamma_es_dir)
        gamma_es = run_esolver_subprocess(gamma_es_dir)
        assert gamma_es['converged'], \
            f"Gamma ESolver did not converge after {gamma_es['niter']} iters"

        multik_es_dir = tmp_path / "multik_esolver"
        write_test_directory(multik_data, multik_es_dir)
        multik_es = run_esolver_subprocess(multik_es_dir)
        assert multik_es['converged'], \
            f"Multi-k ESolver did not converge after {multik_es['niter']} iters"

        # --- Driver references ---
        gamma_dr_dir = tmp_path / "gamma_driver"
        write_test_directory(gamma_data, gamma_dr_dir)
        gamma_dr = run_driver_subprocess(gamma_dr_dir)
        assert gamma_dr['converged'], "Gamma driver did not converge"

        multik_dr_dir = tmp_path / "multik_driver"
        write_test_directory(multik_data, multik_dr_dir)
        multik_dr = run_driver_subprocess(multik_dr_dir)
        assert multik_dr['converged'], "Multi-k driver did not converge"

        # --- Compare gamma ---
        gamma_ediff = abs(gamma_es['energy_ev'] - gamma_dr['energy_ev'])
        print(f"\n=== Sequential: Gamma (run 1) ===")
        print(f"ESolver: E={gamma_es['energy_ev']:.8f} eV, "
              f"niter={gamma_es['niter']}")
        print(f"Driver:  E={gamma_dr['energy_ev']:.8f} eV, "
              f"niter={gamma_dr['niter']}")
        print(f"Energy diff: {gamma_ediff:.2e} eV")

        assert gamma_ediff < 1e-4, \
            f"Gamma energy mismatch: {gamma_ediff:.2e} eV > 1e-4 eV"
        assert gamma_es['niter'] == gamma_dr['niter'], \
            f"Gamma niter mismatch: {gamma_es['niter']} vs {gamma_dr['niter']}"

        # --- Compare multi-k ---
        multik_ediff = abs(multik_es['energy_ev'] - multik_dr['energy_ev'])
        print(f"\n=== Sequential: Multi-k (run 2) ===")
        print(f"ESolver: E={multik_es['energy_ev']:.8f} eV, "
              f"niter={multik_es['niter']}")
        print(f"Driver:  E={multik_dr['energy_ev']:.8f} eV, "
              f"niter={multik_dr['niter']}")
        print(f"Energy diff: {multik_ediff:.2e} eV")

        assert multik_ediff < 1e-2, \
            f"Multi-k energy mismatch: {multik_ediff:.2e} eV > 1e-2 eV"
        assert multik_es['niter'] == multik_dr['niter'], \
            f"Multi-k niter mismatch: {multik_es['niter']} vs {multik_dr['niter']}"

    def test_sequential_multik_then_gamma(self, tmp_path):
        """
        Run multi-k ESolver -> gamma ESolver.

        Reversed order from test_sequential_gamma_then_multik.
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")
        if not MULTIK_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {MULTIK_TEST_CASE}")

        # --- Build data dicts ---
        multik_data = build_test_data(
            MULTIK_TEST_CASE,
            input_modifications={'init_chg': 'atomic'},
        )
        gamma_data = build_test_data(GAMMA_TEST_CASE)

        # --- ESolver runs (reversed order) ---
        multik_es_dir = tmp_path / "multik_esolver"
        write_test_directory(multik_data, multik_es_dir)
        multik_es = run_esolver_subprocess(multik_es_dir)
        assert multik_es['converged'], \
            f"Multi-k ESolver did not converge after {multik_es['niter']} iters"

        gamma_es_dir = tmp_path / "gamma_esolver"
        write_test_directory(gamma_data, gamma_es_dir)
        gamma_es = run_esolver_subprocess(gamma_es_dir)
        assert gamma_es['converged'], \
            f"Gamma ESolver did not converge after {gamma_es['niter']} iters"

        # --- Driver references ---
        multik_dr_dir = tmp_path / "multik_driver"
        write_test_directory(multik_data, multik_dr_dir)
        multik_dr = run_driver_subprocess(multik_dr_dir)
        assert multik_dr['converged'], "Multi-k driver did not converge"

        gamma_dr_dir = tmp_path / "gamma_driver"
        write_test_directory(gamma_data, gamma_dr_dir)
        gamma_dr = run_driver_subprocess(gamma_dr_dir)
        assert gamma_dr['converged'], "Gamma driver did not converge"

        # --- Compare multi-k ---
        multik_ediff = abs(multik_es['energy_ev'] - multik_dr['energy_ev'])
        print(f"\n=== Sequential: Multi-k (run 1) ===")
        print(f"ESolver: E={multik_es['energy_ev']:.8f} eV, "
              f"niter={multik_es['niter']}")
        print(f"Driver:  E={multik_dr['energy_ev']:.8f} eV, "
              f"niter={multik_dr['niter']}")
        print(f"Energy diff: {multik_ediff:.2e} eV")

        assert multik_ediff < 1e-2, \
            f"Multi-k energy mismatch: {multik_ediff:.2e} eV > 1e-2 eV"
        assert multik_es['niter'] == multik_dr['niter'], \
            f"Multi-k niter mismatch: {multik_es['niter']} vs {multik_dr['niter']}"

        # --- Compare gamma ---
        gamma_ediff = abs(gamma_es['energy_ev'] - gamma_dr['energy_ev'])
        print(f"\n=== Sequential: Gamma (run 2) ===")
        print(f"ESolver: E={gamma_es['energy_ev']:.8f} eV, "
              f"niter={gamma_es['niter']}")
        print(f"Driver:  E={gamma_dr['energy_ev']:.8f} eV, "
              f"niter={gamma_dr['niter']}")
        print(f"Energy diff: {gamma_ediff:.2e} eV")

        assert gamma_ediff < 1e-4, \
            f"Gamma energy mismatch: {gamma_ediff:.2e} eV > 1e-4 eV"
        assert gamma_es['niter'] == gamma_dr['niter'], \
            f"Gamma niter mismatch: {gamma_es['niter']} vs {gamma_dr['niter']}"

    def test_sequential_same_calculation_twice(self, tmp_path):
        """
        Run the same gamma-only calculation twice.

        Both runs must produce identical energy (< 1e-8 eV) and niter.
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")

        data = build_test_data(GAMMA_TEST_CASE)

        # --- First run ---
        run1_dir = tmp_path / "run1"
        write_test_directory(data, run1_dir)
        result1 = run_esolver_subprocess(run1_dir)
        assert result1['converged'], \
            f"Run 1 did not converge after {result1['niter']} iters"

        # --- Second run ---
        run2_dir = tmp_path / "run2"
        write_test_directory(data, run2_dir)
        result2 = run_esolver_subprocess(run2_dir)
        assert result2['converged'], \
            f"Run 2 did not converge after {result2['niter']} iters"

        # --- Compare ---
        ediff = abs(result1['energy_ev'] - result2['energy_ev'])
        print(f"\n=== Same Calculation Twice ===")
        print(f"Run 1: E={result1['energy_ev']:.10f} eV, niter={result1['niter']}")
        print(f"Run 2: E={result2['energy_ev']:.10f} eV, niter={result2['niter']}")
        print(f"Energy diff: {ediff:.2e} eV")

        assert ediff < 1e-8, \
            f"Energies not identical: {ediff:.2e} eV > 1e-8 eV"
        assert result1['niter'] == result2['niter'], \
            f"Niter mismatch: {result1['niter']} vs {result2['niter']}"

    def test_sequential_different_nspin(self, tmp_path):
        """
        Run nspin=1 ESolver -> nspin=2 ESolver.

        Each result must match its driver reference.
        """
        if not GAMMA_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_TEST_CASE}")
        if not GAMMA_SPIN2_TEST_CASE.exists():
            pytest.skip(f"Test case not found: {GAMMA_SPIN2_TEST_CASE}")

        # --- Build data dicts ---
        nspin1_data = build_test_data(GAMMA_TEST_CASE)
        nspin2_data = build_test_data(GAMMA_SPIN2_TEST_CASE)

        # --- ESolver runs ---
        nspin1_es_dir = tmp_path / "nspin1_esolver"
        write_test_directory(nspin1_data, nspin1_es_dir)
        nspin1_es = run_esolver_subprocess(nspin1_es_dir)
        assert nspin1_es['converged'], \
            f"nspin=1 ESolver did not converge after {nspin1_es['niter']} iters"

        nspin2_es_dir = tmp_path / "nspin2_esolver"
        write_test_directory(nspin2_data, nspin2_es_dir)
        nspin2_es = run_esolver_subprocess(nspin2_es_dir)
        assert nspin2_es['converged'], \
            f"nspin=2 ESolver did not converge after {nspin2_es['niter']} iters"

        # --- Driver references ---
        nspin1_dr_dir = tmp_path / "nspin1_driver"
        write_test_directory(nspin1_data, nspin1_dr_dir)
        nspin1_dr = run_driver_subprocess(nspin1_dr_dir)
        assert nspin1_dr['converged'], "nspin=1 driver did not converge"

        nspin2_dr_dir = tmp_path / "nspin2_driver"
        write_test_directory(nspin2_data, nspin2_dr_dir)
        nspin2_dr = run_driver_subprocess(nspin2_dr_dir)
        assert nspin2_dr['converged'], "nspin=2 driver did not converge"

        # --- Compare nspin=1 ---
        nspin1_ediff = abs(nspin1_es['energy_ev'] - nspin1_dr['energy_ev'])
        print(f"\n=== Sequential: nspin=1 (run 1) ===")
        print(f"ESolver: E={nspin1_es['energy_ev']:.8f} eV, "
              f"niter={nspin1_es['niter']}")
        print(f"Driver:  E={nspin1_dr['energy_ev']:.8f} eV, "
              f"niter={nspin1_dr['niter']}")
        print(f"Energy diff: {nspin1_ediff:.2e} eV")

        assert nspin1_ediff < 1e-4, \
            f"nspin=1 energy mismatch: {nspin1_ediff:.2e} eV > 1e-4 eV"
        assert nspin1_es['niter'] == nspin1_dr['niter'], \
            f"nspin=1 niter mismatch: {nspin1_es['niter']} vs {nspin1_dr['niter']}"

        # --- Compare nspin=2 ---
        nspin2_ediff = abs(nspin2_es['energy_ev'] - nspin2_dr['energy_ev'])
        print(f"\n=== Sequential: nspin=2 (run 2) ===")
        print(f"ESolver: E={nspin2_es['energy_ev']:.8f} eV, "
              f"niter={nspin2_es['niter']}")
        print(f"Driver:  E={nspin2_dr['energy_ev']:.8f} eV, "
              f"niter={nspin2_dr['niter']}")
        print(f"Energy diff: {nspin2_ediff:.2e} eV")

        assert nspin2_ediff < 1e-4, \
            f"nspin=2 energy mismatch: {nspin2_ediff:.2e} eV > 1e-4 eV"
        assert nspin2_es['niter'] == nspin2_dr['niter'], \
            f"nspin=2 niter mismatch: {nspin2_es['niter']} vs {nspin2_dr['niter']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])
