"""
Subprocess runner for ABACUS calculations.

This module contains functions to run ABACUS via subprocess
when C++ bindings are not available.
"""

from typing import Optional
import os
import subprocess

from .result import CalculationResult
from .parsers import (
    parse_running_log,
    get_suffix_from_input,
    collect_output_files,
    parse_forces_from_log,
    parse_stress_from_log,
)


def find_abacus_executable() -> Optional[str]:
    """Find the abacus executable in PATH or common locations."""
    import shutil

    # Check PATH first
    abacus_path = shutil.which("abacus")
    if abacus_path:
        return abacus_path

    # Check common locations
    common_paths = [
        "/usr/local/bin/abacus",
        "/usr/bin/abacus",
        os.path.expanduser("~/abacus/build/abacus"),
        os.path.expanduser("~/.local/bin/abacus"),
    ]

    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def modify_input_file(input_dir: str, calculate_force: bool, calculate_stress: bool) -> Optional[str]:
    """
    Modify INPUT file to add cal_force and cal_stress parameters.

    Returns the path to the backup file if modifications were made, None otherwise.
    """
    input_file = os.path.join(input_dir, "INPUT")
    if not os.path.exists(input_file):
        return None

    # Read original content
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Check if cal_force/cal_stress already exist
    has_cal_force = False
    has_cal_stress = False
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith('cal_force'):
            has_cal_force = True
        if line_lower.startswith('cal_stress'):
            has_cal_stress = True

    # If both already exist, no need to modify
    if has_cal_force and has_cal_stress:
        return None

    # Create backup
    backup_file = input_file + ".pyabacus_backup"
    with open(backup_file, 'w') as f:
        f.writelines(lines)

    # Add missing parameters
    new_lines = lines.copy()
    if not has_cal_force:
        new_lines.append(f"cal_force {1 if calculate_force else 0}\n")
    if not has_cal_stress:
        new_lines.append(f"cal_stress {1 if calculate_stress else 0}\n")

    # Write modified file
    with open(input_file, 'w') as f:
        f.writelines(new_lines)

    return backup_file


def restore_input_file(input_dir: str, backup_file: Optional[str]):
    """Restore INPUT file from backup."""
    if backup_file is None:
        return

    input_file = os.path.join(input_dir, "INPUT")
    if os.path.exists(backup_file):
        # Restore original
        with open(backup_file, 'r') as f:
            content = f.read()
        with open(input_file, 'w') as f:
            f.write(content)
        # Remove backup
        os.remove(backup_file)


def run_abacus_subprocess(
    input_dir: str,
    output_dir: str,
    verbosity: int,
    calculate_force: bool = True,
    calculate_stress: bool = False,
    nprocs: int = 1,
    nthreads: int = 1,
) -> CalculationResult:
    """Run ABACUS using subprocess and parse results."""
    import shutil

    # Find abacus executable
    abacus_exe = find_abacus_executable()
    if abacus_exe is None:
        raise RuntimeError(
            "ABACUS executable not found. Please ensure 'abacus' is in your PATH "
            "or install ABACUS from https://github.com/deepmodeling/abacus-develop"
        )

    # Convert to absolute path
    input_dir = os.path.abspath(input_dir)

    # Modify INPUT file to add cal_force and cal_stress
    backup_file = modify_input_file(input_dir, calculate_force, calculate_stress)

    try:
        # Get the suffix from INPUT file to know where output will be
        suffix = get_suffix_from_input(input_dir)
        expected_out_dir = os.path.join(input_dir, f"OUT.{suffix}")

        # Set up environment
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(nthreads)

        # Build command
        if nprocs > 1:
            # Find mpirun or mpiexec
            mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
            if mpirun is None:
                raise RuntimeError(
                    f"MPI requested (nprocs={nprocs}) but mpirun/mpiexec not found. "
                    "Please install MPI or set nprocs=1."
                )
            cmd = [mpirun, "-np", str(nprocs), abacus_exe]
        else:
            cmd = [abacus_exe]

        # Set up stdout/stderr based on verbosity
        if verbosity >= 2:
            stdout = None
            stderr = None
        elif verbosity == 1:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        try:
            proc = subprocess.run(
                cmd,
                cwd=input_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=None,  # No timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ABACUS calculation timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to run ABACUS: {e}")

        # Find and parse the output
        # First try the expected output directory based on suffix
        log_path = None
        if os.path.exists(expected_out_dir):
            for log_name in ["running_scf.log", "running_relax.log", "running_cell-relax.log", "running_nscf.log"]:
                candidate = os.path.join(expected_out_dir, log_name)
                if os.path.exists(candidate):
                    log_path = candidate
                    break

        # Fallback: find the most recently modified OUT.* directory
        if log_path is None:
            out_dirs = [d for d in os.listdir(input_dir) if d.startswith("OUT.") and os.path.isdir(os.path.join(input_dir, d))]
            if out_dirs:
                latest_out = max(out_dirs, key=lambda d: os.path.getmtime(os.path.join(input_dir, d)))
                out_dir_path = os.path.join(input_dir, latest_out)
                for log_name in ["running_scf.log", "running_relax.log", "running_cell-relax.log", "running_nscf.log"]:
                    candidate = os.path.join(out_dir_path, log_name)
                    if os.path.exists(candidate):
                        log_path = candidate
                        break

        if log_path and os.path.exists(log_path):
            result = parse_running_log(log_path)
            # Set output tracking fields
            result.log_file = os.path.abspath(log_path)
            result.output_dir = os.path.abspath(os.path.dirname(log_path))
            result.output_files = collect_output_files(result.output_dir)

            # Parse forces if requested
            if calculate_force and result.nat > 0:
                forces = parse_forces_from_log(log_path, result.nat)
                if forces is not None:
                    result.forces = forces

            # Parse stress if requested
            if calculate_stress:
                stress = parse_stress_from_log(log_path)
                if stress is not None:
                    result.stress = stress
        else:
            result = CalculationResult()
            # Try to find output directory even if log file wasn't found
            if os.path.exists(expected_out_dir):
                result.output_dir = os.path.abspath(expected_out_dir)
                result.output_files = collect_output_files(result.output_dir)

    finally:
        # Restore original INPUT file
        restore_input_file(input_dir, backup_file)

    return result
