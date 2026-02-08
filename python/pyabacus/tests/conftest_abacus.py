"""
ABACUS test utilities and fixtures.

This module provides common utilities for setting up ABACUS calculations:
1. Setup structure - copy STRU file, configure INPUT with correct paths
2. Execute calculation and verify convergence
3. Extract results for comparison
"""

import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Initialize MPI before any other imports
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# Path configuration
# __file__ is at python/pyabacus/tests/conftest_abacus.py
# Need 4 parents to get to abacus-develop root
ABACUS_ROOT = Path(__file__).parent.parent.parent.parent
PP_ORB_DIR = ABACUS_ROOT / 'tests' / 'PP_ORB'
GAMMA_TEST_CASE = ABACUS_ROOT / 'tests' / '02_NAO_Gamma' / '001_NO_GO_OHK'
MULTIK_TEST_CASE = ABACUS_ROOT / 'tests' / '03_NAO_multik' / '02_NO_KP_15'
GAMMA_SPIN2_TEST_CASE = ABACUS_ROOT / 'tests' / '02_NAO_Gamma' / '002_NO_GO_OHK2'

# Unit conversion constants
RY_TO_EV = 13.605698  # Must match source/source_base/constants.h


@dataclass
class ABACUSTestConfig:
    """Configuration for an ABACUS test calculation."""
    test_case_dir: Path
    gamma_only: bool = True
    input_modifications: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.input_modifications is None:
            self.input_modifications = {}


def setup_calculation_directory(
    src_dir: Path,
    dst_dir: Path,
    input_modifications: Optional[Dict[str, str]] = None
) -> bool:
    """
    Setup ABACUS calculation directory with proper file paths.

    Workflow:
    1. Copy STRU, KPT files from source test case
    2. Create INPUT file with correct pseudo_dir and orbital_dir paths
    3. Apply any additional INPUT modifications

    Parameters
    ----------
    src_dir : Path
        Source test case directory
    dst_dir : Path
        Destination directory for calculation
    input_modifications : dict, optional
        Additional INPUT parameters to modify

    Returns
    -------
    bool
        True if setup successful, False otherwise
    """
    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        return False

    # Ensure destination exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy structure files (STRU, KPT)
    for filename in ['STRU', 'KPT']:
        src_file = src_dir / filename
        if src_file.exists():
            shutil.copy(src_file, dst_dir / filename)
        elif filename == 'STRU':
            print(f"Error: Required file {filename} not found in {src_dir}")
            return False

    # Copy dmrs1_nao.csr if exists (for restart tests)
    dmr_file = src_dir / 'dmrs1_nao.csr'
    if dmr_file.exists():
        shutil.copy(dmr_file, dst_dir / 'dmrs1_nao.csr')

    # Step 2: Create INPUT file with correct paths
    src_input = src_dir / 'INPUT'
    if not src_input.exists():
        print(f"Error: INPUT file not found in {src_dir}")
        return False

    # Read original INPUT
    with open(src_input, 'r') as f:
        lines = f.readlines()

    # Parse and modify INPUT parameters
    modifications = {
        'pseudo_dir': str(PP_ORB_DIR),
        'orbital_dir': str(PP_ORB_DIR),
    }
    if input_modifications:
        modifications.update(input_modifications)

    # Process each line
    new_lines = []
    modified_keys = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            new_lines.append(line)
            continue

        parts = stripped.split()
        if not parts:
            new_lines.append(line)
            continue

        key = parts[0]
        if key in modifications:
            new_lines.append(f"{key}    {modifications[key]}\n")
            modified_keys.add(key)
        else:
            new_lines.append(line)

    # Add any modifications that weren't in original file
    for key, value in modifications.items():
        if key not in modified_keys:
            new_lines.append(f"{key}    {value}\n")

    # Write modified INPUT
    with open(dst_dir / 'INPUT', 'w') as f:
        f.writelines(new_lines)

    # Create output directory based on suffix parameter
    # ABACUS creates OUT.{suffix}/ directory for output files
    suffix = 'autotest'  # default
    for line in new_lines:
        stripped = line.strip()
        if stripped.startswith('suffix'):
            parts = stripped.split()
            if len(parts) >= 2:
                suffix = parts[1]
                break
    out_dir = dst_dir / f'OUT.{suffix}'
    out_dir.mkdir(exist_ok=True)

    return True


def verify_setup(work_dir: Path) -> bool:
    """Verify that calculation directory is properly set up."""
    required_files = ['INPUT', 'STRU']
    for filename in required_files:
        if not (work_dir / filename).exists():
            print(f"Error: Missing required file: {filename}")
            return False

    # Verify INPUT has valid pseudo_dir and orbital_dir
    with open(work_dir / 'INPUT', 'r') as f:
        content = f.read()

    if 'pseudo_dir' not in content:
        print("Error: pseudo_dir not set in INPUT")
        return False

    return True


class WorkingDirectory:
    """Context manager for changing working directory."""

    def __init__(self, path: Path):
        self.path = path
        self.original_dir = None

    def __enter__(self):
        self.original_dir = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_dir:
            os.chdir(self.original_dir)
        return False
