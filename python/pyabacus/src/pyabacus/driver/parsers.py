"""
Log parsing functions for ABACUS output files.

This module contains functions to parse ABACUS running log files
and extract calculation results.
"""

from typing import Optional, Dict
import numpy as np
import os
import re

from .result import CalculationResult


def parse_running_log(log_path: str) -> CalculationResult:
    """Parse the running log file to extract calculation results (all energies in eV)."""
    result = CalculationResult()

    if not os.path.exists(log_path):
        return result

    with open(log_path, 'r') as f:
        content = f.read()

    # Parse convergence - check multiple patterns
    convergence_patterns = [
        r"#SCF IS CONVERGED#",
        r"charge density convergence is achieved",
        r"convergence is achieved",
        r"SCF CONVERGED",
    ]
    for pattern in convergence_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            result.converged = True
            break

    # Parse total energy (look for final energy first)
    # Pattern 1: !FINAL_ETOT_IS -57.02190809937956 eV
    final_match = re.search(r"!FINAL_ETOT_IS\s+([-\d.]+)\s+eV", content)
    if final_match:
        result.etot = float(final_match.group(1))  # Already in eV
    else:
        # Pattern 2: E_KohnSham lines - get the last one (Ry, eV)
        ks_matches = re.findall(r"E_KohnSham\s+([-\d.]+)\s+([-\d.]+)", content)
        if ks_matches:
            result.etot = float(ks_matches[-1][1])  # Second value is in eV

    # Parse number of SCF iterations - count ELEC ITER lines
    iter_matches = re.findall(r"#ELEC ITER#\s+(\d+)", content)
    if iter_matches:
        result.niter = int(iter_matches[-1])

    # Parse drho - get the last value (Electron density deviation)
    drho_matches = re.findall(r"Electron density deviation\s+([\d.eE+-]+)", content)
    if drho_matches:
        result.drho = float(drho_matches[-1])

    # Parse number of atoms
    nat_match = re.search(r"TOTAL ATOM NUMBER\s*=\s*(\d+)", content)
    if nat_match:
        result.nat = int(nat_match.group(1))

    # Parse number of types - count "READING ATOM TYPE" lines
    ntype_matches = re.findall(r"READING ATOM TYPE\s+(\d+)", content)
    if ntype_matches:
        result.ntype = int(ntype_matches[-1])

    # Parse number of bands
    nbands_match = re.search(r"Number of electronic states \(NBANDS\)\s*=\s*(\d+)", content)
    if nbands_match:
        result.nbands = int(nbands_match.group(1))

    # Parse number of k-points (nkstot now = X after reduction)
    nkstot_match = re.search(r"nkstot now\s*=\s*(\d+)", content)
    if nkstot_match:
        result.nks = int(nkstot_match.group(1))
    else:
        # Fallback to original nkstot
        nkstot_match = re.search(r"nkstot\s*=\s*(\d+)", content)
        if nkstot_match:
            result.nks = int(nkstot_match.group(1))

    # Parse Fermi energy - format: E_Fermi  0.4657978215  6.3375044881 (Ry, eV)
    fermi_match = re.search(r"E_Fermi\s+([-\d.]+)\s+([-\d.]+)", content)
    if fermi_match:
        result.fermi_energy = float(fermi_match.group(2))  # Second value is in eV

    # Parse band gap - format: E_gap(k)  0.1070873708  1.4569984261 (Ry, eV)
    gap_match = re.search(r"E_gap\(k\)\s+([-\d.]+)\s+([-\d.]+)", content)
    if gap_match:
        result.bandgap = float(gap_match.group(2))  # Second value is in eV

    # Parse energy components from the final SCF iteration
    # Format: E_xxx  value_Ry  value_eV - we take the eV value (second column)

    # E_band (band energy)
    eband_match = re.search(r"E_band\s+([-\d.]+)\s+([-\d.]+)", content)
    if eband_match:
        result.eband = float(eband_match.group(2))  # eV

    # E_Hartree
    hartree_match = re.search(r"E_Hartree\s+([-\d.]+)\s+([-\d.]+)", content)
    if hartree_match:
        result.hartree_energy = float(hartree_match.group(2))  # eV

    # E_xc (exchange-correlation)
    etxc_match = re.search(r"E_xc\s+([-\d.]+)\s+([-\d.]+)", content)
    if etxc_match:
        result.etxc = float(etxc_match.group(2))  # eV

    # E_Ewald
    ewald_match = re.search(r"E_Ewald\s+([-\d.]+)\s+([-\d.]+)", content)
    if ewald_match:
        result.ewald_energy = float(ewald_match.group(2))  # eV

    # E_entropy(-TS) for metals
    demet_match = re.search(r"E_entropy\(-TS\)\s+([-\d.]+)\s+([-\d.]+)", content)
    if demet_match:
        result.demet = float(demet_match.group(2))  # eV

    # E_exx (exact exchange)
    exx_match = re.search(r"E_exx\s+([-\d.]+)\s+([-\d.]+)", content)
    if exx_match:
        result.exx = float(exx_match.group(2))  # eV

    return result


def get_suffix_from_input(input_dir: str) -> str:
    """Parse the suffix from INPUT file."""
    input_file = os.path.join(input_dir, "INPUT")
    suffix = "ABACUS"  # default suffix

    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                # Parse suffix parameter
                if 'suffix' in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        suffix = parts[1]
                        break
    return suffix


def collect_output_files(output_dir: str) -> Dict[str, str]:
    """
    Collect all output files from the output directory.

    Parameters
    ----------
    output_dir : str
        Path to the output directory (OUT.$suffix)

    Returns
    -------
    dict
        Dictionary mapping filename to full path
    """
    output_files = {}
    if output_dir and os.path.isdir(output_dir):
        try:
            for entry in os.listdir(output_dir):
                full_path = os.path.join(output_dir, entry)
                if os.path.isfile(full_path):
                    output_files[entry] = full_path
        except OSError:
            pass  # Ignore errors during directory iteration
    return output_files


def parse_forces_from_log(log_path: str, nat: int) -> Optional[np.ndarray]:
    """Parse forces from the running log file (returns forces in eV/Angstrom)."""
    if not os.path.exists(log_path) or nat <= 0:
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Try multiple force block formats
    # Format 1: #TOTAL-FORCE (eV/Angstrom)#
    #           -------------------------------------------------------------------------
    #               Atoms              Force_x              Force_y              Force_z
    #           -------------------------------------------------------------------------
    #                 Al1         0.0000000000         0.0000000000         0.0000000000
    force_pattern1 = r"#TOTAL-FORCE \(eV/Angstrom\)#.*?-{10,}\s*\n\s*Atoms\s+Force_x\s+Force_y\s+Force_z\s*\n\s*-{10,}\s*\n((?:\s*\S+\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n)+)"
    match = re.search(force_pattern1, content, re.DOTALL)

    if not match:
        # Format 2: TOTAL-FORCE (eV/Angstrom)
        #           ----------------------------
        #            atom    x       y       z
        #             Si1  0.001   0.002   0.003
        force_pattern2 = r"TOTAL-FORCE \(eV/Angstrom\).*?-{10,}\s*\n\s*atom.*?\n((?:\s*\S+\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n)+)"
        match = re.search(force_pattern2, content, re.DOTALL)

    if match:
        force_lines = match.group(1).strip().split('\n')
        forces = []
        for line in force_lines:
            parts = line.split()
            if len(parts) >= 4:
                # parts[0] is atom label, parts[1:4] are fx, fy, fz in eV/Angstrom
                try:
                    fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                    forces.append([fx, fy, fz])  # Already in eV/Angstrom
                except (ValueError, IndexError):
                    continue

        if len(forces) == nat:
            return np.array(forces)

    return None


def parse_stress_from_log(log_path: str) -> Optional[np.ndarray]:
    """Parse stress tensor from the running log file (returns stress in kbar)."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Try multiple stress block formats
    # Format 1: #TOTAL-STRESS (kbar)#
    #           ----------------------------------------------------------------
    #                    Stress_x             Stress_y             Stress_z
    #           ----------------------------------------------------------------
    #               15.7976835472         0.0000000000         0.0000000000
    stress_pattern1 = r"#TOTAL-STRESS \(kbar\)#.*?-{10,}\s*\n\s*Stress_x\s+Stress_y\s+Stress_z\s*\n\s*-{10,}\s*\n((?:\s*[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n){3})"
    match = re.search(stress_pattern1, content, re.DOTALL | re.IGNORECASE)

    if not match:
        # Format 2: TOTAL-STRESS (KBAR)
        #           ----------------------------
        #             1.234   0.000   0.000
        stress_pattern2 = r"TOTAL-STRESS \(KBAR\).*?-{10,}\s*\n((?:\s*[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n){3})"
        match = re.search(stress_pattern2, content, re.DOTALL | re.IGNORECASE)

    if match:
        stress_lines = match.group(1).strip().split('\n')
        stress = []
        for line in stress_lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    stress.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except (ValueError, IndexError):
                    continue

        if len(stress) == 3:
            return np.array(stress)

    return None
