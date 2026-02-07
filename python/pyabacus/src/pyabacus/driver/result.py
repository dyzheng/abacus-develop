"""
Result dataclass for ABACUS calculations.

This module contains the CalculationResult dataclass and unit conversion constants.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import os

from ..constants import RY_TO_EV, BOHR_TO_ANG


@dataclass
class CalculationResult:
    """
    Container for ABACUS calculation results.

    All energies are stored in eV units.

    Attributes
    ----------
    converged : bool
        Whether SCF converged
    niter : int
        Number of SCF iterations
    etot : float
        Total energy in eV
    forces : np.ndarray, optional
        Forces on atoms (nat, 3) in eV/Angstrom
    stress : np.ndarray, optional
        Stress tensor (3, 3) in kbar
    energies : dict
        Dictionary of energy components (all in eV)
    fermi_energy : float
        Fermi energy in eV
    bandgap : float
        Band gap in eV
    nat : int
        Number of atoms
    ntype : int
        Number of atom types
    nbands : int
        Number of bands
    nks : int
        Number of k-points
    """
    # Convergence info
    converged: bool = False
    niter: int = 0
    drho: float = 0.0

    # Energies (all in eV)
    etot: float = 0.0
    eband: float = 0.0
    hartree_energy: float = 0.0
    etxc: float = 0.0
    ewald_energy: float = 0.0
    demet: float = 0.0
    exx: float = 0.0
    evdw: float = 0.0

    # Forces (in eV/Angstrom) and stress (in kbar)
    forces: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None

    # Electronic structure info
    fermi_energy: float = 0.0  # in eV
    bandgap: float = 0.0       # in eV

    # System info
    nat: int = 0
    ntype: int = 0
    nbands: int = 0
    nks: int = 0

    # Output file tracking
    output_dir: str = ""  # Path to OUT.$suffix folder
    log_file: str = ""    # Path to the main log file (running_*.log)
    output_files: Dict[str, str] = field(default_factory=dict)  # filename -> full path

    @property
    def etot_ev(self) -> float:
        """Total energy in eV (same as etot, for compatibility)."""
        return self.etot

    @property
    def energies(self) -> Dict[str, float]:
        """Dictionary of all energy components (all in eV)."""
        return {
            'etot': self.etot,
            'eband': self.eband,
            'hartree_energy': self.hartree_energy,
            'etxc': self.etxc,
            'ewald_energy': self.ewald_energy,
            'demet': self.demet,
            'exx': self.exx,
            'evdw': self.evdw,
        }

    @property
    def forces_ev_ang(self) -> Optional[np.ndarray]:
        """Forces in eV/Angstrom (same as forces, for compatibility)."""
        return self.forces

    @property
    def has_forces(self) -> bool:
        """Whether forces are available."""
        return self.forces is not None

    @property
    def has_stress(self) -> bool:
        """Whether stress is available."""
        return self.stress is not None

    @property
    def has_output_dir(self) -> bool:
        """Whether output directory exists and is set."""
        return bool(self.output_dir) and os.path.isdir(self.output_dir)

    def get_output_file(self, filename: str) -> Optional[str]:
        """
        Get full path to a specific output file.

        Parameters
        ----------
        filename : str
            Name of the output file (e.g., 'running_scf.log', 'BANDS_1.dat')

        Returns
        -------
        str or None
            Full path to the file if it exists, None otherwise
        """
        return self.output_files.get(filename)

    def list_output_files(self) -> List[str]:
        """
        List all output file names.

        Returns
        -------
        list of str
            List of output file names
        """
        return list(self.output_files.keys())

    def summary(self) -> str:
        """Return a summary string of the calculation result."""
        lines = [
            "=== ABACUS Calculation Result ===",
            f"Converged: {'Yes' if self.converged else 'No'}",
            f"SCF iterations: {self.niter}",
            f"Final drho: {self.drho:.2e}",
            "",
            "Energies (eV):",
            f"  Total energy: {self.etot:.8f}",
            f"  Band energy:  {self.eband:.8f}",
            f"  Hartree:      {self.hartree_energy:.8f}",
            f"  XC energy:    {self.etxc:.8f}",
            f"  Ewald:        {self.ewald_energy:.8f}",
            f"  Entropy(-TS): {self.demet:.8f}",
            f"  EXX:          {self.exx:.8f}",
            f"  VdW:          {self.evdw:.8f}",
        ]

        lines.extend([
            "",
            "System info:",
            f"  Atoms: {self.nat}, Types: {self.ntype}",
            f"  Bands: {self.nbands}, K-points: {self.nks}",
            f"  Fermi energy: {self.fermi_energy:.6f} eV",
            f"  Band gap: {self.bandgap:.6f} eV",
        ])

        lines.append("")
        lines.append("Forces (eV/Angstrom):")
        if self.has_forces and self.forces is not None:
            max_force = np.max(np.abs(self.forces))
            lines.append(f"  Calculated ({self.nat} atoms), Max force: {max_force:.6f}")
            for i, f in enumerate(self.forces):
                lines.append(f"    Atom {i+1}: [{f[0]:12.8f}, {f[1]:12.8f}, {f[2]:12.8f}]")
        else:
            lines.append("  Not calculated")

        lines.append("")
        lines.append("Stress (kbar):")
        if self.has_stress and self.stress is not None:
            lines.append("  Calculated:")
            for i, row in enumerate(self.stress):
                lines.append(f"    [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}]")
        else:
            lines.append("  Not calculated")

        # Output file tracking
        lines.extend([
            "",
            "Output:",
            f"  Directory: {self.output_dir if self.output_dir else 'N/A'}",
            f"  Log file: {os.path.basename(self.log_file) if self.log_file else 'N/A'}",
            f"  Files: {len(self.output_files)} output files",
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<CalculationResult converged={self.converged} "
            f"etot={self.etot:.6f} eV>"
        )
