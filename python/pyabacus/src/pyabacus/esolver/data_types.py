"""
Data types for PyABACUS ESolver module.

This module defines dataclasses for storing calculation results
in a structured and type-safe manner.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from ..constants import RY_TO_EV, BOHR_TO_ANG, RY_BOHR_TO_EV_ANG, KBAR_TO_EV_ANG3


@dataclass
class ChargeData:
    """
    Container for charge density data.

    Attributes
    ----------
    rho : np.ndarray
        Real-space charge density with shape (nspin, nrxx)
    rhog : np.ndarray, optional
        Reciprocal-space charge density with shape (nspin, ngmc)
    nspin : int
        Number of spin channels (1, 2, or 4)
    nrxx : int
        Number of real-space grid points
    ngmc : int, optional
        Number of G-vectors for charge density
    """
    rho: np.ndarray
    nspin: int
    nrxx: int
    rhog: Optional[np.ndarray] = None
    ngmc: Optional[int] = None

    def total_charge(self) -> float:
        """Calculate total charge by integrating rho."""
        return np.sum(self.rho)

    def spin_density(self) -> Optional[np.ndarray]:
        """
        Calculate spin density (rho_up - rho_down) for spin-polarized calculations.

        Returns None for non-spin-polarized calculations.
        """
        if self.nspin == 2:
            return self.rho[0] - self.rho[1]
        return None


@dataclass
class EnergyData:
    """
    Container for energy data.

    All energies are in Rydberg units.

    Attributes
    ----------
    etot : float
        Total energy
    eband : float
        Band (kinetic + local potential) energy
    hartree_energy : float
        Hartree (electron-electron Coulomb) energy
    etxc : float
        Exchange-correlation energy
    ewald_energy : float
        Ewald (ion-ion Coulomb) energy
    demet : float
        -TS term for metallic systems (smearing correction)
    exx : float
        Exact exchange energy (for hybrid functionals)
    evdw : float
        van der Waals correction energy
    """
    etot: float = 0.0
    eband: float = 0.0
    hartree_energy: float = 0.0
    etxc: float = 0.0
    ewald_energy: float = 0.0
    demet: float = 0.0
    exx: float = 0.0
    evdw: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
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

    def to_eV(self) -> 'EnergyData':
        """
        Convert all energies from Rydberg to eV.

        Returns a new EnergyData instance with energies in eV.
        """
        return EnergyData(
            etot=self.etot * RY_TO_EV,
            eband=self.eband * RY_TO_EV,
            hartree_energy=self.hartree_energy * RY_TO_EV,
            etxc=self.etxc * RY_TO_EV,
            ewald_energy=self.ewald_energy * RY_TO_EV,
            demet=self.demet * RY_TO_EV,
            exx=self.exx * RY_TO_EV,
            evdw=self.evdw * RY_TO_EV,
        )


@dataclass
class HamiltonianData:
    """
    Container for Hamiltonian matrix data.

    Attributes
    ----------
    Hk : List[np.ndarray]
        List of H(k) matrices for each k-point
    Sk : List[np.ndarray]
        List of S(k) overlap matrices for each k-point
    HR : Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray], optional
        H(R) in sparse format: {(iat1, iat2, (R1, R2, R3)): matrix}
    SR : Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray], optional
        S(R) in sparse format: {(iat1, iat2, (R1, R2, R3)): matrix}
    nbasis : int
        Number of basis functions
    nks : int
        Number of k-points
    """
    Hk: List[np.ndarray] = field(default_factory=list)
    Sk: List[np.ndarray] = field(default_factory=list)
    HR: Optional[Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray]] = None
    SR: Optional[Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray]] = None
    nbasis: int = 0
    nks: int = 0

    def get_Hk(self, ik: int) -> np.ndarray:
        """Get H(k) matrix for k-point ik."""
        if ik < 0 or ik >= len(self.Hk):
            raise IndexError(f"K-point index {ik} out of range [0, {len(self.Hk)})")
        return self.Hk[ik]

    def get_Sk(self, ik: int) -> np.ndarray:
        """Get S(k) matrix for k-point ik."""
        if ik < 0 or ik >= len(self.Sk):
            raise IndexError(f"K-point index {ik} out of range [0, {len(self.Sk)})")
        return self.Sk[ik]


@dataclass
class DensityMatrixData:
    """
    Container for density matrix data.

    Attributes
    ----------
    DMK : List[np.ndarray]
        List of DM(k) matrices for each k-point
    DMR : Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray], optional
        DM(R) in sparse format: {(iat1, iat2, (R1, R2, R3)): matrix}
    nks : int
        Number of k-points
    nrow : int
        Number of rows in density matrix
    ncol : int
        Number of columns in density matrix
    """
    DMK: List[np.ndarray] = field(default_factory=list)
    DMR: Optional[Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray]] = None
    nks: int = 0
    nrow: int = 0
    ncol: int = 0

    def get_DMK(self, ik: int) -> np.ndarray:
        """Get DM(k) matrix for k-point ik."""
        if ik < 0 or ik >= len(self.DMK):
            raise IndexError(f"K-point index {ik} out of range [0, {len(self.DMK)})")
        return self.DMK[ik]

    def trace(self, ik: int) -> complex:
        """Calculate trace of DM(k) for k-point ik."""
        return np.trace(self.get_DMK(ik))


@dataclass
class ForceData:
    """
    Container for force data.

    Forces are stored in Rydberg/Bohr units internally.

    Attributes
    ----------
    forces : np.ndarray
        Forces on atoms with shape (nat, 3) in Ry/Bohr
    nat : int
        Number of atoms
    """
    forces: np.ndarray
    nat: int

    def to_eV_Ang(self) -> np.ndarray:
        """
        Convert forces from Ry/Bohr to eV/Angstrom.

        Returns
        -------
        np.ndarray
            Forces in eV/Angstrom with shape (nat, 3)
        """
        return self.forces * RY_BOHR_TO_EV_ANG

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'forces': self.forces,
            'forces_eV_Ang': self.to_eV_Ang(),
            'nat': self.nat,
        }


@dataclass
class StressData:
    """
    Container for stress tensor data.

    Stress is stored in kbar units internally.

    Attributes
    ----------
    stress : np.ndarray
        Stress tensor with shape (3, 3) in kbar
    """
    stress: np.ndarray

    def to_voigt(self) -> np.ndarray:
        """
        Return stress in Voigt notation.

        Returns
        -------
        np.ndarray
            Stress in Voigt notation (6,): xx, yy, zz, yz, xz, xy
        """
        s = self.stress
        return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])

    def to_eV_Ang3(self) -> np.ndarray:
        """
        Convert stress to eV/Angstrom^3 in Voigt notation.

        This is the format expected by ASE.

        Returns
        -------
        np.ndarray
            Stress in eV/Å³ with Voigt notation (6,)
        """
        return self.to_voigt() * KBAR_TO_EV_ANG3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'stress': self.stress,
            'stress_voigt': self.to_voigt(),
            'stress_eV_Ang3': self.to_eV_Ang3(),
        }


@dataclass
class SCFResult:
    """
    Container for SCF calculation results.

    Attributes
    ----------
    converged : bool
        Whether SCF converged
    niter : int
        Number of iterations performed
    drho : float
        Final charge density difference
    energy : EnergyData
        Final energy data
    charge : ChargeData, optional
        Final charge density
    hamiltonian : HamiltonianData, optional
        Final Hamiltonian matrices
    density_matrix : DensityMatrixData, optional
        Final density matrix
    """
    converged: bool
    niter: int
    drho: float
    energy: EnergyData
    charge: Optional[ChargeData] = None
    hamiltonian: Optional[HamiltonianData] = None
    density_matrix: Optional[DensityMatrixData] = None

    def summary(self) -> str:
        """Return a summary string of the SCF result."""
        status = "converged" if self.converged else "not converged"
        return (
            f"SCF Result: {status}\n"
            f"  Iterations: {self.niter}\n"
            f"  Final drho: {self.drho:.2e}\n"
            f"  Total energy: {self.energy.etot:.8f} Ry "
            f"({self.energy.etot * RY_TO_EV:.8f} eV)"
        )
