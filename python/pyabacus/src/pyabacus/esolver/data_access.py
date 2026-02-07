"""
Data access mixin for LCAOWorkflow.

This module provides the DataAccessMixin class that handles
data access properties and methods for the workflow.
"""

from typing import TYPE_CHECKING
import numpy as np

from .data_types import (
    ChargeData,
    EnergyData,
    HamiltonianData,
    DensityMatrixData,
    ForceData,
    StressData,
)

if TYPE_CHECKING:
    pass


class DataAccessMixin:
    """
    Mixin class providing data access functionality.

    This mixin provides properties and methods for accessing
    calculation data like charge, energy, Hamiltonian, etc.
    """

    @property
    def charge(self) -> ChargeData:
        """
        Get current charge density.

        Returns
        -------
        ChargeData
            Charge density data container
        """
        accessor = self._esolver.get_charge()
        if hasattr(accessor, 'is_valid') and not accessor.is_valid():
            return ChargeData(rho=np.array([]), nspin=0, nrxx=0)

        kwargs = dict(
            rho=accessor.get_rho(),
            nspin=accessor.nspin,
            nrxx=accessor.nrxx,
        )
        if hasattr(accessor, 'get_rhog'):
            kwargs['rhog'] = accessor.get_rhog()
        if hasattr(accessor, 'ngmc'):
            kwargs['ngmc'] = accessor.ngmc
        return ChargeData(**kwargs)

    @property
    def energy(self) -> EnergyData:
        """
        Get current energy data.

        Returns
        -------
        EnergyData
            Energy data container with all energy components
        """
        accessor = self._esolver.get_energy()
        return EnergyData(
            etot=accessor.etot,
            eband=accessor.eband,
            hartree_energy=accessor.hartree_energy,
            etxc=accessor.etxc,
            ewald_energy=accessor.ewald_energy,
            demet=accessor.demet,
            exx=accessor.exx,
            evdw=accessor.evdw,
        )

    @property
    def hamiltonian(self) -> HamiltonianData:
        """
        Get current Hamiltonian matrices.

        Returns
        -------
        HamiltonianData
            Hamiltonian data container with H(k), S(k), H(R), S(R)
        """
        accessor = self._esolver.get_hamiltonian()
        if not accessor.is_valid():
            return HamiltonianData()

        nks = accessor.nks
        Hk = [accessor.get_Hk(ik) for ik in range(nks)]
        Sk = [accessor.get_Sk(ik) for ik in range(nks)]

        return HamiltonianData(
            Hk=Hk,
            Sk=Sk,
            HR=accessor.get_HR(),
            SR=accessor.get_SR(),
            nbasis=accessor.nbasis,
            nks=nks,
        )

    @property
    def density_matrix(self) -> DensityMatrixData:
        """
        Get current density matrix.

        Returns
        -------
        DensityMatrixData
            Density matrix data container with DM(k) and DM(R)
        """
        accessor = self._esolver.get_density_matrix()
        if not accessor.is_valid():
            return DensityMatrixData()

        nks = accessor.nks
        DMK = [accessor.get_DMK(ik) for ik in range(nks)]

        return DensityMatrixData(
            DMK=DMK,
            DMR=accessor.get_DMR(),
            nks=nks,
            nrow=accessor.nrow,
            ncol=accessor.ncol,
        )

    @property
    def force(self) -> ForceData:
        """
        Get force data (call cal_force first).

        Returns
        -------
        ForceData
            Force data container with forces in Ry/Bohr
        """
        accessor = self._esolver.get_force()
        return ForceData(
            forces=accessor.get_forces(),
            nat=accessor.nat,
        )

    @property
    def stress(self) -> StressData:
        """
        Get stress data (call cal_stress first).

        Returns
        -------
        StressData
            Stress data container with stress tensor in kbar
        """
        accessor = self._esolver.get_stress()
        return StressData(stress=accessor.get_stress())

    @property
    def is_converged(self) -> bool:
        """Check if SCF is converged."""
        return self._esolver.is_converged()

    @property
    def niter(self) -> int:
        """Get current iteration number."""
        return self._esolver.niter

    @property
    def drho(self) -> float:
        """Get current charge density difference."""
        return self._esolver.drho

    @property
    def nks(self) -> int:
        """Get number of k-points."""
        return self._esolver.nks

    @property
    def nbasis(self) -> int:
        """Get number of basis functions."""
        return self._esolver.nbasis

    @property
    def nbands(self) -> int:
        """Get number of bands."""
        return self._esolver.nbands

    @property
    def nspin(self) -> int:
        """Get number of spin channels."""
        return self._esolver.nspin

    @property
    def nat(self) -> int:
        """Get number of atoms."""
        return self._esolver.nat

    def get_psi(self, ik: int) -> np.ndarray:
        """
        Get wave function coefficients for k-point ik.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        np.ndarray
            Wave function coefficients with shape (nbands, nbasis)
        """
        return self._esolver.get_psi(ik)

    def get_eigenvalues(self, ik: int) -> np.ndarray:
        """
        Get eigenvalues for k-point ik.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        np.ndarray
            Eigenvalues with shape (nbands,)
        """
        return self._esolver.get_eigenvalues(ik)

    def get_occupations(self, ik: int) -> np.ndarray:
        """
        Get occupation numbers for k-point ik.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        np.ndarray
            Occupation numbers with shape (nbands,)
        """
        return self._esolver.get_occupations(ik)

    def get_kvec(self, ik: int) -> np.ndarray:
        """
        Get k-vector in direct coordinates.

        Parameters
        ----------
        ik : int
            K-point index

        Returns
        -------
        np.ndarray
            K-vector with shape (3,)
        """
        return self._esolver.get_kvec_d(ik)

    def get_kweights(self) -> np.ndarray:
        """
        Get k-point weights.

        Returns
        -------
        np.ndarray
            K-point weights with shape (nks,)
        """
        return self._esolver.get_wk()
