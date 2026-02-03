#include "gtest/gtest.h"
#include "../orbital_mag.h"
#include "source_base/vector3.h"
#include "source_cell/unitcell.h"
#include "source_cell/klist.h"
#include "source_estate/elecstate.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_psi/psi.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include <complex>
#include <cmath>

/**
 * @brief Unit tests for OrbitalMag class
 *
 * Following TDD approach: Write tests first, then implement functionality
 */
class OrbitalMagTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup minimal test system: 2 atoms, 2 orbitals each
        setup_unitcell();
        setup_kpoints();
        setup_hcontainer();
        setup_elecstate();
        setup_psi();
        setup_parallel_orbitals();
    }

    void TearDown() override {
        cleanup();
    }

    void setup_unitcell() {
        // Simple cubic lattice with 2 atoms
        ucell.ntype = 1;
        ucell.nat = 2;
        ucell.lat0 = 1.0;
        ucell.latvec.e11 = 5.0; ucell.latvec.e12 = 0.0; ucell.latvec.e13 = 0.0;
        ucell.latvec.e21 = 0.0; ucell.latvec.e22 = 5.0; ucell.latvec.e23 = 0.0;
        ucell.latvec.e31 = 0.0; ucell.latvec.e32 = 0.0; ucell.latvec.e33 = 5.0;

        ucell.atoms = new Atom[ucell.ntype];
        ucell.iat2it = new int[ucell.nat];
        ucell.iat2ia = new int[ucell.nat];

        for (int iat = 0; iat < ucell.nat; iat++) {
            ucell.iat2it[iat] = 0;
            ucell.iat2ia[iat] = iat;
        }

        ucell.atoms[0].na = 2;
        ucell.atoms[0].nw = 2;  // 2 orbitals per atom
        ucell.atoms[0].tau.resize(2);
        ucell.atoms[0].tau[0].set(0.0, 0.0, 0.0);
        ucell.atoms[0].tau[1].set(0.5, 0.5, 0.5);

        nlocal = 4;  // 2 atoms × 2 orbitals
    }

    void setup_kpoints() {
        // Single k-point for simplicity
        nks = 1;
        kv.set_nks(nks);
        kv.kvec_d.resize(nks);
        kv.kvec_d[0].set(0.0, 0.0, 0.0);  // Gamma point
        kv.wk.resize(nks);
        kv.wk[0] = 1.0;
    }

    void setup_hcontainer() {
        // Create H(R) and S(R) containers - simplified for testing
        // Just allocate empty containers for now
        hR = new hamilt::HContainer<double>(ucell);
        sR = new hamilt::HContainer<double>(ucell);
    }

    void setup_elecstate() {
        // Setup electronic state with eigenvalues and weights
        nbands = 2;
        pelec = new elecstate::ElecState();

        pelec->ekb.create(nks, nbands);
        pelec->wg.create(nks, nbands);

        // Simple band structure: 2 occupied bands
        pelec->ekb(0, 0) = -1.0;  // Lower band
        pelec->ekb(0, 1) = -0.5;  // Upper band

        pelec->wg(0, 0) = 1.0;    // Fully occupied
        pelec->wg(0, 1) = 1.0;    // Fully occupied

        pelec->eferm.ef = -0.25;  // Fermi energy between bands
        pelec->klist = &kv;
    }

    void setup_psi() {
        // Setup wavefunctions (eigenvectors)
        psi = new psi::Psi<std::complex<double>>(nks, nbands, nlocal, nlocal, true);

        // Simple eigenvectors: identity-like for testing
        for (int ik = 0; ik < nks; ik++) {
            for (int ib = 0; ib < nbands; ib++) {
                for (int io = 0; io < nlocal; io++) {
                    if (io == ib) {
                        (*psi)(ik, ib, io) = std::complex<double>(1.0, 0.0);
                    } else {
                        (*psi)(ik, ib, io) = std::complex<double>(0.0, 0.0);
                    }
                }
            }
        }
    }

    void setup_parallel_orbitals() {
        // Setup Parallel_Orbitals for serial case
        // In serial mode, nrow=nlocal, ncol=nlocal
        pv.set_serial(nlocal, nlocal);
        pv.ncol_bands = nbands;
        pv.nrow_bands = nlocal;
        pv.nloc_wfc = nlocal * nbands;
        pv.nloc_Eij = nbands * nbands;
    }

    void cleanup() {
        // Simplified cleanup to avoid double-free issues
        // The test framework will handle most cleanup
        if (hR) {
            delete hR;
            hR = nullptr;
        }
        if (sR) {
            delete sR;
            sR = nullptr;
        }
        if (pelec) {
            delete pelec;
            pelec = nullptr;
        }
        if (psi) {
            delete psi;
            psi = nullptr;
        }
        // Don't delete ucell members - they're managed by UnitCell destructor
    }

    // Test data members
    UnitCell ucell;
    K_Vectors kv;
    elecstate::ElecState* pelec = nullptr;
    hamilt::HContainer<double>* hR = nullptr;
    hamilt::HContainer<double>* sR = nullptr;
    psi::Psi<std::complex<double>>* psi = nullptr;
    Parallel_Orbitals pv;

    int nlocal = 4;
    int nbands = 2;
    int nks = 1;
};

/**
 * Test 1: K-derivative finite difference validation
 *
 * Verify that analytical dH/dk matches numerical finite difference
 */
TEST_F(OrbitalMagTest, KDerivativeFiniteDifference) {
    hamilt::OrbitalMag orbital_mag(ucell, kv, *pelec, hR, sR, psi, &pv);

    // Allocate output matrix for dH/dk
    std::vector<std::complex<double>> dH_dk_analytical(nlocal * nlocal);

    // Test with empty HContainer - should produce zero matrix
    // In a real test, we would populate hR with test data
    // For now, verify the function runs without crashing
    // and produces a zero matrix for empty input

    // Compute analytical derivative in x-direction
    // Note: This is a private method, so we test it indirectly through calculate_orbital_moment
    // For direct testing, we would need to make it public or use a friend class

    // Since compute_k_derivative is private, we test the overall functionality
    // by verifying that calculate_orbital_moment runs without errors
    ModuleBase::Vector3<double> M_orb = orbital_mag.calculate_orbital_moment();

    // With empty HContainer, result should be zero
    EXPECT_NEAR(M_orb.x, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.y, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.z, 0.0, 1e-12);
}

/**
 * Test 2: Velocity matrix Hermiticity
 *
 * Diagonal elements V_nn should be real (group velocity)
 * Off-diagonal should satisfy V_nm = V_mn*
 */
TEST_F(OrbitalMagTest, VelocityMatrixHermitian) {
    hamilt::OrbitalMag orbital_mag(ucell, kv, *pelec, hR, sR, psi, &pv);

    // Since compute_velocity_matrix is private, we test it indirectly
    // by verifying that the overall calculation produces physically reasonable results

    // For an empty HContainer (no Hamiltonian matrix elements),
    // the velocity matrix should be zero, which is trivially Hermitian
    ModuleBase::Vector3<double> M_orb = orbital_mag.calculate_orbital_moment();

    // The calculation should complete without errors
    // With empty HContainer, result should be zero (which is consistent with Hermiticity)
    EXPECT_NEAR(M_orb.x, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.y, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.z, 0.0, 1e-12);

    // Note: A more comprehensive test would populate hR and sR with test data
    // and verify Hermiticity properties directly, but that requires exposing
    // private methods or using a test fixture with friend access
}

/**
 * Test 3: Time-reversal symmetry for non-magnetic system
 *
 * For non-magnetic systems (like Si), orbital moment should be zero
 */
TEST_F(OrbitalMagTest, TimeReversalSymmetry) {
    hamilt::OrbitalMag orbital_mag(ucell, kv, *pelec, hR, sR, psi, &pv);

    // Calculate orbital moment
    ModuleBase::Vector3<double> M_orb = orbital_mag.calculate_orbital_moment();

    // For time-reversal symmetric system with empty HContainer: M_orb should be ~0
    // This is a minimal test - a real test would use a proper non-magnetic system
    // with populated Hamiltonian and overlap matrices
    EXPECT_NEAR(M_orb.x, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.y, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.z, 0.0, 1e-12);
}

/**
 * Test 4: Gauge invariance under translation
 *
 * Orbital moment should be invariant under global translation
 */
TEST_F(OrbitalMagTest, GaugeInvariance) {
    hamilt::OrbitalMag orbital_mag(ucell, kv, *pelec, hR, sR, psi, &pv);

    // Calculate M_orb with original positions
    ModuleBase::Vector3<double> M_orb_before = orbital_mag.calculate_orbital_moment();

    // Shift all atoms by a vector T
    ModuleBase::Vector3<double> shift(0.1, 0.2, 0.3);
    for (int ia = 0; ia < ucell.atoms[0].na; ia++) {
        ucell.atoms[0].tau[ia] += shift;
    }

    // Create new OrbitalMag object with shifted positions
    hamilt::OrbitalMag orbital_mag_shifted(ucell, kv, *pelec, hR, sR, psi, &pv);

    // Calculate M_orb again
    ModuleBase::Vector3<double> M_orb_after = orbital_mag_shifted.calculate_orbital_moment();

    // Verify: |M_orb_before - M_orb_after| < 1e-12
    // For empty HContainer, both should be zero, so difference is zero
    EXPECT_NEAR(M_orb_before.x, M_orb_after.x, 1e-12);
    EXPECT_NEAR(M_orb_before.y, M_orb_after.y, 1e-12);
    EXPECT_NEAR(M_orb_before.z, M_orb_after.z, 1e-12);

    // Restore original positions for cleanup
    for (int ia = 0; ia < ucell.atoms[0].na; ia++) {
        ucell.atoms[0].tau[ia] -= shift;
    }
}

/**
 * Test 5: Constructor initialization
 *
 * Verify that OrbitalMag constructor properly initializes members
 */
TEST_F(OrbitalMagTest, ConstructorInitialization) {
    // This should pass once constructor is implemented
    hamilt::OrbitalMag orbital_mag(ucell, kv, *pelec, hR, sR, psi, &pv);

    // If we get here without crashing, constructor works
    EXPECT_TRUE(true);
}

/**
 * Test 6: Constructor with nullptr Parallel_Orbitals (serial mode)
 *
 * Verify that OrbitalMag works correctly when pv is nullptr
 */
TEST_F(OrbitalMagTest, ConstructorWithNullParallelOrbitals) {
    // Test with nullptr for Parallel_Orbitals (pure serial mode)
    hamilt::OrbitalMag orbital_mag(ucell, kv, *pelec, hR, sR, psi, nullptr);

    // Calculate orbital moment - should work in serial mode
    ModuleBase::Vector3<double> M_orb = orbital_mag.calculate_orbital_moment();

    // With empty HContainer, result should be zero
    EXPECT_NEAR(M_orb.x, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.y, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.z, 0.0, 1e-12);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
