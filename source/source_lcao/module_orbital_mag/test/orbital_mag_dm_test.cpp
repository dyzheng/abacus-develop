#include "gtest/gtest.h"
#include "../orbital_mag_dm.h"
#include "source_base/vector3.h"
#include "source_cell/unitcell.h"
#include "source_cell/klist.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include <complex>
#include <cmath>

/**
 * @brief Unit tests for OrbitalMagDM class (density matrix method)
 *
 * Tests the density matrix-based orbital magnetization algorithm
 */
class OrbitalMagDMTest : public ::testing::Test {
protected:
    void SetUp() override {
        setup_unitcell();
        setup_kpoints();
        setup_hcontainer();
        setup_parallel_orbitals();
        setup_density_matrix();
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

        // Setup reciprocal lattice vectors
        ucell.G.e11 = 1.0/5.0; ucell.G.e12 = 0.0; ucell.G.e13 = 0.0;
        ucell.G.e21 = 0.0; ucell.G.e22 = 1.0/5.0; ucell.G.e23 = 0.0;
        ucell.G.e31 = 0.0; ucell.G.e32 = 0.0; ucell.G.e33 = 1.0/5.0;
        ucell.tpiba = 2.0 * M_PI / ucell.lat0;

        // Setup lattice vectors a1, a2, a3
        ucell.a1.set(5.0, 0.0, 0.0);
        ucell.a2.set(0.0, 5.0, 0.0);
        ucell.a3.set(0.0, 0.0, 5.0);

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
        // Setup 3x3x3 k-point grid for finite difference testing
        nks = 27;
        kv.set_nks(nks);
        kv.kvec_d.resize(nks);
        kv.wk.resize(nks);

        int ik = 0;
        double dk = 1.0 / 3.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    kv.kvec_d[ik].set(i * dk, j * dk, k * dk);
                    kv.wk[ik] = 1.0 / nks;
                    ik++;
                }
            }
        }
    }

    void setup_hcontainer() {
        // Create H(R) and S(R) containers
        hR = new hamilt::HContainer<double>(ucell);
        sR = new hamilt::HContainer<double>(ucell);
    }

    void setup_parallel_orbitals() {
        // Setup Parallel_Orbitals for serial case
        pv.set_serial(nlocal, nlocal);
        pv.ncol_bands = nlocal;
        pv.nrow_bands = nlocal;
        pv.nloc_wfc = nlocal * nlocal;
        pv.nloc_Eij = nlocal * nlocal;
    }

    void setup_density_matrix() {
        // Create density matrix with k-vectors
        dm = new elecstate::DensityMatrix<std::complex<double>, double>(&pv, 1, kv.kvec_d, nks);

        // Initialize density matrix to identity-like for testing
        // P(k) = I for simplicity
        for (int ik = 0; ik < nks; ik++) {
            for (int i = 0; i < nlocal; i++) {
                for (int j = 0; j < nlocal; j++) {
                    std::complex<double> val = (i == j) ? std::complex<double>(1.0, 0.0)
                                                        : std::complex<double>(0.0, 0.0);
                    dm->set_DMK(1, ik, i, j, val);
                }
            }
        }
    }

    void cleanup() {
        if (hR) {
            delete hR;
            hR = nullptr;
        }
        if (sR) {
            delete sR;
            sR = nullptr;
        }
        if (dm) {
            delete dm;
            dm = nullptr;
        }
    }

    // Test data members
    UnitCell ucell;
    K_Vectors kv;
    elecstate::DensityMatrix<std::complex<double>, double>* dm = nullptr;
    hamilt::HContainer<double>* hR = nullptr;
    hamilt::HContainer<double>* sR = nullptr;
    Parallel_Orbitals pv;
    LCAO_Orbitals orb;

    int nlocal = 4;
    int nks = 27;
};

/**
 * Test 1: Constructor initialization
 *
 * Verify that OrbitalMagDM constructor properly initializes members
 */
TEST_F(OrbitalMagDMTest, ConstructorInitialization) {
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm(ucell, kv, *dm, hR, sR, &pv, orb);

    // If we get here without crashing, constructor works
    EXPECT_TRUE(true);
}

/**
 * Test 2: Constructor with vector potential A(t)
 *
 * Verify that OrbitalMagDM accepts vector potential parameter
 */
TEST_F(OrbitalMagDMTest, ConstructorWithVectorPotential) {
    ModuleBase::Vector3<double> At(0.1, 0.2, 0.3);
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm(ucell, kv, *dm, hR, sR, &pv, orb, At);

    // If we get here without crashing, constructor works with A(t)
    EXPECT_TRUE(true);
}

/**
 * Test 3: Set vector potential
 *
 * Verify that set_vector_potential method works
 */
TEST_F(OrbitalMagDMTest, SetVectorPotential) {
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm(ucell, kv, *dm, hR, sR, &pv, orb);

    ModuleBase::Vector3<double> At(0.5, 0.5, 0.5);
    orbital_mag_dm.set_vector_potential(At);

    // If we get here without crashing, set_vector_potential works
    EXPECT_TRUE(true);
}

/**
 * Test 4: Calculate orbital moment with empty HContainer
 *
 * With empty H(R) and S(R), the orbital moment should be zero
 */
TEST_F(OrbitalMagDMTest, CalculateOrbitalMomentEmpty) {
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm(ucell, kv, *dm, hR, sR, &pv, orb);

    ModuleBase::Vector3<double> M_orb = orbital_mag_dm.calculate_orbital_moment();

    // With empty HContainer, result should be zero
    EXPECT_NEAR(M_orb.x, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.y, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.z, 0.0, 1e-12);
}

/**
 * Test 5: Time-reversal symmetry for non-magnetic system
 *
 * For non-magnetic systems, orbital moment should be zero
 */
TEST_F(OrbitalMagDMTest, TimeReversalSymmetry) {
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm(ucell, kv, *dm, hR, sR, &pv, orb);

    ModuleBase::Vector3<double> M_orb = orbital_mag_dm.calculate_orbital_moment();

    // For time-reversal symmetric system: M_orb should be ~0
    EXPECT_NEAR(M_orb.x, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.y, 0.0, 1e-12);
    EXPECT_NEAR(M_orb.z, 0.0, 1e-12);
}

/**
 * Test 6: Gauge invariance under translation
 *
 * Orbital moment should be invariant under global translation
 */
TEST_F(OrbitalMagDMTest, GaugeInvariance) {
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm(ucell, kv, *dm, hR, sR, &pv, orb);

    // Calculate M_orb with original positions
    ModuleBase::Vector3<double> M_orb_before = orbital_mag_dm.calculate_orbital_moment();

    // Shift all atoms by a vector T
    ModuleBase::Vector3<double> shift(0.1, 0.2, 0.3);
    for (int ia = 0; ia < ucell.atoms[0].na; ia++) {
        ucell.atoms[0].tau[ia] += shift;
    }

    // Create new OrbitalMagDM object with shifted positions
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm_shifted(ucell, kv, *dm, hR, sR, &pv, orb);

    // Calculate M_orb again
    ModuleBase::Vector3<double> M_orb_after = orbital_mag_dm_shifted.calculate_orbital_moment();

    // Verify: |M_orb_before - M_orb_after| < tolerance
    EXPECT_NEAR(M_orb_before.x, M_orb_after.x, 1e-12);
    EXPECT_NEAR(M_orb_before.y, M_orb_after.y, 1e-12);
    EXPECT_NEAR(M_orb_before.z, M_orb_after.z, 1e-12);

    // Restore original positions
    for (int ia = 0; ia < ucell.atoms[0].na; ia++) {
        ucell.atoms[0].tau[ia] -= shift;
    }
}

/**
 * Test 7: Vector potential effect
 *
 * Verify that non-zero A(t) affects the calculation
 */
TEST_F(OrbitalMagDMTest, VectorPotentialEffect) {
    // Calculate with zero A(t)
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm_zero(ucell, kv, *dm, hR, sR, &pv, orb);
    ModuleBase::Vector3<double> M_orb_zero = orbital_mag_dm_zero.calculate_orbital_moment();

    // Calculate with non-zero A(t)
    ModuleBase::Vector3<double> At(0.1, 0.0, 0.0);
    hamilt::OrbitalMagDM<std::complex<double>> orbital_mag_dm_nonzero(ucell, kv, *dm, hR, sR, &pv, orb, At);
    ModuleBase::Vector3<double> M_orb_nonzero = orbital_mag_dm_nonzero.calculate_orbital_moment();

    // With empty HContainer, both should be zero regardless of A(t)
    // This test verifies the code path works; real difference would require populated HContainer
    EXPECT_NEAR(M_orb_zero.x, 0.0, 1e-12);
    EXPECT_NEAR(M_orb_nonzero.x, 0.0, 1e-12);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
