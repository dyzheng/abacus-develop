#include "orbital_mag.h"
#include "source_base/libm/libm.h"
#include "source_lcao/module_rt/td_folding.h"
#ifdef __MPI
#include "source_base/module_external/scalapack_connector.h"
#endif
#include "source_base/formatter.h"

namespace hamilt {

OrbitalMag::OrbitalMag(const UnitCell& ucell_in,
                       const K_Vectors& kv_in,
                       const elecstate::ElecState& pelec_in,
                       const HContainer<double>* hR_in,
                       const HContainer<double>* sR_in,
                       const psi::Psi<std::complex<double>>* psi_in,
                       const Parallel_Orbitals* pv_in)
    : ucell(ucell_in),
      kv(kv_in),
      pelec(pelec_in),
      hR(hR_in),
      sR(sR_in),
      psi(psi_in),
      pv(pv_in)
{
    // Initialize global dimensions
    this->nlocal = 0;
    for (int it = 0; it < ucell.ntype; it++) {
        this->nlocal += ucell.atoms[it].na * ucell.atoms[it].nw;
    }

    this->nbands = pelec.ekb.nc;  // Number of bands
}

OrbitalMag::~OrbitalMag()
{
    // Nothing to delete - all pointers are const references
}

void OrbitalMag::compute_k_derivative(const int ik,
                                      const int alpha,
                                      const HContainer<double>* hR_input,
                                      std::complex<double>* dk_matrix)
{
    // Zero output array - use local size for distributed matrices
    const int64_t local_size = get_local_size();
    std::fill_n(dk_matrix, local_size, std::complex<double>(0.0, 0.0));

    const ModuleBase::Vector3<double>& kvec_d = kv.kvec_d[ik];
    const int ld_hk = get_local_nrow();  // Leading dimension for column-major layout

    // Loop over atom pairs in H(R)
    #pragma omp parallel for
    for (int i = 0; i < hR_input->size_atom_pairs(); ++i) {
        hamilt::AtomPair<double>& ap = const_cast<hamilt::AtomPair<double>&>(hR_input->get_atom_pair(i));

        for (int ir = 0; ir < ap.get_R_size(); ++ir) {
            ModuleBase::Vector3<int> R_idx = ap.get_R_index(ir);
            ModuleBase::Vector3<double> dR(R_idx.x, R_idx.y, R_idx.z);

            // Compute k-phase: exp(i*k*R)
            double arg = (kvec_d * dR) * ModuleBase::TWO_PI;
            double sinp, cosp;
            ModuleBase::libm::sincos(arg, &sinp, &cosp);
            std::complex<double> kphase(cosp, sinp);

            // Convert R to Cartesian coordinates
            ModuleBase::Vector3<double> dR_cart = dR * ucell.latvec * ucell.lat0;

            // Multiply by i * R_alpha * exp(i*k*R)
            std::complex<double> factor = kphase * ModuleBase::IMAG_UNIT * dR_cart[alpha];

            // Get matrix elements for this R
            ap.find_R(R_idx);

            // Use add_to_matrix with column-major layout (hk_type=1) for ScaLAPACK compatibility
            // This handles both serial and parallel cases through AtomPair's paraV
            ap.add_to_matrix(dk_matrix, ld_hk, factor, 1);
        }
    }
}

void OrbitalMag::compute_velocity_matrix(const int ik,
                                         const std::complex<double>* dH_dk,
                                         const std::complex<double>* dS_dk,
                                         const double* eigenvalues,
                                         const std::complex<double>* eigenvectors,
                                         std::complex<double>* V_nm)
{
    // V_nm = C_n† * [dH/dk - E_avg * dS/dk] * C_m
    // where E_avg = (E_n + E_m)/2 for numerical stability

#ifdef __MPI
    if (is_parallel()) {
        // Parallel version using ScaLAPACK
        std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);

        // Step 1: dH_psi = dH/dk * psi  (nlocal x nbands, distributed)
        std::vector<std::complex<double>> dH_psi(pv->nloc_wfc, {0.0, 0.0});
        ScalapackConnector::gemm('N', 'N', nlocal, nbands, nlocal, one,
            dH_dk, 1, 1, pv->desc,
            eigenvectors, 1, 1, pv->desc_wfc,
            zero, dH_psi.data(), 1, 1, pv->desc_wfc);

        // Step 2: dS_psi = dS/dk * psi  (nlocal x nbands, distributed)
        std::vector<std::complex<double>> dS_psi(pv->nloc_wfc, {0.0, 0.0});
        ScalapackConnector::gemm('N', 'N', nlocal, nbands, nlocal, one,
            dS_dk, 1, 1, pv->desc,
            eigenvectors, 1, 1, pv->desc_wfc,
            zero, dS_psi.data(), 1, 1, pv->desc_wfc);

        // Step 3: H_nm = psi^H * dH_psi  (nbands x nbands, distributed)
        // Note: desc_Eij uses lld=nrow, so allocate with pv->nloc like band_energy.cpp
        std::vector<std::complex<double>> H_nm(pv->nloc, {0.0, 0.0});
        ScalapackConnector::gemm('C', 'N', nbands, nbands, nlocal, one,
            eigenvectors, 1, 1, pv->desc_wfc,
            dH_psi.data(), 1, 1, pv->desc_wfc,
            zero, H_nm.data(), 1, 1, pv->desc_Eij);

        // Step 4: S_nm = psi^H * dS_psi  (nbands x nbands, distributed)
        std::vector<std::complex<double>> S_nm(pv->nloc, {0.0, 0.0});
        ScalapackConnector::gemm('C', 'N', nbands, nbands, nlocal, one,
            eigenvectors, 1, 1, pv->desc_wfc,
            dS_psi.data(), 1, 1, pv->desc_wfc,
            zero, S_nm.data(), 1, 1, pv->desc_Eij);

        // Step 5: V_nm = H_nm - E_avg * S_nm (local elements only)
        // Use same indexing as band_energy.cpp: stride is pv->nrow (column-major)
        for (int j_local = 0; j_local < pv->ncol_bands; ++j_local) {
            int j_global = pv->local2global_col(j_local);
            if (j_global >= nbands) continue;
            for (int i_local = 0; i_local < pv->nrow_bands; ++i_local) {
                int i_global = pv->local2global_row(i_local);
                if (i_global >= nbands) continue;
                int idx = j_local * pv->nrow + i_local;  // Column-major with stride nrow
                double E_avg = 0.5 * (eigenvalues[i_global] + eigenvalues[j_global]);
                V_nm[idx] = H_nm[idx] - E_avg * S_nm[idx];
            }
        }
        return;
    }
#endif

    // Serial version
    // Zero output
    std::fill_n(V_nm, nbands * nbands, std::complex<double>(0.0, 0.0));

    // Temporary storage for intermediate results
    std::vector<std::complex<double>> temp(nlocal * nbands, std::complex<double>(0.0, 0.0));

    // For each band pair (n, m):
    for (int n = 0; n < nbands; ++n) {
        for (int m = 0; m < nbands; ++m) {
            std::complex<double> v_nm = 0.0;

            // Symmetrized energy
            double E_avg = 0.5 * (eigenvalues[n] + eigenvalues[m]);

            // Matrix multiplication: C_n† * [dH/dk - E_avg * dS/dk] * C_m
            for (int mu = 0; mu < nlocal; ++mu) {
                for (int nu = 0; nu < nlocal; ++nu) {
                    std::complex<double> dH_element = dH_dk[mu * nlocal + nu];
                    std::complex<double> dS_element = dS_dk[mu * nlocal + nu];

                    std::complex<double> v_element = dH_element - E_avg * dS_element;

                    v_nm += std::conj(eigenvectors[n * nlocal + mu])
                          * v_element
                          * eigenvectors[m * nlocal + nu];
                }
            }

            V_nm[n * nbands + m] = v_nm;
        }
    }
}

double OrbitalMag::compute_sos_contribution(const int ik,
                                            const std::complex<double>* V_x,
                                            const std::complex<double>* V_y,
                                            const double* eigenvalues,
                                            const double mu)
{
    double M_z_k = 0.0;
    const double eta = 1e-6;  // Degeneracy protection threshold

    // Get number of occupied bands
    int n_occ = 0;
    for (int n = 0; n < nbands; ++n) {
        if (pelec.wg(ik, n) > 1e-8) {
            n_occ = n + 1;
        }
    }

#ifdef __MPI
    if (is_parallel()) {
        // For parallel case, V_x and V_y are distributed in 2D block-cyclic format
        // We need to gather them to compute the SOS contribution
        // Each process computes partial contribution for its local elements

        // Gather V_x and V_y to all processes for SOS calculation
        std::vector<std::complex<double>> V_x_global(nbands * nbands, {0.0, 0.0});
        std::vector<std::complex<double>> V_y_global(nbands * nbands, {0.0, 0.0});

        // Copy local elements to global array positions
        // Use same indexing as band_energy.cpp: stride is pv->nrow
        for (int j_local = 0; j_local < pv->ncol_bands; ++j_local) {
            int j_global = pv->local2global_col(j_local);
            if (j_global >= nbands) continue;
            for (int i_local = 0; i_local < pv->nrow_bands; ++i_local) {
                int i_global = pv->local2global_row(i_local);
                if (i_global >= nbands) continue;
                int idx_local = j_local * pv->nrow + i_local;  // Column-major with stride nrow
                int idx_global = i_global * nbands + j_global;  // Row-major for SOS loop
                V_x_global[idx_global] = V_x[idx_local];
                V_y_global[idx_global] = V_y[idx_local];
            }
        }

        // Reduce to get full matrices on all processes
        MPI_Allreduce(MPI_IN_PLACE, V_x_global.data(), nbands * nbands * 2,
                      MPI_DOUBLE, MPI_SUM, pv->comm());
        MPI_Allreduce(MPI_IN_PLACE, V_y_global.data(), nbands * nbands * 2,
                      MPI_DOUBLE, MPI_SUM, pv->comm());

        // Compute SOS with global matrices (same on all processes)
        for (int n = 0; n < n_occ; ++n) {
            for (int m = 0; m < nbands; ++m) {
                if (m == n) continue;

                double dE = eigenvalues[n] - eigenvalues[m];
                if (std::abs(dE) < eta) continue;

                std::complex<double> v_x_nm = V_x_global[n * nbands + m];
                std::complex<double> v_y_mn = V_y_global[m * nbands + n];
                std::complex<double> product = v_x_nm * v_y_mn;

                double E_weight = eigenvalues[m] + eigenvalues[n] - 2.0 * mu;
                double contrib = product.imag() * E_weight / (dE * dE);

                M_z_k += contrib;
            }
        }

        return M_z_k * kv.wk[ik];
    }
#endif

    // Serial version: Sum over occupied states n
    for (int n = 0; n < n_occ; ++n) {
        // Sum over all states m (m ≠ n)
        for (int m = 0; m < nbands; ++m) {
            if (m == n) continue;

            double dE = eigenvalues[n] - eigenvalues[m];

            // Skip near-degenerate states
            if (std::abs(dE) < eta) continue;

            // Compute <n|v_x|m><m|v_y|n>
            std::complex<double> v_x_nm = V_x[n * nbands + m];
            std::complex<double> v_y_mn = V_y[m * nbands + n];
            std::complex<double> product = v_x_nm * v_y_mn;

            // Energy weight: (E_m + E_n - 2μ)
            double E_weight = eigenvalues[m] + eigenvalues[n] - 2.0 * mu;

            // Contribution: Im[product] * E_weight / dE²
            double contrib = product.imag() * E_weight / (dE * dE);

            M_z_k += contrib;
        }
    }

    return M_z_k * kv.wk[ik];  // Multiply by k-point weight
}

ModuleBase::Vector3<double> OrbitalMag::calculate_orbital_moment()
{
    ModuleBase::Vector3<double> M_orb(0.0, 0.0, 0.0);

    // Get Fermi energy
    double mu = pelec.eferm.ef;

    // Allocate workspace using local sizes for distributed matrices
    const int64_t local_mat_size = get_local_size();
    std::vector<std::complex<double>> dH_dk(local_mat_size);
    std::vector<std::complex<double>> dS_dk(local_mat_size);

    // V_x, V_y, V_z are band matrices (nbands x nbands)
    // In parallel mode, they use pv->nloc (same as desc_Eij layout)
    int64_t v_size = nbands * nbands;
#ifdef __MPI
    if (is_parallel()) {
        v_size = pv->nloc;  // Use nloc, not nloc_Eij, to match desc_Eij layout
    }
#endif
    std::vector<std::complex<double>> V_x(v_size);
    std::vector<std::complex<double>> V_y(v_size);
    std::vector<std::complex<double>> V_z(v_size);

    // Loop over k-points
    for (int ik = 0; ik < kv.get_nks(); ++ik) {
        // Get eigenvalues for this k-point
        const double* ekb_k = &pelec.ekb(ik, 0);

        // Get eigenvectors for this k-point
        const std::complex<double>* psi_k = &((*psi)(ik, 0, 0));

        // Compute velocity matrices V_x, V_y, V_z
        compute_k_derivative(ik, 0, hR, dH_dk.data());
        compute_k_derivative(ik, 0, sR, dS_dk.data());
        compute_velocity_matrix(ik, dH_dk.data(), dS_dk.data(),
                                ekb_k, psi_k, V_x.data());

        compute_k_derivative(ik, 1, hR, dH_dk.data());
        compute_k_derivative(ik, 1, sR, dS_dk.data());
        compute_velocity_matrix(ik, dH_dk.data(), dS_dk.data(),
                                ekb_k, psi_k, V_y.data());

        compute_k_derivative(ik, 2, hR, dH_dk.data());
        compute_k_derivative(ik, 2, sR, dS_dk.data());
        compute_velocity_matrix(ik, dH_dk.data(), dS_dk.data(),
                                ekb_k, psi_k, V_z.data());

        // Compute all three components of orbital moment:
        // M_x = Im[V_y * V_z], M_y = Im[V_z * V_x], M_z = Im[V_x * V_y]
        double M_x_k = compute_sos_contribution(ik, V_y.data(), V_z.data(),
                                                ekb_k, mu);
        double M_y_k = compute_sos_contribution(ik, V_z.data(), V_x.data(),
                                                ekb_k, mu);
        double M_z_k = compute_sos_contribution(ik, V_x.data(), V_y.data(),
                                                ekb_k, mu);
        M_orb.x += M_x_k;
        M_orb.y += M_y_k;
        M_orb.z += M_z_k;
        // print k-index, k-vector, M_x_k, M_y_k, M_z_k
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,
                                    FmtCore::format("k-point %d: k = (%.6f, %.6f, %.6f), M_orb_k = (%.6f, %.6f, %.6f)",
                                                    ik + 1,
                                                    kv.kvec_d[ik].x,
                                                    kv.kvec_d[ik].y,
                                                    kv.kvec_d[ik].z,    
                                                    M_x_k/kv.wk[ik], M_y_k/kv.wk[ik], M_z_k/kv.wk[ik]));
    }

    // Apply prefactor: 1/2
    M_orb *= 0.5;

    // MPI reduction if parallel (for k-point parallelization)
#ifdef __MPI
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized) {
        MPI_Allreduce(MPI_IN_PLACE, &M_orb.x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
#endif

    return M_orb;
}

} // namespace hamilt
