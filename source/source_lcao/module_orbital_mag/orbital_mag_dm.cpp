#include "orbital_mag_dm.h"
#include "source_base/libm/libm.h"
#include "source_base/constants.h"
#include "source_base/formatter.h"
#include "source_base/global_variable.h"
#include "source_io/cal_r_overlap_R.h"
#ifdef __MPI
#include "source_base/module_external/scalapack_connector.h"
#endif

namespace hamilt {

template <typename TK>
OrbitalMagDM<TK>::OrbitalMagDM(const UnitCell& ucell_in,
                               const K_Vectors& kv_in,
                               const elecstate::DensityMatrix<TK, double>& dm_in,
                               const HContainer<double>* hR_in,
                               const HContainer<double>* sR_in,
                               const Parallel_Orbitals* pv_in,
                               const LCAO_Orbitals& orb_in,
                               const ModuleBase::Vector3<double>& At_in)
    : ucell(ucell_in),
      kv(kv_in),
      dm(dm_in),
      hR(hR_in),
      sR(sR_in),
      pv(pv_in),
      orb(orb_in),
      cart_At(At_in)
{
    // Initialize global dimensions
    this->nlocal = 0;
    for (int it = 0; it < ucell.ntype; it++) {
        this->nlocal += ucell.atoms[it].na * ucell.atoms[it].nw;
    }
}

template <typename TK>
OrbitalMagDM<TK>::~OrbitalMagDM()
{
    // Nothing to delete - all pointers are const references
}

template <typename TK>
void OrbitalMagDM<TK>::set_vector_potential(const ModuleBase::Vector3<double>& At)
{
    this->cart_At = At;
}

template <typename TK>
void OrbitalMagDM<TK>::fold_to_k(int ik, const HContainer<double>* mR, TK* mk)
{
    // Zero output array
    const int64_t local_size = get_local_size();
    std::fill_n(mk, local_size, TK(0.0, 0.0));

    const ModuleBase::Vector3<double>& kvec_d = kv.kvec_d[ik];
    const int ld_hk = get_local_nrow();

    // Convert A(t) from Cartesian to direct coordinates
    // A_direct = A_cart * latvec^T (since kvec_d is in direct coords)
    ModuleBase::Vector3<double> At_direct(0.0, 0.0, 0.0);
    if (cart_At.norm() > 1e-10) {
        // A_direct = A_cart * G^T where G = 2*pi*inv(latvec)
        // For simplicity, we work in Cartesian k-space
        // kvec_c = kvec_d * G, so phase = (kvec_c + A_cart) * R_cart
        // But since we use kvec_d * dR (direct), we need A in direct coords
        At_direct.x = cart_At * ucell.a1;
        At_direct.y = cart_At * ucell.a2;
        At_direct.z = cart_At * ucell.a3;
        At_direct /= ModuleBase::TWO_PI;  // Convert to same units as kvec_d
    }

    // Loop over atom pairs
    #pragma omp parallel for
    for (int i = 0; i < mR->size_atom_pairs(); ++i) {
        hamilt::AtomPair<double>& ap = const_cast<hamilt::AtomPair<double>&>(mR->get_atom_pair(i));

        for (int ir = 0; ir < ap.get_R_size(); ++ir) {
            ModuleBase::Vector3<int> R_idx = ap.get_R_index(ir);
            ModuleBase::Vector3<double> dR(R_idx.x, R_idx.y, R_idx.z);

            // Compute k-phase: exp(i*(k+A)*R)
            double arg = ((kvec_d + At_direct) * dR) * ModuleBase::TWO_PI;
            double sinp, cosp;
            ModuleBase::libm::sincos(arg, &sinp, &cosp);
            std::complex<double> kphase(cosp, sinp);

            // Get matrix elements for this R
            ap.find_R(R_idx);

            // Use add_to_matrix with column-major layout (hk_type=1)
            ap.add_to_matrix(mk, ld_hk, kphase, 1);
        }
    }
}

template <typename TK>
void OrbitalMagDM<TK>::compute_k_derivative(int ik, int alpha, const HContainer<double>* mR, TK* dm_dk)
{
    // Zero output array
    const int64_t local_size = get_local_size();
    std::fill_n(dm_dk, local_size, TK(0.0, 0.0));

    const ModuleBase::Vector3<double>& kvec_d = kv.kvec_d[ik];
    const int ld_hk = get_local_nrow();

    // Convert A(t) to direct coordinates (same as in fold_to_k)
    ModuleBase::Vector3<double> At_direct(0.0, 0.0, 0.0);
    if (cart_At.norm() > 1e-10) {
        At_direct.x = cart_At * ucell.a1;
        At_direct.y = cart_At * ucell.a2;
        At_direct.z = cart_At * ucell.a3;
        At_direct /= ModuleBase::TWO_PI;
    }

    // Loop over atom pairs
    #pragma omp parallel for
    for (int i = 0; i < mR->size_atom_pairs(); ++i) {
        hamilt::AtomPair<double>& ap = const_cast<hamilt::AtomPair<double>&>(mR->get_atom_pair(i));

        for (int ir = 0; ir < ap.get_R_size(); ++ir) {
            ModuleBase::Vector3<int> R_idx = ap.get_R_index(ir);
            ModuleBase::Vector3<double> dR(R_idx.x, R_idx.y, R_idx.z);

            // Compute k-phase: exp(i*(k+A)*R)
            double arg = ((kvec_d + At_direct) * dR) * ModuleBase::TWO_PI;
            double sinp, cosp;
            ModuleBase::libm::sincos(arg, &sinp, &cosp);
            std::complex<double> kphase(cosp, sinp);

            // Convert R to Cartesian coordinates for the derivative factor
            ModuleBase::Vector3<double> dR_cart = dR * ucell.latvec * ucell.lat0;

            // Factor: i * R_alpha * exp(i*(k+A)*R)
            std::complex<double> factor = kphase * ModuleBase::IMAG_UNIT * dR_cart[alpha];

            // Get matrix elements for this R
            ap.find_R(R_idx);

            // Use add_to_matrix with column-major layout (hk_type=1)
            ap.add_to_matrix(dm_dk, ld_hk, factor, 1);
        }
    }
}

template <typename TK>
std::pair<int, int> OrbitalMagDM<TK>::find_k_neighbors(int ik, int alpha)
{
    // Find the nearest k-point neighbors in direction alpha from the actual k-grid
    // Instead of using a fixed dk_finite, we search for the closest k-points
    // that differ primarily in the alpha direction

    const ModuleBase::Vector3<double>& kvec_d = kv.kvec_d[ik];
    const double tol = 1e-6;

    int ik_plus = -1, ik_minus = -1;
    double min_dist_plus = 1e10;
    double min_dist_minus = 1e10;

    // Helper to compute wrapped difference in one direction
    auto wrap_diff = [](double d) {
        while (d > 0.5) d -= 1.0;
        while (d < -0.5) d += 1.0;
        return d;
    };

    for (int jk = 0; jk < kv.get_nks(); ++jk) {
        if (jk == ik) continue;

        const ModuleBase::Vector3<double>& kj = kv.kvec_d[jk];

        // Compute wrapped difference
        ModuleBase::Vector3<double> diff;
        diff.x = wrap_diff(kj.x - kvec_d.x);
        diff.y = wrap_diff(kj.y - kvec_d.y);
        diff.z = wrap_diff(kj.z - kvec_d.z);

        // Check if this k-point differs mainly in the alpha direction
        // The other two directions should be nearly zero
        double diff_alpha = diff[alpha];
        double diff_other1 = diff[(alpha + 1) % 3];
        double diff_other2 = diff[(alpha + 2) % 3];

        // Only consider k-points that are aligned in the alpha direction
        if (std::abs(diff_other1) > tol || std::abs(diff_other2) > tol) {
            continue;
        }

        // Check for positive direction neighbor (k + dk)
        if (diff_alpha > tol && diff_alpha < min_dist_plus) {
            min_dist_plus = diff_alpha;
            ik_plus = jk;
        }

        // Check for negative direction neighbor (k - dk)
        if (diff_alpha < -tol && std::abs(diff_alpha) < min_dist_minus) {
            min_dist_minus = std::abs(diff_alpha);
            ik_minus = jk;
        }
    }

    return std::make_pair(ik_plus, ik_minus);
}

template <typename TK>
void OrbitalMagDM<TK>::compute_dm_derivative(int ik, int alpha, TK* dP_dk)
{
    // Find neighboring k-points
    std::pair<int, int> k_neighbors = find_k_neighbors(ik, alpha);
    int ik_plus = k_neighbors.first;
    int ik_minus = k_neighbors.second;

    const int64_t local_size = get_local_size();

    if (ik_plus < 0 || ik_minus < 0) {
        // Cannot compute finite difference - set to zero
        std::fill_n(dP_dk, local_size, TK(0.0, 0.0));
        return;
    }

    // Get density matrices at neighboring k-points
    const TK* P_plus = dm.get_DMK_pointer(ik_plus);
    const TK* P_minus = dm.get_DMK_pointer(ik_minus);

    // Compute the actual k-point spacing in direct coordinates
    const ModuleBase::Vector3<double>& k_current = kv.kvec_d[ik];
    const ModuleBase::Vector3<double>& k_p = kv.kvec_d[ik_plus];
    const ModuleBase::Vector3<double>& k_m = kv.kvec_d[ik_minus];

    // Helper to compute wrapped difference
    auto wrap_diff = [](double d) {
        while (d > 0.5) d -= 1.0;
        while (d < -0.5) d += 1.0;
        return d;
    };

    // Total dk in direct coordinates (from k_minus to k_plus)
    double dk_direct = wrap_diff(k_p[alpha] - k_current[alpha])
                     - wrap_diff(k_m[alpha] - k_current[alpha]);

    // Convert to Cartesian coordinates
    // The reciprocal lattice vectors are rows of G matrix
    ModuleBase::Vector3<double> b_alpha(0.0, 0.0, 0.0);
    if (alpha == 0) {
        b_alpha.set(ucell.G.e11, ucell.G.e12, ucell.G.e13);
    } else if (alpha == 1) {
        b_alpha.set(ucell.G.e21, ucell.G.e22, ucell.G.e23);
    } else {
        b_alpha.set(ucell.G.e31, ucell.G.e32, ucell.G.e33);
    }
    b_alpha *= ucell.tpiba;  // 2*pi/a * b_alpha

    // dk in Cartesian (Bohr^-1)
    double dk_cart = std::abs(dk_direct) * b_alpha.norm();

    if (dk_cart < 1e-10) {
        // Degenerate case - set to zero
        std::fill_n(dP_dk, local_size, TK(0.0, 0.0));
        return;
    }

    double inv_dk = 1.0 / dk_cart;

    // Compute finite difference: dP/dk = [P(k+) - P(k-)] / dk
    for (int64_t i = 0; i < local_size; ++i) {
        dP_dk[i] = (P_plus[i] - P_minus[i]) * inv_dk;
    }
}

template <typename TK>
void OrbitalMagDM<TK>::compute_energy_density(const TK* P, const TK* H, TK* W)
{
    // W = P * H (energy density matrix)
    const int64_t local_size = get_local_size();

#ifdef __MPI
    if (is_parallel()) {
        // Parallel version using ScaLAPACK
        std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);

        // W = P * H using pzgemm
        ScalapackConnector::gemm('N', 'N', nlocal, nlocal, nlocal, one,
            P, 1, 1, pv->desc,
            H, 1, 1, pv->desc,
            zero, W, 1, 1, pv->desc);
        return;
    }
#endif

    // Serial version: simple matrix multiplication
    std::fill_n(W, local_size, TK(0.0, 0.0));

    for (int i = 0; i < nlocal; ++i) {
        for (int j = 0; j < nlocal; ++j) {
            TK sum(0.0, 0.0);
            for (int k = 0; k < nlocal; ++k) {
                // Column-major: element (i,k) is at index k*nlocal + i
                sum += P[k * nlocal + i] * H[j * nlocal + k];
            }
            W[j * nlocal + i] = sum;
        }
    }
}

template <typename TK>
void OrbitalMagDM<TK>::compute_corrected_velocity(const TK* V, const TK* S_alpha, const TK* W, TK* V_tilde)
{
    // V_tilde = V - S_alpha * W
    const int64_t local_size = get_local_size();

#ifdef __MPI
    if (is_parallel()) {
        // Parallel version using ScaLAPACK
        std::complex<double> one(1.0, 0.0), zero(0.0, 0.0), minus_one(-1.0, 0.0);

        // First copy V to V_tilde
        std::copy(V, V + local_size, V_tilde);

        // Temporary for S_alpha * W
        std::vector<TK> SW(local_size, TK(0.0, 0.0));

        // SW = S_alpha * W
        ScalapackConnector::gemm('N', 'N', nlocal, nlocal, nlocal, one,
            S_alpha, 1, 1, pv->desc,
            W, 1, 1, pv->desc,
            zero, SW.data(), 1, 1, pv->desc);

        // V_tilde = V - SW
        for (int64_t i = 0; i < local_size; ++i) {
            V_tilde[i] -= SW[i];
        }
        return;
    }
#endif

    // Serial version
    // First compute S_alpha * W
    std::vector<TK> SW(local_size, TK(0.0, 0.0));

    for (int i = 0; i < nlocal; ++i) {
        for (int j = 0; j < nlocal; ++j) {
            TK sum(0.0, 0.0);
            for (int k = 0; k < nlocal; ++k) {
                sum += S_alpha[k * nlocal + i] * W[j * nlocal + k];
            }
            SW[j * nlocal + i] = sum;
        }
    }

    // V_tilde = V - SW
    for (int64_t i = 0; i < local_size; ++i) {
        V_tilde[i] = V[i] - SW[i];
    }
}

template <typename TK>
std::complex<double> OrbitalMagDM<TK>::compute_trace_term(const TK* P, const TK* P_x, const TK* V_tilde_y)
{
    // Compute Tr[P * P_x * V_tilde_y]
    // This is sum_ijk P_ij * P_x_jk * V_tilde_y_ki
    // = sum_i (P * P_x * V_tilde_y)_ii

    std::complex<double> trace(0.0, 0.0);
    const int64_t local_size = get_local_size();

#ifdef __MPI
    if (is_parallel()) {
        // Parallel version: compute P * P_x first, then multiply by V_tilde_y and trace
        std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);

        // Temp1 = P * P_x
        std::vector<TK> temp1(local_size, TK(0.0, 0.0));
        ScalapackConnector::gemm('N', 'N', nlocal, nlocal, nlocal, one,
            P, 1, 1, pv->desc,
            P_x, 1, 1, pv->desc,
            zero, temp1.data(), 1, 1, pv->desc);

        // Temp2 = Temp1 * V_tilde_y
        std::vector<TK> temp2(local_size, TK(0.0, 0.0));
        ScalapackConnector::gemm('N', 'N', nlocal, nlocal, nlocal, one,
            temp1.data(), 1, 1, pv->desc,
            V_tilde_y, 1, 1, pv->desc,
            zero, temp2.data(), 1, 1, pv->desc);

        // Compute trace from local elements
        // In 2D block-cyclic distribution, we sum diagonal elements that are local
        for (int j_local = 0; j_local < pv->ncol; ++j_local) {
            int j_global = pv->local2global_col(j_local);
            if (j_global >= nlocal) continue;
            for (int i_local = 0; i_local < pv->nrow; ++i_local) {
                int i_global = pv->local2global_row(i_local);
                if (i_global >= nlocal) continue;
                if (i_global == j_global) {
                    // Diagonal element
                    int idx = j_local * pv->nrow + i_local;
                    trace += temp2[idx];
                }
            }
        }

        // Reduce trace across all processes
        MPI_Allreduce(MPI_IN_PLACE, &trace, 2, MPI_DOUBLE, MPI_SUM, pv->comm());
        return trace;
    }
#endif

    // Serial version: compute full matrix product and trace
    // Temp1 = P * P_x
    std::vector<TK> temp1(local_size, TK(0.0, 0.0));
    for (int i = 0; i < nlocal; ++i) {
        for (int j = 0; j < nlocal; ++j) {
            TK sum(0.0, 0.0);
            for (int k = 0; k < nlocal; ++k) {
                sum += P[k * nlocal + i] * P_x[j * nlocal + k];
            }
            temp1[j * nlocal + i] = sum;
        }
    }

    // Temp2 = Temp1 * V_tilde_y and compute trace
    for (int i = 0; i < nlocal; ++i) {
        TK sum(0.0, 0.0);
        for (int k = 0; k < nlocal; ++k) {
            sum += temp1[k * nlocal + i] * V_tilde_y[i * nlocal + k];
        }
        trace += sum;
    }

    return trace;
}

template <typename TK>
ModuleBase::Vector3<double> OrbitalMagDM<TK>::calculate_orbital_moment()
{
    ModuleBase::Vector3<double> M_orb(0.0, 0.0, 0.0);

    // Allocate workspace
    const int64_t local_size = get_local_size();
    std::vector<TK> Hk(local_size);
    std::vector<TK> Sk(local_size);
    std::vector<TK> W(local_size);           // Energy density matrix
    std::vector<TK> V_x(local_size);         // dH/dk_x
    std::vector<TK> V_y(local_size);         // dH/dk_y
    std::vector<TK> V_z(local_size);         // dH/dk_z
    std::vector<TK> S_x(local_size);         // dS/dk_x
    std::vector<TK> S_y(local_size);         // dS/dk_y
    std::vector<TK> S_z(local_size);         // dS/dk_z
    std::vector<TK> V_tilde_x(local_size);   // Corrected velocity x
    std::vector<TK> V_tilde_y(local_size);   // Corrected velocity y
    std::vector<TK> V_tilde_z(local_size);   // Corrected velocity z
    std::vector<TK> P_x(local_size);         // dP/dk_x
    std::vector<TK> P_y(local_size);         // dP/dk_y
    std::vector<TK> P_z(local_size);         // dP/dk_z

    // Loop over k-points
    for (int ik = 0; ik < kv.get_nks(); ++ik) {
        // Get density matrix P(k)
        const TK* Pk = dm.get_DMK_pointer(ik);

        // 1. Build H(k) and S(k) via folding
        fold_to_k(ik, hR, Hk.data());
        fold_to_k(ik, sR, Sk.data());

        // 2. Compute energy density matrix W = P * H
        compute_energy_density(Pk, Hk.data(), W.data());

        // 3. Compute analytical k-derivatives of H and S
        compute_k_derivative(ik, 0, hR, V_x.data());
        compute_k_derivative(ik, 1, hR, V_y.data());
        compute_k_derivative(ik, 2, hR, V_z.data());
        compute_k_derivative(ik, 0, sR, S_x.data());
        compute_k_derivative(ik, 1, sR, S_y.data());
        compute_k_derivative(ik, 2, sR, S_z.data());

        // 4. Compute density matrix derivatives (finite difference)
        compute_dm_derivative(ik, 0, P_x.data());
        compute_dm_derivative(ik, 1, P_y.data());
        compute_dm_derivative(ik, 2, P_z.data());

        // 5. Build corrected velocities: V_tilde = V - S * W
        compute_corrected_velocity(V_x.data(), S_x.data(), W.data(), V_tilde_x.data());
        compute_corrected_velocity(V_y.data(), S_y.data(), W.data(), V_tilde_y.data());
        compute_corrected_velocity(V_z.data(), S_z.data(), W.data(), V_tilde_z.data());

        // 6. Compute trace terms for each component
        // M_z = (1/2) * Im * Tr[P * (P_x * V_tilde_y - P_y * V_tilde_x)]
        std::complex<double> trace_xy = compute_trace_term(Pk, P_x.data(), V_tilde_y.data());
        std::complex<double> trace_yx = compute_trace_term(Pk, P_y.data(), V_tilde_x.data());
        double M_z_k = 0.5 * (trace_xy - trace_yx).imag() * kv.wk[ik];

        // M_x = (1/2) * Im * Tr[P * (P_y * V_tilde_z - P_z * V_tilde_y)]
        std::complex<double> trace_yz = compute_trace_term(Pk, P_y.data(), V_tilde_z.data());
        std::complex<double> trace_zy = compute_trace_term(Pk, P_z.data(), V_tilde_y.data());
        double M_x_k = 0.5 * (trace_yz - trace_zy).imag() * kv.wk[ik];

        // M_y = (1/2) * Im * Tr[P * (P_z * V_tilde_x - P_x * V_tilde_z)]
        std::complex<double> trace_zx = compute_trace_term(Pk, P_z.data(), V_tilde_x.data());
        std::complex<double> trace_xz = compute_trace_term(Pk, P_x.data(), V_tilde_z.data());
        double M_y_k = 0.5 * (trace_zx - trace_xz).imag() * kv.wk[ik];

        // Debug output: check if P_x, P_y, P_z are non-zero
        double P_x_norm = 0.0, P_y_norm = 0.0, P_z_norm = 0.0;
        for (int64_t i = 0; i < local_size; ++i) {
            P_x_norm += std::norm(P_x[i]);
            P_y_norm += std::norm(P_y[i]);
            P_z_norm += std::norm(P_z[i]);
        }
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,
            FmtCore::format("  Debug: |P_x|=%.6e, |P_y|=%.6e, |P_z|=%.6e, trace_xy=(%.6e,%.6e), trace_yx=(%.6e,%.6e)",
                            std::sqrt(P_x_norm), std::sqrt(P_y_norm), std::sqrt(P_z_norm),
                            trace_xy.real(), trace_xy.imag(),
                            trace_yx.real(), trace_yx.imag()));

        M_orb.x += M_x_k;
        M_orb.y += M_y_k;
        M_orb.z += M_z_k;

        // Output per-k contribution
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,
            FmtCore::format("k-point %d (DM method): k = (%.6f, %.6f, %.6f), M_orb_k = (%.6e, %.6e, %.6e)",
                            ik + 1,
                            kv.kvec_d[ik].x,
                            kv.kvec_d[ik].y,
                            kv.kvec_d[ik].z,
                            M_x_k / kv.wk[ik],
                            M_y_k / kv.wk[ik],
                            M_z_k / kv.wk[ik]));
    }

    // MPI reduction for k-point parallelization
#ifdef __MPI
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized) {
        MPI_Allreduce(MPI_IN_PLACE, &M_orb.x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
#endif

    return M_orb;
}

template <typename TK>
ModuleBase::Vector3<double> OrbitalMagDM<TK>::calculate_local_moment()
{
    // Initialize cal_r_overlap_R to get access to get_psi_L_psi
    cal_r_overlap_R rR;
    rR.init(ucell, *pv, orb);

    // Build the on-site L matrix in global indexing:
    // L_local(mu, nu) = <phi_mu | L | phi_nu> only when mu and nu are on the same atom
    // The matrix is purely imaginary for real orbitals, stored as complex.
    //
    // We store L_local as a dense nlocal x nlocal complex matrix (3 components).
    // Only on-site (same atom) blocks are non-zero.

    std::vector<std::complex<double>> Lx_global(static_cast<int64_t>(nlocal) * nlocal, {0.0, 0.0});
    std::vector<std::complex<double>> Ly_global(static_cast<int64_t>(nlocal) * nlocal, {0.0, 0.0});
    std::vector<std::complex<double>> Lz_global(static_cast<int64_t>(nlocal) * nlocal, {0.0, 0.0});

    // Build orbital index mapping: global orbital index -> (type, atom, L, N, m)
    // Following the same convention as cal_r_overlap_R::init
    int iw = 0;
    for (int it = 0; it < ucell.ntype; it++)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ia++)
        {
            // Record the starting global index for this atom
            int iw_start = iw;
            int nw_atom = ucell.atoms[it].nw;

            // Compute all on-site L matrix elements for orbitals on this atom
            for (int iw1 = 0; iw1 < nw_atom; iw1++)
            {
                int L1 = ucell.atoms[it].iw2l[iw1];
                int N1 = ucell.atoms[it].iw2n[iw1];
                int m1 = ucell.atoms[it].iw2m[iw1];

                for (int iw2 = 0; iw2 < nw_atom; iw2++)
                {
                    int L2 = ucell.atoms[it].iw2l[iw2];
                    int N2 = ucell.atoms[it].iw2n[iw2];
                    int m2 = ucell.atoms[it].iw2m[iw2];

                    // On-site: R1 = R2 = tau (atom position), same type
                    ModuleBase::Vector3<double> R(0.0, 0.0, 0.0);
                    auto L_vec = rR.get_psi_L_psi(R, it, L1, m1, N1, R, it, L2, m2, N2);

                    int mu = iw_start + iw1;
                    int nu = iw_start + iw2;
                    // Column-major: element (mu, nu) at index nu * nlocal + mu
                    int64_t idx = static_cast<int64_t>(nu) * nlocal + mu;
                    Lx_global[idx] = L_vec.x;
                    Ly_global[idx] = L_vec.y;
                    Lz_global[idx] = L_vec.z;
                }
            }
            iw += nw_atom;
        }
    }

    // Now compute M_local = sum_k wk * Tr[P(k) * L_local]
    // Since L_local is on-site (R=0 only), L_local(k) = L_local for all k.
    // Tr[P(k) * L] = sum_{mu,nu} P(k)_{nu,mu} * L_{mu,nu}
    //              = sum_{mu,nu} P(k)_{nu,mu} * L_{mu,nu}
    // In column-major: P(k)[nu * nrow + mu] for element (mu, nu) ... but
    // actually P(k) is stored as column-major with local distribution.
    // For serial: P(k)[nu * nlocal + mu] = P_{mu,nu}

    ModuleBase::Vector3<double> M_local(0.0, 0.0, 0.0);

    for (int ik = 0; ik < kv.get_nks(); ++ik)
    {
        const TK* Pk = dm.get_DMK_pointer(ik);

        std::complex<double> trace_x(0.0, 0.0);
        std::complex<double> trace_y(0.0, 0.0);
        std::complex<double> trace_z(0.0, 0.0);

        if (!is_parallel())
        {
            // Serial: direct trace computation
            // Tr[P * L] = sum_{mu,nu} P_{mu,nu} * L_{nu,mu}
            // Column-major: P_{mu,nu} = Pk[nu * nlocal + mu]
            //               L_{nu,mu} = Lx_global[mu * nlocal + nu]
            for (int mu = 0; mu < nlocal; ++mu)
            {
                for (int nu = 0; nu < nlocal; ++nu)
                {
                    int64_t idx_P = static_cast<int64_t>(nu) * nlocal + mu;   // P_{mu,nu}
                    int64_t idx_L = static_cast<int64_t>(mu) * nlocal + nu;   // L_{nu,mu}
                    std::complex<double> P_mn(Pk[idx_P]);
                    trace_x += P_mn * Lx_global[idx_L];
                    trace_y += P_mn * Ly_global[idx_L];
                    trace_z += P_mn * Lz_global[idx_L];
                }
            }
        }
#ifdef __MPI
        else
        {
            // Parallel: iterate over local elements
            for (int j_local = 0; j_local < pv->ncol; ++j_local)
            {
                int nu = pv->local2global_col(j_local);
                if (nu >= nlocal) continue;
                for (int i_local = 0; i_local < pv->nrow; ++i_local)
                {
                    int mu = pv->local2global_row(i_local);
                    if (mu >= nlocal) continue;

                    int idx_local = j_local * pv->nrow + i_local;
                    std::complex<double> P_mn(Pk[idx_local]);

                    // L_{nu,mu} in global array
                    int64_t idx_L = static_cast<int64_t>(mu) * nlocal + nu;
                    trace_x += P_mn * Lx_global[idx_L];
                    trace_y += P_mn * Ly_global[idx_L];
                    trace_z += P_mn * Lz_global[idx_L];
                }
            }

            // Reduce across processes
            MPI_Allreduce(MPI_IN_PLACE, &trace_x, 2, MPI_DOUBLE, MPI_SUM, pv->comm());
            MPI_Allreduce(MPI_IN_PLACE, &trace_y, 2, MPI_DOUBLE, MPI_SUM, pv->comm());
            MPI_Allreduce(MPI_IN_PLACE, &trace_z, 2, MPI_DOUBLE, MPI_SUM, pv->comm());
        }
#endif

        // The result should be real (L is Hermitian, P is Hermitian)
        // M_local_alpha = sum_k wk * Re[Tr(P * L_alpha)]
        M_local.x += trace_x.real() * kv.wk[ik];
        M_local.y += trace_y.real() * kv.wk[ik];
        M_local.z += trace_z.real() * kv.wk[ik];

        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,
            FmtCore::format("k-point %d (local): Tr[P*Lx]=(%.6e,%.6e), Tr[P*Ly]=(%.6e,%.6e), Tr[P*Lz]=(%.6e,%.6e)",
                            ik + 1,
                            trace_x.real(), trace_x.imag(),
                            trace_y.real(), trace_y.imag(),
                            trace_z.real(), trace_z.imag()));
    }

    // MPI reduction for k-point parallelization
#ifdef __MPI
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized) {
        MPI_Allreduce(MPI_IN_PLACE, &M_local.x, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
#endif

    return M_local;
}

// Explicit template instantiation
template class OrbitalMagDM<std::complex<double>>;

} // namespace hamilt
