#include "lr_spectrum.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/parallel_reduce.h"
#include "source_lcao/module_hcontainer/hcontainer_funcs.h"
#include "math.h"
#include <fstream>
#include <iomanip>
#ifdef __MPI
#include "source_base/module_external/scalapack_connector.h"
#endif

namespace LR
{
    /// helper: compute dot product Re(a.x*b.x + a.y*b.y + a.z*b.z) for real T
    template<typename T>
    inline double rotatory_dot(const ModuleBase::Vector3<T>& mu, const ModuleBase::Vector3<T>& m);

    template<>
    inline double rotatory_dot<double>(const ModuleBase::Vector3<double>& mu, const ModuleBase::Vector3<double>& m)
    {
        return mu.x * m.x + mu.y * m.y + mu.z * m.z;
    }

    template<>
    inline double rotatory_dot<std::complex<double>>(
        const ModuleBase::Vector3<std::complex<double>>& mu,
        const ModuleBase::Vector3<std::complex<double>>& m)
    {
        // R_S = Im[mu . m*] = Im[mu_x*conj(m_x) + mu_y*conj(m_y) + mu_z*conj(m_z)]
        return (mu.x * std::conj(m.x) + mu.y * std::conj(m.y) + mu.z * std::conj(m.z)).imag();
    }

    /// helper: store magnetic dipole result for T=double (store imaginary part)
    inline void store_magnetic_dipole(ModuleBase::Vector3<double>& out,
                                      const std::vector<std::complex<double>>& m_dipole)
    {
        out = ModuleBase::Vector3<double>(m_dipole[0].imag(), m_dipole[1].imag(), m_dipole[2].imag());
    }

    /// helper: store magnetic dipole result for T=complex<double>
    inline void store_magnetic_dipole(ModuleBase::Vector3<std::complex<double>>& out,
                                      const std::vector<std::complex<double>>& m_dipole)
    {
        out = ModuleBase::Vector3<std::complex<double>>(m_dipole[0], m_dipole[1], m_dipole[2]);
    }

    /// declared in lr_spectrum_velocity.cpp
    Velocity_op<std::complex<double>> get_velocity_matrix_R(const UnitCell& ucell,
        const Grid_Driver& gd,
        const Parallel_Orbitals& pmat,
        const TwoCenterBundle& two_center_bundle);

    template<typename T>
    void LR_Spectrum<T>::cal_magnetic_transition_dipoles()
    {
        ModuleBase::TITLE("LR::LR_Spectrum", "cal_magnetic_transition_dipoles");

        if (eig_ks_ == nullptr || nbands_ks_ == 0)
        {
            ModuleBase::WARNING("LR_Spectrum::cal_magnetic_transition_dipoles",
                "eig_ks not provided, cannot compute magnetic transition dipoles via SOS");
            magnetic_transition_dipole_.resize(nstate);
            return;
        }

        // ========== 1. Build velocity operator v(R) in AO basis ==========
        const Velocity_op<std::complex<double>>& vR = get_velocity_matrix_R(ucell, gd_, pmat, two_center_bundle_);

        const int nb = this->nbands_ks_;  // total KS bands (nocc + nvirt)
        const int nocc0 = this->nocc[0];  // number of occupied bands
        const double eta = 1e-6;  // degeneracy protection

        // ========== 2. Set up 2D distribution for band-basis matrix (nb x nb) ==========
        Parallel_2D pb;
#ifdef __MPI
        LR_Util::setup_2d_division(pb, 1, nb, nb, pmat.blacs_ctxt);
#else
        pb.set_serial(nb, nb);
#endif

        // Workspace for AO-basis v(k) and band-basis V(k)
        const int vk_size = pmat.get_local_size();
        std::vector<std::complex<double>> vk(vk_size, {0.0, 0.0});

        // Band-basis velocity matrices V_x, V_y, V_z (global, gathered for SOS)
        std::vector<std::complex<double>> V_global_x(nb * nb);
        std::vector<std::complex<double>> V_global_y(nb * nb);
        std::vector<std::complex<double>> V_global_z(nb * nb);

        // Intermediate for AO→band transformation
#ifdef __MPI
        const int temp_size = pc.get_local_size();  // same layout as C(k): naos_local_row x nbands_local_col
        const int vb_size = pb.get_local_size();    // nb_local_row x nb_local_col
#else
        const int temp_size = this->naos * nb;
        const int vb_size = nb * nb;
#endif
        std::vector<std::complex<double>> temp_mat(temp_size, {0.0, 0.0});
        std::vector<std::complex<double>> V_band(vb_size, {0.0, 0.0});

        // ========== 3. Compute m_{ia}(k) for each k and contract with X^S ==========
        magnetic_transition_dipole_.resize(nstate);

        // Pre-compute m_{ia,alpha}(k) for all k-points and store for contraction
        // m_{ia,alpha}(k) = -(i/2) * sum_{u!=a} eps_{alpha,beta,gamma} * [V^beta_{au} * V^gamma_{ui} - V^gamma_{au} * V^beta_{ui}] / (e_u - e_a)
        // This is the cross product (V_{au} x V_{ui})_alpha / (e_u - e_a)

        // For each excited state S: m_{0S,alpha} = sum_{iak} X^S_{iak} * m_{ia,alpha}(k)
        // We loop over k first to reuse the velocity matrices

        // Allocate m_ia for all k-points: m_ia_all[ik][alpha][local_pair_index]
        // local pair index follows pX layout: nvirt_local_row x nocc_local_col
        const int px_local_size = this->pX[0].get_local_size();
        // 3 directions x nk x px_local_size
        std::vector<std::complex<double>> m_ia_all(3 * nk * px_local_size, {0.0, 0.0});

        for (int ik = 0; ik < nk; ++ik)
        {
            const double* eig_k = eig_ks_ + ik * nb;  // KS eigenvalues at this k (in Ry)

            // Get eigenvectors C(k): psi_ks[0](ik, band_local, basis_local)
            // psi_ks[0] has shape (nk, pc.get_col_size(), pc.get_row_size())
            // pc distributes naos x nbands, so C is stored as naos_local_row x nbands_local_col
            this->psi_ks[0].fix_k(ik);
            const T* C_k = this->psi_ks[0].get_pointer();

            // Convert C_k to complex (needed for pzgemm when T=double)
            const int c_local_size = pc.get_local_size();
            std::vector<std::complex<double>> C_k_complex(c_local_size);
            for (int i = 0; i < c_local_size; ++i) { C_k_complex[i] = static_cast<std::complex<double>>(C_k[i]); }

            // For each Cartesian direction, fold v(R)->v(k), transform to band basis, gather
            for (int dir = 0; dir < 3; ++dir)
            {
                // Fold v_dir(R) -> v_dir(k) in AO basis
                std::fill(vk.begin(), vk.end(), std::complex<double>(0.0, 0.0));
                hamilt::folding_HR(*vR.get_current_term_pointer(dir), vk.data(), kv.kvec_d[ik], pmat.get_row_size(), 1);

                // Transform to band basis: V_dir = C† * v_dir(k) * C
                // Step 1: temp = v_dir(k) * C  [naos x naos] * [naos x nbands] = [naos x nbands]
                // Step 2: V_dir = C† * temp     [nbands x naos] * [naos x nbands] = [nbands x nbands]
#ifdef __MPI
                std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);

                // temp = v(k) * C
                std::fill(temp_mat.begin(), temp_mat.end(), std::complex<double>(0.0, 0.0));
                ScalapackConnector::gemm('N', 'N', this->naos, nb, this->naos, one,
                    vk.data(), 1, 1, pmat.desc,
                    C_k_complex.data(), 1, 1, pc.desc,
                    zero, temp_mat.data(), 1, 1, pc.desc);

                // V_dir = C† * temp
                std::fill(V_band.begin(), V_band.end(), std::complex<double>(0.0, 0.0));
                ScalapackConnector::gemm('C', 'N', nb, nb, this->naos, one,
                    C_k_complex.data(), 1, 1, pc.desc,
                    temp_mat.data(), 1, 1, pc.desc,
                    zero, V_band.data(), 1, 1, pb.desc);

                // Gather V_band to global on all processes
                std::complex<double>* V_global = (dir == 0) ? V_global_x.data() : (dir == 1) ? V_global_y.data() : V_global_z.data();
                std::fill(V_global, V_global + nb * nb, std::complex<double>(0.0, 0.0));
                for (int j_local = 0; j_local < pb.get_col_size(); ++j_local)
                {
                    int j_global = pb.local2global_col(j_local);
                    for (int i_local = 0; i_local < pb.get_row_size(); ++i_local)
                    {
                        int i_global = pb.local2global_row(i_local);
                        V_global[i_global * nb + j_global] = V_band[j_local * pb.get_row_size() + i_local];
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, V_global, nb * nb * 2, MPI_DOUBLE, MPI_SUM, pmat.comm());
#else
                // Serial: temp = v(k) * C  (column-major: vk[col*nrow+row], C stored as col-major too)
                std::fill(temp_mat.begin(), temp_mat.end(), std::complex<double>(0.0, 0.0));
                for (int mu = 0; mu < this->naos; ++mu)
                    for (int m = 0; m < nb; ++m)
                        for (int nu = 0; nu < this->naos; ++nu)
                            temp_mat[mu + m * this->naos] += vk[mu + nu * this->naos] * C_k_complex[nu + m * this->naos];
                // V_dir = C† * temp
                std::complex<double>* V_global = (dir == 0) ? V_global_x.data() : (dir == 1) ? V_global_y.data() : V_global_z.data();
                std::fill(V_global, V_global + nb * nb, std::complex<double>(0.0, 0.0));
                for (int n = 0; n < nb; ++n)
                    for (int m = 0; m < nb; ++m)
                        for (int mu = 0; mu < this->naos; ++mu)
                            V_global[n * nb + m] += std::conj(C_k_complex[mu + n * this->naos]) * temp_mat[mu + m * this->naos];
#endif
            }  // end dir loop

            // Now V_global_x/y/z[n*nb+m] = V^{x,y,z}_{nm}(k) in band basis (global, row-major)
            // Compute m_{ia,alpha}(k) using SOS formula:
            // m_{ia,alpha}(k) = -(i/2) * sum_{u!=a} (V_{au} x V_{ui})_alpha / (e_u - e_a)
            // where (AxB)_x = A_y*B_z - A_z*B_y, etc.
            // Here i=occupied, a=virtual in the KS band indexing

            for (int io_local = 0; io_local < this->pX[0].get_col_size(); ++io_local)
            {
                const int io_global = this->pX[0].local2global_col(io_local);  // occupied index (0..nocc-1)
                const int i_band = io_global;  // band index of occupied state

                for (int iv_local = 0; iv_local < this->pX[0].get_row_size(); ++iv_local)
                {
                    const int iv_global = this->pX[0].local2global_row(iv_local);  // virtual index (0..nvirt-1)
                    const int a_band = nocc0 + iv_global;  // band index of virtual state

                    const int pair_local = io_local * this->pX[0].get_row_size() + iv_local;

                    // SOS: sum over all intermediate states u (u != a)
                    std::complex<double> m_x(0.0, 0.0), m_y(0.0, 0.0), m_z(0.0, 0.0);
                    const double e_a = eig_k[a_band] / ModuleBase::e2;  // Ry -> Hartree

                    for (int u = 0; u < nb; ++u)
                    {
                        if (u == a_band) { continue; }
                        const double e_u = eig_k[u] / ModuleBase::e2;  // Ry -> Hartree
                        const double dE = e_u - e_a;
                        if (std::abs(dE) < eta) { continue; }

                        // V_{au}(k) and V_{ui}(k) for each direction
                        const std::complex<double> Vx_au = V_global_x[a_band * nb + u];
                        const std::complex<double> Vy_au = V_global_y[a_band * nb + u];
                        const std::complex<double> Vz_au = V_global_z[a_band * nb + u];
                        const std::complex<double> Vx_ui = V_global_x[u * nb + i_band];
                        const std::complex<double> Vy_ui = V_global_y[u * nb + i_band];
                        const std::complex<double> Vz_ui = V_global_z[u * nb + i_band];

                        // Cross product (V_{au} x V_{ui})
                        const std::complex<double> cross_x = Vy_au * Vz_ui - Vz_au * Vy_ui;
                        const std::complex<double> cross_y = Vz_au * Vx_ui - Vx_au * Vz_ui;
                        const std::complex<double> cross_z = Vx_au * Vy_ui - Vy_au * Vx_ui;

                        m_x += cross_x / dE;
                        m_y += cross_y / dE;
                        m_z += cross_z / dE;
                    }

                    // m_{ia,alpha} = -(i/2) * sum
                    const std::complex<double> prefac(-0.0, -0.5);  // -i/2
                    const int base = ik * px_local_size;
                    m_ia_all[0 * nk * px_local_size + base + pair_local] = prefac * m_x;
                    m_ia_all[1 * nk * px_local_size + base + pair_local] = prefac * m_y;
                    m_ia_all[2 * nk * px_local_size + base + pair_local] = prefac * m_z;
                }
            }
        }  // end ik loop

        // ========== 4. Contract with X^S to get magnetic transition dipole ==========
        for (int istate = 0; istate < nstate; ++istate)
        {
            std::vector<std::complex<double>> m_dipole(3, {0.0, 0.0});

            const int offset_b = istate * this->ldim;  // start of this state in X array

            for (int alpha = 0; alpha < 3; ++alpha)
            {
                for (int ik = 0; ik < nk; ++ik)
                {
                    const int x_offset = offset_b + ik * px_local_size;
                    const int m_offset = alpha * nk * px_local_size + ik * px_local_size;

                    for (int p = 0; p < px_local_size; ++p)
                    {
                        // X is type T, m_ia is complex
                        m_dipole[alpha] += static_cast<std::complex<double>>(X[x_offset + p]) * m_ia_all[m_offset + p];
                    }
                }
            }

            // Apply normalization
            // Note: unlike electric dipole which goes through DM_trans (has 1/nk built in),
            // here X is contracted directly with m_ia, so NO nk recovery is needed.
            // The sqrt(2) factor accounts for spin degeneracy and X normalization convention.
            for (int alpha = 0; alpha < 3; ++alpha)
            {
                if (this->nspin_x == 1) { m_dipole[alpha] *= std::sqrt(2.0); }
                Parallel_Reduce::reduce_all(m_dipole[alpha]);
            }

            store_magnetic_dipole(magnetic_transition_dipole_[istate], m_dipole);
        }
    }

    template<typename T>
    void LR_Spectrum<T>::cal_rotatory_strength()
    {
        ModuleBase::TITLE("LR::LR_Spectrum", "cal_rotatory_strength");
        rotatory_strength_.resize(nstate, 0.0);
        for (int istate = 0; istate < nstate; ++istate)
        {
            rotatory_strength_[istate] = rotatory_dot<T>(
                transition_dipole_[istate], magnetic_transition_dipole_[istate]);
        }
    }

    template<typename T>
    void LR_Spectrum<T>::ecd_spectrum(const std::vector<double>& freq, const double eta)
    {
        ModuleBase::TITLE("LR::LR_Spectrum", "ecd_spectrum");

        // 1. Compute magnetic transition dipoles and rotatory strengths
        this->cal_magnetic_transition_dipoles();
        this->cal_rotatory_strength();

        // 2. Print rotatory strengths to running log
        std::ofstream& ofs_running = GlobalV::ofs_running;
        ofs_running << "\n==================================================================== " << std::endl;
        ofs_running << "  Electronic Circular Dichroism (ECD) Analysis" << std::endl;
        ofs_running << "==================================================================== " << std::endl;
        ofs_running << std::setw(8) << "State"
                    << std::setw(20) << "Energy (eV)"
                    << std::setw(30) << "Osc. Strength (a.u.)"
                    << std::setw(30) << "Rot. Strength (a.u.)" << std::endl;
        ofs_running << "------------------------------------------------------------------------------------ " << std::endl;
        for (int istate = 0; istate < nstate; ++istate)
        {
            ofs_running << std::setw(8) << istate
                        << std::setw(20) << std::setprecision(6) << eig[istate] * ModuleBase::Ry_to_eV
                        << std::setw(30) << std::setprecision(6) << oscillator_strength_[istate]
                        << std::setw(30) << std::setprecision(6) << rotatory_strength_[istate] << std::endl;
        }
        ofs_running << "==================================================================== " << std::endl;

        // 2.5 Output per-state dipole details (ecd_detail.dat)
        if (GlobalV::MY_RANK == 0)
        {
            std::ofstream ofs_ecd_detail(PARAM.globalv.global_out_dir + "ecd_detail.dat");
            ofs_ecd_detail << std::scientific << std::setprecision(10);
            ofs_ecd_detail << "# ECD per-state detail" << std::endl;
            ofs_ecd_detail << "# State  E(Ry)  E(eV)  E(Ha)  "
                           << "mu_x  mu_y  mu_z  |mu|^2/3  osc_str  "
                           << "m_x  m_y  m_z  R_S" << std::endl;
            for (int istate = 0; istate < nstate; ++istate)
            {
                const auto& mu = transition_dipole_[istate];
                const auto& m = magnetic_transition_dipole_[istate];
                ofs_ecd_detail << istate << "\t"
                               << eig[istate] << "\t"
                               << eig[istate] * ModuleBase::Ry_to_eV << "\t"
                               << eig[istate] / ModuleBase::e2 << "\t"
                               << mu.x << "\t" << mu.y << "\t" << mu.z << "\t"
                               << mean_squared_transition_dipole_[istate] << "\t"
                               << oscillator_strength_[istate] << "\t"
                               << m.x << "\t" << m.y << "\t" << m.z << "\t"
                               << rotatory_strength_[istate] << std::endl;
            }
            ofs_ecd_detail << "\n# ECD spectrum prefactor: 4*pi/omega*e2/nk = "
                           << 4.0 * M_PI / ucell.omega * ModuleBase::e2 / this->nk << std::endl;
            ofs_ecd_detail << "# omega=" << ucell.omega << " e2=" << ModuleBase::e2 << " nk=" << this->nk << std::endl;
            ofs_ecd_detail.close();
        }

        // 3. Output ECD spectrum with Lorentzian broadening
        if (GlobalV::MY_RANK == 0)
        {
            std::ofstream ofs(PARAM.globalv.global_out_dir + "ecd.dat");
            ofs << "Frequency (eV) | wave length(nm) | Delta_epsilon (a.u.)" << std::endl;

            const double fac = 4.0 * M_PI / ucell.omega * ModuleBase::e2 / this->nk;

            for (size_t f = 0; f < freq.size(); ++f)
            {
                double ecd_value = 0.0;
                for (int i = 0; i < nstate; ++i)
                {
                    const double dw = (freq[f] - eig[i]) / ModuleBase::e2;
                    const double eta_au = eta / ModuleBase::e2;
                    ecd_value += rotatory_strength_[i] * eta_au / (dw * dw + eta_au * eta_au) / M_PI;
                }
                ecd_value *= fac;
                ofs << freq[f] * ModuleBase::Ry_to_eV << "\t"
                    << 91.126664 / freq[f] << "\t"
                    << ecd_value << std::endl;
            }
            ofs.close();
        }
    }
}

template class LR::LR_Spectrum<double>;
template class LR::LR_Spectrum<std::complex<double>>;
