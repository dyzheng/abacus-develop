#include "deltap.h"
#include "source_base/memory.h"
#include "source_base/name_angular.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"

extern "C" {
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetri_(const int* n, double* a, const int* lda, const int* ipiv, double* work, const int* lwork, int* info);
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* info);
}

namespace deltap {

void DeltaP::compute_real_overlaps(const UnitCell& ucell, const Grid_Driver& gd)
{
    ModuleBase::TITLE("DeltaP", "compute_real_overlaps");
    ModuleBase::timer::start("DeltaP", "compute_real_overlaps");

    const int npol = ucell.get_npol();
    overlap_R_.clear();
    overlap_R_.resize(nat_);
    nproj_per_atom_.resize(nat_, 0);

    size_t memory_cost = 0;

    for (int iat = 0; iat < nat_; iat++)
    {
        auto tau0 = ucell.get_tau(iat);
        int T0, I0;
        ucell.iat2iait(iat, &I0, &T0);

        // Find adjacent atoms
        AdjacentAtomInfo adjs;
        gd.Find_atom(ucell, tau0, T0, I0, &adjs);

        // Filter by cutoff radius
        std::vector<bool> is_adj(adjs.adj_num + 1, false);
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = ucell.itia2iat(T1, I1);
            const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad];
            if (ucell.cal_dtau(iat, iat1, R_index1).norm() * ucell.lat0
                < orb_cutoff_[T1] + rm_)
            {
                is_adj[ad] = true;
            }
        }
        filter_adjs(is_adj, adjs);

        // max_l_plus_1 for this atom type (same as DeltaSpin)
        const int max_l_plus_1 = ucell.atoms[T0].nwl + 1;
        nproj_per_atom_[iat] = max_l_plus_1 * max_l_plus_1;

        // Compute <phi_mu | alpha^I_lm(R)> via snap()
        std::vector<std::unordered_map<int, std::vector<double>>> nlm_iat0(adjs.adj_num + 1);

        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = ucell.itia2iat(T1, I1);
            const Atom* atom1 = &ucell.atoms[T1];
            const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];

            auto all_indexes = paraV_->get_indexes_row(iat1);
            auto col_indexes = paraV_->get_indexes_col(iat1);
            all_indexes.insert(all_indexes.end(), col_indexes.begin(), col_indexes.end());
            std::sort(all_indexes.begin(), all_indexes.end());
            all_indexes.erase(std::unique(all_indexes.begin(), all_indexes.end()), all_indexes.end());

            for (int iw1l = 0; iw1l < (int)all_indexes.size(); iw1l += npol)
            {
                const int iw1 = all_indexes[iw1l] / npol;
                std::vector<double> nlm_target(max_l_plus_1 * max_l_plus_1);
                const int L1 = atom1->iw2l[iw1];
                const int N1 = atom1->iw2n[iw1];
                const int m1 = atom1->iw2m[iw1];

                std::vector<std::vector<double>> nlm;
                const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                ModuleBase::Vector3<double> dtau = tau0 - tau1;
                // Use intor_ (<phi|phi_onsite>) for D_mat computation
                // This gives <phi_onsite_lm | phi_mu> which is needed for
                // D_mat = sum_mu <phi_onsite_lm | phi_mu> * c_{m,mu} = <phi_onsite_lm | psi_m>
                intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 0, nlm);

                // Select first zeta of each l, same as DeltaSpin
                int target_L = 0, index = 0;
                for (int iw = 0; iw < ucell.atoms[T0].nw; iw++)
                {
                    const int L0 = ucell.atoms[T0].iw2l[iw];
                    if (L0 == target_L)
                    {
                        for (int m = 0; m < 2 * L0 + 1; m++)
                        {
                            nlm_target[index] = nlm[0][iw + m];
                            index++;
                        }
                        target_L++;
                    }
                }
                // Key the nlm map by the GLOBAL orbital index of the neighbor
                // atom's basis function: compute_S_k consumes these keys via
                // paraV_->global2local_row(key) and the D_I contraction reads
                // psi rows at the corresponding local row.  get_indexes_row/
                // get_indexes_col(iat) return atom-RELATIVE indices; storing
                // them directly would collide across atoms in serial (e.g.
                // F_rel0 and H_rel0 both at slot 0) and make the per-rank S_k
                // partial sums rank-dependent under MPI (S_k/D_I MPI bug).
                nlm_iat0[ad].insert({paraV_->iat2iwt_[iat1] + all_indexes[iw1l], nlm_target});
            }
        }

        // Store as OverlapData
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            OverlapData od;
            od.iat_adj = ucell.itia2iat(adjs.ntype[ad], adjs.natom[ad]);
            od.R_index = adjs.box[ad];
            od.nlm = nlm_iat0[ad];
            overlap_R_[iat].push_back(std::move(od));
        }
    }

    nproj_max_ = 0;
    for (int iat = 0; iat < nat_; iat++)
    {
        nproj_max_ = std::max(nproj_max_, nproj_per_atom_[iat]);
    }

    ModuleBase::Memory::record("DeltaP:overlap_R", memory_cost);
    ModuleBase::timer::end("DeltaP", "compute_real_overlaps");
}

void DeltaP::compute_smo_overlap_matrix(const UnitCell& ucell)
{
    // Compute SMO overlap matrix S_{ab} = <alpha_a | alpha_b>
    // where alpha_a are the SMO basis functions (first zeta of each l)
    // Uses the same TwoCenterIntegrator (intor_) as compute_real_overlaps

    ModuleBase::TITLE("DeltaP", "compute_smo_overlap_matrix");
    ModuleBase::timer::start("DeltaP", "compute_smo_overlap_matrix");

    const int npol = ucell.get_npol();

    // Build global SMO basis index: for each atom iat, list (type, l, m) of SMO functions
    struct SmoEntry {
        int iat;   // global atom index
        int it;    // atom type
        int l;     // angular momentum
        int m_idx; // m index in the SMO ordering (0..2l)
    };
    std::vector<SmoEntry> smo_entries;

    for (int iat = 0; iat < nat_; iat++)
    {
        auto tau0 = ucell.get_tau(iat);
        int T0 = 0, I0 = 0;
        ucell.iat2iait(iat, &I0, &T0);
        const int max_l_plus_1 = ucell.atoms[T0].nwl + 1;

        int idx = 0;
        for (int l = 0; l < max_l_plus_1; l++)
        {
            for (int m = 0; m < 2 * l + 1; m++)
            {
                smo_entries.push_back({iat, T0, l, m});
                idx++;
            }
        }
    }

    smo_m_dim_ = smo_entries.size();
    smo_overlap_.assign(smo_m_dim_ * smo_m_dim_, 0.0);

    // For each pair (a, b), compute <alpha_a | alpha_b>
    // alpha_a is on atom iat_a, alpha_b is on atom iat_b (possibly in a different cell)
    // Use the grid driver to find neighboring atoms within the orbital cutoff

    for (int a = 0; a < smo_m_dim_; a++)
    {
        int iat_a = smo_entries[a].iat;
        int T_a = smo_entries[a].it;
        int L_a = smo_entries[a].l;
        int M_a = smo_entries[a].m_idx;
        auto tau_a = ucell.get_tau(iat_a);
        int I_a = 0, T_a_tmp = 0;
        ucell.iat2iait(iat_a, &I_a, &T_a_tmp);
        AdjacentAtomInfo adjs;
        gd_->Find_atom(ucell, tau_a, T_a, I_a, &adjs);

        // Find_atom may not include self-atom. Always compute self-overlap first.
        {
            std::vector<std::vector<double>> nlm;
            ModuleBase::Vector3<double> zero(0, 0, 0);
            int M1_a = M_a_for_snap(ucell, T_a, L_a, M_a);
            if (onsite_onsite_intor_)
                onsite_onsite_intor_->snap(T_a, L_a, 0, M1_a, T_a, zero, 0, nlm);
            else if (overlap_intor_)
                overlap_intor_->snap(T_a, L_a, 0, M1_a, T_a, zero, 0, nlm);

            if (a < 3)
            {
                std::cout << "   DeltaP SELF: a=" << a << " iat=" << iat_a
                          << " T=" << T_a << " L=" << L_a << " M=" << M_a
                          << " M1=" << M1_a
                          << " nlm_size=" << (nlm.empty() ? 0 : (nlm[0].empty() ? 0 : nlm[0].size()))
                          << " onsite=" << (onsite_onsite_intor_ ? "Y" : "N")
                          << " overlap=" << (overlap_intor_ ? "Y" : "N")
                          << std::endl;
                if (!nlm.empty() && !nlm[0].empty() && a < 3)
                {
                    std::cout << "     nlm[0][0]=" << nlm[0][0];
                    if (nlm[0].size() > 1) std::cout << " nlm[0][1]=" << nlm[0][1];
                    if (nlm[0].size() > 2) std::cout << " nlm[0][2]=" << nlm[0][2];
                    std::cout << std::endl;
                }
            }

            if (!nlm.empty() && !nlm[0].empty())
            {
                if (a < 3)
                {
                    std::cout << "   DeltaP SELF_EXTRACT: a=" << a
                              << " nw=" << ucell.atoms[T_a].nw
                              << " nlm_size=" << nlm[0].size() << std::endl;
                }
                int target_L = 0;
                for (int iw = 0; iw < ucell.atoms[T_a].nw; iw++)
                {
                    int L0 = ucell.atoms[T_a].iw2l[iw];
                    if (L0 == target_L)
                    {
                        for (int m = 0; m < 2 * L0 + 1; m++)
                        {
                            if (iw + m >= ucell.atoms[T_a].nw) break;
                            double val = nlm[0][iw + m];
                            for (int b = 0; b < smo_m_dim_; b++)
                            {
                                if (smo_entries[b].iat == iat_a &&
                                    smo_entries[b].it == T_a &&
                                    smo_entries[b].l == L0 &&
                                    smo_entries[b].m_idx == m)
                                {
                                    smo_overlap_[a + b * smo_m_dim_] = val;
                                    if (a < 3 && b < 3)
                                        std::cout << "     S[" << a << "," << b
                                                  << "] = " << val << " (iw=" << iw
                                                  << " m=" << m << " L0=" << L0 << ")" << std::endl;
                                    break;
                                }
                            }
                        }
                        target_L++;
                    }
                }
            }
        }

        for (int ad = 0; ad < adjs.adj_num + 1; ad++)
        {
            int T_b = adjs.ntype[ad];
            int I_b = adjs.natom[ad];
            int iat_b = ucell.itia2iat(T_b, I_b);

            // Skip self-atom entirely (including periodic images)
            // Self-overlap and intra-atom cross terms already handled above
            if (iat_b == iat_a)
                continue;

            // Check if within orbital cutoff (use rm_ for SMO basis)
            if (ucell.cal_dtau(iat_a, iat_b, adjs.box[ad]).norm() * ucell.lat0
                > orb_cutoff_[T_a] + orb_cutoff_[T_b])
                continue;

            // Compute <phi_onsite_{T_a, L_a, 0, M_a} | phi_onsite_{T_b, *, 0, *}>
            // using onsite_onsite_intor_ (both bra and ket are phi_onsite)
            ModuleBase::Vector3<double> dtau = tau_a - adjs.adjacent_tau[ad];
            std::vector<std::vector<double>> nlm;
            if (onsite_onsite_intor_)
            {
                int M1_a = M_a_for_snap(ucell, T_a, L_a, M_a);
                onsite_onsite_intor_->snap(T_a, L_a, 0, M1_a,
                                            T_b, dtau * ucell.lat0, 0, nlm);
            }
            else
            {
                int M1_a = M_a_for_snap(ucell, T_a, L_a, M_a);
                overlap_intor_->snap(T_a, L_a, 0, M1_a,
                                      T_b, dtau * ucell.lat0, 0, nlm);
            }
            if (nlm.empty() || nlm[0].empty()) continue;

            // Extract the SMO components (first zeta of each l) for type T_b
            // The nlm array has nlm[0][iw] = <bra | ket_{iw}>
            // where iw indexes all LCAO orbitals of type T_b in order:
            //   l=0(n=0,m=0), l=0(n=1,m=0), l=1(n=0,m=0,1,2), l=1(n=1,m=0,1,2), ...
            // We want the first zeta (n=0) of each l, so:
            //   l=0: iw=0 (n=0, m=0)
            //   l=1: iw=2 (n=0, m=0,1,2) → nlm[0][2], nlm[0][3], nlm[0][4]
            //   l=2: iw=6 (n=0, m=0,1,2,3,4) → nlm[0][6..10]
            // The extraction logic finds the first iw with L0==target_L
            int target_L = 0;
            for (int iw = 0; iw < ucell.atoms[T_b].nw; iw++)
            {
                int L0 = ucell.atoms[T_b].iw2l[iw];
                if (L0 == target_L)
                {
                    // iw is the start of the first zeta for angular momentum L0
                    // The next 2*L0+1 entries (iw, iw+1, ..., iw+2*L0) are the m components
                    for (int m = 0; m < 2 * L0 + 1; m++)
                    {
                        if (iw + m >= ucell.atoms[T_b].nw) break;
                        double val = nlm[0][iw + m];
                        // Find the global SMO index for (iat_b, T_b, L0, m)
                        for (int b = 0; b < smo_m_dim_; b++)
                        {
                            if (smo_entries[b].iat == iat_b &&
                                smo_entries[b].it == T_b &&
                                smo_entries[b].l == L0 &&
                                smo_entries[b].m_idx == m)
                            {
                                smo_overlap_[a + b * smo_m_dim_] = val;
                                break;
                            }
                        }
                    }
                    target_L++;
                }
            }
        }
    }

    // Debug: check S before eigenvalue decomposition
    {
        double trace = 0.0;
        double max_asym = 0.0;
        for (int i = 0; i < smo_m_dim_; i++)
        {
            trace += smo_overlap_[i + i * smo_m_dim_];
            for (int j = 0; j < i; j++)
                max_asym = std::max(max_asym, std::abs(smo_overlap_[i+j*smo_m_dim_] - smo_overlap_[j+i*smo_m_dim_]));
        }
        std::cout << "   DeltaP: S trace=" << trace << " (should be " << smo_m_dim_
                  << "), max_asym=" << max_asym << std::endl;
        std::cout << "   DeltaP: S diagonal:";
        for (int i = 0; i < std::min(smo_m_dim_, 17); i++)
            std::cout << " " << std::fixed << std::setprecision(4) << smo_overlap_[i+i*smo_m_dim_];
        std::cout << std::endl;
    }

    // Compute S^{-1/2} via eigenvalue decomposition: S = V * D * V^T
    // S^{-1/2} = V * D^{-1/2} * V^T
    // IMPORTANT: dsyev with jobz="V" destroys the input matrix!
    // Must use a copy.
    smo_overlap_inv_.assign(smo_m_dim_ * smo_m_dim_, 0.0);
    {
        int n = smo_m_dim_;
        std::vector<double> eigvals(n);
        std::vector<double> work(1);
        int lwork = -1, info = 0;
        std::vector<double> S_eig = smo_overlap_;  // copy (dsyev will destroy S_eig)
        dsyev_((char*)"V", (char*)"U", &n, S_eig.data(), &n, eigvals.data(), work.data(), &lwork, &info);
        if (info != 0)
        {
            std::cerr << "DeltaP: SMO overlap dsyev query failed, info=" << info << std::endl;
            for (int i = 0; i < n; i++) smo_overlap_inv_[i + i * n] = 1.0;
        }
        else
        {
            lwork = static_cast<int>(work[0]);
            work.resize(std::max(lwork, 1));
            dsyev_((char*)"V", (char*)"U", &n, S_eig.data(), &n, eigvals.data(), work.data(), &lwork, &info);
            if (info != 0)
            {
                std::cerr << "DeltaP: SMO overlap dsyev failed, info=" << info << std::endl;
                for (int i = 0; i < n; i++) smo_overlap_inv_[i + i * n] = 1.0;
            }
            else
            {
                std::cout << "   DeltaP: S eigenvalue range: [" << eigvals[0] << ", " << eigvals[n-1] << "]" << std::endl;
                // S_eig now contains eigenvectors V (column-major)
                // Compute S^{-1/2} = V * D^{-1/2} * V^T
                for (int i = 0; i < n; i++)
                {
                    double d_inv_half = (eigvals[i] > 1e-10) ? 1.0 / std::sqrt(eigvals[i]) : 0.0;
                    for (int j = 0; j < n; j++)
                        smo_overlap_inv_[j + i * n] = S_eig[j + i * n] * d_inv_half;
                }
                std::vector<double> tmp(n * n, 0.0);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double s = 0.0;
                        for (int k = 0; k < n; k++)
                            s += smo_overlap_inv_[i + k * n] * S_eig[j + k * n];
                        tmp[i + j * n] = s;
                    }
                smo_overlap_inv_ = tmp;
            }
        }
    }

    std::cout << "   DeltaP: SMO overlap matrix " << smo_m_dim_ << "×" << smo_m_dim_
              << " computed and inverted" << std::endl;

    ModuleBase::timer::end("DeltaP", "compute_smo_overlap_matrix");
}

int DeltaP::M_a_for_snap(const UnitCell& ucell, int T, int L, int m_idx) const
{
    // Convert SMO m_idx (0..2l) to the M1 convention used by snap()
    // snap() uses: M1 = (m % 2 == 0) ? -m / 2 : (m + 1) / 2
    // But the SMO ordering in compute_real_overlaps uses:
    // for m = 0..2l: M1 = m (in the iw2m convention)
    // Actually, looking at compute_real_overlaps:
    //   const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
    // where m1 = atom1->iw2m[iw1]
    // And the SMO m_idx corresponds to the m loop: for (int m = 0; m < 2*L0+1; m++)
    // which uses nlm[0][iw + m], where iw is the first zeta of L0
    // So m_idx = 0..2l, and the corresponding M1 = atom->iw2m[iw + m_idx]

    // Find the first zeta orbital of angular momentum L for type T
    int iw_first = 0;
    for (int iw = 0; iw < ucell.atoms[T].nw; iw++)
    {
        if (ucell.atoms[T].iw2l[iw] == L)
        {
            iw_first = iw;
            break;
        }
    }
    int m1 = ucell.atoms[T].iw2m[iw_first + m_idx];
    return (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
}

} // namespace deltap
