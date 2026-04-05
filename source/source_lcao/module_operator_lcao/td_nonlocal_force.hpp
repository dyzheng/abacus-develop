#pragma once
#include "td_nonlocal_lcao.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"

namespace hamilt
{

template <typename TK, typename TR>
void TDNonlocal<OperatorLCAO<TK, TR>>::cal_force(
    const bool cal_force,
    const HContainer<TR>* dmR,
    ModuleBase::matrix& force)
{
    ModuleBase::TITLE("TDNonlocal", "cal_force");
    ModuleBase::timer::tick("TDNonlocal", "cal_force");

    if (!cal_force)
    {
        ModuleBase::timer::tick("TDNonlocal", "cal_force");
        return;
    }

    force.zero_out();

    const Parallel_Orbitals* paraV = dmR->get_paraV();
    const int npol = this->ucell->get_npol();

    // Update vector potential
    this->cart_At = TD_info::cart_At;

    // Loop over all atoms and calculate force contributions
    #pragma omp parallel
    {
        ModuleBase::matrix force_local(force.nr, force.nc);

        #pragma omp for schedule(dynamic)
        for (int iat0 = 0; iat0 < this->ucell->nat; iat0++)
        {
            auto tau0 = ucell->get_tau(iat0);
            int T0 = 0, I0 = 0;
            ucell->iat2iait(iat0, &I0, &T0);

            // Find adjacent atoms
            AdjacentAtomInfo adjs;
            this->Grid->Find_atom(*ucell, tau0, T0, I0, &adjs);

            // Filter adjacent atoms based on cutoff
            std::vector<bool> is_adj(adjs.adj_num + 1, false);
            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T1 = adjs.ntype[ad];
                const int I1 = adjs.natom[ad];
                const int iat1 = ucell->itia2iat(T1, I1);
                const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad];
                if (this->ucell->cal_dtau(iat0, iat1, R_index1).norm() * this->ucell->lat0
                    < orb_.Phi[T1].getRcut() + this->ucell->infoNL.Beta[T0].get_rcut_max())
                {
                    is_adj[ad] = true;
                }
            }
            filter_adjs(is_adj, adjs);

            // Calculate <psi|beta> for each neighbor with derivatives
            std::vector<std::unordered_map<int, std::vector<std::complex<double>>>> nlm_iat0(adjs.adj_num + 1);
            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T1 = adjs.ntype[ad];
                const int I1 = adjs.natom[ad];
                const int iat1 = ucell->itia2iat(T1, I1);
                const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];
                const Atom* atom1 = &ucell->atoms[T1];

                auto all_indexes = paraV->get_indexes_row(iat1);
                auto col_indexes = paraV->get_indexes_col(iat1);
                all_indexes.insert(all_indexes.end(), col_indexes.begin(), col_indexes.end());
                std::sort(all_indexes.begin(), all_indexes.end());
                all_indexes.erase(std::unique(all_indexes.begin(), all_indexes.end()), all_indexes.end());

                for (size_t iw1l = 0; iw1l < all_indexes.size(); iw1l += npol)
                {
                    const int iw1 = all_indexes[iw1l] / npol;
                    std::vector<std::vector<std::complex<double>>> nlm;
                    module_rt::snap_psibeta_half_tddft(orb_,
                                                       this->ucell->infoNL,
                                                       nlm,
                                                       tau1 * this->ucell->lat0,
                                                       T1,
                                                       atom1->iw2l[iw1],
                                                       atom1->iw2m[iw1],
                                                       atom1->iw2n[iw1],
                                                       tau0 * this->ucell->lat0,
                                                       T0,
                                                       this->cart_At,
                                                       false /*calc_r*/,
                                                       true /*calc_deri*/);

                    // Store value + 3 derivatives interleaved
                    const int length = nlm[0].size();
                    std::vector<std::complex<double>> nlm_target(length * 4);
                    for (int idx = 0; idx < length; idx++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            nlm_target[idx + n * length] = nlm[n][idx];
                        }
                    }
                    // Key is all_indexes[iw1l], matching the original code
                    nlm_iat0[ad].insert({all_indexes[iw1l], nlm_target});
                }
            }

            // Second iteration to calculate force
            for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
            {
                const int T1 = adjs.ntype[ad1];
                const int I1 = adjs.natom[ad1];
                const int iat1 = ucell->itia2iat(T1, I1);
                double* force_tmp1 = &force_local(iat1, 0);
                double* force_tmp2 = &force_local(iat0, 0);

                for (int ad2 = 0; ad2 < adjs.adj_num + 1; ++ad2)
                {
                    const int T2 = adjs.ntype[ad2];
                    const int I2 = adjs.natom[ad2];
                    const int iat2 = ucell->itia2iat(T2, I2);
                    const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad1];
                    const ModuleBase::Vector3<int>& R_index2 = adjs.box[ad2];
                    const ModuleBase::Vector3<int> R_vector(R_index2[0] - R_index1[0],
                                                            R_index2[1] - R_index1[1],
                                                            R_index2[2] - R_index1[2]);

                    const hamilt::BaseMatrix<TR>* tmp
                        = dmR->find_matrix(iat1, iat2, R_vector[0], R_vector[1], R_vector[2]);

                    if (tmp == nullptr)
                    {
                        continue;
                    }

                    int row_size = paraV->get_row_size();
                    int col_size = paraV->get_col_size();
                    if (row_size == 0 || col_size == 0)
                    {
                        continue;
                    }

                    this->cal_force_IJR(iat1,
                                        iat2,
                                        T0,
                                        paraV,
                                        nlm_iat0[ad1],
                                        nlm_iat0[ad2],
                                        tmp,
                                        force_tmp1,
                                        force_tmp2);
                }
            }
        }

        #pragma omp critical
        {
            force += force_local;
        }
    }

    // MPI reduction
#ifdef __MPI
    Parallel_Reduce::reduce_all(force.c, force.nr * force.nc);
#endif

    // Multiply by 2 for Hermitian matrix
    for (int i = 0; i < force.nr * force.nc; i++)
    {
        force.c[i] *= 2.0;
    }

    ModuleBase::timer::tick("TDNonlocal", "cal_force");
}

template <typename TK, typename TR>
void TDNonlocal<OperatorLCAO<TK, TR>>::cal_force_IJR(
    const int& iat1,
    const int& iat2,
    const int& T0,
    const Parallel_Orbitals* paraV,
    const std::unordered_map<int, std::vector<std::complex<double>>>& nlm1_all,
    const std::unordered_map<int, std::vector<std::complex<double>>>& nlm2_all,
    const hamilt::BaseMatrix<TR>* dmR_pointer,
    double* force1,
    double* force2)
{
    const int npol = this->ucell->get_npol();

    auto row_indexes = paraV->get_indexes_row(iat1);
    auto col_indexes = paraV->get_indexes_col(iat2);

    const TR* tmp_d = nullptr;
    const TR* dm_pointer = dmR_pointer->get_pointer();

    // IMPORTANT: Loop WITHOUT npol stride, matching the original code
    for (int iw1l = 0; iw1l < row_indexes.size(); iw1l++)
    {
        auto it1 = nlm1_all.find(row_indexes[iw1l]);
        if (it1 == nlm1_all.end()) continue;
        const std::vector<std::complex<double>>& nlm1 = it1->second;
        const int length = nlm1.size() / 4;

        for (int iw2l = 0; iw2l < col_indexes.size(); iw2l++)
        {
            auto it2 = nlm2_all.find(col_indexes[iw2l]);
            if (it2 == nlm2_all.end()) continue;
            const std::vector<std::complex<double>>& nlm2 = it2->second;

            std::vector<std::complex<double>> nlm_tmp(3, std::complex<double>{0, 0});
            // Use only is=0 for non-SOC case (npol=1), matching original code
            for (int no = 0; no < this->ucell->atoms[T0].ncpp.non_zero_count_soc[0]; no++)
            {
                const int p1 = this->ucell->atoms[T0].ncpp.index1_soc[0][no];
                const int p2 = this->ucell->atoms[T0].ncpp.index2_soc[0][no];
                this->ucell->atoms[T0].ncpp.get_d(0, p1, p2, tmp_d);
                nlm_tmp[0] += nlm1[p1 + length] * nlm2[p2] * (*tmp_d);
                nlm_tmp[1] += nlm1[p1 + length * 2] * nlm2[p2] * (*tmp_d);
                nlm_tmp[2] += nlm1[p1 + length * 3] * nlm2[p2] * (*tmp_d);
            }

            for (int i = 0; i < 3; i++)
            {
                double tmp = (dm_pointer[0] * nlm_tmp[i]).real();
                force1[i] += tmp;
                force2[i] -= tmp;
            }
            dm_pointer++;
        }
    }
}

} // namespace hamilt