/**
 * @file deltaqs_projector.cpp
 * @brief Implementation of CSZ projector for DeltaQS.
 *
 * @par Key difference from DeltaSpin cal_pre_HR
 * DeltaSpin: only first zeta per l -> nlm size = (l_max+1)^2
 * DeltaQS:   all n_zeta zetas per l -> nlm size = sum_l(n_zeta_l * (2l+1))
 *
 * The cal_HR_IJR logic is IDENTICAL; only the nlm content differs.
 *
 * @par Phase 1 verification
 * Fe (4s2p2d1f): CSZ nlm size = 4*1 + 2*3 + 2*5 + 1*7 = 27
 *                  DeltaSpin nlm size = (3+1)^2 = 16
 *
 * @par Development log
 * 2026-06-30: Initial implementation based on DeltaSpin cal_pre_HR pattern.
 *   - Adapted nlm selection to use csz_per_l from ValenceConfig
 *   - Reused cal_HR_IJR logic (identical for nspin=2 npol=1)
 *   - Added cal_charge() using the same pattern as DeltaSpin cal_moment()
 */
#include "deltaqs_projector.h"

#include <iostream>
#include <algorithm>
#include <cmath>

#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_base/memory.h"
#include "source_base/parallel_reduce.h"
#include "source_io/module_parameter/parameter.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"

namespace deltaqs {

CSZProjector::~CSZProjector()
{
    for (auto& hr : pre_hr_)
    {
        delete hr;
    }
    pre_hr_.clear();
}

void CSZProjector::build(const UnitCell& ucell,
                          const std::vector<ValenceConfig>& configs,
                          const Parallel_Orbitals* paraV,
                          const Grid_Driver* gridD,
                          const TwoCenterIntegrator* intor,
                          const std::vector<double>& orb_cutoff,
                          const hamilt::HContainer<double>* hR,
                          const std::vector<bool>& constraint_atoms)
{
    ModuleBase::TITLE("DeltaQS", "CSZProjector::build");
    ModuleBase::timer::start("DeltaQS", "CSZProjector::build");

    this->ucell_ = &ucell;
    this->paraV_ = paraV;
    this->constraint_atom_list_ = constraint_atoms;

    pre_hr_.clear();
    pre_hr_.resize(ucell.nat, nullptr);
    nproj_.resize(ucell.nat, 0);

    const int npol = ucell.get_npol();
    size_t memory_cost = 0;

    for (int iat = 0; iat < ucell.nat; iat++)
    {
        if (!constraint_atom_list_[iat]) continue;

        int T0, I0;
        ucell.iat2iait(iat, &I0, &T0);

        auto tau0 = ucell.get_tau(iat);

        // ---- Step 1: find adjacent atoms (same as DeltaSpin) ----
        AdjacentAtomInfo adjs;
        gridD->Find_atom(ucell, tau0, T0, I0, &adjs);

        std::vector<bool> is_adj(adjs.adj_num + 1, false);
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = ucell.itia2iat(T1, I1);
            const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad];
            if (ucell.cal_dtau(iat, iat1, R_index1).norm() * ucell.lat0
                < orb_cutoff[T1] + PARAM.inp.onsite_radius)
            {
                is_adj[ad] = true;
            }
        }
        filter_adjs(is_adj, adjs);

        // ---- Step 2: prepare <IJR> atom pairs (same as DeltaSpin) ----
        pre_hr_[iat] = new hamilt::HContainer<double>(paraV);
        for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
        {
            const int T1 = adjs.ntype[ad1];
            const int I1 = adjs.natom[ad1];
            const int iat1 = ucell.itia2iat(T1, I1);
            const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad1];
            for (int ad2 = 0; ad2 < adjs.adj_num + 1; ++ad2)
            {
                const int T2 = adjs.ntype[ad2];
                const int I2 = adjs.natom[ad2];
                const int iat2 = ucell.itia2iat(T2, I2);
                ModuleBase::Vector3<int>& R_index2 = adjs.box[ad2];
                int r_vector[3] = {R_index2.x - R_index1.x,
                                   R_index2.y - R_index1.y,
                                   R_index2.z - R_index1.z};
                if (hR->find_matrix(iat1, iat2, r_vector[0], r_vector[1], r_vector[2]) == nullptr)
                {
                    continue;
                }
                hamilt::AtomPair<double> tmp(iat1, iat2,
                                              r_vector[0], r_vector[1], r_vector[2],
                                              paraV);
                pre_hr_[iat]->insert_pair(tmp);
            }
        }
        pre_hr_[iat]->allocate(nullptr, true);

        // ---- Step 3: calculate <phi|alpha> overlap integrals ----
        // CSZ EXTENSION: use n_zeta zetas per l instead of just the first one
        const ValenceConfig& vc = configs[T0];
        const int total_nproj = vc.total_csz_orbitals;
        nproj_[iat] = total_nproj;

        std::cout << "[DeltaQS] CSZ projector for atom " << iat
                  << " (" << ucell.atoms[T0].label << "): "
                  << total_nproj << " projector functions" << std::endl;

        std::vector<std::unordered_map<int, std::vector<double>>> nlm_iat0(adjs.adj_num + 1);
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = ucell.itia2iat(T1, I1);
            const Atom* atom1 = &ucell.atoms[T1];
            const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];

            auto all_indexes = paraV->get_indexes_row(iat1);
            auto col_indexes = paraV->get_indexes_col(iat1);
            all_indexes.insert(all_indexes.end(), col_indexes.begin(), col_indexes.end());
            std::sort(all_indexes.begin(), all_indexes.end());
            all_indexes.erase(std::unique(all_indexes.begin(), all_indexes.end()), all_indexes.end());

            for (int iw1l = 0; iw1l < (int)all_indexes.size(); iw1l += npol)
            {
                const int iw1 = all_indexes[iw1l] / npol;
                std::vector<double> nlm_target(total_nproj, 0.0);

                const int L1 = atom1->iw2l[iw1];
                const int N1 = atom1->iw2n[iw1];
                const int m1 = atom1->iw2m[iw1];
                std::vector<std::vector<double>> nlm;

                const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                ModuleBase::Vector3<double> dtau = tau0 - tau1;
                intor->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 0, nlm);

                // ===== CSZ EXTENSION: select n_zeta zetas per l =====
                // DeltaSpin: only first zeta per l (target_L increments after first match)
                // DeltaQS:   first n_zeta_l zetas per l
                int index = 0;
                for (int l = 0; l <= vc.l_max; l++)
                {
                    int n_zeta = 0;
                    auto it_csz = vc.csz_per_l.find(l);
                    if (it_csz != vc.csz_per_l.end())
                    {
                        n_zeta = it_csz->second;
                    }

                    int zeta_count = 0;
                    for (int iw = 0; iw < ucell.atoms[T0].nw; iw++)
                    {
                        const int L0 = ucell.atoms[T0].iw2l[iw];
                        if (L0 == l && zeta_count < n_zeta)
                        {
                            for (int m = 0; m < 2 * L0 + 1; m++)
                            {
                                nlm_target[index] = nlm[0][iw + m];
                                index++;
                            }
                            zeta_count++;
                        }
                    }
                }

                nlm_iat0[ad].insert({all_indexes[iw1l], nlm_target});
            }
        }

        // ---- Step 4: calculate <phi|alpha><alpha|phi> (same logic as DeltaSpin) ----
        for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
        {
            const int T1 = adjs.ntype[ad1];
            const int I1 = adjs.natom[ad1];
            const int iat1 = ucell.itia2iat(T1, I1);
            ModuleBase::Vector3<int>& R_index1 = adjs.box[ad1];
            const std::unordered_map<int, std::vector<double>>& nlm1 = nlm_iat0[ad1];
            for (int ad2 = 0; ad2 < adjs.adj_num + 1; ++ad2)
            {
                const int T2 = adjs.ntype[ad2];
                const int I2 = adjs.natom[ad2];
                const int iat2 = ucell.itia2iat(T2, I2);
                const std::unordered_map<int, std::vector<double>>& nlm2 = nlm_iat0[ad2];
                ModuleBase::Vector3<int>& R_index2 = adjs.box[ad2];
                ModuleBase::Vector3<int> R_vector(R_index2[0] - R_index1[0],
                                                   R_index2[1] - R_index1[1],
                                                   R_index2[2] - R_index1[2]);
                hamilt::BaseMatrix<double>* tmp = pre_hr_[iat]->find_matrix(
                    iat1, iat2, R_vector[0], R_vector[1], R_vector[2]);
                if (tmp != nullptr)
                {
                    this->cal_hr_ijr(iat1, iat2, nlm1, nlm2, tmp->get_pointer());
                }
            }
        }
        memory_cost += pre_hr_[iat]->get_memory_size();
    }
    ModuleBase::Memory::record("DeltaQS:CSZ_pre_HR", memory_cost);
    ModuleBase::timer::end("DeltaQS", "CSZProjector::build");
}

// cal_hr_ijr: identical logic to DeltaSpin cal_HR_IJR for TR=double (nspin=2)
void CSZProjector::cal_hr_ijr(int iat1, int iat2,
                               const std::unordered_map<int, std::vector<double>>& nlm1_all,
                               const std::unordered_map<int, std::vector<double>>& nlm2_all,
                               double* data_pointer)
{
    const int npol = 1; // For nspin=2, npol=1 (no spinor structure)
    
    auto row_indexes = paraV_->get_indexes_row(iat1);
    auto col_indexes = paraV_->get_indexes_col(iat2);

    for (size_t iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
    {
        auto it1 = nlm1_all.find(row_indexes[iw1l]);
        if (it1 == nlm1_all.end()) {
            data_pointer += npol * col_indexes.size();
            continue;
        }
        const std::vector<double>& nlm1 = it1->second;

        for (size_t iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
        {
            auto it2 = nlm2_all.find(col_indexes[iw2l]);
            if (it2 == nlm2_all.end()) {
                data_pointer += npol;
                continue;
            }
            const std::vector<double>& nlm2 = it2->second;

            double nlm_tmp = 0.0;
            for (size_t m = 0; m < nlm1.size(); m++)
            {
                nlm_tmp += nlm1[m] * nlm2[m];
            }
            
            // For nspin=2 (npol=1), add to all npol*npol = 1 positions
            data_pointer[0] += nlm_tmp;
            data_pointer += npol;
        }
        data_pointer += (npol - 1) * col_indexes.size();
    }
}

const hamilt::HContainer<double>* CSZProjector::get_pre_hr(int iat) const
{
    if (iat < 0 || iat >= (int)pre_hr_.size()) return nullptr;
    return pre_hr_[iat];
}

int CSZProjector::get_nproj(int iat) const
{
    if (iat < 0 || iat >= (int)nproj_.size()) return 0;
    return nproj_[iat];
}

std::vector<double> CSZProjector::cal_charge(const hamilt::HContainer<double>* dmR) const
{
    const int nat = ucell_->nat;
    std::vector<double> charge(nat, 0.0);

    if (dmR == nullptr) return charge;

    for (int iat = 0; iat < nat; iat++)
    {
        if (!constraint_atom_list_[iat] || pre_hr_[iat] == nullptr) continue;

        for (int iap = 0; iap < pre_hr_[iat]->size_atom_pairs(); iap++)
        {
            const hamilt::AtomPair<double>& tmp = pre_hr_[iat]->get_atom_pair(iap);
            int iat1 = tmp.get_atom_i();
            int iat2 = tmp.get_atom_j();

            for (int ir = 0; ir < tmp.get_R_size(); ++ir)
            {
                const ModuleBase::Vector3<int> r_index = tmp.get_R_index(ir);
                const double* dmr_data = dmR->find_matrix(
                    iat1, iat2, r_index[0], r_index[1], r_index[2])->get_pointer();
                const double* hr_data = tmp.get_pointer(ir);
                int mat_size = tmp.get_size();

                for (int i = 0; i < mat_size; i++)
                {
                    charge[iat] += dmr_data[i] * hr_data[i];
                }
            }
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_all(charge.data(), charge.size());
#endif

    return charge;
}

std::vector<double> CSZProjector::cal_moment(const hamilt::HContainer<double>* dmR_diff) const
{
    return cal_charge(dmR_diff);
}

} // namespace deltaqs
