#include "deltap.h"
#include "source_base/memory.h"
#include "source_base/name_angular.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"

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
                nlm_iat0[ad].insert({all_indexes[iw1l], nlm_target});
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

} // namespace deltap
