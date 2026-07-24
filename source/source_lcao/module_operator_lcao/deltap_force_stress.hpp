#pragma once
#include "deltap_lcao.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"

namespace hamilt
{

template <typename TK, typename TR>
void DeltaPOperator<TK, TR>::cal_force_stress(const bool cal_force,
                                              const bool cal_stress,
                                              const HContainer<double>* dmR,
                                              ModuleBase::matrix& force,
                                              ModuleBase::matrix& stress)
{
    ModuleBase::TITLE("DeltaPOperator", "cal_force_stress");
    ModuleBase::timer::start("DeltaPOperator", "cal_force_stress");

    const Parallel_Orbitals* paraV = dmR->get_paraV();
    const int npol = this->ucell->get_npol();
    const int gdir = this->gdir_ - 1; // 0-based direction index
    std::vector<double> stress_tmp(6, 0);

    if (cal_force) force.zero_out();

    #pragma omp parallel
    {
        std::vector<double> stress_local(6, 0);
        ModuleBase::matrix force_local(force.nr, force.nc);
        #pragma omp for schedule(dynamic)
        for (int iat0 = 0; iat0 < this->ucell->nat; iat0++)
        {
            // Skip atoms without lambda (unconstrained)
            if (static_cast<size_t>(iat0) >= this->lambda_.size() || this->lambda_[iat0] == 0.0)
                continue;

            double lam = this->lambda_[iat0];
            auto tau0 = this->ucell->get_tau(iat0);
            int T0, I0;
            this->ucell->iat2iait(iat0, &I0, &T0);

            // Find adjacent atoms
            AdjacentAtomInfo adjs;
            this->gridD->Find_atom(*this->ucell, tau0, T0, I0, &adjs);

            std::vector<bool> is_adj(adjs.adj_num + 1, false);
            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T1 = adjs.ntype[ad];
                const int I1 = adjs.natom[ad];
                const int iat1 = this->ucell->itia2iat(T1, I1);
                const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad];
                if (this->ucell->cal_dtau(iat0, iat1, R_index1).norm() * this->ucell->lat0
                    < this->orb_cutoff_[T1] + this->rm_)
                {
                    is_adj[ad] = true;
                }
            }
            filter_adjs(is_adj, adjs);

            // Compute nlm projections with derivatives (cal_deri=1)
            const int max_l_plus_1 = this->ucell->atoms[T0].nwl + 1;
            const int length = max_l_plus_1 * max_l_plus_1;
            std::vector<std::unordered_map<int, std::vector<double>>> nlm_iat0(adjs.adj_num + 1);

            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T1 = adjs.ntype[ad];
                const int I1 = adjs.natom[ad];
                const int iat1 = this->ucell->itia2iat(T1, I1);
                const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];
                const Atom* atom1 = &this->ucell->atoms[T1];

                auto all_indexes = paraV->get_indexes_row(iat1);
                auto col_indexes = paraV->get_indexes_col(iat1);
                all_indexes.insert(all_indexes.end(), col_indexes.begin(), col_indexes.end());
                std::sort(all_indexes.begin(), all_indexes.end());
                all_indexes.erase(std::unique(all_indexes.begin(), all_indexes.end()), all_indexes.end());

                for (size_t iw1l = 0; iw1l < all_indexes.size(); iw1l += npol)
                {
                    const int iw1 = all_indexes[iw1l] / npol;
                    std::vector<std::vector<double>> nlm;
                    int L1 = atom1->iw2l[iw1];
                    int N1 = atom1->iw2n[iw1];
                    int m1 = atom1->iw2m[iw1];
                    int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;

                    ModuleBase::Vector3<double> dtau = tau0 - tau1;
                    this->intor_->snap(T1, L1, N1, M1, T0, dtau * this->ucell->lat0,
                                        1 /*cal_deri*/, nlm);

                    // Extract nlm data for all target L0 values (with derivatives)
                    std::vector<double> nlm_target(length * 4);
                    int target_L = 0, index = 0;
                    for (int iw = 0; iw < this->ucell->atoms[T0].nw; iw++)
                    {
                        const int L0 = this->ucell->atoms[T0].iw2l[iw];
                        if (L0 == target_L)
                        {
                            for (int m = 0; m < 2 * L0 + 1; m++)
                            {
                                for (int channel = 0; channel < 4; channel++)
                                    nlm_target[index + channel * length] = nlm[channel][iw + m];
                                index++;
                            }
                        }
                        else
                        {
                            target_L = L0;
                            index = target_L * target_L;
                            for (int m = 0; m < 2 * L0 + 1; m++)
                            {
                                for (int channel = 0; channel < 4; channel++)
                                    nlm_target[index + channel * length] = nlm[channel][iw + m];
                                index++;
                            }
                        }
                    }
                    nlm_iat0[ad].insert({all_indexes[iw1l], nlm_target});
                }
            }

            // Iterate over atom pairs and compute forces
            for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
            {
                const int T1 = adjs.ntype[ad1];
                const int I1 = adjs.natom[ad1];
                const int iat1 = this->ucell->itia2iat(T1, I1);
                const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad1];

                for (int ad2 = 0; ad2 < adjs.adj_num + 1; ++ad2)
                {
                    const int T2 = adjs.ntype[ad2];
                    const int I2 = adjs.natom[ad2];
                    const int iat2 = this->ucell->itia2iat(T2, I2);
                    ModuleBase::Vector3<int>& R_index2 = adjs.box[ad2];
                    int r_vector[3] = {R_index2.x - R_index1.x, R_index2.y - R_index1.y, R_index2.z - R_index1.z};

                    const hamilt::BaseMatrix<double>* dmR_pointer =
                        dmR->find_matrix(iat1, iat2, r_vector[0], r_vector[1], r_vector[2]);
                    if (dmR_pointer == nullptr) continue;

                    if (cal_force)
                    {
                        double force1[3] = {0, 0, 0};
                        double force2[3] = {0, 0, 0};
                        cal_force_IJR(iat1, iat2, paraV, nlm_iat0[ad1], nlm_iat0[ad2],
                                       dmR_pointer, lam, force1, force2);

                        for (int ipol = 0; ipol < 3; ipol++)
                        {
                            force_local(iat1, ipol) += force1[ipol];
                            force_local(iat2, ipol) += force2[ipol];
                        }
                    }

                    if (cal_stress)
                    {
                        cal_stress_IJR(iat1, iat2, r_vector, paraV, nlm_iat0[ad1], nlm_iat0[ad2],
                                        dmR_pointer, lam, gdir, stress_local.data());
                    }
                }
            }
        }

        #pragma omp critical(cal_fs_deltap)
        {
            if (cal_force)
                force += force_local;

            if (cal_stress)
                for (int i = 0; i < 6; i++)
                    stress_tmp[i] += stress_local[i];
        }
    }

    if (cal_force)
    {
        force = force * 2.0;  // Hermitian conjugate contribution
        Parallel_Reduce::reduce_all(force.c, force.nr * force.nc);
    }

    if (cal_stress)
    {
        Parallel_Reduce::reduce_all(stress_tmp.data(), 6);
        stress.zero_out();
        stress.c[0] = stress_tmp[0] / this->ucell->omega; // xx
        stress.c[1] = stress_tmp[1] / this->ucell->omega; // xy
        stress.c[2] = stress_tmp[2] / this->ucell->omega; // xz
        stress.c[3] = stress_tmp[1] / this->ucell->omega; // yx
        stress.c[4] = stress_tmp[3] / this->ucell->omega; // yy
        stress.c[5] = stress_tmp[4] / this->ucell->omega; // yz
        stress.c[6] = stress_tmp[2] / this->ucell->omega; // zx
        stress.c[7] = stress_tmp[4] / this->ucell->omega; // zy
        stress.c[8] = stress_tmp[5] / this->ucell->omega; // zz
    }

    ModuleBase::timer::end("DeltaPOperator", "cal_force_stress");
}

template <typename TK, typename TR>
void DeltaPOperator<TK, TR>::cal_force_IJR(const int& iat1,
                                            const int& iat2,
                                            const Parallel_Orbitals* paraV,
                                            const std::unordered_map<int, std::vector<double>>& nlm1_all,
                                            const std::unordered_map<int, std::vector<double>>& nlm2_all,
                                            const hamilt::BaseMatrix<double>* dmR_pointer,
                                            double lambda,
                                            double* force1,
                                            double* force2)
{
    const int npol = this->ucell->get_npol();
    const int nspin = (npol == 2) ? 4 : 2;  // DM spin channels

    auto row_indexes = paraV->get_indexes_row(iat1);
    auto col_indexes = paraV->get_indexes_col(iat2);

    std::vector<int> step_trace(nspin, 0);
    if (nspin == 4)
    {
        step_trace[1] = 1;
        step_trace[2] = col_indexes.size();
        step_trace[3] = col_indexes.size() + 1;
    }

    double tmp[3] = {0.0};
    for (int is = 1; is < nspin; is++)
    {
        const double* dm_pointer = dmR_pointer->get_pointer();
        for (size_t iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
        {
            auto it1 = nlm1_all.find(row_indexes[iw1l]);
            if (it1 == nlm1_all.end()) { dm_pointer += npol * col_indexes.size(); continue; }
            const std::vector<double>& nlm1 = it1->second;

            for (size_t iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
            {
                auto it2 = nlm2_all.find(col_indexes[iw2l]);
                if (it2 == nlm2_all.end()) { dm_pointer += npol; continue; }
                const std::vector<double>& nlm2 = it2->second;

                const int nlm_size = nlm1.size();
                const int length = nlm_size / 4;
                const int lmax = sqrt(length);

                int index = 0;
                for (int l = 0; l < lmax; l++)
                {
                    for (int m = 0; m < 2 * l + 1; m++)
                    {
                        index = l * l + m;
                        // Force = -lambda * (d<nlm1|/dR * nlm2 + nlm1 * d<nlm2|/dR) * DM
                        // nlm layout: [value(0..length-1) | deri_x(length..2*len-1) | deri_y(2*len..3*len-1) | deri_z(3*len..4*len-1)]
                        double dbb = nlm1[index + length] * nlm2[index] * dm_pointer[step_trace[is]];
                        tmp[0] = lambda * dbb;
                        dbb = nlm1[index + length * 2] * nlm2[index] * dm_pointer[step_trace[is]];
                        tmp[1] = lambda * dbb;
                        dbb = nlm1[index + length * 3] * nlm2[index] * dm_pointer[step_trace[is]];
                        tmp[2] = lambda * dbb;

                        force1[0] += tmp[0]; force1[1] += tmp[1]; force1[2] += tmp[2];
                        force2[0] -= tmp[0]; force2[1] -= tmp[1]; force2[2] -= tmp[2];
                    }
                }
                dm_pointer += npol;
            }
        }
    }
}

template <typename TK, typename TR>
void DeltaPOperator<TK, TR>::cal_stress_IJR(const int& iat1,
                                             const int& iat2,
                                             const int* r_vector,
                                             const Parallel_Orbitals* paraV,
                                             const std::unordered_map<int, std::vector<double>>& nlm1_all,
                                             const std::unordered_map<int, std::vector<double>>& nlm2_all,
                                             const hamilt::BaseMatrix<double>* dmR_pointer,
                                             double lambda,
                                             int gdir,
                                             double* stress)
{
    const int npol = this->ucell->get_npol();
    const int nspin = (npol == 2) ? 4 : 2;

    auto row_indexes = paraV->get_indexes_row(iat1);
    auto col_indexes = paraV->get_indexes_col(iat2);

    std::vector<int> step_trace(nspin, 0);
    if (nspin == 4)
    {
        step_trace[1] = 1;
        step_trace[2] = col_indexes.size();
        step_trace[3] = col_indexes.size() + 1;
    }

    // Stress: σ_αβ = (1/Ω) * Σ dE/dε_αβ
    // For atom-pair (R1, R2), stress contribution:
    // σ_αβ += F_α(R1) * R1_β + F_α(R2) * R2_β
    // (simplified — the exact formula needs λ * deriv * DM * R_vector projection)

    double tmp[3] = {0.0};
    for (int is = 1; is < nspin; is++)
    {
        const double* dm_pointer = dmR_pointer->get_pointer();
        for (size_t iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
        {
            auto it1 = nlm1_all.find(row_indexes[iw1l]);
            if (it1 == nlm1_all.end()) { dm_pointer += npol * col_indexes.size(); continue; }
            const std::vector<double>& nlm1 = it1->second;

            for (size_t iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
            {
                auto it2 = nlm2_all.find(col_indexes[iw2l]);
                if (it2 == nlm2_all.end()) { dm_pointer += npol; continue; }
                const std::vector<double>& nlm2 = it2->second;

                const int nlm_size = nlm1.size();
                const int length = nlm_size / 4;
                const int lmax = sqrt(length);

                int index = 0;
                for (int l = 0; l < lmax; l++)
                {
                    for (int m = 0; m < 2 * l + 1; m++)
                    {
                        index = l * l + m;
                        double dbb = lambda * nlm1[index + length] * nlm2[index] * dm_pointer[step_trace[is]];
                        tmp[0] = dbb;
                        dbb = lambda * nlm1[index + length * 2] * nlm2[index] * dm_pointer[step_trace[is]];
                        tmp[1] = dbb;
                        dbb = lambda * nlm1[index + length * 3] * nlm2[index] * dm_pointer[step_trace[is]];
                        tmp[2] = dbb;

                        // Stress: σ_αβ = Σ_k F_α[k] * R_vector[k]_β
                        for (int ipol = 0; ipol < 3; ipol++)
                        {
                            double F_alpha = tmp[ipol];
                            stress[ipol * 3 + 0] += F_alpha * r_vector[0];  // σ_x,ε  → xx,xy,xz
                            stress[ipol * 3 + 1] += F_alpha * r_vector[1];
                            stress[ipol * 3 + 2] += F_alpha * r_vector[2];
                        }
                    }
                }
                dm_pointer += npol;
            }
        }
    }
}

} // namespace hamilt
