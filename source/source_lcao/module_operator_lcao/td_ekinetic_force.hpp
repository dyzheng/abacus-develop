#pragma once
#include "td_ekinetic_lcao.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/libm/libm.h"
#include "source_estate/module_dm/cal_dm_psi.h"
namespace hamilt
{
template <typename TK, typename TR>
void TDEkinetic<OperatorLCAO<TK, TR>>::cal_force(
    const bool cal_force,
    const Parallel_Orbitals* paraV,
    const psi::Psi<std::complex<double>>* psi,
    const elecstate::ElecState* pelec,
    ModuleBase::matrix& force)
{
    const int npol = ucell->get_npol();
    ModuleBase::Vector3<double> At = TD_info::cart_At;
    if (cal_force)
    {
        force.zero_out();
    }
    else return;
    // calculate dmr
    const int nspin0 = PARAM.inp.nspin;
    const int nspin_dm = std::map<int, int>({ {1,1},{2,2},{4,1} })[nspin0];
    elecstate::DensityMatrix<std::complex<double>, double> tmp_dm(paraV, nspin_dm, kv->kvec_d, kv->get_nks() / nspin_dm);
    elecstate::cal_dm_psi(paraV, pelec->wg, psi[0], tmp_dm);
    tmp_dm.init_DMR(Grid, ucell);
    tmp_dm.cal_DMR();
    if (PARAM.inp.nspin == 2)
    {
        tmp_dm.switch_dmr(1);
    }
    // Convert to complex HContainer for TDDFT velocity gauge
    hamilt::HContainer<std::complex<double>> dmR_complex(paraV);
    tmp_dm.cal_DMR_full(&dmR_complex);
    const hamilt::HContainer<std::complex<double>>* dmR = &dmR_complex;
    // Loop over all atom pairs and calculate force contributions
    #pragma omp parallel
    {
        ModuleBase::matrix force_local(force.nr, force.nc);

        #pragma omp for schedule(dynamic)
        for (int iat1 = 0; iat1 < ucell->nat; iat1++)
        {
            auto tau1 = ucell->get_tau(iat1);
            int T1 = 0, I1 = 0;
            ucell->iat2iait(iat1, &I1, &T1);
            Atom& atom1 = ucell->atoms[T1];

            // Find adjacent atoms
            AdjacentAtomInfo adjs;
            Grid->Find_atom(*ucell, tau1, T1, I1, &adjs);

            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T2 = adjs.ntype[ad];
                const int I2 = adjs.natom[ad];
                const int iat2 = ucell->itia2iat(T2, I2);
                const ModuleBase::Vector3<int>& R_index = adjs.box[ad];

                // Check cutoff
                ModuleBase::Vector3<double> dtau = ucell->cal_dtau(iat1, iat2, R_index);
                if (dtau.norm() * ucell->lat0 >= orb_cutoff_[T1] + orb_cutoff_[T2])
                {
                    continue;
                }

                // Find density matrix for this atom pair
                const hamilt::BaseMatrix<std::complex<double>>* dm_matrix = dmR->find_matrix(iat1, iat2, R_index[0], R_index[1], R_index[2]);
                if (dm_matrix == nullptr)
                {
                    continue;
                }

                // Calculate force for this atom pair
                double* force_tmp1 = (cal_force) ? &force_local(iat1, 0) : nullptr;
                double* force_tmp2 = (cal_force) ? &force_local(iat2, 0) : nullptr;

                Atom& atom2 = ucell->atoms[T2];
                auto row_indexes = paraV->get_indexes_row(iat1);
                auto col_indexes = paraV->get_indexes_col(iat2);

                if (row_indexes.size() == 0 || col_indexes.size() == 0)
                {
                    continue;
                }
                const std::complex<double>* dm_pointer = dm_matrix->get_pointer();
                double overlap = 0;
                double grad[3] = {0, 0, 0};
                double hess[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

                // step_trace = 0 for npol=1; ={0, 1, col_size, col_size+1} for npol=2
                std::vector<int> step_trace(npol * npol, 0);
                if (npol == 2)
                {
                    step_trace[1] = 1;
                    step_trace[2] = col_indexes.size();
                    step_trace[3] = col_indexes.size() + 1;
                }

                // Loop over orbital pairs
                for (int iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
                {
                    const int iw1 = row_indexes[iw1l] / npol;
                    const int L1 = atom1.iw2l[iw1];
                    const int N1 = atom1.iw2n[iw1];
                    const int m1 = atom1.iw2m[iw1];
                    const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;

                    for (int iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
                    {
                        const int iw2 = col_indexes[iw2l] / npol;
                        const int L2 = atom2.iw2l[iw2];
                        const int N2 = atom2.iw2n[iw2];
                        const int m2 = atom2.iw2m[iw2];
                        const int M2 = (m2 % 2 == 0) ? -m2 / 2 : (m2 + 1) / 2;

                        // Calculate integral and its gradient using provided functor
                        intor_->calculate(T1, L1, N1, M1, T2, L2, N2, M2, dtau * this->ucell->lat0, &overlap, grad, hess);

                        // Calculate force contribution with compile-time sign
                        if (cal_force)
                        {
                            // Factor of 2 for Hermitian matrix will be applied later
                            for (int i = 0; i < 3; i++)
                            {
                                // force_tmp1[i] += 2 * dm_pointer[0] * grad[i] * At * At;
                                force_tmp2[i] -= dm_pointer[0].real() * grad[i] * At * At;
                                for (int j = 0; j < 3; j++)
                                {
                                    force_tmp2[i] -= 2 * dm_pointer[0].imag() * hess[j * 3 + i] * At[j] * At[j];
                                }
                            }
                        }
                        dm_pointer += npol;
                    }
                    dm_pointer += (npol - 1) * col_indexes.size();
                }
            }
        }

        #pragma omp critical
        {
            if (cal_force)
            {
                force += force_local;
            }
        }
    }

    // Finalize with MPI reduction and post-processing
    if (cal_force)
    {
#ifdef __MPI
        Parallel_Reduce::reduce_all(force.c, force.nr * force.nc);
#endif
    }
}
}// namespace hamilt
