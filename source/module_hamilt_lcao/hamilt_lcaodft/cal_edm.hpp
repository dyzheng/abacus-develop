#ifndef MODULE_HAMILT_LCAODFT_CAL_EDM_HPP
#define MODULE_HAMILT_LCAODFT_CAL_EDM_HPP

#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_cell/klist.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_base/lapack_connector.h"
#include "module_base/scalapack_connector.h"
#include <vector>
#include <complex>

namespace hamilt
{
/**
 * @brief calculate the energy density matrix
 * EDM = 0.5 * (S^-1 * H * DM + DM * H * S^-1)
 * S contains the overlap matrix, comes from the HamiltLCAO::matrix function
 * H contains the hamiltonian matrix, comes from the HamiltLCAO::matrix function
 * S and H are calculated in the HamiltLCAO::updateHk(ik) function for each k point
 * DM is the density matrix, comes from the ElecStateLCAO::get_DM() function
 * EDM is stored in the input edmk
 * parallel information is stored in the Parallel_Orbitals class
 * @param ParaV the parallel orbitals
 * @param kv the k point
 * @param p_hamilt the hamiltonian
 * 
 */
template <typename T>
void cal_edm(
    const Parallel_Orbitals* pv, 
    const int nks, 
    Hamilt<T>* p_hamilt,
    const std::vector<std::vector<T>>& dmk,
    std::vector<std::vector<std::complex<double>>>& edmk);

template <>
void cal_edm(
    const Parallel_Orbitals* pv, 
    const int nks, 
    Hamilt<std::complex<double>>* p_hamilt,
    const std::vector<std::vector<std::complex<double>>>& dmk,
    std::vector<std::vector<std::complex<double>>>& edmk)
{
    const int* desc = pv->desc;
    const long nloc = pv->nloc;
    const int ncol = pv->ncol;
    const int nrow = pv->nrow;
    const int nlocal = desc[2];

    auto* hamilt = dynamic_cast<HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_hamilt);

#ifdef __MPI
    std::vector<std::complex<double>> Htmp(nloc);
    std::vector<std::complex<double>> Sinv(nloc);
    std::vector<std::complex<double>> tmp1(nloc);
    std::vector<std::complex<double>> tmp2(nloc);
    std::vector<std::complex<double>> tmp3(nloc);
    std::vector<std::complex<double>> tmp4(nloc);
#endif
    for (int ik = 0; ik < nks; ++ik) 
    {
        const std::complex<double>* p_dmk = dmk[ik].data();

#ifdef __MPI

        // mohan add 2024-03-27
        //! be careful, the type of nloc is 'long'
        //! whether the long type is safe, needs more discussion

        Htmp.assign(nloc, 0);
        Sinv.assign(nloc, 0);
        tmp1.assign(nloc, 0);
        tmp2.assign(nloc, 0);
        tmp3.assign(nloc, 0);
        tmp4.assign(nloc, 0);

        const int inc = 1;
        
        // update HK and SK for each k point
        p_hamilt->updateHk(ik);

        hamilt::MatrixBlock<complex<double>> h_mat;
        hamilt::MatrixBlock<complex<double>> s_mat;

        p_hamilt->matrix(h_mat, s_mat);
        zcopy_(&nloc, h_mat.p, &inc, Htmp.data(), &inc);
        zcopy_(&nloc, s_mat.p, &inc, Sinv.data(), &inc);

        vector<int> ipiv(nloc, 0);
        int info = 0;
        const int one_int = 1;

        pzgetrf_(&nlocal, &nlocal, Sinv.data(), &one_int, &one_int, desc, ipiv.data(), &info);

        int lwork = -1;
        int liwork = -1;

        // if lwork == -1, then the size of work is (at least) of length 1.
        std::vector<std::complex<double>> work(1, 0);

        // if liwork = -1, then the size of iwork is (at least) of length 1.
        std::vector<int> iwork(1, 0);

        pzgetri_(&nlocal,
                 Sinv.data(),
                 &one_int,
                 &one_int,
                 desc,
                 ipiv.data(),
                 work.data(),
                 &lwork,
                 iwork.data(),
                 &liwork,
                 &info);

        lwork = work[0].real();
        work.resize(lwork, 0);
        liwork = iwork[0];
        iwork.resize(liwork, 0);

        pzgetri_(&nlocal,
                 Sinv.data(),
                 &one_int,
                 &one_int,
                 desc,
                 ipiv.data(),
                 work.data(),
                 &lwork,
                 iwork.data(),
                 &liwork,
                 &info);

        const char N_char = 'N';
        const char T_char = 'T';
        const complex<double> one_float = {1.0, 0.0};
        const complex<double> zero_float = {0.0, 0.0};
        const complex<double> half_float = {0.5, 0.0};

        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                p_dmk,
                &one_int,
                &one_int,
                desc,
                Htmp.data(),
                &one_int,
                &one_int,
                desc,
                &zero_float,
                tmp1.data(),
                &one_int,
                &one_int,
                desc);

        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                tmp1.data(),
                &one_int,
                &one_int,
                desc,
                Sinv.data(),
                &one_int,
                &one_int,
                desc,
                &zero_float,
                tmp2.data(),
                &one_int,
                &one_int,
                desc);

        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                Sinv.data(),
                &one_int,
                &one_int,
                desc,
                Htmp.data(),
                &one_int,
                &one_int,
                desc,
                &zero_float,
                tmp3.data(),
                &one_int,
                &one_int,
                desc);

        pzgemm_(&N_char,
                &N_char,
                &nlocal,
                &nlocal,
                &nlocal,
                &one_float,
                tmp3.data(),
                &one_int,
                &one_int,
                desc,
                p_dmk,
                &one_int,
                &one_int,
                desc,
                &zero_float,
                tmp4.data(),
                &one_int,
                &one_int,
                desc);

        pzgeadd_(&N_char,
                 &nlocal,
                 &nlocal,
                 &half_float,
                 tmp2.data(),
                 &one_int,
                 &one_int,
                 desc,
                 &half_float,
                 tmp4.data(),
                 &one_int,
                 &one_int,
                 desc);

        zcopy_(&nloc, tmp4.data(), &inc, edmk[ik].data(), &inc);

#else
        // for serial version
        ModuleBase::ComplexMatrix tmp_edmk(nlocal, nlocal);
        ModuleBase::ComplexMatrix Sinv(nlocal, nlocal);
        ModuleBase::ComplexMatrix Htmp(nlocal, nlocal);

        hamilt::MatrixBlock<std::complex<double>> h_mat;
        hamilt::MatrixBlock<std::complex<double>> s_mat;

        p_hamilt->matrix(h_mat, s_mat);
        // cout<<"hmat "<<h_mat.p[0]<<endl;
        for (int i = 0; i < nlocal; i++)
        {
            for (int j = 0; j < nlocal; j++)
            {
                Htmp(i, j) = h_mat.p[i * nlocal + j];
                Sinv(i, j) = s_mat.p[i * nlocal + j];
            }
        }
        int INFO = 0;

        int lwork = 3 * nlocal - 1; // tmp
        std::complex<double>* work = new std::complex<double>[lwork];
        ModuleBase::GlobalFunc::ZEROS(work, lwork);

        int IPIV[nlocal];

        LapackConnector::zgetrf(nlocal, nlocal, Sinv, nlocal, IPIV, &INFO);
        LapackConnector::zgetri(nlocal, Sinv, nlocal, IPIV, work, lwork, &INFO);
        delete[] work;
        // I just use ModuleBase::ComplexMatrix temporarily, and will change it
        // to complex<double>*
        ModuleBase::ComplexMatrix tmp_dmk_base(nlocal, nlocal);
        for (int i = 0; i < nlocal; i++)
        {
            for (int j = 0; j < nlocal; j++)
            {
                tmp_dmk_base(i, j) = p_dmk[i * nlocal + j];
            }
        }
        tmp_edmk = 0.5 * (Sinv * Htmp * tmp_dmk_base + tmp_dmk_base * Htmp * Sinv);
        int inc = 1;
        long nloc = nlocal * nlocal;
        zcopy_(&nloc, tmp_edmk.c, &inc, edmk[ik].data(), &inc);
#endif
    }
}

template <>
void cal_edm(
    const Parallel_Orbitals* pv, 
    const int nks, 
    Hamilt<double>* p_hamilt,
    const std::vector<std::vector<double>>& dmk,
    std::vector<std::vector<std::complex<double>>>& edmk)
{
    return;
}

}//namespace hamilt

#endif