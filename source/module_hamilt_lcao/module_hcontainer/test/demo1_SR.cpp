#include "../hcontainer.h"
#include "module_basis/module_ao/ORB_gen_tables.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/operator_lcao.h"
#include "overlap_new.h"

template <typename T>
hamilt::OverlapNew<hamilt::OperatorLCAO<T>>::OverlapNew(LCAO_Matrix* LM_in,
                                                        const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                                                        hamilt::HContainer<T>* SR_in,
                                                        std::vector<T>* SK_pointer_in,
                                                        const UnitCell* ucell_in,
                                                        Grid_Driver* GridD_in,
                                                        const Parallel_Orbitals* paraV)
    : OperatorLCAO<T>(LM_in, kvec_d_in)
{
    this->ucell = ucell_in;
    this->SR = SR_in;
    this->SK_pointer = SK_pointer_in;
#ifdef __DEBUG
    assert(this->ucell != nullptr);
    assert(this->SR != nullptr);
    assert(this->SK_pointer != nullptr);
#endif
    // initialize SR to allocate sparse overlap matrix memory
    this->initialize_SR(GridD_in);
}

// initialize_SR()
template <typename T>
void hamilt::OverlapNew<hamilt::OperatorLCAO<T>>::initialize_SR(Grid_Driver* GridD, const Parallel_Orbitals* paraV)
{
    for (int iat1 = 0; iat1 < ucell.nat; iat1++)
    {
        auto tau1 = ucell.get_tau(iat1);
        int T1, I1;
        ucell.iat2iait(iat1, &I1, &T1);
        GridD.Find_atom(ucell, tau1, T1, I1);
        for (int ad = 0; ad < GridD.getAdjacentNum() + 1; ++ad)
        {
            int iat2 = GridD.getAdjacent(ad);
            auto R_index = GridD.getBox(ad);
            hamilt::AtomPair<double> tmp(iat1, iat2, paraV);
            tmp.get_HR_values(R_index.x, R_index.y, R_index.z);
        }
    }
}

template <typename T>
void hamilt::OverlapNew<hamilt::OperatorLCAO<T>>::calculate_SR()
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int iap = 0; iap < this->SR->size_atom_pairs(); ++iap)
    {
        hamilt::AtomPair<T>& tmp = this->SR->get_atom_pair(iap);
        int iat1 = tmp.get_atom_i();
        int iat2 = tmp.get_atom_j();
        const Parallel_Orbitals* paraV = tmp.get_paraV();

        for (int iR = 0; iR < tmp.get_R_size(); ++iR)
        {
            const int* R_index = tmp.get_R_index(iR);
            auto dtau = ucell.get_dtau(iat1, iat2, R_index[0], R_index[1], R_index[2]);
            T* data_pointer = tmp.get_pointer(iR);
            this->cal_SR_IJR(iat1, iat2, paraV, dtau, data_pointer);
        }
    }
}

//cal_SR_IJR()
template <typename T>
void hamilt::OverlapNew<hamilt::OperatorLCAO<T>>::cal_SR_IJR(
    const int& iat1, 
    const int& iat2, 
    const Parallel_Orbitals* paraV,
    const ModuleBase::Vector3<double>& dtau,
    T* data_pointer )
{
    int T1, I1;
    this->ucell->iat2iait(iat1, &I1, &T1);
    int T2, I2;
    this->ucell->iat2iait(iat2, &I2, &T2);
    int nw1 = this->ucell->get_nw(T1);
    int nw2 = this->ucell->get_nw(T2);
    const int* iw2l1 = this->ucell->atoms[T1].iw2l;
    const int* iw2n1 = this->ucell->atoms[T1].iw2n;
    const int* iw2m1 = this->ucell->atoms[T1].iw2m;
    const int* iw2l2 = this->ucell->atoms[T2].iw2l;
    const int* iw2n2 = this->ucell->atoms[T2].iw2n;
    const int* iw2m2 = this->ucell->atoms[T2].iw2m;
    double olm[3] = {0, 0, 0};
    for (int iw1 = 0; iw1 < nw1; ++iw1)
    {
        if (paraV->skip_row(iat1, iw1))
            continue;
        const int L1 = iw2l1[iw1];
        const int N1 = iw2n1[iw1];
        const int m1 = iw2m1[iw1];
        for (int iw2 = 0; iw2 < nw2; ++iw2)
        {
            if (paraV->skip_col(iat2, iw2))
                continue;
            const int L2 = iw2l2[iw2];
            const int N2 = iw2n2[iw2];
            const int m2 = iw2m2[iw2];
            GlobalC::UOT.snap_psipsi(GlobalC::ORB,
                                        olm,
                                        0,
                                        'S',
                                        tau1,
                                        T1,
                                        L1,
                                        m1,
                                        N1,
                                        tau2,
                                        T2,
                                        L2,
                                        m2,
                                        N2,
                                        GlobalV::NSPIN,
                                        nullptr, // for soc
                                        false,
                                        0.0);
            *data_pointer++ = olm[0];
        }
    }
}

// contributeHR()
template <typename T>
void hamilt::OverlapNew<hamilt::OperatorLCAO<T>>::contributeHR()
{
    if (this->SR_fixed_done)
    {
        return;
    }
    this->calculate_SR();
    this->SR_fixed_done = true;
}

// contributeHk()
template <typename T>
void hamilt::OverlapNew<hamilt::OperatorLCAO<T>>::contributeHk(int ik)
{
    const int ncol = this->LM->ParaV->ncol;
    folding_HR(*this->SR, this->SK_pointer, this->kvec_d[ik], ncol);
}

template class hamilt::OverlapNew<hamilt::OperatorLCAO<double>>;
template class hamilt::OverlapNew<hamilt::OperatorLCAO<std::complex<double>>>;