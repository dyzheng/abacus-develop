#include "op_dftu_lcao.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_lcao/module_dftu/dftu.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace hamilt
{

template class OperatorDFTU<OperatorLCAO<double, double>>;

template class OperatorDFTU<OperatorLCAO<std::complex<double>, double>>;

template class OperatorDFTU<OperatorLCAO<std::complex<double>, std::complex<double>>>;

template<typename TK, typename TR>
OperatorDFTU<OperatorLCAO<TK, TR>>::OperatorDFTU(LCAO_Matrix* LM_in,
                                  const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                                  hamilt::HContainer<TR>* hR_in,
                                  hamilt::HContainer<TR>* sR_in,
                                  std::vector<TK>* hK_in,
                                  const UnitCell* ucell_in,
                                  const std::vector<int>& isk_in)
    : isk(isk_in), OperatorLCAO<TK, TR>(LM_in, kvec_d_in, hR_in, hK_in), sR_(sR_in)
{
    this->cal_type = lcao_dftu;
    this->initialize(ucell_in);
}

// destructor
template<typename TK, typename TR>
OperatorDFTU<OperatorLCAO<TK, TR>>::~OperatorDFTU()
{
    delete this->VU_;
}

template<typename TK, typename TR>
void OperatorDFTU<OperatorLCAO<TK, TR>>::initialize(const UnitCell* ucell_in)
{
    ModuleBase::TITLE("OperatorDFTU", "initialize");
    if(this->VU_ != nullptr)
    {
        delete this->VU_;
    }
    this->VU_ = new hamilt::HContainer<TR>(ucell_in->nat);
    
    std::vector<int> orb_index(ucell_in->nat + 1);
	orb_index[0] = 0;
	for(int i=1;i<orb_index.size();i++)
	{
		int type = ucell_in->iat2it[i-1];
		orb_index[i] = orb_index[i-1] + ucell_in->atoms[type].nw;
	}

    for(int it = 0; it < ucell_in->ntype; it++)
    {
        if(GlobalC::dftu.orbital_corr[it] == -1)
        {
            continue;
        }
        for(int ia = 0; ia < ucell_in->atoms[it].na; ia++)
        {
            const int iat = ucell_in->itia2iat(it, ia);
            hamilt::AtomPair<TR> atom_pair(iat, iat, orb_index.data(), orb_index.data(), ucell_in->nat);
            this->VU_->insert_pair(atom_pair);
        }
        this->VU_->allocate(nullptr, 0);
    }
    return;
}

template<typename TK, typename TR>
void OperatorDFTU<OperatorLCAO<TK, TR>>::contributeHR()
{
    ModuleBase::TITLE("OperatorDFTU", "contributeHR");
    ModuleBase::timer::tick("OperatorDFTU", "contributeHR");
    //update VU_ for current spin
    this->VU_->set_zero();
    GlobalC::dftu.cal_VU_pot_atompair(GlobalV::CURRENT_SPIN, true, this->VU_);
    // 1/2 * VU_(i,i) * SR_(i,j,R) + 1/2 * SR_(i,j,R) * VU_(j, j)) -> hR(i,j,R)
    for(int iap = 0;iap<this->sR_->size_atom_pairs();iap++)
    {
        hamilt::AtomPair<TR>& sR_ap = this->sR_->get_atom_pair(iap);
        const int iat = sR_ap.get_atom_i();
        const int jat = sR_ap.get_atom_j();
        hamilt::BaseMatrix<TR>* vu1 = this->VU_->find_matrix(iat, iat, 0, 0, 0);
        hamilt::BaseMatrix<TR>* vu2 = this->VU_->find_matrix(jat, jat, 0, 0, 0);
        if(vu1 == nullptr && vu2 == nullptr) continue;
        const int row_size = sR_ap.get_row_size();
        const int col_size = sR_ap.get_col_size();
        for(int ir=0;ir<sR_ap.get_R_size();ir++)
        {
            const int* rindex = sR_ap.get_R_index(ir);
            hamilt::BaseMatrix<TR>* tmp_hR = this->hR->find_matrix(iat, jat, rindex[0], rindex[1], rindex[2]);

            constexpr char transa='N', transb='N';
            const double gemm_alpha = 0.5, gemm_beta = 1.0;
            if(vu1 != nullptr)
            {
                dgemm_(
                    &transa, &transb, 
                    &col_size, 
                    &row_size,
                    &row_size, 
                    &gemm_alpha, 
                    sR_ap.get_pointer(ir),  
                    &col_size,     
                    vu1->get_pointer(), 
                    &row_size,
                    &gemm_beta,      
                    tmp_hR->get_pointer(),    
                    &col_size);
            }
            if(vu2 != nullptr)
            {
                dgemm_(
                    &transa, &transb, 
                    &col_size, 
                    &row_size,
                    &col_size, 
                    &gemm_alpha, 
                    vu2->get_pointer(),  
                    &col_size, 
                    sR_ap.get_pointer(ir), 
                    &col_size,    
                    &gemm_beta,      
                    tmp_hR->get_pointer(),    
                    &col_size);
            }
        }
    }
    ModuleBase::timer::tick("OperatorDFTU", "contributeHR");
    return;
}

template<>
void OperatorDFTU<OperatorLCAO<std::complex<double>, std::complex<double>>>::contributeHR()
{
    ModuleBase::TITLE("OperatorDFTU", "contributeHR");
    //do nothing
}

template<>
void OperatorDFTU<OperatorLCAO<double, double>>::contributeHk(int ik)
{
    ModuleBase::TITLE("OperatorDFTU", "contributeHk");
    ModuleBase::timer::tick("OperatorDFTU", "contributeHk");
    // do nothing
    // Effective potential of DFT+U is added to total Hamiltonian here; Quxin adds on 20201029
    /*std::vector<double> eff_pot(this->LM->ParaV->nloc);
    GlobalC::dftu.cal_eff_pot_mat_real(ik, &eff_pot[0], isk);

    for (int irc = 0; irc < this->LM->ParaV->nloc; irc++)
    {
        this->LM->Hloc[irc] += eff_pot[irc];
    }*/

    ModuleBase::timer::tick("OperatorDFTU", "contributeHk");
}

template<>
void OperatorDFTU<OperatorLCAO<std::complex<double>, double>>::contributeHk(int ik)
{
    ModuleBase::TITLE("OperatorDFTU", "contributeHk");
    /*ModuleBase::timer::tick("OperatorDFTU", "contributeHk");
    // Effective potential of DFT+U is added to total Hamiltonian here; Quxin adds on 20201029
    std::vector<std::complex<double>> eff_pot(this->LM->ParaV->nloc);
    GlobalC::dftu.cal_eff_pot_mat_complex(ik, &eff_pot[0], isk);

    for (int irc = 0; irc < this->LM->ParaV->nloc; irc++)
    {
        this->LM->Hloc2[irc] += eff_pot[irc];
    }

    ModuleBase::timer::tick("OperatorDFTU", "contributeHk");
        */
}

template<>
void OperatorDFTU<OperatorLCAO<std::complex<double>, std::complex<double>>>::contributeHk(int ik)
{
    ModuleBase::TITLE("OperatorDFTU", "contributeHk");
    ModuleBase::timer::tick("OperatorDFTU", "contributeHk");
    // Effective potential of DFT+U is added to total Hamiltonian here; Quxin adds on 20201029
    std::vector<std::complex<double>> eff_pot(this->LM->ParaV->nloc);
    GlobalC::dftu.cal_eff_pot_mat_complex(ik, &eff_pot[0], isk);

    for (int irc = 0; irc < this->LM->ParaV->nloc; irc++)
    {
        this->LM->Hloc2[irc] += eff_pot[irc];
    }

    ModuleBase::timer::tick("OperatorDFTU", "contributeHk");
}

}