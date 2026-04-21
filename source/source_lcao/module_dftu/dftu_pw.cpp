#include "dftu.h"
#include "source_estate/module_charge/charge_mixing.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_base/parallel_reduce.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"


/// calculate occupation matrix for DFT+U
void Plus_U::cal_occ_pw(const int iter,
		const void* psi_in,
		const ModuleBase::matrix& wg_in,
		const UnitCell& cell,
		Charge_Mixing* p_chgmix)
{
    ModuleBase::timer::tick("Plus_U", "cal_occ_pw");
    this->copy_locale(cell);

    if(this->initialed_locale == false)
    {
    this->zero_locale(cell);

#if 0  // DIAG disabled
    // DIAGNOSTIC [N4]: state right after zero_locale
    std::cout << "[DIAG-PW] cal_occ_pw iter=" << iter << " right after zero_locale:" << std::endl;
    std::cout << "[DIAG-PW]   uom_array.size=" << this->uom_array.size()
              << " eff_pot_pw.size=" << this->eff_pot_pw.size() << std::endl;
    if (this->uom_array.size() > 0) {
        std::cout << "[DIAG-PW]   uom_array[0..9]=";
        for(int i=0;i<10 && i<(int)this->uom_array.size();i++) std::cout << " " << this->uom_array[i];
        std::cout << std::endl;
    }
#endif

    if(PARAM.inp.device == "cpu")
    {
        auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
        const psi::Psi<std::complex<double>>* psi_p = (const psi::Psi<std::complex<double>>*)psi_in;
        // loop over k-points to calculate Mi of \sum_{k,i,l,m}<Psi_{k,i}|alpha_{l,m}><alpha_{l,m}|Psi_{k,i}>
        const int nbands = psi_p->get_nbands();
        for(int ik = 0; ik < psi_p->get_nk(); ik++)
        {
            int is = 0;
            if(PARAM.inp.nspin == 2 && ik >= psi_p->get_nk()/2)
            {
                is = 1;
            }
            psi_p->fix_k(ik);
            onsite_p->tabulate_atomic(ik);

            // DIAGNOSTIC: print psi pointer and psi values BEFORE overlap_proj_psi
            const std::complex<double>* psi_ptr_before = psi_p->get_pointer();
            std::cout << "[PSI-BEFORE] pw-port ik=" << ik << " psi_ptr=" << psi_ptr_before << " nbands=" << nbands << " npol=" << psi_p->get_npol() << std::endl;
            std::cout << "[PSI-BEFORE] pw-port ik=" << ik << " psi[0..9]=";
            for(int i=0;i<10;i++) std::cout << " (" << psi_ptr_before[i].real() << "," << psi_ptr_before[i].imag() << ")";
            std::cout << std::endl;

            onsite_p->overlap_proj_psi(nbands*psi_p->get_npol(), psi_p->get_pointer());
            const std::complex<double>* becp = onsite_p->get_h_becp();
            // becp(nbands*npol , nkb)
            // mag = wg * \sum_{nh}becp * becp
            int nkb = onsite_p->get_size_becp() / nbands / psi_p->get_npol();
            std::cout << "[BECP] pw-port ik=" << ik << " becp_ptr=" << becp << " nkb=" << nkb << " size=" << nbands*psi_p->get_npol()*nkb << std::endl;
            // print first 10 becp values
            std::cout << "[BECP] pw-port ik=" << ik << " becp[0..9]=";
            for(int i=0;i<10 && i<nbands*psi_p->get_npol()*nkb; i++) std::cout << " (" << becp[i].real() << "," << becp[i].imag() << ")";
            std::cout << std::endl;
            int begin_ih = 0;
            for(int iat = 0; iat < cell.nat; iat++)
            {
                const int it = cell.iat2it[iat];
                const int nh = onsite_p->get_nh(iat);
                const int target_l = this->orbital_corr[it];
                if(target_l == -1)
                {
                    begin_ih += nh;
                    continue;
                }
                // m = l^2, l^2+1, ..., (l+1)^2-1
                const int m_begin = target_l * target_l;
                const int tlp1 = 2 * target_l + 1;
                const int tlp1_2 = tlp1 * tlp1;
                if(PARAM.inp.nspin == 4)
                {
                for(int ib = 0;ib<nbands;ib++)
                {
                    const double weight = wg_in(ik, ib);
                    int ind_m1m2 = 0;
                    for(int m1 = 0; m1 < tlp1; m1++)
                    {
                        const int index_m1 = ib*2*nkb + begin_ih + m_begin + m1;
                        for(int m2 = 0; m2 < tlp1; m2++)
                        {
                            const int index_m2 = ib*2*nkb + begin_ih + m_begin + m2;
                            std::complex<double> occ[4];
                            occ[0] = weight * conj(becp[index_m1]) * becp[index_m2];
                            occ[1] = weight * conj(becp[index_m1]) * becp[index_m2 + nkb];
                            occ[2] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2];
                            occ[3] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2 + nkb];
                            this->locale[iat][target_l][0][0].c[ind_m1m2] += (occ[0] + occ[3]).real();
                            this->locale[iat][target_l][0][0].c[ind_m1m2 + tlp1_2] += (occ[1] + occ[2]).real();
                            this->locale[iat][target_l][0][0].c[ind_m1m2 + 2 * tlp1_2] += (occ[1] - occ[2]).imag();
                            this->locale[iat][target_l][0][0].c[ind_m1m2 + 3 * tlp1_2] += (occ[0] - occ[3]).real();
                            ind_m1m2++;
                        }
                    }
                }// ib
                }
                else
                {
                for(int ib = 0;ib<nbands;ib++)
                {
                    // DIAGNOSTIC: print wg for first band
                    if(iter <= 2 && ib == 0 && ik == 0)
                    {
                        std::cout << "[WG-PW] iter=" << iter << " ik=" << ik << " wg[0..7]=";
                        for(int b=0;b<8;b++) std::cout << " " << wg_in(ik, b);
                        std::cout << std::endl;
                    }
                    const double weight = wg_in(ik, ib);
                    int ind_m1m2 = 0;
                    for(int m1 = 0; m1 < tlp1; m1++)
                    {
                        const int index_m1 = ib*nkb + begin_ih + m_begin + m1;
                        for(int m2 = 0; m2 < tlp1; m2++)
                        {
                            const int index_m2 = ib*nkb + begin_ih + m_begin + m2;
                            this->locale[iat][target_l][0][is].c[ind_m1m2] += weight * (conj(becp[index_m1]) * becp[index_m2]).real();
                            ind_m1m2++;
                        }
                    }
                }// ib
                }
                begin_ih += nh;
            }// iat
            // DIAG: locale snapshot per ik
            if(iter <= 3 && (ik == 0 || ik == 1)) {
                std::cout << "[DFTU-IK] pw-port iter=" << iter << " ik=" << ik << " after locale acc:" << std::endl;
                for(int iat2=0; iat2<cell.nat; iat2++){
                    const int it2 = cell.iat2it[iat2];
                    const int tl2 = this->orbital_corr[it2];
                    if(tl2 == -1) continue;
                    const int sz2 = (2*tl2+1)*(2*tl2+1);
                    std::cout << "[DFTU-IK]   locale[iat=" << iat2 << "][0](spin" << is << ")=";
                    for(int ii=0;ii<sz2;ii++) std::cout << (ii>0?",":"") << this->locale[iat2][tl2][0][is].c[ii];
                    std::cout << std::endl;
                }
            }
        }// ik
    }
#if defined(__CUDA) || defined(__ROCM)
    else
    {
        auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_GPU>::get_instance();
        const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi_p = (const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*)psi_in;
        // loop over k-points to calculate Mi of \sum_{k,i,l,m}<Psi_{k,i}|alpha_{l,m}><alpha_{l,m}|Psi_{k,i}>
        const int nbands = psi_p->get_nbands();
        for(int ik = 0; ik < psi_p->get_nk(); ik++)
        {
            int is = 0;
            if(PARAM.inp.nspin == 2 && ik >= psi_p->get_nk()/2)
            {
                is = 1;
            }
            psi_p->fix_k(ik);
            onsite_p->tabulate_atomic(ik);

            // DIAGNOSTIC: print psi pointer and becp info per ik
            std::cout << "[BECP] pw-port ik=" << ik << " psi_ptr=" << psi_p->get_pointer() << " nbands=" << nbands << " npol=" << psi_p->get_npol() << std::endl;

            onsite_p->overlap_proj_psi(nbands*psi_p->get_npol(), psi_p->get_pointer());
            const std::complex<double>* becp = onsite_p->get_h_becp();
            // becp(nbands*npol , nkb)
            // mag = wg * \sum_{nh}becp * becp
            int nkb = onsite_p->get_size_becp() / nbands / psi_p->get_npol();
            std::cout << "[BECP] pw-port ik=" << ik << " becp_ptr=" << becp << " nkb=" << nkb << " size=" << nbands*psi_p->get_npol()*nkb << std::endl;
            // print first 10 becp values
            std::cout << "[BECP] pw-port ik=" << ik << " becp[0..9]=";
            for(int i=0;i<10 && i<nbands*psi_p->get_npol()*nkb; i++) std::cout << " (" << becp[i].real() << "," << becp[i].imag() << ")";
            std::cout << std::endl;
            int begin_ih = 0;
            for(int iat = 0; iat < cell.nat; iat++)
            {
                const int it = cell.iat2it[iat];
                const int nh = onsite_p->get_nh(iat);
                const int target_l = this->orbital_corr[it];
                if(target_l == -1)
                {
                    begin_ih += nh;
                    continue;
                }
                // m = l^2, l^2+1, ..., (l+1)^2-1
                const int m_begin = target_l * target_l;
                const int tlp1 = 2 * target_l + 1;
                const int tlp1_2 = tlp1 * tlp1;
                if(PARAM.inp.nspin == 4)
                {
                    for(int ib = 0;ib<nbands;ib++)
                    {
                        const double weight = wg_in(ik, ib);
                        int ind_m1m2 = 0;
                        for(int m1 = 0; m1 < tlp1; m1++)
                        {
                            const int index_m1 = ib*2*nkb + begin_ih + m_begin + m1;
                            for(int m2 = 0; m2 < tlp1; m2++)
                            {
                                const int index_m2 = ib*2*nkb + begin_ih + m_begin + m2;
                                std::complex<double> occ[4];
                                occ[0] = weight * conj(becp[index_m1]) * becp[index_m2];
                                occ[1] = weight * conj(becp[index_m1]) * becp[index_m2 + nkb];
                                occ[2] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2];
                                occ[3] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2 + nkb];
                                this->locale[iat][target_l][0][0].c[ind_m1m2] += (occ[0] + occ[3]).real();
                                this->locale[iat][target_l][0][0].c[ind_m1m2 + tlp1_2] += (occ[1] + occ[2]).real();
                                this->locale[iat][target_l][0][0].c[ind_m1m2 + 2 * tlp1_2] += (occ[1] - occ[2]).imag();
                                this->locale[iat][target_l][0][0].c[ind_m1m2 + 3 * tlp1_2] += (occ[0] - occ[3]).real();
                                ind_m1m2++;
                            }
                        }
                    }// ib
                }
                else
                {
                    for(int ib = 0;ib<nbands;ib++)
                    {
                        const double weight = wg_in(ik, ib);
                        int ind_m1m2 = 0;
                        for(int m1 = 0; m1 < tlp1; m1++)
                        {
                            const int index_m1 = ib*nkb + begin_ih + m_begin + m1;
                            for(int m2 = 0; m2 < tlp1; m2++)
                            {
                                const int index_m2 = ib*nkb + begin_ih + m_begin + m2;
                                this->locale[iat][target_l][0][is].c[ind_m1m2] += weight * (conj(becp[index_m1]) * becp[index_m2]).real();
                                ind_m1m2++;
                            }
                        }
                    }// ib
                }
                begin_ih += nh;
            }// iat
        }// ik
    }
#endif

    // reduce locale from all k-pools
    for(int iat = 0; iat < cell.nat; iat++)
    {
        const int it = cell.iat2it[iat];
        const int target_l = this->orbital_corr[it];
        if(target_l == -1)
        {
            continue;
        }
        const int fold = PARAM.inp.nspin == 4 ? 4 : 1;
        const int size = (2 * target_l + 1) * (2 * target_l + 1);

        Parallel_Reduce::reduce_double_allpool(PARAM.inp.kpar,
            PARAM.globalv.nproc_in_pool,
            this->locale[iat][target_l][0][0].c,
            size * fold);

        // save to uom_array
        if(this->uom_array.size() != 0)
        {
            for(int mm = 0; mm < size * fold; mm++)
                this->uom_array[eff_pot_pw_index[iat] + mm] = this->locale[iat][target_l][0][0].c[mm];
        }

        if(PARAM.inp.nspin == 2)
        {
            Parallel_Reduce::reduce_double_allpool(PARAM.inp.kpar,
                PARAM.globalv.nproc_in_pool,
                this->locale[iat][target_l][0][1].c, size);
            if(this->uom_array.size() != 0)
            {
                for(int mm = 0; mm < size; mm++)
                    this->uom_array[this->uom_array.size()/2 + eff_pot_pw_index[iat] + mm] = this->locale[iat][target_l][0][1].c[mm];
            }
        }
    }
    } // end if(initialed_locale == false)
    else
    {
        for(int iat = 0; iat < cell.nat; iat++)
        {
            const int it = cell.iat2it[iat];
            const int target_l = this->orbital_corr[it];
            if(target_l == -1)
            {
                continue;
            }
            const int fold = PARAM.inp.nspin == 4 ? 4 : 1;
            const int size = (2 * target_l + 1) * (2 * target_l + 1);
            if(this->uom_array.size() != 0)
            {
                for(int mm = 0; mm < size * fold; mm++)
                {
                    this->uom_array[eff_pot_pw_index[iat] + mm] = this->locale[iat][target_l][0][0].c[mm];
                    if(PARAM.inp.nspin == 2)
                        this->uom_array[this->uom_array.size()/2 + eff_pot_pw_index[iat] + mm] = this->locale[iat][target_l][0][1].c[mm];
                }
            }
        }
    }

    if(mixing_dftu && p_chgmix != nullptr)
    {
        p_chgmix->mix_uom(this->uom_array, this->uom_save);
        this->set_locale(cell);
    }

    Plus_U::energy_u = 0.0;

    // DIAGNOSTIC [N5]: state before VU calculation
#if 0  // disabled: locale[0][1] access crashes for nspin=4
    if(iter <= 2)
    std::cout << "[DIAG-PW] cal_occ_pw iter=" << iter << " before VU calculation:" << std::endl;
    if(iter <= 2)
    std::cout << "[DIAG-PW]   uom_array.size=" << this->uom_array.size()
              << " eff_pot_pw.size=" << this->eff_pot_pw.size() << std::endl;
    if(iter <= 2 && this->uom_array.size() > 0) {
        std::cout << "[DIAG-PW]   uom_array[0..29]=";
        for(int i=0;i<30 && i<(int)this->uom_array.size();i++) std::cout << " " << this->uom_array[i];
        std::cout << std::endl;
    }
    if(iter <= 2)
    for(int iat = 0; iat < cell.nat; iat++) {
        const int it2 = cell.iat2it[iat];
        const int tl = this->orbital_corr[it2];
        if(tl == -1) continue;
        const int sz = (2*tl+1)*(2*tl+1);
        std::cout << "[DIAG-PW]   locale[iat=" << iat << "][0][0](";
        for(int i=0;i<sz;i++) std::cout << (i>0?",":"") << this->locale[iat][tl][0][0].c[i];
        std::cout << ") locale[0][1](";
        for(int i=0;i<sz;i++) std::cout << (i>0?",":"") << this->locale[iat][tl][0][1].c[i];
        std::cout << ")" << std::endl;
    }
#endif

    // calculate effective potential and energy
    for(int iat = 0; iat < cell.nat; iat++)
    {
        const int it = cell.iat2it[iat];
        const int target_l = this->orbital_corr[it];
        if(target_l == -1)
        {
            continue;
        }
        const int size = (2 * target_l + 1) * (2 * target_l + 1);
        //update effective potential
        const double u_value = this->U[it];
        std::complex<double>* vu_iat = &(this->eff_pot_pw[this->eff_pot_pw_index[iat]]);
        const int m_size = 2 * target_l + 1;

        double weight_eu = 1;
        switch(PARAM.inp.nspin)
        {
            case 1: weight_eu = 1.0; break;
            case 2: weight_eu = 0.5; break;
            case 4: weight_eu = 0.25; break;
            default: break;
        }
        const double diag_coeff = PARAM.inp.nspin == 4 ? 1.0 : 0.5;

        for (int m1 = 0; m1 < m_size; m1++)
        {
            for (int m2 = 0; m2 < m_size; m2++)
            {
                vu_iat[m1 * m_size + m2] = u_value *
                  (diag_coeff * (m1 == m2) - this->locale[iat][target_l][0][0].c[m2 * m_size + m1]);
                Plus_U::energy_u += u_value * weight_eu * this->locale[iat][target_l][0][0].c[m2 * m_size + m1]
                         * this->locale[iat][target_l][0][0].c[m1 * m_size + m2];
            }
        }
        if(PARAM.inp.nspin == 2)
        {
            std::complex<double>* vu_iat1 = &(this->eff_pot_pw[this->eff_pot_pw.size()/2 + this->eff_pot_pw_index[iat]]);
            for (int m1 = 0; m1 < m_size; m1++)
            {
                for (int m2 = 0; m2 < m_size; m2++)
                {
                    vu_iat1[m1 * m_size + m2] = u_value *
                      (diag_coeff * (m1 == m2) - this->locale[iat][target_l][0][1].c[m2 * m_size + m1]);
                    Plus_U::energy_u += u_value * weight_eu * this->locale[iat][target_l][0][1].c[m2 * m_size + m1]
                             * this->locale[iat][target_l][0][1].c[m1 * m_size + m2];
                }
            }
        }
        if(PARAM.inp.nspin == 4)
        {
        for (int is = 1; is < 4; ++is)
        {
            int start = is * m_size * m_size;
            for (int m1 = 0; m1 < m_size; m1++)
            {
                for (int m2 = 0; m2 < m_size; m2++)
                {
                    vu_iat[start + m1 * m_size + m2] = u_value *
                      (0 - this->locale[iat][target_l][0][0].c[start + m2 * m_size + m1]);
                    Plus_U::energy_u += u_value * weight_eu
                             * this->locale[iat][target_l][0][0].c[start + m2 * m_size + m1]
                             * this->locale[iat][target_l][0][0].c[start + m1 * m_size + m2];
                }
            }
        }
        // transfer from Pauli matrix representation to spin representation
        for (int m1 = 0; m1 < m_size; m1++)
        {
            for (int m2 = 0; m2 < m_size; m2++)
            {
                int index[4];
                index[0] = m1 * m_size + m2;
                index[1] = m1 * m_size + m2 + size;
                index[2] = m1 * m_size + m2 + size * 2;
                index[3] = m1 * m_size + m2 + size * 3;
                std::complex<double> vu_tmp[4];
                for (int i = 0; i < 4; i++)
                {
                    vu_tmp[i] = vu_iat[index[i]];
                }
                vu_iat[index[0]] = 0.5 * (vu_tmp[0] + vu_tmp[3]);
                vu_iat[index[3]] = 0.5 * (vu_tmp[0] - vu_tmp[3]);
                vu_iat[index[1]] = 0.5 * (vu_tmp[1] + std::complex<double>(0.0, 1.0) * vu_tmp[2]);
                vu_iat[index[2]] = 0.5 * (vu_tmp[1] - std::complex<double>(0.0, 1.0) * vu_tmp[2]);
            }
        }
        }
    }

    // DIAGNOSTIC [N6]: state after VU calculation
    if(iter <= 2)
    std::cout << "[DIAG-PW] cal_occ_pw iter=" << iter << " after VU calculation:" << std::endl;
    if(iter <= 2)
    std::cout << "[DIAG-PW]   eff_pot_pw.size=" << this->eff_pot_pw.size() << std::endl;
    if(iter <= 2 && this->eff_pot_pw.size() > 0) {
        std::cout << "[DIAG-PW]   eff_pot_pw[0..29]=";
        for(int i=0;i<30 && i<(int)this->eff_pot_pw.size();i++)
            std::cout << " (" << this->eff_pot_pw[i].real() << "," << this->eff_pot_pw[i].imag() << ")";
        std::cout << std::endl;
    }
    if(iter <= 2)
    std::cout << "[DIAG-PW]   energy_u=" << Plus_U::energy_u << std::endl;
    if(iter <= 2)
    std::cout << "[DIAG-PW]   initialed_locale=false" << std::endl;

    initialed_locale = false;

    // update effective potential
    ModuleBase::timer::tick("Plus_U", "cal_occ_pw");
}
/// calculate the local DFT+U effective potential matrix for PW base.
void Plus_U::cal_VU_pot_pw(const int spin)
{

}
