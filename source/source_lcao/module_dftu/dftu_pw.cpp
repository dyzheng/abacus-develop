     1|#include "dftu.h"
     2|#include "source_estate/module_charge/charge_mixing.h"
     3|#include "source_pw/module_pwdft/onsite_projector.h"
     4|#include "source_base/parallel_reduce.h"
     5|#include "source_io/module_parameter/parameter.h"
     6|#include "source_base/timer.h"
     7|
     8|
     9|/// calculate occupation matrix for DFT+U
    10|void Plus_U::cal_occ_pw(const int iter,
    11|		const void* psi_in,
    12|		const ModuleBase::matrix& wg_in,
    13|		const UnitCell& cell,
    14|		Charge_Mixing* p_chgmix)
    15|{
    16|    ModuleBase::timer::tick("Plus_U", "cal_occ_pw");
    17|    this->copy_locale(cell);
    18|
    19|    if(this->initialed_locale == false)
    20|    {
    21|    this->zero_locale(cell);
    22|
    23|    // DIAGNOSTIC [N4]: state right after zero_locale
    24|    // std::cout << "[DIAG-PW] cal_occ_pw iter=" << iter << " right after zero_locale:" << std::endl;
    25|    // std::cout << "[DIAG-PW]   uom_array.size=" << this->uom_array.size()
    26|              << " eff_pot_pw.size=" << this->eff_pot_pw.size() << std::endl;
    27|    if (this->uom_array.size() > 0) {
    28|        // std::cout << "[DIAG-PW]   uom_array[0..9]=";
    29|        // for(int i=0;i<10 && i<(int)this->uom_array.size();i++) std::cout << " " << this->uom_array[i];
    30|        std::cout << std::endl;
    31|    }
    32|
    33|    if(PARAM.inp.device == "cpu")
    34|    {
    35|        auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_CPU>::get_instance();
    36|        const psi::Psi<std::complex<double>>* psi_p = (const psi::Psi<std::complex<double>>*)psi_in;
    37|        // loop over k-points to calculate Mi of \sum_{k,i,l,m}<Psi_{k,i}|alpha_{l,m}><alpha_{l,m}|Psi_{k,i}>
    38|        const int nbands = psi_p->get_nbands();
    39|        for(int ik = 0; ik < psi_p->get_nk(); ik++)
    40|        {
    41|            int is = 0;
    42|            if(PARAM.inp.nspin == 2 && ik >= psi_p->get_nk()/2)
    43|            {
    44|                is = 1;
    45|            }
    46|            psi_p->fix_k(ik);
    47|            onsite_p->tabulate_atomic(ik);
    48|
    49|            // DIAGNOSTIC: print psi pointer and psi values BEFORE overlap_proj_psi
    50|            const std::complex<double>* psi_ptr_before = psi_p->get_pointer();
    51|            // std::cout << "[PSI-BEFORE] pw-port ik=" << ik << " psi_ptr=" << psi_ptr_before << " nbands=" << nbands << " npol=" << psi_p->get_npol() << std::endl;
    52|            // std::cout << "[PSI-BEFORE] pw-port ik=" << ik << " psi[0..9]=";
    53|        // for(int i=0;i<10;i++) std::cout << " (" << psi_ptr_before[i].real() << "," << psi_ptr_before[i].imag() << ")";
    54|            std::cout << std::endl;
    55|
    56|            onsite_p->overlap_proj_psi(nbands*psi_p->get_npol(), psi_p->get_pointer());
    57|            const std::complex<double>* becp = onsite_p->get_h_becp();
    58|            // becp(nbands*npol , nkb)
    59|            // mag = wg * \sum_{nh}becp * becp
    60|            int nkb = onsite_p->get_size_becp() / nbands / psi_p->get_npol();
    61|            // std::cout << "[BECP] pw-port ik=" << ik << " becp_ptr=" << becp << " nkb=" << nkb << " size=" << nbands*psi_p->get_npol()*nkb << std::endl;
    62|            // print first 10 becp values
    63|            // std::cout << "[BECP] pw-port ik=" << ik << " becp[0..9]=";
    64|        // for(int i=0;i<10 && i<nbands*psi_p->get_npol()*nkb; i++) std::cout << " (" << becp[i].real() << "," << becp[i].imag() << ")";
    65|            std::cout << std::endl;
    66|            int begin_ih = 0;
    67|            for(int iat = 0; iat < cell.nat; iat++)
    68|            {
    69|                const int it = cell.iat2it[iat];
    70|                const int nh = onsite_p->get_nh(iat);
    71|                const int target_l = this->orbital_corr[it];
    72|                if(target_l == -1)
    73|                {
    74|                    begin_ih += nh;
    75|                    continue;
    76|                }
    77|                // m = l^2, l^2+1, ..., (l+1)^2-1
    78|                const int m_begin = target_l * target_l;
    79|                const int tlp1 = 2 * target_l + 1;
    80|                const int tlp1_2 = tlp1 * tlp1;
    81|                if(PARAM.inp.nspin == 4)
    82|                {
    83|                for(int ib = 0;ib<nbands;ib++)
    84|                {
    85|                    const double weight = wg_in(ik, ib);
    86|                    int ind_m1m2 = 0;
    87|                    for(int m1 = 0; m1 < tlp1; m1++)
    88|                    {
    89|                        const int index_m1 = ib*2*nkb + begin_ih + m_begin + m1;
    90|                        for(int m2 = 0; m2 < tlp1; m2++)
    91|                        {
    92|                            const int index_m2 = ib*2*nkb + begin_ih + m_begin + m2;
    93|                            std::complex<double> occ[4];
    94|                            occ[0] = weight * conj(becp[index_m1]) * becp[index_m2];
    95|                            occ[1] = weight * conj(becp[index_m1]) * becp[index_m2 + nkb];
    96|                            occ[2] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2];
    97|                            occ[3] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2 + nkb];
    98|                            this->locale[iat][target_l][0][0].c[ind_m1m2] += (occ[0] + occ[3]).real();
    99|                            this->locale[iat][target_l][0][0].c[ind_m1m2 + tlp1_2] += (occ[1] + occ[2]).real();
   100|                            this->locale[iat][target_l][0][0].c[ind_m1m2 + 2 * tlp1_2] += (occ[1] - occ[2]).imag();
   101|                            this->locale[iat][target_l][0][0].c[ind_m1m2 + 3 * tlp1_2] += (occ[0] - occ[3]).real();
   102|                            ind_m1m2++;
   103|                        }
   104|                    }
   105|                }// ib
   106|                }
   107|                else
   108|                {
   109|                for(int ib = 0;ib<nbands;ib++)
   110|                {
   111|                    // DIAGNOSTIC: print wg for first band
   112|                    if(iter <= 2 && ib == 0 && ik == 0)
   113|                    {
   114|                        // std::cout << "[WG-PW] iter=" << iter << " ik=" << ik << " wg[0..7]=";
   115|        // for(int b=0;b<8;b++) std::cout << " " << wg_in(ik, b);
   116|                        std::cout << std::endl;
   117|                    }
   118|                    const double weight = wg_in(ik, ib);
   119|                    int ind_m1m2 = 0;
   120|                    for(int m1 = 0; m1 < tlp1; m1++)
   121|                    {
   122|                        const int index_m1 = ib*nkb + begin_ih + m_begin + m1;
   123|                        for(int m2 = 0; m2 < tlp1; m2++)
   124|                        {
   125|                            const int index_m2 = ib*nkb + begin_ih + m_begin + m2;
   126|                            this->locale[iat][target_l][0][is].c[ind_m1m2] += weight * (conj(becp[index_m1]) * becp[index_m2]).real();
   127|                            ind_m1m2++;
   128|                        }
   129|                    }
   130|                }// ib
   131|                }
   132|                begin_ih += nh;
   133|            }// iat
   134|            // DIAG: locale snapshot per ik
   135|            if(iter <= 3 && (ik == 0 || ik == 1)) {
   136|                // std::cout << "[DFTU-IK] pw-port iter=" << iter << " ik=" << ik << " after locale acc:" << std::endl;
   137|                for(int iat2=0; iat2<cell.nat; iat2++){
   138|                    const int it2 = cell.iat2it[iat2];
   139|                    const int tl2 = this->orbital_corr[it2];
   140|                    if(tl2 == -1) continue;
   141|                    const int sz2 = (2*tl2+1)*(2*tl2+1);
   142|                    // std::cout << "[DFTU-IK]   locale[iat=" << iat2 << "][0](spin" << is << ")=";
   143|        // for(int ii=0;ii<sz2;ii++) std::cout << (ii>0?",":"") << this->locale[iat2][tl2][0][is].c[ii];
   144|                    std::cout << std::endl;
   145|                }
   146|            }
   147|        }// ik
   148|    }
   149|#if defined(__CUDA) || defined(__ROCM)
   150|    else
   151|    {
   152|        auto* onsite_p = projectors::OnsiteProjector<double, base_device::DEVICE_GPU>::get_instance();
   153|        const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi_p = (const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*)psi_in;
   154|        // loop over k-points to calculate Mi of \sum_{k,i,l,m}<Psi_{k,i}|alpha_{l,m}><alpha_{l,m}|Psi_{k,i}>
   155|        const int nbands = psi_p->get_nbands();
   156|        for(int ik = 0; ik < psi_p->get_nk(); ik++)
   157|        {
   158|            int is = 0;
   159|            if(PARAM.inp.nspin == 2 && ik >= psi_p->get_nk()/2)
   160|            {
   161|                is = 1;
   162|            }
   163|            psi_p->fix_k(ik);
   164|            onsite_p->tabulate_atomic(ik);
   165|
   166|            // DIAGNOSTIC: print psi pointer and becp info per ik
   167|            // std::cout << "[BECP] pw-port ik=" << ik << " psi_ptr=" << psi_p->get_pointer() << " nbands=" << nbands << " npol=" << psi_p->get_npol() << std::endl;
   168|
   169|            onsite_p->overlap_proj_psi(nbands*psi_p->get_npol(), psi_p->get_pointer());
   170|            const std::complex<double>* becp = onsite_p->get_h_becp();
   171|            // becp(nbands*npol , nkb)
   172|            // mag = wg * \sum_{nh}becp * becp
   173|            int nkb = onsite_p->get_size_becp() / nbands / psi_p->get_npol();
   174|            // std::cout << "[BECP] pw-port ik=" << ik << " becp_ptr=" << becp << " nkb=" << nkb << " size=" << nbands*psi_p->get_npol()*nkb << std::endl;
   175|            // print first 10 becp values
   176|            // std::cout << "[BECP] pw-port ik=" << ik << " becp[0..9]=";
   177|        // for(int i=0;i<10 && i<nbands*psi_p->get_npol()*nkb; i++) std::cout << " (" << becp[i].real() << "," << becp[i].imag() << ")";
   178|            std::cout << std::endl;
   179|            int begin_ih = 0;
   180|            for(int iat = 0; iat < cell.nat; iat++)
   181|            {
   182|                const int it = cell.iat2it[iat];
   183|                const int nh = onsite_p->get_nh(iat);
   184|                const int target_l = this->orbital_corr[it];
   185|                if(target_l == -1)
   186|                {
   187|                    begin_ih += nh;
   188|                    continue;
   189|                }
   190|                // m = l^2, l^2+1, ..., (l+1)^2-1
   191|                const int m_begin = target_l * target_l;
   192|                const int tlp1 = 2 * target_l + 1;
   193|                const int tlp1_2 = tlp1 * tlp1;
   194|                if(PARAM.inp.nspin == 4)
   195|                {
   196|                    for(int ib = 0;ib<nbands;ib++)
   197|                    {
   198|                        const double weight = wg_in(ik, ib);
   199|                        int ind_m1m2 = 0;
   200|                        for(int m1 = 0; m1 < tlp1; m1++)
   201|                        {
   202|                            const int index_m1 = ib*2*nkb + begin_ih + m_begin + m1;
   203|                            for(int m2 = 0; m2 < tlp1; m2++)
   204|                            {
   205|                                const int index_m2 = ib*2*nkb + begin_ih + m_begin + m2;
   206|                                std::complex<double> occ[4];
   207|                                occ[0] = weight * conj(becp[index_m1]) * becp[index_m2];
   208|                                occ[1] = weight * conj(becp[index_m1]) * becp[index_m2 + nkb];
   209|                                occ[2] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2];
   210|                                occ[3] = weight * conj(becp[index_m1 + nkb]) * becp[index_m2 + nkb];
   211|                                this->locale[iat][target_l][0][0].c[ind_m1m2] += (occ[0] + occ[3]).real();
   212|                                this->locale[iat][target_l][0][0].c[ind_m1m2 + tlp1_2] += (occ[1] + occ[2]).real();
   213|                                this->locale[iat][target_l][0][0].c[ind_m1m2 + 2 * tlp1_2] += (occ[1] - occ[2]).imag();
   214|                                this->locale[iat][target_l][0][0].c[ind_m1m2 + 3 * tlp1_2] += (occ[0] - occ[3]).real();
   215|                                ind_m1m2++;
   216|                            }
   217|                        }
   218|                    }// ib
   219|                }
   220|                else
   221|                {
   222|                    for(int ib = 0;ib<nbands;ib++)
   223|                    {
   224|                        const double weight = wg_in(ik, ib);
   225|                        int ind_m1m2 = 0;
   226|                        for(int m1 = 0; m1 < tlp1; m1++)
   227|                        {
   228|                            const int index_m1 = ib*nkb + begin_ih + m_begin + m1;
   229|                            for(int m2 = 0; m2 < tlp1; m2++)
   230|                            {
   231|                                const int index_m2 = ib*nkb + begin_ih + m_begin + m2;
   232|                                this->locale[iat][target_l][0][is].c[ind_m1m2] += weight * (conj(becp[index_m1]) * becp[index_m2]).real();
   233|                                ind_m1m2++;
   234|                            }
   235|                        }
   236|                    }// ib
   237|                }
   238|                begin_ih += nh;
   239|            }// iat
   240|        }// ik
   241|    }
   242|#endif
   243|
   244|    // reduce locale from all k-pools
   245|    for(int iat = 0; iat < cell.nat; iat++)
   246|    {
   247|        const int it = cell.iat2it[iat];
   248|        const int target_l = this->orbital_corr[it];
   249|        if(target_l == -1)
   250|        {
   251|            continue;
   252|        }
   253|        const int fold = PARAM.inp.nspin == 4 ? 4 : 1;
   254|        const int size = (2 * target_l + 1) * (2 * target_l + 1);
   255|
   256|        Parallel_Reduce::reduce_double_allpool(PARAM.inp.kpar,
   257|            PARAM.globalv.nproc_in_pool,
   258|            this->locale[iat][target_l][0][0].c,
   259|            size * fold);
   260|
   261|        // save to uom_array
   262|        if(this->uom_array.size() != 0)
   263|        {
   264|            for(int mm = 0; mm < size * fold; mm++)
   265|                this->uom_array[eff_pot_pw_index[iat] + mm] = this->locale[iat][target_l][0][0].c[mm];
   266|        }
   267|
   268|        if(PARAM.inp.nspin == 2)
   269|        {
   270|            Parallel_Reduce::reduce_double_allpool(PARAM.inp.kpar,
   271|                PARAM.globalv.nproc_in_pool,
   272|                this->locale[iat][target_l][0][1].c, size);
   273|            if(this->uom_array.size() != 0)
   274|            {
   275|                for(int mm = 0; mm < size; mm++)
   276|                    this->uom_array[eff_pot_pw_index[iat] + mm + size] = this->locale[iat][target_l][0][1].c[mm];
   277|            }
   278|        }
   279|    }
   280|    } // end if(initialed_locale == false)
   281|    else
   282|    {
   283|        for(int iat = 0; iat < cell.nat; iat++)
   284|        {
   285|            const int it = cell.iat2it[iat];
   286|            const int target_l = this->orbital_corr[it];
   287|            if(target_l == -1)
   288|            {
   289|                continue;
   290|            }
   291|            const int fold = PARAM.inp.nspin == 4 ? 4 : 1;
   292|            const int size = (2 * target_l + 1) * (2 * target_l + 1);
   293|            if(this->uom_array.size() != 0)
   294|            {
   295|                for(int mm = 0; mm < size * fold; mm++)
   296|                {
   297|                    this->uom_array[eff_pot_pw_index[iat] + mm] = this->locale[iat][target_l][0][0].c[mm];
   298|                    if(PARAM.inp.nspin == 2)
   299|                        this->uom_array[eff_pot_pw_index[iat] + mm + locale[iat][target_l][0][0].nr * locale[iat][target_l][0][0].nc] = this->locale[iat][target_l][0][1].c[mm];
   300|                }
   301|            }
   302|        }
   303|    }
   304|
   305|    if(mixing_dftu && p_chgmix != nullptr)
   306|    {
   307|        p_chgmix->mix_uom(this->uom_array, this->uom_save);
   308|        this->set_locale(cell);
   309|    }
   310|
   311|    Plus_U::energy_u = 0.0;
   312|
   313|    // DIAGNOSTIC [N5]: state before VU calculation
   314|    if(iter <= 2)
   315|    // std::cout << "[DIAG-PW] cal_occ_pw iter=" << iter << " before VU calculation:" << std::endl;
   316|    if(iter <= 2)
   317|    // std::cout << "[DIAG-PW]   uom_array.size=" << this->uom_array.size()
   318|              << " eff_pot_pw.size=" << this->eff_pot_pw.size() << std::endl;
   319|    if(iter <= 2 && this->uom_array.size() > 0) {
   320|        // std::cout << "[DIAG-PW]   uom_array[0..29]=";
   321|        // for(int i=0;i<30 && i<(int)this->uom_array.size();i++) std::cout << " " << this->uom_array[i];
   322|        std::cout << std::endl;
   323|    }
   324|    if(iter <= 2)
   325|    for(int iat = 0; iat < cell.nat; iat++) {
   326|        const int it2 = cell.iat2it[iat];
   327|        const int tl = this->orbital_corr[it2];
   328|        if(tl == -1) continue;
   329|        const int sz = (2*tl+1)*(2*tl+1);
   330|        // std::cout << "[DIAG-PW]   locale[iat=" << iat << "][0][0](";
   331|        // for(int i=0;i<sz;i++) std::cout << (i>0?",":"") << this->locale[iat][tl][0][0].c[i];
   332|        std::cout << ") locale[0][1](";
   333|        // for(int i=0;i<sz;i++) std::cout << (i>0?",":"") << this->locale[iat][tl][0][1].c[i];
   334|        std::cout << ")" << std::endl;
   335|    }
   336|
   337|    // calculate effective potential and energy
   338|    for(int iat = 0; iat < cell.nat; iat++)
   339|    {
   340|        const int it = cell.iat2it[iat];
   341|        const int target_l = this->orbital_corr[it];
   342|        if(target_l == -1)
   343|        {
   344|            continue;
   345|        }
   346|        const int size = (2 * target_l + 1) * (2 * target_l + 1);
   347|        //update effective potential
   348|        const double u_value = this->U[it];
   349|        std::complex<double>* vu_iat = &(this->eff_pot_pw[this->eff_pot_pw_index[iat]]);
   350|        const int m_size = 2 * target_l + 1;
   351|
   352|        double weight_eu = 1;
   353|        switch(PARAM.inp.nspin)
   354|        {
   355|            case 1: weight_eu = 1.0; break;
   356|            case 2: weight_eu = 0.5; break;
   357|            case 4: weight_eu = 0.25; break;
   358|            default: break;
   359|        }
   360|        const double diag_coeff = PARAM.inp.nspin == 4 ? 1.0 : 0.5;
   361|
   362|        for (int m1 = 0; m1 < m_size; m1++)
   363|        {
   364|            for (int m2 = 0; m2 < m_size; m2++)
   365|            {
   366|                vu_iat[m1 * m_size + m2] = u_value *
   367|                  (diag_coeff * (m1 == m2) - this->locale[iat][target_l][0][0].c[m2 * m_size + m1]);
   368|                Plus_U::energy_u += u_value * weight_eu * this->locale[iat][target_l][0][0].c[m2 * m_size + m1]
   369|                         * this->locale[iat][target_l][0][0].c[m1 * m_size + m2];
   370|            }
   371|        }
   372|        if(PARAM.inp.nspin == 2)
   373|        {
   374|            std::complex<double>* vu_iat1 = &(this->eff_pot_pw[this->eff_pot_pw.size()/2 + this->eff_pot_pw_index[iat]]);
   375|            for (int m1 = 0; m1 < m_size; m1++)
   376|            {
   377|                for (int m2 = 0; m2 < m_size; m2++)
   378|                {
   379|                    vu_iat1[m1 * m_size + m2] = u_value *
   380|                      (diag_coeff * (m1 == m2) - this->locale[iat][target_l][0][1].c[m2 * m_size + m1]);
   381|                    Plus_U::energy_u += u_value * weight_eu * this->locale[iat][target_l][0][1].c[m2 * m_size + m1]
   382|                             * this->locale[iat][target_l][0][1].c[m1 * m_size + m2];
   383|                }
   384|            }
   385|        }
   386|        if(PARAM.inp.nspin == 4)
   387|        {
   388|        for (int is = 1; is < 4; ++is)
   389|        {
   390|            int start = is * m_size * m_size;
   391|            for (int m1 = 0; m1 < m_size; m1++)
   392|            {
   393|                for (int m2 = 0; m2 < m_size; m2++)
   394|                {
   395|                    vu_iat[start + m1 * m_size + m2] = u_value *
   396|                      (0 - this->locale[iat][target_l][0][0].c[start + m2 * m_size + m1]);
   397|                    Plus_U::energy_u += u_value * weight_eu
   398|                             * this->locale[iat][target_l][0][0].c[start + m2 * m_size + m1]
   399|                             * this->locale[iat][target_l][0][0].c[start + m1 * m_size + m2];
   400|                }
   401|            }
   402|        }
   403|        // transfer from Pauli matrix representation to spin representation
   404|        for (int m1 = 0; m1 < m_size; m1++)
   405|        {
   406|            for (int m2 = 0; m2 < m_size; m2++)
   407|            {
   408|                int index[4];
   409|                index[0] = m1 * m_size + m2;
   410|                index[1] = m1 * m_size + m2 + size;
   411|                index[2] = m1 * m_size + m2 + size * 2;
   412|                index[3] = m1 * m_size + m2 + size * 3;
   413|                std::complex<double> vu_tmp[4];
   414|                for (int i = 0; i < 4; i++)
   415|                {
   416|                    vu_tmp[i] = vu_iat[index[i]];
   417|                }
   418|                vu_iat[index[0]] = 0.5 * (vu_tmp[0] + vu_tmp[3]);
   419|                vu_iat[index[3]] = 0.5 * (vu_tmp[0] - vu_tmp[3]);
   420|                vu_iat[index[1]] = 0.5 * (vu_tmp[1] + std::complex<double>(0.0, 1.0) * vu_tmp[2]);
   421|                vu_iat[index[2]] = 0.5 * (vu_tmp[1] - std::complex<double>(0.0, 1.0) * vu_tmp[2]);
   422|            }
   423|        }
   424|        }
   425|    }
   426|
   427|    // DIAGNOSTIC [N6]: state after VU calculation
   428|    if(iter <= 2)
   429|    // std::cout << "[DIAG-PW] cal_occ_pw iter=" << iter << " after VU calculation:" << std::endl;
   430|    if(iter <= 2)
   431|    // std::cout << "[DIAG-PW]   eff_pot_pw.size=" << this->eff_pot_pw.size() << std::endl;
   432|    if(iter <= 2 && this->eff_pot_pw.size() > 0) {
   433|        // std::cout << "[DIAG-PW]   eff_pot_pw[0..29]=";
   434|        for(int i=0;i<30 && i<(int)this->eff_pot_pw.size();i++)
   435|        // std::cout << " (" << this->eff_pot_pw[i].real() << "," << this->eff_pot_pw[i].imag() << ")";
   436|        std::cout << std::endl;
   437|    }
   438|    if(iter <= 2)
   439|    // std::cout << "[DIAG-PW]   energy_u=" << Plus_U::energy_u << std::endl;
   440|    if(iter <= 2)
   441|    // std::cout << "[DIAG-PW]   initialed_locale=false" << std::endl;
   442|
   443|    initialed_locale = false;
   444|
   445|    // update effective potential
   446|    ModuleBase::timer::tick("Plus_U", "cal_occ_pw");
   447|}
   448|/// calculate the local DFT+U effective potential matrix for PW base.
   449|void Plus_U::cal_VU_pot_pw(const int spin)
   450|{
   451|
   452|}
   453|