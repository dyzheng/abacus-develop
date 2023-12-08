#include "dftu.h"
#include "module_base/timer.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace ModuleDFTU
{

void DFTU::cal_VU_pot_mat_complex(const int spin, const bool newlocale, std::complex<double>* VU)
{
    ModuleBase::TITLE("DFTU", "cal_VU_pot_mat_complex");
    ModuleBase::GlobalFunc::ZEROS(VU, this->LM->ParaV->nloc);

    for (int it = 0; it < GlobalC::ucell.ntype; ++it)
    {
        if (this->orbital_corr[it] == -1)
            continue;
        for (int ia = 0; ia < GlobalC::ucell.atoms[it].na; ia++)
        {
            const int iat = GlobalC::ucell.itia2iat(it, ia);
            for (int L = 0; L <= GlobalC::ucell.atoms[it].nwl; L++)
            {
                if (L != this->orbital_corr[it])
                    continue;

                for (int n = 0; n < GlobalC::ucell.atoms[it].l_nchi[L]; n++)
                {
                    if (n != 0)
                        continue;

                    for (int m1 = 0; m1 < 2 * L + 1; m1++)
                    {
                        for (int ipol1 = 0; ipol1 < GlobalV::NPOL; ipol1++)
                        {
                            const int mu = this->LM->ParaV->global2local_row(this->iatlnmipol2iwt[iat][L][n][m1][ipol1]);
                            if (mu < 0)
                                continue;

                            for (int m2 = 0; m2 < 2 * L + 1; m2++)
                            {
                                for (int ipol2 = 0; ipol2 < GlobalV::NPOL; ipol2++)
                                {
                                    const int nu
                                        = this->LM->ParaV->global2local_col(this->iatlnmipol2iwt[iat][L][n][m2][ipol2]);
                                    if (nu < 0)
                                        continue;

                                    int m1_all = m1 + (2 * L + 1) * ipol1;
                                    int m2_all = m2 + (2 * L + 1) * ipol2;

                                    double val = get_onebody_eff_pot(it, iat, L, n, spin, m1_all, m2_all, newlocale);
                                    VU[nu * this->LM->ParaV->nrow + mu] = std::complex<double>(val, 0.0);
                                } // ipol2
                            } // m2
                        } // ipol1
                    } // m1
                } // n
            } // l
        } // ia
    } // it

    return;
}

void DFTU::cal_VU_pot_mat_real(const int spin, const bool newlocale, double* VU)
{
    ModuleBase::TITLE("DFTU", "cal_VU_pot_mat_real");
    ModuleBase::GlobalFunc::ZEROS(VU, this->LM->ParaV->nloc);

    for (int it = 0; it < GlobalC::ucell.ntype; ++it)
    {
        if (this->orbital_corr[it] == -1)
            continue;
        for (int ia = 0; ia < GlobalC::ucell.atoms[it].na; ia++)
        {
            const int iat = GlobalC::ucell.itia2iat(it, ia);
            for (int L = 0; L <= GlobalC::ucell.atoms[it].nwl; L++)
            {
                if (L != this->orbital_corr[it])
                    continue;

                for (int n = 0; n < GlobalC::ucell.atoms[it].l_nchi[L]; n++)
                {
                    if (n != 0)
                        continue;

                    for (int m1 = 0; m1 < 2 * L + 1; m1++)
                    {
                        for (int ipol1 = 0; ipol1 < GlobalV::NPOL; ipol1++)
                        {
                            const int mu = this->LM->ParaV->global2local_row(this->iatlnmipol2iwt[iat][L][n][m1][ipol1]);
                            if (mu < 0)
                                continue;
                            for (int m2 = 0; m2 < 2 * L + 1; m2++)
                            {
                                for (int ipol2 = 0; ipol2 < GlobalV::NPOL; ipol2++)
                                {
                                    const int nu
                                        = this->LM->ParaV->global2local_col(this->iatlnmipol2iwt[iat][L][n][m2][ipol2]);
                                    if (nu < 0)
                                        continue;

                                    int m1_all = m1 + (2 * L + 1) * ipol1;
                                    int m2_all = m2 + (2 * L + 1) * ipol2;

                                    VU[nu * this->LM->ParaV->nrow + mu]
                                        = this->get_onebody_eff_pot(it, iat, L, n, spin, m1_all, m2_all, newlocale);

                                } // ipol2
                            } // m2
                        } // ipol1
                    } // m1
                } // n
            } // l
        } // ia
    } // it

    return;
}

void DFTU::cal_VU_pot_atompair(const int spin, const bool newlocale, hamilt::HContainer<double>* vu)
{
    ModuleBase::TITLE("DFTU", "cal_VU_pot_mat_real");
    vu->set_zero();
    if (!this->initialed_locale)
    {
        return;
    }
    const int npol = GlobalV::NPOL;

    for (int it = 0; it < GlobalC::ucell.ntype; ++it)
    {
        // skip elements without plus-U
        if (this->orbital_corr[it] == -1)
            continue;
        for (int ia = 0; ia < GlobalC::ucell.atoms[it].na; ia++)
        {
            const int iat = GlobalC::ucell.itia2iat(it, ia);
            // choose the target L-orbital by user input
            const int L = this->orbital_corr[it];
            if (L > GlobalC::ucell.atoms[it].nwl || GlobalC::ucell.atoms[it].l_nchi[L]<=0)
                continue;
            const int n = 0;
            /*for (int L = 0; L <= GlobalC::ucell.atoms[it].nwl; L++)
            {
                if (L != this->orbital_corr[it])
                    continue;*/

                /*for (int n = 0; n < GlobalC::ucell.atoms[it].l_nchi[L]; n++)
                {
                    if (n != 0)
                        continue;*/

            // find the target atom-pair
            hamilt::AtomPair<double>* ap_iat = vu->find_pair(iat, iat);
            if (ap_iat == nullptr)
                continue;
            const int iw_begin = this->iatlnmipol2iwt[iat][L][n][0][0] - this->iatlnmipol2iwt[iat][0][0][0][0];
            const int iw_end = this->iatlnmipol2iwt[iat][L][n][2 * L][npol-1] - this->iatlnmipol2iwt[iat][0][0][0][0];
            double* data_pointer = &(ap_iat->get_value(iw_begin, iw_begin));
            const int col_size = ap_iat->get_col_size();
            // loop for m, m'
            int m1 = -1;
            for (int irow = iw_begin; irow <= iw_end; irow += npol)
            {
                const int ipol1 = irow % npol;
                if(!ipol1) m1++;
                int m2 = -1;
                int m1_all = m1 + (2 * L + 1) * ipol1;
                for (int icol = 0; icol <= iw_end - iw_begin; icol++)
                {
                    int ipol2 = icol % npol;
                    if(!ipol2) m2++;
                    int m2_all = m2 + (2 * L + 1) * ipol2;
                    data_pointer[icol] = this->get_onebody_eff_pot(it, iat, L, n, spin, m1_all, m2_all, newlocale);
                }
                data_pointer += col_size;
            }
        } // ia
    } // it

    return;
}

double DFTU::get_onebody_eff_pot(const int T,
                                 const int iat,
                                 const int L,
                                 const int N,
                                 const int spin,
                                 const int m0,
                                 const int m1,
                                 const bool newlocale)
{
    ModuleBase::TITLE("DFTU", "get_onebody_eff_pot");

    double VU = 0.0;

    switch (cal_type)
    {
    case 1: // rotationally invarient formalism and FLL double counting

        break;

    case 2: // rotationally invarient formalism and AMF double counting

        break;

    case 3: // simplified formalism and FLL double counting
        if (newlocale)
        {
            if (Yukawa)
            {
                if (m0 == m1)
                    VU = (this->U_Yukawa[T][L][N] - this->J_Yukawa[T][L][N])
                         * (0.5 - this->locale[iat][L][N][spin](m0, m1));
                else
                    VU = -(this->U_Yukawa[T][L][N] - this->J_Yukawa[T][L][N]) * this->locale[iat][L][N][spin](m0, m1);
            }
            else
            {
                if (m0 == m1)
                    VU = (this->U[T]) * (0.5 - this->locale[iat][L][N][spin](m0, m1));
                else
                    VU = -(this->U[T]) * this->locale[iat][L][N][spin](m0, m1);
            }
        }
        else
        {
            if (Yukawa)
            {
                if (m0 == m1)
                    VU = (this->U_Yukawa[T][L][N] - this->J_Yukawa[T][L][N])
                         * (0.5 - this->locale_save[iat][L][N][spin](m0, m1));
                else
                    VU = -(this->U_Yukawa[T][L][N] - this->J_Yukawa[T][L][N])
                         * this->locale_save[iat][L][N][spin](m0, m1);
            }
            else
            {
                if (m0 == m1)
                    VU = (this->U[T]) * (0.5 - this->locale_save[iat][L][N][spin](m0, m1));
                else
                    VU = -(this->U[T]) * this->locale_save[iat][L][N][spin](m0, m1);
            }
        }

        break;

    case 4: // simplified formalism and AMF double counting

        break;
    }

    return VU;
}
} // namespace ModuleDFTU