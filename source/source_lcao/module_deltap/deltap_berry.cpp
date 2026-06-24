#include "deltap.h"
#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"

#include <cmath>

namespace deltap {

void DeltaP::compute_S_k(int ik)
{
    ModuleBase::TITLE("DeltaP", "compute_S_k");
    ModuleBase::timer::start("DeltaP", "compute_S_k");

    const int nat = nat_;
    kstring_data_[ik].S_k.resize(nat);
    kstring_data_[ik].dS_k.resize(nat);

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];
        kstring_data_[ik].S_k[iat].resize(r);
        kstring_data_[ik].dS_k[iat].resize(3);
        for (int alpha = 0; alpha < 3; alpha++)
        {
            kstring_data_[ik].dS_k[iat][alpha].resize(r);
        }

        for (int lm = 0; lm < r; lm++)
        {
            for (const auto& od : overlap_R_[iat])
            {
                const double arg = ModuleBase::TWO_PI * (
                    kstring_data_[ik].kvec_d.x * od.R_index.x +
                    kstring_data_[ik].kvec_d.y * od.R_index.y +
                    kstring_data_[ik].kvec_d.z * od.R_index.z);
                const std::complex<double> phase(std::cos(arg), std::sin(arg));

                for (const auto& nlm_entry : od.nlm)
                {
                    const int iw_global = nlm_entry.first;
                    const std::vector<double>& nlm_vec = nlm_entry.second;
                    const int iw_local = paraV_->global2local_row(iw_global);
                    if (iw_local < 0) continue;

                    if ((int)kstring_data_[ik].S_k[iat][lm].size() <= iw_local)
                    {
                        kstring_data_[ik].S_k[iat][lm].resize(iw_local + 1, {0.0, 0.0});
                        for (int a = 0; a < 3; a++)
                            kstring_data_[ik].dS_k[iat][a][lm].resize(iw_local + 1, {0.0, 0.0});
                    }

                    kstring_data_[ik].S_k[iat][lm][iw_local] += phase * nlm_vec[lm];

                    const std::complex<double> i_phase(0.0, 1.0);
                    for (int a = 0; a < 3; a++)
                    {
                        double R_alpha = (a == 0) ? od.R_index.x : (a == 1) ? od.R_index.y : od.R_index.z;
                        kstring_data_[ik].dS_k[iat][a][lm][iw_local]
                            += i_phase * R_alpha * phase * nlm_vec[lm];
                    }
                }
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_S_k");
}

void DeltaP::compute_D_I(int ik, const std::complex<double>* psi_k, int nbands, int nrow_local)
{
    ModuleBase::TITLE("DeltaP", "compute_D_I");
    ModuleBase::timer::start("DeltaP", "compute_D_I");

    const int nat = nat_;
    kstring_data_[ik].D_I.resize(nat);

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];
        kstring_data_[ik].D_I[iat].resize(r);
        for (int lm = 0; lm < r; lm++)
        {
            kstring_data_[ik].D_I[iat][lm].resize(nbands, {0.0, 0.0});
        }

        for (int lm = 0; lm < r; lm++)
        {
            const int s_size = kstring_data_[ik].S_k[iat][lm].size();
            for (int mu_local = 0; mu_local < s_size; mu_local++)
            {
                const std::complex<double> s_val = kstring_data_[ik].S_k[iat][lm][mu_local];
                if (std::abs(s_val) < 1e-15) continue;

                const std::complex<double> s_conj = std::conj(s_val);
                for (int n = 0; n < nbands; n++)
                {
                    kstring_data_[ik].D_I[iat][lm][n] += s_conj * psi_k[mu_local + n * nrow_local];
                }
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_D_I");
}

void DeltaP::compute_berry_connection(int ik, const std::complex<double>* psi_k,
                                       int nbands, int nrow_local, const double* wg)
{
    ModuleBase::TITLE("DeltaP", "compute_berry_connection");
    ModuleBase::timer::start("DeltaP", "compute_berry_connection");

    const int nat = nat_;
    const int nppstr = nppstr_;

    if (A_nk_.empty())
    {
        A_nk_.resize(nat);
        for (int iat = 0; iat < nat; iat++)
        {
            A_nk_[iat].resize(nppstr);
            for (int j = 0; j < nppstr; j++)
            {
                A_nk_[iat][j].resize(nbands, ModuleBase::Vector3<std::complex<double>>(0.0, 0.0, 0.0));
            }
        }
    }

    int ik_next = (ik + 1) % nppstr;
    int ik_prev = (ik - 1 + nppstr) % nppstr;
    const double dk_dir = 1.0 / (nppstr - 1);
    const double inv_2dk = 1.0 / (2.0 * dk_dir);

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];

        for (int n = 0; n < nbands; n++)
        {
            for (int alpha = 0; alpha < 3; alpha++)
            {
                std::complex<double> term1 = {0.0, 0.0};
                std::complex<double> term2 = {0.0, 0.0};

                for (int lm = 0; lm < r; lm++)
                {
                    // term1: <psi_nk | d_k_alpha alpha> * <alpha | psi_nk>
                    std::complex<double> bra_grad = {0.0, 0.0};
                    const int s_size = kstring_data_[ik].dS_k[iat][alpha][lm].size();
                    for (int mu = 0; mu < s_size; mu++)
                    {
                        const std::complex<double> ds_val = kstring_data_[ik].dS_k[iat][alpha][lm][mu];
                        const std::complex<double> c_val = psi_k[mu + n * nrow_local];
                        bra_grad += std::conj(ds_val) * c_val;
                    }
                    term1 += bra_grad * kstring_data_[ik].D_I[iat][lm][n];

                    // term2: <psi_nk | alpha> * d_k <alpha | psi_nk>
                    std::complex<double> d_D = {0.0, 0.0};
                    if (ik_next != ik && ik_prev != ik)
                    {
                        d_D = (kstring_data_[ik_next].D_I[iat][lm][n]
                               - kstring_data_[ik_prev].D_I[iat][lm][n]) * inv_2dk;
                    }
                    term2 += std::conj(kstring_data_[ik].D_I[iat][lm][n]) * d_D;
                }

                A_nk_[iat][ik][n][alpha] = term1 + term2;
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_berry_connection");
}

void DeltaP::integrate_polarization(const UnitCell& ucell, int nbands)
{
    ModuleBase::TITLE("DeltaP", "integrate_polarization");
    ModuleBase::timer::start("DeltaP", "integrate_polarization");

    const int nat = nat_;
    const int alpha_idx = gdir_ - 1;

    // Lattice vector length along gdir (in Bohr), matching berryphase convention
    double a_alpha = 0.0;
    if (gdir_ == 1) { a_alpha = ucell.lat0 * ucell.a1.norm(); }
    else if (gdir_ == 2) { a_alpha = ucell.lat0 * ucell.a2.norm(); }
    else { a_alpha = ucell.lat0 * ucell.a3.norm(); }

    const double dk_dir = 1.0 / (nppstr_ - 1);
    const double prefactor = -1.0 / (2.0 * ModuleBase::PI * a_alpha) * dk_dir;

    results_.P_I.resize(nat, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I.resize(nat, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));

    double occupied_bands = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occupied_bands - std::floor(occupied_bands)) > 0.0)
    {
        occupied_bands = std::floor(occupied_bands) + 1.0;
    }
    const int occ_nbands = static_cast<int>(occupied_bands);

    for (int iat = 0; iat < nat; iat++)
    {
        double gamma = 0.0;
        for (int j = 0; j < nppstr_; j++)
        {
            for (int n = 0; n < occ_nbands && n < nbands; n++)
            {
                double a_imag = A_nk_[iat][j][n][alpha_idx].imag();
                gamma += a_imag;
            }
        }
        results_.gamma_I[iat][alpha_idx] = gamma;
        results_.P_I[iat][alpha_idx] = prefactor * gamma;
    }

    results_.P_total = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    for (int iat = 0; iat < nat; iat++)
    {
        results_.P_total += results_.P_I[iat];
    }

    ModuleBase::timer::end("DeltaP", "integrate_polarization");
}

} // namespace deltap
