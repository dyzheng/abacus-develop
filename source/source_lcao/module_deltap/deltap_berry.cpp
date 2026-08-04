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
                            += ModuleBase::TWO_PI * i_phase * R_alpha * phase * nlm_vec[lm];
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
    // nbands is the GLOBAL band count (callers pass paraV_->get_wfc_global_nbands()).
    // D_I is indexed by GLOBAL band so that the per-rank partial sums can be
    // combined with a single uniform-count Allreduce (A' scheme):
    //   - rows:   each rank sums over its local orbitals (mu_local);
    //   - bands:  psi's columns are 2D-block-cyclically distributed, so local
    //             column n maps to global band g = local2global_col(n).  Bands
    //             not owned by this rank stay zero (filled by other col-ranks).
    // psi_k is the local nrow x ncol_bands block, column-major with LLD nrow_local.
    // In serial, local2global_col is the identity, so this reduces to the
    // original single-rank formula (zero behavior change).
    const int ncol_local = paraV_->ncol_bands;
    kstring_data_[ik].D_I.resize(nat);

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];
        kstring_data_[ik].D_I[iat].resize(r);
        for (int lm = 0; lm < r; lm++)
        {
            kstring_data_[ik].D_I[iat][lm].assign(nbands, {0.0, 0.0});
        }

        for (int lm = 0; lm < r; lm++)
        {
            const int s_size = kstring_data_[ik].S_k[iat][lm].size();
            for (int mu_local = 0; mu_local < s_size; mu_local++)
            {
                const std::complex<double> s_val = kstring_data_[ik].S_k[iat][lm][mu_local];
                if (std::abs(s_val) < 1e-15) continue;

                const std::complex<double> s_conj = std::conj(s_val);
                for (int n = 0; n < ncol_local; n++)
                {
                    const int g = paraV_->local2global_col(n);
                    if (g < 0 || g >= nbands) continue;
                    kstring_data_[ik].D_I[iat][lm][g] += s_conj * psi_k[mu_local + n * nrow_local];
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
                    // Determine gauge phase for this (ik, n)
                    std::complex<double> g_nk(1.0, 0.0);
                    if (gauge_enabled_ && static_cast<int>(gauge_phase_.size()) > ik
                        && static_cast<int>(gauge_phase_[ik].size()) > n)
                    {
                        g_nk = gauge_phase_[ik][n];
                    }

                    // term1: <psi|d_k alpha> * <alpha|psi>
                    // Bug fix: conj(C)*dS (not conj(dS)*C) for correct <psi|d_k alpha>
                    // Gauge: C -> C * g_nk
                    std::complex<double> bra_grad = {0.0, 0.0};
                    const int s_size = kstring_data_[ik].dS_k[iat][alpha][lm].size();
                    for (int mu = 0; mu < s_size; mu++)
                    {
                        const std::complex<double> ds_val = kstring_data_[ik].dS_k[iat][alpha][lm][mu];
                        const std::complex<double> c_val = psi_k[mu + n * nrow_local];
                        bra_grad += std::conj(c_val * g_nk) * ds_val;
                    }
                    std::complex<double> D_I_gauge = kstring_data_[ik].D_I[iat][lm][n] * g_nk;
                    term1 += bra_grad * D_I_gauge;

                    // term2: conj(D_I) * d_k D_I
                    // Gauge: apply gauge phases to D_I at ik, ik_next, ik_prev
                    std::complex<double> d_D = {0.0, 0.0};
                    if (ik_next != ik && ik_prev != ik)
                    {
                        std::complex<double> g_next(1.0, 0.0);
                        std::complex<double> g_prev(1.0, 0.0);
                        if (gauge_enabled_)
                        {
                            if (static_cast<int>(gauge_phase_.size()) > ik_next
                                && static_cast<int>(gauge_phase_[ik_next].size()) > n)
                                g_next = gauge_phase_[ik_next][n];
                            if (static_cast<int>(gauge_phase_.size()) > ik_prev
                                && static_cast<int>(gauge_phase_[ik_prev].size()) > n)
                                g_prev = gauge_phase_[ik_prev][n];
                        }
                        std::complex<double> D_next = kstring_data_[ik_next].D_I[iat][lm][n] * g_next;
                        std::complex<double> D_prev = kstring_data_[ik_prev].D_I[iat][lm][n] * g_prev;
                        d_D = (D_next - D_prev) * inv_2dk;
                    }
                    term2 += std::conj(D_I_gauge) * d_D;
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

    results_.P_I.resize(nat, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I.resize(nat, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));

    double occupied_bands = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occupied_bands - std::floor(occupied_bands)) > 0.0)
        occupied_bands = std::floor(occupied_bands) + 1.0;
    const int occ_nbands = static_cast<int>(occupied_bands);
    const double dk_dir = 1.0 / (nppstr_ - 1);

    // Integrate A_nk over k-points for all three directions
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        double a_alpha = 0.0;
        if (alpha == 0)      a_alpha = ucell.lat0 * ucell.a1.norm();
        else if (alpha == 1) a_alpha = ucell.lat0 * ucell.a2.norm();
        else                 a_alpha = ucell.lat0 * ucell.a3.norm();
        const double omega = ucell.omega;
        const double prefactor = -a_alpha / (2.0 * ModuleBase::PI * omega) * dk_dir;

        for (int iat = 0; iat < nat; iat++)
        {
            double gamma = 0.0;
            for (int j = 0; j < nppstr_; j++)
                for (int n = 0; n < occ_nbands && n < nbands; n++)
                    gamma += A_nk_[iat][j][n][alpha].imag();
            results_.gamma_I[iat][alpha] = gamma;
            results_.P_I[iat][alpha] = prefactor * gamma;
        }
    }

    results_.P_total = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    for (int iat = 0; iat < nat; iat++)
        results_.P_total += results_.P_I[iat];

    std::cout << "   [A_nk integration] P_total = ("
              << std::scientific << std::setprecision(6)
              << results_.P_total.x << ", " << results_.P_total.y << ", "
              << results_.P_total.z << ")" << std::endl;

    ModuleBase::timer::end("DeltaP", "integrate_polarization");
}

} // namespace deltap
