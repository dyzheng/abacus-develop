#include "deltap.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include <cmath>

namespace deltap {

void DeltaP::gauge_fix_smo_anchored(int nbands)
{
    ModuleBase::TITLE("DeltaP", "gauge_fix_smo_anchored");
    ModuleBase::timer::start("DeltaP", "gauge_fix_smo_anchored");

    if (nppstr_ == 0)
    {
        ModuleBase::timer::end("DeltaP", "gauge_fix_smo_anchored");
        return;
    }

    gauge_phase_.resize(nppstr_);
    for (int j = 0; j < nppstr_; ++j)
    {
        gauge_phase_[j].resize(nbands, std::complex<double>(1.0, 0.0));
    }
    anchor_iat_.resize(nbands, -1);
    anchor_lm_.resize(nbands, -1);
    phase_corrections_.resize(nbands, std::complex<double>(1.0, 0.0));

    // Phase 1: Determine anchor SMO at k_0
    for (int n = 0; n < nbands; ++n)
    {
        double max_proj = 0.0;
        for (int iat = 0; iat < nat_; ++iat)
        {
            int r = nproj_per_atom_[iat];
            if (r == 0) continue;
            if (kstring_data_[0].D_I.size() <= static_cast<size_t>(iat)) continue;
            if (kstring_data_[0].D_I[iat].size() == 0) continue;

            for (int lm = 0; lm < r; ++lm)
            {
                if (kstring_data_[0].D_I[iat][lm].size() <= static_cast<size_t>(n)) continue;
                double proj = std::abs(kstring_data_[0].D_I[iat][lm][n]);
                if (proj > max_proj)
                {
                    max_proj = proj;
                    anchor_iat_[n] = iat;
                    anchor_lm_[n] = lm;
                }
            }
        }

        if (anchor_iat_[n] < 0)
        {
            gauge_phase_[0][n] = std::complex<double>(1.0, 0.0);
            continue;
        }

        std::complex<double> D_anchor = kstring_data_[0].D_I[anchor_iat_[n]][anchor_lm_[n]][n];
        double abs_D = std::abs(D_anchor);
        if (abs_D < 1e-15)
        {
            gauge_phase_[0][n] = std::complex<double>(1.0, 0.0);
        }
        else
        {
            gauge_phase_[0][n] = std::conj(D_anchor) / abs_D;
        }
    }

    // Phase 2: Compute gauge phases at k_1, ..., k_{nppstr-1}
    for (int j = 1; j < nppstr_; ++j)
    {
        for (int n = 0; n < nbands; ++n)
        {
            if (anchor_iat_[n] < 0)
            {
                gauge_phase_[j][n] = gauge_phase_[j - 1][n];
                continue;
            }

            int iat = anchor_iat_[n];
            int lm = anchor_lm_[n];

            if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat) ||
                kstring_data_[j].D_I[iat].size() <= static_cast<size_t>(lm) ||
                kstring_data_[j].D_I[iat][lm].size() <= static_cast<size_t>(n))
            {
                gauge_phase_[j][n] = gauge_phase_[j - 1][n];
                continue;
            }

            std::complex<double> D_anchor = kstring_data_[j].D_I[iat][lm][n];
            double abs_D = std::abs(D_anchor);

            // Anchor jump detection
            if (abs_D < anchor_thr_)
            {
                double new_max = 0.0;
                int new_iat = -1, new_lm = -1;
                for (int iat2 = 0; iat2 < nat_; ++iat2)
                {
                    int r = nproj_per_atom_[iat2];
                    if (r == 0) continue;
                    if (kstring_data_[j].D_I.size() <= static_cast<size_t>(iat2)) continue;
                    if (kstring_data_[j].D_I[iat2].size() == 0) continue;

                    for (int lm2 = 0; lm2 < r; ++lm2)
                    {
                        if (kstring_data_[j].D_I[iat2][lm2].size() <= static_cast<size_t>(n)) continue;
                        double proj = std::abs(kstring_data_[j].D_I[iat2][lm2][n]);
                        if (proj > new_max)
                        {
                            new_max = proj;
                            new_iat = iat2;
                            new_lm = lm2;
                        }
                    }
                }

                if (new_iat >= 0 && new_iat != iat)
                {
                    std::complex<double> D_old = D_anchor;
                    std::complex<double> D_new = kstring_data_[j].D_I[new_iat][new_lm][n];
                    double delta_phi = std::arg(D_new) - std::arg(D_old);
                    phase_corrections_[n] *= std::polar(1.0, -delta_phi);

                    anchor_iat_[n] = new_iat;
                    anchor_lm_[n] = new_lm;
                    D_anchor = D_new;
                    abs_D = std::abs(D_new);
                }
            }

            std::complex<double> g(1.0, 0.0);
            if (abs_D >= 1e-15)
            {
                g = std::conj(D_anchor) / abs_D;
            }

            // Continuous phase tracking
            std::complex<double> g_prev = gauge_phase_[j - 1][n];
            std::complex<double> overlap = g * std::conj(g_prev);
            if (overlap.real() < 0.0)
            {
                g = -g;
            }

            gauge_phase_[j][n] = g;
        }
    }

    ModuleBase::timer::end("DeltaP", "gauge_fix_smo_anchored");
}

} // namespace deltap
