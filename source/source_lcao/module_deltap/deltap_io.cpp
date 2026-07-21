#include "deltap.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace deltap {

void DeltaP::verify_sum_rule()
{
    ModuleBase::TITLE("DeltaP", "verify_sum_rule");

    std::cout << " * DeltaP Sum Rule Check:" << std::endl;
    std::cout << "   P_total (DeltaP)  = (" << std::scientific << std::setprecision(6)
              << results_.P_total.x << ", " << results_.P_total.y << ", " << results_.P_total.z
              << ")" << std::endl;
    std::cout << "   P_total (ABACUS)  = (" << results_.P_abacus.x << ", "
              << results_.P_abacus.y << ", " << results_.P_abacus.z << ")" << std::endl;

    if ((std::abs(results_.P_abacus.x) + std::abs(results_.P_abacus.y) + std::abs(results_.P_abacus.z)) > 1e-10)
    {
        double num = 0.0, denom = 0.0;
        double p_ref[3] = {results_.P_abacus.x, results_.P_abacus.y, results_.P_abacus.z};
        double p_cal[3] = {results_.P_total.x, results_.P_total.y, results_.P_total.z};
        for (int a = 0; a < 3; ++a)
        {
            num += (p_cal[a] - p_ref[a]) * (p_cal[a] - p_ref[a]);
            denom += p_ref[a] * p_ref[a];
        }
        const double rel_error = std::sqrt(num) / std::sqrt(denom);
        std::cout << "   Relative error    = " << rel_error << std::endl;
        if (rel_error < 0.01)
        {
            std::cout << "   [PASS] Sum rule satisfied (< 1%)" << std::endl;
        }
        else
        {
            std::cout << "   [WARN] Sum rule NOT satisfied (> 1%)" << std::endl;
        }
    }
    else
    {
        std::cout << "   (no ABACUS reference for comparison)" << std::endl;
    }
}

void DeltaP::write_results(const UnitCell& ucell) const
{
    const std::string out_dir = "OUT." + PARAM.inp.suffix;
    const std::string filename = out_dir + "/deltap_results.dat";

    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        std::cerr << "Warning: cannot open " << filename << " for writing" << std::endl;
        return;
    }

    ofs << "# DeltaP atomic polarization decomposition" << std::endl;
    ofs << "# Direction: " << gdir_ << " (1=x, 2=y, 3=z)" << std::endl;
    ofs << "# SMO radius: " << rm_ << " Bohr" << std::endl;
    ofs << "#" << std::endl;
    ofs << "# Atom    Px          Py          Pz          (a.u.)" << std::endl;
    ofs << std::scientific << std::setprecision(8);

    for (int iat = 0; iat < nat_; iat++)
    {
        int ia, it;
        ucell.iat2iait(iat, &ia, &it);
        ofs << "  " << std::setw(4) << ucell.atom_label[it]
            << " " << std::setw(4) << ia
            << "  " << std::setw(14) << results_.P_I[iat].x
            << " " << std::setw(14) << results_.P_I[iat].y
            << " " << std::setw(14) << results_.P_I[iat].z
            << std::endl;
    }

    ofs << "#" << std::endl;
    ofs << "# Total   " << std::setw(14) << results_.P_total.x
        << " " << std::setw(14) << results_.P_total.y
        << " " << std::setw(14) << results_.P_total.z << std::endl;
    ofs << "# ABACUS  " << std::setw(14) << results_.P_abacus.x
        << " " << std::setw(14) << results_.P_abacus.y
        << " " << std::setw(14) << results_.P_abacus.z << std::endl;

    // Output per-atom electronic center displacement (from phase unwrapping)
    const int alpha_idx_io = gdir_ - 1;
    double a_alpha_io = 0.0;
    if (gdir_ == 1) a_alpha_io = ucell.lat0 * ucell.a1.norm();
    else if (gdir_ == 2) a_alpha_io = ucell.lat0 * ucell.a2.norm();
    else a_alpha_io = ucell.lat0 * ucell.a3.norm();

    ofs << "#" << std::endl;
    ofs << "# Per-atom electronic center displacement (phase unwrapping)" << std::endl;
    ofs << "# Atom    r_elec(bohr)   r_ion(bohr)   delta_r(bohr)  delta_r(A)" << std::endl;
    for (int iat = 0; iat < nat_; iat++)
    {
        int ia, it;
        ucell.iat2iait(iat, &ia, &it);
        double r_elec = results_.r_elec_center[iat][alpha_idx_io];
        double r_ion = 0.0;
        if (gdir_ == 1) r_ion = ucell.get_tau(iat).x * ucell.lat0;
        else if (gdir_ == 2) r_ion = ucell.get_tau(iat).y * ucell.lat0;
        else r_ion = ucell.get_tau(iat).z * ucell.lat0;
        double delta_r = r_ion - r_elec;
        double delta_r_A = delta_r / 1.8897259886;
        ofs << "  " << std::setw(4) << ucell.atom_label[it]
            << " " << std::setw(4) << ia
            << "  " << std::setw(14) << r_elec
            << "  " << std::setw(14) << r_ion
            << "  " << std::setw(14) << delta_r
            << "  " << std::setw(14) << delta_r_A
            << std::endl;
    }

    (void)a_alpha_io;

    // Output full w_In matrix for post-processing with Wannier90 WF centers
    {
        const std::string wfile = out_dir + "/deltap_smo_weights.dat";
        std::ofstream wfs(wfile);
        if (wfs.is_open())
        {
            int nocc_w = results_.smo_weights.size();
            int nat_w = (nocc_w > 0) ? results_.smo_weights[0].size() : 0;

            wfs << "# DeltaP SMO projection weights w_In" << std::endl;
            wfs << "# w_In = sum_{a in I} |<alpha_a|v_n>|^2" << std::endl;
            wfs << "# Non-orthogonal: normalize per n as w_In_norm = w_In / sum_I w_In" << std::endl;
            wfs << "# Row = WF index n (0-based), Column = atom index I (0-based)" << std::endl;
            wfs << "# n_occ = " << nocc_w << "  nat = " << nat_w << std::endl;
            wfs << std::scientific << std::setprecision(8);
            for (int n = 0; n < nocc_w; n++)
            {
                for (int iat = 0; iat < nat_w; iat++)
                {
                    if (iat > 0) wfs << " ";
                    wfs << results_.smo_weights[n][iat];
                }
                wfs << "\n";
            }
            wfs.close();
            std::cout << " * DeltaP SMO weights written to " << wfile << std::endl;
        }
    }

    ofs.close();
    std::cout << " * DeltaP results written to " << filename << std::endl;
}

} // namespace deltap
