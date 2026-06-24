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

    const int alpha_idx = gdir_ - 1;
    const double p_total = results_.P_total[alpha_idx];
    const double p_abacus = results_.P_abacus[alpha_idx];

    std::cout << " * DeltaP Sum Rule Check:" << std::endl;
    std::cout << "   P_total (DeltaP)  = " << std::scientific << std::setprecision(6)
              << p_total << std::endl;
    std::cout << "   P_total (ABACUS)  = " << p_abacus << std::endl;

    if (std::abs(p_abacus) > 1e-10)
    {
        const double rel_error = std::abs(p_total - p_abacus) / std::abs(p_abacus);
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

    ofs.close();
    std::cout << " * DeltaP results written to " << filename << std::endl;
}

} // namespace deltap
