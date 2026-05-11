// gga_grad=3: SF noncollinear GGA using built-in PBE.
// Sequential implementation (no OMP) for correctness verification.

#include "xc_functional_gga_noncol_sf_builtin.h"

#include "module_base/constants.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/vector3.h"
#include "module_parameter/parameter.h"
#include "xc_functional.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace ModuleXC
{
namespace NCGGA_SF_Builtin
{

std::tuple<double, double, ModuleBase::matrix> v_xc_ncgga_sf_builtin(
    const int& nrxx, const double& omega, const double tpiba, const Charge* const chr)
{
    ModuleBase::TITLE("XC_Functional", "v_xc_ncgga_sf_builtin");
    ModuleBase::timer::tick("XC_Functional", "v_xc_ncgga_sf_builtin");

    if (GlobalV::NSPIN != 4 || (!GlobalV::DOMAG && !GlobalV::DOMAG_Z))
        throw std::domain_error("v_xc_ncgga_sf_builtin requires NSPIN==4.");

    ModulePW::PW_Basis* rhopw = chr->rhopw;
    const int npw = rhopw->npw;
    const double e2 = ModuleBase::e2;
    constexpr double vanishing = 1e-10;
    constexpr double epsr = 1e-6;
    const double fac = 0.5;
    const bool is_gga = (XC_Functional::get_func_type() == 2 || XC_Functional::get_func_type() == 4);

    // --- Compute rhotmp1/2, mag_part (same as gradcorr NSPIN=4) ---
    std::vector<double> rhotmp1(nrxx), rhotmp2(nrxx), amag(nrxx);
    std::vector<double> mag_part(3 * nrxx, 0.0);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double mx = chr->rho[1][ir], my = chr->rho[2][ir], mz = chr->rho[3][ir];
        amag[ir] = std::sqrt(mx*mx + my*my + mz*mz);
        rhotmp1[ir] = 0.5 * (chr->rho[0][ir] + amag[ir]);
        rhotmp2[ir] = 0.5 * (chr->rho[0][ir] - amag[ir]);
        if (amag[ir] > 1e-12)
        {
            mag_part[ir] = mx / amag[ir];
            mag_part[ir + nrxx] = my / amag[ir];
            mag_part[ir + 2*nrxx] = mz / amag[ir];
        }
    }
    for (int ir = 0; ir < nrxx; ++ir)
    {
        rhotmp1[ir] += fac * chr->rho_core[ir];
        rhotmp2[ir] += fac * chr->rho_core[ir];
    }

    // --- FFT and gradients (matching gradcorr gga_grad=2 exactly) ---
    // Step 1: compute grad(rho[0] + rho_core) = grad of total charge
    std::vector<std::complex<double>> rhogsum1(npw), tmp_recip(npw);
    rhopw->real2recip(chr->rho[0], rhogsum1.data());
    for (int ig = 0; ig < npw; ++ig)
        rhogsum1[ig] += chr->rhog_core[ig];

    std::vector<ModuleBase::Vector3<double>> gdr1(nrxx), gdr2(nrxx);
    std::vector<ModuleBase::Vector3<double>> gdr_mag(nrxx);
    XC_Functional::grad_rho(rhogsum1.data(), gdr1.data(), rhopw, tpiba);

    // Step 2: split total charge gradient, then add/subtract magnetic gradients
    // gdr_total = grad(rho[0] + rho_core)
    // gdr1 = 0.5*gdr_total + Σ 0.5*m̂_μ*grad(m_μ) = grad(rho_up + 0.5*core)
    // gdr2 = 0.5*gdr_total - Σ 0.5*m̂_μ*grad(m_μ) = grad(rho_dw + 0.5*core)
    for (int ir = 0; ir < nrxx; ++ir)
    {
        gdr_mag[ir] = gdr1[ir];
        gdr1[ir] = 0.5 * gdr_mag[ir];
        gdr2[ir] = 0.5 * gdr_mag[ir];
    }
    for (int is = 1; is <= 3; ++is)
    {
        rhopw->real2recip(chr->rho[is], tmp_recip.data());
        XC_Functional::grad_rho(tmp_recip.data(), gdr_mag.data(), rhopw, tpiba);
        const double* mp = mag_part.data() + (is-1)*nrxx;
        for (int ir = 0; ir < nrxx; ++ir)
        {
            const ModuleBase::Vector3<double> g = 0.5 * gdr_mag[ir] * mp[ir];
            gdr1[ir] += g;
            gdr2[ir] -= g;
        }
    }

    // --- LDA potential (4-component basis) ---
    double etxc = 0, vtxc = 0;
    ModuleBase::matrix v(GlobalV::NSPIN, nrxx);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double arho = std::abs(chr->rho[0][ir] + chr->rho_core[ir]);
        if (arho <= vanishing) continue;

        double zeta = amag[ir] / arho;
        if (std::abs(zeta) > 1.0) zeta = (zeta > 0) ? 1.0 : -1.0;
        double exc = 0, vxc[2] = {0, 0};
        XC_Functional::xc_spin(arho, zeta, exc, vxc[0], vxc[1]);

        v(0, ir) = e2 * 0.5 * (vxc[0] + vxc[1]);
        vtxc += v(0, ir) * chr->rho[0][ir];

        if (amag[ir] > vanishing)
        {
            const double vs = e2 * 0.5 * (vxc[0] - vxc[1]);
            const double inv_a = 1.0 / amag[ir];
            for (int mu = 1; mu < 4; ++mu)
            {
                v(mu, ir) = vs * chr->rho[mu][ir] * inv_a;
                vtxc += v(mu, ir) * chr->rho[mu][ir];
            }
        }
        etxc += e2 * exc * arho;
    }

    // --- GGA: compute in (up, down) basis, then rotate ---
    if (is_gga)
    {
        double etxcgc = 0, vtxcgc = 0;
        std::vector<double> vup_gga(nrxx, 0), vdw_gga(nrxx, 0);
        std::vector<ModuleBase::Vector3<double>> h1(nrxx), h2(nrxx);

        for (int ir = 0; ir < nrxx; ++ir)
        {
            double sx = 0, v1xup = 0, v1xdw = 0, v2xup = 0, v2xdw = 0;
            double sc = 0, v1cup = 0, v1cdw = 0, v2c = 0;
            const double grho2a = gdr1[ir] * gdr1[ir];
            const double grho2b = gdr2[ir] * gdr2[ir];
            const double rh = rhotmp1[ir] + rhotmp2[ir];

            XC_Functional::gcx_spin(rhotmp1[ir], rhotmp2[ir], grho2a, grho2b,
                                    sx, v1xup, v1xdw, v2xup, v2xdw);

            if (rh > epsr)
            {
                double zeta = (rhotmp1[ir] - rhotmp2[ir]) / rh;
                zeta = std::fabs(zeta);
                if (zeta > 1.0 - epsr) zeta = 1.0 - epsr;
                const double grh2 = (gdr1[ir] + gdr2[ir]) * (gdr1[ir] + gdr2[ir]);
                if (std::sqrt(std::abs(grh2)) > 1e-10)
                {
                    XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
                }
            }

            vup_gga[ir] = e2 * (v1xup + v1cup);
            vdw_gga[ir] = e2 * (v1xdw + v1cdw);

            const double v2cup = v2c, v2cdw = v2c, v2cud = v2c;
            h1[ir] = e2 * ((v2xup + v2cup) * gdr1[ir] + v2cud * gdr2[ir]);
            h2[ir] = e2 * ((v2xdw + v2cdw) * gdr2[ir] + v2cud * gdr1[ir]);

            vtxcgc += vup_gga[ir] * (rhotmp1[ir] - chr->rho_core[ir] * fac);
            vtxcgc += vdw_gga[ir] * (rhotmp2[ir] - chr->rho_core[ir] * fac);
            etxcgc += e2 * (sx + sc);
        }

        // --- div(h) correction (gga_grad=2 style) ---
        // v(0) += 0.5*(vup_gga + vdw_gga), then -= div(0.5*(h1+h2))
        // v(1-3) += 0.5*(vup_gga - vdw_gga)*mag_part, then -= div(0.5*(h1-h2)*mag_part)*mag_part

        // Add GGA density derivative to v
        for (int ir = 0; ir < nrxx; ++ir)
        {
            v(0, ir) += 0.5 * (vup_gga[ir] + vdw_gga[ir]);
            const double vdiff = 0.5 * (vup_gga[ir] - vdw_gga[ir]);
            for (int mu = 1; mu < 4; ++mu)
            {
                v(mu, ir) += vdiff * mag_part[ir + (mu-1)*nrxx];
            }
        }

        // div(h) correction
        std::vector<double> dh(nrxx);
        std::vector<ModuleBase::Vector3<double>> tmp_h(nrxx);

        // div(0.5*(h1+h2)) -> v(0)
        for (int ir = 0; ir < nrxx; ++ir)
            tmp_h[ir] = 0.5 * (h1[ir] + h2[ir]);
        XC_Functional::grad_dot(tmp_h.data(), dh.data(), rhopw, tpiba);
        for (int ir = 0; ir < nrxx; ++ir)
            v(0, ir) -= dh[ir];
        double sum = 0;
        for (int ir = 0; ir < nrxx; ++ir)
            sum += dh[ir] * chr->rho[0][ir];
        vtxcgc -= sum;

        // SF: div(0.5*(h1-h2)*t_mu) -> v(mu) for each mu independently
        // This retains the (h1-h2)·∇t_mu correction term that gga_grad=2 drops
        // via the t_mu * Σ t_nu * div(...) projection.
        for (int mu = 1; mu < 4; ++mu)
        {
            const double* mp = mag_part.data() + (mu-1)*nrxx;
            for (int ir = 0; ir < nrxx; ++ir)
                tmp_h[ir] = 0.5 * (h1[ir] - h2[ir]) * mp[ir];
            XC_Functional::grad_dot(tmp_h.data(), dh.data(), rhopw, tpiba);
            for (int ir = 0; ir < nrxx; ++ir)
                v(mu, ir) -= dh[ir];
            double sum_mu = 0;
            for (int ir = 0; ir < nrxx; ++ir)
                sum_mu += dh[ir] * chr->rho[mu][ir];
            vtxcgc -= sum_mu;
        }

        etxc += etxcgc;
        vtxc += vtxcgc;
    }

#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    etxc *= omega / rhopw->nxyz;
    vtxc *= omega / rhopw->nxyz;

    ModuleBase::timer::tick("XC_Functional", "v_xc_ncgga_sf_builtin");
    return std::make_tuple(etxc, vtxc, std::move(v));
}

void gradcorr_ncgga_sf_builtin(const Charge* const chr, ModulePW::PW_Basis* rhopw,
                                const UnitCell* ucell, std::vector<double>& stress_gga)
{
    stress_gga.assign(9, 0.0);
}

} // namespace NCGGA_SF_Builtin
} // namespace ModuleXC
