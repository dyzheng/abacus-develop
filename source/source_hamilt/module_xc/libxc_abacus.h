#ifndef LIBXC_ABACUS_H
#define LIBXC_ABACUS_H

#ifdef __LIBXC

#include "source_base/matrix.h"
#include "source_base/vector3.h"
#include "xc_ncgga_radial.h"

#include <xc.h>
#include <xc_funcs.h>

#include <tuple>
#include <vector>

#include <map>
#include <utility>

class Charge;

namespace XC_Functional_Libxc
{
    // Complete forward data for the gga_grad=2 noncollinear Libxc graph:
    //   rho_s = N_s(x),
    //   g_s   = G_h N_s(x).
    // Reverse the FFT gradient before applying the local map Jacobian;
    // this uses the same clipping branches as the forward energy.
    struct NclSfDiscreteData
    {
        std::vector<ModuleXC::NcggaSpinMapPoint> spin_map;
        std::vector<double> rho;
        std::vector<std::vector<ModuleBase::Vector3<double>>> spin_gradient;
    };

//-------------------
//  libxc_setup.cpp
//-------------------

    // sets functional type, which allows combination of LIBXC keyword connected by "+"
    //        for example: "XC_LDA_X+XC_LDA_C_PZ"
    extern std::pair<int, std::vector<int>> set_xc_type_libxc(const std::string& xc_func_in);

    extern std::vector<xc_func_type> init_func(
        const std::vector<int> &func_id,
        const int xc_polarized,
        const double hybrid_alpha = 0.0,
        const double hse_omega = 0.0);

    extern void finish_func(std::vector<xc_func_type> &funcs);


//-------------------
//  libxc_pot.cpp
//-------------------

    extern std::tuple<double, double, ModuleBase::matrix> v_xc_libxc(
        const std::vector<int> &func_id,
        const int &nrxx, // number of real-space grid
        const double &omega, // volume of cell
        const double tpiba,
        const Charge* const chr, // charge density
        const int nspin,
        const bool domag,
        const bool domag_z,
        const std::map<int, double>* scaling_factor = nullptr,
        const double hybrid_alpha = 0.0,
        const double hse_omega = 0.0);

    // Gradient-metric stress of the gga_grad=2 noncollinear Libxc graph.
    extern void gradcorr_ncgga_sf_libxc(
        const std::vector<int>& func_id,
        const std::size_t nrxx,
        const double tpiba,
        const Charge* const chr,
        const std::map<int, double>* scaling_factor,
        const double hybrid_alpha,
        const double hse_omega,
        std::vector<double>& stress_gga);

    // for mGGA functional
    extern std::tuple<double, double, ModuleBase::matrix, ModuleBase::matrix> v_xc_meta(
        const std::vector<int> &func_id,
        const int &nrxx, // number of real-space grid
        const double &omega, // volume of cell
        const double tpiba,
        const Charge* const chr,
        const int nspin,
        const double hybrid_alpha = 0.0,
        const double hse_omega = 0.0);


//-------------------
//  libxc_tools.cpp
//-------------------

    // converting rho (abacus=>libxc)
    extern std::vector<double> convert_rho(
        const int nspin,
        const std::size_t nrxx,
        const Charge* const chr);

    // converting rho (abacus=>libxc)
    extern std::tuple<std::vector<double>, std::vector<double>> convert_rho_amag_nspin4(
        const int nspin,
        const std::size_t nrxx,
        const Charge* const chr);

    // calculating grho
    extern std::vector<std::vector<ModuleBase::Vector3<double>>> cal_gdr(
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &rho,
        const double tpiba,
        const Charge* const chr);

    // converting grho (abacus=>libxc)
    extern std::vector<double> convert_sigma(
        const std::vector<std::vector<ModuleBase::Vector3<double>>> &gdr);

    // sgn for threshold mask
    extern std::vector<double> cal_sgn(
        const double rho_threshold,
        const double grho_threshold,
        const xc_func_type &func,
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &rho,
        const std::vector<double> &sigma);

    // threshold masks for the xc potential (Quantum ESPRESSO convention):
    // the first mask applies to exc and vrho, the second one only to vsigma
    extern std::pair<std::vector<double>, std::vector<double>> cal_sgn_vxc(
        const double rho_threshold_vrho,
        const double rho_threshold_vsigma,
        const double grho_threshold_vsigma,
        const xc_func_type &func,
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &rho,
        const std::vector<double> &sigma);

    // converting etxc from exc (libxc=>abacus)
    extern double convert_etxc(
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &sgn,
        const std::vector<double> &rho,
        std::vector<double> exc);

    // converting vtxc and v from vrho and vsigma (libxc=>abacus)
    extern std::pair<double, ModuleBase::matrix> convert_vtxc_v(
        const xc_func_type &func,
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &sgn_vrho,
        const std::vector<double> &sgn_vsigma,
        const std::vector<double> &rho,
        const std::vector<std::vector<ModuleBase::Vector3<double>>> &gdr,
        const std::vector<double> &vrho,
        const std::vector<double> &vsigma,
        const double tpiba,
        const Charge* const chr);

    // dh for gga v
    extern std::vector<std::vector<double>> cal_dh(
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &sgn,
        const std::vector<std::vector<ModuleBase::Vector3<double>>> &gdr,
        const std::vector<double> &vsigma,
        const double tpiba,
        const Charge* const chr);

    // convert v for NSPIN=4
    extern ModuleBase::matrix convert_v_nspin4(
        const std::size_t nrxx,
        const Charge* const chr,
        const std::vector<double> &amag,
        const ModuleBase::matrix &v);

    extern std::vector<double> compute_mag_part_nspin4(
        const std::size_t nrxx,
        const Charge* const chr);

    extern std::vector<std::vector<ModuleBase::Vector3<double>>> cal_gdr_sf(
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &rho,
        const std::vector<double> &mag_part,
        const double tpiba,
        const Charge* const chr);

    extern std::vector<std::vector<double>> cal_dh_sf(
        const int nspin,
        const std::size_t nrxx,
        const std::vector<double> &sgn,
        const std::vector<std::vector<ModuleBase::Vector3<double>>> &gdr,
        const std::vector<double> &vsigma,
        const std::vector<double> &mag_part,
        const double tpiba,
        const Charge* const chr);

    extern ModuleBase::matrix convert_v_nspin4_sf(
        const std::size_t nrxx,
        const Charge* const chr,
        const std::vector<double> &mag_part,
        const ModuleBase::matrix &v);

    // Build the gga_grad=2 local spin map and, when requested, its
    // mapped-density FFT-gradient graph. LDA-only callers set need_gradient=false.
    extern NclSfDiscreteData make_ncl_sf_discrete_data(
        const std::size_t nrxx,
        const double tpiba,
        const Charge* const chr,
        const bool need_gradient);

    // Reverse the mapped-density FFT-gradient graph: apply the adjoint of the
    // gradient operator before the local map Jacobian.
    extern ModuleBase::matrix reverse_ncl_sf_discrete(
        const std::size_t nrxx,
        const NclSfDiscreteData& data,
        const std::vector<double>& drho,
        const std::vector<double>& dsigma,
        const double tpiba,
        const Charge* const chr);


//-------------------
//  libxc_lda_wrap.cpp
//-------------------

    extern void xc_spin_libxc(
        const std::vector<int> &func_id,
        const double &rhoup,
        const double &rhodw,
        double &exc,
        double &vxcup,
        double &vxcdw,
        const double hybrid_alpha = 0.0,
        const double hse_omega = 0.0);


//-------------------
//  libxc_gga_wrap.cpp
//-------------------

    // the entire GGA functional, for nspin=1 case
    extern void gcxc_libxc(
        const std::vector<int> &func_id,
        const double &rho,
        const double &grho,
        double &sxc,
        double &v1xc,
        double &v2xc,
        const double hybrid_alpha = 0.0,
        const double hse_omega = 0.0);

    // Overload accepting an already-initialized functional vector. The caller
    // is responsible for init_func/finish_func; this overload does not touch
    // the lifetime of funcs. Useful for per-thread reuse inside OpenMP loops.
    extern void gcxc_libxc(
        const std::vector<xc_func_type>& funcs,
        const double &rho,
        const double &grho,
        double &sxc,
        double &v1xc,
        double &v2xc);

    // the entire GGA functional, for nspin=2 case
    extern void gcxc_spin_libxc(
        const std::vector<int> &func_id,
        const double rhoup,
        const double rhodw,
        const ModuleBase::Vector3<double> gdr1,
        const ModuleBase::Vector3<double> gdr2,
        double &sxc,
        double &v1xcup,
        double &v1xcdw,
        double &v2xcup,
        double &v2xcdw,
        double &v2xcud,
        const double hybrid_alpha = 0.0,
        const double hse_omega = 0.0);

    // Overload accepting an already-initialized functional vector.
    extern void gcxc_spin_libxc(
        const std::vector<xc_func_type>& funcs,
        const double rhoup,
        const double rhodw,
        const ModuleBase::Vector3<double> gdr1,
        const ModuleBase::Vector3<double> gdr2,
        double &sxc,
        double &v1xcup,
        double &v1xcdw,
        double &v2xcup,
        double &v2xcdw,
        double &v2xcud);


//-------------------
//  libxc_mgga_wrap.cpp
//-------------------

    // wrapper for the mGGA functionals
    extern void tau_xc(
        const std::vector<int> &func_id,
        const double &rho,
        const double &grho,
        const double &atau,
        double &sxc,
        double &v1xc,
        double &v2xc,
        double &v3xc,
        const double &hybrid_alpha = 0.0,
        const double &hse_omega = 0.0);

    // Overload accepting an already-initialized functional vector.
    // hybrid_alpha scales the semilocal SCAN exchange for SCAN0.
    extern void tau_xc(
        const std::vector<xc_func_type>& funcs,
        const double &rho,
        const double &grho,
        const double &atau,
        double &sxc,
        double &v1xc,
        double &v2xc,
        double &v3xc,
        const double hybrid_alpha);

    extern void tau_xc_spin(
        const std::vector<int> &func_id,
        double rhoup,
        double rhodw,
        ModuleBase::Vector3<double> gdr1,
        ModuleBase::Vector3<double> gdr2,
        double tauup,
        double taudw,
        double &sxc,
        double &v1xcup,
        double &v1xcdw,
        double &v2xcup,
        double &v2xcdw,
        double &v2xcud,
        double &v3xcup,
        double &v3xcdw,
        const double &hybrid_alpha = 0.0,
        const double &hse_omega = 0.0);

    // Overload accepting an already-initialized functional vector.
    // hybrid_alpha scales the semilocal SCAN exchange for SCAN0.
    extern void tau_xc_spin(
        const std::vector<xc_func_type>& funcs,
        double rhoup,
        double rhodw,
        ModuleBase::Vector3<double> gdr1,
        ModuleBase::Vector3<double> gdr2,
        double tauup,
        double taudw,
        double &sxc,
        double &v1xcup,
        double &v1xcdw,
        double &v2xcup,
        double &v2xcdw,
        double &v2xcud,
        double &v3xcup,
        double &v3xcdw,
        const double hybrid_alpha);

} // namespace XC_Functional_Libxc

#endif // __LIBXC

#endif // LIBXC_ABACUS_H
