#include "deltap.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"

namespace deltap {

void DeltaP::init(const UnitCell& ucell, const Grid_Driver& gd, const K_Vectors& kv,
                  const TwoCenterIntegrator* intor, const std::vector<double>& orb_cutoff,
                  double rm, int gdir)
{
    intor_ = intor;
    orb_cutoff_ = orb_cutoff;
    rm_ = rm;
    gdir_ = gdir;
    nat_ = ucell.nat;
    gd_ = &gd;
    kv_ = &kv;
    ModuleBase::TITLE("DeltaP", "init");
}

void DeltaP::compute_atomic_polarization(const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi, const elecstate::ElecState* pelec) {}

void DeltaP::setup_kstring(const K_Vectors& kv) {}
void DeltaP::compute_S_k(int ik) {}
void DeltaP::compute_D_I(int ik, const std::complex<double>* psi_k, int nbands, int nrow_local) {}
void DeltaP::compute_berry_connection(int ik, const std::complex<double>* psi_k,
                                      int nbands, int nrow_local, const double* wg) {}
void DeltaP::integrate_polarization(const UnitCell& ucell, int nbands) {}
void DeltaP::verify_sum_rule() {}
void DeltaP::write_results(const UnitCell& ucell) const {}

} // namespace deltap
