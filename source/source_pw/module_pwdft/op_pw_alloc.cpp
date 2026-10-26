#include "op_pw_alloc.h"

#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"
#include "source_base/global_variable.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_constraint/alloc_pw.h"
#include "source_estate/module_constraint/constraint_io.h"
#include "source_estate/module_constraint/constraint_loop.h"
#include "source_estate/module_constraint/weight_grid.h"
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_pwdft/onsite_proj.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace alloc
{
namespace
{

struct PwTargetContext
{
    int npw = 0;
    PwTargetSpace space;
    std::vector<std::vector<std::complex<double>>> phi;
};

void reduce_pool(std::complex<double>& value)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(value);
#endif
}

void reduce_pool(std::vector<double>& values)
{
#ifdef __MPI
    Parallel_Reduce::reduce_pool(values.data(), static_cast<int>(values.size()));
#endif
}

std::vector<std::complex<double>> apply_local_potential(
    const ModulePW::PW_Basis_K& pw,
    const int ik,
    const std::vector<double>& potential,
    const std::complex<double>* input,
    const int npw)
{
    std::vector<std::complex<double>> work(pw.nmaxgr,
                                           std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> porter(pw.nmaxgr,
                                             std::complex<double>(0.0, 0.0));
    for (int ig = 0; ig < npw; ++ig)
    {
        work[ig] = input[ig];
    }
    pw.template recip_to_real<std::complex<double>, base_device::DEVICE_CPU>(
        work.data(), porter.data(), ik);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        porter[ir] *= potential[ir];
    }
    pw.template real_to_recip<std::complex<double>, base_device::DEVICE_CPU>(
        porter.data(), work.data(), ik);
    std::vector<std::complex<double>> result(npw,
                                             std::complex<double>(0.0, 0.0));
    for (int ig = 0; ig < npw; ++ig)
    {
        result[ig] = work[ig];
    }
    return result;
}

bool build_target_context(
    projectors::OnsiteProjector<double, base_device::DEVICE_CPU>* onsite_p,
    const constraint::WeightGrid& weight_grid,
    const int ik,
    const int target_l,
    PwTargetContext& context,
    std::string& error)
{
    // Branch: allocation needs an initialized onsite-projector table and a
    // selected angular-momentum shell.  Both are configuration, not runtime
    // numerical failures, so report a precise reason.
    if (onsite_p == nullptr || onsite_p->get_tot_nproj() <= 0)
    {
        error = "PW allocation requires onsite projectors "
                "(onsite_radius > 0 and NUMERICAL_ORBITAL)";
        return false;
    }
    onsite_p->tabulate_atomic(ik);
    const std::complex<double>* tab = onsite_p->get_tab_atomic();
    const int npw = onsite_p->get_npw();
    // Branch: a missing tabulation leaves no plane-wave target vectors.
    if (tab == nullptr || npw <= 0)
    {
        error = "PW allocation could not tabulate onsite projectors";
        return false;
    }

    const int nat = static_cast<int>(weight_grid.nat());
    std::vector<int> nh_per_atom(nat, 0);
    for (int iat = 0; iat < nat; ++iat)
    {
        nh_per_atom[iat] = onsite_p->get_nh(iat);
    }
    const PwTargetShell selection
        = select_pw_target_shell(nh_per_atom, target_l);
    // Branch: an empty shell would silently reduce the allocation operator to
    // the legacy local weight, which is exactly what S5 must replace.
    if (selection.row.empty())
    {
        error = "PW allocation selected zero target projectors";
        return false;
    }

    const int ntarget = static_cast<int>(selection.row.size());
    const int nconstraint = weight_grid.nconstraint();
    const auto& constraint_atoms = weight_grid.constraint_atoms();
    std::vector<int> target_constraint(ntarget, -1);
    for (int a = 0; a < ntarget; ++a)
    {
        int owner = -1;
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            // Branch: each target atom must belong to at most one constraint;
            // a duplicate owner makes Q_i ambiguous.
            if (std::find(constraint_atoms[alpha].begin(),
                          constraint_atoms[alpha].end(),
                          selection.atom[a])
                != constraint_atoms[alpha].end())
            {
                if (owner >= 0)
                {
                    error = "PW target atom belongs to more than one constraint";
                    return false;
                }
                owner = alpha;
            }
        }
        target_constraint[a] = owner;
    }

    std::vector<std::complex<double>> compact_tab(
        static_cast<size_t>(ntarget) * npw);
    context.phi.assign(
        ntarget,
        std::vector<std::complex<double>>(npw,
                                          std::complex<double>(0.0, 0.0)));
    for (int a = 0; a < ntarget; ++a)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            const std::complex<double> value
                = tab[static_cast<size_t>(selection.row[a]) * npw + ig];
            compact_tab[static_cast<size_t>(a) * npw + ig] = value;
            context.phi[a][ig] = value;
        }
    }
    std::vector<double> ST;
    build_pw_projector_overlap(compact_tab.data(), npw, ntarget, ST);
    build_pw_target(ST, ntarget, selection.atom, target_constraint,
                    nconstraint, context.space);
    // Branch: a zero-rank projector shell has no complement subspace to
    // constrain; this is a rank/configuration failure, not a no-op.
    if (context.space.rank == 0)
    {
        error = "PW allocation target projector has zero numerical rank";
        return false;
    }
    context.npw = npw;
    return true;
}

bool build_mu_potential(const constraint::WeightGrid& weight_grid,
                        const constraint::ConstraintLoop& loop,
                        const int spin,
                        std::vector<double>& potential,
                        std::vector<double>& lambda,
                        std::string& error)
{
    const int nconstraint = weight_grid.nconstraint();
    const std::vector<double>& mu = loop.mu_eff();
    const std::vector<constraint::ChannelProfile>& channels = loop.channels();
    // Guard: the multiplier, channel and weight lists must stay parallel.
    if (static_cast<int>(mu.size()) != nconstraint
        || static_cast<int>(channels.size()) != nconstraint)
    {
        error = "PW allocation multiplier/constraint count mismatch";
        return false;
    }

    potential.assign(weight_grid.nrxx(), 0.0);
    lambda.assign(nconstraint, 0.0);
    const auto& weights = weight_grid.constraint_weights();
    for (int alpha = 0; alpha < nconstraint; ++alpha)
    {
        // Branch A: spin-up (or the single nspin=1 channel) uses inj_up.
        // Branch B: spin-down uses inj_dn, which is the opposite sign for a
        // spin constraint and the same sign for a charge constraint.
        const double sign = (spin == 0) ? channels[alpha].inj_up
                                       : channels[alpha].inj_dn;
        const double coefficient = mu[alpha] * sign;
        lambda[alpha] = coefficient;
        for (int ir = 0; ir < weight_grid.nrxx(); ++ir)
        {
            potential[ir] += coefficient * weights[alpha][ir];
        }
    }
    return true;
}

bool build_target_products(
    const ModulePW::PW_Basis_K& pw,
    const int ik,
    const std::vector<double>& potential,
    const PwTargetContext& context,
    std::vector<std::vector<std::complex<double>>>& vphi,
    std::vector<std::complex<double>>& m_raw,
    std::string& error)
{
    const int r = context.space.ntarget;
    vphi.assign(r,
                std::vector<std::complex<double>>(context.npw,
                                                  std::complex<double>(0.0, 0.0)));
    m_raw.assign(static_cast<size_t>(r) * r,
                 std::complex<double>(0.0, 0.0));
    for (int a = 0; a < r; ++a)
    {
        vphi[a] = apply_local_potential(pw, ik, potential,
                                        context.phi[a].data(), context.npw);
    }
    for (int a = 0; a < r; ++a)
    {
        for (int b = 0; b < r; ++b)
        {
            std::complex<double> value(0.0, 0.0);
            for (int ig = 0; ig < context.npw; ++ig)
            {
                value += std::conj(context.phi[a][ig]) * vphi[b][ig];
            }
            reduce_pool(value);
            m_raw[static_cast<size_t>(a) * r + b] = value;
        }
    }
    (void)error;
    return true;
}

void build_raw_overlap(const PwTargetContext& context,
                       const std::complex<double>* psi,
                       std::vector<std::complex<double>>& raw)
{
    raw.assign(context.space.ntarget, std::complex<double>(0.0, 0.0));
    for (int a = 0; a < context.space.ntarget; ++a)
    {
        std::complex<double> value(0.0, 0.0);
        for (int ig = 0; ig < context.npw; ++ig)
        {
            value += std::conj(context.phi[a][ig]) * psi[ig];
        }
        reduce_pool(value);
        raw[a] = value;
    }
}

} // namespace

bool compute_pw_alloc_observables(
    const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi,
    const ModuleBase::matrix& occupations,
    const K_Vectors& kv,
    const ModulePW::PW_Basis_K& wfcpw,
    const int nspin,
    const constraint::WeightGrid& weight_grid,
    const int target_l,
    std::vector<double>& n_alloc,
    std::vector<double>& m_alloc,
    std::string& error)
{
    const int nconstraint = weight_grid.nconstraint();
    n_alloc.assign(nconstraint, 0.0);
    m_alloc.assign(nconstraint, 0.0);
    // Branch: the observable is defined for the collinear scalar-band path.
    if (nspin != 1 && nspin != 2)
    {
        error = "PW allocation observable supports only nspin 1/2";
        return false;
    }
    // Branch: the spinor band layout has two components per band and needs a
    // separate Pauli contraction, which this collinear closure does not form.
    if (psi == nullptr || psi->get_npol() != 1)
    {
        error = "PW allocation observable requires scalar CPU wavefunctions";
        return false;
    }
    const int nks = psi->get_nk();
    // Branch: no k-point means no occupied wavefunction to project.
    if (nks <= 0)
    {
        error = "PW allocation observable has no k-points";
        return false;
    }
    auto* onsite_p
        = projectors::OnsiteProjector<double,
                                      base_device::DEVICE_CPU>::get_instance();
    for (int ik = 0; ik < nks; ++ik)
    {
        // Branch: every k-point has its own spin label and projector basis.
        if (static_cast<int>(kv.isk.size()) <= ik)
        {
            error = "PW allocation observable has no k-point spin label";
            return false;
        }
        psi->fix_k(ik);
        PwTargetContext context;
        if (!build_target_context(onsite_p, weight_grid, ik, target_l,
                                  context, error))
        {
            return false;
        }
        const int nband = psi->get_nbands();
        const int ld_psi = psi->get_nbasis();
        const int spin_sign = (kv.isk[ik] == 1) ? -1 : 1;
        const std::complex<double>* psi_k = psi->get_pointer();
        for (int alpha = 0; alpha < nconstraint; ++alpha)
        {
            const std::vector<double>& potential
                = weight_grid.constraint_weight(alpha);
            std::vector<std::vector<std::complex<double>>> vphi;
            std::vector<std::complex<double>> m_raw;
            if (!build_target_products(wfcpw, ik, potential,
                                       context, vphi, m_raw, error))
            {
                return false;
            }
            std::vector<double> lambda(nconstraint, 0.0);
            lambda[alpha] = 1.0;
            for (int band = 0; band < nband; ++band)
            {
                const double occupation = occupations(ik, band);
                // Branch: an empty band contributes nothing to Tr(D K_i).
                if (occupation == 0.0)
                {
                    continue;
                }
                const std::complex<double>* band_ptr
                    = psi_k + static_cast<size_t>(band) * ld_psi;
                std::vector<std::complex<double>> raw;
                build_raw_overlap(context, band_ptr, raw);
                const std::vector<std::complex<double>> vpsi
                    = apply_local_potential(wfcpw, ik, potential,
                                            band_ptr, context.npw);
                std::vector<std::complex<double>> delta;
                apply_pw_alloc_delta_h(context.space, context.phi, 
                                        std::vector<std::complex<double>>(
                                            band_ptr,
                                            band_ptr + context.npw),
                                        raw, vpsi, vphi, m_raw, lambda, delta);
                std::complex<double> trace(0.0, 0.0);
                for (int ig = 0; ig < context.npw; ++ig)
                {
                    trace += std::conj(band_ptr[ig]) * delta[ig];
                }
                reduce_pool(trace);
                n_alloc[alpha] += occupation * trace.real();
                // Branch: only a two-channel collinear run has a spin
                // difference observable.
                if (nspin == 2)
                {
                    m_alloc[alpha] += spin_sign * occupation * trace.real();
                }
            }
        }
    }
    // Every rank in a pool owns the same local k list after the pool
    // reductions above.  Average the identical copies and sum the disjoint
    // k-point sets of different pools.
    Parallel_Reduce::reduce_double_allpool(
        PARAM.inp.kpar, GlobalV::NPROC_IN_POOL, n_alloc.data(), nconstraint);
    Parallel_Reduce::reduce_double_allpool(
        PARAM.inp.kpar, GlobalV::NPROC_IN_POOL, m_alloc.data(), nconstraint);
    (void)ucell;
    return true;
}

} // namespace alloc

namespace hamilt
{

template <typename T, typename Device>
AllocConstraintPW<OperatorPW<T, Device>>::AllocConstraintPW(
    const int* isk_in,
    const UnitCell* ucell_in,
    const ModulePW::PW_Basis_K* wfcpw_in)
    : isk(isk_in), ucell(ucell_in), wfcpw(wfcpw_in)
{
    this->classname = "AllocConstraintPW";
    this->cal_type = calculation_type::pw_onsite;
}

template <typename T, typename Device>
template <typename T_in, typename Device_in>
AllocConstraintPW<OperatorPW<T, Device>>::AllocConstraintPW(
    const AllocConstraintPW<OperatorPW<T_in, Device_in>>* op)
    : isk(op->get_isk()),
      ucell(op->get_ucell()),
      wfcpw(op->get_wfcpw())
{
    this->classname = "AllocConstraintPW";
    this->cal_type = calculation_type::pw_onsite;
}

template <typename T, typename Device>
void AllocConstraintPW<OperatorPW<T, Device>>::init(const int ik_in)
{
    // The allocation operator reads the same onsite tabulation as DFT+U and
    // DeltaSpin.  Refresh it here because this operator can be the only PW
    // onsite consumer in an allocation-only run.
    auto* onsite_p
        = projectors::OnsiteProjector<double, Device>::get_instance();
    // Branch: the projector singleton exists even before init; only a
    // tabulated instance has rows to refresh.
    if (onsite_p != nullptr && onsite_p->get_tot_nproj() > 0)
    {
        onsite_p->tabulate_atomic(ik_in);
    }
    OperatorPW<T, Device>::init(ik_in);
}

template <typename T, typename Device>
void AllocConstraintPW<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node) const
{
    (void)is_first_node;
    if constexpr (std::is_same<T, std::complex<double>>::value
                  && std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        // Branch: this operator is inert unless the allocation switch is on.
        if (!PARAM.inp.constraint || !PARAM.inp.constraint_alloc_switch)
        {
            return;
        }
        constraint::ConstraintLoop& loop
            = constraint::ConstraintLoop::instance();
        // Branch: before the loop is armed (or in the legacy grid mode) the
        // normal Hamiltonian must remain untouched.
        if (!loop.enabled() || !loop.allocation_mode())
        {
            return;
        }
        // Guard: the scalar PW allocation algebra has one component per band.
        if (npol != 1)
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
                "PW allocation currently supports only collinear npol = 1");
        }
        // Guard: dipole weights are a global coordinate, not per-atom target
        // ownership; the allocation backend is not defined for them.
        if (loop.dipole_constraint())
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
                "PW allocation does not support dipole constraints");
        }
        // Branch: an empty band range is a valid no-op.
        if (nbands <= 0)
        {
            return;
        }
        const int spin = (PARAM.inp.nspin == 2) ? this->isk[this->ik] : 0;
        // Guard: an unexpected spin label would silently apply the wrong
        // channel sign to a spin constraint.
        if (spin != 0 && spin != 1)
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
                "PW allocation received an invalid spin label");
        }
        const constraint::WeightGrid& wg = loop.weight_grid();
        // Guard: the FFT potential and Wavefunction grids must coincide; the
        // constraint configuration already rejects double_grid.
        if (wg.nrxx() != this->wfcpw->nrxx)
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
                "PW allocation weight and wavefunction grids differ");
        }

        std::vector<double> potential;
        std::vector<double> lambda;
        std::string error;
        if (!alloc::build_mu_potential(wg, loop, spin, potential, lambda,
                                       error))
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
        }
        auto* onsite_p
            = projectors::OnsiteProjector<double,
                                          base_device::DEVICE_CPU>::get_instance();
        alloc::PwTargetContext context;
        if (!alloc::build_target_context(onsite_p, wg, this->ik,
                                         PARAM.inp.constraint_alloc_target_l,
                                         context, error))
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
        }
        const int npw = ngk_ik > 0 ? ngk_ik : context.npw;
        // Guard: the operator chain and onsite tabulation must describe one
        // plane-wave partition.
        if (npw != context.npw || npw > nbasis)
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act",
                "PW allocation plane-wave partition mismatch");
        }
        std::vector<std::vector<std::complex<double>>> vphi;
        std::vector<std::complex<double>> m_raw;
        if (!alloc::build_target_products(*this->wfcpw, this->ik, potential,
                                          context, vphi, m_raw, error))
        {
            ModuleBase::WARNING_QUIT("AllocConstraintPW::act", error);
        }

        for (int ib = 0; ib < nbands; ++ib)
        {
            const std::complex<double>* psi_band
                = reinterpret_cast<const std::complex<double>*>(tmpsi_in)
                  + static_cast<size_t>(ib) * nbasis;
            std::complex<double>* hpsi_band
                = reinterpret_cast<std::complex<double>*>(tmhpsi)
                  + static_cast<size_t>(ib) * nbasis;
            std::vector<std::complex<double>> raw;
            alloc::build_raw_overlap(context, psi_band, raw);
            const std::vector<std::complex<double>> vpsi
                = alloc::apply_local_potential(*this->wfcpw, this->ik,
                                               potential, psi_band, npw);
            std::vector<std::complex<double>> delta;
            alloc::apply_pw_alloc_delta_h(
                context.space, context.phi,
                std::vector<std::complex<double>>(psi_band, psi_band + npw),
                raw, vpsi, vphi, m_raw, lambda, delta);
            for (int ig = 0; ig < npw; ++ig)
            {
                hpsi_band[ig] += delta[ig];
            }
        }
    }
}

template class AllocConstraintPW<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class AllocConstraintPW<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;

#if ((defined __CUDA) || (defined __ROCM))
template class AllocConstraintPW<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class AllocConstraintPW<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
#endif

} // namespace hamilt
