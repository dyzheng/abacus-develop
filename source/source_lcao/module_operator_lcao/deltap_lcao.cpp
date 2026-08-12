#include "deltap_lcao.h"
#include "source_base/timer.h"
#include "source_base/memory.h"
#include "source_base/tool_title.h"
#include "source_base/parallel_reduce.h"
#include "source_io/module_parameter/parameter.h"
#include <fstream>
#include <type_traits>

// Static storage for force/stress computation
template <typename TK, typename TR>
std::vector<double> hamilt::DeltaPOperator<TK, TR>::s_stored_lambda;
template <typename TK, typename TR>
std::vector<double> hamilt::DeltaPOperator<TK, TR>::s_stored_hk_force;
template <typename TK, typename TR>
double hamilt::DeltaPOperator<TK, TR>::s_stored_e_hk = 0.0;

template <typename TK, typename TR>
hamilt::DeltaPOperator<TK, TR>::DeltaPOperator(
    HS_Matrix_K<TK>* hsk_in,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
    hamilt::HContainer<TR>* hR_in,
    const UnitCell& ucell_in,
    const Grid_Driver* gridD_in,
    const TwoCenterIntegrator* intor,
    const std::vector<double>& orb_cutoff,
    double rm)
    : hamilt::OperatorLCAO<TK, TR>(hsk_in, kvec_d_in, hR_in),
      intor_(intor), orb_cutoff_(orb_cutoff), rm_(rm)
{
    this->cal_type = calculation_type::lcao_dp_lambda;
    this->ucell = &ucell_in;
    this->gridD = gridD_in;
    // Branch A: SCF path (hR != null) — cache the parallel orbital layout for
    // cal_pre_HR()/contributeHR().
    // Branch B: force/stress path (hR == null, see FORCE_STRESS.cpp) — paraV is
    // not needed here; cal_force_stress() takes it from the density matrix.
    // Guarding the dereference fixes a null-pointer crash in relax+cal_force.
    this->paraV = this->hR ? this->hR->get_paraV() : nullptr;
    // Per-atom λ init: if deltap_lambda_init_file is set, read one value per
    // atom (used by the FD protocol to freeze the base-run converged λ for
    // every displaced geometry; group-1 of T7-b).  Otherwise fall back to the
    // scalar deltap_lambda_init for all atoms.
    if (!PARAM.inp.deltap_lambda_init_file.empty())
    {
        std::ifstream ifs(PARAM.inp.deltap_lambda_init_file);
        if (!ifs)
        {
            ModuleBase::WARNING_QUIT("DeltaPOperator",
                "cannot open deltap_lambda_init_file: "
                + PARAM.inp.deltap_lambda_init_file);
        }
        this->lambda_.resize(this->ucell->nat);
        this->lambda_save_.resize(this->ucell->nat);
        for (int iat = 0; iat < this->ucell->nat; ++iat)
        {
            if (!(ifs >> this->lambda_[iat]))
            {
                ModuleBase::WARNING_QUIT("DeltaPOperator",
                    "deltap_lambda_init_file has fewer values than nat");
            }
        }
        this->lambda_save_ = this->lambda_;
    }
    else
    {
        this->lambda_.assign(this->ucell->nat, PARAM.inp.deltap_lambda_init);
        this->lambda_save_.assign(this->ucell->nat, PARAM.inp.deltap_lambda_init);
    }
    this->pre_hr.resize(this->ucell->nat, nullptr);
}

template <typename TK, typename TR>
hamilt::DeltaPOperator<TK, TR>::~DeltaPOperator()
{
    for (auto& hr : this->pre_hr)
    {
        if (hr != nullptr) { delete hr; hr = nullptr; }
    }
}

template <typename TK, typename TR>
void hamilt::DeltaPOperator<TK, TR>::contributeHR()
{
    // Branch A: first time ever — lambda_save_ already equals lambda_
    // from the constructor, so dλ = 0 (no HR perturbation at iter=1).
    // This avoids destabilizing the initial SCF convergence.
    // Previously (B16-era), lambda_save_ was reset to 0 here, causing
    // dλ = λ_init at iter=1, which created a large Hamiltonian
    // perturbation that prevented gamma convergence.
    // HR is applied only when set_lambda() changes λ, which resets
    // dp_hr_done → the else-if below is skipped → dλ computed.
    if (!this->hr_done)
    {
        // hR is being rebuilt from scratch (e.g., new ionic step).
        if (this->initialized)
        {
            // Not the first time — reset lambda_save_ to zero so the FULL
            // current lambda_ is re-added to the newly zeroed hR.
            // (The cumulative incremental model relied on hR persistence.)
            this->lambda_save_.assign(this->ucell->nat, 0.0);
        }
        // Else: first time ever — lambda_save_ already equals lambda_
        // (both were set to lambda_init in the constructor), so dλ = 0.
    }
    else if (this->dp_hr_done)
    {
        return;
    }

    // T-6' (Ô_w): in the exact weight-channel operator mode
    // (deltap_operator_mode = "ow"), the τ_α·P̂ geometric proxy H_HR is
    // REPLACED by the per-k H_ow = Σ_n θ_n·(P̂_λ·C)·C† built in
    // DeltaP::compute_hk_correction (contributeHk).  Adding the real-space
    // H_HR here as well would double-count the on-site projector term.
    // Legacy gamma mode and the historical "proxy" operator mode are
    // unchanged (zero regression).
    if (PARAM.inp.deltap_operator_mode == "ow"
        && PARAM.inp.deltap_observable == "operator")
    {
        // Keep the λ bookkeeping consistent (the H_ow path is the sole
        // operator contribution), then hand the k-space part to contributeHk.
        this->dp_hr_done = true;
        for (int iat = 0; iat < this->ucell->nat; iat++)
        {
            this->lambda_save_[iat] = this->lambda_[iat];
        }
        return;
    }

    if (!this->initialized)
    {
        this->cal_pre_HR();
        this->initialized = true;
    }

    const int alpha_idx = this->gdir_ - 1;

    for (int iat = 0; iat < this->ucell->nat; iat++)
    {
        if (this->pre_hr[iat] == nullptr) continue;

        int I0, T0;
        this->ucell->iat2iait(iat, &I0, &T0);
        // B-6 (fixed): H_HR = Σ λ·τ_α·P̂ with τ_α the Direct (fractional)
        // coordinate taud — must match the force side (deltap_force_stress)
        // and the E-field equivalent E=−πλ/(2a).  lat0-unit tau amplified
        // the operator by L = a/lat0 (~15.87 here).
        double tau_alpha = this->ucell->atoms[T0].taud[I0][alpha_idx];

        double dlambda = this->lambda_[iat] - this->lambda_save_[iat];
        double coeff = dlambda * tau_alpha;

        if (std::abs(coeff) < 1e-15) continue;

        for (int iap = 0; iap < this->pre_hr[iat]->size_atom_pairs(); iap++)
        {
            hamilt::AtomPair<TR>& tmp = this->pre_hr[iat]->get_atom_pair(iap);
            int iat1 = tmp.get_atom_i();
            int iat2 = tmp.get_atom_j();
            for (int ir = 0; ir < tmp.get_R_size(); ++ir)
            {
                const ModuleBase::Vector3<int> r_index = tmp.get_R_index(ir);
                const TR* pre_hr_data = tmp.get_pointer(ir);
                hamilt::BaseMatrix<TR>* dmat = this->hR->find_matrix(iat1, iat2, r_index[0], r_index[1], r_index[2]);
                if (dmat == nullptr) continue;
                TR* dhr_data = dmat->get_pointer();
                for (int i = 0; i < tmp.get_size(); i++)
                {
                    dhr_data[i] += pre_hr_data[i] * coeff;
                }
            }
        }

        this->lambda_save_[iat] = this->lambda_[iat];
    }

    this->dp_hr_done = true;
}

template <typename TK, typename TR>
void hamilt::DeltaPOperator<TK, TR>::cal_pre_HR()
{
    ModuleBase::TITLE("DeltaPOperator", "cal_pre_HR");
    ModuleBase::timer::start("DeltaPOperator", "cal_pre_HR");

    const int npol = this->ucell->get_npol();

    for (int iat = 0; iat < this->ucell->nat; iat++)
    {
        auto tau0 = this->ucell->get_tau(iat);
        int T0, I0;
        this->ucell->iat2iait(iat, &I0, &T0);

        AdjacentAtomInfo adjs;
        this->gridD->Find_atom(*this->ucell, tau0, T0, I0, &adjs);

        std::vector<bool> is_adj(adjs.adj_num + 1, false);
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = this->ucell->itia2iat(T1, I1);
            const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad];
            if (this->ucell->cal_dtau(iat, iat1, R_index1).norm() * this->ucell->lat0
                < this->orb_cutoff_[T1] + this->rm_)
            {
                is_adj[ad] = true;
            }
        }
        filter_adjs(is_adj, adjs);

        this->pre_hr[iat] = new hamilt::HContainer<TR>(this->paraV);
        for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
        {
            const int T1 = adjs.ntype[ad1];
            const int I1 = adjs.natom[ad1];
            const int iat1 = this->ucell->itia2iat(T1, I1);
            const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad1];
            for (int ad2 = 0; ad2 < adjs.adj_num + 1; ++ad2)
            {
                const int T2 = adjs.ntype[ad2];
                const int I2 = adjs.natom[ad2];
                const int iat2 = this->ucell->itia2iat(T2, I2);
                ModuleBase::Vector3<int>& R_index2 = adjs.box[ad2];
                int r_vector[3] = {R_index2.x - R_index1.x, R_index2.y - R_index1.y, R_index2.z - R_index1.z};
                if (this->hR->find_matrix(iat1, iat2, r_vector[0], r_vector[1], r_vector[2]) == nullptr) continue;
                hamilt::AtomPair<TR> tmp(iat1, iat2, r_vector[0], r_vector[1], r_vector[2], this->paraV);
                this->pre_hr[iat]->insert_pair(tmp);
            }
        }
        this->pre_hr[iat]->allocate(nullptr, true);

        const int max_l_plus_1 = this->ucell->atoms[T0].nwl + 1;
        std::vector<std::unordered_map<int, std::vector<double>>> nlm_iat0(adjs.adj_num + 1);

        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = this->ucell->itia2iat(T1, I1);
            const Atom* atom1 = &this->ucell->atoms[T1];
            const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];

            auto all_indexes = this->paraV->get_indexes_row(iat1);
            auto col_indexes = this->paraV->get_indexes_col(iat1);
            all_indexes.insert(all_indexes.end(), col_indexes.begin(), col_indexes.end());
            std::sort(all_indexes.begin(), all_indexes.end());
            all_indexes.erase(std::unique(all_indexes.begin(), all_indexes.end()), all_indexes.end());

            for (int iw1l = 0; iw1l < (int)all_indexes.size(); iw1l += npol)
            {
                const int iw1 = all_indexes[iw1l] / npol;
                std::vector<double> nlm_target(max_l_plus_1 * max_l_plus_1);
                const int L1 = atom1->iw2l[iw1];
                const int N1 = atom1->iw2n[iw1];
                const int m1 = atom1->iw2m[iw1];

                std::vector<std::vector<double>> nlm;
                const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                ModuleBase::Vector3<double> dtau = tau0 - tau1;
                this->intor_->snap(T1, L1, N1, M1, T0, dtau * this->ucell->lat0, 0, nlm);

                int target_L = 0, index = 0;
                for (int iw = 0; iw < this->ucell->atoms[T0].nw; iw++)
                {
                    const int L0 = this->ucell->atoms[T0].iw2l[iw];
                    if (L0 == target_L)
                    {
                        for (int m = 0; m < 2 * L0 + 1; m++)
                        {
                            nlm_target[index] = nlm[0][iw + m];
                            index++;
                        }
                        target_L++;
                    }
                }
                nlm_iat0[ad].insert({all_indexes[iw1l], nlm_target});
            }
        }

        for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
        {
            const int T1 = adjs.ntype[ad1];
            const int I1 = adjs.natom[ad1];
            const int iat1 = this->ucell->itia2iat(T1, I1);
            const std::unordered_map<int, std::vector<double>>& nlm1 = nlm_iat0[ad1];
            for (int ad2 = 0; ad2 < adjs.adj_num + 1; ++ad2)
            {
                const int T2 = adjs.ntype[ad2];
                const int I2 = adjs.natom[ad2];
                const int iat2 = this->ucell->itia2iat(T2, I2);
                const std::unordered_map<int, std::vector<double>>& nlm2 = nlm_iat0[ad2];
                ModuleBase::Vector3<int> R_vector(adjs.box[ad2][0] - adjs.box[ad1][0],
                                                  adjs.box[ad2][1] - adjs.box[ad1][1],
                                                  adjs.box[ad2][2] - adjs.box[ad1][2]);
                hamilt::BaseMatrix<TR>* tmp = this->pre_hr[iat]->find_matrix(iat1, iat2, R_vector[0], R_vector[1], R_vector[2]);
                if (tmp != nullptr)
                {
                    this->cal_HR_IJR(iat1, iat2, nlm1, nlm2, tmp->get_pointer());
                }
            }
        }
    }

    ModuleBase::timer::end("DeltaPOperator", "cal_pre_HR");
}

template <typename TK, typename TR>
void hamilt::DeltaPOperator<TK, TR>::cal_HR_IJR(
    const int& iat1,
    const int& iat2,
    const std::unordered_map<int, std::vector<double>>& nlm1_all,
    const std::unordered_map<int, std::vector<double>>& nlm2_all,
    TR* data_pointer)
{
    const int npol = this->ucell->get_npol();
    auto row_indexes = this->paraV->get_indexes_row(iat1);
    auto col_indexes = this->paraV->get_indexes_col(iat2);
    std::vector<int> step_trace(npol * npol, 0);
    for (int is = 0; is < npol; is++)
    {
        for (int is2 = 0; is2 < npol; is2++)
        {
            step_trace[is * npol + is2] = this->paraV->get_col_size(iat2) * is + is2;
        }
    }
    for (int iw1l = 0; iw1l < (int)row_indexes.size(); iw1l += npol)
    {
        const std::vector<double>& nlm1 = nlm1_all.find(row_indexes[iw1l])->second;
        for (int iw2l = 0; iw2l < (int)col_indexes.size(); iw2l += npol)
        {
            const std::vector<double>& nlm2 = nlm2_all.find(col_indexes[iw2l])->second;
            TR nlm_tmp = TR(0);
            for (int m1 = 0; m1 < (int)nlm1.size(); m1++)
            {
                nlm_tmp += nlm1[m1] * nlm2[m1];
            }
            for (int is = 0; is < npol * npol; ++is)
            {
                data_pointer[step_trace[is]] += nlm_tmp;
            }
            data_pointer += npol;
        }
        data_pointer += (npol - 1) * col_indexes.size();
    }
}

namespace {
    template <typename TK>
    inline void add_hk_correction(TK* hk, const std::vector<std::complex<double>>& H_sym) {
        for (size_t i = 0; i < H_sym.size(); ++i) {
            hk[i] += H_sym[i];
        }
    }

    template <>
    inline void add_hk_correction<double>(double* hk, const std::vector<std::complex<double>>& H_sym) {
        for (size_t i = 0; i < H_sym.size(); ++i) {
            hk[i] += H_sym[i].real();
        }
    }
}

template <typename TK, typename TR>
void hamilt::DeltaPOperator<TK, TR>::contributeHk(int ik)
{
    if (hk_correction_.count(ik) == 0) return;

    const auto& H_sym = hk_correction_.at(ik);
    TK* hk = this->hsk->get_hk();

    // Debug: print max correction magnitude
    double max_corr = 0.0;
    for (size_t i = 0; i < H_sym.size(); ++i) {
        max_corr = std::max(max_corr, std::abs(H_sym[i]));
    }
    static int hk_call_count = 0;
    if (hk_call_count < 5) {
        std::cout << "   [DeltaPOp] contributeHk ik=" << ik
                  << " size=" << H_sym.size()
                  << " max|corr|=" << max_corr << std::endl;
        hk_call_count++;
    }

    add_hk_correction<TK>(hk, H_sym);
}

// Include force/stress template implementations
#include "deltap_force_stress.hpp"

template class hamilt::DeltaPOperator<std::complex<double>, double>;
template class hamilt::DeltaPOperator<std::complex<double>, std::complex<double>>;
template class hamilt::DeltaPOperator<double, double>;
