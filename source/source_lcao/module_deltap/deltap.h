#ifndef DELTAP_H
#define DELTAP_H

#include "source_base/vector3.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"
#include "source_psi/psi.h"

#include <complex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class cal_r_overlap_R;
class unkOverlap_lcao;

namespace deltap {

struct OverlapData {
    int iat_adj = -1;
    ModuleBase::Vector3<int> R_index;
    std::unordered_map<int, std::vector<double>> nlm;
};

struct KSpaceData {
    ModuleBase::Vector3<double> kvec_d;
    // S_k[iat][lm][mu_local]
    std::vector<std::vector<std::vector<std::complex<double>>>> S_k;
    // dS_k[iat][alpha][lm][mu_local]  (alpha: 0=x, 1=y, 2=z)
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> dS_k;
    // D_I[iat][lm][nband]
    std::vector<std::vector<std::vector<std::complex<double>>>> D_I;
};

struct AtomicPolarization {
    std::vector<ModuleBase::Vector3<double>> P_I;
    std::vector<ModuleBase::Vector3<double>> gamma_I;
    std::vector<double> smo_weight_sum;
    std::vector<ModuleBase::Vector3<double>> gamma_I_raw;
    std::vector<ModuleBase::Vector3<double>> r_elec_center;
    std::vector<std::vector<double>> smo_weights;  // w_In: [n_occ][nat]
    ModuleBase::Vector3<double> P_total;
    ModuleBase::Vector3<double> P_abacus;
};

class DeltaP {
public:
    DeltaP() = default;
    ~DeltaP() = default;

    void init(const UnitCell& ucell,
              const Grid_Driver& gd,
              const K_Vectors& kv,
              const TwoCenterIntegrator* intor,
              const TwoCenterIntegrator* overlap_intor,
              const TwoCenterIntegrator* onsite_onsite_intor,
              const std::vector<double>& orb_cutoff,
              double rm,
              int gdir,
              const Parallel_Orbitals* paraV,
              cal_r_overlap_R* r_overlap = nullptr,
              unkOverlap_lcao* berry_overlap = nullptr);

    void compute_atomic_polarization(
        const UnitCell& ucell,
        const psi::Psi<std::complex<double>>* psi,
        const elecstate::ElecState* pelec);

    void compute_atomic_polarization(
        const UnitCell& ucell,
        const psi::Psi<double>* psi,
        const elecstate::ElecState* pelec)
    {
        throw std::logic_error("DeltaP decomposition supports only multi-k");
    }

    const AtomicPolarization& get_results() const { return results_; }

    /// Lightweight Wilson loop for SCF inner loop (public for esolver access).
    void compute_gamma_scf(const UnitCell& ucell,
                           const psi::Psi<std::complex<double>>* psi,
                           const elecstate::ElecState* pelec);

    /// Compute k-dependent HK correction for constrained DFT (Berry connection operator).
    /// M(k_j) = (i/2) * S(k_j,k_{j+1}) * C(k_{j+1}) * W_eff(k_j) * C†(k_j)
    /// Serial only for now (nrow == ncol).
    void compute_hk_correction(const UnitCell& ucell,
                               const psi::Psi<std::complex<double>>* psi,
                               const std::vector<double>& lambda,
                               std::unordered_map<int, std::vector<std::complex<double>>>& hk_correction);

    /// Load branch state from file (public for esolver access).
    void load_branch();

private:
    void compute_real_overlaps(const UnitCell& ucell, const Grid_Driver& gd);
    void setup_kstring(const K_Vectors& kv);
    void compute_S_k(int ik);
    void compute_S_dk(const UnitCell& ucell);
    void compute_S_dk_link(const UnitCell& ucell,
                           const ModuleBase::Vector3<double>& kvec_d_R,
                           const ModuleBase::Vector3<double>& kvec_c_L,
                           const ModuleBase::Vector3<double>& kvec_c_R);
    void compute_D_I(int ik, const std::complex<double>* psi_k, int nbands, int nrow_local);
    void compute_berry_connection(int ik, const std::complex<double>* psi_k,
                                   int nbands, int nrow_local, const double* wg);
    void integrate_polarization(const UnitCell& ucell, int nbands);
    void compute_smo_overlap_matrix(const UnitCell& ucell);
    int M_a_for_snap(const UnitCell& ucell, int T, int L, int m_idx) const;
    void compute_resta_z(const UnitCell& ucell,
                         const psi::Psi<std::complex<double>>* psi,
                         const elecstate::ElecState* pelec);
    void gauge_fix_smo_anchored(int nbands);
    void compute_wannier_polarization(const UnitCell& ucell,
                                      const psi::Psi<std::complex<double>>* psi,
                                      const elecstate::ElecState* pelec);

    void verify_sum_rule();
    void write_results(const UnitCell& ucell) const;
    void save_branch() const;

    /// Select the 2π branch nearest to a reference value.
    /// Given principal value γ^I_0 = Σ_n w^I_n·arg(λ_n) and weights w^I_n,
    /// searches the set {γ^I_0 + 2π·w^I·k : k ∈ Z^N_occ} for the element
    /// closest to gamma_prev.  Returns the selected value and the integer
    /// shift vector k (for diagnostics).
    double select_branch_set(
        const std::vector<double>& weights,   // w^I_n  [nbands]
        const std::vector<double>& arg_evals, // arg(λ_n) [nbands]
        int nbands,
        double gamma_prev,
        std::vector<int>& k_selected) const;  // output: k_n [nbands]

    // Configuration
    const TwoCenterIntegrator* intor_ = nullptr;
    const TwoCenterIntegrator* overlap_intor_ = nullptr;
    const TwoCenterIntegrator* onsite_onsite_intor_ = nullptr;  // <phi_onsite|phi_onsite>
    cal_r_overlap_R* r_overlap_ = nullptr;  // for <phi|r|phi(R)> position matrix
    unkOverlap_lcao* berry_overlap_ = nullptr;  // for berry_phase-exact overlap matrix
    std::vector<double> orb_cutoff_;
    double rm_ = 3.0;
    int gdir_ = 3;
    int nat_ = 0;
    int nproj_max_ = 0;

    // S(dk) local block for the exact Wilson-loop overlap O = C^dagger(k_j) * S(dk) * C(k_{j+1})
    std::vector<std::complex<double>> S_dk_;
    int S_dk_nrow_ = 0;
    int S_dk_ncol_ = 0;

    // Cache for per-link S_dk computation: stores raw overlap data
    // so that only the phase needs to be recomputed per link
    struct S_dk_cache_entry {
        int lr, lc;
        double ov;
        double Rx, Ry, Rz;
        double tau_x, tau_y, tau_z;
        // Orbital info for get_psi_r_psi
        ModuleBase::Vector3<double> R1_cart;
        int T1, L1, m1, N1;
        ModuleBase::Vector3<double> R2_cart;
        int T2, L2, m2, N2;
        // Cached position matrix (local part only, in Bohr)
        double r_local_x = 0, r_local_y = 0, r_local_z = 0;
        bool r_computed = false;
    };
    std::vector<S_dk_cache_entry> S_dk_cache_;
    bool S_dk_cache_valid_ = false;

    // Branch tracking for cross-SCF phase smoothness
    std::vector<std::complex<double>> W_prev_;
    bool has_prev_ = false;

    // SCF mode: skip file I/O during inner loop iterations
    bool scf_mode_ = false;
    bool scf_initialized_ = false;

    // Branch-set selection diagnostics: per-atom principal value, selected
    // value, and integer shift vector k_n that was applied.
    std::vector<double> gamma_principal_;       // γ^I_0 before selection
    std::vector<double> gamma_selected_;        // γ^I after selection
    std::vector<std::vector<int>> branch_k_;    // k_n[iat][n] shift applied

    // Infrastructure pointers
    const Parallel_Orbitals* paraV_ = nullptr;
    const Grid_Driver* gd_ = nullptr;
    const K_Vectors* kv_ = nullptr;

    // Real-space overlaps: overlap_R_[iat][adj_index]
    std::vector<std::vector<OverlapData>> overlap_R_;
    std::vector<int> nproj_per_atom_;

    // SMO overlap matrix S_{ab} = <alpha_a | alpha_b> and its inverse
    std::vector<double> smo_overlap_;     // m_dim × m_dim
    std::vector<double> smo_overlap_inv_; // m_dim × m_dim
    int smo_m_dim_ = 0;

    // k-string data
    std::vector<KSpaceData> kstring_data_;
    int nppstr_ = 0;
    int total_string_ = 0;
    std::vector<std::vector<int>> k_index_;

    // Berry connection: A_nk_[iat][ik][nband][3] (alpha=x,y,z)
    std::vector<std::vector<std::vector<ModuleBase::Vector3<std::complex<double>>>>> A_nk_;

    // Gauge fixing data (SMO-anchored gauge, Method 5)
    std::vector<std::vector<std::complex<double>>> gauge_phase_;  // [ik][n]
    std::vector<int> anchor_iat_;                                 // [n]
    std::vector<int> anchor_lm_;                                  // [n]
    std::vector<std::complex<double>> phase_corrections_;         // [n]
    bool gauge_enabled_ = false;
    double anchor_thr_ = 1e-8;

    // Results
    AtomicPolarization results_;
};

} // namespace deltap

#endif
