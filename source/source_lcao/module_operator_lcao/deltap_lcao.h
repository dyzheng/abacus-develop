#ifndef DELTAP_LCAO_H
#define DELTAP_LCAO_H

#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/unitcell.h"
#include "source_lcao/module_operator_lcao/operator_lcao.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include <unordered_map>
#include <complex>
#include <vector>

namespace hamilt
{

template <typename TK, typename TR>
class DeltaPOperator : public OperatorLCAO<TK, TR>
{
  public:
    DeltaPOperator(HS_Matrix_K<TK>* hsk_in,
                   const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                   hamilt::HContainer<TR>* hR_in,
                   const UnitCell& ucell_in,
                   const Grid_Driver* gridD_in,
                   const TwoCenterIntegrator* intor,
                   const std::vector<double>& orb_cutoff,
                   double rm);

    ~DeltaPOperator();

    virtual void contributeHR() override;

    virtual void contributeHk(int ik) override;

    void update_lambda() { this->dp_hr_done = false; }

    void set_lambda(const std::vector<double>& lambda_in)
    {
        this->lambda_ = lambda_in;
        this->dp_hr_done = false;
    }

    const std::vector<double>& get_lambda() const { return this->lambda_; }
    const std::vector<double>& get_lambda_save() const { return this->lambda_save_; }

    void set_gdir(int gdir) { this->gdir_ = gdir; }

    void set_hk_correction(const std::unordered_map<int, std::vector<std::complex<double>>>& correction)
    {
        hk_correction_ = correction;
    }

    /**
     * @brief Static storage of per-atom lambda for force/stress computation.
     *
     * Set by ESolver_KS_LCAO before calling getForceStress.
     * Accessed by FORCE_STRESS.cpp via DeltaPOperator::get_stored_lambda().
     */
    static void store_lambda_for_force(const std::vector<double>& lam) { s_stored_lambda = lam; }
    static const std::vector<double>& get_stored_lambda() { return s_stored_lambda; }

    /**
     * @brief Static storage of the H_HK (Berry-connection) analytic force
     *        contribution, computed by deltap::DeltaP::compute_hk_force in
     *        ESolver_KS_LCAO::cal_force.  Added to the total force inside
     *        FORCE_STRESS so that the printed TOTAL-FORCE includes it.
     */
    static void store_hk_force_for_force(const std::vector<double>& f_hk, double e_hk)
    {
        s_stored_hk_force = f_hk;
        s_stored_e_hk = e_hk;
    }
    static const std::vector<double>& get_stored_hk_force() { return s_stored_hk_force; }
    static double get_stored_e_hk() { return s_stored_e_hk; }

    /**
     * @brief Compute force and stress from the DeltaP constraint Hamiltonian.
     *
     * Follows the same pattern as DeltaSpin::cal_force_stress().
     * Uses intor_->snap(cal_deri=1) for projector derivatives.
     *
     * @param cal_force  Compute forces if true
     * @param cal_stress Compute stresses if true
     * @param dmR        Density matrix in HContainer format
     * @param force      Output force [nat][3]
     * @param stress     Output stress [3][3] (Voigt order)
     */
    void cal_force_stress(const bool cal_force,
                          const bool cal_stress,
                          const HContainer<double>* dmR,
                          ModuleBase::matrix& force,
                          ModuleBase::matrix& stress);

  private:
    const UnitCell* ucell = nullptr;
    const Grid_Driver* gridD = nullptr;
    const Parallel_Orbitals* paraV = nullptr;
    hamilt::HContainer<TR>* HR = nullptr;
    const TwoCenterIntegrator* intor_ = nullptr;
    std::vector<double> orb_cutoff_;
    double rm_ = 3.0;
    int gdir_ = 3;

    std::vector<hamilt::HContainer<TR>*> pre_hr;

    std::vector<double> lambda_;
    std::vector<double> lambda_save_;
    static std::vector<double> s_stored_lambda;  // for force/stress access
    static std::vector<double> s_stored_hk_force;  // H_HK force (Ry/Bohr), nat*3
    static double s_stored_e_hk;                   // E_HK (Ry) for diagnostics
    bool initialized = false;
    bool dp_hr_done = false;

    // k-dependent HK correction for Berry connection operator
    std::unordered_map<int, std::vector<std::complex<double>>> hk_correction_;

    void cal_pre_HR();
    void cal_HR_IJR(const int& iat1,
                    const int& iat2,
                    const std::unordered_map<int, std::vector<double>>& nlm1_all,
                    const std::unordered_map<int, std::vector<double>>& nlm2_all,
                    TR* data_pointer);

    void cal_force_IJR(const int& iat1,
                       const int& iat2,
                       const Parallel_Orbitals* paraV,
                       const std::unordered_map<int, std::vector<double>>& nlm1_all,
                       const std::unordered_map<int, std::vector<double>>& nlm2_all,
                       const hamilt::BaseMatrix<double>* dmR_pointer,
                       double lambda,
                       double* force1,
                       double* force2,
                       double* p_hat = nullptr);

    void cal_stress_IJR(const int& iat1,
                        const int& iat2,
                        const int* r_vector,
                        const Parallel_Orbitals* paraV,
                        const std::unordered_map<int, std::vector<double>>& nlm1_all,
                        const std::unordered_map<int, std::vector<double>>& nlm2_all,
                        const hamilt::BaseMatrix<double>* dmR_pointer,
                        double lambda,
                        int gdir,
                        double* stress);
};

} // namespace hamilt

#endif
