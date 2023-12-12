#ifndef HAMILT_MIXING_H
#define HAMILT_MIXING_H
#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/module_mixing/mixing.h"
#include "module_base/module_mixing/plain_mixing.h"
namespace hamilt
{
template <typename T>
class HamiltMixing
{
  public:
    HamiltMixing();
    ~HamiltMixing();
    Base_Mixing::Mixing* mixing = nullptr; ///< Mixing object to mix Hamiltonian
    Base_Mixing::Mixing_Data hamilt_mdata;    ///< Mixing data for Hamiltonian

    /**
     * @brief reset mixing
     *
     */
    void mix_reset();

    /**
     * @brief hamilt mixing
     *
     */
    void mix_hamilt(T* hamilt_data);

    /**
     * @brief charge mixing for real space
     *
     */
    void mix_hamilt_real(T* hamilt_data, int size);

    /**
     * @brief Inner product of two double vectors
     *
     */
    double inner_product_real(double* rho1, double* rho2);

    /**
     * @brief Set the mixing object
     *
     * @param mixing_mode_in mixing mode: "plain", "broyden", "pulay"
     * @param mixing_beta_in mixing beta
     * @param mixing_ndim_in mixing ndim
     * @param data_length_in data length
     */
    void set_mixing(const std::string& mixing_mode_in,
                    const double& mixing_beta_in,
                    const int& mixing_ndim_in,
                    const int& data_length_in); 

    /**
     * @brief Get the delta of Hamiltonian
     *
     */
    double get_dhamilt(const T* hamilt_new);

    void save_data(const T* data_in);

    // extracting parameters
    // normally these parameters will not be used
    // outside charge mixing, but Exx is using them
    // as well as some other places
    const std::string& get_mixing_mode() const
    {
        return mixing_mode;
    }
    double get_mixing_beta() const
    {
        return mixing_beta;
    }
    int get_mixing_ndim() const
    {
        return mixing_ndim;
    }
    double get_mixing_gg0() const
    {
        return mixing_gg0;
    }

  private:
    //======================================
    // General parameters
    //======================================
    std::string mixing_mode = "broyden"; ///< mixing mode: "plain", "broyden", "pulay"
    double mixing_beta = 0.8;            ///< mixing beta for density
    double mixing_beta_mag = 1.6;        ///< mixing beta for magnetism
    int mixing_ndim = 8;                 ///< mixing ndim for broyden and pulay
    double mixing_gg0 = 0.0;             ///< mixing gg0 for Kerker screen
    bool mixing_tau = false;             ///< whether to use tau mixing

    int data_length = 0;
    std::vector<T> data_save;

};

}  // namespace hamilt

#endif
