#include "hamilt_mixing.h"

#include "module_base/element_elec_config.h"
#include "module_base/inverse_matrix.h"
#include "module_base/module_mixing/broyden_mixing.h"
#include "module_base/module_mixing/plain_mixing.h"
#include "module_base/module_mixing/pulay_mixing.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace hamilt
{

template <typename T>
HamiltMixing<T>::HamiltMixing()
{
}

template <typename T>
HamiltMixing<T>::~HamiltMixing()
{
    delete this->mixing;
}

template <typename T>
void HamiltMixing<T>::set_mixing(const std::string& mixing_mode_in,
                               const double& mixing_beta_in,
                               const int& mixing_ndim_in,
                               const int& data_length_in)
{
    this->mixing_mode = mixing_mode_in;
    this->mixing_beta = mixing_beta_in;
    this->mixing_ndim = mixing_ndim_in;
    this->data_length = data_length_in;
    this->data_save.resize(data_length_in);

    GlobalV::ofs_running<<"\n----------- Double Check Mixing Parameters Begin ------------"<<std::endl;
    GlobalV::ofs_running<<"mixing_type: "<< this->mixing_mode <<std::endl;
    GlobalV::ofs_running<<"mixing_beta: "<< this->mixing_beta <<std::endl;
    GlobalV::ofs_running<<"mixing_ndim: "<< this->mixing_ndim <<std::endl;
    GlobalV::ofs_running<<"----------- Double Check Mixing Parameters End ------------"<<std::endl;

    if (this->mixing_mode == "broyden")
    {
        delete this->mixing;
        this->mixing = new Base_Mixing::Broyden_Mixing(this->mixing_ndim, this->mixing_beta);
    }
    else if (this->mixing_mode == "plain")
    {
        delete this->mixing;
        this->mixing = new Base_Mixing::Plain_Mixing(this->mixing_beta);
    }
    else if (this->mixing_mode == "pulay")
    {
        delete this->mixing;
        this->mixing = new Base_Mixing::Pulay_Mixing(this->mixing_ndim, this->mixing_beta);
    }
    else
    {
        ModuleBase::WARNING_QUIT("HamiltMixing", "This Mixing mode is not implemended yet,coming soon.");
    }

    this->mixing->init_mixing_data(this->hamilt_mdata, this->data_length, sizeof(double));
    return;
}

template <typename T>
double HamiltMixing<T>::get_dhamilt(const T* hamilt_new)
{
    ModuleBase::TITLE("HamiltMixing", "get_drho");
    ModuleBase::timer::tick("HamiltMixing", "get_drho");
    double dhamilt = 0.0;

    for (int ir = 0; ir < this->data_length; ir++)
    {
        dhamilt += std::norm(hamilt_new[ir] - this->data_save[ir]);
    }
#ifdef __MPI
    Parallel_Reduce::reduce_pool(dhamilt);
#endif

    ModuleBase::timer::tick("HamiltMixing", "get_dhamilt");
    return dhamilt;
}

template <typename T>
void HamiltMixing<T>::mix_hamilt_real(T* hamilt_data, int size)
{
    T* hamilt_in = this->data_save.data();
    T* hamilt_out = hamilt_data;

    this->mixing->push_data(this->hamilt_mdata, hamilt_in, hamilt_out, nullptr, true);
    
    auto inner_product
        = std::bind(&HamiltMixing::inner_product_real, this, std::placeholders::_1, std::placeholders::_2);
    this->mixing->cal_coef(this->hamilt_mdata, inner_product);
    this->mixing->mix_data(this->hamilt_mdata, hamilt_out);

}

template <typename T>
void HamiltMixing<T>::mix_reset()
{
    this->mixing->reset();
    this->hamilt_mdata.reset();
}

template <typename T>
void HamiltMixing<T>::mix_hamilt(T* hamilt_data)
{
    ModuleBase::TITLE("HamiltMixing", "mix_hamilt");
    ModuleBase::timer::tick("HamiltMixing", "mix_hamilt");

    // the charge before mixing.
    std::vector<T> rho123(this->data_length);
    for(int ir=0; ir<this->data_length; ir++)
    {
        rho123[ir] = hamilt_data[ir];
    }
    
    mix_hamilt_real(hamilt_data, this->data_length);
    // ---------------------------------------------------
    // update this->data_save
    // ---------------------------------------------------
    for(int ir=0; ir<this->data_length; ir++)
    {
        this->data_save[ir] = rho123[ir];
    }
    ModuleBase::timer::tick("HamiltMixing", "mix_hamilt");
    return;
}

template <typename T>
double HamiltMixing<T>::inner_product_real(double* hamilt1, double* hamilt2)
{
    double rnorm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : rnorm)
#endif
    for (int ir = 0; ir < this->data_length; ++ir)
    {
        rnorm += hamilt1[ir] * hamilt2[ir];
    }
#ifdef __MPI
    Parallel_Reduce::reduce_pool(rnorm);
#endif
    return rnorm;
}

template <typename T>
void HamiltMixing<T>::save_data(const T* data_in)
{
    for(int ir=0; ir<this->data_length; ir++)
    {
        this->data_save[ir] = data_in[ir];
    }
}

template class HamiltMixing<double>;
template class HamiltMixing<std::complex<double>>;

} // namespace hamilt