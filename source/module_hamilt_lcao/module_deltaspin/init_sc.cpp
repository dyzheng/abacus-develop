#include "spin_constrain.h"

// init sc
template <typename FPTYPE>
void SpinConstrain<FPTYPE>::init_sc(double sc_thr_in,
                                            int nsc_in,
                                            int nsc_min_in,
                                            double alpha_trial_in,
                                            double sccut_in,
                                            bool decay_grad_switch_in,
                                            const UnitCell& ucell,
                                            std::string sc_file,
                                            int NPOL,
                                            Parallel_Orbitals* ParaV_in,
                                            int nspin_in,
                                            K_Vectors& kv_in,
                                            std::string KS_SOLVER_in,
                                            void* phsol_in,
                                            void* p_hamilt_in,
                                            void* psi_in,
                                            elecstate::ElecState* pelec_in)
{
    this->set_input_parameters(sc_thr_in, nsc_in, nsc_min_in, alpha_trial_in, sccut_in, decay_grad_switch_in);
    this->set_atomCounts(ucell.get_atom_Counts());
    this->set_orbitalCounts(ucell.get_orbital_Counts());
    this->set_lnchiCounts(ucell.get_lnchi_Counts());
    this->set_nspin(nspin_in);
    this->set_target_mag(ucell.get_target_mag());
    this->lambda_ = ucell.get_lambda();
    this->constrain_ = ucell.get_constrain();
    this->atomLabels_ = ucell.get_atomLabels();
    this->set_decay_grad();
    this->set_npol(NPOL);
    if(ParaV_in != nullptr) this->set_ParaV(ParaV_in);
    this->set_solver_parameters(kv_in, phsol_in, p_hamilt_in, psi_in, pelec_in, KS_SOLVER_in);
}

template class SpinConstrain<std::complex<double>>;
template class SpinConstrain<double>;