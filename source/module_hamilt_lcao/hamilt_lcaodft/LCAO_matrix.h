#ifndef LCAO_MATRIX_H
#define LCAO_MATRIX_H

#include "module_base/complexmatrix.h"
#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/vector3.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_HS_arrays.hpp"

// add by jingan for map<> in 2021-12-2, will be deleted in the future
#include "module_base/abfs-vector3_order.h"
#ifdef __EXX
#include <RI/global/Tensor.h>
#endif

class LCAO_Matrix
{
  public:
    LCAO_Matrix();
    ~LCAO_Matrix();

    void divide_HS_in_frag(const bool isGamma, Parallel_Orbitals& pv, const int& nks);

    Parallel_Orbitals* ParaV;

#ifdef __EXX
    using TAC = std::pair<int, std::array<int, 3>>;
    std::vector<std::map<int, std::map<TAC, RI::Tensor<double>>>>* Hexxd;
    std::vector<std::map<int, std::map<TAC, RI::Tensor<std::complex<double>>>>>* Hexxc;
    /// @brief Hexxk for all k-points, only for the 1st scf loop ofrestart load
    std::vector<std::vector<double>> Hexxd_k_load;
    std::vector<std::vector<std::complex<double>>> Hexxc_k_load;
#endif

  public:
    // LiuXh add 2019-07-15
    double**** Hloc_fixedR_tr;
    double**** SlocR_tr;
    double**** HR_tr;

    std::complex<double>**** Hloc_fixedR_tr_soc;
    std::complex<double>**** SlocR_tr_soc;
    std::complex<double>**** HR_tr_soc;

    // jingan add 2021-6-4, modify 2021-12-2
    // Sparse form of HR and SR, the format is [R_direct_coor][orbit_row][orbit_col]

    // For HR_sparse[2], when nspin=1, only 0 is valid, when nspin=2, 0 means spin up, 1 means spin down
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> HR_sparse[2];
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> SR_sparse;
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> TR_sparse;

    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> dHRx_sparse[2];
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> dHRy_sparse[2];
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> dHRz_sparse[2];

    // For nspin = 4
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, std::complex<double>>>> HR_soc_sparse;
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, std::complex<double>>>> SR_soc_sparse;
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, std::complex<double>>>> TR_soc_sparse;

    // Record all R direct coordinate information, even if HR or SR is a zero matrix
    std::set<Abfs::Vector3_Order<int>> all_R_coor;

    // Records the R direct coordinates of HR and SR output, This variable will be filled with data when HR and SR files
    // are output.
    std::set<Abfs::Vector3_Order<int>> output_R_coor;

    template <typename T>
    static void set_mat2d(const int& global_ir, const int& global_ic, const T& v, const Parallel_Orbitals& pv, T* mat);

    void set_HSgamma(const int& iw1_all, const int& iw2_all, const double& v, double* HSloc);

    void set_HR_tr(const int& Rx,
                   const int& Ry,
                   const int& Rz,
                   const int& iw1_all,
                   const int& iw2_all,
                   const double& v);

    void set_HR_tr_soc(const int& Rx,
                       const int& Ry,
                       const int& Rz,
                       const int& iw1_all,
                       const int& iw2_all,
                       const std::complex<double>& v); // LiuXh add 2019-07-16

    void zeros_HSgamma(const char& mtype);

    void zeros_HSk(const char& mtype);

    void update_Hloc(void);

    void update_Hloc2(const int& ik);

    void output_HSk(const char& mtype, std::string& fn);

    // jingan add 2021-6-4, modify 2021-12-2
    void destroy_HS_R_sparse(void);

    void destroy_T_R_sparse(void);

    void destroy_dH_R_sparse(LCAO_HS_Arrays& HS_Arrays);
};

#include "LCAO_matrix.hpp"

#endif
