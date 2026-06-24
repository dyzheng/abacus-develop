#ifndef DELTAP_H
#define DELTAP_H

#include "source_base/vector3.h"
#include <complex>
#include <unordered_map>
#include <vector>

namespace deltap {

struct OverlapData {
    int iat_adj = -1;
    ModuleBase::Vector3<int> R_index;
    std::unordered_map<int, std::vector<double>> nlm;
};

struct KSpaceData {
    ModuleBase::Vector3<double> kvec_d;
    std::vector<std::vector<std::vector<std::complex<double>>>> S_k;
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> dS_k;
    std::vector<std::vector<std::vector<std::complex<double>>>> D_I;
};

struct AtomicPolarization {
    std::vector<ModuleBase::Vector3<double>> P_I;
    std::vector<ModuleBase::Vector3<double>> gamma_I;
    ModuleBase::Vector3<double> P_total;
    ModuleBase::Vector3<double> P_abacus;
};

class DeltaP {
public:
    DeltaP() = default;
    ~DeltaP() = default;
};

} // namespace deltap

#endif
