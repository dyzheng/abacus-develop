// Mock implementations for DensityMatrix tests
// This provides a minimal DensityMatrix implementation for unit testing

#include "source_estate/module_dm/density_matrix.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include <vector>
#include <complex>

namespace elecstate {

// Constructor for multi-k calculation
template <typename TK, typename TR>
DensityMatrix<TK, TR>::DensityMatrix(const Parallel_Orbitals* _paraV,
                                      const int nspin,
                                      const std::vector<ModuleBase::Vector3<double>>& kvec_d,
                                      const int nk)
    : _paraV(_paraV), _nspin(nspin), _kvec_d(kvec_d), _nk(nk > 0 ? nk : static_cast<int>(kvec_d.size()))
{
    // Allocate DMK storage
    int nrow = _paraV ? _paraV->nrow : 0;
    int ncol = _paraV ? _paraV->ncol : 0;
    int dmk_size = nrow * ncol;

    _DMK.resize(_nspin * _nk);
    for (auto& dmk : _DMK) {
        dmk.resize(dmk_size, TK(0.0));
    }
}

// Constructor for gamma-only calculation
template <typename TK, typename TR>
DensityMatrix<TK, TR>::DensityMatrix(const Parallel_Orbitals* _paraV, const int nspin)
    : _paraV(_paraV), _nspin(nspin), _nk(1)
{
    int nrow = _paraV ? _paraV->nrow : 0;
    int ncol = _paraV ? _paraV->ncol : 0;
    int dmk_size = nrow * ncol;

    _DMK.resize(_nspin);
    for (auto& dmk : _DMK) {
        dmk.resize(dmk_size, TK(0.0));
    }
}

template <typename TK, typename TR>
DensityMatrix<TK, TR>::~DensityMatrix()
{
    for (auto* dmr : _DMR) {
        delete dmr;
    }
    _DMR.clear();
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::set_DMK(const int ispin, const int ik, const int i, const int j, const TK value)
{
    int index = (ispin - 1) * _nk + ik;
    if (index >= 0 && index < static_cast<int>(_DMK.size())) {
        int nrow = _paraV ? _paraV->nrow : 0;
        int idx = j * nrow + i;  // Column-major
        if (idx >= 0 && idx < static_cast<int>(_DMK[index].size())) {
            _DMK[index][idx] = value;
        }
    }
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::set_DMK_zero()
{
    for (auto& dmk : _DMK) {
        std::fill(dmk.begin(), dmk.end(), TK(0.0));
    }
}

template <typename TK, typename TR>
TK DensityMatrix<TK, TR>::get_DMK(const int ispin, const int ik, const int i, const int j) const
{
    int index = (ispin - 1) * _nk + ik;
    if (index >= 0 && index < static_cast<int>(_DMK.size())) {
        int nrow = _paraV ? _paraV->nrow : 0;
        int idx = j * nrow + i;
        if (idx >= 0 && idx < static_cast<int>(_DMK[index].size())) {
            return _DMK[index][idx];
        }
    }
    return TK(0.0);
}

template <typename TK, typename TR>
int DensityMatrix<TK, TR>::get_DMK_nks() const
{
    return _nspin * _nk;
}

template <typename TK, typename TR>
int DensityMatrix<TK, TR>::get_DMK_size() const
{
    return _DMK.empty() ? 0 : static_cast<int>(_DMK[0].size());
}

template <typename TK, typename TR>
int DensityMatrix<TK, TR>::get_DMK_nrow() const
{
    return _paraV ? _paraV->nrow : 0;
}

template <typename TK, typename TR>
int DensityMatrix<TK, TR>::get_DMK_ncol() const
{
    return _paraV ? _paraV->ncol : 0;
}

template <typename TK, typename TR>
TK* DensityMatrix<TK, TR>::get_DMK_pointer(const int ik) const
{
    if (ik >= 0 && ik < static_cast<int>(_DMK.size())) {
        return const_cast<TK*>(_DMK[ik].data());
    }
    return nullptr;
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::set_DMK_pointer(const int ik, TK* DMK_in)
{
    if (ik >= 0 && ik < static_cast<int>(_DMK.size()) && DMK_in != nullptr) {
        int size = static_cast<int>(_DMK[ik].size());
        for (int i = 0; i < size; ++i) {
            _DMK[ik][i] = DMK_in[i];
        }
    }
}

template <typename TK, typename TR>
hamilt::HContainer<TR>* DensityMatrix<TK, TR>::get_DMR_pointer(const int ispin) const
{
    if (ispin >= 0 && ispin < static_cast<int>(_DMR.size())) {
        return _DMR[ispin];
    }
    return nullptr;
}

// Stub implementations for methods not needed in tests
template <typename TK, typename TR>
void DensityMatrix<TK, TR>::init_DMR(const Grid_Driver* GridD_in, const UnitCell* ucell)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::init_DMR(Record_adj& ra, const UnitCell* ucell)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::init_DMR(const hamilt::HContainer<TR>& _DMR_in)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::init_DMR(const hamilt::HContainer<typename ShiftRealComplex<TR>::type>& _DMR_in)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::cal_DMR(const int ik_in)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::cal_DMR_td(const UnitCell& ucell, const ModuleBase::Vector3<double> At, const int ik_in)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::cal_DMR_full(hamilt::HContainer<std::complex<double>>* dmR_out, const int ik_in) const
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::switch_dmr(const int mode)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::write_DMK(const std::string directory, const int ispin, const int ik)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::read_DMK(const std::string directory, const int ispin, const int ik)
{
    // Empty stub
}

template <typename TK, typename TR>
void DensityMatrix<TK, TR>::save_DMR()
{
    // Empty stub
}

// Explicit template instantiations
template class DensityMatrix<double, double>;
template class DensityMatrix<std::complex<double>, double>;
template class DensityMatrix<std::complex<double>, std::complex<double>>;

} // namespace elecstate
