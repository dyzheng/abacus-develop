#include "atom_pair.h"

namespace hamilt
{

//----------------------------------------------------
//atom pair class
//----------------------------------------------------

template<typename T>
AtomPair<T>::AtomPair(
    const int& atom_i_,
    const int& atom_j_,
    const Parallel_Orbitals* paraV_,
    const T* existed_matrix
):atom_i(atom_i_), atom_j(atom_j_), paraV(paraV_) 
{
    assert(this->paraV != nullptr);
    this->row_ap = this->paraV->atom_begin_row[atom_i];
    this->col_ap = this->paraV->atom_begin_col[atom_j];
    if(this->row_ap == -1 || this->col_ap == -1)
    {
        throw std::string("Atom-pair not belong this process");
    }
    this->row_size = this->paraV->get_row_size(atom_i);
    this->col_size = this->paraV->get_col_size(atom_j);
    this->ldc = this->paraV->get_col_size();
    this->R_index.resize(3, 0);
    this->current_R = 0;
    if(existed_matrix != nullptr)
    {
        BaseMatrix<T> tmp(row_size, col_size, (existed_matrix + row_ap * ldc + col_ap));
        this->values.push_back(tmp);
    }
    else
    {
        BaseMatrix<T> tmp(row_size, col_size);
        this->values.push_back(tmp);
        this->ldc = col_size;
    }
}

template<typename T>
AtomPair<T>::AtomPair(
    const int& atom_i,
    const int& atom_j,
    const int& rx,
    const int& ry,
    const int& rz,
    const Parallel_Orbitals* ParaV_,
    const T* existed_matrix
):atom_i(atom_i_), atom_j(atom_j_), paraV(paraV_)
{
    assert(this->paraV != nullptr);
    this->row_ap = this->paraV->atom_begin_row[atom_i];
    this->col_ap = this->paraV->atom_begin_col[atom_j];
    if(this->row_ap == -1 || this->col_ap == -1)
    {
        throw std::string("Atom-pair not belong this process");
    }
    this->row_size = this->paraV->get_row_size(atom_i);
    this->col_size = this->paraV->get_col_size(atom_j);
    this->ldc = this->paraV->get_col_size();
    this->R_index.resize(3, 0);
    this->current_R = 0;
    this->R_index[0] = rx;
    this->R_index[1] = ry;
    this->R_index[2] = rz;
    if(existed_matrix != nullptr)
    {
        BaseMatrix<T> tmp(row_size, col_size, (existed_matrix + row_ap * ldc + col_ap));
        this->values.push_back(tmp);
    }
    else
    {
        BaseMatrix<T> tmp(row_size, col_size);
        this->values.push_back(tmp);
        this->ldc = col_size;
    }
}
//direct save whole matrix of atom-pair
template<typename T>
AtomPair<T>::AtomPair(
    const int& atom_i_,
    const int& atom_j_,
    const int* row_atom_begin,
    const int* col_atom_begin,
    const int& natom,
    const T* existed_matrix
):atom_i(atom_i_), atom_j(atom_j_)
{
    assert(row_atom_begin != nullptr && col_atom_begin != nullptr);
    this->row_ap = row_atom_begin[atom_i];
    this->col_ap = col_atom_begin[atom_j];
    this->row_size = row_atom_begin[atom_i+1] - row_atom_begin[atom_i];
    this->col_size = col_atom_begin[atom_j+1] - col_atom_begin[atom_j];
    this->R_index.resize(3, 0);
    this->current_R = 0;
    if(existed_matrix != nullptr)
    {
        this->ldc = row_atom_begin[natom] - row_atom_begin[0];
        BaseMatrix<T> tmp(row_size, col_size, (existed_matrix + row_ap * ldc + col_ap));
        this->values.push_back(tmp);
    }
    else
    {
        BaseMatrix<T> tmp(row_size, col_size);
        this->values.push_back(tmp);
        this->ldc = col_size;
    }
}
//
template<typename T>
AtomPair<T>::AtomPair(
    const int& atom_i,
    const int& atom_j,
    const int& rx,
    const int& ry,
    const int& rz,
    const int* row_atom_begin,
    const int* col_atom_begin,
    const int& natom,
    const T* existed_matrix
):atom_i(atom_i_), atom_j(atom_j_)
{
    assert(row_atom_begin != nullptr && col_atom_begin != nullptr);
    this->row_ap = row_atom_begin[atom_i];
    this->col_ap = col_atom_begin[atom_j];
    this->row_size = row_atom_begin[atom_i+1] - row_atom_begin[atom_i];
    this->col_size = col_atom_begin[atom_j+1] - col_atom_begin[atom_j];
    this->R_index.resize(3, 0);
    this->current_R = 0;
    if(existed_matrix != nullptr)
    {
        this->ldc = row_atom_begin[natom] - row_atom_begin[0];
        BaseMatrix<T> tmp(row_size, col_size, (existed_matrix + row_ap * ldc + col_ap));
        this->values.push_back(tmp);
    }
    else
    {
        BaseMatrix<T> tmp(row_size, col_size);
        this->values.push_back(tmp);
        this->ldc = col_size;
    }
}

template<typename T>
BaseMatrix<T>& AtomPair<T>::get_R_values(int rx_in, int ry_in, int rz_in)const
{
    for (int i = 0; i < R_index.size(); i+=3) {
        if (R_index[i] == rx_in && R_index[i+1] == ry_in && R_index[i+2] == rz_in) {
            current_R = i/3;
            return values[current_R];
        }
    }
    throw std::out_of_range("Invalid cell index");
    //add a new BaseMatrix for this R index
    /*R_index.push_back(rx_in);
    R_index.push_back(ry_in);
    R_index.push_back(rz_in);
    values.emplace_back(BaseMatrix<T>());
    current_R = values.size()-1;
    values.back().num_orb_i = element_i->num_orbitals;
    values.back().num_orb_j = element_j->num_orbitals;
    return values.back();*/
}

template<typename T>
void AtomPair<T>::convert_add(const BaseMatrix<T>& target, int rx_in, int ry_in, int rz_in){
    BaseMatrix<T>& matrix = get_R_values(rx_in, ry_in, rz_in);
    if (matrix.memory_type == 1) {
        matrix.add_array(target.get_pointer());
    }
    else {
        for (int i = 0; i < matrix.num_orb_i; i++) {
            for (int j = 0; j < matrix.num_orb_j; j++) {
                matrix.add_element(i, j, target.get_value(i, j));
            }
        }
    }
}
template<typename T>
void AtomPair<T>::convert_save(const BaseMatrix<T>& target, int rx_in, int ry_in, int rz_in){
    BaseMatrix<T>& matrix = get_R_values(rx_in, ry_in, rz_in);
    if (matrix.memory_type == 1) {
        matrix.save_array(target.get_pointer());
    }
    else {
        for (int i = 0; i < matrix.num_orb_i; i++) {
            for (int j = 0; j < matrix.num_orb_j; j++) {
                matrix.save_element(i, j, target.get_value(i, j));
            }
        }
    }
}

template<typename T>
void AtomPair<T>::add_to_matrix(std::complex<T>* hk, const int ncol_hk, const std::complex<T> &kphase) const{
    /*for (int i = 0; i < R_index.size(); i+=3) {
        int rx = R_index[i];
        int ry = R_index[i+1];
        int rz = R_index[i+2];
        const BaseMatrix<T>& matrix = get_R_values(rx, ry, rz);*/
    assert(ncol_hk == this->ldc);
    const BaseMatrix<T>& matrix = values[current_R];
    std::complex<T>* hk_tmp = hk + this->row_ap * ncol_hk + this->col_ap;
    for (int mu = 0; mu < matrix.num_orb_i; mu++) {
        hk += ncol_hk;
        for (int nu = 0; nu < matrix.num_orb_j; nu++) {
            hk[nu] += matrix.get_value(mu, nu, this->ldc) * kphase;
        }
    }
}

template<typename T>
T& AtomPair<T>::get_matrix_value(const size_t &i_row_global, const size_t &j_col_global) const
{
    size_t i_row_local = this->ParaV->trace_loc_row(i_row_global);
    size_t j_col_local = this->paraV->trace_loc_col(j_col_global);
    //assert(i_row_local != -1 && i_col_local != -1)
    size_t i_row_in = i_row_local - row_ap;
    size_t j_col_in = j_col_local - col_ap;
    return this->values[current_R].get_value(i_row_in, j_col_in, this->ldc);
}

}