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
AtomPair<T>::AtomPair(
    const int& atom_i_,
    const int& atom_j_
):atom_i(atom_i_), atom_j(atom_j_)
{}

// copy constructor
template<typename T>
AtomPair<T>::AtomPair(const AtomPair<T>& other)
    : R_index(other.R_index), values(other.values), paraV(other.paraV),
        current_R(other.current_R), atom_i(other.atom_i), atom_j(other.atom_j),
        row_ap(other.row_ap), col_ap(other.col_ap), row_size(other.row_size),
        col_size(other.col_size), ldc(other.ldc) 
{}

// The copy assignment operator
template<typename T>
AtomPair<T>& AtomPair<T>::operator=(const AtomPair<T>& other) {
    if (this != &other) {
        R_index = other.R_index;
        values = other.values;
        paraV = other.paraV;
        current_R = other.current_R;
        atom_i = other.atom_i;
        atom_j = other.atom_j;
        row_ap = other.row_ap;
        col_ap = other.col_ap;
        row_size = other.row_size;
        col_size = other.col_size;
        ldc = other.ldc;
    }
    return *this;
}

// move constructor
template<typename T>
AtomPair<T>::AtomPair(AtomPair<T>&& other) noexcept
    : R_index(std::move(other.R_index)), values(std::move(other.values)), paraV(other.paraV),
        current_R(other.current_R), atom_i(other.atom_i), atom_j(other.atom_j),
        row_ap(other.row_ap), col_ap(other.col_ap), row_size(other.row_size),
        col_size(other.col_size), ldc(other.ldc) 
{
    other.paraV = nullptr;
}

// move assignment operator
template<typename T>
AtomPair<T>& AtomPair<T>::operator=(AtomPair<T>&& other) noexcept 
{
    if (this != &other) {
        R_index = std::move(other.R_index);
        values = std::move(other.values);
        paraV = other.paraV;
        other.paraV = nullptr;
        current_R = other.current_R;
        atom_i = other.atom_i;
        atom_j = other.atom_j;
        row_ap = other.row_ap;
        col_ap = other.col_ap;
        row_size = other.row_size;
        col_size = other.col_size;
        ldc = other.ldc;
    }
    return *this;
}

template<typename T>
bool AtomPair<T>::operator<(const AtomPair<T>& other) const
{
    if (atom_i < other.atom_i) {
        return true;
    } else if (atom_i == other.atom_i) {
        return atom_j < other.atom_j;
    } else {
        return false;
    }
}

template<typename T>
BaseMatrix<T>& AtomPair<T>::get_HR_values(int rx_in, int ry_in, int rz_in)const
{
    if(this->current_R > -1)
    {
        //if current_R is not -1, R index has been fixed, just return this->values[current_R]
        return this->values[current_R];
    }
    //if current_R is -1, R index has not been fixed, find existed R index
    for (int i = 0; i < this->R_index.size(); i+=3) 
    {
        if (R_index[i] == rx_in && R_index[i+1] == ry_in && R_index[i+2] == rz_in) 
        {
            return values[i/3];
        }
    }
    // no existed value with this R index found
    // add a new BaseMatrix for this R index
    R_index.push_back(rx_in);
    R_index.push_back(ry_in);
    R_index.push_back(rz_in);
    values.push_back(BaseMatrix<T>(this->row_size, this->col_size));
    return values.back();
}

template<typename T>
void AtomPair<T>::convert_add(const BaseMatrix<T>& target, int rx_in, int ry_in, int rz_in){
    BaseMatrix<T>& matrix = this->get_HR_values(rx_in, ry_in, rz_in);
    //memory type is 2d-block
    //for 2d-block memory type, the data of input matrix is expected storing in a linear array
    //so we can use pointer to access data, and will be separate to 2d-block in this function
    matrix.add_array(target.get_pointer());
}

//function merge
template<typename T>
void AtomPair<T>::merge(const AtomPair<T>& other)
{
    if (other.atom_i != atom_i || other.atom_j != atom_j) {
        throw std::string("AtomPair::merge: atom pair not match");
    }
    for (int i = 0; i < other.R_index.size(); i+=3) {
        int rx = other.R_index[i];
        int ry = other.R_index[i+1];
        int rz = other.R_index[i+2];
        const BaseMatrix<T>& matrix = other.get_HR_values(rx, ry, rz);
        convert_add(matrix, rx, ry, rz);
    }
}

template<typename T>
void AtomPair<T>::add_to_matrix(std::complex<T>* hk, const int ld_hk, const std::complex<T> &kphase, const int hk_type) const{
    /*for (int i = 0; i < R_index.size(); i+=3) {
        int rx = R_index[i];
        int ry = R_index[i+1];
        int rz = R_index[i+2];
        const BaseMatrix<T>& matrix = get_HR_values(rx, ry, rz);*/
    const BaseMatrix<T>& matrix = values[current_R];
    std::complex<T>* hk_tmp = hk + this->row_ap * ld_hk + this->col_ap;
    if(hk_type == 0)
    {
        for (int mu = 0; mu < this->row_size; mu++) 
        {
            for (int nu = 0; nu < this->col_size; nu++) 
            {
                hk_tmp[nu] += matrix.get_value(mu, nu, this->ldc) * kphase;
            }
            hk_tmp += ld_hk;
        }
    }
    else if(hk_type == 1)
    {
        for (int nu = 0; nu < this->col_size; nu++) 
        {
            for (int mu = 0; mu < this->row_size; mu++) 
            {
                hk_tmp[mu] += matrix.get_value(mu, nu, this->ldc) * kphase;
            }
            hk_tmp += ld_hk;
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

//interface for get (rx, ry, rz) of index-th R-index in this->R_index, the return should be int[3]
template<typename T>
int* AtomPair<T>::get_R_index(const int& index) const
{
    return &(R_index[index*3]);
}

}