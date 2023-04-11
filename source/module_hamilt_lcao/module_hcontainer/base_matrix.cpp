#include "base_matrix.h"

namespace hamilt
{

template<typename T>
BaseMatrix<T>::BaseMatrix(){
    value_begin.clear();
    current_multiple = 0;
    num_orb_i = 0;
    num_orb_j = 0;
    memory_type = 1;
}
template<typename T>
BaseMatrix<T>::~BaseMatrix(){
    for (auto it = value_begin.begin(); it != value_begin.end(); ++it) {
        delete[] *it;
    }
    value_begin.clear();
}
template<typename T>
void BaseMatrix<T>::save_array(T* array){
    value_begin.push_back(array);
}
template<typename T>
void BaseMatrix<T>::add_array(T* array){
    value_begin.push_back(array);
}
template<typename T>
void BaseMatrix<T>::save_element(int mu, int nu, const T& value){
    int index = mu*num_orb_j + nu;
    value_begin[current_multiple][index] = value;
}
template<typename T>
void BaseMatrix<T>::add_element(int mu, int nu, const T& value){
    int index = mu*num_orb_j + nu;
    value_begin[current_multiple][index] += value;
}
template<typename T>
T& BaseMatrix<T>::get_value(const size_t &i_row, const size_t &j_col) const{
    int index = i_row*num_orb_j + j_col;
    return value_begin[current_multiple][index];
}
template<typename T>
T& BaseMatrix<T>::get_value(const size_t &i_row, const size_t &j_col, const size_t &ldc) const
{
    int index = i_row*ldc + j_col;
    return value_begin[current_multiple][index];
}

template<typename T>
T* BaseMatrix<T>::get_pointer() const{
    return value_begin[current_multiple];
}


}