#include "base_matrix.h"
#include "module_base/global_function.h"

namespace hamilt
{

template<typename T>
BaseMatrix<T>::BaseMatrix(const int &nrow_, const int &ncol_, T* data_existed = nullptr)
{
    value_begin.clear();
    nrow_local = nrow_;
    ncol_local = ncol_;
    //the default memory_type is 1 (2d-block format), it doesn't matter for initialization
    if(data_existed == nullptr)
    {
        value_begin = new T[nrow_local * ncol_local];
        ModuleBase::GlobalFunc::ZEROS(value_begin, nrow_local * ncol_local);
        this->allocated = true;
    }
    else
    {
        value_begin = data_existed;
    }
}

//move constructor
template<typename T>
BaseMatrix<T>::BaseMatrix(BaseMatrix<T>&& matrix)
{
    this->nrow_local = matrix.nrow_local;
    this->ncol_local = matrix.ncol_local;
    this->value_begin = matrix.value_begin;
    this->allocated = matrix.allocated;
    if(matrix.allocated)
    {
        matrix.allocated = false;
    }
    matrix.value_begin = nullptr;
}

//copy constructor
template<typename T>
BaseMatrix<T>::BaseMatrix(const BaseMatrix<T>& matrix)
{
    this->nrow_local = matrix.nrow_local;
    this->ncol_local = matrix.ncol_local;
    this->memory_type = matrix.memory_type;
    this->ldc = matrix.ldc;
    if(matrix.allocated)
    {
        this->value_begin = new T[nrow_local * ncol_local];
        ModuleBase::GlobalFunc::ZEROS(this->value_begin, nrow_local * ncol_local);
        this->allocated = true;
        for(int i = 0; i < nrow_local * ncol_local; i++)
        {
            this->value_begin[i] = matrix.value_begin[i];
        }
    }
    else
    {
        this->value_begin = matrix.value_begin;
        this->allocated = false;
    }
}

template<typename T>
BaseMatrix<T>::~BaseMatrix()
{
    if(this->allocated)
    {
        delete[] value_begin;
    }
}

//set_memory_type
template<typename T>
void BaseMatrix<T>::set_memory_type(const int &memory_type_in)
{
    this->memory_type = memory_type_in;
}

//add_array
template<typename T>
void BaseMatrix<T>::add_array(T* array)
{
    //if allocated, save data from array into matrix
    //if whole matrix and 2d-block format, save data from array into matrix either
    if(this->allocated || this->memory_type == 0 || this->memory_type == 1)
    {
        for(int i = 0; i < nrow_local * ncol_local; i++)
        {
            value_begin[i] += array[i];
        }
    }
    else
    {//if not allocated, then it is a wrapper
        if(this->memory_type == 2 )
        {
            for (int i = 0; i < matrix.nrow_local; i++) 
            {
                for (int j = 0; j < matrix.ncol_local; j++) 
                {
                    matrix.add_element(i, j, *array);
                    array++;
                }
            }
        }
        
    }
}

template<typename T>
void BaseMatrix<T>::add_element(int mu, int nu, const T& value){
    const int dim = this->memory_type==2?this->ldc:this->ncol_local;
    int index = mu * dim + nu;
    value_begin[index] += value;
}

template<typename T>
T& BaseMatrix<T>::get_value(const size_t &i_row, const size_t &j_col) const{
    const int dim = this->memory_type==2?this->ldc:this->ncol_local;
    int index = i_row * dim + j_col;
    return value_begin[index];
}

template<typename T>
T* BaseMatrix<T>::get_pointer() const{
    return value_begin;
}


}