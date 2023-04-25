// this class used for dump data in DEBUG mode
#ifndef EXPORT_H
#define EXPORT_H

#ifdef __DEBUG
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include "global_variable.h"
#include <complex>
#ifdef __MPI
#include "mpi.h"
#include "parallel_reduce.h"
#endif

namespace ModuleBase
{

extern bool out_alllog;
extern bool out_mat_hs;
extern bool out_mat_hsR;

//used for output matrix info in header of output file
struct MatrixHeader {
    int size; // total size of matrix
    int rows; // number of row of matrix
    int cols; // number of column of matrix
    bool is_triangular; // if the matrix is upper trangular matrix
	int data_bits;
};

struct ArrayHeader {
	int size;
	int data_bits;
};

template<typename T>
bool if_not_zero(const T& data);

template<typename T>
void dump_matrix(const T* mat, const int& row, const int& col, const int& size, const std::string& filename) {
    MatrixHeader header;
    header.size = size;
    header.rows = row;
    header.cols = col;
    header.is_triangular = false; // default setting 
	header.data_bits = sizeof(T);

    std::stringstream ss;
    ss << GlobalV::global_out_dir << filename;
    std::ofstream fout;
    if(out_alllog)
    {//print for all processes
        ss << GlobalV::MY_RANK;
    }
    else if( GlobalV::MY_RANK != 0)
    {//print for first process
        return;
    }
    //print matrix to file
    fout.open(ss.str());
    if (!fout) {
        std::cerr << "Error: failed to open file " << filename << " for writing" << std::endl;
        return;
    }
    fout<<header.size<<" "<<header.rows<<" "<<header.cols<<std::endl;
    for(int ir = 0;ir<row; ir++)
    {
        for(int ic = 0;ic<col; ic++)
        {
            if(if_not_zero(mat[ir + ic * col])) 
            {
                fout<<ir<<" "<<ic<<" "<<mat[ir + ic * col]<<std::endl;
            }
        }
    }
    /** binary mode
    std::ofstream fout(ss.str(), std::ios::binary);
    if (!fout) {
        std::cerr << "Error: failed to open file " << filename << " for writing" << std::endl;
        return;
    }
    fout.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (int i = 0; i < mat.size(); i++) {
        fout.write(reinterpret_cast<const char*>(mat[i].data()), mat[i].size() * sizeof(double));
    }
    */
    fout.close();
}

template<typename T>
void dump_reduced_matrix(
    const T* mat, 
    const int& row, 
    const int& col, 
    const int* trace_loc_row, 
    const int* trace_loc_col, 
    const int& target_dim, 
    const std::string& filename)
{
#ifdef __MPI
    std::ofstream g1;

    std::stringstream ss;
    ss << GlobalV::global_out_dir << filename;
    bool error = false;
    if (GlobalV::MY_RANK == 0)
    {
        g1.open(ss.str());
        if (!g1) {
            std::cerr << "Error: failed to open file " << filename << " for writing" << std::endl;
            error = true;
        }
        g1 << target_dim<<std::endl;
    }
    if(error)
    {
        return;
    }

    int ir, ic;
    std::vector<T> lineH(target_dim);
    for (int i = 0; i < target_dim; i++)
    {
        ModuleBase::GlobalFunc::ZEROS(lineH.data(), target_dim);

        ir = trace_loc_row[i];
        if (ir >= 0)
        {
            // data collection
            for (int j = 0; j < target_dim; j++)
            {
                ic = trace_loc_col[j];
                if (ic >= 0)
                {
                    int iic;
                    if (ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER())
                    {
                        iic = ir + ic * row;
                    }
                    else
                    {
                        iic = ir * col + ic;
                    }
                    lineH[j] = mat[iic];
                }
            }
        }

        Parallel_Reduce::reduce_complex_double_pool(lineH.data(), target_dim);

        if (GlobalV::MY_RANK == 0)
        {
            for (int j = 0; j < target_dim; j++)
            {
                if(if_not_zero(lineH[j])) 
                {
                    g1 << i<<" "<<j<<" " << lineH[j] << std::endl;
                }
            }
        }
    }

    if (GlobalV::MY_RANK == 0)
    {
        g1.close();
    }
#endif
}

template<typename T>
void dump_array(const T* array, const int& size, const std::string& filename) {
    ArrayHeader header;
    header.size = size;
	header.data_bits = sizeof(T);

    std::stringstream ss;
    ss << GlobalV::global_out_dir << filename;
    std::ofstream fout;
    if(out_alllog)
    {//print for all processes
        ss << GlobalV::MY_RANK;
    }
    else if( GlobalV::MY_RANK != 0)
    {//print for first process
        return;
    }
    //print array to file
    fout.open(ss.str());
    if (!fout) {
        std::cerr << "Error: failed to open file " << filename << " for writing" << std::endl;
        return;
    }
    fout<<header.size<<std::endl;
    for(int i = 0;i<size; i++)
    {
        if(if_not_zero(array[i])) 
        {
            fout<<i<<" "<<array[i]<<std::endl;
        }
    }
    fout.close();
}

}

#endif

#endif 
