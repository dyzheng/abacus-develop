//#include "module_cell/atom_spec.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "base_matrix.h"

namespace hamilt
{
/**
Class: AtomPair
Template Parameters:

T: The type of the matrix elements.

Member Variables:

R_index: A std::vector of integers representing the three indices of the cell for each matrix in values.

values: A std::vector of UnitMatrix<T> representing the matrix for each cell.

element_i: A pointer to the first atom in the pair.

element_j: A pointer to the second atom in the pair.

current_R: The index of the current cell.

Member Functions:

AtomPair(): The constructor of the AtomPair class. It initializes the member variables.

~AtomPair(): The destructor of the AtomPair class. It frees the memory allocated for the matrix elements.

get_R_values(int rx_in, int ry_in, int rz_in): Returns a reference to the matrix in the cell with the specified indices.

get_R_values(int rx_in, int ry_in, int rz_in) const: Returns a const reference to the matrix in the cell with the specified indices.

convert_add(const UnitMatrix<T>& target, int rx_in, int ry_in, int rz_in): Adds the matrix target to the matrix in the cell with the specified indices.

convert_save(const UnitMatrix<T>& target, int rx_in, int ry_in, int rz_in): Saves the matrix target to the matrix in the cell with the specified indices.

add_to_matrix(container::Tensor& hk, const std::complex<T> &kphase) const: Adds the matrix elements to the hk tensor with the specified phase factor.
*/

template<typename T>
class AtomPair
{
  public:
    // Constructor of class AtomPair
    // Only for 2d-block MPI parallel case
    // This constructor used for initialize a atom-pair local Hamiltonian with only center cell
    // which is used for constructing HK (k space Hamiltonian) objects, (gamma_only case)
    AtomPair(
        const int& atom_i_,        // atomic index of atom i, used to identify atom
        const int& atom_j_,        // atomic index of atom j, used to identify atom
        const Parallel_Orbitals* paraV_,  // information for 2d-block parallel
        const T* existed_matrix = nullptr // if nullptr, new memory will be allocated, otherwise this class is a data wrapper
    );
    // Constructor of class AtomPair
    // Only for 2d-block MPI parallel case
    // This constructor used for initialize a atom-pair local Hamiltonian with non-zero cell indexes, 
    // which is used for constructing HR (real space Hamiltonian) objects.  
    AtomPair(
        const int& atom_i_,        // atomic index of atom i, used to identify atom
        const int& atom_j_,        // atomic index of atom j, used to identify atom
        const int& rx,             // x coordinate of cell
        const int& ry,             // y coordinate of cell
        const int& rz,             // z coordinate of cell
        const Parallel_Orbitals* paraV_,  // information for 2d-block parallel
        const T* existed_array = nullptr  // if nullptr, new memory will be allocated, otherwise this class is a data wrapper
    );
    // This constructor used for initialize a atom-pair local Hamiltonian with only center cell
    // which is used for constructing HK (k space Hamiltonian) objects, (gamma_only case)
    AtomPair(
        const int& atom_i,         // atomic index of atom i, used to identify atom
        const int& atom_j,         // atomic index of atom j, used to identify atom
        const int* row_atom_begin, // array, contains starting indexes in Hamiltonian matrix of atom i
        const int* col_atom_begin,  // array, contains starting indexes in Hamiltonian matrix of atom j
        const int& natom,
        const T* existed_matrix = nullptr
    );

    // This constructor used for initialize a atom-pair local Hamiltonian with non-zero cell indexes, 
    // which is used for constructing HR (real space Hamiltonian) objects.  
    AtomPair(
        const int& atom_i,         // atomic index of atom i, used to identify atom
        const int& atom_j,         // atomic index of atom j, used to identify atom
        const int& rx,             // x coordinate of cell
        const int& ry,             // y coordinate of cell
        const int& rz,             // z coordinate of cell
        const int* row_atom_begin, // array, contains starting indexes in Hamiltonian matrix of atom i
        const int* col_atom_begin,  // array, contains starting indexes in Hamiltonian matrix of atom j
        const int& natom,
        const T* existed_matrix = nullptr
    );
    //Destructor of class AtomPair
    ~AtomPair(){};

    //interface for get target matrix of target cell
    BaseMatrix<T> &get_R_values(int rx_in, int ry_in, int rz_in) const;
    
    //this interface will call get_value in this->values
    T& get_value(const int& i) const;
    T& get_value(const int& row, const int& col) const;
    T& get_matrix_value(const size_t &i_row_global, const size_t &j_col_global) const;

    // add another BaseMatrix<T> to 
    void convert_add(const BaseMatrix<T>& target, int rx_in, int ry_in, int rz_in);
    void convert_save(const BaseMatrix<T>& target, int rx_in, int ry_in, int rz_in);
    
    void add_to_matrix(std::complex<T>* hk, const int ncol_hk, const std::complex<T> &kphase)const;

    /*const Atom* element_i = nullptr;
    const Atom* element_j = nullptr;*/

  private:
  
    //it contains 3 index of cell, size of R_index is three times of values.
    std::vector<int> R_index;

    //it contains containers for accessing matrix of this atom-pair 
    std::vector<BaseMatrix<T>> values;

    //only for 2d-block
    const Parallel_Orbitals* paraV = nullptr;

    //the default R index is (0, 0, 0)
    int current_R = 0;

    //index for identifying atom I and J for this atom-pair
    int atom_i = -1;
    int atom_j = -1;
    //start index of row for this Atom-Pair
    int row_ap = -1;
    //start index of col for this Atom-pair
    int col_ap = -1;
    int row_size = 0;
    int col_size = 0;
    int ldc = -1; //leading_dimention_colomn

};

}