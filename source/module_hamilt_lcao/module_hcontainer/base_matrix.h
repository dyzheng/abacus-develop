#include<complex>
#include<map>
#include<vector>

namespace hamilt
{

/**
 * Class: BaseMatrix
 * Template Parameters:
 * T: The type of the matrix elements.
 * 
 * Member Variables:
 * 
 * value_begin: A std::vector of pointers to the matrix elements. It can be used to access the matrix elements in different ways depending on the memory type.
 * 
 * current_multiple: An integer indicating the current multiple of the matrix size. It is used to calculate the index of an element in the value_begin vector.
 * 
 * num_orb_i: An integer indicating the number of rows in the matrix.
 * 
 * num_orb_j: An integer indicating the number of columns in the matrix.
 * 
 * memory_type: An integer indicating the memory type. It can be:
 * 0: The whole matrix is stored in a linear array.
 * 1: The matrix is divided into 2D-blocks.
 * 2: A submatrix of the whole matrix is stored in a linear array.
 * 3: The whole matrix is sparse and stored in a compressed format.

 * Member Functions:
 * BaseMatrix(): The constructor of the baseMatrix class. It initializes the member variables.
 * ~BaseMatrix(): The destructor of the baseMatrix class. It frees the memory allocated for the matrix elements.
 * save_array(T* array): Stores an array of matrix elements in the value_begin vector.
 * add_array(T* array): Adds an array of matrix elements to the value_begin vector.
 * save_element(int mu, int nu, const T& value): Saves a single matrix element at the specified row and column.
 * add_element(int mu, int nu, const T& value): Adds a single matrix element to the one already stored at the specified row and column.
 * get_value(int mu, int nu): Returns a reference to the matrix element at the specified row and column.
 * get_pointer(): Returns a pointer to the first element of the value_begin vector.
 * get_value(const size_t &i_row, const size_t &j_col, const size_t &ldc) const: Returns a reference to the matrix element at the specified row and column in a block matrix.
 * get_pointer() const: Returns a const pointer to the first element of the value_begin vector.          */

template<typename T>
class BaseMatrix
{
    public:
    //Constructor of class BaseMatrix
    BaseMatrix(const int &nrow_, const int &ncol_, T* data_existed = nullptr);
    //Destructor of class BaseMatrix
    ~BaseMatrix(){};

    //pointer for accessing data
    //two ways to arrange data:
    //1. allocate data itself
    //2. only access data but be arranged by RealSparseHamiltonian
    std::vector<T*> value_begin;
    int current_multiple = 0;

    int nrow_local = 0;
    int ncol_local = 0;

    //memory type, choose how to access value via pointer of array
    //0 is whole matrix
    //1 is 2d-block
    //2 is submatrix in whole matrix
    //3 is sparse matrix in whole matrix
    int memory_type = 1;

    void save_array(T* array);
    void add_array(T* array);
    void save_element(int mu, int nu, const T& value);
    void add_element(int mu, int nu, const T& value);
    //for inside matrix
    T& get_value(const size_t &i_row, const size_t &j_col) const;
    //for block matrix
    T& get_value(const size_t &i_row, const size_t &j_col, const size_t &ldc) const;
    T* get_pointer() const;

};

}