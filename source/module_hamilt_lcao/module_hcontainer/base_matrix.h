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
    //copy constructor
    BaseMatrix(const BaseMatrix<T>& matrix);
    //move constructor
    BaseMatrix(BaseMatrix<T>&& matrix);
    //Destructor of class BaseMatrix
    ~BaseMatrix();

    /**
     * @brief save an array to the matrix
     * 
     * @param array array to be saved
    */
    void add_array(T* array);
    /**
     * @brief add a single element to the matrix
     * 
     * @param mu row index
     * @param nu column index
     * @param value value to be added
    */
    void add_element(int mu, int nu, const T& value);
    //for inside matrix
    /**
     * @brief get value from a whole matrix
     * for memory_type = 0 or 1, ncol_local will be used to calculate the index
     * for memory_type = 2, ldc will be used to calculate the index
     * 
     * @param i_row row index
     * @param j_col column index
     * @return T&
    */
    T& get_value(const size_t &i_row, const size_t &j_col) const;
    /**
     * @brief get pointer of value from a submatrix
    */
    T* get_pointer() const;

    void set_memory_type(const int &memory_type_in);

    private:
    bool allocated = false;

    //pointer for accessing data
    //two ways to arrange data:
    //1. allocate data itself
    //2. only access data but be arranged by RealSparseHamiltonian
    T* value_begin = nullptr;
    
    //int current_multiple = 0;

    //number of rows and columns
    int nrow_local = 0;
    int ncol_local = 0;

    //memory type, choose how to access value via pointer of array
    //0 is whole matrix
    //1 is 2d-block
    //2 is submatrix in whole matrix
    //3 is sparse matrix in whole matrix , not implemented yet
    int memory_type = 1;

    //leading dimension of matrix, used with memory_type = 2
    int ldc = 0;

};

}