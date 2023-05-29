#include<complex>
#include<map>
#include<vector>
#include "atom_pair.h"
#include "module_cell/unitcell.h"

namespace hamilt
{

template<typename T>
class HContainer
{
  public:
    //Destructor of class HContainer
    ~HContainer(){};

    //Constructor of class HContainer
    HContainer(
        const UnitCell& ucell_,
        const int strategy
    );
    HContainer(
        const UnitCell& ucell_,
        const int strategy,
        const Parallel_Orbitals* paraV
    );
    HContainer(
        const UnitCell& ucell_,
        const int strategy,
        const Parallel_Orbitals* paraV,
        const T* whole_matrix
    );
    HContainer(
        const HContainer<T>& HR_in,
        const int strategy
    );

    /**
     * @brief a AtomPair object can be inserted into HContainer, two steps:
     * 1, find AtomPair with atom index atom_i and atom_j
     * 2.1, if found, add to exist AtomPair, 
     * 2.2, if not found, insert new one and sort.
     * 
     * @param i atom index of atom i
     * @param j atom index of atom j
     * @param atom_ij AtomPair object
     */
    void insert_pair(int i, int j, const AtomPair<T>& atom_ij);
    
    /**
     * @brief find AtomPair with atom index atom_i and atom_j
     * This interface can be used to find AtomPair,
     * if found, return pointer will be the exist one,
     * if not found, return pointer will be nullptr.
     * 
     * @param i atom index of atom i
     * @param j atom index of atom j
     * @return AtomPair<T>*
    */
    AtomPair<T>* find_pair(int i, int j) const;

    /**
     * @brief add AtomPair with atom index atom_i and atom_j
     * This interface can be used to add AtomPair,
     * if found, add to exist one,
     * if not found, insert new one and sort.
     * used for calculate H(K) in H(K) += H(R) * kphase for non-soc case (T -> complex<T>)
     * 
     * @param i atom index of atom i
     * @param j atom index of atom j
     * @param atom_ij AtomPair object
     */
    void add_atom_pair(int i, int j, const AtomPair<std::complex<T>>& atom_ij, std::complex<T> alpha);
    /**
     * @brief add AtomPair with atom index atom_i and atom_j
     * This interface can be used to add AtomPair,
     * if found, add to exist one,
     * if not found, insert new one and sort.
     * used for calculate H(K) in H(K) += H(R) * kphase for soc case (complex<T> -> complex<T>)
     * 
     * @param i atom index of atom i
     * @param j atom index of atom j
     * @param atom_ij AtomPair object
     */
    void add_atom_pair(int i, int j, const AtomPair<T>& atom_ij, T alpha);
    /**
     * return a reference of AtomPair with index of atom I and J in
     * atom_pairs (R is not fixed)
     * tmp_atom_pairs (R is fixed)
     */
    AtomPair<T>& get_atom_pair(int i, int j) const;
    /**
     * return a reference of AtomPair with index in 
     * atom_pairs (R is not fixed)
     * tmp_atom_pairs (R is fixed)
     */
    AtomPair<T>& get_atom_pair(int index) const;

    /**
     * @brief operator() for accessing value with indexes
     * 
     * @param atom_i index of atom i
     * @param atom_j index of atom j
     * @param rx_in index of R in x direction
     * @param ry_in index of R in y direction
     * @param rz_in index of R in z direction
     * @param mu index of orbital mu
     * @param nu index of orbital nu
     * @return T&
    */
    T &operator()(int atom_i, int atom_j, int rx_in, int ry_in, int rz_in, int mu, int nu) const
    {
        return this->get_atom_pair(atom_i, atom_j).get_R_values(rx_in, ry_in, rz_in).get_value(mu, nu);
    }

    //used for choosing spin for Hamiltonian
    //used for choosing x/y/z for DH/Dtau matrix
    //used for choosing 11/12/13/22/23/33 for DH/strain matrix
    //void fix_index(int multiple_in);

    //save atom-pair pointers into this->tmp_atom_pairs for selected R index
    /**
     * @brief save atom-pair pointers into this->tmp_atom_pairs for selected R index
     * 
     * @param rx_in index of R in x direction
     * @param ry_in index of R in y direction
     * @param rz_in index of R in z direction
     * @return true if success
    */
    bool fix_R(int rx_in, int ry_in, int rz_in);
    /**
     * @brief restrict R indexes of all atom-pair to 0, 0, 0
     * add BaseMatrix<T> with non-zero R index to this->atom_pairs[i].values[0]
    */
    void fix_gamma();
    
    //interface for call a R loop for HContainer
    //it can return a new R-index with (rx,ry,rz) for each loop
    //if index==0, a new loop of R will be initialized 
    /**
     * @brief interface for call a R loop for HContainer
     * it can return a new R-index with (rx,ry,rz) for each loop
     * if index==0, a new loop of R will be initialized
     * 
     * @param index index of R loop
     * @param rx index of R in x direction, would be set in the function
     * @param ry index of R in y direction, would be set in the function
     * @param rz index of R in z direction, would be set in the function
    */
    void loop_R(const size_t &index, int &rx, int &ry, int &rz) const;

    /**
     * @brief calculate number of R index which has counted AtomPairs
    */
    size_t get_size_for_loop_R();

    /**
     * @brief calculate number of AtomPairs for current R index
    */
    size_t get_AP_size();

    /**
     * @brief get data pointer of AtomPair with index of I, J
     * 
     * @param i index of atom i
     * @param j index of atom j
     * @return T* pointer of data
    */
    T* get_ap_data(int i, int j){return get_atom_pair(i,j).get_pointer();}

    /**
     * @brief get data pointer of AtomPair with index of I, J, Rx, Ry, Rz
     * 
     * @param i index of atom i
     * @param j index of atom j
     * @param R int[3] of R index
     * @return T* pointer of data
    */
    T* get_ap_data(int i, int j, int* R){return get_atom_pair(i,j).get_R_value(R[0], R[1], R[2]).get_pointer();}

  private:
    //i-j atom pairs, sorted by matrix of (i, j)
    std::vector<AtomPair<T>> atom_pairs;

    //temporary atom-pair lists to loop selected R index 
    std::vector<AtomPair<T>*> tmp_atom_pairs;
    //it contains 3 index of cell, size of R_index is three times of values.
    std::vector<int> tmp_R_index;
    //current index of R in tmp_atom_pairs, -1 means not initialized
    int current_R = -1;
    //it contains a map of (rx, ry, rz) to int* in tmp_R_index
    std::unordered_map<std::tuple<int, int, int>, int> tmp_R_index_map;

    //int multiple = 1;
    //int current_multiple = 0;
    
};

}