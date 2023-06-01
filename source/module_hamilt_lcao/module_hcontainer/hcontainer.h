#ifndef HCONTAINER_H
#define HCONTAINER_H

#include <complex>
#include <unordered_map>
#include <vector>

#include "atom_pair.h"
#include "module_cell/unitcell.h"

namespace hamilt
{

template <typename T>
class HContainer
{
  public:
    // Destructor of class HContainer
    ~HContainer(){};

    // copy constructor
    HContainer(const HContainer<T>& HR_in);

    // move constructor
    HContainer(HContainer<T>&& HR_in);

    // simple constructor
    HContainer();

    // use unitcell to initialize atom_pairs
    HContainer(const UnitCell& ucell_);

    /**
     * @brief a AtomPair object can be inserted into HContainer, two steps:
     * 1, find AtomPair with atom index atom_i and atom_j
     * 2.1, if found, add to exist AtomPair,
     * 2.2, if not found, insert new one and sort.
     *
     * @param atom_ij AtomPair object
     */
    void insert_pair(const AtomPair<T>& atom_ij);

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
     * @brief return a reference of AtomPair with index of atom I and J in atom_pairs
     *
     * @param i index of atom i
     * @param j index of atom j
     */
    AtomPair<T>& get_atom_pair(int i, int j) const;
    /**
     * @brief return a reference of AtomPair with index in
     * atom_pairs (R is not fixed)
     * tmp_atom_pairs (R is fixed)
     *
     * @param index index of atom-pair
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
    T& operator()(int atom_i, int atom_j, int rx_in, int ry_in, int rz_in, int mu, int nu) const
    {
        return this->get_atom_pair(atom_i, atom_j).get_HR_values(rx_in, ry_in, rz_in).get_value(mu, nu);
    }

    // save atom-pair pointers into this->tmp_atom_pairs for selected R index
    /**
     * @brief save atom-pair pointers into this->tmp_atom_pairs for selected R index
     *
     * @param rx_in index of R in x direction
     * @param ry_in index of R in y direction
     * @param rz_in index of R in z direction
     * @return true if success
     */
    bool fix_R(int rx_in, int ry_in, int rz_in) const;

    /**
     * @brief set current_R to -1, which means R is not fixed
     * clear this->tmp_atom_pairs
     */
    void unfix_R() const;

    /**
     * @brief restrict R indexes of all atom-pair to 0, 0, 0
     * add BaseMatrix<T> with non-zero R index to this->atom_pairs[i].values[0]
     * set gamma_only = true
     * in this mode:
     *   1. fix_R() can not be used
     *   2. there is no interface to set gamma_only = false, user should create a new HContainer if needed
     *   3. get_size_for_loop_R() and loop_R() can not be used
     *   4. get_AP_size() can be used
     *   5. data(i, j) can be used to get pointer of target atom-pair with R = 0, 0, 0 , data(i,j,R) can not be used
     *   6. insert_pair() can be used, but the R index will be ignored
     *   7. get_atom_pair(), find_atom_pair() can be used
     *   8. operator() can be used, but the R index will be ignored
     */
    void fix_gamma();

    // interface for call a R loop for HContainer
    // it can return a new R-index with (rx,ry,rz) for each loop
    // if index==0, a new loop of R will be initialized
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
    void loop_R(const size_t& index, int& rx, int& ry, int& rz) const;

    /**
     * @brief calculate number of R index which has counted AtomPairs
     */
    size_t size_R_loop() const;

    /**
     * @brief calculate number of AtomPairs for current R index
     */
    size_t size_atom_pairs() const;

    /**
     * @brief get data pointer of AtomPair with index of I, J
     *
     * @param i index of atom i
     * @param j index of atom j
     * @return T* pointer of data
     */
    T* data(int i, int j) const;

    /**
     * @brief get data pointer of AtomPair with index of I, J, Rx, Ry, Rz
     *
     * @param i index of atom i
     * @param j index of atom j
     * @param R int[3] of R index
     * @return T* pointer of data
     */
    T* data(int i, int j, int* R) const;

  private:
    // i-j atom pairs, sorted by matrix of (i, j)
    std::vector<AtomPair<T>> atom_pairs;

    /**
     * @brief temporary atom-pair lists to loop selected R index
     */
    mutable std::vector<const AtomPair<T> const*> tmp_atom_pairs;
    // it contains 3 index of cell, size of R_index is three times of values.
    mutable std::vector<int> tmp_R_index;
    // current index of R in tmp_atom_pairs, -1 means not initialized
    mutable int current_R = -1;
    /**
     * @brief find index of R in tmp_R_index, used when current_R is fixed
     *
     * @param rx_in index of R in x direction
     * @param ry_in index of R in y direction
     * @param rz_in index of R in z direction
     * @return int index of R in tmp_R_index
     */
    int find_R(const int& rx_in, const int& ry_in, const int& rz_in) const;

    bool gamma_only = false;

    // int multiple = 1;
    // int current_multiple = 0;
};

} // namespace hamilt

#endif