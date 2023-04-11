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
    //first level is i-j atom pairs
    std::vector<int> atom_i;
    std::vector<int> atom_j;
    std::vector<AtomPair<T>> atom_ij;

    void insert_pair(int i, int j, const AtomPair<T>& atom_ij);
    void add_atom_pair(int i, int j, const AtomPair<T>& atom_ij, std::complex<T> alpha);
    AtomPair<T>& get_atom_pair(int i, int j) const;
    AtomPair<T>& get_atom_pair(int index) const;

    T &operator()(int atom_i, int atom_j, int rx_in, int ry_in, int rz_in, int mu, int nu)
    {
        return this->get_atom_pair(atom_i, atom_j).get_R_values(rx_in, ry_in, rz_in).get_value(mu, nu);
    }

    const T &operator()(int atom_i, int atom_j, int rx_in, int ry_in, int rz_in, int mu, int nu) const
    {
        return this->get_atom_pair(atom_i, atom_j).get_R_values(rx_in, ry_in, rz_in).get_value(mu, nu);
    }

    //used for choosing spin for Hamiltonian
    //used for choosing x/y/z for DH/Dtau matrix
    //used for choosing 11/12/13/22/23/33 for DH/strain matrix
    void fix_index(int multiple_in);

    int multiple = 1;
    int current_multiple = 0;

    void fix_R(int rx_in, int ry_in, int rz_in);
    void fix_gamma();
    
    //interface for call a R loop for HContainer
    void loop_R(const size_t &index, int &rx, int &ry, int &rz) const;

    //calculate number of R index which has counted AtomPairs
    size_t get_R_size();

    //calculate number of AtomPairs for current R index
    size_t get_AP_size();

    //get data pointer of AtomPair with index of I, J, (R)
    T* get_ap_data(int i, int j){return get_atom_pair(i,j).get_pointer();}
    T* get_ap_data(int i, int j, int* R){return get_atom_pair(i,j).get_R_value(R[0], R[1], R[2]).get_pointer();}
    
};

}