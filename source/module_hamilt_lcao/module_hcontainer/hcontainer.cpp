#include "hcontainer.h"

namespace hamilt
{

// class HContainer

// T of HContainer can be double or complex<double>
template class HContainer<double>;
template class HContainer<std::complex<double>>;

// copy constructor
template <typename T>
HContainer<T>::HContainer(const HContainer<T>& HR_in)
{
    this->atom_pairs = HR_in.atom_pairs;
    this->gamma_only = HR_in.gamma_only;
    this->current_R = -1;
    // tmp terms not copied
}

// move constructor
template <typename T>
HContainer<T>::HContainer(HContainer<T>&& HR_in)
{
    this->atom_pairs = std::move(HR_in.atom_pairs);
    this->gamma_only = HR_in.gamma_only;
    this->current_R = -1;
    // tmp terms not moved
}

// simple constructor
template <typename T>
HContainer<T>::HContainer()
{
    this->gamma_only = false;
    this->current_R = -1;
}

// use unitcell to initialize atom_pairs
template <typename T>
HContainer<T>::HContainer(const UnitCell& ucell_)
{
    this->gamma_only = false;
    this->current_R = -1;
    // initialize atom_pairs
    for (int i = 0; i < ucell_.nat; i++)
    {
        for (int j = 0; j < ucell_.nat; j++)
        {
            AtomPair<T> atom_ij(i, j);
            int it1 = ucell_.iat2it[i];
            int it2 = ucell_.iat2it[j];
            atom_ij.set_size(ucell_.atoms[it1].nw, ucell_.atoms[it2].nw);
            this->atom_pairs.push_back(atom_ij);
        }
    }
    // sort atom_pairs
    std::sort(this->atom_pairs.begin(), this->atom_pairs.end());
}

template <typename T>
AtomPair<T>* HContainer<T>::find_pair(int atom_i, int atom_j) const
{
    AtomPair<T> target(atom_i, atom_j);
    auto it = std::lower_bound(this->atom_pairs.begin(), this->atom_pairs.end(), target);
    if (it != this->atom_pairs.end() && it->identify(atom_i, atom_j))
    {
        AtomPair<T>* tmp_pointer = const_cast<AtomPair<T>*>(&(*it));
        return tmp_pointer;
    }
    else
    {
        return nullptr;
    }
}

// get_atom_pair with atom_ij
template <typename T>
AtomPair<T>& HContainer<T>::get_atom_pair(int atom_i, int atom_j) const
{
    AtomPair<T> target(atom_i, atom_j);
    auto it = std::lower_bound(this->atom_pairs.begin(), this->atom_pairs.end(), target);
    if (it != this->atom_pairs.end() && it->identify(atom_i, atom_j))
    {
        AtomPair<T>* tmp = const_cast<AtomPair<T>*>(&(*it));
        return *tmp;
    }
    else
    {
        std::cout << "Error: atom pair not found in get_atom_pair" << std::endl;
        exit(1);
    }
}

// get_atom_pair with index
template <typename T>
AtomPair<T>& HContainer<T>::get_atom_pair(int index) const
{
#ifdef __DEBUG
    if (this->current_R > -1)
    {
        if (index >= this->tmp_atom_pairs.size() || index < 0)
        {
            std::cout << "Error: index out of range in get_atom_pair" << std::endl;
            exit(1);
        }
    }
    else
    {
        if (index >= this->atom_pairs.size() || index < 0)
        {
            std::cout << "Error: index out of range in get_atom_pair" << std::endl;
            exit(1);
        }
    }
#endif
    if (this->current_R > -1)
    {
        return const_cast<AtomPair<T>&>(*this->tmp_atom_pairs[index]);
    }
    else
    {
        return const_cast<AtomPair<T>&>(this->atom_pairs[index]);
    }
}

template <typename T>
bool HContainer<T>::fix_R(int rx_in, int ry_in, int rz_in) const
{
    // clear and reallocate the memory of this->tmp_atom_pairs
    this->tmp_atom_pairs.clear();
    this->tmp_atom_pairs.shrink_to_fit();
    this->tmp_atom_pairs.reserve(this->atom_pairs.size());

    // find (rx, ry, rz) in this->atom_pairs[i].R_values
    for (auto it = this->atom_pairs.begin(); it != this->atom_pairs.end(); ++it)
    {
        if (it->find_R(rx_in, ry_in, rz_in))
        {
            // push bach the pointer of AtomPair to this->tmp_atom_pairs
            const AtomPair<T> const* tmp_pointer = &(*it);
            this->tmp_atom_pairs.push_back(tmp_pointer);
        }
    }
    if (this->tmp_atom_pairs.size() == 0)
    {
        std::cout << "Error: no atom pair found in fix_R" << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

// unfix_R
template <typename T>
void HContainer<T>::unfix_R() const
{
    this->current_R = -1;
    this->tmp_atom_pairs.clear();
    this->tmp_atom_pairs.shrink_to_fit();
}

// fix_gamma
template <typename T>
void HContainer<T>::fix_gamma()
{
    // every AtomPair in this->atom_pairs has the (0, 0, 0) cell index
    // fix every AtomPair in this->atom_pairs to only center cell
    this->gamma_only = true;
    for (auto it = this->atom_pairs.begin(); it != this->atom_pairs.end(); ++it)
    {
        it->merge_to_gamma();
    }
    // in gamma_only case, R_index is not needed, tmp_R_index should be empty
    if (this->current_R != -1)
    {
        this->current_R = -1;
        this->tmp_R_index.clear();
        this->tmp_R_index.shrink_to_fit();
    }
}

// find_R
template <typename T>
int HContainer<T>::find_R(const int& rx_in, const int& ry_in, const int& rz_in) const
{
    // search (rx, ry, rz) in this->tmp_R_index
    if (this->tmp_atom_pairs.empty())
    {
        return -1;
    }
    for (int i = 0; i < this->tmp_R_index.size() / 3; i++)
    {
        if (this->tmp_R_index[i * 3] == rx_in && this->tmp_R_index[i * 3 + 1] == ry_in
            && this->tmp_R_index[i * 3 + 2] == rz_in)
        {
            return i;
        }
    }
    return -1;
}

// size_R_loop, return the number of different cells in this->atom_pairs
template <typename T>
size_t HContainer<T>::size_R_loop() const
{
    /**
     * start a new iteration of loop_R
     * there is different R-index in this->atom_pairs[i].R_values
     * search them one by one and store them in this->tmp_R_index
     */
    this->tmp_R_index.clear();
    for (auto it = this->atom_pairs.begin(); it != this->atom_pairs.end(); ++it)
    {
        /**
         * search (rx, ry, rz) with (it->R_values[i*3+0], it->R_values[i*3+1], it->R_values[i*3+2])
         * if (rx, ry, rz) not found in this->tmp_R_index,
         * insert the (rx, ry, rz) into end of this->tmp_R_index
         * no need to sort this->tmp_R_index, using find_R() to find the (rx, ry, rz) -> int in tmp_R_index
         */
        for (int iR = 0; iR < it->get_R_size(); iR++)
        {
            int* R_pointer = it->get_R_index(iR);
            int it_tmp = this->find_R(R_pointer[0], R_pointer[1], R_pointer[2]);
            if (it_tmp == -1)
            {
                this->tmp_R_index.push_back(R_pointer[0]);
                this->tmp_R_index.push_back(R_pointer[1]);
                this->tmp_R_index.push_back(R_pointer[2]);
            }
        }
    }
    return this->tmp_R_index.size() / 3;
}

template <typename T>
void HContainer<T>::loop_R(const size_t& index, int& rx, int& ry, int& rz) const
{
#ifdef __DEBUG
    if (index >= this->tmp_R_index.size() / 3)
    {
        std::cout << "Error: index out of range in loop_R" << std::endl;
        exit(1);
    }
#endif
    // set rx, ry, rz
    rx = this->tmp_R_index[index * 3];
    ry = this->tmp_R_index[index * 3 + 1];
    rz = this->tmp_R_index[index * 3 + 2];
    return;
}

// get_AP_size
template <typename T>
size_t HContainer<T>::size_atom_pairs() const
{
    // R index is fixed
    if (this->current_R > -1)
    {
        return this->tmp_atom_pairs.size();
    }
    // R index is not fixed
    else
    {
        return this->atom_pairs.size();
    }
}

// data() interface with atom_i and atom_j
template <typename T>
T* HContainer<T>::data(int atom_i, int atom_j) const
{
    AtomPair<T>* atom_ij = this->find_pair(atom_i, atom_j);
    if (atom_ij != nullptr)
    {
        return atom_ij->get_pointer();
    }
    else
    {
        std::cout << "Error: atom pair not found in data" << std::endl;
        exit(1);
    }
}

// data() interface with atom_i and atom_j ad R_pointer
template <typename T>
T* HContainer<T>::data(int atom_i, int atom_j, int* R_pointer) const
{
    AtomPair<T>* atom_ij = this->find_pair(atom_i, atom_j);
    if (atom_ij != nullptr)
    {
        return atom_ij->get_HR_values(R_pointer[0], R_pointer[1], R_pointer[2]).get_pointer();
    }
    else
    {
        std::cout << "Error: atom pair not found in data" << std::endl;
        exit(1);
    }
}

// insert_pair
template <typename T>
void HContainer<T>::insert_pair(const AtomPair<T>& atom_ij)
{
    // find atom_ij in this->atom_pairs
    auto it = std::lower_bound(this->atom_pairs.begin(), this->atom_pairs.end(), atom_ij);
    // if found, merge
    if (it != this->atom_pairs.end() && it->identify(atom_ij))
    {
        it->merge(atom_ij, this->gamma_only);
    }
    // if not found, insert
    else
    {
        this->atom_pairs.insert(it, atom_ij);
    }
}

} // end namespace hamilt