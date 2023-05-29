#include "hcontainer.h"

namespace hamilt
{

template<typename T>
HContainer::HContainer(
    const UnitCell& ucell_,
    const int strategy
)
{

}

template<typename T>
HContainer::HContainer(
    const UnitCell& ucell_,
    const int strategy,
    const Parallel_Orbitals* paraV
)
{

}

template<typename T>
HContainer::HContainer(
    const UnitCell& ucell_,
    const int strategy,
    const Parallel_Orbitals* paraV,
    const T* whole_matrix
)
{

}

template<typename T>
HContainer::HContainer(
    const HContainer<T>& HR_in,
    const int strategy
)
{

}

template<typename T>
AtomPair<T>* HContainer::find_pair(int atom_i, int atom_j) const
{
    AtomPair<T> target(atom_i, atom_j);
    auto it = std::lower_bound(this->atom_pairs.begin(), this->atom_pairs.end(), target);
    if (it != this->atom_pairs.end() && it->identify(atom_i, atom_j)) {
        return &(*it);
    } else {
        return nullptr;
    }
}

template<typename T>
bool HContainer::fix_R(int rx_in, int ry_in, int rz_in)
{
    //clear and reallocate the memory of this->tmp_atom_pairs
    this->tmp_atom_pairs.clear();
    this->tmp_atom_pairs.shrink_to_fit();
    this->tmp_atom_pairs.reserve(this->atom_pairs.size());

    //find (rx, ry, rz) in this->atom_pairs[i].R_values
    for (auto it = this->atom_pairs.begin(); it != this->atom_pairs.end(); ++it) 
    {
        if (it->find_R(rx_in, ry_in, rz_in)) 
        {
            this->tmp_atom_pairs.push_back(&(*it));
        }
    }
    if(this->tmp_atom_pairs.size() == 0)
    {
        std::cout << "Error: no atom pair found in fix_R" << std::endl;
        return false;
    }
    else 
    {
        return true;
    }
}

template<typename T>
size_t HContainer::get_size_for_loop_R()
{
    /**
     * start a new iteration of loop_R
     * there is different R-index in this->atom_pairs[i].R_values
     * search them one by one and initialize a localized static map in this function to store the R-index
    */
    this->tmp_R_index.clear();
    //init this->tmp_R_index_map
    this->tmp_R_index_map.clear();
    this->tmp_R_index_map.shrink_to_fit();
    for (auto it = this->atom_pairs.begin(); it != this->atom_pairs.end(); ++it) 
    {
        /**
         * search (rx, ry, rz) with (it->R_values[i*3+0], it->R_values[i*3+1], it->R_values[i*3+2])
         * if (rx, ry, rz) not found in this->tmp_R_index, 
         * insert the (rx, ry, rz) into end of this->tmp_R_index
         * no need to sort this->tmp_R_index, using tmp_R_index_map to find the (rx, ry, rz) -> int* in tmp_R_index
         */
        for(int iR =0; iR < it->R_index.size(); iR+=3)
        {
            if (this->tmp_R_index.empty()) 
            {
                this->tmp_R_index.push_back(it->R_index[iR]);
                this->tmp_R_index.push_back(it->R_index[iR+1]);
                this->tmp_R_index.push_back(it->R_index[iR+2]);
                //insert &tmp_R_index[0] to tmp_R_index_map
                this->tmp_R_index_map.insert(
                    std::pair(
                        std::tuple<this->tmp_R_index[0], this->tmp_R_index[1], this->tmp_R_index[2]>, 
                        &this->tmp_R_index[0]
                    )
                );
            } 
            else 
            {
                auto it_tmp = this->tmp_R_index_map.find(
                    std::tuple(it->R_index[iR], it->R_index[iR+1], it->R_index[iR+2])
                );
                if (it_tmp == this->tmp_R_index_map.end()) 
                {
                    const int current_size = this->tmp_R_index.size();
                    this->tmp_R_index.push_back(it->R_index[iR]);
                    this->tmp_R_index.push_back(it->R_index[iR+1]);
                    this->tmp_R_index.push_back(it->R_index[iR+2]);
                    //insert &tmp_R_index[current_size] to tmp_R_index_map
                    this->tmp_R_index_map.insert(
                        std::pair(
                            std::tuple<this->tmp_R_index[current_size], this->tmp_R_index[current_size+1], this->tmp_R_index[current_size+2]>, 
                            &this->tmp_R_index[current_size]
                        )
                    );
                }
            }
        }
    }
    return this->tmp_R_index.size()/3;
}

template<typename T>
void HContainer::loop_R(const size_t &index, int &rx, int &ry, int &rz) const
{
#ifdef __DEBUG
    if (index >= this->tmp_R_index.size()/3) 
    {
        std::cout << "Error: index out of range in loop_R" << std::endl;
        exit(1);
    }
#endif
    //set rx, ry, rz
    rx = this->tmp_R_index[index*3];
    ry = this->tmp_R_index[index*3+1];
    rz = this->tmp_R_index[index*3+2];
    return;
}

template<typename T>
void HContainer::loop_atom_pairs(
    const std::function<void(AtomPair<T>&)>& func
)
{
    for (auto it = this->tmp_atom_pairs.begin(); it != this->tmp_atom_pairs.end(); ++it) 
    {
        func(**it);
    }
}

//get_AP_size
template<typename T>
size_t HContainer::get_AP_size() const
{
    //R index is fixed
    if(this->current_R > -1)
    {
        return this->tmp_atom_pairs.size();
    }
    //R index is not fixed
    else
    {
        return this->atom_pairs.size();
    }
}

//insert_pair
template<typename T>
void HContainer::insert_pair(const AtomPair<T>& atom_ij)
{
    auto it = std::lower_bound(this->atom_pairs.begin(), this->atom_pairs.end(), atom_ij);
    if (it != this->atom_pairs.end()) 
    {
        it->merge(atom_ij);
    } 
    else 
    {
        this->atom_pairs.insert(it, atom_ij);
    }
}