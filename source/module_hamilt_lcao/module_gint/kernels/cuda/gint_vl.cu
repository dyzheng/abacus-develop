#include "gint_vl.cuh"
#include "interp.cuh"
#include "cuda_tools.cuh"
#include "sph.cuh"

// __ldg is not available in CUDA < 3.5 compute capability
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 350
#define USE_LDG
#endif

namespace GintKernel
{

__global__ void get_psi_and_vldr3(const double* const ylmcoef,
                                  const double delta_r,
                                  const int bxyz,
                                  const double nwmax,
                                  const double max_atom,
                                  const int* const ucell_atom_nwl,
                                  const bool* const atom_iw2_new,
                                  const int* const atom_iw2_ylm,
                                  const int* const atom_nw,
                                  const double* const rcut,
                                  const int nr_max,
                                  const double* const psi_u,
                                  const double* const mcell_pos,
                                  const double* const dr_part,
                                  const double* const vldr3,
                                  const uint8_t* const atoms_type,
                                  const int* const atoms_num_info,
                                  double* psi,
                                  double* psi_vldr3)
{
    const int bcell_id = blockIdx.x;
    const int num_atoms = atoms_num_info[2 * bcell_id];
    const int pre_atoms = atoms_num_info[2 * bcell_id + 1];
    const int mcell_id = blockIdx.y;
    const double vldr3_value = vldr3[bcell_id * bxyz + mcell_id];
    const double mcell_pos_x = mcell_pos[3 * mcell_id];
    const double mcell_pos_y = mcell_pos[3 * mcell_id + 1];
    const double mcell_pos_z = mcell_pos[3 * mcell_id + 2];

    for(int atom_id = threadIdx.x; atom_id < num_atoms; atom_id += blockDim.x)
    {
        const int dr_start = 3 * (pre_atoms + atom_id);
        const double dr_x = dr_part[dr_start] + mcell_pos_x;
        const double dr_y = dr_part[dr_start + 1] + mcell_pos_y;
        const double dr_z = dr_part[dr_start + 2] + mcell_pos_z;
        double dist = sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z);
#ifdef USE_LDG
        const int atype = __ldg(atoms_type + pre_atoms + atom_id);
#else
        const int atype = atoms_type[pre_atoms + atom_id];
#endif
        if(dist < rcut[atype])
        {
            if (dist < 1.0E-9)
            {
                dist += 1.0E-9;
            }
            double dr[3] = {dr_x / dist, dr_y / dist, dr_z / dist};
            double ylma[49];
#ifdef USE_LDG
            const int nwl = __ldg(ucell_atom_nwl + atype);
#else
            const int nwl = ucell_atom_nwl[atype];
#endif
            spherical_harmonics(dr, nwl, ylma, ylmcoef);
            int psi_idx = (bcell_id * max_atom + atom_id) * bxyz * nwmax + mcell_id;
            interp_vl(dist,
                      delta_r,
                      atype,
                      nwmax,
                      bxyz,
                      nr_max,
                      atom_nw,
                      atom_iw2_new,
                      psi_u,
                      ylma,
                      atom_iw2_ylm,
                      vldr3_value,
                      psi,
                      psi_vldr3,
                      psi_idx);
        }
    }
}

__global__ void extract_vldr3_kernel(const double* __restrict__ d_vlocal,
                                     const int* __restrict__ d_start_ind,
                                     double* vldr3,
                                     const int grid_index_ij,
                                     const int nbzp,
                                     const int bx,
                                     const int by,
                                     const int bz,
                                     const int bxyz,
                                     const int ncy,
                                     const int nczp,
                                     const double vfactor)
{
    // Each thread handles one element of vldr3[nbzp * bxyz]
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nbzp * bxyz;
    if (tid >= total)
        return;

    const int z_index = tid / bxyz;
    const int id_in_z = tid % bxyz;

    // Decompose id_in_z into bx_index, by_index, bz_index
    const int bz_index = id_in_z % bz;
    const int tmp = id_in_z / bz;
    const int by_index = tmp % by;
    const int bx_index = tmp / by;

    const int grid_index = grid_index_ij + z_index;
    const int start_ind_grid = d_start_ind[grid_index];
    const int vindex_global = bx_index * ncy * nczp
                              + by_index * nczp + bz_index
                              + start_ind_grid;

    vldr3[tid] = d_vlocal[vindex_global] * vfactor;
}

} // namespace GintKernel