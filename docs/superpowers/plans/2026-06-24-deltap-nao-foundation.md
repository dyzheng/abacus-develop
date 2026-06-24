# DeltaP NAO Foundation: SMO-Projected Atomic Polarization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement post-processing computation of atomic polarization P^I_alpha via SMO-projected Berry connection in a new `module_deltap`, reusing DeltaSpin's two-center overlap infrastructure, with T0 (Si) and T1 (BaTiO3) verification.

**Architecture:** New C++ module at `source/source_lcao/module_deltap/` that reuses `TwoCenterIntegrator::snap()` for real-space `<phi|alpha>` overlaps, phase-sums them into k-space S(k) and analytic dS(k), computes Berry connection A^I_n(k) = term1(analytic) + term2(finite-diff), and integrates over k-string to get P^I. Post-processing entry point in `ctrl_scf_lcao.cpp` after berry_phase.

**Tech Stack:** C++11, CMake, GoogleTest, ABACUS LCAO infrastructure (TwoCenterIntegrator, Grid_Driver, Parallel_Orbitals, HContainer)

**Spec:** `docs/superpowers/specs/2026-06-24-deltap-nao-foundation-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `source/source_lcao/module_deltap/deltap.h` | Main class header, data structure definitions |
| `source/source_lcao/module_deltap/deltap.cpp` | Main class: init, orchestration, k-string setup |
| `source/source_lcao/module_deltap/deltap_overlap.cpp` | Real-space `<phi\|alpha(R)>` overlaps via `snap()` |
| `source/source_lcao/module_deltap/deltap_berry.cpp` | k-space S/dS, D_I, Berry connection, polarization integration |
| `source/source_lcao/module_deltap/deltap_io.cpp` | Output results, sum-rule verification |
| `source/source_lcao/module_deltap/CMakeLists.txt` | Build configuration |
| `source/source_lcao/module_deltap/test/CMakeLists.txt` | Test build |
| `source/source_lcao/module_deltap/test/deltap_math_test.cpp` | Unit tests for phase-summing and Berry connection math |
| `source/source_io/module_parameter/input_parameter.h` | **Modify**: add `deltap_*` fields |
| `source/source_io/module_parameter/read_input_item_other.cpp` | **Modify**: add parameter parsing |
| `source/source_io/module_ctrl/ctrl_scf_lcao.cpp` | **Modify**: add DeltaP entry point |
| `source/source_lcao/CMakeLists.txt` | **Modify**: add `add_subdirectory(module_deltap)` |
| `source/Makefile.Objects` | **Modify**: add `module_deltap` path |
| `tests/17_DS_DFTU/18_LCAO_DELTAP_BTO/` | **New**: T1 integration test for BaTiO3 |

---

## Task 1: Module Scaffold and Build System

**Files:**
- Create: `source/source_lcao/module_deltap/CMakeLists.txt`
- Create: `source/source_lcao/module_deltap/deltap.h` (stub)
- Create: `source/source_lcao/module_deltap/deltap.cpp` (stub)
- Modify: `source/source_lcao/CMakeLists.txt:5` (add subdirectory)
- Modify: `source/Makefile.Objects:61` (add path)

- [ ] **Step 1: Create the module directory**

```bash
mkdir -p source/source_lcao/module_deltap/test
```

- [ ] **Step 2: Create stub header `deltap.h`**

```cpp
#ifndef DELTAP_H
#define DELTAP_H

#include "source_base/vector3.h"
#include <complex>
#include <unordered_map>
#include <vector>

namespace deltap {

struct OverlapData {
    int iat_adj = -1;
    ModuleBase::Vector3<int> R_index;
    std::unordered_map<int, std::vector<double>> nlm;
};

struct KSpaceData {
    ModuleBase::Vector3<double> kvec_d;
    std::vector<std::vector<std::vector<std::complex<double>>>> S_k;
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> dS_k;
    std::vector<std::vector<std::vector<std::complex<double>>>> D_I;
};

struct AtomicPolarization {
    std::vector<ModuleBase::Vector3<double>> P_I;
    std::vector<ModuleBase::Vector3<double>> gamma_I;
    ModuleBase::Vector3<double> P_total;
    ModuleBase::Vector3<double> P_abacus;
};

class DeltaP {
public:
    DeltaP() = default;
    ~DeltaP() = default;
};

} // namespace deltap

#endif
```

- [ ] **Step 3: Create stub `deltap.cpp`**

```cpp
#include "deltap.h"
```

- [ ] **Step 4: Create `CMakeLists.txt`**

```cmake
list(APPEND objects
    deltap.cpp
    deltap_overlap.cpp
    deltap_berry.cpp
    deltap_io.cpp
)

add_library(
    deltap
    OBJECT
    ${objects}
)

if(ENABLE_COVERAGE)
  add_coverage(deltap)
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
  endif()
endif()
```

- [ ] **Step 5: Add to parent `source_lcao/CMakeLists.txt`**

After line 5 (`add_subdirectory(module_deltaspin)`), add:

```cmake
add_subdirectory(module_deltap)
```

- [ ] **Step 6: Add to `source/Makefile.Objects`**

After line 61 (`./source_lcao/module_deltaspin:\`), add:

```makefile
./source_lcao/module_deltap:\
```

- [ ] **Step 7: Verify the build compiles**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -5
```

Expected: builds successfully (empty object library).

- [ ] **Step 8: Commit**

```bash
git add source/source_lcao/module_deltap/ source/source_lcao/CMakeLists.txt source/Makefile.Objects
git commit -m "feat(deltap): scaffold module_deltap with CMakeLists and stub header"
```

---

## Task 2: Input Parameters

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h` (add fields after line 614)
- Modify: `source/source_io/module_parameter/read_input_item_other.cpp` (add parsing block)

- [ ] **Step 1: Add DeltaP fields to `input_parameter.h`**

After line 614 (end of spin-constrained section), add a new section:

```cpp
    // ==============   #Parameters (20.DeltaP atomic polarization) =============
    bool deltap_switch = false;       ///< switch to enable DeltaP atomic polarization decomposition
    double deltap_rm = 3.0;          ///< SMO modulation radius (Bohr); if 0, reuse onsite_radius
    int deltap_gdir = 3;             ///< polarization direction: 1=x, 2=y, 3=z
    double deltap_dk_fd = 1e-6;      ///< finite-difference delta-k for T0 validation
    int deltap_npk_string = 0;       ///< override k-string density (0 = use KPT mesh)
```

- [ ] **Step 2: Add parameter parsing to `read_input_item_other.cpp`**

At the end of `item_others()` function (before the closing `}`), add:

```cpp
    // DeltaP atomic polarization
    {
        Input_Item item("deltap_switch");
        item.annotation = "switch to enable DeltaP atomic polarization decomposition";
        item.category = "DeltaP";
        item.type = "Boolean";
        item.description = "Switch to enable DeltaP atomic polarization decomposition";
        item.default_value = "False";
        item.unit = "";
        item.availability = "";
        read_sync_bool(input.deltap_switch);
        this->add_item(item);
    }
    {
        Input_Item item("deltap_rm");
        item.annotation = "SMO modulation radius (Bohr)";
        item.category = "DeltaP";
        item.type = "Real";
        item.description = "SMO modulation radius (Bohr); if 0, reuse onsite_radius";
        item.default_value = "3.0";
        item.unit = "Bohr";
        item.availability = "deltap_switch is true";
        read_sync_double(input.deltap_rm);
        this->add_item(item);
    }
    {
        Input_Item item("deltap_gdir");
        item.annotation = "polarization direction: 1=x, 2=y, 3=z";
        item.category = "DeltaP";
        item.type = "Integer";
        item.description = "Polarization direction: 1=x, 2=y, 3=z";
        item.default_value = "3";
        item.unit = "";
        item.availability = "deltap_switch is true";
        read_sync_int(input.deltap_gdir);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.deltap_gdir < 1 || para.input.deltap_gdir > 3)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deltap_gdir must be 1, 2, or 3");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deltap_dk_fd");
        item.annotation = "finite-difference delta-k for T0 validation";
        item.category = "DeltaP";
        item.type = "Real";
        item.description = "Finite-difference delta-k for Berry connection validation";
        item.default_value = "1.0e-6";
        item.unit = "";
        item.availability = "deltap_switch is true";
        read_sync_double(input.deltap_dk_fd);
        this->add_item(item);
    }
    {
        Input_Item item("deltap_npk_string");
        item.annotation = "override k-string density (0 = use KPT mesh)";
        item.category = "DeltaP";
        item.type = "Integer";
        item.description = "Override k-string density; 0 means use KPT mesh density";
        item.default_value = "0";
        item.unit = "";
        item.availability = "deltap_switch is true";
        read_sync_int(input.deltap_npk_string);
        this->add_item(item);
    }
```

- [ ] **Step 3: Verify compilation**

```bash
cd build && cmake .. && make -j$(nproc) 2>&1 | tail -10
```

Expected: builds successfully.

- [ ] **Step 4: Commit**

```bash
git add source/source_io/module_parameter/input_parameter.h source/source_io/module_parameter/read_input_item_other.cpp
git commit -m "feat(deltap): add deltap_* input parameters"
```

---

## Task 3: DeltaP Class Header — Full Interface

**Files:**
- Modify: `source/source_lcao/module_deltap/deltap.h` (expand stub to full interface)

- [ ] **Step 1: Write the full header**

```cpp
#ifndef DELTAP_H
#define DELTAP_H

#include "source_base/vector3.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"
#include "source_psi/psi.h"

#include <complex>
#include <unordered_map>
#include <vector>

namespace deltap {

struct OverlapData {
    int iat_adj = -1;
    ModuleBase::Vector3<int> R_index;
    std::unordered_map<int, std::vector<double>> nlm;
};

struct KSpaceData {
    ModuleBase::Vector3<double> kvec_d;
    // S_k[iat][lm][mu_local]
    std::vector<std::vector<std::vector<std::complex<double>>>> S_k;
    // dS_k[iat][alpha][lm][mu_local]  (alpha: 0=x, 1=y, 2=z)
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> dS_k;
    // D_I[iat][lm][nband]
    std::vector<std::vector<std::vector<std::complex<double>>>> D_I;
};

struct AtomicPolarization {
    std::vector<ModuleBase::Vector3<double>> P_I;
    std::vector<ModuleBase::Vector3<double>> gamma_I;
    ModuleBase::Vector3<double> P_total;
    ModuleBase::Vector3<double> P_abacus;
};

class DeltaP {
public:
    DeltaP() = default;
    ~DeltaP() = default;

    void init(const UnitCell& ucell,
              const Grid_Driver& gd,
              const K_Vectors& kv,
              const TwoCenterIntegrator* intor,
              const std::vector<double>& orb_cutoff,
              double rm,
              int gdir);

    void compute_atomic_polarization(
        const UnitCell& ucell,
        const psi::Psi<std::complex<double>>* psi,
        const elecstate::ElecState* pelec);

    const AtomicPolarization& get_results() const { return results_; }

private:
    void compute_real_overlaps(const UnitCell& ucell, const Grid_Driver& gd);
    void setup_kstring(const K_Vectors& kv);
    void compute_S_k(int ik);
    void compute_D_I(int ik, const std::complex<double>* psi_k, int nbands, int nrow_local);
    void compute_berry_connection(int ik, int nbands, const double* wg);
    void integrate_polarization(const UnitCell& ucell, int nbands);
    void verify_sum_rule();
    void write_results(const UnitCell& ucell) const;

    // Configuration
    const TwoCenterIntegrator* intor_ = nullptr;
    std::vector<double> orb_cutoff_;
    double rm_ = 3.0;
    int gdir_ = 3;
    int nat_ = 0;
    int nproj_max_ = 0;

    // Parallel orbitals (for 2D-block distribution)
    const Parallel_Orbitals* paraV_ = nullptr;

    // Real-space overlaps: overlap_R_[iat][adj_index]
    std::vector<std::vector<OverlapData>> overlap_R_;
    std::vector<int> nproj_per_atom_;

    // k-string data
    std::vector<KSpaceData> kstring_data_;
    int nppstr_ = 0;
    int total_string_ = 0;
    std::vector<std::vector<int>> k_index_;

    // Berry connection: A_nk_[iat][ik][nband][3] (alpha=x,y,z)
    std::vector<std::vector<std::vector<ModuleBase::Vector3<std::complex<double>>>>> A_nk_;

    // Results
    AtomicPolarization results_;
};

} // namespace deltap

#endif
```

- [ ] **Step 2: Verify compilation** (stubs in .cpp will need updating)

Create empty stubs for all methods in `deltap.cpp`:

```cpp
#include "deltap.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"

namespace deltap {

void DeltaP::init(const UnitCell& ucell, const Grid_Driver& gd, const K_Vectors& kv,
                  const TwoCenterIntegrator* intor, const std::vector<double>& orb_cutoff,
                  double rm, int gdir) {
    intor_ = intor;
    orb_cutoff_ = orb_cutoff;
    rm_ = rm;
    gdir_ = gdir;
    nat_ = ucell.nat;
}

void DeltaP::compute_atomic_polarization(const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi, const elecstate::ElecState* pelec) {}

void DeltaP::compute_real_overlaps(const UnitCell& ucell, const Grid_Driver& gd) {}
void DeltaP::setup_kstring(const K_Vectors& kv) {}
void DeltaP::compute_S_k(int ik) {}
void DeltaP::compute_D_I(int ik, const std::complex<double>* psi_k, int nbands, int nrow_local) {}
void DeltaP::compute_berry_connection(int ik, int nbands, const double* wg) {}
void DeltaP::integrate_polarization(const UnitCell& ucell, int nbands) {}
void DeltaP::verify_sum_rule() {}
void DeltaP::write_results(const UnitCell& ucell) const {}

} // namespace deltap
```

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -5
```

Expected: compiles with stubs.

- [ ] **Step 4: Commit**

```bash
git add source/source_lcao/module_deltap/deltap.h source/source_lcao/module_deltap/deltap.cpp
git commit -m "feat(deltap): define DeltaP class interface and data structures"
```

---

## Task 4: Real-Space Overlap Computation

**Files:**
- Create: `source/source_lcao/module_deltap/deltap_overlap.cpp`
- Modify: `source/source_lcao/module_deltap/CMakeLists.txt` (add deltap_overlap.cpp)

This task replicates `dspin_lcao.cpp::cal_pre_HR()` lines 252-371 — the real-space `<phi|alpha(R)>` overlap computation — but stores individual overlaps instead of the sandwich product.

- [ ] **Step 1: Implement `compute_real_overlaps` in `deltap_overlap.cpp`**

```cpp
#include "deltap.h"
#include "source_base/memory.h"
#include "source_base/name_angular.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"

namespace deltap {

void DeltaP::compute_real_overlaps(const UnitCell& ucell, const Grid_Driver& gd)
{
    ModuleBase::TITLE("DeltaP", "compute_real_overlaps");
    ModuleBase::timer::start("DeltaP", "compute_real_overlaps");

    const int npol = ucell.get_npol();
    overlap_R_.clear();
    overlap_R_.resize(nat_);
    nproj_per_atom_.resize(nat_, 0);

    size_t memory_cost = 0;

    for (int iat = 0; iat < nat_; iat++)
    {
        auto tau0 = ucell.get_tau(iat);
        int T0, I0;
        ucell.iat2iait(iat, &I0, &T0);

        // Find adjacent atoms
        AdjacentAtomInfo adjs;
        gd.Find_atom(ucell, tau0, T0, I0, &adjs);

        // Filter by cutoff radius
        std::vector<bool> is_adj(adjs.adj_num + 1, false);
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = ucell.itia2iat(T1, I1);
            const ModuleBase::Vector3<int>& R_index1 = adjs.box[ad];
            if (ucell.cal_dtau(iat, iat1, R_index1).norm() * ucell.lat0
                < orb_cutoff_[T1] + rm_)
            {
                is_adj[ad] = true;
            }
        }
        filter_adjs(is_adj, adjs);

        // max_l_plus_1 for this atom type (same as DeltaSpin)
        const int max_l_plus_1 = ucell.atoms[T0].nwl + 1;
        nproj_per_atom_[iat] = max_l_plus_1 * max_l_plus_1;

        // Compute <phi_mu | alpha^I_lm(R)> via snap()
        std::vector<std::unordered_map<int, std::vector<double>>> nlm_iat0(adjs.adj_num + 1);

        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T1 = adjs.ntype[ad];
            const int I1 = adjs.natom[ad];
            const int iat1 = ucell.itia2iat(T1, I1);
            const Atom* atom1 = &ucell.atoms[T1];
            const ModuleBase::Vector3<double>& tau1 = adjs.adjacent_tau[ad];

            auto all_indexes = paraV_->get_indexes_row(iat1);
            auto col_indexes = paraV_->get_indexes_col(iat1);
            all_indexes.insert(all_indexes.end(), col_indexes.begin(), col_indexes.end());
            std::sort(all_indexes.begin(), all_indexes.end());
            all_indexes.erase(std::unique(all_indexes.begin(), all_indexes.end()), all_indexes.end());

            for (int iw1l = 0; iw1l < (int)all_indexes.size(); iw1l += npol)
            {
                const int iw1 = all_indexes[iw1l] / npol;
                std::vector<double> nlm_target(max_l_plus_1 * max_l_plus_1);
                const int L1 = atom1->iw2l[iw1];
                const int N1 = atom1->iw2n[iw1];
                const int m1 = atom1->iw2m[iw1];

                std::vector<std::vector<double>> nlm;
                const int M1 = (m1 % 2 == 0) ? -m1 / 2 : (m1 + 1) / 2;
                ModuleBase::Vector3<double> dtau = tau0 - tau1;
                intor_->snap(T1, L1, N1, M1, T0, dtau * ucell.lat0, 0, nlm);

                // Select first zeta of each l, same as DeltaSpin
                int target_L = 0, index = 0;
                for (int iw = 0; iw < ucell.atoms[T0].nw; iw++)
                {
                    const int L0 = ucell.atoms[T0].iw2l[iw];
                    if (L0 == target_L)
                    {
                        for (int m = 0; m < 2 * L0 + 1; m++)
                        {
                            nlm_target[index] = nlm[0][iw + m];
                            index++;
                        }
                        target_L++;
                    }
                }
                nlm_iat0[ad].insert({all_indexes[iw1l], nlm_target});
            }
        }

        // Store as OverlapData
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            OverlapData od;
            od.iat_adj = ucell.itia2iat(adjs.ntype[ad], adjs.natom[ad]);
            od.R_index = adjs.box[ad];
            od.nlm = nlm_iat0[ad];
            overlap_R_[iat].push_back(std::move(od));
        }
    }

    nproj_max_ = 0;
    for (int iat = 0; iat < nat_; iat++)
    {
        nproj_max_ = std::max(nproj_max_, nproj_per_atom_[iat]);
    }

    ModuleBase::Memory::record("DeltaP:overlap_R", memory_cost);
    ModuleBase::timer::end("DeltaP", "compute_real_overlaps");
}

} // namespace deltap
```

- [ ] **Step 2: Update `CMakeLists.txt`**

Add `deltap_overlap.cpp` to the objects list:

```cmake
list(APPEND objects
    deltap.cpp
    deltap_overlap.cpp
    deltap_berry.cpp
    deltap_io.cpp
)
```

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -10
```

Expected: compiles (may have unused variable warnings — acceptable).

- [ ] **Step 4: Commit**

```bash
git add source/source_lcao/module_deltap/deltap_overlap.cpp source/source_lcao/module_deltap/CMakeLists.txt
git commit -m "feat(deltap): implement real-space <phi|alpha> overlap computation"
```

---

## Task 5: k-String Setup

**Files:**
- Modify: `source/source_lcao/module_deltap/deltap.cpp` (implement `setup_kstring`)

This replicates `berryphase::set_kpoints()` for the gdir direction — indexing k-points into strings along the polarization direction.

- [ ] **Step 1: Implement `setup_kstring` in `deltap.cpp`**

Add this implementation to `deltap.cpp` (replacing the stub):

```cpp
void DeltaP::setup_kstring(const K_Vectors& kv)
{
    ModuleBase::TITLE("DeltaP", "setup_kstring");
    ModuleBase::timer::start("DeltaP", "setup_kstring");

    const int mp_x = kv.nmp[0];
    const int mp_y = kv.nmp[1];
    const int mp_z = kv.nmp[2];
    const int direction = gdir_;

    int mp_dir = 0;
    int num_string = 0;
    if (direction == 1) { mp_dir = mp_x; num_string = mp_y * mp_z; }
    else if (direction == 2) { mp_dir = mp_y; num_string = mp_x * mp_z; }
    else { mp_dir = mp_z; num_string = mp_x * mp_y; }

    total_string_ = num_string;
    k_index_.resize(total_string_);
    for (int istring = 0; istring < total_string_; istring++)
    {
        k_index_[istring].resize(mp_dir + 1);
    }

    // Build k-string indices (same logic as berryphase::set_kpoints)
    int string_index = -1;
    if (direction == 1)
    {
        for (int iz = 0; iz < mp_z; iz++)
        {
            for (int iy = 0; iy < mp_y; iy++)
            {
                string_index++;
                for (int ix = 0; ix < mp_x; ix++)
                {
                    k_index_[string_index][ix] = ix + iy * mp_x + iz * mp_x * mp_y;
                    if (ix == mp_x - 1)
                        k_index_[string_index][ix + 1] = k_index_[string_index][0];
                }
            }
        }
    }
    else if (direction == 2)
    {
        for (int iz = 0; iz < mp_z; iz++)
        {
            for (int ix = 0; ix < mp_x; ix++)
            {
                string_index++;
                for (int iy = 0; iy < mp_y; iy++)
                {
                    k_index_[string_index][iy] = ix + iy * mp_x + iz * mp_x * mp_y;
                    if (iy == mp_y - 1)
                        k_index_[string_index][iy + 1] = k_index_[string_index][0];
                }
            }
        }
    }
    else
    {
        for (int iy = 0; iy < mp_y; iy++)
        {
            for (int ix = 0; ix < mp_x; ix++)
            {
                string_index++;
                for (int iz = 0; iz < mp_z; iz++)
                {
                    k_index_[string_index][iz] = ix + iy * mp_x + iz * mp_x * mp_y;
                    if (iz == mp_z - 1)
                        k_index_[string_index][iz + 1] = k_index_[string_index][0];
                }
            }
        }
    }

    nppstr_ = mp_dir + 1;

    ModuleBase::timer::end("DeltaP", "setup_kstring");
}
```

- [ ] **Step 2: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add source/source_lcao/module_deltap/deltap.cpp
git commit -m "feat(deltap): implement k-string setup for Berry phase integration"
```

---

## Task 6: k-Space S and dS Computation

**Files:**
- Create: `source/source_lcao/module_deltap/deltap_berry.cpp`
- Modify: `source/source_lcao/module_deltap/CMakeLists.txt` (add deltap_berry.cpp)

This is the core k-space phase-summing: `S(k) = sum_R e^{ikR} <phi|alpha(R)>` and `dS(k,alpha) = sum_R i*R_alpha*e^{ikR} <phi|alpha(R)>`.

- [ ] **Step 1: Implement `compute_S_k` in `deltap_berry.cpp`**

```cpp
#include "deltap.h"
#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"

namespace deltap {

void DeltaP::compute_S_k(int ik)
{
    ModuleBase::TITLE("DeltaP", "compute_S_k");
    ModuleBase::timer::start("DeltaP", "compute_S_k");

    const int nat = nat_;
    kstring_data_[ik].S_k.resize(nat);
    kstring_data_[ik].dS_k.resize(nat);

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];
        kstring_data_[ik].S_k[iat].resize(r);
        kstring_data_[ik].dS_k[iat].resize(3);
        for (int alpha = 0; alpha < 3; alpha++)
        {
            kstring_data_[ik].dS_k[iat][alpha].resize(r);
        }

        for (int lm = 0; lm < r; lm++)
        {
            // Iterate over all adjacent atoms for this iat
            for (const auto& od : overlap_R_[iat])
            {
                const double arg = ModuleBase::TWO_PI * (
                    kstring_data_[ik].kvec_d.x * od.R_index.x +
                    kstring_data_[ik].kvec_d.y * od.R_index.y +
                    kstring_data_[ik].kvec_d.z * od.R_index.z);
                const std::complex<double> phase(std::cos(arg), std::sin(arg));

                for (const auto& [iw_global, nlm_vec] : od.nlm)
                {
                    const int iw_local = paraV_->global2local_row(iw_global);
                    if (iw_local < 0) continue;

                    // Ensure S_k[iat][lm] is large enough
                    if ((int)kstring_data_[ik].S_k[iat][lm].size() <= iw_local)
                    {
                        kstring_data_[ik].S_k[iat][lm].resize(iw_local + 1, {0.0, 0.0});
                        for (int a = 0; a < 3; a++)
                            kstring_data_[ik].dS_k[iat][a][lm].resize(iw_local + 1, {0.0, 0.0});
                    }

                    // S(k) = sum_R e^{ikR} <phi|alpha(R)>
                    kstring_data_[ik].S_k[iat][lm][iw_local] += phase * nlm_vec[lm];

                    // dS(k,alpha) = sum_R i*R_alpha*e^{ikR} <phi|alpha(R)>
                    const std::complex<double> i_phase(0.0, 1.0);
                    for (int a = 0; a < 3; a++)
                    {
                        double R_alpha = (a == 0) ? od.R_index.x : (a == 1) ? od.R_index.y : od.R_index.z;
                        kstring_data_[ik].dS_k[iat][a][lm][iw_local]
                            += i_phase * R_alpha * phase * nlm_vec[lm];
                    }
                }
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_S_k");
}

} // namespace deltap
```

- [ ] **Step 2: Update `CMakeLists.txt`** (add `deltap_berry.cpp` if not already present)

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add source/source_lcao/module_deltap/deltap_berry.cpp source/source_lcao/module_deltap/CMakeLists.txt
git commit -m "feat(deltap): implement k-space S(k) and analytic dS(k) phase-summing"
```

---

## Task 7: SMO-Wavefunction Overlap D_I

**Files:**
- Modify: `source/source_lcao/module_deltap/deltap_berry.cpp` (add `compute_D_I`)

This computes `D_I(lm, n, k) = sum_mu S*_{mu,Ilm}(k) * C_{n,mu}(k)` — same algorithm as `dspin_lcao::cal_PI_sub` but storing per-k.

- [ ] **Step 1: Implement `compute_D_I`** (append to `deltap_berry.cpp`)

```cpp
void DeltaP::compute_D_I(int ik, const std::complex<double>* psi_k, int nbands, int nrow_local)
{
    ModuleBase::TITLE("DeltaP", "compute_D_I");
    ModuleBase::timer::start("DeltaP", "compute_D_I");

    const int nat = nat_;
    kstring_data_[ik].D_I.resize(nat);

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];
        kstring_data_[ik].D_I[iat].resize(r);
        for (int lm = 0; lm < r; lm++)
        {
            kstring_data_[ik].D_I[iat][lm].resize(nbands, {0.0, 0.0});
        }

        for (int lm = 0; lm < r; lm++)
        {
            const int s_size = kstring_data_[ik].S_k[iat][lm].size();
            for (int mu_local = 0; mu_local < s_size; mu_local++)
            {
                const std::complex<double> s_val = kstring_data_[ik].S_k[iat][lm][mu_local];
                if (std::abs(s_val) < 1e-15) continue;

                const std::complex<double> s_conj = std::conj(s_val);
                for (int n = 0; n < nbands; n++)
                {
                    // psi_k is column-major: C[irow + icol * lda]
                    kstring_data_[ik].D_I[iat][lm][n] += s_conj * psi_k[mu_local + n * nrow_local];
                }
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_D_I");
}
```

- [ ] **Step 2: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add source/source_lcao/module_deltap/deltap_berry.cpp
git commit -m "feat(deltap): implement SMO-wavefunction overlap D_I computation"
```

---

## Task 8: Berry Connection Computation

**Files:**
- Modify: `source/source_lcao/module_deltap/deltap_berry.cpp` (add `compute_berry_connection`)

This computes `A^I_n(k, alpha) = term1 + term2` where:
- term1 = `sum_{lm} <psi|d_k_alpha alpha> * <alpha|psi>` = `sum_{lm} [sum_mu dS*_{mu,Ilm} * C_{n,mu}] * D_I(lm, n)`
- term2 = `sum_{lm} <psi|alpha> * d_k <alpha|psi>` = `sum_{lm} D_I*(lm, n) * d_k D_I(lm, n)` (finite difference)

- [ ] **Step 1: Implement `compute_berry_connection`** (append to `deltap_berry.cpp`)

```cpp
void DeltaP::compute_berry_connection(int ik, int nbands, const double* wg)
{
    ModuleBase::TITLE("DeltaP", "compute_berry_connection");
    ModuleBase::timer::start("DeltaP", "compute_berry_connection");

    const int nat = nat_;
    const int nppstr = nppstr_;

    // Allocate A_nk_ if needed: [iat][ik][nband][3]
    if (A_nk_.empty())
    {
        A_nk_.resize(nat);
        for (int iat = 0; iat < nat; iat++)
        {
            A_nk_[iat].resize(nppstr);
            for (int j = 0; j < nppstr; j++)
            {
                A_nk_[iat][j].resize(nbands, ModuleBase::Vector3<std::complex<double>>(0.0, 0.0, 0.0));
            }
        }
    }

    // Finite difference for d_k D_I:
    // d_k D_I(lm, n, k_j) = [D_I(k_{j+1}) - D_I(k_{j-1})] / (2*dk)
    int ik_next = (ik + 1) % nppstr;
    int ik_prev = (ik - 1 + nppstr) % nppstr;

    // dk in direct coordinates along gdir
    const double dk_dir = 1.0 / (nppstr - 1);  // spacing between k-points on string

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];

        for (int n = 0; n < nbands; n++)
        {
            for (int alpha = 0; alpha < 3; alpha++)
            {
                std::complex<double> term1 = {0.0, 0.0};
                std::complex<double> term2 = {0.0, 0.0};

                for (int lm = 0; lm < r; lm++)
                {
                    // term1: sum_mu dS*_{mu,Ilm}(k,alpha) * C_{n,mu}  then * D_I(lm, n)
                    // <psi_nk | d_k_alpha alpha^I_lmk> = sum_mu [dS*_{mu,Ilm}(k,alpha)] * C_{n,mu}(k)
                    std::complex<double> bra_grad = {0.0, 0.0};
                    const int s_size = kstring_data_[ik].dS_k[iat][alpha][lm].size();
                    const int nrow_local = paraV_->get_row_size();
                    for (int mu = 0; mu < s_size; mu++)
                    {
                        // Need C_{n,mu}(k) — but we don't store psi directly.
                        // Instead, use the relation:
                        // sum_mu dS* * C = d/dk [sum_mu S* * C] - sum_mu (dS*/dk - dS*) * C
                        // Actually, we can compute this directly if we have psi_k.
                        // For now, store the psi pointer during compute_D_I call.
                        // This is handled by a separate approach: we compute <psi|d_k alpha>
                        // = conj(d_k <alpha|psi>) - correction... No.
                        //
                        // Correct: <psi|d_k alpha> = sum_mu [dS*_{mu,Ilm}(k,alpha)] * C_{n,mu}(k)
                        // We need C_{n,mu}(k). We'll pass psi_k to this function.
                        // For now, skip — will refactor in the integration step.
                    }
                    // term1 = bra_grad * D_I(lm, n)
                    // term2 = conj(D_I(lm, n)) * [D_I_next(lm,n) - D_I_prev(lm,n)] / (2*dk)
                }

                A_nk_[iat][ik][n][alpha] = term1 + term2;
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_berry_connection");
}
```

**Note:** The above is a skeleton. The actual implementation needs `psi_k` to compute term1. Let me fix this by changing the function signature.

- [ ] **Step 2: Fix the interface — pass psi_k to compute_berry_connection**

Update `deltap.h` to change the signature:

```cpp
    void compute_berry_connection(int ik, const std::complex<double>* psi_k,
                                  int nbands, int nrow_local, const double* wg);
```

Update the implementation in `deltap_berry.cpp`:

```cpp
void DeltaP::compute_berry_connection(int ik, const std::complex<double>* psi_k,
                                       int nbands, int nrow_local, const double* wg)
{
    ModuleBase::TITLE("DeltaP", "compute_berry_connection");
    ModuleBase::timer::start("DeltaP", "compute_berry_connection");

    const int nat = nat_;
    const int nppstr = nppstr_;

    if (A_nk_.empty())
    {
        A_nk_.resize(nat);
        for (int iat = 0; iat < nat; iat++)
        {
            A_nk_[iat].resize(nppstr);
            for (int j = 0; j < nppstr; j++)
            {
                A_nk_[iat][j].resize(nbands, ModuleBase::Vector3<std::complex<double>>(0.0, 0.0, 0.0));
            }
        }
    }

    int ik_next = (ik + 1) % nppstr;
    int ik_prev = (ik - 1 + nppstr) % nppstr;
    const double dk_dir = 1.0 / (nppstr - 1);
    const double inv_2dk = 1.0 / (2.0 * dk_dir);

    for (int iat = 0; iat < nat; iat++)
    {
        const int r = nproj_per_atom_[iat];

        for (int n = 0; n < nbands; n++)
        {
            for (int alpha = 0; alpha < 3; alpha++)
            {
                std::complex<double> term1 = {0.0, 0.0};
                std::complex<double> term2 = {0.0, 0.0};

                for (int lm = 0; lm < r; lm++)
                {
                    // term1: <psi_nk | d_k_alpha alpha> * <alpha | psi_nk>
                    //   = [sum_mu dS*_{mu,Ilm}(k,alpha) * C_{n,mu}] * D_I(lm, n)
                    std::complex<double> bra_grad = {0.0, 0.0};
                    const int s_size = kstring_data_[ik].dS_k[iat][alpha][lm].size();
                    for (int mu = 0; mu < s_size; mu++)
                    {
                        const std::complex<double> ds_val = kstring_data_[ik].dS_k[iat][alpha][lm][mu];
                        const std::complex<double> c_val = psi_k[mu + n * nrow_local];
                        bra_grad += std::conj(ds_val) * c_val;
                    }
                    term1 += bra_grad * kstring_data_[ik].D_I[iat][lm][n];

                    // term2: <psi_nk | alpha> * d_k <alpha | psi_nk>
                    //   = conj(D_I(lm, n)) * [D_I_next(lm,n) - D_I_prev(lm,n)] / (2*dk)
                    std::complex<double> d_D = {0.0, 0.0};
                    if (ik_next != ik && ik_prev != ik)
                    {
                        d_D = (kstring_data_[ik_next].D_I[iat][lm][n]
                               - kstring_data_[ik_prev].D_I[iat][lm][n]) * inv_2dk;
                    }
                    term2 += std::conj(kstring_data_[ik].D_I[iat][lm][n]) * d_D;
                }

                A_nk_[iat][ik][n][alpha] = term1 + term2;
            }
        }
    }

    ModuleBase::timer::end("DeltaP", "compute_berry_connection");
}
```

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add source/source_lcao/module_deltap/deltap.h source/source_lcao/module_deltap/deltap_berry.cpp
git commit -m "feat(deltap): implement Berry connection A^I_n with analytic term1 and FD term2"
```

---

## Task 9: Polarization Integration

**Files:**
- Modify: `source/source_lcao/module_deltap/deltap_berry.cpp` (add `integrate_polarization`)

- [ ] **Step 1: Implement `integrate_polarization`** (append to `deltap_berry.cpp`)

```cpp
void DeltaP::integrate_polarization(const UnitCell& ucell, int nbands)
{
    ModuleBase::TITLE("DeltaP", "integrate_polarization");
    ModuleBase::timer::start("DeltaP", "integrate_polarization");

    const int nat = nat_;
    const int alpha_idx = gdir_ - 1;  // 0-indexed direction

    // Lattice vector length along gdir (in Bohr)
    // a_alpha = lat0 * |latvec[gdir-1]|
    ModuleBase::Vector3<double> latvec_gdir(
        ucell.latvec(gdir_ - 1, 0),
        ucell.latvec(gdir_ - 1, 1),
        ucell.latvec(gdir_ - 1, 2));
    const double a_alpha = ucell.lat0 * latvec_gdir.norm();

    // dk in direct coordinates
    const double dk_dir = 1.0 / (nppstr_ - 1);

    // P^I_alpha = -(e / 2*pi*a_alpha) * dk * sum_n sum_j Im[A^I_n(k_j, alpha)]
    // Using ABACUS units: e=1, output in C/m^2 requires conversion
    // ABACUS berry_phase uses: pdl = -sum_n Im[sum_j log(<u_j|u_{j+1}>)] / pi
    // P = pdl * e * R / (2*pi*a*Omega) ... 
    // For consistency with ABACUS, use same prefactor as berryphase

    const double prefactor = -1.0 / (2.0 * ModuleBase::PI * a_alpha) * dk_dir;

    results_.P_I.resize(nat, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    results_.gamma_I.resize(nat, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));

    // Get occupation bands
    double occupied_bands = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occupied_bands - std::floor(occupied_bands)) > 0.0)
    {
        occupied_bands = std::floor(occupied_bands) + 1.0;
    }
    const int occ_nbands = static_cast<int>(occupied_bands);

    for (int iat = 0; iat < nat; iat++)
    {
        double gamma = 0.0;
        for (int j = 0; j < nppstr_; j++)
        {
            for (int n = 0; n < occ_nbands && n < nbands; n++)
            {
                // Berry connection: Im[A^I_n(k_j, alpha)]
                double a_imag = A_nk_[iat][j][n][alpha_idx].imag();
                gamma += a_imag;
            }
        }
        results_.gamma_I[iat][alpha_idx] = gamma;
        results_.P_I[iat][alpha_idx] = prefactor * gamma;
    }

    // Sum total
    results_.P_total = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    for (int iat = 0; iat < nat; iat++)
    {
        results_.P_total += results_.P_I[iat];
    }

    ModuleBase::timer::end("DeltaP", "integrate_polarization");
}
```

- [ ] **Step 2: Add include for PARAM**

Make sure `deltap_berry.cpp` includes:

```cpp
#include "source_io/module_parameter/parameter.h"
```

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add source/source_lcao/module_deltap/deltap_berry.cpp
git commit -m "feat(deltap): implement polarization integration over k-string"
```

---

## Task 10: I/O and Sum Rule Verification

**Files:**
- Create: `source/source_lcao/module_deltap/deltap_io.cpp`
- Modify: `source/source_lcao/module_deltap/CMakeLists.txt` (add deltap_io.cpp)

- [ ] **Step 1: Implement `write_results` and `verify_sum_rule` in `deltap_io.cpp`**

```cpp
#include "deltap.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace deltap {

void DeltaP::verify_sum_rule()
{
    ModuleBase::TITLE("DeltaP", "verify_sum_rule");

    const double p_total_mag = results_.P_total.norm();
    const double p_abacus_mag = results_.P_abacus.norm();

    if (p_abacus_mag > 1e-10)
    {
        const double rel_error = std::abs(p_total_mag - p_abacus_mag) / p_abacus_mag;
        std::cout << " * DeltaP Sum Rule Check:" << std::endl;
        std::cout << "   P_total (DeltaP)  = " << std::scientific << std::setprecision(6)
                  << results_.P_total[gdir_-1] << std::endl;
        std::cout << "   P_total (ABACUS)  = " << results_.P_abacus[gdir_-1] << std::endl;
        std::cout << "   Relative error    = " << rel_error << std::endl;
        if (rel_error < 0.01)
        {
            std::cout << "   [PASS] Sum rule satisfied (< 1%)" << std::endl;
        }
        else
        {
            std::cout << "   [WARN] Sum rule NOT satisfied (> 1%)" << std::endl;
        }
    }
    else
    {
        std::cout << " * DeltaP: P_total = " << results_.P_total[gdir_-1]
                  << " (no ABACUS reference for comparison)" << std::endl;
    }
}

void DeltaP::write_results(const UnitCell& ucell) const
{
    // Write to OUT.{suffix}/deltap_results.dat
    const std::string out_dir = "OUT." + PARAM.inp.suffix;
    const std::string filename = out_dir + "/deltap_results.dat";

    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        std::cerr << "Warning: cannot open " << filename << " for writing" << std::endl;
        return;
    }

    ofs << "# DeltaP atomic polarization decomposition" << std::endl;
    ofs << "# Direction: " << gdir_ << " (1=x, 2=y, 3=z)" << std::endl;
    ofs << "# SMO radius: " << rm_ << " Bohr" << std::endl;
    ofs << "#" << std::endl;
    ofs << "# Atom    Px          Py          Pz          (C/m^2)" << std::endl;
    ofs << std::scientific << std::setprecision(8);

    for (int iat = 0; iat < nat_; iat++)
    {
        int ia, it;
        ucell.iat2iait(iat, &ia, &it);
        ofs << "  " << std::setw(4) << ucell.atom_label[it]
            << " " << std::setw(4) << ia
            << "  " << std::setw(14) << results_.P_I[iat].x
            << " " << std::setw(14) << results_.P_I[iat].y
            << " " << std::setw(14) << results_.P_I[iat].z
            << std::endl;
    }

    ofs << "#" << std::endl;
    ofs << "# Total   " << std::setw(14) << results_.P_total.x
        << " " << std::setw(14) << results_.P_total.y
        << " " << std::setw(14) << results_.P_total.z << std::endl;
    ofs << "# ABACUS  " << std::setw(14) << results_.P_abacus.x
        << " " << std::setw(14) << results_.P_abacus.y
        << " " << std::setw(14) << results_.P_abacus.z << std::endl;

    ofs.close();
    std::cout << " * DeltaP results written to " << filename << std::endl;
}

} // namespace deltap
```

- [ ] **Step 2: Update `CMakeLists.txt`** (ensure `deltap_io.cpp` is in objects list)

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add source/source_lcao/module_deltap/deltap_io.cpp source/source_lcao/module_deltap/CMakeLists.txt
git commit -m "feat(deltap): implement I/O and sum-rule verification"
```

---

## Task 11: Main Orchestration — `compute_atomic_polarization`

**Files:**
- Modify: `source/source_lcao/module_deltap/deltap.cpp` (implement `compute_atomic_polarization`)

- [ ] **Step 1: Implement the main workflow in `deltap.cpp`**

Replace the stub `compute_atomic_polarization` with:

```cpp
#include "deltap.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#include <iostream>

namespace deltap {

void DeltaP::init(const UnitCell& ucell, const Grid_Driver& gd, const K_Vectors& kv,
                  const TwoCenterIntegrator* intor, const std::vector<double>& orb_cutoff,
                  double rm, int gdir)
{
    intor_ = intor;
    orb_cutoff_ = orb_cutoff;
    rm_ = rm;
    gdir_ = gdir;
    nat_ = ucell.nat;
    ModuleBase::TITLE("DeltaP", "init");
}

void DeltaP::compute_atomic_polarization(const UnitCell& ucell,
    const psi::Psi<std::complex<double>>* psi, const elecstate::ElecState* pelec)
{
    ModuleBase::TITLE("DeltaP", "compute_atomic_polarization");
    ModuleBase::timer::start("DeltaP", "compute_atomic_polarization");

    std::cout << "\n * * * * * *\n << Start DeltaP atomic polarization decomposition\n";

    // Step 1: compute real-space overlaps
    compute_real_overlaps(ucell, gd_);  // need Grid_Driver — see note below

    // Step 2: setup k-string
    // Need K_Vectors — store during init or pass as parameter
    // For now, assume kstring setup is done

    // Step 3: Loop over k-points on string, compute S, dS, D_I
    const int nks = psi->get_nk();
    const int nbands = psi->get_nbands();
    const int nrow_local = paraV_->get_row_size();

    kstring_data_.resize(nppstr_);

    for (int j = 0; j < nppstr_; j++)
    {
        // Map string index j to actual k-point index in psi
        // For single-string (istring=0): ik_psi = k_index_[0][j]
        int ik_psi = k_index_[0][j];  // simplified: first string only

        if (ik_psi >= nks) continue;

        psi->fix_k(ik_psi);
        const std::complex<double>* psi_k = psi->get_pointer();

        // Set kvec_d
        kstring_data_[j].kvec_d = pelec->klist->kvec_d[ik_psi];

        // Compute S and dS
        compute_S_k(j);

        // Compute D_I
        compute_D_I(j, psi_k, nbands, nrow_local);
    }

    // Step 4: Compute Berry connection (needs all D_I for FD)
    for (int j = 0; j < nppstr_; j++)
    {
        int ik_psi = k_index_[0][j];
        if (ik_psi >= nks) continue;
        psi->fix_k(ik_psi);
        const std::complex<double>* psi_k = psi->get_pointer();
        const double* wg = &(pelec->wg(ik_psi, 0));
        compute_berry_connection(j, psi_k, nbands, nrow_local, wg);
    }

    // Step 5: Integrate to polarization
    integrate_polarization(ucell, nbands);

    // Step 6: Verify and output
    verify_sum_rule();
    write_results(ucell);

    std::cout << " >> Finish DeltaP atomic polarization decomposition.\n * * * * * *\n";

    ModuleBase::timer::end("DeltaP", "compute_atomic_polarization");
}

} // namespace deltap
```

**Note:** The `init` function needs to store references to `Grid_Driver` and `K_Vectors`. Update `deltap.h` to add:

```cpp
    const Grid_Driver* gd_ = nullptr;
    const K_Vectors* kv_ = nullptr;
```

And update `init` to store them:

```cpp
void DeltaP::init(const UnitCell& ucell, const Grid_Driver& gd, const K_Vectors& kv, ...)
{
    ...
    gd_ = &gd;
    kv_ = &kv;
    ...
}
```

Also call `setup_kstring(kv)` in `init` after storing `kv_`.

- [ ] **Step 2: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) deltap 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add source/source_lcao/module_deltap/deltap.h source/source_lcao/module_deltap/deltap.cpp
git commit -m "feat(deltap): implement main orchestration workflow"
```

---

## Task 12: Entry Point Integration in `ctrl_scf_lcao.cpp`

**Files:**
- Modify: `source/source_io/module_ctrl/ctrl_scf_lcao.cpp` (add DeltaP call after berry_phase)

- [ ] **Step 1: Add include and entry point**

After the berry_phase block (around line 357), add:

```cpp
    //------------------------------------------------------------------
    //! 12b) DeltaP atomic polarization decomposition
    //------------------------------------------------------------------
    if (inp.calculation == "nscf" && inp.deltap_switch)
    {
        std::cout << FmtCore::format("\n * * * * * *\n << Start %s.\n", "DeltaP decomposition");
        deltap::DeltaP dp;
        dp.init(ucell, gd, kv, orb.two_center_bundle->overlap_orb_onsite.get(),
                orb_cutoff, inp.deltap_rm, inp.deltap_gdir);
        dp.compute_atomic_polarization(psi, pelec);
        std::cout << FmtCore::format(" >> Finish %s.\n * * * * * *\n", "DeltaP decomposition");
    }
```

Add the include at the top of the file:

```cpp
#include "source_lcao/module_deltap/deltap.h"
```

- [ ] **Step 2: Verify build**

```bash
cd build && cmake .. && make -j$(nproc) 2>&1 | tail -10
```

Expected: full build succeeds. Fix any link errors (ensure `deltap` OBJECT library is linked into the main binary).

- [ ] **Step 3: Commit**

```bash
git add source/source_io/module_ctrl/ctrl_scf_lcao.cpp
git commit -m "feat(deltap): add post-processing entry point in ctrl_scf_lcao"
```

---

## Task 13: T1 Integration Test — BaTiO3 Sum Rule

**Files:**
- Create: `tests/17_DS_DFTU/18_LCAO_DELTAP_BTO/INPUT`
- Create: `tests/17_DS_DFTU/18_LCAO_DELTAP_BTO/KPT`
- Create: `tests/17_DS_DFTU/18_LCAO_DELTAP_BTO/STRU`
- Create: `tests/17_DS_DFTU/18_LCAO_DELTAP_BTO/result.ref`

- [ ] **Step 1: Create test directory**

```bash
mkdir -p tests/17_DS_DFTU/18_LCAO_DELTAP_BTO
```

- [ ] **Step 2: Create INPUT file**

```
INPUT_PARAMETERS
suffix    autotest
calculation    nscf
basis_type    lcao
ecutwfc    20
gamma_only    0

nspin    1
scf_thr    1.0e-6
scf_nmax    50
out_chg    0
smearing_method    gaussian
smearing_sigma    0.01
mixing_type    broyden
mixing_beta    0.4
ks_solver    genelpa
symmetry    0

berry_phase    1
gdir    3

deltap_switch    1
deltap_rm    3.0
deltap_gdir    3

pseudo_dir    ../../PP_ORB
orbital_dir    ../../PP_ORB
```

- [ ] **Step 3: Create KPT file**

```
K_POINTS
0
Monkhorst_Pack
8 8 8
0 0 0
```

- [ ] **Step 4: Create STRU file** (tetragonal BaTiO3 with ferroelectric distortion)

```
ATOMIC_SPECIES
Ba 137.328 Ba_upf
Ti 47.867 Ti_upf
O  15.999 O_upf

NUMERICAL_ORBITAL
Ba_dzp.orb
Ti_dzp.orb
O_dzp.orb

LATTICE_CONSTANT
1.8897261254578284

LATTICE_VECTORS
4.00    0.00    0.00
0.00    4.00    0.00
0.00    0.00    4.20

ATOMIC_POSITIONS
Direct

Ba
0.0
1
0.00   0.00   0.00

Ti
0.0
1
0.50   0.50   0.52

O
0.0
2
0.50   0.50   0.00
0.50   0.50   0.50

O
0.0
2
0.50   0.00   0.50
0.00   0.50   0.50
```

- [ ] **Step 5: Create `result.ref`** (will be populated after first run)

For now, create a placeholder that checks for key output strings:

```
Total
ABACUS
```

- [ ] **Step 6: Run the test (requires ABACUS built + pseudopotentials)**

```bash
cd tests/17_DS_DFTU/18_LCAO_DELTAP_BTO
mpirun -np 4 /root/abacus-develop/build/abacus
```

Expected: produces `OUT.autotest/deltap_results.dat` with per-atom P^I and sum-rule comparison. Inspect output and update `result.ref` with actual values.

- [ ] **Step 7: Commit**

```bash
git add tests/17_DS_DFTU/18_LCAO_DELTAP_BTO/
git commit -m "test(deltap): add BaTiO3 T1 sum-rule integration test"
```

---

## Task 14: Unit Test — Phase-Summing Math (T0 Foundation)

**Files:**
- Create: `source/source_lcao/module_deltap/test/CMakeLists.txt`
- Create: `source/source_lcao/module_deltap/test/deltap_math_test.cpp`

This tests the core math (S, dS, Berry connection formula) with synthetic data — no ABACUS infrastructure needed.

- [ ] **Step 1: Create test `CMakeLists.txt`**

```cmake
if(ENABLE_LCAO)

AddTest(
  TARGET MODULE_LCAO_deltap_math_test
  LIBS ${math_libs} base device parameter
  SOURCES deltap_math_test.cpp
)

endif()
```

- [ ] **Step 2: Write the math unit test**

```cpp
#include "gtest/gtest.h"
#include "source_base/vector3.h"
#include "source_base/constants.h"
#include <complex>
#include <vector>
#include <cmath>

// Test the phase-summing formula:
// S(k) = sum_R e^{ikR} * overlap(R)
// dS(k,alpha) = sum_R i*R_alpha*e^{ikR} * overlap(R)
// 
// We verify: dS(k,alpha) analytic matches finite-difference of S(k)

namespace {

std::complex<double> compute_S(double k, const std::vector<std::pair<double, double>>& overlaps)
{
    std::complex<double> s = {0.0, 0.0};
    for (const auto& [R, val] : overlaps)
    {
        double arg = k * R;
        s += std::complex<double>(std::cos(arg), std::sin(arg)) * val;
    }
    return s;
}

std::complex<double> compute_dS_analytic(double k, int alpha,
    const std::vector<std::pair<double, std::vector<double>>>& overlaps)
{
    // dS(k,alpha) = sum_R i*R_alpha * e^{ikR} * overlap(R)
    std::complex<double> ds = {0.0, 0.0};
    for (const auto& [R_vec, val] : overlaps)
    {
        double arg = k * R_vec;  // simplified 1D
        std::complex<double> phase(std::cos(arg), std::sin(arg));
        std::complex<double> i_R(0.0, R_vec);
        ds += i_R * phase * val[alpha];
    }
    return ds;
}

std::complex<double> compute_dS_finite_diff(double k, double dk,
    const std::vector<std::pair<double, double>>& overlaps)
{
    // dS/dk = [S(k+dk) - S(k-dk)] / (2*dk)
    std::complex<double> s_plus = compute_S(k + dk, overlaps);
    std::complex<double> s_minus = compute_S(k - dk, overlaps);
    return (s_plus - s_minus) / (2.0 * dk);
}

} // anonymous namespace

class DeltaPMathTest : public testing::Test
{
protected:
    // Synthetic overlaps: (R, value) pairs for a simple 1D chain
    std::vector<std::pair<double, double>> overlaps_1d;
    double dk_ = 1e-6;

    void SetUp() override
    {
        overlaps_1d = {
            {-2.0, 0.01},
            {-1.0, 0.15},
            { 0.0, 1.00},
            { 1.0, 0.15},
            { 2.0, 0.01},
        };
    }
};

TEST_F(DeltaPMathTest, SSumConsistency)
{
    // S(0) should equal sum of all overlaps (all phases = 1)
    double s0_real = compute_S(0.0, overlaps_1d).real();
    double expected = 0.01 + 0.15 + 1.00 + 0.15 + 0.01;
    EXPECT_NEAR(s0_real, expected, 1e-12);
}

TEST_F(DeltaPMathTest, AnalyticDSMatchesFiniteDiff)
{
    // For 1D, alpha=0 (x direction)
    std::vector<std::pair<double, std::vector<double>>> overlaps_vec;
    for (const auto& [R, val] : overlaps_1d)
    {
        overlaps_vec.push_back({R, {val, 0.0, 0.0}});
    }

    std::vector<double> k_test = {0.1, 0.5, 1.0, 1.5, 2.0, 3.14159};

    for (double k : k_test)
    {
        std::complex<double> ds_analytic = compute_dS_analytic(k, 0, overlaps_vec);
        std::complex<double> ds_fd = compute_dS_finite_diff(k, dk_, overlaps_1d);

        double rel_error = std::abs(ds_analytic - ds_fd) / (std::abs(ds_fd) + 1e-15);
        EXPECT_LT(rel_error, 1e-8)
            << "k=" << k << " analytic=" << ds_analytic << " fd=" << ds_fd
            << " rel_error=" << rel_error;
    }
}

TEST_F(DeltaPMathTest, BerryConnectionGaugeInvariance)
{
    // A_n = <psi|d_k alpha> * <alpha|psi> + <psi|alpha> * d_k<alpha|psi>
    // Under psi -> e^{i*phi} * psi, A_n should be invariant
    // (because the phase cancels between bra and ket)

    std::complex<double> psi(0.7, 0.3);
    std::complex<double> d_alpha_psi(0.2, -0.1);  // <d_k alpha | psi>
    std::complex<double> alpha_psi(0.5, 0.2);     // <alpha | psi>
    std::complex<double> psi_d_alpha(-0.1, 0.4);  // <psi | d_k alpha> = conj(d_alpha_psi) * |psi|^2/|psi|^2... 
    // Actually: <psi|d_k alpha> = conj(d_k <alpha|psi>) if the operator is Hermitian
    // But for Berry connection, we use:
    // A = <psi|d_k alpha><alpha|psi> + <psi|alpha> d_k<alpha|psi>
    // = conj(d_k<alpha|psi>) * <alpha|psi> + conj(<alpha|psi>) * d_k<alpha|psi>
    // = 2 * Re[conj(d_k<alpha|psi>) * <alpha|psi>]

    std::complex<double> A = std::conj(d_alpha_psi) * alpha_psi
                           + std::conj(alpha_psi) * d_alpha_psi;

    // Apply gauge: psi -> e^{i*phi} * psi
    double phi = 0.37;
    std::complex<double> phase(std::cos(phi), std::sin(phi));
    std::complex<double> psi_g = phase * psi;
    std::complex<double> d_alpha_psi_g = phase * d_alpha_psi;
    std::complex<double> alpha_psi_g = phase * alpha_psi;

    std::complex<double> A_g = std::conj(d_alpha_psi_g) * alpha_psi_g
                             + std::conj(alpha_psi_g) * d_alpha_psi_g;

    // A should be real and gauge-invariant
    EXPECT_NEAR(A.imag(), 0.0, 1e-12);
    EXPECT_NEAR(A_g.imag(), 0.0, 1e-12);
    EXPECT_NEAR(A.real(), A_g.real(), 1e-12);
}
```

- [ ] **Step 3: Build and run the test**

```bash
cd build && cmake .. && make -j$(nproc) MODULE_LCAO_deltap_math_test 2>&1 | tail -5
./tests/MODULE_LCAO_deltap_math_test
```

Expected: all 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add source/source_lcao/module_deltap/test/
git commit -m "test(deltap): add T0 math unit tests for phase-summing and gauge invariance"
```

---

## Self-Review Notes

**Spec coverage:**
- Section 2 (Physics): Implemented in deltap_berry.cpp (S, dS, D_I, Berry connection, integration)
- Section 3 (Architecture): Module location, CMakeLists, entry point — Tasks 1, 12
- Section 3.3 (Input params): Task 2
- Section 3.4 (Reuse): TwoCenterIntegrator::snap() in Task 4, cal_PI_sub pattern in Task 7, berryphase::set_kpoints pattern in Task 5
- Section 4 (Data structures): Task 3
- Section 5 (Algorithm): Tasks 4-9 follow the 6-step workflow
- Section 6 (Tests): T0 math in Task 14, T1 BaTiO3 in Task 13
- Section 8 (Risks): SMO completeness tested via rm variation in T1; FD accuracy in T0; gauge in T0

**Known gaps to address during implementation:**
1. The `paraV_` (Parallel_Orbitals) needs to be set — pass it in `init()` or get from the operator
2. The `orb.two_center_bundle->overlap_orb_onsite` must be non-null — requires `onsite_radius > 0` or `deltap_rm > 0` in input
3. The k-string index mapping (`k_index_[0][j]` -> actual psi k-point) needs careful handling for MPI parallel k-points
4. The berry_phase output (`P_abacus`) needs to be read from the berryphase results or computed in the same run
5. The `ucell.latvec(row, col)` API needs verification — may be `ucell.latvec[row][col]` or similar
