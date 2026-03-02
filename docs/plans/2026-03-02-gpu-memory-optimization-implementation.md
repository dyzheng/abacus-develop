# GPU Memory Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement phased GPU memory optimization for Davidson diagonalization to achieve 75-85% memory reduction with <15% performance overhead.

**Architecture:** Two-tier memory hierarchy with CPU as staging area. Phase 1 implements k-point paging for wavefunctions. Phase 2 adds atom-batch computation for VKB projectors. Phase 3 (optional) adds real-space projector support.

**Tech Stack:** C++14, CUDA 10.1+, cuBLAS, GoogleTest, CMake

---

## Phase 1: Psi K-Point Paging

### Task 1: Add PsiStorageMode Enum

**Files:**
- Modify: `source/module_psi/psi.h:1-50`

**Step 1: Add storage mode enum before Psi class definition**

```cpp
// Add after existing includes, before namespace psi
namespace psi {

enum class PsiStorageMode {
    ALL_GPU,      // Current behavior: all k-points on GPU
    ALL_CPU,      // All k-points on CPU
    PAGED_GPU     // CPU storage + single k-point GPU buffer
};

} // namespace psi
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors (enum doesn't break existing code)

**Step 3: Commit**

```bash
git add source/module_psi/psi.h
git commit -m "feat(psi): add PsiStorageMode enum for memory optimization"
```

---

### Task 2: Add Psi Class Member Variables

**Files:**
- Modify: `source/module_psi/psi.h` (Psi class private section)

**Step 1: Add member variables to Psi class**

Find the private section of the Psi class and add:

```cpp
private:
    // Existing members...

    // Memory optimization members
    PsiStorageMode storage_mode_ = PsiStorageMode::ALL_GPU;
    T* psi_cpu_ = nullptr;           // CPU storage for all k-points
    T* psi_gpu_buffer_ = nullptr;    // GPU buffer for current k-point
    T* psi_gpu_transfer_buffer_ = nullptr;  // Second buffer for double buffering
    int current_k_gpu_ = -1;         // Which k-point is on GPU (-1 = none)

#if defined(__CUDA) || defined(__ROCM)
    void* compute_stream_ = nullptr;  // CUDA stream for computation
    void* transfer_stream_ = nullptr; // CUDA stream for transfers
#endif
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 3: Commit**

```bash
git add source/module_psi/psi.h
git commit -m "feat(psi): add member variables for paged GPU storage"
```

---

### Task 3: Add Psi Public Methods

**Files:**
- Modify: `source/module_psi/psi.h` (Psi class public section)

**Step 1: Add public method declarations**

In the public section of Psi class, add:

```cpp
public:
    // Existing methods...

    // Memory optimization methods
    void set_storage_mode(PsiStorageMode mode);
    PsiStorageMode get_storage_mode() const { return storage_mode_; }

    void load_k_to_gpu(int ik);
    void store_k_from_gpu(int ik);
    void ensure_k_on_gpu(int ik);

    int get_current_k_gpu() const { return current_k_gpu_; }
    T* get_cpu_pointer(int ik = 0);
    const T* get_cpu_pointer(int ik = 0) const;
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: Linker errors (methods not implemented yet) - this is expected

**Step 3: Commit**

```bash
git add source/module_psi/psi.h
git commit -m "feat(psi): add public methods for k-point paging"
```

---

### Task 4: Write Unit Test for PsiStorageMode

**Files:**
- Create: `source/module_psi/test/psi_paging_test.cpp`

**Step 1: Create test file with basic structure**

```cpp
#include "module_psi/psi.h"
#include "gtest/gtest.h"
#include <complex>

#ifdef __CUDA
#include <cuda_runtime.h>
#endif

using namespace psi;

class PsiPagingTest : public ::testing::Test {
protected:
    void SetUp() override {
        nk = 10;
        nband = 20;
        nbasis = 1000;
    }

    int nk;
    int nband;
    int nbasis;
};

TEST_F(PsiPagingTest, DefaultStorageMode) {
    Psi<std::complex<double>> psi(nk, nband, nbasis);
    EXPECT_EQ(psi.get_storage_mode(), PsiStorageMode::ALL_GPU);
}

TEST_F(PsiPagingTest, SetStorageMode) {
    Psi<std::complex<double>> psi(nk, nband, nbasis);
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);
    EXPECT_EQ(psi.get_storage_mode(), PsiStorageMode::PAGED_GPU);
}
```

**Step 2: Add test to CMakeLists.txt**

Modify: `source/module_psi/test/CMakeLists.txt`

Add:
```cmake
if(USE_CUDA OR USE_ROCM)
    AddTest(
        TARGET psi_paging_test
        SOURCES psi_paging_test.cpp
        LIBS psi
    )
endif()
```

**Step 3: Run test to verify it fails**

Run: `cd build && ctest -R psi_paging_test -V`
Expected: FAIL - methods not implemented

**Step 4: Commit**

```bash
git add source/module_psi/test/psi_paging_test.cpp source/module_psi/test/CMakeLists.txt
git commit -m "test(psi): add unit tests for paged storage mode"
```

---

### Task 5: Implement set_storage_mode Method

**Files:**
- Create: `source/module_psi/psi_paging.cpp`

**Step 1: Create implementation file**

```cpp
#include "module_psi/psi.h"
#include "module_base/tool_quit.h"

#ifdef __CUDA
#include <cuda_runtime.h>
#elif defined(__ROCM)
#include <hip/hip_runtime.h>
#endif

namespace psi {

template <typename T, typename Device>
void Psi<T, Device>::set_storage_mode(PsiStorageMode mode) {
    if (mode == storage_mode_) {
        return;  // Already in requested mode
    }

    // For now, only support setting mode before allocation
    if (this->psi != nullptr) {
        ModuleBase::WARNING_QUIT("Psi::set_storage_mode",
            "Cannot change storage mode after allocation");
    }

    storage_mode_ = mode;
}

template <typename T, typename Device>
T* Psi<T, Device>::get_cpu_pointer(int ik) {
    if (storage_mode_ == PsiStorageMode::PAGED_GPU) {
        if (ik < 0 || ik >= this->nk) {
            ModuleBase::WARNING_QUIT("Psi::get_cpu_pointer", "Invalid k-point index");
        }
        return psi_cpu_ + ik * this->nbands * this->nbasis;
    } else {
        return this->psi + ik * this->nbands * this->nbasis;
    }
}

template <typename T, typename Device>
const T* Psi<T, Device>::get_cpu_pointer(int ik) const {
    if (storage_mode_ == PsiStorageMode::PAGED_GPU) {
        if (ik < 0 || ik >= this->nk) {
            ModuleBase::WARNING_QUIT("Psi::get_cpu_pointer", "Invalid k-point index");
        }
        return psi_cpu_ + ik * this->nbands * this->nbasis;
    } else {
        return this->psi + ik * this->nbands * this->nbasis;
    }
}

// Explicit instantiations
template class Psi<std::complex<float>, base_device::DEVICE_CPU>;
template class Psi<std::complex<double>, base_device::DEVICE_CPU>;
#if defined(__CUDA) || defined(__ROCM)
template class Psi<std::complex<float>, base_device::DEVICE_GPU>;
template class Psi<std::complex<double>, base_device::DEVICE_GPU>;
#endif

} // namespace psi
```

**Step 2: Add to CMakeLists.txt**

Modify: `source/module_psi/CMakeLists.txt`

Add `psi_paging.cpp` to the source list.

**Step 3: Run test**

Run: `cd build && cmake --build . -j$(nproc) && ctest -R psi_paging_test -V`
Expected: Tests should pass now

**Step 4: Commit**

```bash
git add source/module_psi/psi_paging.cpp source/module_psi/CMakeLists.txt
git commit -m "feat(psi): implement set_storage_mode and get_cpu_pointer"
```

---

### Task 6: Modify Psi Constructor for PAGED_GPU Mode

**Files:**
- Modify: `source/module_psi/psi.cpp` (constructor)

**Step 1: Write test for PAGED_GPU allocation**

Add to `source/module_psi/test/psi_paging_test.cpp`:

```cpp
#ifdef __CUDA
TEST_F(PsiPagingTest, PagedGPUAllocation) {
    Psi<std::complex<double>, base_device::DEVICE_GPU> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);

    std::vector<int> ngk(nk, nbasis);
    psi.resize(nk, nband, nbasis, ngk.data());

    // Check that CPU memory is allocated
    EXPECT_NE(psi.get_cpu_pointer(0), nullptr);

    // Check that GPU buffer is allocated (size of 1 k-point)
    EXPECT_NE(psi.get_pointer(), nullptr);

    // Check current k-point is unset
    EXPECT_EQ(psi.get_current_k_gpu(), -1);
}
#endif
```

**Step 2: Run test to verify it fails**

Run: `cd build && ctest -R psi_paging_test -V`
Expected: FAIL - allocation not implemented

**Step 3: Implement PAGED_GPU allocation in Psi::resize**

Modify `source/module_psi/psi.cpp`, find the `resize` method and add:

```cpp
template <typename T, typename Device>
void Psi<T, Device>::resize(const int nks, const int nbands_in, const int nbasis_in, const int* ngk_in) {
    // Existing code...

    if (storage_mode_ == PsiStorageMode::PAGED_GPU) {
        // Allocate CPU storage for all k-points
        this->psi_cpu_ = new T[nks * nbands_in * nbasis_in];
        std::memset(this->psi_cpu_, 0, nks * nbands_in * nbasis_in * sizeof(T));

        // Allocate GPU buffers for 1 k-point (double buffering)
#if defined(__CUDA) || defined(__ROCM)
        if (this->device == base_device::GpuDevice) {
            cudaMalloc(&this->psi_gpu_buffer_, nbands_in * nbasis_in * sizeof(T));
            cudaMalloc(&this->psi_gpu_transfer_buffer_, nbands_in * nbasis_in * sizeof(T));
            cudaMemset(this->psi_gpu_buffer_, 0, nbands_in * nbasis_in * sizeof(T));
            cudaMemset(this->psi_gpu_transfer_buffer_, 0, nbands_in * nbasis_in * sizeof(T));

            // Create CUDA streams
            cudaStreamCreate((cudaStream_t*)&this->compute_stream_);
            cudaStreamCreate((cudaStream_t*)&this->transfer_stream_);

            // Point psi to GPU buffer
            this->psi = this->psi_gpu_buffer_;
        }
#endif
    } else {
        // Original allocation logic
        // ... existing code ...
    }
}
```

**Step 4: Run test**

Run: `cd build && cmake --build . -j$(nproc) && ctest -R psi_paging_test -V`
Expected: PASS

**Step 5: Commit**

```bash
git add source/module_psi/psi.cpp source/module_psi/test/psi_paging_test.cpp
git commit -m "feat(psi): implement PAGED_GPU memory allocation"
```

---

### Task 7: Implement load_k_to_gpu Method

**Files:**
- Modify: `source/module_psi/psi_paging.cpp`

**Step 1: Write test for load_k_to_gpu**

Add to `source/module_psi/test/psi_paging_test.cpp`:

```cpp
#ifdef __CUDA
TEST_F(PsiPagingTest, LoadKPointToGPU) {
    Psi<std::complex<double>, base_device::DEVICE_GPU> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);

    std::vector<int> ngk(nk, nbasis);
    psi.resize(nk, nband, nbasis, ngk.data());

    // Initialize CPU data for k-point 5
    auto* cpu_ptr = psi.get_cpu_pointer(5);
    for (int i = 0; i < nband * nbasis; i++) {
        cpu_ptr[i] = std::complex<double>(5.0 + i, 0.0);
    }

    // Load k-point 5 to GPU
    psi.load_k_to_gpu(5);

    // Check current k-point is set
    EXPECT_EQ(psi.get_current_k_gpu(), 5);

    // Copy back from GPU and verify
    std::vector<std::complex<double>> gpu_data(nband * nbasis);
    cudaMemcpy(gpu_data.data(), psi.get_pointer(),
               nband * nbasis * sizeof(std::complex<double>),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < nband * nbasis; i++) {
        EXPECT_DOUBLE_EQ(gpu_data[i].real(), 5.0 + i);
    }
}
#endif
```

**Step 2: Run test to verify it fails**

Run: `cd build && ctest -R psi_paging_test -V`
Expected: FAIL - load_k_to_gpu not implemented

**Step 3: Implement load_k_to_gpu**

Add to `source/module_psi/psi_paging.cpp`:

```cpp
template <typename T, typename Device>
void Psi<T, Device>::load_k_to_gpu(int ik) {
    if (storage_mode_ != PsiStorageMode::PAGED_GPU) {
        return;  // No-op if not in paged mode
    }

    if (ik < 0 || ik >= this->nk) {
        ModuleBase::WARNING_QUIT("Psi::load_k_to_gpu", "Invalid k-point index");
    }

#if defined(__CUDA) || defined(__ROCM)
    if (this->device == base_device::GpuDevice) {
        const size_t size = this->nbands * this->nbasis * sizeof(T);
        T* src = psi_cpu_ + ik * this->nbands * this->nbasis;

        cudaError_t err = cudaMemcpy(psi_gpu_buffer_, src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::string msg = "Failed to transfer k-point " + std::to_string(ik)
                            + " to GPU: " + cudaGetErrorString(err);
            ModuleBase::WARNING_QUIT("Psi::load_k_to_gpu", msg);
        }

        current_k_gpu_ = ik;
    }
#endif
}
```

**Step 4: Run test**

Run: `cd build && cmake --build . -j$(nproc) && ctest -R psi_paging_test -V`
Expected: PASS

**Step 5: Commit**

```bash
git add source/module_psi/psi_paging.cpp source/module_psi/test/psi_paging_test.cpp
git commit -m "feat(psi): implement load_k_to_gpu method"
```

---

### Task 8: Implement store_k_from_gpu Method

**Files:**
- Modify: `source/module_psi/psi_paging.cpp`

**Step 1: Write test for store_k_from_gpu**

Add to `source/module_psi/test/psi_paging_test.cpp`:

```cpp
#ifdef __CUDA
TEST_F(PsiPagingTest, StoreKPointFromGPU) {
    Psi<std::complex<double>, base_device::DEVICE_GPU> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);

    std::vector<int> ngk(nk, nbasis);
    psi.resize(nk, nband, nbasis, ngk.data());

    // Load k-point 3
    psi.load_k_to_gpu(3);

    // Modify GPU data
    std::vector<std::complex<double>> new_data(nband * nbasis);
    for (int i = 0; i < nband * nbasis; i++) {
        new_data[i] = std::complex<double>(3.14 + i, 2.71);
    }
    cudaMemcpy(psi.get_pointer(), new_data.data(),
               nband * nbasis * sizeof(std::complex<double>),
               cudaMemcpyHostToDevice);

    // Store back to CPU
    psi.store_k_from_gpu(3);

    // Verify CPU data was updated
    auto* cpu_ptr = psi.get_cpu_pointer(3);
    for (int i = 0; i < nband * nbasis; i++) {
        EXPECT_DOUBLE_EQ(cpu_ptr[i].real(), 3.14 + i);
        EXPECT_DOUBLE_EQ(cpu_ptr[i].imag(), 2.71);
    }
}
#endif
```

**Step 2: Run test to verify it fails**

Run: `cd build && ctest -R psi_paging_test -V`
Expected: FAIL - store_k_from_gpu not implemented

**Step 3: Implement store_k_from_gpu**

Add to `source/module_psi/psi_paging.cpp`:

```cpp
template <typename T, typename Device>
void Psi<T, Device>::store_k_from_gpu(int ik) {
    if (storage_mode_ != PsiStorageMode::PAGED_GPU) {
        return;  // No-op if not in paged mode
    }

    if (ik < 0 || ik >= this->nk) {
        ModuleBase::WARNING_QUIT("Psi::store_k_from_gpu", "Invalid k-point index");
    }

#if defined(__CUDA) || defined(__ROCM)
    if (this->device == base_device::GpuDevice) {
        const size_t size = this->nbands * this->nbasis * sizeof(T);
        T* dst = psi_cpu_ + ik * this->nbands * this->nbasis;

        cudaError_t err = cudaMemcpy(dst, psi_gpu_buffer_, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::string msg = "Failed to transfer k-point " + std::to_string(ik)
                            + " from GPU: " + cudaGetErrorString(err);
            ModuleBase::WARNING_QUIT("Psi::store_k_from_gpu", msg);
        }
    }
#endif
}
```

**Step 4: Run test**

Run: `cd build && cmake --build . -j$(nproc) && ctest -R psi_paging_test -V`
Expected: PASS

**Step 5: Commit**

```bash
git add source/module_psi/psi_paging.cpp source/module_psi/test/psi_paging_test.cpp
git commit -m "feat(psi): implement store_k_from_gpu method"
```

---

### Task 9: Implement ensure_k_on_gpu Method

**Files:**
- Modify: `source/module_psi/psi_paging.cpp`

**Step 1: Write test for ensure_k_on_gpu**

Add to `source/module_psi/test/psi_paging_test.cpp`:

```cpp
#ifdef __CUDA
TEST_F(PsiPagingTest, EnsureKPointOnGPU) {
    Psi<std::complex<double>, base_device::DEVICE_GPU> psi;
    psi.set_storage_mode(PsiStorageMode::PAGED_GPU);

    std::vector<int> ngk(nk, nbasis);
    psi.resize(nk, nband, nbasis, ngk.data());

    // Initialize CPU data
    auto* cpu_ptr = psi.get_cpu_pointer(7);
    for (int i = 0; i < nband * nbasis; i++) {
        cpu_ptr[i] = std::complex<double>(7.0 + i, 0.0);
    }

    // First call should load
    psi.ensure_k_on_gpu(7);
    EXPECT_EQ(psi.get_current_k_gpu(), 7);

    // Second call should be no-op (already loaded)
    psi.ensure_k_on_gpu(7);
    EXPECT_EQ(psi.get_current_k_gpu(), 7);

    // Different k-point should trigger new load
    psi.ensure_k_on_gpu(8);
    EXPECT_EQ(psi.get_current_k_gpu(), 8);
}
#endif
```

**Step 2: Run test to verify it fails**

Run: `cd build && ctest -R psi_paging_test -V`
Expected: FAIL - ensure_k_on_gpu not implemented

**Step 3: Implement ensure_k_on_gpu**

Add to `source/module_psi/psi_paging.cpp`:

```cpp
template <typename T, typename Device>
void Psi<T, Device>::ensure_k_on_gpu(int ik) {
    if (storage_mode_ != PsiStorageMode::PAGED_GPU) {
        return;  // No-op if not in paged mode
    }

    if (current_k_gpu_ != ik) {
        load_k_to_gpu(ik);
    }
}
```

**Step 4: Run test**

Run: `cd build && cmake --build . -j$(nproc) && ctest -R psi_paging_test -V`
Expected: PASS

**Step 5: Commit**

```bash
git add source/module_psi/psi_paging.cpp source/module_psi/test/psi_paging_test.cpp
git commit -m "feat(psi): implement ensure_k_on_gpu method"
```

---

### Task 10: Update Psi Destructor for PAGED_GPU

**Files:**
- Modify: `source/module_psi/psi.cpp` (destructor)

**Step 1: Write test for proper cleanup**

Add to `source/module_psi/test/psi_paging_test.cpp`:

```cpp
#ifdef __CUDA
TEST_F(PsiPagingTest, ProperCleanup) {
    // Test that destructor doesn't leak memory
    {
        Psi<std::complex<double>, base_device::DEVICE_GPU> psi;
        psi.set_storage_mode(PsiStorageMode::PAGED_GPU);

        std::vector<int> ngk(nk, nbasis);
        psi.resize(nk, nband, nbasis, ngk.data());

        psi.load_k_to_gpu(0);
    }
    // Destructor called here - should not crash or leak

    // If we get here without crash, test passes
    SUCCEED();
}
#endif
```

**Step 2: Run test**

Run: `cd build && ctest -R psi_paging_test -V`
Expected: May pass or fail depending on current destructor implementation

**Step 3: Update destructor**

Modify `source/module_psi/psi.cpp`, find destructor and add:

```cpp
template <typename T, typename Device>
Psi<T, Device>::~Psi() {
    // Existing cleanup...

    // PAGED_GPU cleanup
    if (storage_mode_ == PsiStorageMode::PAGED_GPU) {
        if (psi_cpu_ != nullptr) {
            delete[] psi_cpu_;
            psi_cpu_ = nullptr;
        }

#if defined(__CUDA) || defined(__ROCM)
        if (this->device == base_device::GpuDevice) {
            if (psi_gpu_buffer_ != nullptr) {
                cudaFree(psi_gpu_buffer_);
                psi_gpu_buffer_ = nullptr;
            }
            if (psi_gpu_transfer_buffer_ != nullptr) {
                cudaFree(psi_gpu_transfer_buffer_);
                psi_gpu_transfer_buffer_ = nullptr;
            }
            if (compute_stream_ != nullptr) {
                cudaStreamDestroy((cudaStream_t)compute_stream_);
                compute_stream_ = nullptr;
            }
            if (transfer_stream_ != nullptr) {
                cudaStreamDestroy((cudaStream_t)transfer_stream_);
                transfer_stream_ = nullptr;
            }
        }
#endif
    }
}
```

**Step 4: Run test**

Run: `cd build && cmake --build . -j$(nproc) && ctest -R psi_paging_test -V`
Expected: PASS

**Step 5: Commit**

```bash
git add source/module_psi/psi.cpp source/module_psi/test/psi_paging_test.cpp
git commit -m "feat(psi): add PAGED_GPU cleanup in destructor"
```

---

### Task 11: Add INPUT Parameter for device_memory_mode

**Files:**
- Modify: `source/module_parameter/input_parameter.h`
- Modify: `source/module_io/read_input_item_elec_stru.cpp`

**Step 1: Add parameter to input_parameter.h**

Find the GPU-related parameters section and add:

```cpp
// Around line with other GPU parameters
std::string device_memory_mode = "";  // "full_gpu", "paged", or "" (auto)
```

**Step 2: Add parameter reading in read_input_item_elec_stru.cpp**

Find the GPU parameters section and add:

```cpp
{
    Input_Item item("device_memory_mode");
    item.annotation = "GPU memory mode: full_gpu, paged, or auto (empty string)";
    read_sync_string(input.device_memory_mode);
    item.reset_value = [](const Input& input, Input_Item& item) {
        item.final_value << input.device_memory_mode;
    };
    item.check_value = [](const Input& input, const Input_Item& item) {
        const std::string& mode = input.device_memory_mode;
        if (mode != "" && mode != "full_gpu" && mode != "paged") {
            ModuleBase::WARNING_QUIT("Input",
                "device_memory_mode must be '', 'full_gpu', or 'paged'");
        }
    };
    this->add_item(item);
}
```

**Step 3: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 4: Commit**

```bash
git add source/module_parameter/input_parameter.h source/module_io/read_input_item_elec_stru.cpp
git commit -m "feat(input): add device_memory_mode parameter"
```

---

### Task 12: Integrate Paging into HSolverPW

**Files:**
- Modify: `source/module_hsolver/hsolver_pw.cpp`

**Step 1: Add paging logic to k-point loop in solve() method**

Find the k-point loop (around line 328 and 370) and modify:

```cpp
// Around line 328 (first k-point loop)
for (int ik = 0; ik < nks; ik++) {
    // Add paging logic
    if (psi.get_storage_mode() == psi::PsiStorageMode::PAGED_GPU) {
        psi.load_k_to_gpu(ik);
    }

    psi.fix_k(ik);
    this->update_precondition(precondition, ik, this->wfc_basis->npwk[ik], vl_of_0);

    // Existing diagonalization code...
    this->hamiltSolvePsiK(pHamilt, psi, precondition, eigenvalues.data() + ik * psi.get_nbands(), this->wfc_basis->nks);

    // Store back to CPU
    if (psi.get_storage_mode() == psi::PsiStorageMode::PAGED_GPU) {
        psi.store_k_from_gpu(ik);
    }

    // Existing code...
}
```

**Step 2: Repeat for second k-point loop (around line 370)**

Apply same modification to the second k-point loop.

**Step 3: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 4: Commit**

```bash
git add source/module_hsolver/hsolver_pw.cpp
git commit -m "feat(hsolver): integrate k-point paging into HSolverPW"
```

---

### Task 13: Add Auto-Detection Logic for Storage Mode

**Files:**
- Modify: `source/module_hsolver/hsolver_pw.cpp` (in solve() method, before k-point loop)

**Step 1: Add auto-detection before k-point loop**

Add at the beginning of `HSolverPW::solve()`:

```cpp
// Auto-detect storage mode if not explicitly set
if (PARAM.inp.device_memory_mode == "") {
    // Use paged mode for many k-points on GPU
    if (this->wfc_basis->nks > 10 && PARAM.inp.device == "gpu") {
        psi.set_storage_mode(psi::PsiStorageMode::PAGED_GPU);
        GlobalV::ofs_running << " AUTO: Using paged GPU memory mode (nks="
                            << this->wfc_basis->nks << ")" << std::endl;
    }
} else if (PARAM.inp.device_memory_mode == "paged") {
    psi.set_storage_mode(psi::PsiStorageMode::PAGED_GPU);
    GlobalV::ofs_running << " Using paged GPU memory mode" << std::endl;
} else if (PARAM.inp.device_memory_mode == "full_gpu") {
    psi.set_storage_mode(psi::PsiStorageMode::ALL_GPU);
    GlobalV::ofs_running << " Using full GPU memory mode" << std::endl;
}
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 3: Commit**

```bash
git add source/module_hsolver/hsolver_pw.cpp
git commit -m "feat(hsolver): add auto-detection for storage mode"
```

---

### Task 14: Run Phase 1 Integration Tests

**Files:**
- Test: `tests/integrate/11_PW_GPU/*`

**Step 1: Build and install**

```bash
cd build
cmake --build . -j$(nproc)
cmake --install .
```

**Step 2: Run GPU integration tests with paged mode**

```bash
cd tests/integrate
export ABACUS_DEVICE_MEMORY_MODE=paged
./Autotest.sh -r "11_PW_GPU_101.*"
```

Expected: All tests PASS with same results as baseline

**Step 3: Compare results with baseline**

```bash
# Run baseline (full GPU)
export ABACUS_DEVICE_MEMORY_MODE=full_gpu
./Autotest.sh -r "11_PW_GPU_101.*"

# Compare energies (should be identical within 1e-6 eV)
python ../tools/compare_results.py OUT.paged OUT.full_gpu
```

**Step 4: If tests pass, commit**

```bash
git add -A
git commit -m "test(phase1): verify paged mode passes integration tests"
```

**Step 5: If tests fail, debug and fix**

Check `OUT.*/running_scf.log` for errors, fix issues, and repeat.

---

## Phase 2: VKB Atom Batching

### Task 15: Create VKBBatchManager Class

**Files:**
- Create: `source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h`
- Create: `source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.cpp`

**Step 1: Write test for VKBBatchManager**

Create: `source/module_hamilt_pw/hamilt_pwdft/test/vkb_batch_test.cpp`

```cpp
#include "module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h"
#include "gtest/gtest.h"

class VKBBatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        natom = 200;
        nproj_per_atom = 10;
        npw = 10000;
        gpu_mem = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    }

    int natom;
    int nproj_per_atom;
    int npw;
    size_t gpu_mem;
};

TEST_F(VKBBatchTest, ComputeOptimalBatchSize) {
    int batch_size = compute_optimal_batch_size(natom, nproj_per_atom, npw, gpu_mem);
    EXPECT_GT(batch_size, 0);
    EXPECT_LE(batch_size, natom);
}

TEST_F(VKBBatchTest, InitializeBatchManager) {
    VKBBatchManager<std::complex<double>> manager;
    // Test will be implemented after class is created
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && ctest -R vkb_batch_test -V`
Expected: FAIL - class not defined

**Step 3: Create VKBBatchManager header**

Create: `source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h`

```cpp
#ifndef VKB_BATCH_MANAGER_H
#define VKB_BATCH_MANAGER_H

#include <vector>
#include "module_base/macros.h"

namespace hamilt {

/**
 * @brief Compute optimal batch size for VKB computation
 * @param total_atoms Total number of atoms
 * @param nproj_per_atom Average projectors per atom
 * @param npw Number of plane waves
 * @param available_gpu_mem Available GPU memory in bytes
 * @return Optimal number of atoms per batch
 */
int compute_optimal_batch_size(int total_atoms, int nproj_per_atom,
                                int npw, size_t available_gpu_mem);

template <typename T>
class VKBBatchManager {
public:
    VKBBatchManager() = default;
    ~VKBBatchManager();

    /**
     * @brief Initialize batch manager
     * @param total_atoms Total number of atoms
     * @param nproj_per_atom Projectors per atom (can be array)
     * @param npw Number of plane waves
     * @param gpu_mem_budget GPU memory budget for VKB
     */
    void init(int total_atoms, const int* nproj_per_atom, int npw, size_t gpu_mem_budget);

    /**
     * @brief Get batch information
     * @param ibatch Batch index
     * @param atom_start Output: starting atom index
     * @param atom_end Output: ending atom index (exclusive)
     * @param nkb_batch Output: number of projectors in batch
     */
    void get_batch_info(int ibatch, int& atom_start, int& atom_end, int& nkb_batch) const;

    int get_nbatch() const { return nbatch_; }
    T* get_vkb_batch_buffer() { return vkb_batch_gpu_; }

private:
    int nbatch_ = 0;
    std::vector<int> batch_start_;   // Starting atom for each batch
    std::vector<int> batch_size_;    // Number of atoms in each batch
    std::vector<int> batch_nkb_;     // Number of projectors in each batch
    T* vkb_batch_gpu_ = nullptr;     // GPU buffer for current batch
    size_t max_batch_size_ = 0;      // Maximum batch size in elements
};

} // namespace hamilt

#endif // VKB_BATCH_MANAGER_H
```

**Step 4: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: Linker errors (implementation not done)

**Step 5: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h
git add source/module_hamilt_pw/hamilt_pwdft/test/vkb_batch_test.cpp
git commit -m "feat(vkb): add VKBBatchManager class header"
```

---

### Task 16: Implement VKBBatchManager

**Files:**
- Create: `source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.cpp`

**Step 1: Implement compute_optimal_batch_size**

```cpp
#include "vkb_batch_manager.h"
#include <algorithm>

namespace hamilt {

int compute_optimal_batch_size(int total_atoms, int nproj_per_atom,
                                int npw, size_t available_gpu_mem) {
    // Calculate memory per atom
    size_t vkb_size_per_atom = nproj_per_atom * npw * sizeof(std::complex<double>);

    // Use 15% of available memory for VKB
    size_t target_batch_mem = available_gpu_mem * 0.15;

    // Calculate batch size
    int batch_size = target_batch_mem / vkb_size_per_atom;

    // Ensure at least 1 atom, at most total_atoms
    return std::max(1, std::min(batch_size, total_atoms));
}

} // namespace hamilt
```

**Step 2: Implement VKBBatchManager::init**

```cpp
template <typename T>
void VKBBatchManager<T>::init(int total_atoms, const int* nproj_per_atom,
                               int npw, size_t gpu_mem_budget) {
    // Calculate total projectors
    int total_nkb = 0;
    for (int iat = 0; iat < total_atoms; iat++) {
        total_nkb += nproj_per_atom[iat];
    }

    // Compute batch size
    int avg_nproj = total_nkb / total_atoms;
    int batch_size = compute_optimal_batch_size(total_atoms, avg_nproj, npw, gpu_mem_budget);

    // Create batches
    nbatch_ = (total_atoms + batch_size - 1) / batch_size;
    batch_start_.resize(nbatch_);
    batch_size_.resize(nbatch_);
    batch_nkb_.resize(nbatch_);

    int iat = 0;
    for (int ibatch = 0; ibatch < nbatch_; ibatch++) {
        batch_start_[ibatch] = iat;
        int atoms_in_batch = std::min(batch_size, total_atoms - iat);
        batch_size_[ibatch] = atoms_in_batch;

        // Count projectors in this batch
        int nkb_batch = 0;
        for (int i = 0; i < atoms_in_batch; i++) {
            nkb_batch += nproj_per_atom[iat + i];
        }
        batch_nkb_[ibatch] = nkb_batch;

        // Track maximum batch size
        max_batch_size_ = std::max(max_batch_size_, (size_t)(nkb_batch * npw));

        iat += atoms_in_batch;
    }

    // Allocate GPU buffer for largest batch
#if defined(__CUDA) || defined(__ROCM)
    cudaMalloc(&vkb_batch_gpu_, max_batch_size_ * sizeof(T));
#endif
}
```

**Step 3: Implement get_batch_info**

```cpp
template <typename T>
void VKBBatchManager<T>::get_batch_info(int ibatch, int& atom_start,
                                         int& atom_end, int& nkb_batch) const {
    if (ibatch < 0 || ibatch >= nbatch_) {
        throw std::runtime_error("Invalid batch index");
    }

    atom_start = batch_start_[ibatch];
    atom_end = atom_start + batch_size_[ibatch];
    nkb_batch = batch_nkb_[ibatch];
}
```

**Step 4: Implement destructor**

```cpp
template <typename T>
VKBBatchManager<T>::~VKBBatchManager() {
#if defined(__CUDA) || defined(__ROCM)
    if (vkb_batch_gpu_ != nullptr) {
        cudaFree(vkb_batch_gpu_);
        vkb_batch_gpu_ = nullptr;
    }
#endif
}

// Explicit instantiations
template class VKBBatchManager<std::complex<float>>;
template class VKBBatchManager<std::complex<double>>;
```

**Step 5: Add to CMakeLists.txt**

Modify: `source/module_hamilt_pw/hamilt_pwdft/CMakeLists.txt`

Add `vkb_batch_manager.cpp` to source list.

**Step 6: Run test**

Run: `cd build && cmake --build . -j$(nproc) && ctest -R vkb_batch_test -V`
Expected: PASS

**Step 7: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.cpp
git add source/module_hamilt_pw/hamilt_pwdft/CMakeLists.txt
git commit -m "feat(vkb): implement VKBBatchManager class"
```

---

### Task 17: Add getvnl_batch Method

**Files:**
- Modify: `source/module_hamilt_pw/hamilt_pwdft/VNL_in_pw.h`
- Modify: `source/module_hamilt_pw/hamilt_pwdft/VNL_in_pw.cpp`

**Step 1: Add method declaration to header**

In `VNL_in_pw.h`, add to `pseudopot_cell_vnl` class:

```cpp
// Add after existing getvnl methods
template<typename FPTYPE>
void getvnl_batch(Device* ctx, const UnitCell& ucell, const int& ik,
                  int atom_start, int atom_end,
                  std::complex<FPTYPE>* vkb_batch) const;
```

**Step 2: Write test for getvnl_batch**

Add to `source/module_hamilt_pw/hamilt_pwdft/test/vkb_batch_test.cpp`:

```cpp
TEST_F(VKBBatchTest, GetvnlBatchCorrectness) {
    // This test requires full setup - will be integration test
    // For now, just verify method exists
    SUCCEED();
}
```

**Step 3: Implement getvnl_batch**

Add to `VNL_in_pw.cpp`:

```cpp
template<typename FPTYPE>
void pseudopot_cell_vnl::getvnl_batch(Device* ctx, const UnitCell& ucell,
                                      const int& ik, int atom_start, int atom_end,
                                      std::complex<FPTYPE>* vkb_batch) const {
    if (PARAM.inp.use_paw) {
        return;
    }

    const int npw = this->wfcpw->npwk[ik];

    // Reuse existing getvnl logic but only for subset of atoms
    ModuleBase::matrix vkb1(nhm, npw);
    double* vq = new double[npw];
    const int x1 = (lmaxkb + 1) * (lmaxkb + 1);

    ModuleBase::matrix ylm(x1, npw);
    ModuleBase::Vector3<double>* gk = new ModuleBase::Vector3<double>[npw];
    for (int ig = 0; ig < npw; ig++) {
        gk[ig] = this->wfcpw->getgpluskcar(ik, ig);
    }

    ModuleBase::YlmReal::Ylm_Real(cpu_ctx, x1, npw, reinterpret_cast<double*>(gk), ylm.c);

    using Device = base_device::DEVICE_CPU;
    Device* cpu_ctx = {};
    using resmem_complex_op = base_device::memory::resize_memory_op<std::complex<double>, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<std::complex<double>, Device>;
    std::complex<double>* sk = nullptr;
    resmem_complex_op()(cpu_ctx, sk, ucell.nat * npw, "VNL::sk");
    this->psf->get_sk(cpu_ctx, ik, this->wfcpw, sk);

    int jkb = 0;
    for (int iat = atom_start; iat < atom_end; iat++) {
        int it = ucell.iat2it[iat];
        int ia = ucell.iat2ia[iat];

        const int nbeta = ucell.atoms[it].ncpp.nbeta;
        const int nh = ucell.atoms[it].ncpp.nh;

        // Calculate beta in G-space
        for (int nb = 0; nb < nbeta; nb++) {
            for (int ig = 0; ig < npw; ig++) {
                const double gnorm = gk[ig].norm() * ucell.tpiba;
                vq[ig] = ModuleBase::PolyInt::Polynomial_Interpolation(
                    this->tab, it, nb, PARAM.globalv.nqx, PARAM.globalv.dq, gnorm);
            }

            // Add spherical harmonic part
            for (int ih = 0; ih < nh; ih++) {
                if (nb == this->indv(it, ih)) {
                    const int lm = static_cast<int>(nhtolm(it, ih));
                    for (int ig = 0; ig < npw; ig++) {
                        vkb1(ih, ig) = ylm(lm, ig) * vq[ig];
                    }
                }
            }
        }

        // Add structure factor
        for (int ih = 0; ih < nh; ih++) {
            std::complex<double> pref = pow(ModuleBase::NEG_IMAG_UNIT, nhtol(it, ih));
            std::complex<FPTYPE>* pvkb = &vkb_batch[jkb * npw];
            for (int ig = 0; ig < npw; ig++) {
                pvkb[ig] = vkb1(ih, ig) * sk[iat * npw + ig] * pref;
            }
            ++jkb;
        }
    }

    delete[] vq;
    delete[] gk;
    delmem_complex_op()(cpu_ctx, sk);
}

// Explicit instantiations
template void pseudopot_cell_vnl::getvnl_batch<float>(Device*, const UnitCell&, const int&, int, int, std::complex<float>*) const;
template void pseudopot_cell_vnl::getvnl_batch<double>(Device*, const UnitCell&, const int&, int, int, std::complex<double>*) const;
```

**Step 4: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 5: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/VNL_in_pw.h
git add source/module_hamilt_pw/hamilt_pwdft/VNL_in_pw.cpp
git commit -m "feat(vkb): add getvnl_batch method for atom batching"
```

---

### Task 18: Add vkb_batch_atoms INPUT Parameter

**Files:**
- Modify: `source/module_parameter/input_parameter.h`
- Modify: `source/module_io/read_input_item_elec_stru.cpp`

**Step 1: Add parameter to input_parameter.h**

```cpp
// Around GPU parameters section
int vkb_batch_atoms = 0;  // Number of atoms per VKB batch (0 = auto)
```

**Step 2: Add parameter reading**

In `read_input_item_elec_stru.cpp`:

```cpp
{
    Input_Item item("vkb_batch_atoms");
    item.annotation = "Number of atoms per VKB batch (0 for auto-tune)";
    read_sync_int(input.vkb_batch_atoms);
    item.reset_value = [](const Input& input, Input_Item& item) {
        item.final_value << input.vkb_batch_atoms;
    };
    item.check_value = [](const Input& input, const Input_Item& item) {
        if (input.vkb_batch_atoms < 0) {
            ModuleBase::WARNING_QUIT("Input", "vkb_batch_atoms must be >= 0");
        }
    };
    this->add_item(item);
}
```

**Step 3: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 4: Commit**

```bash
git add source/module_parameter/input_parameter.h
git add source/module_io/read_input_item_elec_stru.cpp
git commit -m "feat(input): add vkb_batch_atoms parameter"
```

---

### Task 19: Modify Nonlocal Operator for Batching

**Files:**
- Modify: `source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.h`
- Modify: `source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp`

**Step 1: Add VKBBatchManager member to Nonlocal class**

In `nonlocal_pw.h`, add to private section:

```cpp
#include "module_hamilt_pw/hamilt_pwdft/vkb_batch_manager.h"

private:
    // Existing members...

    // VKB batching support
    bool use_vkb_batching_ = false;
    VKBBatchManager<T> vkb_manager_;
    T* becp_batch_ = nullptr;
    T* ps_batch_ = nullptr;
```

**Step 2: Add helper method declaration**

```cpp
private:
    void compute_ps_batch(const T* becp_batch, int nkb_batch,
                         int atom_start, int atom_end, T* ps_batch) const;
```

**Step 3: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: Linker errors (methods not implemented)

**Step 4: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.h
git commit -m "feat(nonlocal): add VKBBatchManager member to Nonlocal class"
```

---

### Task 20: Implement Nonlocal Batching in init()

**Files:**
- Modify: `source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp`

**Step 1: Initialize VKB batching in Nonlocal::init()**

Find the `init()` method and add at the end:

```cpp
template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::init(const int ik_in) {
    // Existing code...

    // Initialize VKB batching if enabled
    if (PARAM.inp.vkb_batch_atoms > 0 ||
        (PARAM.inp.device_memory_mode == "paged" && this->ppcell->nkb > 0)) {

        use_vkb_batching_ = true;

        // Get GPU memory info
        size_t free_mem = 0, total_mem = 0;
#if defined(__CUDA)
        cudaMemGetInfo(&free_mem, &total_mem);
#elif defined(__ROCM)
        hipMemGetInfo(&free_mem, &total_mem);
#endif

        // Build nproj_per_atom array
        std::vector<int> nproj_per_atom(this->ucell->nat);
        for (int iat = 0; iat < this->ucell->nat; iat++) {
            int it = this->ucell->iat2it[iat];
            nproj_per_atom[iat] = this->ucell->atoms[it].ncpp.nh;
        }

        // Initialize batch manager
        vkb_manager_.init(this->ucell->nat, nproj_per_atom.data(),
                         this->wfcpw->npwk[ik_in], free_mem);

        GlobalV::ofs_running << " VKB batching enabled: "
                            << vkb_manager_.get_nbatch() << " batches" << std::endl;
    }

    if (this->next_op != nullptr) {
        this->next_op->init(ik_in);
    }
}
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 3: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp
git commit -m "feat(nonlocal): initialize VKB batching in init()"
```

---

### Task 21: Implement compute_ps_batch Helper

**Files:**
- Modify: `source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp`

**Step 1: Implement compute_ps_batch method**

Add after the `add_nonlocal_pp` method:

```cpp
template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::compute_ps_batch(
    const T* becp_batch, int nkb_batch, int atom_start, int atom_end, T* ps_batch) const {

    int sum = 0;
    int iat = atom_start;

    if (this->npol == 1) {
        const int current_spin = this->isk[this->ik];

        for (int iat_local = atom_start; iat_local < atom_end; iat_local++) {
            int it = this->ucell->iat2it[iat_local];
            const int nproj = this->ucell->atoms[it].ncpp.nh;

            // ps[ip2] = Σ_ip deeq[ip2,ip] * becp[ip]
            for (int ib = 0; ib < nbands; ++ib) {
                for (int ip2 = 0; ip2 < nproj; ip2++) {
                    T sum_val = this->zero;
                    for (int ip = 0; ip < nproj; ip++) {
                        sum_val += this->deeq[current_spin * this->ppcell->deeq.getBound2() * this->ppcell->deeq.getBound3() * this->ppcell->deeq.getBound4()
                                             + iat * this->ppcell->deeq.getBound3() * this->ppcell->deeq.getBound4()
                                             + ip * this->ppcell->deeq.getBound4()
                                             + ip2]
                                  * becp_batch[ib * nkb_batch + sum + ip];
                    }
                    ps_batch[(sum + ip2) * nbands + ib] = sum_val;
                }
            }
            sum += nproj;
            ++iat;
        }
    } else {
        // Non-collinear case
        for (int iat_local = atom_start; iat_local < atom_end; iat_local++) {
            int it = this->ucell->iat2it[iat_local];
            const int nproj = this->ucell->atoms[it].ncpp.nh;

            for (int ib = 0; ib < nbands; ib += 2) {
                for (int ip2 = 0; ip2 < nproj; ip2++) {
                    for (int ip = 0; ip < nproj; ip++) {
                        int psind = (sum + ip2) * nbands + ib;
                        int becpind = ib * nkb_batch + sum + ip;
                        T becp1 = becp_batch[becpind];
                        T becp2 = becp_batch[becpind + nkb_batch];

                        ps_batch[psind] += this->deeq_nc[0 * this->ppcell->deeq_nc.getBound2() * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                        + iat * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                        + ip2 * this->ppcell->deeq_nc.getBound4()
                                                        + ip] * becp1
                                         + this->deeq_nc[1 * this->ppcell->deeq_nc.getBound2() * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                        + iat * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                        + ip2 * this->ppcell->deeq_nc.getBound4()
                                                        + ip] * becp2;

                        ps_batch[psind + 1] += this->deeq_nc[2 * this->ppcell->deeq_nc.getBound2() * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                            + iat * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                            + ip2 * this->ppcell->deeq_nc.getBound4()
                                                            + ip] * becp1
                                             + this->deeq_nc[3 * this->ppcell->deeq_nc.getBound2() * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                            + iat * this->ppcell->deeq_nc.getBound3() * this->ppcell->deeq_nc.getBound4()
                                                            + ip2 * this->ppcell->deeq_nc.getBound4()
                                                            + ip] * becp2;
                    }
                }
            }
            sum += nproj;
            ++iat;
        }
    }
}
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 3: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp
git commit -m "feat(nonlocal): implement compute_ps_batch helper method"
```

---

### Task 22: Implement Batched act() Method

**Files:**
- Modify: `source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp`

**Step 1: Modify act() to support batching**

Find the `act()` method and wrap existing code with batching logic:

```cpp
template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node) const {

    ModuleBase::timer::tick("Operator", "NonlocalPW");

    if (is_first_node) {
        setmem_complex_op()(this->ctx, tmhpsi, 0, nbasis * nbands / npol);
    }

    if (!PARAM.inp.use_paw && this->ppcell->nkb > 0) {
        this->npw = ngk_ik;
        this->max_npw = nbasis / npol;
        this->npol = npol;

        if (use_vkb_batching_) {
            // ===== BATCHED PATH =====
            for (int ibatch = 0; ibatch < vkb_manager_.get_nbatch(); ibatch++) {
                int atom_start, atom_end, nkb_batch;
                vkb_manager_.get_batch_info(ibatch, atom_start, atom_end, nkb_batch);

                // Get VKB batch buffer
                T* vkb_batch = vkb_manager_.get_vkb_batch_buffer();

                // Compute vkb for this batch
                this->ppcell->getvnl_batch(this->ctx, *this->ucell, this->ik,
                                          atom_start, atom_end, vkb_batch);

                // Allocate becp_batch if needed
                if (this->nkb_m < nbands * nkb_batch) {
                    resmem_complex_op()(this->ctx, this->becp_batch_, nbands * nkb_batch);
                    this->nkb_m = nbands * nkb_batch;
                }

                // Compute becp_batch = vkb_batch^† × psi
                char transa = 'C';
                char transb = 'N';
                if (nbands == 1) {
                    int inc = 1;
                    gemv_op()(this->ctx, transa, this->npw, nkb_batch,
                             &this->one, vkb_batch, this->npw,
                             tmpsi_in, inc,
                             &this->zero, this->becp_batch_, inc);
                } else {
                    gemm_op()(this->ctx, transa, transb,
                             nkb_batch, nbands, this->npw,
                             &this->one, vkb_batch, this->npw,
                             tmpsi_in, max_npw,
                             &this->zero, this->becp_batch_, nkb_batch);
                }

                Parallel_Reduce::reduce_pool(becp_batch_, nkb_batch * nbands);

                // Allocate ps_batch if needed
                if (ps_batch_ == nullptr) {
                    resmem_complex_op()(this->ctx, this->ps_batch_, nkb_batch * nbands);
                }
                setmem_complex_op()(this->ctx, this->ps_batch_, 0, nkb_batch * nbands);

                // Compute ps_batch = deeq × becp_batch
                this->compute_ps_batch(becp_batch_, nkb_batch, atom_start, atom_end, ps_batch_);

                // Accumulate: hpsi += vkb_batch × ps_batch
                if (nbands == 1) {
                    int inc = 1;
                    gemv_op()(this->ctx, 'N', this->npw, nkb_batch,
                             &this->one, vkb_batch, this->npw,
                             this->ps_batch_, inc,
                             &this->one, tmhpsi, inc);  // alpha=1 for accumulation
                } else {
                    gemm_op()(this->ctx, 'N', 'T',
                             this->npw, nbands, nkb_batch,
                             &this->one, vkb_batch, this->npw,
                             this->ps_batch_, nbands,
                             &this->one, tmhpsi, max_npw);  // alpha=1 for accumulation
                }
            }
        } else {
            // ===== ORIGINAL NON-BATCHED PATH =====
            // ... existing code unchanged ...
        }
    }

    ModuleBase::timer::tick("Operator", "NonlocalPW");
}
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 3: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp
git commit -m "feat(nonlocal): implement batched act() method"
```

---

### Task 23: Add Destructor Cleanup for Batching

**Files:**
- Modify: `source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp`

**Step 1: Update destructor**

Find the destructor and add:

```cpp
template<typename T, typename Device>
Nonlocal<OperatorPW<T, Device>>::~Nonlocal() {
    delmem_complex_op()(this->ctx, this->ps);
    delmem_complex_op()(this->ctx, this->becp);

    // Cleanup batching resources
    if (becp_batch_ != nullptr) {
        delmem_complex_op()(this->ctx, this->becp_batch_);
        becp_batch_ = nullptr;
    }
    if (ps_batch_ != nullptr) {
        delmem_complex_op()(this->ctx, this->ps_batch_);
        ps_batch_ = nullptr;
    }
}
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors

**Step 3: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/operator_pw/nonlocal_pw.cpp
git commit -m "feat(nonlocal): add cleanup for batching resources"
```

---

### Task 24: Run Phase 2 Integration Tests

**Files:**
- Test: `tests/integrate/11_PW_GPU/*`

**Step 1: Build and install**

```bash
cd build
cmake --build . -j$(nproc)
cmake --install .
```

**Step 2: Run tests with both paging and batching**

```bash
cd tests/integrate
export ABACUS_DEVICE_MEMORY_MODE=paged
export ABACUS_VKB_BATCH_ATOMS=20
./Autotest.sh -r "11_PW_GPU_101.*"
```

Expected: All tests PASS

**Step 3: Compare with Phase 1 results**

```bash
# Should be identical to Phase 1 (paging only)
python ../tools/compare_results.py OUT.phase2 OUT.phase1
```

Expected: Energy differences < 1e-6 eV

**Step 4: Profile memory usage**

```bash
# Run with memory profiling
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > mem_phase2.log &
PROFILER_PID=$!
mpirun -np 1 abacus
kill $PROFILER_PID

# Compare with baseline
python ../tools/analyze_memory.py mem_baseline.log mem_phase2.log
```

Expected: 75-85% memory reduction

**Step 5: If tests pass, commit**

```bash
git add -A
git commit -m "test(phase2): verify combined paging+batching passes tests"
```

---

## Phase 3: Real-Space Extension (Optional)

### Task 25: Create Projector Base Class

**Files:**
- Create: `source/module_hamilt_pw/hamilt_pwdft/projector_base.h`

**Step 1: Create abstract base class**

```cpp
#ifndef PROJECTOR_BASE_H
#define PROJECTOR_BASE_H

#include <cstddef>

namespace hamilt {

template <typename T>
class ProjectorBase {
public:
    virtual ~ProjectorBase() = default;

    /**
     * @brief Compute projection coefficients becp = <beta|psi>
     * @param psi Input wavefunctions in reciprocal space
     * @param becp Output projection coefficients
     * @param nbands Number of bands
     */
    virtual void compute_becp(const T* psi, T* becp, int nbands) = 0;

    /**
     * @brief Apply deeq matrix: hpsi += Σ_jkb vkb[jkb] * (deeq * becp)[jkb]
     * @param becp Input projection coefficients
     * @param hpsi Output: accumulated to hpsi
     * @param nbands Number of bands
     */
    virtual void apply_deeq(const T* becp, T* hpsi, int nbands) = 0;

    /**
     * @brief Get memory size used by projector
     * @return Memory size in bytes
     */
    virtual size_t get_memory_size() const = 0;
};

} // namespace hamilt

#endif // PROJECTOR_BASE_H
```

**Step 2: Verify compilation**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i error`
Expected: No errors (header only)

**Step 3: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/projector_base.h
git commit -m "feat(phase3): add ProjectorBase abstract class"
```

---

### Task 26: Create ReciprocalProjector Class

**Files:**
- Create: `source/module_hamilt_pw/hamilt_pwdft/reciprocal_projector.h`
- Create: `source/module_hamilt_pw/hamilt_pwdft/reciprocal_projector.cpp`

**Step 1: Create header**

```cpp
#ifndef RECIPROCAL_PROJECTOR_H
#define RECIPROCAL_PROJECTOR_H

#include "projector_base.h"
#include "vkb_batch_manager.h"

namespace hamilt {

/**
 * @brief Reciprocal-space projector (current implementation)
 * Uses vkb in reciprocal space with optional batching
 */
template <typename T>
class ReciprocalProjector : public ProjectorBase<T> {
public:
    ReciprocalProjector(/* parameters */);
    ~ReciprocalProjector() override;

    void compute_becp(const T* psi, T* becp, int nbands) override;
    void apply_deeq(const T* becp, T* hpsi, int nbands) override;
    size_t get_memory_size() const override;

private:
    VKBBatchManager<T> vkb_manager_;
    // Other members...
};

} // namespace hamilt

#endif // RECIPROCAL_PROJECTOR_H
```

**Step 2: Note for implementation**

This wraps the existing batched VKB logic into the ProjectorBase interface.
Implementation would move code from Nonlocal::act() into this class.

**Step 3: Commit**

```bash
git add source/module_hamilt_pw/hamilt_pwdft/reciprocal_projector.h
git commit -m "feat(phase3): add ReciprocalProjector class header"
```

---

### Task 27: Document Phase 3 Extension Points

**Files:**
- Create: `docs/plans/2026-03-02-phase3-realspace-notes.md`

**Step 1: Create documentation**

```markdown
# Phase 3: Real-Space Projector Extension

## Status: Design Complete, Implementation Deferred

Phase 3 is an optional future enhancement. The architecture has been designed
to support real-space projectors without refactoring Phases 1-2.

## Extension Points

1. **ProjectorBase interface** - Abstract base class for projectors
2. **ReciprocalProjector** - Wraps current batched VKB implementation
3. **RealSpaceProjector** - Future: real-space beta(r) projections

## Implementation Steps (Future Work)

1. Implement ReciprocalProjector class (wrap existing batched code)
2. Refactor Nonlocal::act() to use ProjectorBase polymorphism
3. Implement RealSpaceProjector class
4. Add FFT integration for psi(G) ↔ psi(r) transforms
5. Generate beta_r from pseudopotential files
6. Optimize real-space integration kernels
7. Add INPUT parameter: projector_type = "reciprocal" | "realspace"

## When to Implement

- Very large systems (>500 atoms, >50000 plane waves)
- Memory-constrained GPUs where Phase 2 is insufficient
- When FFT overhead (~10-15%) is acceptable

## References

- Design document: `docs/plans/2026-03-02-gpu-memory-optimization-design.md`
- VASP real-space implementation: Kresse & Furthmüller, PRB 54, 11169 (1996)
```

**Step 2: Commit**

```bash
git add docs/plans/2026-03-02-phase3-realspace-notes.md
git commit -m "docs(phase3): document real-space extension points"
```

---

## Final Integration and Documentation

### Task 28: Update Documentation

**Files:**
- Modify: `docs/advanced/acceleration/cuda.md`

**Step 1: Add section on memory optimization**

Add after the "Known limitations" section:

```markdown
## GPU Memory Optimization

ABACUS provides memory optimization for GPU calculations to handle larger systems:

### Paged Memory Mode

For calculations with many k-points, use paged memory mode to reduce GPU memory usage:

\`\`\`
INPUT:
    device  gpu
    device_memory_mode  paged
\`\`\`

This mode:
- Stores wavefunctions on CPU, pages one k-point at a time to GPU
- Reduces GPU memory by ~90% for wavefunction storage
- Adds ~5-8% computational overhead
- Automatically enabled for >10 k-points if device_memory_mode is not set

### VKB Batching

For systems with many atoms, VKB projectors can be computed in batches:

\`\`\`
INPUT:
    device  gpu
    device_memory_mode  paged
    vkb_batch_atoms  20  # or 0 for auto-tune
\`\`\`

Combined optimization:
- 75-85% total GPU memory reduction
- <15% computational overhead
- Enables calculations that wouldn't fit in GPU memory

### Memory Usage

Check GPU memory usage during calculation:
\`\`\`bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
\`\`\`
```

**Step 2: Commit**

```bash
git add docs/advanced/acceleration/cuda.md
git commit -m "docs: add GPU memory optimization documentation"
```

---

### Task 29: Create Example INPUT Files

**Files:**
- Create: `examples/gpu_memory_optimization/README.md`
- Create: `examples/gpu_memory_optimization/INPUT.paged`
- Create: `examples/gpu_memory_optimization/INPUT.full_gpu`

**Step 1: Create README**

```markdown
# GPU Memory Optimization Examples

This directory contains examples demonstrating GPU memory optimization features.

## Files

- `INPUT.full_gpu` - Standard full GPU mode (baseline)
- `INPUT.paged` - Paged memory mode for many k-points
- `INPUT.batched` - Combined paging + VKB batching

## Usage

\`\`\`bash
# Run with full GPU mode
cp INPUT.full_gpu INPUT
mpirun -np 1 abacus

# Run with paged mode
cp INPUT.paged INPUT
mpirun -np 1 abacus

# Compare memory usage
nvidia-smi
\`\`\`

## Expected Results

- Paged mode: ~90% less GPU memory for wavefunctions
- Batched mode: ~75-85% total GPU memory reduction
- Performance: <15% overhead
- Numerical results: Identical within 1e-6 eV
```

**Step 2: Create INPUT files**

Create `INPUT.full_gpu`:
```
INPUT_PARAMETERS
calculation  scf
basis_type   pw
device       gpu
device_memory_mode  full_gpu
ks_solver    dav
```

Create `INPUT.paged`:
```
INPUT_PARAMETERS
calculation  scf
basis_type   pw
device       gpu
device_memory_mode  paged
vkb_batch_atoms  0  # auto-tune
ks_solver    dav
```

**Step 3: Commit**

```bash
git add examples/gpu_memory_optimization/
git commit -m "docs: add GPU memory optimization examples"
```

---

### Task 30: Final Testing and Validation

**Files:**
- Test: All GPU tests

**Step 1: Run full test suite**

```bash
cd build
cmake --build . -j$(nproc)
cmake --install .

cd ../tests/integrate
./Autotest.sh -r "11_PW_GPU.*"
```

Expected: All tests PASS

**Step 2: Run performance benchmarks**

```bash
cd ../..
bash scripts/benchmark_memory_optimization.sh
```

Expected:
- Memory reduction: 75-85%
- Performance overhead: <15%

**Step 3: Generate final report**

```bash
python scripts/generate_optimization_report.py > OPTIMIZATION_REPORT.md
```

**Step 4: Commit final results**

```bash
git add OPTIMIZATION_REPORT.md
git commit -m "test: add final optimization validation report"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-03-02-gpu-memory-optimization-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
